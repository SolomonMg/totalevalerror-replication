#!/usr/bin/env Rscript
# 31_arena_dstudy_predicts_improvement.R
# On the dev split, per task category: fit a variance decomposition on Likert
# scores to estimate the (judge + prompt + interaction) variance budget. Apply
# the D-study formula to project single->TEE SE reduction. Then compare the
# projected reduction ordering across categories to the observed improvement
# in Arena-human agreement (computed in 30).
#
# Claim the paper wants to make: TEE's D-study, fit in advance on dev, predicts
# which task categories benefit most from multi-judge averaging. Verify on test.
#
# Outputs:
#   data/processed/arena_dstudy_per_category.csv
#   data/processed/arena_dstudy_vs_observed.csv

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

set.seed(42)

TRAIN_FRAC <- 0.80
TARGET_CATEGORIES <- c("creative_writing", "persuasion", "coding", "factual_qa")

lik <- read_csv("data/processed/arena_likert_clean.csv", show_col_types = FALSE)
pw  <- read_csv("data/processed/arena_pairwise_clean.csv", show_col_types = FALSE)

# ---- Build dev/test split on battles (stratified by category) ----
battles_df <- lik %>% distinct(battle_id, category_llm)
dev_ids <- battles_df %>% group_by(category_llm) %>%
  slice_sample(prop = TRAIN_FRAC) %>% ungroup() %>% pull(battle_id)
test_ids <- setdiff(battles_df$battle_id, dev_ids)
cat(sprintf("Dev battles: %d   |   Test battles: %d\n", length(dev_ids), length(test_ids)))

lik_dev <- lik %>% filter(battle_id %in% dev_ids)

# ---- Per-category variance decomposition on dev (Likert scores) ----
# We treat each response (A or B) as a separate "item" and fit:
#   score ~ (1 | battle_response) + (1 | variant_id) + (1 | judge_model) +
#           (1 | battle_response:variant_id) + (1 | battle_response:judge_model) +
#           (1 | variant_id:judge_model)
# This is the same decomposition structure used elsewhere in the paper, adapted
# to Arena's factorial (no temperature facet; 1 rep per cell).

fit_per_cat <- function(cat) {
  d <- lik_dev %>% filter(category_llm == cat, !is.na(score)) %>%
    mutate(br = paste(battle_id, response_side, sep = "_"),
           variant_id = as.factor(variant_id),
           judge_model = as.factor(judge_model))
  if (nrow(d) < 100) {
    warning(sprintf("Too few dev rows (%d) for category %s; skipping", nrow(d), cat))
    return(NULL)
  }
  mod <- lmer(
    score ~ (1 | br) + (1 | variant_id) + (1 | judge_model) +
            (1 | br:variant_id) + (1 | br:judge_model) + (1 | variant_id:judge_model),
    data = d, REML = TRUE,
    control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000))
  )
  vc <- as.data.frame(VarCorr(mod)) %>% select(grp, vcov) %>% rename(component = grp, variance = vcov)
  labels <- c("br"="item_response", "variant_id"="prompt", "judge_model"="judge",
              "br:variant_id"="item x prompt",
              "br:judge_model"="item x judge",
              "variant_id:judge_model"="prompt x judge",
              "Residual"="residual")
  vc$tle_label <- labels[vc$component]
  vc$category <- cat
  vc
}

per_cat_vc <- map_dfr(TARGET_CATEGORIES, fit_per_cat)

cat("\n=== Per-category variance components (dev split) ===\n")
print(per_cat_vc %>% mutate(variance = round(variance, 4)) %>%
        pivot_wider(id_cols = tle_label, names_from = category,
                    values_from = variance))

# ---- D-study: single->TEE SE reduction per category ----
# Single pipeline: V=1, M=1, R=1 (scoring one response, one judge, one prompt)
# TEE pipeline:    V=5, M=3, R=1 (5 variants averaged, 3 judges averaged)
# Since each row is one response's score, "N" for item isn't a D-study facet here;
# we're asking how much SE of the per-response score shrinks under TEE.

dstudy_se <- function(vc, V, M) {
  g <- function(lbl) {
    v <- vc$variance[vc$tle_label == lbl]
    if (length(v) == 0) 0 else v
  }
  s2_item     <- g("item_response")
  s2_prompt   <- g("prompt")
  s2_judge    <- g("judge")
  s2_ip       <- g("item x prompt")
  s2_ij       <- g("item x judge")
  s2_pj       <- g("prompt x judge")
  s2_res      <- g("residual")
  # SE^2 of the MEAN score for one response under V prompts x M judges (averaged):
  # Per-call variance contributing to the per-response mean: all non-item terms
  # scale with 1/(VM), while s2_item is fixed (it's the item's true score).
  # We want the variance of the *mean score estimate* around the true item score.
  s2_prompt/V + s2_judge/M + s2_ip/V + s2_ij/M + s2_pj/(V*M) + s2_res/(V*M)
}

proj <- map_dfr(TARGET_CATEGORIES, function(cat) {
  vc <- per_cat_vc %>% filter(category == cat)
  if (nrow(vc) == 0) return(NULL)
  se_single <- sqrt(dstudy_se(vc, V = 1, M = 1))
  se_tee    <- sqrt(dstudy_se(vc, V = 5, M = 3))
  tibble(category = cat,
         se_single = se_single,
         se_tee = se_tee,
         se_reduction = 1 - se_tee / se_single)
})
cat("\n=== D-study projected SE reduction (dev) ===\n")
print(proj %>% mutate(across(where(is.numeric), ~ round(., 3))))

# ---- Compare projected vs. observed improvement ----
observed <- read_csv("data/processed/arena_agreement_improvement.csv",
                     show_col_types = FALSE)

# If improvement CSV includes all categories (not just 4 targets), filter
if ("category_llm" %in% names(observed)) {
  observed <- observed %>% rename(category = category_llm)
}
observed <- observed %>% filter(category %in% TARGET_CATEGORIES)

cmp <- proj %>% left_join(observed, by = "category")
cat("\n=== Projected SE reduction vs observed accuracy improvement ===\n")
print(cmp %>% mutate(across(where(is.numeric), ~ round(., 3))))

# Spearman correlation over the 4 points (underpowered but directional)
if (nrow(cmp) >= 3 && all(!is.na(cmp$se_reduction))) {
  rho_lik <- suppressWarnings(cor(cmp$se_reduction, cmp$improvement_likert,
                                   method = "spearman", use = "complete.obs"))
  rho_pw  <- suppressWarnings(cor(cmp$se_reduction, cmp$improvement_pairwise,
                                   method = "spearman", use = "complete.obs"))
  cat(sprintf("\nSpearman rho(projected SE reduction, observed Likert improvement): %.3f\n", rho_lik))
  cat(sprintf("Spearman rho(projected SE reduction, observed pairwise improvement): %.3f\n", rho_pw))
}

write_csv(per_cat_vc, "data/processed/arena_dstudy_per_category.csv")
write_csv(cmp,         "data/processed/arena_dstudy_vs_observed.csv")
cat("\nWrote:\n")
cat("  data/processed/arena_dstudy_per_category.csv\n")
cat("  data/processed/arena_dstudy_vs_observed.csv\n")
