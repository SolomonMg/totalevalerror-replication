#!/usr/bin/env Rscript
# 30_arena_pipeline_comparison.R
# Compare 4 scoring-pipeline configurations against Arena human preferences.
#
# Configurations (all derived from a single factorial scoring run):
#   single_likert:     1 judge (gpt-oss-120b) x 1 prompt variant (V0) x Likert
#   single_pairwise:   1 judge (gpt-oss-120b) x 1 prompt variant (V0) x pairwise (listed only)
#   tee_likert:        3 judges x 5 variants averaged
#   tee_pairwise:      3 judges x 5 variants x 2 orderings averaged
#
# Per-battle prediction of Arena human winner (A / B / tie):
#   Likert: winner = argmax(mean_score_a, mean_score_b); tie if |diff| < 0.5
#   Pairwise: winner = majority of A/B judgments; tie if 50/50 split
#
# Output:
#   data/processed/arena_agreement_summary.csv  -- accuracy per (config x category)
#   data/processed/arena_battle_predictions.csv -- per-battle predicted winner per config

suppressPackageStartupMessages({
  library(tidyverse)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

set.seed(42)

# Single-judge baseline anchor
BASELINE_JUDGE <- "openai/gpt-oss-120b"
BASELINE_VARIANT <- 0L

# Tie-detection tolerance for Likert difference-of-means (on the 1-5 scale)
LIKERT_TIE_TOL <- 0.5

lik <- read_csv("data/processed/arena_likert_clean.csv",  show_col_types = FALSE)
pw  <- read_csv("data/processed/arena_pairwise_clean.csv", show_col_types = FALSE)

cat(sprintf("Loaded %d likert rows, %d pairwise rows\n", nrow(lik), nrow(pw)))
cat(sprintf("Battles covered: %d likert, %d pairwise\n",
            n_distinct(lik$battle_id), n_distinct(pw$battle_id)))

normalize_winner <- function(w) {
  case_when(
    w == "model_a"        ~ "a",
    w == "model_b"        ~ "b",
    w %in% c("tie", "tie (bothbad)") ~ "tie",
    TRUE ~ NA_character_
  )
}

# ---- Per-battle mean scores per response side, per (judge, variant) cell ----
lik_cell <- lik %>% filter(!is.na(score)) %>%
  group_by(battle_id, category_llm, winner, response_side, judge_model, variant_id) %>%
  summarize(score = mean(score), .groups = "drop")  # should already be unique; safety

# Pairwise cell = per (battle, judge, variant) model_a_wins rate across 2 orderings
pw_cell <- pw %>% filter(!is.na(model_a_wins)) %>%
  group_by(battle_id, category_llm, winner, judge_model, variant_id) %>%
  summarize(a_win_rate = mean(model_a_wins), n = n(), .groups = "drop")

# ---- Configuration 1: single-judge Likert ----
single_lik <- lik_cell %>%
  filter(judge_model == BASELINE_JUDGE, variant_id == BASELINE_VARIANT) %>%
  select(battle_id, category_llm, winner, response_side, score) %>%
  pivot_wider(names_from = response_side, values_from = score,
              names_prefix = "score_") %>%
  mutate(
    config = "single_likert",
    pred_winner = case_when(
      is.na(score_a) | is.na(score_b) ~ NA_character_,
      abs(score_a - score_b) < LIKERT_TIE_TOL ~ "tie",
      score_a > score_b ~ "a",
      score_a < score_b ~ "b",
      TRUE ~ "tie"
    )
  )

# ---- Configuration 2: single-judge pairwise ----
single_pw <- pw_cell %>%
  filter(judge_model == BASELINE_JUDGE, variant_id == BASELINE_VARIANT) %>%
  mutate(
    config = "single_pairwise",
    # a_win_rate is mean across the 2 orderings (each in {0,1}); so it's in {0, 0.5, 1}.
    pred_winner = case_when(
      is.na(a_win_rate) ~ NA_character_,
      abs(a_win_rate - 0.5) < 1e-9 ~ "tie",
      a_win_rate > 0.5 ~ "a",
      TRUE ~ "b"
    )
  ) %>% select(battle_id, category_llm, winner, config, pred_winner)

# ---- Configuration 3: TEE Likert (3 judges x 5 variants averaged) ----
tee_lik <- lik_cell %>%
  group_by(battle_id, category_llm, winner, response_side) %>%
  summarize(score = mean(score), .groups = "drop") %>%
  pivot_wider(names_from = response_side, values_from = score,
              names_prefix = "score_") %>%
  mutate(
    config = "tee_likert",
    pred_winner = case_when(
      is.na(score_a) | is.na(score_b) ~ NA_character_,
      abs(score_a - score_b) < LIKERT_TIE_TOL ~ "tie",
      score_a > score_b ~ "a",
      score_a < score_b ~ "b",
      TRUE ~ "tie"
    )
  )

# ---- Configuration 4: TEE pairwise (3 judges x 5 variants x 2 orderings averaged) ----
tee_pw <- pw_cell %>%
  group_by(battle_id, category_llm, winner) %>%
  summarize(a_win_rate = mean(a_win_rate), .groups = "drop") %>%
  mutate(
    config = "tee_pairwise",
    # Tie-zone half-width 0.175 (calibrated to match Arena's ~35% human tie
    # rate; see analysis/39_pairwise_tie_threshold_sweep.R). Train/test stable.
    pred_winner = case_when(
      is.na(a_win_rate) ~ NA_character_,
      a_win_rate > 0.675 ~ "a",
      a_win_rate < 0.325 ~ "b",
      TRUE ~ "tie"
    )
  ) %>% select(battle_id, category_llm, winner, config, pred_winner)

# ---- Combine ----
preds <- bind_rows(
  single_lik %>% select(battle_id, category_llm, winner, config, pred_winner),
  single_pw,
  tee_lik    %>% select(battle_id, category_llm, winner, config, pred_winner),
  tee_pw
) %>%
  mutate(
    winner_n = normalize_winner(winner),
    correct  = pred_winner == winner_n,
    # "strict-correct" counting ties as their own class
    correct_strict = correct
  ) %>%
  filter(!is.na(pred_winner), !is.na(winner_n))

# Also compute a "AB-only" accuracy: filter to battles where human winner is
# decisive (model_a or model_b) and the pipeline makes an AB prediction.
preds_ab <- preds %>% filter(winner_n %in% c("a", "b"),
                              pred_winner %in% c("a", "b")) %>%
  mutate(correct_ab = pred_winner == winner_n)

# ---- Summary: accuracy per (config x category) ----
summary_by_cat <- preds %>%
  group_by(config, category_llm) %>%
  summarize(
    n_battles = n(),
    acc_all = mean(correct_strict, na.rm = TRUE),
    .groups = "drop"
  )

summary_ab <- preds_ab %>%
  group_by(config, category_llm) %>%
  summarize(
    n_battles_ab = n(),
    acc_ab = mean(correct_ab, na.rm = TRUE),
    .groups = "drop"
  )

summary_full <- summary_by_cat %>%
  left_join(summary_ab, by = c("config", "category_llm")) %>%
  arrange(category_llm, config)

cat("\n=== Accuracy by pipeline config x task category ===\n")
cat("  acc_all: ties predicted as ties count as correct if human also tied\n")
cat("  acc_ab:  AB-only (both pipeline and human decisive)\n\n")
print(summary_full %>% mutate(across(where(is.numeric), ~ round(., 3))))

# Pooled (ignoring category)
summary_pool <- preds %>%
  group_by(config) %>%
  summarize(n = n(), acc_all = mean(correct_strict), .groups = "drop") %>%
  left_join(
    preds_ab %>% group_by(config) %>%
      summarize(n_ab = n(), acc_ab = mean(correct_ab), .groups = "drop"),
    by = "config"
  )
cat("\n=== Pooled accuracy ===\n")
print(summary_pool %>% mutate(across(where(is.numeric), ~ round(., 3))))

# ---- TEE vs. single improvement per category ----
imp <- summary_full %>%
  select(config, category_llm, acc_ab) %>%
  pivot_wider(names_from = config, values_from = acc_ab) %>%
  mutate(
    improvement_likert  = tee_likert  - single_likert,
    improvement_pairwise = tee_pairwise - single_pairwise
  )
cat("\n=== TEE - single improvement (acc_ab) per category ===\n")
print(imp %>% mutate(across(where(is.numeric), ~ round(., 3))))

# ---- Stratified bootstrap CIs on (TEE - single) improvement per category ----
cat("\n=== Bootstrap CIs (2000 reps, stratified by category) ===\n")
N_BOOT <- 2000

bootstrap_imp <- function(preds_ab, n_boot = N_BOOT) {
  # Need a per-battle record with all 4 config predictions joined.
  wide <- preds_ab %>%
    select(battle_id, category_llm, config, correct_ab) %>%
    pivot_wider(names_from = config, values_from = correct_ab)
  # Ensure all 4 config columns exist (pivot_wider drops missing configs)
  for (c in c("single_likert", "single_pairwise", "tee_likert", "tee_pairwise")) {
    if (!c %in% names(wide)) wide[[c]] <- NA_real_
  }
  wide <- wide %>% filter(
    !is.na(single_likert), !is.na(single_pairwise),
    !is.na(tee_likert),    !is.na(tee_pairwise)
  )
  if (nrow(wide) < 20) {
    message("Bootstrap: only ", nrow(wide), " battles with all 4 configs; skipping CIs")
    return(tibble(category_llm = character(),
                   imp_likert_lo = numeric(), imp_likert_hi = numeric(),
                   imp_pairwise_lo = numeric(), imp_pairwise_hi = numeric()))
  }
  boots <- map_dfr(seq_len(n_boot), function(b) {
    # Stratified resample within category
    samp <- wide %>% group_by(category_llm) %>%
      slice_sample(prop = 1, replace = TRUE) %>% ungroup()
    samp %>% group_by(category_llm) %>%
      summarize(
        imp_likert   = mean(tee_likert)   - mean(single_likert),
        imp_pairwise = mean(tee_pairwise) - mean(single_pairwise),
        .groups = "drop"
      )
  })
  boots %>% group_by(category_llm) %>%
    summarize(
      imp_likert_lo   = quantile(imp_likert,   0.025),
      imp_likert_hi   = quantile(imp_likert,   0.975),
      imp_pairwise_lo = quantile(imp_pairwise, 0.025),
      imp_pairwise_hi = quantile(imp_pairwise, 0.975),
      .groups = "drop"
    )
}

if (nrow(preds_ab) > 0) {
  boot_ci <- bootstrap_imp(preds_ab)
  imp <- imp %>% left_join(boot_ci, by = c("category_llm"))
  cat("\n=== TEE - single improvement with 95% bootstrap CIs ===\n")
  print(imp %>% mutate(across(where(is.numeric), ~ round(., 3))))
}

# ---- Write outputs ----
write_csv(preds,        "data/processed/arena_battle_predictions.csv")
write_csv(summary_full, "data/processed/arena_agreement_summary.csv")
write_csv(summary_pool, "data/processed/arena_agreement_pooled.csv")
write_csv(imp,          "data/processed/arena_agreement_improvement.csv")

cat("\nWrote:\n")
cat("  data/processed/arena_battle_predictions.csv\n")
cat("  data/processed/arena_agreement_summary.csv\n")
cat("  data/processed/arena_agreement_pooled.csv\n")
cat("  data/processed/arena_agreement_improvement.csv (+ bootstrap CIs)\n")
