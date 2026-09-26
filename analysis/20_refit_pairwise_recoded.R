#!/usr/bin/env Rscript
# 20_refit_pairwise_recoded.R
# Refit the pairwise ideology variance decomposition with outcome RECODED from
# position-based (1 if judge said "A") to model-based (1 if model_a_original
# won). Removes position bias from the raw outcome and isolates real content
# disagreement.
#
# Compares to the original position-based decomposition (current paper numbers).

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
})

# -------- Load existing cleaned pairwise data --------
df <- read_csv("data/processed/pairwise_clean.csv", show_col_types = FALSE) %>%
  filter(!is.na(outcome), !is.na(true_order))

cat(sprintf("=== 20_refit_pairwise_recoded.R ===\n"))
cat(sprintf("Rows: %d | judges: %d | items: %d | variants: %d\n",
            nrow(df),
            n_distinct(df$judge_model),
            n_distinct(df$item_id),
            n_distinct(df$variant_id)))

# -------- Recode outcome --------
# outcome = 1 if judge said "A" (position-based)
# model_a_wins = 1 if the original-model-A response won (content-based)
#   true_order == "original" -> response_a = model_a_original -> model_a_wins = outcome
#   true_order == "swapped"  -> response_a = model_b_original -> model_a_wins = 1 - outcome
df <- df %>% mutate(
  model_a_wins = ifelse(true_order == "original", outcome, 1 - outcome)
)

cat("\n=== Per-judge means (position-based vs recoded) ===\n")
df %>% group_by(judge_model) %>%
  summarize(
    n = n(),
    mean_position_based = mean(outcome),
    mean_recoded        = mean(model_a_wins),
    .groups = "drop"
  ) %>% print()

# -------- Fit recoded model --------
# Same structure as 02_variance_decomposition.R with true_order as a fixed
# covariate (retained for fair comparison -- under recoding, true_order
# absorbs nothing since it's already used to construct model_a_wins, but we
# keep it to match the original specification mechanically).
#
# Under the recoded outcome, we DROP true_order from the formula because it's
# redundant with the recoding step. Any residual position-bias effect would
# show up as a judge fixed-effect shift, which is what we want to measure.

df$temperature  <- as.factor(df$temperature)
df$judge_model  <- as.factor(df$judge_model)

cat("\n=== Fitting lmer on recoded outcome (model_a_wins) ===\n")
formula_recoded <- model_a_wins ~ temperature + judge_model +
  (1 | category) +
  (1 | item_id) +
  (1 | variant_id) +
  (1 | item_id:variant_id) +
  (1 | item_id:temperature) +
  (1 | variant_id:temperature) +
  (1 | item_id:judge_model) +
  (1 | variant_id:judge_model)

mod <- lmer(
  formula_recoded,
  data = df,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa",
                        optCtrl = list(maxfun = 50000))
)

if (any(grepl("singular", mod@optinfo$conv$lme4$messages))) {
  warning("Singular fit -- some variance components near zero.")
}

# -------- Variance components --------
vc <- as.data.frame(VarCorr(mod)) %>%
  select(grp, vcov) %>% rename(component = grp, variance = vcov)

# Label map matching 02_variance_decomposition.R
tle_labels <- c(
  "category"                = "between-category",
  "item_id"                 = "within-category item",
  "variant_id"              = "prompt",
  "item_id:variant_id"      = "item x prompt",
  "item_id:temperature"     = "item x temperature",
  "variant_id:temperature"  = "prompt x temperature",
  "item_id:judge_model"     = "item x judge",
  "variant_id:judge_model"  = "prompt x judge",
  "Residual"                = "generation"
)
vc <- vc %>% mutate(tle_label = ifelse(component %in% names(tle_labels),
                                       tle_labels[component], component))

# Fixed-effect sensitivity indices (population variance, 1/L)
fe <- fixef(mod)
temp_contrasts  <- c(0, fe[grepl("^temperature", names(fe))])
judge_contrasts <- c(0, fe[grepl("^judge_model", names(fe))])
sigma2_tau      <- var(temp_contrasts) * (length(temp_contrasts) - 1) / length(temp_contrasts)
sigma2_lambda   <- var(judge_contrasts) * (length(judge_contrasts) - 1) / length(judge_contrasts)

vc <- bind_rows(
  vc,
  tibble(component = "temperature (fixed)",  variance = sigma2_tau,
         tle_label = "temperature (design sensitivity)"),
  tibble(component = "judge_model (fixed)", variance = sigma2_lambda,
         tle_label = "judge model (design sensitivity)")
)

total_var <- sum(vc$variance)
vc <- vc %>% mutate(pct = 100 * variance / total_var) %>% arrange(desc(pct))

cat("\n=== RECODED pairwise variance decomposition ===\n")
print(vc %>% mutate(variance = round(variance, 5), pct = round(pct, 1)))

cat(sprintf("\nTotal variance: %.4f\n", total_var))

# -------- Comparison to original (position-based) --------
orig_vc <- read_csv("data/processed/variance_components_pairwise.csv",
                    show_col_types = FALSE) %>%
  filter(!is.na(variance))

cat("\n=== Side-by-side: ORIGINAL (position-based) vs RECODED ===\n")
cmp <- orig_vc %>%
  select(tle_label, variance_orig = variance) %>%
  full_join(vc %>% select(tle_label, variance_recoded = variance),
            by = "tle_label")

cmp$total_orig    <- sum(cmp$variance_orig, na.rm = TRUE)
cmp$total_recoded <- sum(cmp$variance_recoded, na.rm = TRUE)
cmp <- cmp %>% mutate(
  pct_orig    = 100 * variance_orig    / total_orig,
  pct_recoded = 100 * variance_recoded / total_recoded
) %>% select(tle_label, pct_orig, pct_recoded)

cmp <- cmp %>% arrange(desc(coalesce(pct_orig, 0)))
print(cmp %>% mutate(across(where(is.numeric), ~ round(., 1))))

write_csv(vc, "data/processed/variance_components_pairwise_recoded.csv")
write_csv(cmp, "data/processed/pairwise_decomp_comparison.csv")

cat("\nWrote:\n")
cat("  data/processed/variance_components_pairwise_recoded.csv\n")
cat("  data/processed/pairwise_decomp_comparison.csv\n")
