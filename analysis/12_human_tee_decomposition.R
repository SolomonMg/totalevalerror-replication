# 12_human_tee_decomposition.R
# Apply TEE variance decomposition to the propaganda_llm human-coded audit data.
# 261 items × 9 human raters (fully crossed), binary favorability/refusal/vagueness judgments.
#
# Usage:
#   Rscript analysis/12_human_tee_decomposition.R

library(tidyverse)
library(lme4)

set.seed(42)

cat("=== 12_human_tee_decomposition.R ===\n")
cat("TEE variance decomposition for human-coded propaganda/LLM audit data\n\n")

# --- Load data ---
propaganda_root <- Sys.getenv("PROPAGANDA_LLM_GH_PATH",
                              file.path(Sys.getenv("HOME"), "workspace", "propaganda_llm_gh"))
input_path <- file.path(propaganda_root,
                        "code_public/study4_production_model_audit_human_audit/data/human_label_final/combined_final.csv")
if (!file.exists(input_path)) {
  stop("Human data not found at: ", input_path,
       "\nSet PROPAGANDA_LLM_GH_PATH to the propaganda_llm_gh repo root.")
}
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df_raw <- read.csv(input_path, stringsAsFactors = FALSE)
cat("Raw data:", nrow(df_raw), "rows,", length(unique(df_raw$question_ID)), "items,",
    length(unique(df_raw$ra)), "raters\n")

# Clean and prepare
df <- df_raw %>%
  mutate(
    item_id       = as.factor(question_ID),
    rater         = as.factor(ra),
    category      = as.factor(question_type),
    focus         = as.factor(question_focus),
    favorability  = as.integer(value_1_final),
    confidence    = as.numeric(value_2),
    refusal       = as.integer(value_3_final),
    vagueness     = as.integer(value_5_final)
  )

cat("Question types:", paste(levels(df$category), collapse = ", "), "\n")
cat("Question focus:", paste(levels(df$focus), collapse = ", "), "\n")

# ============================================================================
# 1. Summary statistics
# ============================================================================
cat("\n=== Summary Statistics ===\n")

cat("\n--- Favorability (value_1_final) ---\n")
cat("  N valid:", sum(!is.na(df$favorability)), "\n")
cat("  N missing:", sum(is.na(df$favorability)), "\n")
fav_rate <- mean(df$favorability, na.rm = TRUE)
cat("  Overall favorability rate (Chinese > English):", round(fav_rate, 3), "\n")
cat("  By question type:\n")
df %>%
  filter(!is.na(favorability)) %>%
  group_by(category) %>%
  summarise(
    n = n(),
    mean_fav = round(mean(favorability), 3),
    .groups = "drop"
  ) %>%
  print()

cat("\n--- Confidence (value_2) ---\n")
cat("  Mean:", round(mean(df$confidence, na.rm = TRUE), 2), "\n")
cat("  SD:", round(sd(df$confidence, na.rm = TRUE), 2), "\n")
cat("  Median:", median(df$confidence, na.rm = TRUE), "\n")

cat("\n--- Refusal (value_3_final) ---\n")
cat("  Overall refusal rate:", round(mean(df$refusal, na.rm = TRUE), 3), "\n")

cat("\n--- Vagueness (value_5_final) ---\n")
cat("  Overall vagueness rate:", round(mean(df$vagueness, na.rm = TRUE), 3), "\n")

# ============================================================================
# 2. Fit TEE-style lmer: FAVORABILITY (primary)
# ============================================================================
cat("\n\n=== Variance Decomposition: Favorability ===\n")

df_fav <- df %>% filter(!is.na(favorability))
cat("Fitting lmer on", nrow(df_fav), "observations...\n")

# Model: item nested within category, crossed with rater.
# With exactly 1 obs per item x rater cell, the item:rater interaction is
# NOT identifiable — it has as many levels as observations. The residual
# therefore absorbs BOTH within-cell noise and any item x rater interaction.
# This is the standard limitation of unreplicated fully-crossed designs.
# We fit without the interaction term and note this confounding.
mod_fav <- lmer(
  favorability ~ (1 | category) + (1 | item_id) + (1 | rater),
  data = df_fav,
  REML = TRUE,
  control = lmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 50000)
  )
)

cat("\nModel summary:\n")
print(summary(mod_fav))

# Check convergence / singularity
if (isSingular(mod_fav)) {
  cat("\nWARNING: Singular fit — some variance components estimated at zero.\n")
  cat("With binary outcomes and 1 obs/cell, the residual and item:rater interaction\n")
  cat("are partially confounded. The model may push one to the boundary.\n")
}

# Extract variance components
vc_fav <- as.data.frame(VarCorr(mod_fav))
cat("\nRaw VarCorr output:\n")
print(vc_fav)

# Map to TEE labels
# Note: residual absorbs both within-cell noise and item x rater interaction
# (confounded in unreplicated design)
tle_labels_fav <- c(
  "category"       = "between-category (question_type)",
  "item_id"        = "within-category item",
  "rater"          = "rater (judge)",
  "Residual"       = "residual (+ item x rater)"
)

vc_fav_clean <- vc_fav %>%
  select(grp, vcov, sdcor) %>%
  rename(component = grp, variance = vcov, sd = sdcor) %>%
  mutate(
    tle_label = tle_labels_fav[component],
    tle_label = ifelse(is.na(tle_label), component, tle_label),
    variance = pmax(variance, 0)
  )

total_var_fav <- sum(vc_fav_clean$variance)
vc_fav_clean <- vc_fav_clean %>%
  mutate(
    pct_total = round(100 * variance / total_var_fav, 2),
    outcome = "favorability"
  ) %>%
  arrange(desc(variance))

cat("\n=== TEE Variance Components: Favorability ===\n")
cat(sprintf("%-35s %10s %10s\n", "Component", "Variance", "% Total"))
cat(paste(rep("-", 57), collapse = ""), "\n")
for (i in seq_len(nrow(vc_fav_clean))) {
  cat(sprintf("%-35s %10.5f %9.1f%%\n",
              vc_fav_clean$tle_label[i],
              vc_fav_clean$variance[i],
              vc_fav_clean$pct_total[i]))
}
cat(sprintf("%-35s %10.5f %9.1f%%\n", "TOTAL", total_var_fav, 100))

# Interpretation note
cat("\nNote: With 1 observation per item x rater cell (unreplicated design),\n")
cat("the item x rater interaction is NOT identifiable and is confounded\n")
cat("with the residual. For binary outcomes, the residual also includes\n")
cat("Bernoulli sampling variance (p*(1-p) ~ 0.25 at p=0.5).\n")
cat("The residual therefore absorbs: (a) item x rater disagreement,\n")
cat("(b) Bernoulli noise, and (c) any other unmodeled variation.\n")

# ============================================================================
# 3. Fit TEE-style lmer: CONFIDENCE (secondary, continuous)
# ============================================================================
cat("\n\n=== Variance Decomposition: Confidence ===\n")

df_conf <- df %>% filter(!is.na(confidence))
cat("Fitting lmer on", nrow(df_conf), "observations...\n")

mod_conf <- lmer(
  confidence ~ (1 | category) + (1 | item_id) + (1 | rater),
  data = df_conf,
  REML = TRUE,
  control = lmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 50000)
  )
)

cat("\nModel summary:\n")
print(summary(mod_conf))

if (isSingular(mod_conf)) {
  cat("\nWARNING: Singular fit for confidence model.\n")
}

vc_conf <- as.data.frame(VarCorr(mod_conf))
vc_conf_clean <- vc_conf %>%
  select(grp, vcov, sdcor) %>%
  rename(component = grp, variance = vcov, sd = sdcor) %>%
  mutate(
    tle_label = tle_labels_fav[component],
    tle_label = ifelse(is.na(tle_label), component, tle_label),
    variance = pmax(variance, 0)
  )

total_var_conf <- sum(vc_conf_clean$variance)
vc_conf_clean <- vc_conf_clean %>%
  mutate(
    pct_total = round(100 * variance / total_var_conf, 2),
    outcome = "confidence"
  ) %>%
  arrange(desc(variance))

cat("\n=== TEE Variance Components: Confidence ===\n")
cat(sprintf("%-35s %10s %10s\n", "Component", "Variance", "% Total"))
cat(paste(rep("-", 57), collapse = ""), "\n")
for (i in seq_len(nrow(vc_conf_clean))) {
  cat(sprintf("%-35s %10.5f %9.1f%%\n",
              vc_conf_clean$tle_label[i],
              vc_conf_clean$variance[i],
              vc_conf_clean$pct_total[i]))
}
cat(sprintf("%-35s %10.5f %9.1f%%\n", "TOTAL", total_var_conf, 100))

# ============================================================================
# 4. Fit TEE-style lmer: REFUSAL and VAGUENESS
# ============================================================================
fit_binary_outcome <- function(outcome_name, outcome_vec, df_base) {
  cat(sprintf("\n\n=== Variance Decomposition: %s ===\n", outcome_name))
  df_sub <- df_base %>%
    mutate(.outcome = outcome_vec) %>%
    filter(!is.na(.outcome))
  cat("Fitting lmer on", nrow(df_sub), "observations...\n")

  mod <- lmer(
    .outcome ~ (1 | category) + (1 | item_id) + (1 | rater),
    data = df_sub,
    REML = TRUE,
    control = lmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 50000)
    )
  )

  if (isSingular(mod)) {
    cat("WARNING: Singular fit for", outcome_name, "model.\n")
  }

  vc <- as.data.frame(VarCorr(mod))
  vc_clean <- vc %>%
    select(grp, vcov, sdcor) %>%
    rename(component = grp, variance = vcov, sd = sdcor) %>%
    mutate(
      tle_label = tle_labels_fav[component],
      tle_label = ifelse(is.na(tle_label), component, tle_label),
      variance = pmax(variance, 0)
    )

  total_v <- sum(vc_clean$variance)
  vc_clean <- vc_clean %>%
    mutate(
      pct_total = round(100 * variance / total_v, 2),
      outcome = outcome_name
    ) %>%
    arrange(desc(variance))

  cat(sprintf("\n=== TEE Variance Components: %s ===\n", outcome_name))
  cat(sprintf("%-35s %10s %10s\n", "Component", "Variance", "% Total"))
  cat(paste(rep("-", 57), collapse = ""), "\n")
  for (i in seq_len(nrow(vc_clean))) {
    cat(sprintf("%-35s %10.5f %9.1f%%\n",
                vc_clean$tle_label[i],
                vc_clean$variance[i],
                vc_clean$pct_total[i]))
  }
  cat(sprintf("%-35s %10.5f %9.1f%%\n", "TOTAL", total_v, 100))

  return(list(model = mod, vc = vc_clean))
}

res_refusal   <- fit_binary_outcome("refusal",   df$refusal,   df)
res_vagueness <- fit_binary_outcome("vagueness", df$vagueness, df)

# ============================================================================
# 5. Item-level disagreement
# ============================================================================
cat("\n\n=== Item-Level Disagreement ===\n")

item_disagree <- df %>%
  filter(!is.na(favorability)) %>%
  group_by(item_id, category, focus) %>%
  summarise(
    n_raters     = n(),
    mean_fav     = mean(favorability),
    sd_fav       = sd(favorability),
    mean_conf    = mean(confidence, na.rm = TRUE),
    sd_conf      = sd(confidence, na.rm = TRUE),
    mean_refusal = mean(refusal, na.rm = TRUE),
    mean_vague   = mean(vagueness, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # Simple disagreement: SD of binary judgments (max at 0.5 when split is 50/50)
  mutate(
    disagreement = sd_fav,
    consensus = ifelse(mean_fav >= 0.5, "Chinese favored", "English favored")
  ) %>%
  arrange(desc(disagreement))

cat("\nItems with HIGHEST disagreement (most rater split):\n")
item_disagree %>%
  head(15) %>%
  select(item_id, category, focus, mean_fav, sd_fav, mean_conf) %>%
  print(n = 15)

cat("\nItems with LOWEST disagreement (most consensus):\n")
item_disagree %>%
  tail(15) %>%
  select(item_id, category, focus, mean_fav, sd_fav, mean_conf) %>%
  print(n = 15)

# Correlation between disagreement and confidence
valid_items <- item_disagree %>% filter(!is.na(disagreement) & !is.na(mean_conf))
cor_disagree_conf <- cor.test(valid_items$disagreement, valid_items$mean_conf)
cat("\nCorrelation between item disagreement (SD) and mean confidence:\n")
cat("  r =", round(cor_disagree_conf$estimate, 3), "\n")
cat("  95% CI: [", round(cor_disagree_conf$conf.int[1], 3), ",",
    round(cor_disagree_conf$conf.int[2], 3), "]\n")
cat("  p =", format.pval(cor_disagree_conf$p.value, digits = 3), "\n")

# Disagreement by category
cat("\nMean disagreement (SD of favorability) by question type:\n")
item_disagree %>%
  group_by(category) %>%
  summarise(
    n_items = n(),
    mean_disagreement = round(mean(disagreement, na.rm = TRUE), 3),
    mean_confidence   = round(mean(mean_conf, na.rm = TRUE), 2),
    .groups = "drop"
  ) %>%
  print()

# Disagreement by focus
cat("\nMean disagreement by question focus:\n")
item_disagree %>%
  group_by(focus) %>%
  summarise(
    n_items = n(),
    mean_disagreement = round(mean(disagreement, na.rm = TRUE), 3),
    mean_fav          = round(mean(mean_fav), 3),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_disagreement)) %>%
  print()

# ============================================================================
# 6. Rater-level effects
# ============================================================================
cat("\n\n=== Rater-Level Effects ===\n")

rater_stats <- df %>%
  filter(!is.na(favorability)) %>%
  group_by(rater) %>%
  summarise(
    n = n(),
    mean_fav     = round(mean(favorability), 3),
    mean_conf    = round(mean(confidence, na.rm = TRUE), 2),
    mean_refusal = round(mean(refusal, na.rm = TRUE), 3),
    mean_vague   = round(mean(vagueness, na.rm = TRUE), 3),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_fav))

cat("\nRater summary (sorted by favorability rate):\n")
print(rater_stats)

# Rater BLUPs from favorability model
rater_blups <- ranef(mod_fav)$rater
rater_blups$rater <- rownames(rater_blups)
names(rater_blups)[1] <- "blup_intercept"
cat("\nRater BLUPs (favorability model):\n")
print(rater_blups %>% arrange(desc(blup_intercept)))

# ============================================================================
# 7. ICC (inter-rater reliability)
# ============================================================================
cat("\n\n=== Inter-Rater Reliability ===\n")

# ICC from the variance components (treating rater as random)
vc_fav_vals <- setNames(vc_fav_clean$variance, vc_fav_clean$component)
sigma2_cat   <- vc_fav_vals["category"]
sigma2_item  <- vc_fav_vals["item_id"]
sigma2_rater <- vc_fav_vals["rater"]
sigma2_resid <- vc_fav_vals["Residual"]

# ICC(1): proportion of variance due to items (signal)
icc1 <- (sigma2_cat + sigma2_item) / total_var_fav
cat("ICC(1) — signal (item) share of total variance:", round(icc1, 3), "\n")

# ICC(k): reliability of the mean of k=9 raters
# Residual includes confounded item x rater interaction
icc_k <- (sigma2_cat + sigma2_item) /
  (sigma2_cat + sigma2_item + (sigma2_rater + sigma2_resid) / 9)
cat("ICC(k=9) — reliability of 9-rater mean:", round(icc_k, 3), "\n")

# ============================================================================
# 8. Save outputs
# ============================================================================
cat("\n\n=== Saving Results ===\n")

# Combine all variance components
vc_all <- bind_rows(vc_fav_clean, vc_conf_clean, res_refusal$vc, res_vagueness$vc)
vc_path <- file.path(output_dir, "human_tee_variance_components.csv")
write_csv(vc_all, vc_path)
cat("Variance components saved to:", vc_path, "\n")

# Item-level disagreement
item_path <- file.path(output_dir, "human_tee_item_disagreement.csv")
write_csv(item_disagree, item_path)
cat("Item disagreement saved to:", item_path, "\n")

# Rater statistics
rater_path <- file.path(output_dir, "human_tee_rater_stats.csv")
write_csv(rater_stats, rater_path)
cat("Rater stats saved to:", rater_path, "\n")

# Summary table for paper
cat("\n\n========================================\n")
cat("=== SUMMARY TABLE FOR PAPER ===\n")
cat("========================================\n\n")

cat("Human audit: 261 political prompts × 9 raters (fully crossed)\n")
cat("Binary favorability: which LLM completion (Chinese vs English) is more favorable?\n\n")

cat("--- Favorability Variance Decomposition ---\n")
cat(sprintf("%-35s %10s %10s\n", "Source", "Variance", "% Total"))
cat(paste(rep("=", 57), collapse = ""), "\n")
for (i in seq_len(nrow(vc_fav_clean))) {
  cat(sprintf("%-35s %10.5f %9.1f%%\n",
              vc_fav_clean$tle_label[i],
              vc_fav_clean$variance[i],
              vc_fav_clean$pct_total[i]))
}
cat(paste(rep("=", 57), collapse = ""), "\n")
cat(sprintf("%-35s %10.5f %9.1f%%\n", "Total", total_var_fav, 100))
cat(sprintf("\nICC(1) = %.3f  |  ICC(k=9) = %.3f\n", icc1, icc_k))
cat(sprintf("Overall favorability rate = %.3f\n", fav_rate))

cat("\n--- Confidence Variance Decomposition ---\n")
cat(sprintf("%-35s %10s %10s\n", "Source", "Variance", "% Total"))
cat(paste(rep("=", 57), collapse = ""), "\n")
for (i in seq_len(nrow(vc_conf_clean))) {
  cat(sprintf("%-35s %10.5f %9.1f%%\n",
              vc_conf_clean$tle_label[i],
              vc_conf_clean$variance[i],
              vc_conf_clean$pct_total[i]))
}
cat(paste(rep("=", 57), collapse = ""), "\n")
cat(sprintf("%-35s %10.5f %9.1f%%\n", "Total", total_var_conf, 100))

cat("\nDone.\n")
