# 02c_lojo_judge_safety.R
# Leave-one-judge-out (LOJO) sensitivity for the AILuminate safety decomposition
# (SI si:lojo, tab:lojo; main text sec:safety). Promoted from the NeurIPS 2026
# rebuttal (rebuttal_neurips2026/analysis_B_lojo_safety.R, reviewer 6HLW Q1).
#
# Mirrors analysis/02_variance_decomposition_safety.R, refits the model on each
# 2-judge subset (dropping one judge in turn), and recomputes the Var(theta-hat)
# shares at the operational (averaged) design. Also reports the empirical
# per-judge SAFE rates, which show whether the between-judge mean offset
# (sigma^2_lambda) is driven by one judge.
#
# Output: data/processed/lojo_judge_safety.csv
#
# Usage: Rscript analysis/02c_lojo_judge_safety.R   (from project root, ~75 s)

library(tidyverse)
library(lme4)

set.seed(42)

input_path  <- "data/processed/safety_clean.csv"
out_path    <- "data/processed/lojo_judge_safety.csv"

df_all <- read_csv(input_path, show_col_types = FALSE) %>%
  filter(!is.na(outcome))

# Empirical per-judge SAFE rates (transparent; no model needed) ------------
judge_rates <- df_all %>%
  group_by(judge_model) %>%
  summarize(safe_rate = mean(outcome), n = n(), .groups = "drop")
cat("=== Empirical per-judge SAFE rates (full data) ===\n")
print(judge_rates)

# Fit function mirroring 02_variance_decomposition_safety.R -----------------
fit_and_decompose <- function(df, label) {
  df <- df %>%
    droplevels() %>%
    mutate(
      item_id     = as.factor(item_id),
      variant_id  = as.factor(variant_id),
      temperature = as.factor(temperature),
      judge_model = as.factor(judge_model),
      category    = as.factor(category)
    )

  n_judges_here <- n_distinct(df$judge_model)

  mod <- lmer(
    outcome ~ temperature + judge_model +
      (1 | category) +
      (1 | item_id) +
      (1 | variant_id) +
      (1 | item_id:variant_id) +
      (1 | item_id:temperature) +
      (1 | variant_id:temperature) +
      (1 | item_id:judge_model) +
      (1 | variant_id:judge_model) +
      (1 | item_id:variant_id:judge_model:temperature),
    data = df, REML = TRUE,
    control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000))
  )

  singular <- isSingular(mod)

  vc <- as.data.frame(VarCorr(mod))
  getv <- function(grp) {
    v <- vc$vcov[vc$grp == grp]
    if (length(v) == 0) 0 else v
  }
  s2_cat      <- getv("category")
  s2_item     <- getv("item_id")
  s2_prompt   <- getv("variant_id")
  s2_ip       <- getv("item_id:variant_id")
  s2_it       <- getv("item_id:temperature")
  s2_pt       <- getv("variant_id:temperature")
  s2_ij       <- getv("item_id:judge_model")
  s2_pj       <- getv("variant_id:judge_model")
  s2_eps_cell <- getv("item_id:variant_id:judge_model:temperature")
  s2_rho_rep  <- getv("Residual")

  # Fixed-effect design sensitivity indices (population variance, 1/L)
  fe <- fixef(mod)
  gm <- fe["(Intercept)"]
  temp_c  <- fe[grepl("temperature", names(fe))]
  temp_means  <- c(gm, gm + temp_c)
  s2_temp  <- sum((temp_means - mean(temp_means))^2) / length(temp_means)
  judge_c <- fe[grepl("judge_model", names(fe))]
  judge_means <- c(gm, gm + judge_c)
  s2_judge <- sum((judge_means - mean(judge_means))^2) / length(judge_means)

  # Operational (averaged) design parameters
  N <- n_distinct(df$item_id)
  K <- n_distinct(df$variant_id)
  L <- n_distinct(df$temperature)
  M <- n_judges_here
  R <- max(df$replication) + 1

  # Var(theta-hat) contributions at the averaged design (mirror dstudy_avg)
  contrib <- c(
    `judge model (design sensitivity)` = s2_judge / M,
    `within-category item`             = (s2_cat + s2_item) / N,
    `item x judge`                     = s2_ij / (N * M),
    `prompt x judge`                   = s2_pj / (K * M),
    `prompt`                           = s2_prompt / K,
    `item x prompt`                    = s2_ip / (N * K),
    `item x temperature`               = s2_it / (N * L),
    `prompt x temperature`             = s2_pt / (K * L),
    `temperature (design sensitivity)` = s2_temp / L,
    `cell-level (3-way+)`              = s2_eps_cell / (N * K * L * M),
    `replicate noise`                  = s2_rho_rep / (N * K * L * M * R)
  )
  total <- sum(contrib)
  pct <- 100 * contrib / total

  judge_attrib_pct <- pct["judge model (design sensitivity)"] +
    pct["item x judge"] + pct["prompt x judge"]

  tibble(
    config           = label,
    n_judges         = M,
    total_var_theta  = total,
    se_theta         = sqrt(total),
    pct_judge_ds     = as.numeric(pct["judge model (design sensitivity)"]),
    pct_item_judge   = as.numeric(pct["item x judge"]),
    pct_prompt_judge = as.numeric(pct["prompt x judge"]),
    pct_judge_total  = as.numeric(judge_attrib_pct),
    pct_item         = as.numeric(pct["within-category item"]),
    sigma2_lambda    = s2_judge,
    singular         = singular
  )
}

configs <- list(
  list(label = "Full (3 judges)",        keep = NULL),
  list(label = "Drop trinity (strict)",  keep = setdiff(judge_rates$judge_model, "arcee-ai/trinity-large-preview:free")),
  list(label = "Drop gemini-3-flash",    keep = setdiff(judge_rates$judge_model, "google/gemini-3-flash-preview")),
  list(label = "Drop gpt-oss-120b",      keep = setdiff(judge_rates$judge_model, "openai/gpt-oss-120b"))
)

results <- map_dfr(configs, function(cfg) {
  d <- if (is.null(cfg$keep)) df_all else df_all %>% filter(judge_model %in% cfg$keep)
  cat("\n\n### Fitting:", cfg$label, "(", n_distinct(d$judge_model), "judges ) ###\n")
  fit_and_decompose(d, cfg$label)
})

cat("\n\n=== LOJO RESULTS ===\n")
print(as.data.frame(results), digits = 4)

write_csv(results, out_path)
cat("\nSaved to:", out_path, "\n")

# Transparent sigma^2_lambda from empirical safe rates (pop variance of means)
cat("\n=== sigma^2_lambda from empirical SAFE-rate spread (pop var of judge means) ===\n")
sr <- judge_rates$safe_rate
names(sr) <- judge_rates$judge_model
popvar <- function(x) sum((x - mean(x))^2) / length(x)
cat(sprintf("All 3 judges:        sigma^2_lambda = %.6e\n", popvar(sr)))
for (j in names(sr)) {
  cat(sprintf("Drop %-42s sigma^2_lambda = %.6e\n", paste0(j, ":"), popvar(sr[setdiff(names(sr), j)])))
}
