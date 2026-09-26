#!/usr/bin/env Rscript
# 10_naive_vs_tee_se.R — Compute naive vs TEE SE comparison on real data
#
# Real-data version of Figure 1: each pipeline configuration (V prompts × M judges/SUTs × H temps)
# is a "researcher" who computes a naive SE from item means. The TEE SE (D-study at V=1, M=1)
# accounts for the omitted prompt and judge/SUT variance that the naive SE cannot see.
#
# Domains:
#   - likert, pairwise, safety: judge-layer TEE; target = overall grand mean
#     (averaged across judges, prompts, temps).
#   - mmlu: SUT-layer TEE; target = per-SUT grand mean (each SUT is evaluated
#     separately, so coverage is tested against that SUT's mean).

library(tidyverse)
set.seed(42)

cat("=== Naive vs TEE SE Comparison on Real Data ===\n\n")

# --- Helper: D-study variance at fixed temperature (Case 1/3) ---
# Under the paper's fixed-category convention, s2_kappa absorbs into s2_delta
# and both scale with N. Pass facet_label ("judge" or "SUT") to pick the right
# interaction component. Pass M=1 for single-model deployment.
dstudy_se <- function(vc, N, V, M, R, facet_label = "judge") {
  vc_vec <- setNames(vc$variance, vc$tle_label)
  get_vc <- function(label) if (label %in% names(vc_vec)) vc_vec[[label]] else 0

  s2_delta <- get_vc("within-category item")
  s2_kappa <- get_vc("between-category")
  s2_rho   <- get_vc("prompt")
  s2_ar    <- get_vc("item x prompt")
  s2_at    <- get_vc("item x temperature")
  s2_rt    <- get_vc("prompt x temperature")
  s2_al    <- get_vc(sprintf("item x %s", facet_label))
  s2_rl    <- get_vc(sprintf("prompt x %s", facet_label))
  s2_eps   <- get_vc("generation")

  # Categories fixed: s2_kappa absorbs into s2_delta, both /N.
  var_d <- (s2_kappa + s2_delta) / N + s2_rho / V + s2_ar / (N * V) +
    s2_at / N + s2_rt / V +
    s2_al / (N * M) + s2_rl / (V * M) +
    s2_eps / (N * V * M * R)
  sqrt(var_d)
}

results_all <- list()

for (scoring in c("likert", "pairwise", "safety", "mmlu")) {
  cat(sprintf("--- %s ---\n", toupper(scoring)))

  facet_col   <- if (scoring == "mmlu") "sut_model" else "judge_model"
  facet_label <- if (scoring == "mmlu") "SUT" else "judge"

  df <- read_csv(sprintf("data/processed/%s_clean.csv", scoring),
                 show_col_types = FALSE) %>%
    filter(!is.na(outcome))

  vc <- read_csv(sprintf("data/processed/variance_components_%s.csv", scoring),
                 show_col_types = FALSE)

  N <- n_distinct(df$item_id)
  V <- n_distinct(df$variant_id)
  M <- n_distinct(df[[facet_col]])
  H <- n_distinct(df$temperature)
  R <- df %>% count(item_id, variant_id, !!sym(facet_col), temperature) %>%
    pull(n) %>% median()

  # Target grand mean:
  # - For likert/pairwise/safety: overall mean (averaged across all configs).
  # - For MMLU: per-SUT mean (each SUT is benchmarked separately).
  if (scoring == "mmlu") {
    sut_gm <- df %>% group_by(sut_model) %>%
      summarize(grand_mean = mean(outcome), .groups = "drop")
    overall_gm <- mean(df$outcome)
    cat(sprintf("  Overall mean: %.4f | N=%d V=%d M=%d H=%d R=%.0f\n",
                overall_gm, N, V, M, H, R))
  } else {
    overall_gm <- mean(df$outcome)
    cat(sprintf("  Grand mean: %.4f | N=%d V=%d M=%d H=%d R=%.0f\n",
                overall_gm, N, V, M, H, R))
  }

  se_tee_full   <- dstudy_se(vc, N, V = V, M = M, R = R, facet_label = facet_label)
  se_tee_single <- dstudy_se(vc, N, V = 1, M = 1, R = R, facet_label = facet_label)

  cat(sprintf("  TEE SE (full design, V=%d M=%d): %.4f\n", V, M, se_tee_full))
  cat(sprintf("  TEE SE (single pipeline, V=1 M=1): %.4f\n", se_tee_single))

  configs <- df %>% distinct(variant_id, !!sym(facet_col), temperature)

  config_results <- map_dfr(seq_len(nrow(configs)), function(row_i) {
    cfg <- configs[row_i, ]
    sub <- df %>% filter(
      variant_id == cfg$variant_id,
      !!sym(facet_col) == cfg[[facet_col]],
      temperature == cfg$temperature
    )
    item_means <- sub %>% group_by(item_id) %>%
      summarise(m = mean(outcome, na.rm = TRUE), .groups = "drop")
    n_items <- nrow(item_means)
    config_mean <- mean(item_means$m, na.rm = TRUE)
    config_sd   <- sd(item_means$m, na.rm = TRUE)
    se_naive    <- config_sd / sqrt(n_items)

    # Target for coverage test
    if (scoring == "mmlu") {
      target <- sut_gm$grand_mean[sut_gm$sut_model == cfg[[facet_col]]]
    } else {
      target <- overall_gm
    }

    tibble(
      variant_id  = cfg$variant_id,
      facet_value = cfg[[facet_col]],
      temperature = cfg$temperature,
      target_mean = target,
      config_mean = config_mean,
      se_naive    = se_naive,
      ci_lo       = config_mean - 1.96 * se_naive,
      ci_hi       = config_mean + 1.96 * se_naive,
      covers_grand = target >= (config_mean - 1.96 * se_naive) &
                     target <= (config_mean + 1.96 * se_naive),
      tee_ci_lo   = config_mean - 1.96 * se_tee_single,
      tee_ci_hi   = config_mean + 1.96 * se_tee_single,
      tee_covers  = target >= (config_mean - 1.96 * se_tee_single) &
                    target <= (config_mean + 1.96 * se_tee_single)
    )
  })

  naive_coverage <- mean(config_results$covers_grand, na.rm = TRUE)
  tee_coverage   <- mean(config_results$tee_covers,  na.rm = TRUE)
  se_ratio       <- median(config_results$se_naive, na.rm = TRUE) / se_tee_single
  width_ratio    <- mean((config_results$ci_hi - config_results$ci_lo) /
                         (config_results$tee_ci_hi - config_results$tee_ci_lo),
                         na.rm = TRUE)
  underest_pct   <- 100 * (1 - width_ratio)

  cat(sprintf("  Naive SE (median): %.4f\n", median(config_results$se_naive, na.rm = TRUE)))
  cat(sprintf("  SE ratio (naive / TEE-single): %.2f  |  Width ratio: %.2f  |  Underest: %.1f%%\n",
              se_ratio, width_ratio, underest_pct))
  cat(sprintf("  Naive CI coverage: %.1f%% (%d/%d)\n",
              100 * naive_coverage, sum(config_results$covers_grand, na.rm = TRUE),
              nrow(config_results)))
  cat(sprintf("  TEE CI coverage: %.1f%% (%d/%d)\n",
              100 * tee_coverage, sum(config_results$tee_covers, na.rm = TRUE),
              nrow(config_results)))
  cat("\n")

  results_all[[scoring]] <- tibble(
    scoring          = scoring,
    n_configs        = nrow(config_results),
    se_tee_full      = se_tee_full,
    se_tee_single    = se_tee_single,
    se_naive_median  = median(config_results$se_naive, na.rm = TRUE),
    se_ratio         = se_ratio,
    width_ratio      = width_ratio,
    underest_pct     = underest_pct,
    naive_coverage   = naive_coverage,
    tee_coverage     = tee_coverage
  )

  write_csv(config_results, sprintf("data/processed/naive_vs_tee_%s.csv", scoring))
}

summary_df <- bind_rows(results_all)
write_csv(summary_df, "data/processed/naive_vs_tee_summary.csv")

cat("=== Summary Table ===\n")
summary_df %>%
  select(scoring, se_naive_median, se_tee_single, width_ratio, underest_pct,
         naive_coverage, tee_coverage) %>%
  mutate(across(where(is.numeric), ~ round(., 3))) %>%
  print()

cat("\nResults saved to data/processed/naive_vs_tee_*.csv\n")
