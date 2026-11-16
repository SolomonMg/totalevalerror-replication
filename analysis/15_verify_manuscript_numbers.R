#!/usr/bin/env Rscript
# 15_verify_manuscript_numbers.R
#
# This script (1) runs the testthat suite that ports the historical numerical
# checks (variance-component shares, D-study reductions, sample sizes, etc.)
# and (2) prints the post-Phase-4 headline numbers the prose phase (Phase 6)
# will cite in the paper.
#
# The testthat checks live at tests/testthat/test_manuscript_numbers.R (same
# tolerances, same claims, same CSV sources). Run the full suite via:
#
#   Rscript tests/run_tests.R
#
# Phase 4 update: D-study formulas now split the lumped residual into
#   sigma2_eps_cell ("cell-level (3-way+)", NOT reducible by R)
#   sigma2_rho_rep  ("replicate noise", reducible by R via averaging)
# This script prints the new headline values so the prose phase can cite them.

suppressPackageStartupMessages({
  library(tidyverse)
})

root <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(root)

# ---- 1. Run the testthat suite -----------------------------------------------
cat("[15_verify_manuscript_numbers] Running testthat suite...\n\n")
status <- system2("Rscript", file.path("tests", "run_tests.R"))

# ---- 2. Print Phase 4 headline numbers ---------------------------------------
cat("\n\n=== Phase 4 headline numbers (post sigma2_eps_cell / sigma2_rho_rep split) ===\n")

pull_vc <- function(vc, label) {
  v <- vc$variance[vc$tle_label == label]
  if (length(v) == 0) 0 else v[1]
}
pull_pct <- function(vc, label) {
  v <- vc$pct_total[vc$tle_label == label]
  if (length(v) == 0) NA_real_ else v[1]
}

# --- Safety cost-efficiency frontier (paper §2.1, Figure 2) ---
vc_safety <- read_csv("data/processed/variance_components_safety.csv",
                       show_col_types = FALSE)
s2_alpha    <- pull_vc(vc_safety, "within-category item")
s2_gamma    <- pull_vc(vc_safety, "between-category")
s2_rho_var  <- pull_vc(vc_safety, "prompt")
s2_lambda   <- pull_vc(vc_safety, "judge model (design sensitivity)")
s2_ap       <- pull_vc(vc_safety, "item x prompt")
s2_al       <- pull_vc(vc_safety, "item x judge")
s2_pl       <- pull_vc(vc_safety, "prompt x judge")
s2_eps_cell <- pull_vc(vc_safety, "cell-level (3-way+)")
s2_rho_rep  <- pull_vc(vc_safety, "replicate noise")

dstudy_var_safety <- function(N, V, M, R) {
  (s2_gamma + s2_alpha)/N + s2_rho_var/V + s2_lambda/M +
    s2_ap/(N*V) + s2_al/(N*M) + s2_pl/(V*M) +
    s2_eps_cell/(N*V*M) +              # NOT reducible by R
    s2_rho_rep/(N*V*M*R)               # reducible by R
}

sq_se   <- sqrt(dstudy_var_safety(141, 1, 1, 1))
tee_se  <- sqrt(dstudy_var_safety(141, 3, 3, 1))
full_se <- sqrt(dstudy_var_safety(141, 5, 3, 8))

cat("\n--- Safety cost-efficiency frontier (paper §2.1, Figure 2) ---\n")
cat(sprintf("  Status quo (V=1, M=1, R=1):    SE = %.4f  (cost: 141 calls)\n", sq_se))
cat(sprintf("  TEE-guided (V=3, M=3, R=1):    SE = %.4f  (cost: 1,269 calls; %.1f%% reduction at 9x cost)\n",
            tee_se, (1 - tee_se/sq_se)*100))
cat(sprintf("  Full factorial (V=5, M=3, R=8): SE = %.4f  (cost: 16,920 calls; %.1f%% reduction at 120x cost)\n",
            full_se, (1 - full_se/sq_se)*100))
cat(sprintf("  Incremental Full vs TEE: +%.1f%% SE reduction\n",
            (1 - full_se/tee_se)*100))

# --- Per-domain sigma2_eps_cell vs sigma2_rho_rep shares ---
cat("\n--- Residual split shares (post-Phase 3 refit) ---\n")
cat(sprintf("  %-10s  %-22s  %-22s  %-12s\n",
            "Domain", "sigma2_eps_cell (%)", "sigma2_rho_rep (%)", "Combined (%)"))
for (domain in c("safety", "mmlu", "likert", "pairwise")) {
  path <- sprintf("data/processed/variance_components_%s.csv", domain)
  if (!file.exists(path)) next
  vc <- read_csv(path, show_col_types = FALSE)
  eps_pct <- pull_pct(vc, "cell-level (3-way+)")
  rho_pct <- pull_pct(vc, "replicate noise")
  if (is.na(eps_pct)) eps_pct <- 0
  if (is.na(rho_pct)) rho_pct <- 0
  cat(sprintf("  %-10s  %-22s  %-22s  %-12s\n",
              domain,
              sprintf("%.2f", eps_pct),
              sprintf("%.2f", rho_pct),
              sprintf("%.2f", eps_pct + rho_pct)))
}

# --- D-study Double-items reductions ---
cat("\n--- D-study 'Double items' reduction (paper §2.1, §2.2, §SI.8) ---\n")
for (domain in c("safety", "mmlu", "likert", "pairwise")) {
  path <- sprintf("data/processed/dstudy_%s.csv", domain)
  if (!file.exists(path)) next
  ds <- read_csv(path, show_col_types = FALSE)
  red <- ds$reduction_pct[ds$scenario == "Double items"]
  if (length(red) > 0) {
    cat(sprintf("  %-10s: %.1f%% reduction\n", domain, red))
  }
}

# --- MMLU budget allocation summary (paper §2.2, Figure 3) ---
mmlu_summary_path <- "data/processed/mmlu_budget_allocation_summary.csv"
if (file.exists(mmlu_summary_path)) {
  cat("\n--- MMLU budget allocation (paper §2.2, Figure 3) ---\n")
  bs <- read_csv(mmlu_summary_path, show_col_types = FALSE)
  print(bs %>%
          select(researcher, budget, rmse, coverage, mean_K, mean_R) %>%
          arrange(budget, researcher),
        n = 30)
  cat("\nKey claim: TEE-guided RMSE / Naive RMSE at largest budget = ")
  naive_rmse <- bs$rmse[bs$researcher == "Naive" & bs$budget == max(bs$budget)]
  tee_rmse   <- bs$rmse[bs$researcher == "TEE"   & bs$budget == max(bs$budget)]
  if (length(naive_rmse) == 1 && length(tee_rmse) == 1) {
    cat(sprintf("%.3f / %.3f = %.2fx (TEE is %.0f%% of naive RMSE)\n",
                tee_rmse, naive_rmse,
                naive_rmse / tee_rmse,
                100 * tee_rmse / naive_rmse))
  } else {
    cat("(missing)\n")
  }
}

cat("\n=== End headline numbers ===\n")
quit(save = "no", status = status)
