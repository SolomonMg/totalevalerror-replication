#!/usr/bin/env Rscript
# 41_arena_gaming_surface_comparison.R
# Compare best-of-K gaming surface across:
#   - Human voting (sigma_human_pipeline from analysis/40)
#   - Single LLM-judge cell, Likert (sigma_naive)
#   - TEE Likert (sigma_tee, full factorial)
#   - Single LLM-judge cell, Pairwise
#   - TEE Pairwise
#
# Output: data/processed/arena_gaming_surface_comparison.csv
# All SEs are in BT log_pi units. Inflation = E[max(Z_1...Z_K)] * sigma.

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

# E[max] of K i.i.d. standard normals (Monte Carlo, large N for accuracy)
e_max <- function(K, n_sim = 200000) {
  set.seed(42)
  z <- matrix(rnorm(K * n_sim), nrow = K)
  mean(apply(z, 2, max))
}

# ---- Load SEs ----

human <- fread("data/processed/arena_human_bt_bootstrap_se.csv")
llm   <- fread("data/processed/arena_bt_bootstrap_se.csv")

# Per pipeline: median sigma across models
sigma_human <- median(human$se_human)
sigma_llm <- llm[, .(sigma_naive = median(se_naive),
                     sigma_tee   = median(se_tee)), by = config]
cat("Median sigma per pipeline (BT log_pi):\n")
cat(sprintf("  Human voting:                 %.4f\n", sigma_human))
print(sigma_llm)

# ---- Gaming surface at K = 10 and K = 27 ----

K_values <- c(10, 27)
e10 <- e_max(10)
e27 <- e_max(27)
cat(sprintf("\nE[max(K=10)]  = %.4f\n", e10))
cat(sprintf("E[max(K=27)]  = %.4f\n", e27))

build_row <- function(label, sigma) {
  data.table(
    pipeline   = label,
    sigma      = sigma,
    inflation_K10 = e10 * sigma,
    inflation_K27 = e27 * sigma
  )
}

rows <- list(
  build_row("Human voting (Arena leaderboard)", sigma_human)
)

for (i in seq_len(nrow(sigma_llm))) {
  cfg <- sigma_llm$config[i]
  rows[[length(rows) + 1]] <- build_row(
    sprintf("Single LLM-cell, %s", cfg),
    sigma_llm$sigma_naive[i]
  )
  rows[[length(rows) + 1]] <- build_row(
    sprintf("TEE LLM, %s", cfg),
    sigma_llm$sigma_tee[i]
  )
}

out <- rbindlist(rows)
out[, ratio_to_human := sigma / sigma_human]
print(out)

OUT_PATH <- "data/processed/arena_gaming_surface_comparison.csv"
fwrite(out, OUT_PATH)
cat(sprintf("\nSaved %s\n", OUT_PATH))
