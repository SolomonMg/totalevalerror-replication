#!/usr/bin/env Rscript
# 40_arena_bt_human_bootstrap.R
# Per-model Bradley-Terry bootstrap SE using HUMAN votes only (no LLM judge).
# Resample the 4,676 battles with replacement, refit BT on human winners,
# compute SD of bootstrap log_pi draws per model.
#
# Output: data/processed/arena_human_bt_bootstrap_se.csv with
#   columns: model, mean_logpi_human, se_human, n_battles_avg.
#
# This sigma_human_pipeline feeds the gaming-surface comparison
# (analysis/41_arena_gaming_surface.R) against the LLM-judge sigmas
# already in data/processed/arena_bt_bootstrap_se.csv.

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)
set.seed(42)

N_BOOT <- 300
OUT_PATH <- "data/processed/arena_human_bt_bootstrap_se.csv"

# ---- BT MM fit (same as analysis/38) ----

fit_bt_mm <- function(p1, p2, wins1, wins2, n_iter = 600, tol = 1e-6) {
  players <- sort(unique(c(p1, p2)))
  K <- length(players)
  if (K < 2) return(NULL)
  idx <- setNames(seq_along(players), players)
  W <- matrix(0, K, K)
  for (i in seq_along(p1)) {
    W[idx[p1[i]], idx[p2[i]]] <- W[idx[p1[i]], idx[p2[i]]] + wins1[i]
    W[idx[p2[i]], idx[p1[i]]] <- W[idx[p2[i]], idx[p1[i]]] + wins2[i]
  }
  N <- W + t(W)
  w <- rowSums(W)
  pi <- rep(1, K)
  for (it in seq_len(n_iter)) {
    denom_sum <- numeric(K)
    for (i in seq_len(K)) {
      for (j in seq_len(K)) {
        if (i == j || N[i, j] == 0) next
        denom_sum[i] <- denom_sum[i] + N[i, j] / (pi[i] + pi[j])
      }
    }
    pi_new <- ifelse(denom_sum > 0, w / denom_sum, pi)
    pi_new <- pi_new / mean(pi_new)
    if (max(abs(pi_new - pi)) < tol) { pi <- pi_new; break }
    pi <- pi_new
  }
  setNames(pi, players)
}

fit_bt_human <- function(battle_df) {
  # battle_df: data.table with model_a, model_b, winner ("model_a"/"model_b"/"tie")
  p_a_win <- fcase(battle_df$winner == "model_a", 1,
                   battle_df$winner == "model_b", 0,
                   default = 0.5)
  swap <- battle_df$model_a > battle_df$model_b
  p1 <- fifelse(swap, battle_df$model_b, battle_df$model_a)
  p2 <- fifelse(swap, battle_df$model_a, battle_df$model_b)
  p1_win <- fifelse(swap, 1 - p_a_win, p_a_win)
  agg <- data.table(p1 = p1, p2 = p2, p1_win = p1_win)[
    , .(wins1 = sum(p1_win), wins2 = sum(1 - p1_win)), by = .(p1, p2)]
  pi <- fit_bt_mm(agg$p1, agg$p2, agg$wins1, agg$wins2)
  if (is.null(pi)) return(NULL)
  log(pi)
}

# ---- Load battles + human winners ----

cat("Loading battle metadata and human winners...\n")
battles_meta <- fread("data/processed/arena_battles_scored_input.csv",
                      select = c("battle_id", "model_a", "model_b", "winner"))

# Restrict to the 4,676 battles used in the paper (the ones present in the
# LLM-judge analysis).
preds <- fread("data/processed/arena_battle_predictions.csv",
               select = c("battle_id", "config"))
paper_battles <- unique(preds[config == "single_likert", battle_id])
cat(sprintf("Paper subset: %d battles\n", length(paper_battles)))

battles <- battles_meta[battle_id %in% paper_battles]
cat(sprintf("Battles after merge: %d\n", nrow(battles)))
cat("Winner distribution:\n")
print(battles[, .N, by = winner])

# Sanity: each row is one battle / one human vote.
stopifnot(uniqueN(battles$battle_id) == nrow(battles))

# ---- Bootstrap ----

cat(sprintf("Running %d bootstrap reps...\n", N_BOOT))
N <- nrow(battles)
all_models <- sort(unique(c(battles$model_a, battles$model_b)))
boot_logpi <- matrix(NA_real_, nrow = N_BOOT, ncol = length(all_models),
                     dimnames = list(NULL, all_models))

t0 <- Sys.time()
for (b in seq_len(N_BOOT)) {
  idx <- sample.int(N, N, replace = TRUE)
  log_pi <- fit_bt_human(battles[idx])
  if (!is.null(log_pi)) {
    common <- intersect(names(log_pi), all_models)
    boot_logpi[b, common] <- log_pi[common]
  }
  if (b %% 30 == 0) cat(sprintf("  rep %d/%d (%.1fs elapsed)\n",
                                b, N_BOOT,
                                as.numeric(Sys.time() - t0, units = "secs")))
}

# ---- Summarize ----

mean_logpi <- apply(boot_logpi, 2, mean, na.rm = TRUE)
se_logpi   <- apply(boot_logpi, 2, sd,   na.rm = TRUE)
n_present  <- apply(!is.na(boot_logpi), 2, sum)

out <- data.table(
  model            = names(mean_logpi),
  mean_logpi_human = mean_logpi,
  se_human         = se_logpi,
  n_boot_present   = n_present
)[order(-mean_logpi_human)]

# Median across models = headline sigma_human_pipeline
cat(sprintf("\nMedian se_human across %d models: %.4f\n",
            nrow(out), median(out$se_human)))
cat(sprintf("Mean   se_human across %d models: %.4f\n",
            nrow(out), mean(out$se_human)))

fwrite(out, OUT_PATH)
cat(sprintf("Saved %s\n", OUT_PATH))
