#!/usr/bin/env Rscript
# 13e_gaming_best_of_k.R
# Best-of-K gaming gain computed directly from the bootstrap fits, without the
# normal approximation sigma * E[max of K standard normals] that 13c uses.
#
# Step 1 (cached): reproduce the bootstrap log-pi draws behind the paper's Figure 4.
#   LLM pipelines: the 13c loop over the (V, M) grid, same seed and order.
#   Human leaderboard: the 40_arena_bt_human_bootstrap.R loop, same seed.
#   Each set is checked against the cached SEs before it is saved.
# Step 2: for each pipeline, model and K, the expected gain from reporting the best of
#   K independent runs is E[max of K draws] - E[draw], both taken over that model's
#   bootstrap distribution. E[max] is exact for K iid draws from the empirical
#   distribution: sum_i x_(i) * ((i/n)^K - ((i-1)/n)^K). The headline is the median over
#   models, as in 13c. The 95% interval resamples the 300 bootstrap fits (1,000 times),
#   so it covers Monte Carlo error from using 300 fits.
#
# Outputs: data/processed/arena_gaming_boot_draws.csv.gz (cached draws, five pipelines)
#          data/processed/arena_gaming_best_of_k.csv

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

N_BOOT  <- 300
N_OUTER <- 1000
K_GRID  <- c(1, 2, 3, 5, 10, 15, 20, 27, 30, 40)
ELO_PER_LOGPI <- 400 / log(10)
DRAWS_PATH <- "data/processed/arena_gaming_boot_draws.csv.gz"
OUT_PATH   <- "data/processed/arena_gaming_best_of_k.csv"
V_GRID <- c(1, 2, 3, 5)
M_GRID <- c(1, 2, 3)
KEEP <- c("1_1", "5_1", "1_3", "5_3")   # V_M keys used in the figure

if (!file.exists(DRAWS_PATH)) {
  # Verbatim from 13c / 40
  fit_bt_mm <- function(p1, p2, wins1, wins2, n_iter = 600, tol = 1e-6) {
    players <- sort(unique(c(p1, p2)))
    K <- length(players); if (K < 2) return(NULL)
    idx <- setNames(seq_along(players), players)
    W <- matrix(0, K, K)
    for (i in seq_along(p1)) {
      W[idx[p1[i]], idx[p2[i]]] <- W[idx[p1[i]], idx[p2[i]]] + wins1[i]
      W[idx[p2[i]], idx[p1[i]]] <- W[idx[p2[i]], idx[p1[i]]] + wins2[i]
    }
    N <- W + t(W); w <- rowSums(W); pi <- rep(1, K)
    for (it in seq_len(n_iter)) {
      denom <- numeric(K)
      for (i in seq_len(K)) for (j in seq_len(K)) {
        if (i == j || N[i, j] == 0) next
        denom[i] <- denom[i] + N[i, j] / (pi[i] + pi[j])
      }
      pi_new <- ifelse(denom > 0, w / denom, pi); pi_new <- pi_new / mean(pi_new)
      if (max(abs(pi_new - pi)) < tol) { pi <- pi_new; break }
      pi <- pi_new
    }
    setNames(pi, players)
  }
  fit_from_probs <- function(model_a, model_b, p_a_win) {
    swap <- model_a > model_b
    p1 <- fifelse(swap, model_b, model_a)
    p2 <- fifelse(swap, model_a, model_b)
    p1_win <- fifelse(swap, 1 - p_a_win, p_a_win)
    agg <- data.table(p1 = p1, p2 = p2, p1_win = p1_win)[
      , .(wins1 = sum(p1_win), wins2 = sum(1 - p1_win)), by = .(p1, p2)]
    fit_bt_mm(agg$p1, agg$p2, agg$wins1, agg$wins2)
  }
  to_long <- function(mat, key) {
    d <- as.data.table(mat)[, boot := .I]
    d <- melt(d, id.vars = "boot", variable.name = "model", value.name = "logpi",
              variable.factor = FALSE)
    d[, pipeline_key := key][]
  }

  # ---- LLM pipelines: 13c loop ----
  set.seed(42)
  lik <- fread("data/processed/arena_likert_clean.csv")
  battles_meta <- fread("data/processed/arena_battles_scored_input.csv",
                        select = c("battle_id", "model_a", "model_b"))
  lik_wide <- dcast(lik, battle_id + variant_id + judge_model ~ response_side,
                    value.var = "score")
  setnames(lik_wide, c("a", "b"), c("s_a", "s_b"))
  lik_wide <- merge(lik_wide, battles_meta, by = "battle_id")
  setorder(lik_wide, battle_id, variant_id, judge_model)
  variants <- sort(unique(lik_wide$variant_id))
  judges   <- sort(unique(lik_wide$judge_model))
  V_max <- length(variants); M_max <- length(judges)
  lik_wide[, variant_idx := match(variant_id, variants)]
  lik_wide[, judge_idx   := match(judge_model, judges)]
  battles_order <- sort(unique(lik_wide$battle_id))
  N_BATTLES <- length(battles_order)
  lik_wide[, battle_row := match(battle_id, battles_order)]
  sa_arr <- array(NA_real_, dim = c(N_BATTLES, V_max, M_max))
  sb_arr <- array(NA_real_, dim = c(N_BATTLES, V_max, M_max))
  for (r in seq_len(nrow(lik_wide))) {
    sa_arr[lik_wide$battle_row[r], lik_wide$variant_idx[r], lik_wide$judge_idx[r]] <- lik_wide$s_a[r]
    sb_arr[lik_wide$battle_row[r], lik_wide$variant_idx[r], lik_wide$judge_idx[r]] <- lik_wide$s_b[r]
  }
  battle_models <- lik_wide[, .(model_a = first(model_a), model_b = first(model_b)),
                            by = battle_row][order(battle_row)]
  all_models <- sort(unique(c(battle_models$model_a, battle_models$model_b)))

  grid_cached <- fread("data/processed/arena_gaming_surface_grid.csv")
  draws <- list()
  for (V in V_GRID) for (M in M_GRID) {
    cat(sprintf("  V=%d, M=%d\n", V, M))
    boot_logpi <- matrix(NA_real_, nrow = N_BOOT, ncol = length(all_models),
                         dimnames = list(NULL, all_models))
    for (b in seq_len(N_BOOT)) {
      boot_rows <- sample.int(N_BATTLES, N_BATTLES, replace = TRUE)
      v_set <- sample.int(V_max, V); m_set <- sample.int(M_max, M)
      sa_mean <- apply(sa_arr[boot_rows, v_set, m_set, drop = FALSE], 1, mean, na.rm = TRUE)
      sb_mean <- apply(sb_arr[boot_rows, v_set, m_set, drop = FALSE], 1, mean, na.rm = TRUE)
      pred <- fcase(abs(sa_mean - sb_mean) < 0.5, "tie",
                    sa_mean > sb_mean, "a", default = "b")
      bm <- battle_models[boot_rows]
      pi <- fit_from_probs(bm$model_a, bm$model_b,
                           fcase(pred == "a", 1, pred == "b", 0, default = 0.5))
      if (!is.null(pi)) {
        common <- intersect(names(pi), all_models)
        boot_logpi[b, common] <- log(pi)[common]
      }
    }
    sigma <- median(apply(boot_logpi, 2, sd, na.rm = TRUE), na.rm = TRUE)
    stopifnot(abs(sigma - grid_cached$sigma_logpi[grid_cached$V == V & grid_cached$M == M]) < 1e-9)
    key <- paste0(V, "_", M)
    if (key %in% KEEP) draws[[key]] <- to_long(boot_logpi, key)
  }

  # ---- Human leaderboard: script 40 loop ----
  set.seed(42)
  battles_h <- fread("data/processed/arena_battles_scored_input.csv",
                     select = c("battle_id", "model_a", "model_b", "winner"))
  preds <- fread("data/processed/arena_battle_predictions.csv",
                 select = c("battle_id", "config"))
  battles_h <- battles_h[battle_id %in% unique(preds[config == "single_likert", battle_id])]
  models_h <- sort(unique(c(battles_h$model_a, battles_h$model_b)))
  boot_h <- matrix(NA_real_, nrow = N_BOOT, ncol = length(models_h),
                   dimnames = list(NULL, models_h))
  for (b in seq_len(N_BOOT)) {
    d <- battles_h[sample.int(nrow(battles_h), nrow(battles_h), replace = TRUE)]
    pi <- fit_from_probs(d$model_a, d$model_b,
                         fcase(d$winner == "model_a", 1, d$winner == "model_b", 0,
                               default = 0.5))
    if (!is.null(pi)) {
      common <- intersect(names(pi), models_h)
      boot_h[b, common] <- log(pi)[common]
    }
  }
  se_cached <- fread("data/processed/arena_human_bt_bootstrap_se.csv")
  se_new <- apply(boot_h, 2, sd, na.rm = TRUE)
  stopifnot(max(abs(se_new[se_cached$model] - se_cached$se_human)) < 1e-9)
  draws[["human"]] <- to_long(boot_h, "human")

  fwrite(rbindlist(draws), DRAWS_PATH)
  cat("Bootstrap draws reproduce the cached SEs; saved", DRAWS_PATH, "\n")
}

# ---- Step 2: best-of-K gain ----

draws <- fread(DRAWS_PATH)[!is.na(logpi)]
labels <- c(human = "Human leaderboard", `1_1` = "One judge, one prompt",
            `5_1` = "5 prompts, one judge", `1_3` = "3 judges, one prompt",
            `5_3` = "TEE: 5 prompts × 3 judges")

# Expected gain of the best of K iid draws from x, for every K in K_GRID
gain_k <- function(x) {
  x <- sort(x); n <- length(x); i <- seq_len(n)
  vapply(K_GRID, function(K) sum(x * ((i / n)^K - ((i - 1) / n)^K)), numeric(1)) - mean(x)
}
# Median over models of the per-model gain; boot_idx picks which fits to use
median_gain <- function(d, boot_idx) {
  per_model <- d[, .(g = list(gain_k(logpi[match(boot_idx, boot, nomatch = 0)]))),
                 by = model]
  apply(do.call(rbind, per_model$g), 2, median) * ELO_PER_LOGPI
}

set.seed(42)
res <- rbindlist(lapply(names(labels), function(key) {
  d <- draws[pipeline_key == key]
  point <- median_gain(d, seq_len(N_BOOT))
  outer <- replicate(N_OUTER, median_gain(d, sample.int(N_BOOT, N_BOOT, replace = TRUE)))
  data.table(pipeline_key = key, pipeline = labels[[key]], K = K_GRID,
             gain_elo = point,
             ci_lo = apply(outer, 1, quantile, 0.025),
             ci_hi = apply(outer, 1, quantile, 0.975))
}))

# Normal approximation used by 13c, for comparison
e_max <- function(K, n_sim = 100000) {
  set.seed(42); z <- matrix(rnorm(K * n_sim), nrow = K); mean(apply(z, 2, max))
}
sig <- draws[, .(s = sd(logpi)), by = .(pipeline_key, model)][, .(sigma = median(s)),
                                                                by = pipeline_key]
res <- merge(res, sig, by = "pipeline_key")
res[, normal_approx_elo := sigma * vapply(K, e_max, numeric(1)) * ELO_PER_LOGPI][, sigma := NULL]
setorder(res, pipeline_key, K)
fwrite(res, OUT_PATH)

print(res[K %in% c(2, 10, 27, 40)], digits = 3)
cat("Saved", OUT_PATH, "\n")
