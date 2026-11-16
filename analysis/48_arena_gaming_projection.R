#!/usr/bin/env Rscript
# 48_arena_gaming_projection.R
# Project the best-of-K gaming gain (paper Fig. 6) from the 4,676-match sample to live-leaderboard vote counts.
#
# The paper's bootstrap (13c / 13e) resamples battles and, for pipelines that use a subset of the 5 prompt
# variants and 3 judges, also draws which variants and judges to use. So each pipeline's per-model variance has
#   a battle-sampling part, which shrinks as a model collects more battles (about 1 / n), and
#   a configuration part (which judge and prompt were drawn), which does not.
# Human voting and TEE (all 5 variants x all 3 judges) have no configuration draw, so their variance is all battle
# sampling. We estimate the battle-sampling part of the other pipelines by re-running the bootstrap with the
# configuration held fixed, for every fixed configuration, and averaging the per-model variance over them:
#   1_1 (one judge, one prompt): the 15 (variant, judge) pairs;  5_1: the 3 judges, all variants;
#   1_3: the 5 variants, all judges.
# Configuration variance = total (from the cached 13e draws) - battle sampling, floored at 0. Projection to n
# battles per model: var(n) = var_battle * n0 / n + var_config, with n0 the model's battles in the sample.
# Gain at K = sigma * E[max of K standard normals], sigma = median over models of sqrt(var(n)), as in 13e's
# normal approximation (the values the paper and figure report).
# Targets n: median launch votes on the live leaderboard, all launches and those with 10k+ debut votes
# (analysis 47, arena_launch_changes.csv, horizon 28), plus a grid.
#
# Assumptions: battle-sampling variance scales as 1/n with the sample's opponent mix; configuration variance does
# not depend on n; the 5 variants and 3 judges stand in for the configurations a leaderboard might fix.
#
# Inputs:  data/processed/arena_likert_clean.csv, arena_battles_scored_input.csv, arena_gaming_boot_draws.csv.gz,
#          arena_launch_changes.csv
# Outputs: data/processed/arena_gaming_fixed_config_var.csv (per model and pipeline: total, battle, config var)
#          data/processed/arena_gaming_projection.csv (pipeline x n: sigma and gain at K = 27 and 10, in Elo)

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
})
PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)
set.seed(42)
N_BOOT <- 300
ELO_PER_LOGPI <- 400 / log(10)
FIXED_PATH <- "data/processed/arena_gaming_fixed_config_var.csv"
OUT_PATH <- "data/processed/arena_gaming_projection.csv"

# Verbatim from 13e
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
e_max <- function(K, n_sim = 100000) {   # as in 13e
  set.seed(42); z <- matrix(rnorm(K * n_sim), nrow = K); mean(apply(z, 2, max))
}

# ---- Data, as in 13e ----
lik <- fread("data/processed/arena_likert_clean.csv")
battles_meta <- fread("data/processed/arena_battles_scored_input.csv", select = c("battle_id", "model_a", "model_b"))
lik_wide <- dcast(lik, battle_id + variant_id + judge_model ~ response_side, value.var = "score")
setnames(lik_wide, c("a", "b"), c("s_a", "s_b"))
lik_wide <- merge(lik_wide, battles_meta, by = "battle_id")
setorder(lik_wide, battle_id, variant_id, judge_model)
variants <- sort(unique(lik_wide$variant_id)); judges <- sort(unique(lik_wide$judge_model))
V_max <- length(variants); M_max <- length(judges)
lik_wide[, variant_idx := match(variant_id, variants)]
lik_wide[, judge_idx := match(judge_model, judges)]
battles_order <- sort(unique(lik_wide$battle_id))
N_BATTLES <- length(battles_order)
lik_wide[, battle_row := match(battle_id, battles_order)]
sa_arr <- array(NA_real_, dim = c(N_BATTLES, V_max, M_max))
sb_arr <- array(NA_real_, dim = c(N_BATTLES, V_max, M_max))
for (r in seq_len(nrow(lik_wide))) {
  sa_arr[lik_wide$battle_row[r], lik_wide$variant_idx[r], lik_wide$judge_idx[r]] <- lik_wide$s_a[r]
  sb_arr[lik_wide$battle_row[r], lik_wide$variant_idx[r], lik_wide$judge_idx[r]] <- lik_wide$s_b[r]
}
battle_models <- lik_wide[, .(model_a = first(model_a), model_b = first(model_b)), by = battle_row][order(battle_row)]
all_models <- sort(unique(c(battle_models$model_a, battle_models$model_b)))
n0 <- table(c(battle_models$model_a, battle_models$model_b))
cat(sprintf("Sample: %d battles, %d models, battles per model median %d\n", N_BATTLES, length(all_models),
            as.integer(median(n0))))

# ---- Fixed-configuration bootstraps (battle sampling only), cached ----
if (!file.exists(FIXED_PATH)) {
  boot_fixed <- function(v_set, m_set) {
    out <- matrix(NA_real_, nrow = N_BOOT, ncol = length(all_models), dimnames = list(NULL, all_models))
    for (b in seq_len(N_BOOT)) {
      rows <- sample.int(N_BATTLES, N_BATTLES, replace = TRUE)
      sa <- apply(sa_arr[rows, v_set, m_set, drop = FALSE], 1, mean, na.rm = TRUE)
      sb <- apply(sb_arr[rows, v_set, m_set, drop = FALSE], 1, mean, na.rm = TRUE)
      pred <- fcase(abs(sa - sb) < 0.5, "tie", sa > sb, "a", default = "b")
      bm <- battle_models[rows]
      pi <- fit_from_probs(bm$model_a, bm$model_b, fcase(pred == "a", 1, pred == "b", 0, default = 0.5))
      if (!is.null(pi)) { common <- intersect(names(pi), all_models); out[b, common] <- log(pi)[common] }
    }
    apply(out, 2, var, na.rm = TRUE)
  }
  configs <- c(
    map(seq_len(V_max * M_max), ~ list(key = "1_1", v = ((.x - 1) %% V_max) + 1, m = ((.x - 1) %/% V_max) + 1)),
    map(seq_len(M_max), ~ list(key = "5_1", v = seq_len(V_max), m = .x)),
    map(seq_len(V_max), ~ list(key = "1_3", v = .x, m = seq_len(M_max))))
  fixed <- imap_dfr(configs, function(cf, i) {
    cat(sprintf("  fixed config %d/%d (%s)\n", i, length(configs), cf$key))
    v <- boot_fixed(cf$v, cf$m)
    tibble(pipeline_key = cf$key, config = i, model = names(v), var_battle = unname(v))
  })
  fixed <- fixed %>% group_by(pipeline_key, model) %>% summarize(var_battle = mean(var_battle, na.rm = TRUE),
                                                                 n_configs = n(), .groups = "drop")
  draws <- fread("data/processed/arena_gaming_boot_draws.csv.gz")[!is.na(logpi)]
  tot <- draws[, .(var_total = var(logpi)), by = .(pipeline_key, model)] %>% as_tibble()
  comp <- tot %>% left_join(fixed, by = c("pipeline_key", "model")) %>%
    mutate(var_battle = if_else(pipeline_key %in% c("human", "5_3"), var_total, var_battle),
           var_config = pmax(var_total - var_battle, 0), n0 = as.integer(n0[model]))
  stopifnot(!anyNA(comp$var_battle), !anyNA(comp$n0))
  fwrite(comp, FIXED_PATH)
}
comp <- fread(FIXED_PATH) %>% as_tibble()

# ---- Projection ----
lc <- fread("data/processed/arena_launch_changes.csv")[horizon == 28]
targets <- tibble(n_label = c("sample (own n)", "live launch, median", "live launch, 10k+ debut votes",
                              "1,000", "10,000", "50,000"),
                  n = c(NA, median(lc$n0), median(lc$n0[lc$pretested]), 1000, 10000, 50000))
labels <- c(human = "Human voting", `1_1` = "One judge, one prompt", `5_1` = "5 prompts, one judge",
            `1_3` = "3 judges, one prompt", `5_3` = "TEE: 5 prompts x 3 judges")
em27 <- e_max(27); em10 <- e_max(10)
proj <- expand_grid(pipeline_key = names(labels), targets) %>%
  pmap_dfr(function(pipeline_key, n_label, n) {
    x <- comp %>% filter(.data$pipeline_key == !!pipeline_key)
    v <- if (is.na(n)) x$var_total else x$var_battle * x$n0 / n + x$var_config
    s <- median(sqrt(v))
    tibble(pipeline_key = pipeline_key, pipeline = labels[[pipeline_key]], n_label = n_label, n = n,
           sigma_elo = s * ELO_PER_LOGPI, gain_k27_elo = s * em27 * ELO_PER_LOGPI,
           gain_k10_elo = s * em10 * ELO_PER_LOGPI,
           config_share = median(x$var_config / x$var_total))
  })
fwrite(proj, OUT_PATH)

# sanity: at the sample's own n the projection reproduces 13e's normal approximation (56, 45, 46, 46, 32)
chk <- fread("data/processed/arena_gaming_best_of_k.csv")[K == 27, .(pipeline_key, normal_approx_elo)]
s0 <- proj %>% filter(n_label == "sample (own n)") %>% left_join(chk, by = "pipeline_key")
stopifnot(max(abs(s0$gain_k27_elo - s0$normal_approx_elo)) < 1e-6)
cat("\nBest-of-27 gain (Elo) by battles per model:\n")
print(as.data.frame(proj %>% transmute(pipeline, n_label, n = round(n), sigma = round(sigma_elo, 1),
                                       gain27 = round(gain_k27_elo, 1), gain10 = round(gain_k10_elo, 1),
                                       config_share = round(config_share, 2)) %>%
                      pivot_wider(id_cols = c(n_label, n), names_from = pipeline, values_from = gain27)),
      row.names = FALSE)
cat("\nMedian share of per-model variance from the configuration draw (sample n):\n")
print(as.data.frame(proj %>% filter(n_label == "sample (own n)") %>% select(pipeline, config_share)), row.names = FALSE)
cat("Saved", FIXED_PATH, "and", OUT_PATH, "\n")
