#!/usr/bin/env Rscript
# 38_arena_bt_bootstrap.R
# Per-model Bradley-Terry bootstrap SE: naive (battles only) vs
# TEE-aware (battles + cells) for each pipeline config.
#
# For each pipeline:
#   Battle-only bootstrap: resample 4,676 battles with replacement, keep
#     the pipeline's per-battle prediction fixed, refit BT. SD of bootstrap
#     BT draws per model = naive SE.
#   Battle + cell bootstrap:
#     single_*: resample battles, pick ONE random cell per battle for the
#       prediction (varying which judge/variant a hypothetical reporter saw).
#     tee_*:    resample battles, resample 15 Likert cells (or 30 pairwise
#       cells) with replacement per battle, re-aggregate prediction.
#   SD of bootstrap BT draws = TEE-aware SE.
#
# Result: per-model (SE_naive, SE_tee, ratio). Naive pipelines have large
# ratios (single cell is noisy); TEE pipelines have ratios near 1 because
# aggregation damps cell sensitivity.

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(ggplot2)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)
set.seed(42)

N_BOOT <- 300
CACHE_PATH <- "data/processed/arena_bt_bootstrap_se.csv"

# ---- BT MM fit ----

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

# Given per-battle predictions (battle_id, model_a, model_b, pred in {a,b,tie}),
# aggregate to pair-wise wins and fit BT. Returns named vector of log(pi).
fit_bt_from_preds <- function(pred_df) {
  # encode wins
  p_a_win <- ifelse(pred_df$pred == "a", 1,
            ifelse(pred_df$pred == "b", 0, 0.5))
  # canonicalize p1 < p2 lexicographic
  swap <- pred_df$model_a > pred_df$model_b
  p1 <- ifelse(swap, pred_df$model_b, pred_df$model_a)
  p2 <- ifelse(swap, pred_df$model_a, pred_df$model_b)
  p1_win <- ifelse(swap, 1 - p_a_win, p_a_win)
  # aggregate
  dt <- data.table(p1 = p1, p2 = p2, p1_win = p1_win)
  w <- dt[, .(wins1 = sum(p1_win), wins2 = sum(1 - p1_win)), by = .(p1, p2)]
  pi <- fit_bt_mm(w$p1, w$p2, w$wins1, w$wins2)
  if (is.null(pi)) return(NULL)
  log(pi)
}

# ---- Build cell matrices ----

cat("Building cell-level prediction matrices...\n")

lik <- fread("data/processed/arena_likert_clean.csv")
battles_meta <- fread("data/processed/arena_battles_scored_input.csv",
                      select = c("battle_id", "model_a", "model_b"))

# Likert cells: (battle, variant, judge) -> (s_a, s_b)
lik_wide <- dcast(lik, battle_id + variant_id + judge_model ~ response_side,
                  value.var = "score")
setnames(lik_wide, c("a", "b"), c("s_a", "s_b"))
lik_wide <- merge(lik_wide, battles_meta, by = "battle_id")
lik_wide[, pred_cell := fcase(
  abs(s_a - s_b) < 0.5, "tie",
  s_a > s_b, "a",
  s_b > s_a, "b",
  default = "tie"
)]
# Assign a cell index 1..15 per battle (sorted by (variant, judge) for stability)
setorder(lik_wide, battle_id, variant_id, judge_model)
lik_wide[, cell_idx := seq_len(.N), by = battle_id]
N_LIK_CELLS <- max(lik_wide$cell_idx)
cat(sprintf("Likert: %d battles x %d cells = %d rows\n",
            uniqueN(lik_wide$battle_id), N_LIK_CELLS, nrow(lik_wide)))

# Matrix form for fast bootstrap: rows = battles (sorted), cols = cell_idx
battles_order <- sort(unique(lik_wide$battle_id))
lik_wide[, battle_row := match(battle_id, battles_order)]
# Encode pred_cell as 1=a, -1=b, 0=tie
lik_wide[, pred_num := fcase(pred_cell == "a", 1L, pred_cell == "b", -1L, default = 0L)]
pred_mat_lik <- matrix(NA_integer_, nrow = length(battles_order), ncol = N_LIK_CELLS)
for (r in seq_len(nrow(lik_wide))) {
  pred_mat_lik[lik_wide$battle_row[r], lik_wide$cell_idx[r]] <- lik_wide$pred_num[r]
}
# Also store s_a, s_b matrices for tee_likert cell bootstrap
sa_mat <- matrix(NA_real_, nrow = length(battles_order), ncol = N_LIK_CELLS)
sb_mat <- matrix(NA_real_, nrow = length(battles_order), ncol = N_LIK_CELLS)
for (r in seq_len(nrow(lik_wide))) {
  sa_mat[lik_wide$battle_row[r], lik_wide$cell_idx[r]] <- lik_wide$s_a[r]
  sb_mat[lik_wide$battle_row[r], lik_wide$cell_idx[r]] <- lik_wide$s_b[r]
}
# Model pair per battle row
battle_models <- lik_wide[, .(model_a = first(model_a), model_b = first(model_b)),
                          by = battle_row][order(battle_row)]

# Cell index for "single_likert" = gpt-oss-120b, variant_id == 0 (smallest)
single_lik_cell <- lik_wide[judge_model == "openai/gpt-oss-120b" & variant_id == 0,
                            unique(cell_idx)]
stopifnot(length(single_lik_cell) == 1)
cat(sprintf("single_likert canonical cell_idx = %d\n", single_lik_cell))

# ---- Pairwise cells: (battle, variant, judge, order) -> pred ----

pw <- fread("data/processed/arena_pairwise_clean.csv")
pw[, pred_cell := fcase(judgment == "A", "a", judgment == "B", "b", default = "tie")]
pw <- merge(pw, battles_meta, by = "battle_id")
setorder(pw, battle_id, variant_id, judge_model, order)
pw[, cell_idx := seq_len(.N), by = battle_id]
N_PW_CELLS <- max(pw$cell_idx)
cat(sprintf("Pairwise: %d battles x %d cells = %d rows\n",
            uniqueN(pw$battle_id), N_PW_CELLS, nrow(pw)))

pw_battles_order <- sort(unique(pw$battle_id))
# Use the same battle ordering as Likert since both cover the same 4,676 battles
stopifnot(identical(pw_battles_order, battles_order))
pw[, battle_row := match(battle_id, battles_order)]
pw[, pred_num := fcase(pred_cell == "a", 1L, pred_cell == "b", -1L, default = 0L)]
pred_mat_pw <- matrix(NA_integer_, nrow = length(battles_order), ncol = N_PW_CELLS)
for (r in seq_len(nrow(pw))) {
  pred_mat_pw[pw$battle_row[r], pw$cell_idx[r]] <- pw$pred_num[r]
}
# Also track a/b win count per cell for tee_pairwise majority vote
# (since all pairwise preds are in {a, b} under forced-choice, we can average directly)
pw_a_mat <- (pred_mat_pw == 1L) + 0  # 1 if a-win
pw_b_mat <- (pred_mat_pw == -1L) + 0

# single_pairwise canonical cell: gpt-oss-120b, variant 0, listed order
single_pw_cell <- pw[judge_model == "openai/gpt-oss-120b" & variant_id == 0 &
                       order == "listed", unique(cell_idx)]
stopifnot(length(single_pw_cell) == 1)
cat(sprintf("single_pairwise canonical cell_idx = %d\n", single_pw_cell))

# ---- Pipeline per-battle preds (the canonical aggregations) ----

# single_likert: the fixed cell
single_lik_pred_num <- pred_mat_lik[, single_lik_cell]
# single_pairwise: fixed cell
single_pw_pred_num  <- pred_mat_pw[, single_pw_cell]

# tee_likert: mean s_a vs mean s_b across all 15 cells
tee_lik_sa <- rowMeans(sa_mat, na.rm = TRUE)
tee_lik_sb <- rowMeans(sb_mat, na.rm = TRUE)
tee_lik_pred_num <- ifelse(abs(tee_lik_sa - tee_lik_sb) < 0.5, 0L,
                    ifelse(tee_lik_sa > tee_lik_sb, 1L, -1L))

# tee_pairwise: majority a-win rate across 30 cells; tie-zone half-width 0.175
# (calibrated to Arena's ~35% human tie rate; see analysis/39).
pw_cells_per_battle <- rowSums(!is.na(pred_mat_pw))
a_win_rate <- rowSums(pw_a_mat, na.rm = TRUE) /
  (rowSums(pw_a_mat, na.rm = TRUE) + rowSums(pw_b_mat, na.rm = TRUE))
tee_pw_pred_num <- ifelse(a_win_rate > 0.675, 1L,
                   ifelse(a_win_rate < 0.325, -1L, 0L))

# Encoding helpers
encode <- function(num) fcase(num == 1L, "a", num == -1L, "b", default = "tie")

# ---- Bootstrap driver ----

# Given a function that, for a given bootstrap iteration, produces a length-N
# vector of pred strings (one per sampled battle) plus the (model_a, model_b)
# pairs, return a matrix of bootstrapped log-pi (rows = reps, cols = models).
bootstrap_bt <- function(n_boot, pred_fn) {
  N <- nrow(battle_models)
  models <- sort(unique(c(battle_models$model_a, battle_models$model_b)))
  bt_mat <- matrix(NA_real_, nrow = n_boot, ncol = length(models))
  colnames(bt_mat) <- models
  t0 <- Sys.time()
  for (b in seq_len(n_boot)) {
    idx <- sample.int(N, N, replace = TRUE)  # battle indices
    preds <- pred_fn(idx)  # vector of "a"/"b"/"tie", length N
    pred_df <- data.frame(
      battle_id = idx,  # not really battle_id; just a placeholder
      model_a = battle_models$model_a[idx],
      model_b = battle_models$model_b[idx],
      pred    = preds,
      stringsAsFactors = FALSE
    )
    # Drop rows where pred is NA (e.g. missing cell)
    ok <- !is.na(pred_df$pred)
    if (sum(ok) < 100) next
    lp <- fit_bt_from_preds(pred_df[ok, ])
    if (!is.null(lp)) bt_mat[b, names(lp)] <- lp
    if (b %% 50 == 0)
      cat(sprintf("    rep %d/%d (elapsed %.1fs)\n",
                  b, n_boot,
                  as.numeric(Sys.time() - t0, units = "secs")))
  }
  bt_mat
}

# ---- Prediction functions ----

# Naive (battle-only) for each pipeline: fixed pred vector, just indexed
make_naive_fn <- function(pred_num_vec) {
  function(idx) encode(pred_num_vec[idx])
}

# TEE-aware for single_likert: hierarchical bootstrap.
# Each rep picks ONE (judge, variant) cell and applies it to ALL sampled battles
# -- the right analog to "if a different practitioner had made a different
# single-cell pipeline choice, what BT would they have reported?"
make_single_lik_tee_fn <- function() {
  function(idx) {
    cell_rep <- sample.int(N_LIK_CELLS, 1)
    preds <- pred_mat_lik[idx, cell_rep]
    encode(preds)
  }
}

# TEE-aware for single_pairwise: one (judge, variant, order) cell per rep
make_single_pw_tee_fn <- function() {
  function(idx) {
    cell_rep <- sample.int(N_PW_CELLS, 1)
    preds <- pred_mat_pw[idx, cell_rep]
    encode(preds)
  }
}

# TEE-aware for tee_likert: resample N_LIK_CELLS cells with replacement per
# battle, compute mean, threshold. Vectorized.
make_tee_lik_tee_fn <- function() {
  function(idx) {
    N <- length(idx)
    # For each of N battles, draw N_LIK_CELLS cell indices with replacement
    # Organize as an N x N_LIK_CELLS matrix
    cell_samp <- matrix(sample.int(N_LIK_CELLS, N * N_LIK_CELLS, replace = TRUE),
                        nrow = N, ncol = N_LIK_CELLS)
    # Expand to (N * N_LIK_CELLS) indexing pairs into sa_mat, sb_mat
    flat_idx_battle <- rep(idx, N_LIK_CELLS)
    flat_idx_cell   <- as.vector(cell_samp)
    sa_flat <- sa_mat[cbind(flat_idx_battle, flat_idx_cell)]
    sb_flat <- sb_mat[cbind(flat_idx_battle, flat_idx_cell)]
    # Reshape and row-mean
    sa_mean <- rowMeans(matrix(sa_flat, nrow = N, ncol = N_LIK_CELLS), na.rm = TRUE)
    sb_mean <- rowMeans(matrix(sb_flat, nrow = N, ncol = N_LIK_CELLS), na.rm = TRUE)
    encode(ifelse(abs(sa_mean - sb_mean) < 0.5, 0L,
           ifelse(sa_mean > sb_mean, 1L, -1L)))
  }
}

# TEE-aware for tee_pairwise: resample 30 cells with replacement, majority vote
make_tee_pw_tee_fn <- function() {
  function(idx) {
    N <- length(idx)
    cell_samp <- matrix(sample.int(N_PW_CELLS, N * N_PW_CELLS, replace = TRUE),
                        nrow = N, ncol = N_PW_CELLS)
    flat_idx_battle <- rep(idx, N_PW_CELLS)
    flat_idx_cell   <- as.vector(cell_samp)
    pred_flat <- pred_mat_pw[cbind(flat_idx_battle, flat_idx_cell)]
    # a-win rate per sampled battle
    a_mat  <- matrix(as.integer(pred_flat == 1L), nrow = N, ncol = N_PW_CELLS)
    b_mat  <- matrix(as.integer(pred_flat == -1L), nrow = N, ncol = N_PW_CELLS)
    a_rate <- rowSums(a_mat, na.rm = TRUE) /
      pmax(1, rowSums(a_mat, na.rm = TRUE) + rowSums(b_mat, na.rm = TRUE))
    encode(ifelse(a_rate > 0.675, 1L,
           ifelse(a_rate < 0.325, -1L, 0L)))
  }
}

# ---- Run bootstraps ----

if (!file.exists(CACHE_PATH)) {
  cat("\nRunning bootstraps (N_BOOT =", N_BOOT, " per config x method)...\n")
  configs <- list(
    list(name = "single_likert",  naive_fn = make_naive_fn(single_lik_pred_num),
         tee_fn = make_single_lik_tee_fn()),
    list(name = "tee_likert",     naive_fn = make_naive_fn(tee_lik_pred_num),
         tee_fn = make_tee_lik_tee_fn()),
    list(name = "single_pairwise", naive_fn = make_naive_fn(single_pw_pred_num),
         tee_fn = make_single_pw_tee_fn()),
    list(name = "tee_pairwise",    naive_fn = make_naive_fn(tee_pw_pred_num),
         tee_fn = make_tee_pw_tee_fn())
  )

  results <- list()
  for (cfg in configs) {
    cat(sprintf("\n[%s] naive bootstrap...\n", cfg$name))
    bt_naive <- bootstrap_bt(N_BOOT, cfg$naive_fn)
    cat(sprintf("[%s] TEE-aware bootstrap...\n", cfg$name))
    bt_tee   <- bootstrap_bt(N_BOOT, cfg$tee_fn)

    se_naive <- apply(bt_naive, 2, sd, na.rm = TRUE)
    se_tee   <- apply(bt_tee,   2, sd, na.rm = TRUE)
    mean_naive <- colMeans(bt_naive, na.rm = TRUE)
    mean_tee   <- colMeans(bt_tee,   na.rm = TRUE)
    n_naive <- apply(bt_naive, 2, function(x) sum(!is.na(x)))
    n_tee   <- apply(bt_tee,   2, function(x) sum(!is.na(x)))

    results[[cfg$name]] <- tibble(
      config = cfg$name,
      model  = names(se_naive),
      mean_logpi_naive = mean_naive,
      mean_logpi_tee   = mean_tee,
      se_naive = se_naive,
      se_tee   = se_tee,
      n_naive  = n_naive,
      n_tee    = n_tee
    )
  }
  bs_df <- bind_rows(results)
  write_csv(bs_df, CACHE_PATH)
  cat(sprintf("\nSaved %s\n", CACHE_PATH))
} else {
  cat("Using cached", CACHE_PATH, "\n")
  bs_df <- read_csv(CACHE_PATH, show_col_types = FALSE)
}

# ---- Summary ----

bs_df <- bs_df %>% mutate(ratio = se_tee / se_naive)

cat("\n=== SE ratios (tee / naive) by config ===\n")
print(bs_df %>% group_by(config) %>%
        summarize(n_models = n(),
                  mean_se_naive = mean(se_naive, na.rm = TRUE),
                  mean_se_tee   = mean(se_tee,   na.rm = TRUE),
                  median_ratio  = median(ratio, na.rm = TRUE),
                  mean_ratio    = mean(ratio, na.rm = TRUE),
                  q10_ratio     = quantile(ratio, 0.10, na.rm = TRUE),
                  q90_ratio     = quantile(ratio, 0.90, na.rm = TRUE)) %>%
        mutate(across(where(is.numeric), ~ round(., 3))))

# ---- Plot ----

config_order <- c("Likert (single)", "Likert (TEE)",
                  "Pairwise (single)", "Pairwise (TEE)")
bs_plot <- bs_df %>%
  mutate(config_label = case_when(
    config == "single_likert"    ~ "Likert (single)",
    config == "tee_likert"       ~ "Likert (TEE)",
    config == "single_pairwise"  ~ "Pairwise (single)",
    config == "tee_pairwise"     ~ "Pairwise (TEE)"
  ),
  config_label = factor(config_label, levels = config_order))

# Summary stats for annotation
bs_summary <- bs_plot %>%
  group_by(config_label) %>%
  summarize(median_ratio = median(ratio, na.rm = TRUE),
            q25 = quantile(ratio, 0.25, na.rm = TRUE),
            q75 = quantile(ratio, 0.75, na.rm = TRUE),
            .groups = "drop")

# ---- Panel A: horizontal dot plot of SE ratio per model by pipeline ----

ratio_max <- max(bs_plot$ratio, na.rm = TRUE)
p_ratio <- ggplot(bs_plot, aes(x = ratio, y = config_label)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_jitter(height = 0.18, alpha = 0.45, size = 1.7, color = "#0072B2") +
  geom_point(data = bs_summary,
             aes(x = median_ratio, y = config_label),
             inherit.aes = FALSE,
             size = 4, color = "#D55E00", shape = 18) +
  geom_text(data = bs_summary,
            aes(x = median_ratio, y = config_label,
                label = sprintf("median = %.2f", median_ratio)),
            inherit.aes = FALSE,
            hjust = -0.2, vjust = -1.0, size = 3.3, color = "#D55E00") +
  scale_x_log10(breaks = c(1, 1.25, 1.5, 2, 3, 5, 8),
                limits = c(0.85, ceiling(ratio_max * 1.1)),
                expand = expansion(mult = c(0.01, 0.02))) +
  labs(
    x = "SE(TEE-aware bootstrap) / SE(naive bootstrap)  (log scale)",
    y = NULL,
    subtitle = sprintf("Per-model ratio across %d Arena models (diamond: median). Dashed: naive = TEE-aware.",
                       length(unique(bs_df$model)))
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.subtitle = element_text(color = "gray30"))

# ---- Panel B: scatter SE_naive vs SE_tee, one color per pipeline ----

se_max <- max(c(bs_plot$se_naive, bs_plot$se_tee), na.rm = TRUE) * 1.05
p_scatter <- ggplot(bs_plot, aes(x = se_naive, y = se_tee, color = config_label)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 2.2, alpha = 0.75) +
  scale_color_manual(values = c("Likert (single)"    = "#D55E00",
                                "Likert (TEE)"       = "#0072B2",
                                "Pairwise (single)"  = "#CC79A7",
                                "Pairwise (TEE)"     = "#009E73")) +
  scale_x_continuous(limits = c(0, se_max), labels = scales::number_format(accuracy = 0.01)) +
  scale_y_continuous(limits = c(0, se_max), labels = scales::number_format(accuracy = 0.01)) +
  coord_fixed() +
  labs(
    x = "Naive bootstrap SE (battles only, fixed cell)",
    y = "TEE-aware bootstrap SE\n(battles + cell draw per rep)",
    color = NULL,
    subtitle = "Each point: one model's BT log-pi SE under one pipeline. Dashed: equal SE."
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank(),
        plot.subtitle = element_text(color = "gray30"))

# Combine
library(patchwork)
p_combined <- (p_ratio / p_scatter) +
  plot_annotation(
    title = "Naive single-cell pipelines under-report BT SE by 1.5x to 3.5x",
    subtitle = "TEE-aware bootstrap draws a fresh (judge, variant) cell per rep for single pipelines; resamples cells with replacement per battle for TEE pipelines.",
    theme = theme(plot.title = element_text(face = "bold", size = 13),
                  plot.subtitle = element_text(color = "gray30", size = 10.5))
  ) +
  plot_layout(heights = c(0.9, 1.1))

ggsave("figures/fig_arena_bt_bootstrap_se.pdf", p_combined, width = 8, height = 10)
ggsave("figures/fig_arena_bt_bootstrap_se.png", p_combined, width = 8, height = 10, dpi = 300)
cat("\nSaved figures/fig_arena_bt_bootstrap_se.{pdf,png}\n")
