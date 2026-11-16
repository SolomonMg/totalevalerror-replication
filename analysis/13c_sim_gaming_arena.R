#!/usr/bin/env Rscript
# 13c_sim_gaming_arena.R
# Gaming-surface figure calibrated to Arena Likert variance components,
# with human-voting reference line.
#
# Inputs:
#   data/processed/arena_gaming_surface_grid.csv (LLM sigma per V,M)
#   data/processed/arena_human_bt_bootstrap_se.csv (sigma_human)
#
# Output: figures/fig_gaming_surface.pdf
#
# Plot: horizontal dot plot of best-of-K=27 gaming surface in Elo
# for five named pipeline configurations, with the human-voting reference
# line. The grid CSV is written by the bootstrap section above; if it
# already exists, the bootstrap is skipped (set FORCE_RERUN=TRUE to
# regenerate).

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(ggplot2)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)
set.seed(42)

N_BOOT  <- 300
K_GAME  <- 27
ELO_PER_LOGPI <- 400 / log(10)
GRID_PATH  <- "data/processed/arena_gaming_surface_grid.csv"
HUMAN_PATH <- "data/processed/arena_human_bt_bootstrap_se.csv"
FORCE_RERUN <- FALSE

V_GRID <- c(1, 2, 3, 5)
M_GRID <- c(1, 2, 3)

# ---- Bootstrap sigma_LLM(V, M) if grid not cached ----

if (FORCE_RERUN || !file.exists(GRID_PATH)) {
  cat("Running bootstrap for sigma_LLM(V, M) grid...\n")

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

  lik <- fread("data/processed/arena_likert_clean.csv")
  battles_meta <- fread("data/processed/arena_battles_scored_input.csv",
                        select = c("battle_id", "model_a", "model_b"))
  lik_wide <- dcast(lik, battle_id + variant_id + judge_model ~ response_side,
                    value.var = "score")
  setnames(lik_wide, c("a", "b"), c("s_a", "s_b"))
  lik_wide <- merge(lik_wide, battles_meta, by = "battle_id")
  setorder(lik_wide, battle_id, variant_id, judge_model)
  lik_wide[, cell_idx := seq_len(.N), by = battle_id]

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

  results <- list()
  for (V in V_GRID) for (M in M_GRID) {
    cat(sprintf("  V=%d, M=%d: ", V, M)); t0 <- Sys.time()
    boot_logpi <- matrix(NA_real_, nrow = N_BOOT, ncol = length(all_models),
                         dimnames = list(NULL, all_models))
    for (b in seq_len(N_BOOT)) {
      boot_rows <- sample.int(N_BATTLES, N_BATTLES, replace = TRUE)
      v_set <- sample.int(V_max, V); m_set <- sample.int(M_max, M)
      sa_sub <- sa_arr[boot_rows, v_set, m_set, drop = FALSE]
      sb_sub <- sb_arr[boot_rows, v_set, m_set, drop = FALSE]
      sa_mean <- apply(sa_sub, 1, mean, na.rm = TRUE)
      sb_mean <- apply(sb_sub, 1, mean, na.rm = TRUE)
      pred <- fcase(abs(sa_mean - sb_mean) < 0.5, "tie",
                    sa_mean > sb_mean, "a", default = "b")
      bm <- battle_models[boot_rows]
      p_a_win <- fcase(pred == "a", 1, pred == "b", 0, default = 0.5)
      swap <- bm$model_a > bm$model_b
      p1 <- fifelse(swap, bm$model_b, bm$model_a)
      p2 <- fifelse(swap, bm$model_a, bm$model_b)
      p1_win <- fifelse(swap, 1 - p_a_win, p_a_win)
      agg <- data.table(p1 = p1, p2 = p2, p1_win = p1_win)[
        , .(wins1 = sum(p1_win), wins2 = sum(1 - p1_win)), by = .(p1, p2)]
      pi <- fit_bt_mm(agg$p1, agg$p2, agg$wins1, agg$wins2)
      if (!is.null(pi)) {
        log_pi <- log(pi)
        common <- intersect(names(log_pi), all_models)
        boot_logpi[b, common] <- log_pi[common]
      }
    }
    se_per_model <- apply(boot_logpi, 2, sd, na.rm = TRUE)
    results[[length(results) + 1]] <- data.table(
      V = V, M = M, sigma_logpi = median(se_per_model, na.rm = TRUE)
    )
    cat(sprintf("median sigma = %.4f log_pi  (%.1fs)\n",
                median(se_per_model, na.rm = TRUE),
                as.numeric(Sys.time() - t0, units = "secs")))
  }
  fwrite(rbindlist(results), GRID_PATH)
}

# ---- Load + pick the named pipeline points ----

grid <- fread(GRID_PATH)
human <- fread(HUMAN_PATH)
sigma_human <- median(human$se_human)

E_max_K <- function(K, n_sim = 100000) {
  set.seed(42); z <- matrix(rnorm(K * n_sim), nrow = K); mean(apply(z, 2, max))
}
e_K <- E_max_K(K_GAME)

inflation_human_elo <- sigma_human * e_K * ELO_PER_LOGPI
grid[, inflation_elo := sigma_logpi * e_K * ELO_PER_LOGPI]

pull_grid <- function(V_, M_) grid[V == V_ & M == M_, inflation_elo]

named <- data.table(
  pipeline = c("Arena human leaderboard",
               "Single LLM judge\n(V=1, M=1)",
               "Multi-prompt only\n(V=5, M=1)",
               "Multi-judge only\n(V=1, M=3)",
               "TEE: prompts + judges\n(V=5, M=3)"),
  inflation_elo = c(inflation_human_elo,
                    pull_grid(1, 1), pull_grid(5, 1),
                    pull_grid(1, 3), pull_grid(5, 3)),
  category = c("Human", "Single config",
               "Partial averaging", "Partial averaging", "Full TEE")
)
named[, pipeline := factor(pipeline, levels = pipeline)]

cat("\nNamed pipelines (Elo at K=27):\n")
print(named)

# ---- Plot ----

x_min <- floor((min(named$inflation_elo) - 5) / 5) * 5
x_max <- ceiling((max(named$inflation_elo) + 8) / 5) * 5

p <- ggplot(named, aes(x = inflation_elo, y = fct_rev(pipeline),
                       color = category)) +
  geom_point(size = 4) +
  geom_text(aes(label = sprintf("%.0f Elo", inflation_elo)),
            hjust = -0.30, vjust = 0.5, size = 3.5, color = "black") +
  scale_color_manual(values = c("Human"             = "grey35",
                                "Single config"     = "#d73027",
                                "Partial averaging" = "#fc8d59",
                                "Full TEE"          = "#1a9850")) +
  scale_x_continuous(limits = c(x_min, x_max),
                     expand = expansion(mult = c(0, 0))) +
  labs(
    x = sprintf("Best-of-K=%d gaming surface (Elo)", K_GAME),
    y = NULL,
    color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.y = element_text(size = 10),
    legend.position = "none",
    plot.margin = margin(2, 2, 2, 2, "pt")
  )

ggsave("figures/fig_gaming_surface.pdf", p, width = 7.0, height = 3.6)
ggsave("figures/fig_gaming_surface.png", p, width = 7.0, height = 3.6, dpi = 300)
cat("\nSaved figures/fig_gaming_surface.{pdf,png}\n")
