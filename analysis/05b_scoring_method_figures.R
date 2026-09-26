# 05b_scoring_method_figures.R
# Publication-quality figures for the scoring method simulation (D.6/D.7).
# Reads CSVs from 06_sim_scoring_recovery.R and generates 3 figures.
#
# Figure 2: Degradation curves — compression x discretization
# Figure 3: Dose-response — absolute tau for Likert vs BT across 5 pathologies
# Figure 4: Bridging scatter — true latent score vs observed Likert score

library(tidyverse)
library(patchwork)
library(Matrix)

set.seed(42)

fig_dir  <- "figures"
data_dir <- "data/processed"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 05b_scoring_method_figures.R: Scoring Method Figures ===\n")

# =========================================================================
# Shared aesthetics (harmonized with paper palette)
# =========================================================================

method_colors <- c(
  "Likert mean"       = "#2166ac",
  "BT (tournament)"   = "#e08040"
)

method_shapes <- c(
  "Likert mean"       = 16,
  "BT (tournament)"   = 17
)

method_labels <- c(
  "likert_mean"   = "Likert mean",
  "bt_full_rr"    = "BT (full RR)",
  "bt_tournament" = "BT (tournament)",
  "bt_cost_eq"    = "BT (cost-equated)"
)

tau_label <- "Correlation (Kendall tau) with true latent scores"

theme_tle <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "grey90"),
    plot.title = element_text(face = "bold", size = 12, hjust = 0),
    plot.title.position = "plot",
    plot.subtitle = element_text(color = "gray40", size = 9, hjust = 0),
    legend.position = "bottom",
    strip.background = element_rect(fill = "grey85", color = "grey50"),
    strip.text = element_text(face = "bold", size = 10, color = "black"),
    axis.text = element_text(size = 9)
  )

# =========================================================================
# DGP constants and functions (from 06_sim_scoring_recovery.R)
# =========================================================================

N_CATEGORIES  <- 5
ITEMS_PER_CAT <- 30
N_ITEMS       <- N_CATEGORIES * ITEMS_PER_CAT
N_RESP_MODELS <- 4
N_RESPONSES   <- N_ITEMS * N_RESP_MODELS
N_PROMPTS     <- 3
N_JUDGES      <- 3
N_TEMPS       <- 2
N_REPS        <- 2

SIGMA2_DELTA_LEVELS <- c(0.30, 0.60)  # sweep item spread

VC_base <- list(
  sigma2_gamma       = 0.05,
  sigma2_omega       = 0.10,
  sigma2_delta_omega = 0.13,
  sigma2_beta        = 0.006,
  sigma2_phi         = 0.021,
  sigma2_resp_beta   = 0.034,
  sigma2_resp_phi    = 0.155,
  sigma2_beta_phi    = 0.009,
  sigma2_eps         = 3.0
)

# Default VC uses empirical sigma2_delta = 0.60
VC <- c(VC_base, list(sigma2_delta = 0.60))

generate_response_pool <- function(vc = VC) {
  resp_df <- expand.grid(
    model_id = seq_len(N_RESP_MODELS),
    item_id  = seq_len(N_ITEMS)
  )
  resp_df$resp_id  <- seq_len(nrow(resp_df))
  resp_df$category <- ceiling(resp_df$item_id / ITEMS_PER_CAT)
  gamma_v     <- rnorm(N_CATEGORIES, 0, sqrt(vc$sigma2_gamma))
  delta_v     <- rnorm(N_ITEMS, 0, sqrt(vc$sigma2_delta))
  omega_v     <- rnorm(N_RESP_MODELS, 0, sqrt(vc$sigma2_omega))
  delta_omega <- rnorm(N_RESPONSES, 0, sqrt(vc$sigma2_delta_omega))
  resp_df$theta <- 3.0 +
    gamma_v[resp_df$category] +
    delta_v[resp_df$item_id] +
    omega_v[resp_df$model_id] +
    delta_omega
  resp_df
}

generate_cell_means <- function(resp_df, vc = VC) {
  NR <- nrow(resp_df)
  K  <- N_PROMPTS; M <- N_JUDGES
  beta_v     <- rnorm(K, 0, sqrt(vc$sigma2_beta))
  phi_v      <- rnorm(M, 0, sqrt(vc$sigma2_phi))
  resp_beta  <- matrix(rnorm(NR * K, 0, sqrt(vc$sigma2_resp_beta)), NR, K)
  resp_phi   <- matrix(rnorm(NR * M, 0, sqrt(vc$sigma2_resp_phi)), NR, M)
  beta_phi   <- matrix(rnorm(K * M, 0, sqrt(vc$sigma2_beta_phi)), K, M)
  cell_mean <- array(NA_real_, dim = c(NR, K, M))
  for (m in seq_len(M)) {
    cell_mean[, , m] <- resp_df$theta + resp_beta + resp_phi[, m] +
      rep(beta_v + beta_phi[, m], each = NR) + phi_v[m]
  }
  cell_mean
}

observe_likert <- function(cell_mean, sigma2_eps, a_m,
                           kappa_ct, sigma_anchor,
                           nonlinear_gamma = 1.0, n_scale_points = 5,
                           n_temps = N_TEMPS, n_reps = N_REPS) {
  NR <- dim(cell_mean)[1]; K <- dim(cell_mean)[2]; M <- dim(cell_mean)[3]
  n_reps_total <- n_temps * n_reps
  all_y <- numeric(0); all_idx <- integer(0)
  for (m in seq_len(M)) {
    n_obs <- NR * K * n_reps_total
    v_m   <- rep(as.vector(cell_mean[, , m]), each = n_reps_total)
    scaled <- 3 + a_m[m] * (v_m - 3)
    eps    <- rnorm(n_obs, 0, sqrt(sigma2_eps))
    anchor <- if (sigma_anchor > 0) rnorm(n_obs, 0, sigma_anchor) else 0
    latent <- scaled + eps + anchor
    compressed <- 3 + kappa_ct * (latent - 3)
    if (nonlinear_gamma != 1.0) {
      dev <- compressed - 3
      compressed <- 3 + sign(dev) * abs(dev)^nonlinear_gamma
    }
    if (n_scale_points >= 100) {
      y <- pmin(pmax(compressed, 1), 5)
    } else {
      grid <- seq(1, 5, length.out = n_scale_points)
      step <- 4 / (n_scale_points - 1)
      idx_grid <- round((pmin(pmax(compressed, 1), 5) - 1) / step) + 1
      idx_grid <- pmax(1L, pmin(as.integer(n_scale_points), as.integer(idx_grid)))
      y <- grid[idx_grid]
    }
    idx <- rep(rep(seq_len(NR), times = K), each = n_reps_total)
    all_y   <- c(all_y, y)
    all_idx <- c(all_idx, idx)
  }
  as.numeric(tapply(all_y, all_idx, mean))
}

sample_diff_item_within_cat <- function(resp_df, cat, n_pairs) {
  cat_df <- resp_df[resp_df$category == cat, ]
  n_cat  <- nrow(cat_df)
  a_idx <- integer(0); b_idx <- integer(0)
  while (length(a_idx) < n_pairs) {
    need  <- (n_pairs - length(a_idx)) * 2L
    a_new <- sample(n_cat, need, replace = TRUE)
    b_new <- sample(n_cat, need, replace = TRUE)
    ok    <- cat_df$item_id[a_new] != cat_df$item_id[b_new]
    a_idx <- c(a_idx, a_new[ok])
    b_idx <- c(b_idx, b_new[ok])
  }
  tibble(
    resp_a  = cat_df$resp_id[a_idx[seq_len(n_pairs)]],
    resp_b  = cat_df$resp_id[b_idx[seq_len(n_pairs)]],
    stratum = "diff_same_cat"
  )
}

SAME_ITEM_PAIRS        <- N_ITEMS * choose(N_RESP_MODELS, 2)
DIFF_SAME_CAT_PAIRS    <- 700
DIFF_BETWEEN_CAT_PAIRS <- 275

sample_tournament_pairs <- function(resp_df) {
  parts <- list()
  same_item <- do.call(rbind, lapply(seq_len(N_ITEMS), function(i) {
    rids <- resp_df$resp_id[resp_df$item_id == i]
    cc <- combn(rids, 2)
    tibble(resp_a = cc[1, ], resp_b = cc[2, ], stratum = "same_item")
  }))
  parts[[1]] <- same_item
  per_cat <- DIFF_SAME_CAT_PAIRS %/% N_CATEGORIES
  for (cat in seq_len(N_CATEGORIES)) {
    parts[[length(parts) + 1]] <- sample_diff_item_within_cat(resp_df, cat, per_cat)
  }
  cat_combos <- combn(N_CATEGORIES, 2)
  per_cp <- ceiling(DIFF_BETWEEN_CAT_PAIRS / ncol(cat_combos))
  for (cp in seq_len(ncol(cat_combos))) {
    ra <- resp_df$resp_id[resp_df$category == cat_combos[1, cp]]
    rb <- resp_df$resp_id[resp_df$category == cat_combos[2, cp]]
    parts[[length(parts) + 1]] <- tibble(
      resp_a  = sample(ra, per_cp, replace = TRUE),
      resp_b  = sample(rb, per_cp, replace = TRUE),
      stratum = "diff_between_cat"
    )
  }
  bind_rows(parts) %>% mutate(pair_id = row_number())
}

observe_pairwise <- function(cell_mean, pairs_df, sigma2_eps, a_m,
                             n_prompts = N_PROMPTS, n_judges = N_JUDGES,
                             n_temps = N_TEMPS, n_reps = N_REPS) {
  n_pairs  <- nrow(pairs_df)
  noise_sd <- sqrt(sigma2_eps)
  cond <- expand.grid(
    rep    = seq_len(n_reps),
    temp   = seq_len(n_temps),
    judge  = seq_len(n_judges),
    prompt = seq_len(n_prompts)
  )
  n_cond  <- nrow(cond)
  n_total <- n_pairs * n_cond
  pair_idx   <- rep(seq_len(n_pairs), each = n_cond)
  prompt_idx <- rep(cond$prompt, times = n_pairs)
  judge_idx  <- rep(cond$judge, times = n_pairs)
  resp_a <- pairs_df$resp_a[pair_idx]
  resp_b <- pairs_df$resp_b[pair_idx]
  mean_a <- cell_mean[cbind(resp_a, prompt_idx, judge_idx)]
  mean_b <- cell_mean[cbind(resp_b, prompt_idx, judge_idx)]
  scaled_a <- 3 + a_m[judge_idx] * (mean_a - 3)
  scaled_b <- 3 + a_m[judge_idx] * (mean_b - 3)
  lat_a <- scaled_a + rnorm(n_total, 0, noise_sd)
  lat_b <- scaled_b + rnorm(n_total, 0, noise_sd)
  outcome <- as.integer(lat_a > lat_b)
  list(
    wins_a     = as.integer(tapply(outcome, pair_idx, sum)),
    n_per_pair = as.integer(tapply(outcome, pair_idx, length)),
    resp_a     = pairs_df$resp_a,
    resp_b     = pairs_df$resp_b
  )
}

fit_bt_mm <- function(resp_a, resp_b, wins_a, n_per_pair,
                      n_responses = N_RESPONSES, max_iter = 5000,
                      tol = 1e-3) {
  wins_b  <- n_per_pair - wins_a
  n_pairs <- length(resp_a)
  A_a <- sparseMatrix(i = seq_len(n_pairs), j = resp_a, x = 1,
                      dims = c(n_pairs, n_responses))
  A_b <- sparseMatrix(i = seq_len(n_pairs), j = resp_b, x = 1,
                      dims = c(n_pairs, n_responses))
  A   <- A_a + A_b
  W <- as.vector(crossprod(A_a, wins_a) + crossprod(A_b, wins_b))
  pi <- rep(1.0, n_responses)
  for (iter in seq_len(max_iter)) {
    pi_old <- pi
    sum_pi  <- pi[resp_a] + pi[resp_b]
    contrib <- n_per_pair / sum_pi
    D       <- as.vector(crossprod(A, contrib))
    pi <- ifelse(D > 0, W / D, 1e-10)
    pi <- pi / sum(pi) * n_responses
    delta <- max(abs(log(pmax(pi, 1e-20)) - log(pmax(pi_old, 1e-20))))
    if (delta < tol) break
  }
  scores <- log(pmax(pi, 1e-20))
  scores - mean(scores)
}


# =========================================================================
# Figure 3: Compression x Discretization (realistic: 1 prompt, 3 judges)
# Anchoring noise and DIF baked in at realistic levels.
# =========================================================================
cat("\n--- Figure 3: Compression x Discretization (baked-in anchoring + DIF) ---\n")

N_SIM_FIG3 <- 500

# Fixed realistic pathology levels (always present)
FIXED_SIGMA_ANCHOR  <- 0.2   # moderate anchoring noise (literature baseline)
FIXED_SIGMA_SLOPE   <- 0.15  # moderate DIF / scale-use heterogeneity
FIXED_NONLINEAR     <- 1.0   # no non-linear distortion (minor effect)

# Sweep these two
KAPPA_LEVELS        <- c(1.0, 0.7, 0.5, 0.3)
SCALE_POINTS_LEVELS <- c(100, 7, 5, 3)

conditions_grid <- expand.grid(
  kappa_ct       = KAPPA_LEVELS,
  n_scale_points = SCALE_POINTS_LEVELS,
  stringsAsFactors = FALSE
)
n_delta <- length(SIGMA2_DELTA_LEVELS)

fig3_csv <- file.path(data_dir, "sim_scoring_fig3.csv")

if (file.exists(fig3_csv)) {
  cat(sprintf("  Loading cached simulation data from %s\n", fig3_csv))
  full_df <- read_csv(fig3_csv, show_col_types = FALSE)
} else {
  cat(sprintf("  %d delta levels x %d conditions x %d replicates = %d total\n",
              n_delta, nrow(conditions_grid), N_SIM_FIG3,
              n_delta * nrow(conditions_grid) * N_SIM_FIG3))

  all_results <- list()

  for (s2d in SIGMA2_DELTA_LEVELS) {
    cat(sprintf("  --- sigma2_delta = %.2f ---\n", s2d))
    vc_run <- c(VC_base, list(sigma2_delta = s2d))

    for (sim in seq_len(N_SIM_FIG3)) {
      if (sim %% 50 == 0) cat(sprintf("    sim %d/%d\n", sim, N_SIM_FIG3))
      set.seed(42 + sim)

      rdf <- generate_response_pool(vc = vc_run)
      cm  <- generate_cell_means(rdf, vc = vc_run)
      cm_1p <- cm[, 1, , drop = FALSE]

      # BT tournament (once per replicate — immune to Likert pathologies)
      tp  <- sample_tournament_pairs(rdf)
      pw  <- observe_pairwise(cm, tp, vc_run$sigma2_eps, rep(1, N_JUDGES),
                              n_prompts = 1, n_judges = 3, n_temps = 1, n_reps = 1)
      bts <- fit_bt_mm(pw$resp_a, pw$resp_b, pw$wins_a, pw$n_per_pair)
      bt_tau <- cor(rdf$theta, bts, method = "kendall")

      # DIF: draw judge scale-use slopes once per replicate
      a_m <- rnorm(N_JUDGES, 1, FIXED_SIGMA_SLOPE)

      for (ci in seq_len(nrow(conditions_grid))) {
        cond <- conditions_grid[ci, ]

        obs <- observe_likert(cm_1p, vc_run$sigma2_eps, a_m,
                              kappa_ct = cond$kappa_ct,
                              sigma_anchor = FIXED_SIGMA_ANCHOR,
                              nonlinear_gamma = FIXED_NONLINEAR,
                              n_scale_points = cond$n_scale_points,
                              n_temps = 1, n_reps = 1)
        lik_tau <- cor(rdf$theta, obs, method = "kendall")

        all_results[[length(all_results) + 1]] <- tibble(
          sim = sim,
          sigma2_delta = s2d,
          kappa_ct = cond$kappa_ct,
          n_scale_points = cond$n_scale_points,
          likert_tau = lik_tau,
          bt_tau = bt_tau
        )
      }
    }
  }

  full_df <- bind_rows(all_results)
  write_csv(full_df, fig3_csv)
  cat(sprintf("  Saved simulation data to %s\n", fig3_csv))
}

# Count NaN replicates (from collapsed scores, e.g., 3-pt + k=0.3)
nan_counts <- full_df %>%
  group_by(kappa_ct, n_scale_points) %>%
  summarize(n_nan = sum(is.na(likert_tau)), n_total = n(), .groups = "drop") %>%
  filter(n_nan > 0)
if (nrow(nan_counts) > 0) {
  cat("  NaN replicates (collapsed Likert scores):\n")
  print(nan_counts)
}

cat(sprintf("  %d rows\n", nrow(full_df)))

# --- Figure 3: Compression x Discretization heatmap-style dot plot ---

fig3_summary <- full_df %>%
  pivot_longer(cols = c(likert_tau, bt_tau),
               names_to = "method_raw", values_to = "tau") %>%
  mutate(method = recode(method_raw,
    "likert_tau" = "Likert mean", "bt_tau" = "BT (tournament)")) %>%
  group_by(sigma2_delta, kappa_ct, n_scale_points, method) %>%
  summarize(
    tau_mean = mean(tau, na.rm = TRUE),
    tau_lo   = quantile(tau, 0.025, na.rm = TRUE),
    tau_hi   = quantile(tau, 0.975, na.rm = TRUE),
    .groups  = "drop"
  )

# Format labels
fig3_summary <- fig3_summary %>%
  mutate(
    scale_label = case_when(
      n_scale_points >= 100 ~ "Continuous",
      TRUE ~ paste0(n_scale_points, "-point")
    ),
    compression_label = paste0("\u03ba = ", kappa_ct),
    delta_label = paste0("Item spread: s2_d = ", sigma2_delta)
  )

scale_order <- c("3-point", "5-point", "7-point", "Continuous")
fig3_summary$scale_label <- factor(fig3_summary$scale_label, levels = scale_order)
fig3_summary$compression_label <- factor(fig3_summary$compression_label,
  levels = paste0("\u03ba = ", c(1.0, 0.7, 0.5, 0.3)))

fig3 <- ggplot(fig3_summary, aes(x = tau_mean, y = scale_label,
                                  color = method, shape = method)) +
  geom_linerange(aes(xmin = tau_lo, xmax = tau_hi), linewidth = 0.4,
                 position = position_dodge(width = 0.5), alpha = 0.5) +
  geom_point(size = 2.0, position = position_dodge(width = 0.5)) +
  facet_grid(compression_label ~ delta_label) +
  scale_color_manual(values = method_colors, name = NULL) +
  scale_shape_manual(values = method_shapes, name = NULL) +
  labs(
    title = "Compression x Discretization Interaction",
    subtitle = paste0(
      "Baseline: anchoring = ", FIXED_SIGMA_ANCHOR,
      ", DIF = ", FIXED_SIGMA_SLOPE,
      ", s2_eps = ", VC_base$sigma2_eps,
      ". Likert: 1 prompt, 3 judges.\n",
      "BT: 1,875-pair tournament, 3 judges. 95% CIs from ",
      N_SIM_FIG3, " replicates."
    ),
    x = "Correlation (Kendall tau) with true latent scores",
    y = NULL
  ) +
  theme_tle +
  theme(
    panel.spacing = unit(0.8, "lines"),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position = "bottom"
  )

ggsave(file.path(fig_dir, "manuscript_fig_scoring_3.pdf"), fig3,
       width = 7, height = 7.5)
cat("Saved: manuscript_fig_scoring_3.pdf\n")


# =========================================================================
# Figure 2: Degradation Curves (from inline simulation data)
# =========================================================================
cat("\n--- Figure 2: Degradation Curves ---\n")

# Panel (a): tau vs compression (averaged over discretization levels)
# Use empirical sigma2_delta = 0.60 for line plots
fig2a_data <- full_df %>%
  filter(sigma2_delta == 0.60) %>%
  pivot_longer(cols = c(likert_tau, bt_tau),
               names_to = "method_raw", values_to = "tau") %>%
  mutate(method_label = recode(method_raw,
    "likert_tau" = "Likert mean", "bt_tau" = "BT (tournament)")) %>%
  group_by(kappa_ct, method_label) %>%
  summarize(
    tau_mean = mean(tau, na.rm = TRUE),
    tau_se   = sd(tau, na.rm = TRUE) / sqrt(sum(!is.na(tau))),
    .groups  = "drop"
  )

p2a <- ggplot(fig2a_data, aes(x = kappa_ct, y = tau_mean,
                               color = method_label, shape = method_label)) +
  geom_ribbon(aes(ymin = tau_mean - tau_se, ymax = tau_mean + tau_se,
                  fill = method_label), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  scale_x_reverse(breaks = c(1.0, 0.7, 0.5, 0.3)) +
  scale_color_manual(values = method_colors, name = NULL) +
  scale_fill_manual(values = method_colors, guide = "none") +
  scale_shape_manual(values = method_shapes, name = NULL) +
  labs(
    subtitle = "(a) Central tendency compression: judges pull scores toward the scale midpoint",
    x = expression(kappa ~ "(fraction of score deviation retained; 0.3 = severe)"),
    y = "Correlation\n(Kendall tau)"
  ) +
  theme_tle

# Panel (b): tau vs discretization at severe compression
fig2b_data <- full_df %>%
  filter(sigma2_delta == 0.60, kappa_ct == 0.3) %>%
  pivot_longer(cols = c(likert_tau, bt_tau),
               names_to = "method_raw", values_to = "tau") %>%
  mutate(method_label = recode(method_raw,
    "likert_tau" = "Likert mean", "bt_tau" = "BT (tournament)")) %>%
  group_by(n_scale_points, method_label) %>%
  summarize(
    tau_mean = mean(tau, na.rm = TRUE),
    tau_se   = sd(tau, na.rm = TRUE) / sqrt(sum(!is.na(tau))),
    .groups  = "drop"
  )
fig2b_data$n_scale_points <- factor(fig2b_data$n_scale_points,
  levels = c(100, 7, 5, 3))

p2b <- ggplot(fig2b_data, aes(x = n_scale_points, y = tau_mean,
                               color = method_label, shape = method_label,
                               group = method_label)) +
  geom_ribbon(aes(ymin = tau_mean - tau_se, ymax = tau_mean + tau_se,
                  fill = method_label), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  scale_color_manual(values = method_colors, name = NULL) +
  scale_fill_manual(values = method_colors, guide = "none") +
  scale_shape_manual(values = method_shapes, name = NULL) +
  scale_x_discrete(labels = c("100" = "Continuous", "7" = "7-pt",
                               "5" = "5-pt", "3" = "3-pt")) +
  labs(
    subtitle = "(b) Discretization compounds compression (severe compression, kappa = 0.3)",
    x = "Number of Likert scale points (fewer = coarser binning)",
    y = "Correlation\n(Kendall tau)"
  ) +
  theme_tle

fig2 <- p2a / p2b +
  plot_layout(heights = c(1, 1)) +
  plot_annotation(
    title = "Scoring Method Recovery Under Pathologies",
    subtitle = paste0(
      "Baseline: anchoring = ", FIXED_SIGMA_ANCHOR,
      ", DIF = ", FIXED_SIGMA_SLOPE,
      ". ", N_SIM_FIG3, " replicates."
    ),
    theme = theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(color = "gray40", size = 9.5)
    )
  )

ggsave(file.path(fig_dir, "manuscript_fig_scoring_2.pdf"), fig2,
       width = 7, height = 6.5)
cat("Saved: manuscript_fig_scoring_2.pdf\n")


# =========================================================================
# Figure 4: Bridging — True Quality vs Observed Likert Score
# =========================================================================
cat("\n--- Figure 4: Signal Distortion Scatter ---\n")

# Generate one replicate
set.seed(42)
resp_df   <- generate_response_pool()
cell_mean <- generate_cell_means(resp_df)
a_m_none  <- rep(1, N_JUDGES)

# Slice cell_mean to 1 prompt for Likert (typical benchmark: 1 prompt, 3 judges)
cell_mean_1p <- cell_mean[, 1, , drop = FALSE]  # [NR, 1, 3]

# Generate BT tournament scores (3 judges per comparison, 1 prompt, 1 temp, 1 rep)
cat("  Generating BT tournament scores (1,875 pairs, 3 judges)...\n")
set.seed(42)
tourn_pairs <- sample_tournament_pairs(resp_df)

# Connectivity check
adj <- sparseMatrix(
  i = c(tourn_pairs$resp_a, tourn_pairs$resp_b),
  j = c(tourn_pairs$resp_b, tourn_pairs$resp_a),
  x = 1, dims = c(N_RESPONSES, N_RESPONSES)
)
# BFS from node 1
visited <- logical(N_RESPONSES)
queue <- 1L
visited[1] <- TRUE
while (length(queue) > 0) {
  node <- queue[1]; queue <- queue[-1]
  neighbors <- which(adj[node, ] > 0)
  new_nodes <- neighbors[!visited[neighbors]]
  visited[new_nodes] <- TRUE
  queue <- c(queue, new_nodes)
}
n_connected <- sum(visited)
cat(sprintf("  Graph connectivity: %d / %d nodes reachable (connected: %s)\n",
            n_connected, N_RESPONSES, ifelse(n_connected == N_RESPONSES, "YES", "NO")))
min_degree <- min(Matrix::rowSums(adj > 0))
cat(sprintf("  Min degree: %d, Mean degree: %.1f\n", min_degree, mean(Matrix::rowSums(adj > 0))))

pw_tourn   <- observe_pairwise(cell_mean, tourn_pairs, VC$sigma2_eps, a_m_none,
                               n_prompts = 1, n_judges = 3, n_temps = 1, n_reps = 1)
bt_scores <- fit_bt_mm(pw_tourn$resp_a, pw_tourn$resp_b, pw_tourn$wins_a, pw_tourn$n_per_pair)
# Rescale BT scores to 1-5 range for visual comparison
bt_rescaled <- 1 + 4 * (bt_scores - min(bt_scores)) / (max(bt_scores) - min(bt_scores))
cat(sprintf("  BT tournament (3 judges): tau = %.3f\n", cor(resp_df$theta, bt_rescaled, method = "kendall")))

# 4 pathology conditions
conditions <- list(
  list(kappa = 0.5, anchor = 0, gamma = 1.0, scale = 5,
       label = "Moderate compression (k=0.5)"),
  list(kappa = 0.3, anchor = 0, gamma = 1.0, scale = 5,
       label = "Severe compression (k=0.3)")
)

cond_levels <- sapply(conditions, `[[`, "label")

# Likert scatter data
likert_scatter <- bind_rows(lapply(conditions, function(cond) {
  set.seed(42)
  obs <- observe_likert(cell_mean_1p, VC$sigma2_eps, a_m_none,
                        kappa_ct = cond$kappa, sigma_anchor = cond$anchor,
                        nonlinear_gamma = cond$gamma,
                        n_scale_points = cond$scale,
                        n_temps = 1, n_reps = 1)
  tau_val <- cor(resp_df$theta, obs, method = "kendall")
  signal_ratio <- sd(obs) / sd(resp_df$theta)
  tibble(
    theta    = resp_df$theta,
    observed = obs,
    category = factor(resp_df$category),
    condition = cond$label,
    method   = "Likert mean",
    tau_text = sprintf("cor (tau) = %.3f", tau_val)
  )
}))

# BT scatter data (same across all conditions — immune to pathologies)
bt_tau <- cor(resp_df$theta, bt_rescaled, method = "kendall")

bt_scatter <- bind_rows(lapply(conditions, function(cond) {
  tibble(
    theta    = resp_df$theta,
    observed = bt_rescaled,
    category = factor(resp_df$category),
    condition = cond$label,
    method   = "BT (tournament)",
    tau_text = sprintf("cor (tau) = %.3f", bt_tau)
  )
}))

scatter_data <- bind_rows(likert_scatter, bt_scatter)
scatter_data$condition <- factor(scatter_data$condition, levels = cond_levels)
scatter_data$method <- factor(scatter_data$method,
                              levels = c("Likert mean", "BT (tournament)"))

# Category colors (5 categories)
cat_colors <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e")

fig4 <- ggplot(scatter_data, aes(x = theta, y = observed)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted",
              color = "grey65", linewidth = 0.3) +
  geom_point(aes(color = category), size = 0.4, alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE, color = "grey30",
              linewidth = 0.5, linetype = "dashed") +
  geom_text(data = scatter_data %>%
              distinct(condition, method, tau_text),
            aes(x = 1.0, y = 5.0, label = tau_text),
            hjust = 0, vjust = 1, size = 2.8, color = "grey20",
            inherit.aes = FALSE) +
  facet_grid(method ~ condition) +
  scale_color_manual(values = cat_colors, name = "Category") +
  coord_cartesian(xlim = c(0.5, 5.5), ylim = c(0.5, 5.5)) +
  labs(
    title = "How Scoring Pathologies Distort the Quality Signal",
    subtitle = "All panels: 5-point scale. Likert: 1 prompt, 3 judges. BT: 3 judges per comparison, 1,875-pair tournament.",
    x = expression("True latent quality " * theta),
    y = "Observed score (rescaled to 1-5)"
  ) +
  theme_tle +
  theme(
    legend.position = "none",
    panel.spacing = unit(0.8, "lines")
  )

ggsave(file.path(fig_dir, "manuscript_fig_scoring_4.pdf"), fig4,
       width = 12, height = 6)
cat("Saved: manuscript_fig_scoring_4.pdf\n")


# =========================================================================
# Figure 5: Binary Outcome — Likert vs BT at varying prevalence
# =========================================================================
cat("\n--- Figure 5: Binary outcome, varying prevalence ---\n")

N_SIM_FIG5 <- 500
PREVALENCE_LEVELS <- c(0.0005, 0.005, 0.05, 0.50)

# Binary observation: each judge thresholds latent value to 0/1
# sigma_anchor: per-observation anchoring noise (shifts threshold)
# threshold_shifts: per-judge threshold DIF (vector of length M)
observe_binary <- function(cell_mean, sigma2_eps, threshold,
                           sigma_anchor = 0, threshold_shifts = NULL,
                           n_temps = 1, n_reps = 1) {
  NR <- dim(cell_mean)[1]; K <- dim(cell_mean)[2]; M <- dim(cell_mean)[3]
  n_reps_total <- n_temps * n_reps
  if (is.null(threshold_shifts)) threshold_shifts <- rep(0, M)
  all_y <- numeric(0); all_idx <- integer(0)
  for (m in seq_len(M)) {
    n_obs <- NR * K * n_reps_total
    v_m   <- rep(as.vector(cell_mean[, , m]), each = n_reps_total)
    eps   <- rnorm(n_obs, 0, sqrt(sigma2_eps))
    anchor <- if (sigma_anchor > 0) rnorm(n_obs, 0, sigma_anchor) else 0
    latent <- v_m + eps + anchor
    y <- as.numeric(latent > (threshold + threshold_shifts[m]))
    idx <- rep(rep(seq_len(NR), times = K), each = n_reps_total)
    all_y   <- c(all_y, y)
    all_idx <- c(all_idx, idx)
  }
  as.numeric(tapply(all_y, all_idx, mean))
}

# Binary-specific parameters (lower noise, lower DIF than Likert)
BIN_SIGMA2_EPS   <- 0.01   # calibrated for kappa ~ 0.65 at 50% prevalence
BIN_SIGMA_DIF    <- 0.05   # modest threshold disagreement

# Compute threshold for target prevalence from the marginal distribution
# theta ~ N(3, sqrt(sigma2_signal)), noise ~ N(0, sqrt(sigma2_eps))
# P(theta + noise > threshold) = prevalence
sigma2_signal <- VC$sigma2_gamma + VC$sigma2_delta + VC$sigma2_omega +
  VC$sigma2_delta_omega + VC$sigma2_phi + VC$sigma2_beta
total_sd <- sqrt(sigma2_signal + BIN_SIGMA2_EPS)

cat(sprintf("  Binary params: sigma2_eps = %.2f, DIF = %.2f\n",
            BIN_SIGMA2_EPS, BIN_SIGMA_DIF))

fig5_csv <- file.path(data_dir, "sim_scoring_fig5.csv")

if (file.exists(fig5_csv)) {
  cat(sprintf("  Loading cached simulation data from %s\n", fig5_csv))
  bin_df <- read_csv(fig5_csv, show_col_types = FALSE)
} else {
  cat(sprintf("  Prevalence levels: %s\n",
              paste(PREVALENCE_LEVELS, collapse = ", ")))

  bin_results <- list()

  for (prev in PREVALENCE_LEVELS) {
    threshold <- 3.0 + qnorm(1 - prev) * total_sd
    cat(sprintf("  --- prevalence = %.3f, threshold = %.2f ---\n", prev, threshold))

    for (sim in seq_len(N_SIM_FIG5)) {
      if (sim %% 50 == 0) cat(sprintf("    sim %d/%d\n", sim, N_SIM_FIG5))
      set.seed(42 + sim)

      rdf <- generate_response_pool()
      cm  <- generate_cell_means(rdf)
      cm_1p <- cm[, 1, , drop = FALSE]

      # Binary Likert: 1 prompt, 3 judges, threshold to 0/1
      # DIF: per-judge threshold shifts
      dif_shifts <- rnorm(N_JUDGES, 0, BIN_SIGMA_DIF * total_sd)
      obs_bin <- observe_binary(cm_1p, BIN_SIGMA2_EPS, threshold,
                                threshold_shifts = dif_shifts,
                                n_temps = 1, n_reps = 1)
      bin_tau <- cor(rdf$theta, obs_bin, method = "kendall")

      # Inter-judge kappa: per-judge binary decisions, average pairwise Cohen's kappa
      judge_y <- matrix(NA_real_, N_RESPONSES, N_JUDGES)
      for (m in seq_len(N_JUDGES)) {
        latent_m <- cm_1p[, 1, m] + rnorm(N_RESPONSES, 0, sqrt(BIN_SIGMA2_EPS))
        judge_y[, m] <- as.integer(latent_m > (threshold + dif_shifts[m]))
      }
      kappas <- numeric(0)
      for (pair in combn(N_JUDGES, 2, simplify = FALSE)) {
        y1 <- judge_y[, pair[1]]; y2 <- judge_y[, pair[2]]
        p_o <- mean(y1 == y2)
        p1 <- mean(y1); p2 <- mean(y2)
        p_e <- p1 * p2 + (1 - p1) * (1 - p2)
        kappas <- c(kappas, if (p_e < 1) (p_o - p_e) / (1 - p_e) else 1)
      }
      mean_kappa <- mean(kappas)

      # BT tournament (same noise level as binary)
      tp  <- sample_tournament_pairs(rdf)
      pw  <- observe_pairwise(cm, tp, BIN_SIGMA2_EPS, rep(1, N_JUDGES),
                              n_prompts = 1, n_judges = 3, n_temps = 1, n_reps = 1)
      bts <- fit_bt_mm(pw$resp_a, pw$resp_b, pw$wins_a, pw$n_per_pair)
      bt_tau <- cor(rdf$theta, bts, method = "kendall")

      # Pairwise kappa: per-judge binary decisions on each pair
      n_tp <- nrow(tp)
      noise_sd <- sqrt(BIN_SIGMA2_EPS)
      pw_judge_y <- matrix(NA_integer_, n_tp, N_JUDGES)
      for (m in seq_len(N_JUDGES)) {
        lat_a <- cm[tp$resp_a, 1, m] + rnorm(n_tp, 0, noise_sd)
        lat_b <- cm[tp$resp_b, 1, m] + rnorm(n_tp, 0, noise_sd)
        pw_judge_y[, m] <- as.integer(lat_a > lat_b)
      }
      pw_kappas <- numeric(0)
      for (pair in combn(N_JUDGES, 2, simplify = FALSE)) {
        y1 <- pw_judge_y[, pair[1]]; y2 <- pw_judge_y[, pair[2]]
        p_o <- mean(y1 == y2)
        p1 <- mean(y1); p2 <- mean(y2)
        p_e <- p1 * p2 + (1 - p1) * (1 - p2)
        pw_kappas <- c(pw_kappas, if (p_e < 1) (p_o - p_e) / (1 - p_e) else 1)
      }
      pw_mean_kappa <- mean(pw_kappas)

      bin_results[[length(bin_results) + 1]] <- tibble(
        sim = sim,
        prevalence = prev,
        binary_tau = bin_tau,
        bt_tau = bt_tau,
        kappa_binary = mean_kappa,
        kappa_pairwise = pw_mean_kappa
      )
    }
  }

  bin_df <- bind_rows(bin_results)
  write_csv(bin_df, fig5_csv)
  cat(sprintf("  Saved simulation data to %s\n", fig5_csv))
}

# Impute tau=0 for degenerate binary classifications (zero-variance scores,
# i.e., classifier predicted the same label for every item). A classifier
# with no variation in predictions has zero ranking ability.
nan_bin <- bin_df %>%
  group_by(prevalence) %>%
  summarize(n_nan = sum(is.na(binary_tau)), .groups = "drop") %>%
  filter(n_nan > 0)
if (nrow(nan_bin) > 0) {
  cat("  Degenerate replicates (zero-variance binary scores, imputed as tau=0):\n")
  print(nan_bin)
}
bin_df <- bin_df %>%
  mutate(binary_tau = replace_na(binary_tau, 0))

cat(sprintf("  Generated %d rows\n", nrow(bin_df)))

# Summary
fig5_summary <- bin_df %>%
  pivot_longer(cols = c(binary_tau, bt_tau),
               names_to = "method_raw", values_to = "tau") %>%
  mutate(method = recode(method_raw,
    "binary_tau" = "Binary classification", "bt_tau" = "BT (tournament)")) %>%
  group_by(prevalence, method) %>%
  summarize(
    tau_mean = mean(tau, na.rm = TRUE),
    tau_lo   = quantile(tau, 0.025, na.rm = TRUE),
    tau_hi   = quantile(tau, 0.975, na.rm = TRUE),
    .groups  = "drop"
  )

# Compute mean kappa per prevalence level
kappa_by_prev <- bin_df %>%
  group_by(prevalence) %>%
  summarize(
    kappa_bin = mean(kappa_binary, na.rm = TRUE),
    kappa_pw  = mean(kappa_pairwise, na.rm = TRUE),
    .groups = "drop"
  )
cat("  Inter-judge kappa by prevalence:\n")
print(kappa_by_prev)

fig5_summary <- fig5_summary %>%
  mutate(
    prev_label = paste0("Prevalence: ", prevalence * 100, "%")
  ) %>%
  left_join(kappa_by_prev, by = "prevalence") %>%
  mutate(
    prev_label = paste0(prev_label,
      "  (bin \u03ba=", sprintf("%.2f", kappa_bin),
      ", pw \u03ba=", sprintf("%.2f", kappa_pw), ")")
  )

method_colors_5 <- c(
  "Binary classification" = "#2166ac",
  "BT (tournament)"       = "#e08040"
)
method_shapes_5 <- c(
  "Binary classification" = 16,
  "BT (tournament)"       = 17
)

fig5_summary$prev_label <- factor(fig5_summary$prev_label,
  levels = rev(unique(fig5_summary$prev_label[
    order(fig5_summary$prevalence)])))

fig5 <- ggplot(fig5_summary, aes(x = tau_mean, y = method,
                                  color = method, shape = method)) +
  geom_linerange(aes(xmin = tau_lo, xmax = tau_hi), linewidth = 0.4,
                 alpha = 0.5) +
  geom_point(size = 2.5) +
  facet_wrap(~ prev_label, ncol = 1) +
  scale_color_manual(values = method_colors_5, guide = "none") +
  scale_shape_manual(values = method_shapes_5, guide = "none") +
  labs(
    title = "Binary Outcome: Recovery by Prevalence",
    subtitle = paste0(
      "DIF: per-judge threshold shift (sd = ", BIN_SIGMA_DIF,
      " x total sd).\n",
      "Binary: 1 prompt, 3 judges (1,800 calls).\n",
      "BT: 1,875-pair tournament, 3 judges (5,625 calls). ",
      N_SIM_FIG5, " replicates."
    ),
    x = "Correlation (Kendall tau) with true latent scores",
    y = NULL,
    caption = NULL
  ) +
  theme_tle +
  theme(
    strip.background = element_rect(fill = "grey85", color = "grey50"),
    strip.text = element_text(face = "bold", size = 8, color = "black"),
    panel.spacing = unit(1, "lines"),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(fig_dir, "manuscript_fig_scoring_5.pdf"), fig5,
       width = 5, height = 6.5)
cat("Saved: manuscript_fig_scoring_5.pdf\n")


# =========================================================================
# Compound Figure: Scoring pathologies (top) + Prevalence sensitivity (bottom)
# =========================================================================

# Remove individual titles so the compound title is the only one
fig2_notitle <- fig2 & theme(plot.title = element_blank(), plot.subtitle = element_blank())
fig5_notitle <- fig5 + labs(title = NULL, subtitle = NULL)

fig_compound <- (wrap_elements(fig5_notitle) | wrap_elements(fig2_notitle)) +
  plot_layout(widths = c(1, 1.3)) +
  plot_annotation(
    title = "Scoring Method Robustness: Likert Pathologies and Prevalence Sensitivity",
    theme = theme(
      plot.title = element_text(face = "bold", size = 13)
    )
  )

ggsave(file.path(fig_dir, "manuscript_fig_scoring_compound.pdf"), fig_compound,
       width = 13, height = 5.5)
ggsave(file.path(fig_dir, "manuscript_fig_scoring_compound.png"), fig_compound,
       width = 13, height = 5.5, dpi = 300)
cat("Saved: manuscript_fig_scoring_compound.{pdf,png}\n")

cat("\n=== 05b_scoring_method_figures.R complete ===\n")
