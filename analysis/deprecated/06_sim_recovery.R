# 06_sim_recovery.R
# Simulation: Which scoring method better recovers true item quality?
#
# Each arm uses its own empirically-calibrated noise structure.
# Likert noise from empirical Likert variance components;
# Pairwise noise from empirical pairwise variance components.
# Reference-point noise (DIF) affects Likert but cancels in pairwise.

library(tidyverse)

set.seed(42)

N_SIM <- 100
fig_dir <- "figures"
out_dir <- "data/processed"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 06_sim_recovery.R: Scoring Method Recovery Comparison ===\n")

# --- Design parameters ---
N_ITEMS   <- 150
N_PROMPTS <- 5
N_JUDGES  <- 3
N_REPS    <- 8

# --- Noise parameters from empirical variance components ---
# Likert (on 1-5 scale)
LIKERT_NOISE <- list(
  sigma2_delta  = 0.880,   # within-category item (signal)
  sigma2_beta   = 0.006,   # prompt main effect
  sigma2_phi    = 0.021,   # judge main effect
  sigma2_ab     = 0.034,   # item x prompt
  sigma2_aphi   = 0.155,   # item x judge
  sigma2_bphi   = 0.009,   # prompt x judge
  sigma2_eps    = 0.111    # generation (residual)
)

# Pairwise (on 0-1 probability scale, from LPM)
PAIRWISE_NOISE <- list(
  sigma2_delta  = 0.029,   # within-category item (signal)
  sigma2_beta   = 0.000,   # prompt main effect (zero in empirical)
  sigma2_phi    = 0.046,   # judge main effect
  sigma2_ab     = 0.008,   # item x prompt
  sigma2_aphi   = 0.067,   # item x judge
  sigma2_bphi   = 0.001,   # prompt x judge
  sigma2_eps    = 0.070    # generation (residual)
)

# Reference point variance: per-observation anchoring noise (DIF)
# Affects Likert (absolute judgment) but cancels in pairwise (comparative)
SIGMA2_REF_LEVELS <- c(0, 0.25, 0.5, 1.0, 2.0)

# Pairwise design: 40 opponents x 18 reps per pair
PAIRS_PER_ITEM <- 40
REPS_PER_PAIR  <- 18

cat(sprintf("Design: N=%d, K=%d, M=%d, R=%d, N_SIM=%d\n",
            N_ITEMS, N_PROMPTS, N_JUDGES, N_REPS, N_SIM))
cat(sprintf("Pairwise: %d opponents x %d reps per pair\n", PAIRS_PER_ITEM, REPS_PER_PAIR))
cat(sprintf("Likert signal SD = %.3f, Pairwise signal SD = %.3f\n",
            sqrt(LIKERT_NOISE$sigma2_delta), sqrt(PAIRWISE_NOISE$sigma2_delta)))
cat(sprintf("Reference point variance levels: %s\n",
            paste(SIGMA2_REF_LEVELS, collapse = ", ")))

# =========================================================================
# Simulation function
# =========================================================================
run_one_sim <- function(sigma2_ref, theta_std) {
  N <- N_ITEMS; K <- N_PROMPTS; M <- N_JUDGES; R <- N_REPS

  # -----------------------------------------------------------------------
  # LIKERT ARM — noise calibrated to empirical Likert variance components
  # -----------------------------------------------------------------------
  n_l <- LIKERT_NOISE

  # Scale true quality to Likert scale (mean 3, SD from empirical item variance)
  theta_likert <- 3.0 + theta_std * sqrt(n_l$sigma2_delta)

  # Draw Likert-scale random effects
  beta_j_l   <- rnorm(K, 0, sqrt(n_l$sigma2_beta))
  phi_m_l    <- rnorm(M, 0, sqrt(n_l$sigma2_phi))
  ab_ij_l    <- matrix(rnorm(N * K, 0, sqrt(n_l$sigma2_ab)), N, K)
  aphi_im_l  <- matrix(rnorm(N * M, 0, sqrt(n_l$sigma2_aphi)), N, M)
  bphi_jm_l  <- matrix(rnorm(K * M, 0, sqrt(n_l$sigma2_bphi)), K, M)

  # Build cell means
  cell_mean_l <- array(NA_real_, dim = c(N, K, M))
  for (m in seq_len(M)) {
    cell_mean_l[, , m] <- theta_likert + ab_ij_l + aphi_im_l[, m] +
      rep(beta_j_l + bphi_jm_l[, m], each = N) + phi_m_l[m]
  }

  # Generate observations with generation noise + reference-point noise
  n_obs <- N * K * M * R
  cell_vec_l <- rep(as.vector(cell_mean_l), each = R)
  eps_l <- rnorm(n_obs, 0, sqrt(n_l$sigma2_eps))
  ref_shift <- rnorm(n_obs, 0, sqrt(sigma2_ref))
  y_likert <- pmin(pmax(round(cell_vec_l + eps_l + ref_shift), 1), 5)

  item_idx_l <- rep(rep(seq_len(N), each = R), times = K * M)
  theta_hat_likert <- tapply(y_likert, item_idx_l, mean)

  # -----------------------------------------------------------------------
  # PAIRWISE ARM — noise calibrated to empirical pairwise variance components
  # -----------------------------------------------------------------------
  n_p <- PAIRWISE_NOISE

  # Scale true quality to pairwise scale (mean 0.5, SD from empirical item variance)
  theta_pairwise <- 0.5 + theta_std * sqrt(n_p$sigma2_delta)

  # Draw pairwise-scale random effects
  beta_j_p   <- rnorm(K, 0, sqrt(n_p$sigma2_beta))
  phi_m_p    <- rnorm(M, 0, sqrt(n_p$sigma2_phi))
  ab_ij_p    <- matrix(rnorm(N * K, 0, sqrt(n_p$sigma2_ab)), N, K)
  aphi_im_p  <- matrix(rnorm(N * M, 0, sqrt(n_p$sigma2_aphi)), N, M)
  bphi_jm_p  <- matrix(rnorm(K * M, 0, sqrt(n_p$sigma2_bphi)), K, M)

  # Build cell means on pairwise scale
  cell_mean_p <- array(NA_real_, dim = c(N, K, M))
  for (m in seq_len(M)) {
    cell_mean_p[, , m] <- theta_pairwise + ab_ij_p + aphi_im_p[, m] +
      rep(beta_j_p + bphi_jm_p[, m], each = N) + phi_m_p[m]
  }

  # Generate pairwise comparisons
  n_pairs <- (N * PAIRS_PER_ITEM) %/% 2
  i1 <- sample(N, n_pairs, replace = TRUE)
  i2 <- sample(N, n_pairs, replace = TRUE)
  same <- i1 == i2
  i2[same] <- ((i2[same]) %% N) + 1

  wins  <- numeric(N)
  games <- numeric(N)
  bt_i1 <- integer(n_pairs)
  bt_i2 <- integer(n_pairs)
  bt_w1 <- integer(n_pairs)
  bt_n  <- integer(n_pairs)

  for (p in seq_len(n_pairs)) {
    j_samp <- sample(K, REPS_PER_PAIR, replace = TRUE)
    m_samp <- sample(M, REPS_PER_PAIR, replace = TRUE)

    # Shared reference point per comparison (cancels in the difference)
    ref <- rnorm(REPS_PER_PAIR, 0, sqrt(sigma2_ref))

    lat1 <- sapply(seq_len(REPS_PER_PAIR), function(r) {
      cell_mean_p[i1[p], j_samp[r], m_samp[r]] +
        rnorm(1, 0, sqrt(n_p$sigma2_eps)) + ref[r]
    })
    lat2 <- sapply(seq_len(REPS_PER_PAIR), function(r) {
      cell_mean_p[i2[p], j_samp[r], m_samp[r]] +
        rnorm(1, 0, sqrt(n_p$sigma2_eps)) + ref[r]
    })

    w1 <- sum(lat1 > lat2)
    w2 <- REPS_PER_PAIR - w1

    wins[i1[p]]  <- wins[i1[p]] + w1
    wins[i2[p]]  <- wins[i2[p]] + w2
    games[i1[p]] <- games[i1[p]] + REPS_PER_PAIR
    games[i2[p]] <- games[i2[p]] + REPS_PER_PAIR

    bt_i1[p] <- i1[p]
    bt_i2[p] <- i2[p]
    bt_w1[p] <- w1
    bt_n[p]  <- REPS_PER_PAIR
  }

  theta_hat_winrate <- ifelse(games > 0, wins / games, 0.5)

  # Bradley-Terry via logistic regression
  X <- model.matrix(~ factor(bt_i1, levels = seq_len(N)) - 1) -
       model.matrix(~ factor(bt_i2, levels = seq_len(N)) - 1)
  X <- X[, -N]

  bt_fit <- tryCatch(
    suppressWarnings(glm.fit(X, cbind(bt_w1, bt_n - bt_w1), family = binomial())),
    error = function(e) NULL
  )

  if (!is.null(bt_fit)) {
    bt_scores <- c(bt_fit$coefficients, 0)
    bt_scores[is.na(bt_scores)] <- 0
    theta_hat_bt <- bt_scores
  } else {
    theta_hat_bt <- theta_hat_winrate
  }

  # Correlate with theta_std (the shared true quality)
  list(
    cor_likert  = cor(theta_std, theta_hat_likert),
    cor_winrate = cor(theta_std, theta_hat_winrate),
    cor_bt      = cor(theta_std, theta_hat_bt),
    tau_likert  = cor(theta_std, theta_hat_likert, method = "kendall"),
    tau_winrate = cor(theta_std, theta_hat_winrate, method = "kendall"),
    tau_bt      = cor(theta_std, theta_hat_bt, method = "kendall")
  )
}

# =========================================================================
# Main loop
# =========================================================================
results <- list()

for (sigma2_ref in SIGMA2_REF_LEVELS) {
  cat(sprintf("\nReference point variance = %.2f (SD = %.2f)\n",
              sigma2_ref, sqrt(sigma2_ref)))

  for (sim in seq_len(N_SIM)) {
    if (sim %% 50 == 0) cat(sprintf("  sim %d/%d\n", sim, N_SIM))
    set.seed(42 + sim)
    theta_std <- rnorm(N_ITEMS, 0, 1)

    res <- run_one_sim(sigma2_ref, theta_std)
    res$sigma2_ref <- sigma2_ref
    res$sim_id <- sim
    results[[length(results) + 1]] <- res
  }
}

results_df <- bind_rows(results)
write_csv(results_df, file.path(out_dir, "sim_recovery.csv"))
cat("\nSaved: data/processed/sim_recovery.csv\n")

# =========================================================================
# Summary
# =========================================================================
summary_df <- results_df %>%
  group_by(sigma2_ref) %>%
  summarize(
    cor_likert  = sprintf("%.3f (%.3f)", mean(cor_likert), sd(cor_likert)),
    cor_bt      = sprintf("%.3f (%.3f)", mean(cor_bt), sd(cor_bt)),
    tau_likert  = sprintf("%.3f (%.3f)", mean(tau_likert), sd(tau_likert)),
    tau_bt      = sprintf("%.3f (%.3f)", mean(tau_bt), sd(tau_bt)),
    .groups = "drop"
  )

cat("\n--- Recovery Summary (mean (sd)) ---\n")
cat("Each arm uses its own empirically-calibrated noise structure\n")
cat("Pairwise uses Bradley-Terry with 40 opponents per item\n\n")
print(summary_df, width = 120)

# =========================================================================
# Figure
# =========================================================================
plot_df <- results_df %>%
  select(sigma2_ref, sim_id, tau_likert, tau_bt, cor_likert, cor_bt) %>%
  pivot_longer(
    cols = c(tau_likert, tau_bt, cor_likert, cor_bt),
    names_to = c("metric", "method"),
    names_pattern = "(tau|cor)_(likert|bt)",
    values_to = "value"
  ) %>%
  mutate(
    method = ifelse(method == "likert", "Likert (1-5)", "Pairwise (Bradley-Terry)"),
    metric = ifelse(metric == "cor", "Pearson r", "Kendall tau"),
    ref_label = sprintf("ref SD = %.1f", sqrt(sigma2_ref))
  )

scoring_colors <- c("Likert (1-5)" = "#2166ac", "Pairwise (Bradley-Terry)" = "#e08040")

p <- ggplot(plot_df, aes(x = factor(round(sqrt(sigma2_ref), 2)),
                          y = value, fill = method)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.3, outlier.alpha = 0.3) +
  facet_wrap(~metric, scales = "free_y") +
  scale_fill_manual(values = scoring_colors, name = "Scoring method") +
  labs(
    title = "Effect of Reference-Point Noise on Score Recovery",
    subtitle = sprintf(
      "Each arm calibrated to empirical noise. Reference noise affects Likert but cancels in pairwise. %d sims.",
      N_SIM),
    x = "Reference point SD",
    y = "Correlation with true score"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    strip.text = element_text(face = "bold")
  )

ggsave(file.path(fig_dir, "sim_recovery.pdf"), p, width = 10, height = 5)
ggsave(file.path(fig_dir, "sim_recovery.png"), p, width = 10, height = 5, dpi = 300)
cat("Saved: sim_recovery.pdf/png\n")

cat("\n=== 06_sim_recovery.R complete ===\n")
