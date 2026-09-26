# 06_sim_scoring_recovery.R
# Scoring Method Recovery: Item-Level Tournament Design (v5)
#
# Extends Peysakhovich et al. (2015, WWW) with an item-level tournament.
# 150 policy items x 4 response models = 600 responses. ~1875 matchups
# sampled with stratification (same-item cross-model, different-item
# same-category, different-item between-category).
#
# Five Likert pathologies (absent in pairwise):
#   1. Central tendency compression (kappa_ct)
#   2. Per-observation anchoring noise (sigma_anchor)
#   3. Scale-use heterogeneity / DIF (sigma_slope)
#   4. Non-linear scale use (nonlinear_gamma)
#   5. Discretization loss (n_scale_points)
#
# Experiment 1: Ideal conditions (Peysakhovich baseline)
# Experiment 2: Factorial sweep of (kappa, sigma_anchor, sigma_slope)
# Experiment 3: Non-linear and discretization sweep (kappa × gamma × scale_pts)
# Experiment 4: Full 5-way crossing (kappa × anchor × slope × gamma × scale_pts)
#
# Literature basis for pathology parameters:
#   kappa (compression):      Esmaeili 2025 (arXiv:2506.02945)
#   sigma_anchor (noise):     Rating Roulette (arXiv:2510.27106)
#   sigma_slope (DIF):        Shen 2026 (arXiv:2601.03444)
#   nonlinear_gamma (scale):  Preston & Colman 2000, psychometrics literature
#   n_scale_points (discr.):  Likert 1932; Dawes 2008 (fewer points = more info loss)
#
# Usage:
#   Rscript analysis/06_sim_scoring_recovery.R               # default N_SIM=200
#   Rscript analysis/06_sim_scoring_recovery.R --nsim 3       # smoke test
#   Rscript analysis/06_sim_scoring_recovery.R --nsim 500     # HPC full run

library(tidyverse)
library(Matrix)

set.seed(42)

# --- Parse command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)
N_SIM <- 200
if ("--nsim" %in% args) {
  idx <- which(args == "--nsim")
  if (idx < length(args)) N_SIM <- as.integer(args[idx + 1])
}

fig_dir  <- "figures"
out_dir  <- "data/processed"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 06_sim_scoring_recovery.R: Item-Level Tournament (v5) ===\n")
cat(sprintf("N_SIM = %d\n", N_SIM))

# =========================================================================
# Section 1: Design Constants
# =========================================================================

# Response pool: items x response-generating models
N_CATEGORIES  <- 5
ITEMS_PER_CAT <- 30
N_ITEMS       <- N_CATEGORIES * ITEMS_PER_CAT   # 150
N_RESP_MODELS <- 4
N_RESPONSES   <- N_ITEMS * N_RESP_MODELS         # 600

# Evaluation factorial (per pair or per response)
N_PROMPTS <- 3
N_JUDGES  <- 3
N_TEMPS   <- 2
N_REPS    <- 2
EVALS_PER_PAIR <- N_PROMPTS * N_JUDGES * N_TEMPS * N_REPS  # 36

# Likert budget: 600 responses x 36 = 21,600 calls
LIKERT_CALLS <- N_RESPONSES * EVALS_PER_PAIR

# Tournament pair targets (main arm)
SAME_ITEM_PAIRS        <- N_ITEMS * choose(N_RESP_MODELS, 2)  # 900
DIFF_SAME_CAT_PAIRS    <- 700
DIFF_BETWEEN_CAT_PAIRS <- 275
TOTAL_PAIRS <- SAME_ITEM_PAIRS + DIFF_SAME_CAT_PAIRS + DIFF_BETWEEN_CAT_PAIRS  # 1875

# Cost-equated pairwise: same budget as Likert, more pairs + fewer evals
COST_EQ_EVALS     <- 12   # 3 prompts x 2 judges x 1 temp x 2 reps
COST_EQ_N_PROMPTS <- 3
COST_EQ_N_JUDGES  <- 2
COST_EQ_N_TEMPS   <- 1
COST_EQ_N_REPS    <- 2
COST_EQ_TOTAL_PAIRS <- LIKERT_CALLS %/% COST_EQ_EVALS  # 1800
MIN_DEGREE <- 4  # minimum tournament graph degree per response

# Full round-robin: all C(600,2) pairs, same factorial (theoretical ceiling)
FULL_RR_PAIRS     <- choose(N_RESPONSES, 2)  # 179,700
FULL_RR_N_PROMPTS <- N_PROMPTS   # 3
FULL_RR_N_JUDGES  <- N_JUDGES    # 3
FULL_RR_N_TEMPS   <- N_TEMPS     # 2
FULL_RR_N_REPS    <- N_REPS      # 2
FULL_RR_EVALS     <- EVALS_PER_PAIR  # 36 (same as tournament)

# Variance components for response quality DGP
VC <- list(
  sigma2_gamma       = 0.05,   # between-category
  sigma2_delta       = 0.60,   # within-category item
  sigma2_omega       = 0.10,   # response-model main effect
  sigma2_delta_omega = 0.13,   # item x response-model interaction
  # Total response quality variance: ~0.88 (matching empirical)
  sigma2_beta      = 0.006,    # prompt variant
  sigma2_phi       = 0.021,    # judge model
  sigma2_resp_beta = 0.034,    # response x prompt
  sigma2_resp_phi  = 0.155,    # response x judge
  sigma2_beta_phi  = 0.009,    # prompt x judge
  sigma2_eps       = 0.111     # generation noise (residual)
)

# Pathology levels — Experiments 2 and 3
KAPPA_LEVELS        <- c(1.0, 0.7, 0.5, 0.3)
SIGMA_ANCHOR_LEVELS <- c(0, 0.2, 0.4, 0.6)
SIGMA_SLOPE_LEVELS  <- c(0, 0.1, 0.2, 0.3)

# Exp 3: non-linear scale use + discretization
# gamma < 1: large quality diffs compressed on scale (extremes under-used)
# gamma = 1: linear (standard)
NONLINEAR_GAMMA_LEVELS <- c(1.0, 0.7, 0.5)
# n_scale_points: 3-pt loses info, 5-pt standard, 100 = continuous (no loss)
SCALE_POINTS_LEVELS    <- c(3, 5, 7, 100)

n_conditions_exp2 <- length(KAPPA_LEVELS) * length(SIGMA_ANCHOR_LEVELS) *
  length(SIGMA_SLOPE_LEVELS)
n_conditions_exp3 <- length(KAPPA_LEVELS) * length(NONLINEAR_GAMMA_LEVELS) *
  length(SCALE_POINTS_LEVELS)

cat(sprintf("Response pool: %d items x %d models = %d responses\n",
            N_ITEMS, N_RESP_MODELS, N_RESPONSES))
cat(sprintf("Tournament: %d pairs x %d evals = %d calls\n",
            TOTAL_PAIRS, EVALS_PER_PAIR, TOTAL_PAIRS * EVALS_PER_PAIR))
cat(sprintf("Cost-equated: %d pairs x %d evals = %d calls\n",
            COST_EQ_TOTAL_PAIRS, COST_EQ_EVALS,
            COST_EQ_TOTAL_PAIRS * COST_EQ_EVALS))
cat(sprintf("Full RR: %d pairs x %d evals = %d calls\n",
            FULL_RR_PAIRS, FULL_RR_EVALS, FULL_RR_PAIRS * FULL_RR_EVALS))
cat(sprintf("Likert: %d responses x %d evals = %d calls\n",
            N_RESPONSES, EVALS_PER_PAIR, LIKERT_CALLS))
n_conditions_exp4 <- n_conditions_exp2 * length(NONLINEAR_GAMMA_LEVELS) *
  length(SCALE_POINTS_LEVELS)
cat(sprintf("Exp 2: %d, Exp 3: %d, Exp 4: %d conditions\n",
            n_conditions_exp2, n_conditions_exp3, n_conditions_exp4))

# =========================================================================
# Section 2: Response Quality DGP
# =========================================================================
generate_response_pool <- function(vc = VC) {
  resp_df <- expand.grid(
    model_id = seq_len(N_RESP_MODELS),
    item_id  = seq_len(N_ITEMS)
  )
  resp_df$resp_id  <- seq_len(nrow(resp_df))
  resp_df$category <- ceiling(resp_df$item_id / ITEMS_PER_CAT)

  gamma       <- rnorm(N_CATEGORIES, 0, sqrt(vc$sigma2_gamma))
  delta       <- rnorm(N_ITEMS, 0, sqrt(vc$sigma2_delta))
  omega       <- rnorm(N_RESP_MODELS, 0, sqrt(vc$sigma2_omega))
  delta_omega <- rnorm(N_RESPONSES, 0, sqrt(vc$sigma2_delta_omega))

  resp_df$theta <- 3.0 +
    gamma[resp_df$category] +
    delta[resp_df$item_id] +
    omega[resp_df$model_id] +
    delta_omega

  resp_df
}

# =========================================================================
# Section 3: Cell Mean Generation (response x prompt x judge)
# =========================================================================
generate_cell_means <- function(resp_df, vc = VC) {
  NR <- nrow(resp_df)
  K  <- N_PROMPTS
  M  <- N_JUDGES

  beta      <- rnorm(K, 0, sqrt(vc$sigma2_beta))
  phi       <- rnorm(M, 0, sqrt(vc$sigma2_phi))
  resp_beta <- matrix(rnorm(NR * K, 0, sqrt(vc$sigma2_resp_beta)), NR, K)
  resp_phi  <- matrix(rnorm(NR * M, 0, sqrt(vc$sigma2_resp_phi)), NR, M)
  beta_phi  <- matrix(rnorm(K * M, 0, sqrt(vc$sigma2_beta_phi)), K, M)

  cell_mean <- array(NA_real_, dim = c(NR, K, M))
  for (m in seq_len(M)) {
    cell_mean[, , m] <- resp_df$theta + resp_beta + resp_phi[, m] +
      rep(beta + beta_phi[, m], each = NR) + phi[m]
  }
  cell_mean
}

# =========================================================================
# Section 4: Pair Sampling (with minimum degree guarantee)
# =========================================================================

# Helper: sample diff-item pairs within a category
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

# Main tournament: ~1875 pairs
sample_tournament_pairs <- function(resp_df) {
  parts <- list()

  # 1. All same-item cross-model pairs: 150 x C(4,2) = 900
  same_item <- do.call(rbind, lapply(seq_len(N_ITEMS), function(i) {
    rids <- resp_df$resp_id[resp_df$item_id == i]
    cc <- combn(rids, 2)
    tibble(resp_a = cc[1, ], resp_b = cc[2, ], stratum = "same_item")
  }))
  parts[[1]] <- same_item

  # 2. Different-item, same-category: ~700 (140 per cat)
  per_cat <- DIFF_SAME_CAT_PAIRS %/% N_CATEGORIES
  for (cat in seq_len(N_CATEGORIES)) {
    parts[[length(parts) + 1]] <- sample_diff_item_within_cat(resp_df, cat, per_cat)
  }

  # 3. Different-item, between-category: ~275 (~28 per cat-pair)
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

# Cost-equated tournament: ~1800 pairs with min degree guarantee
sample_cost_eq_pairs <- function(resp_df) {
  parts <- list()

  # 1. All same-item cross-model pairs: 900 (degree 3 per response)
  same_item <- do.call(rbind, lapply(seq_len(N_ITEMS), function(i) {
    rids <- resp_df$resp_id[resp_df$item_id == i]
    cc <- combn(rids, 2)
    tibble(resp_a = cc[1, ], resp_b = cc[2, ], stratum = "same_item")
  }))
  parts[[1]] <- same_item

  remaining <- COST_EQ_TOTAL_PAIRS - nrow(same_item)  # ~900

  # 2. Guarantee every response appears in at least 1 diff-item pair.
  #    Shuffle responses and pair consecutive ones (different items).
  shuffled <- sample(N_RESPONSES)
  ga <- shuffled[seq(1, N_RESPONSES, 2)]  # 300 resp_a
  gb <- shuffled[seq(2, N_RESPONSES, 2)]  # 300 resp_b
  # Fix same-item collisions
  same_item_mask <- resp_df$item_id[ga] == resp_df$item_id[gb]
  while (any(same_item_mask)) {
    for (idx in which(same_item_mask)) {
      cands <- resp_df$resp_id[resp_df$item_id != resp_df$item_id[ga[idx]]]
      gb[idx] <- sample(cands, 1)
    }
    same_item_mask <- resp_df$item_id[ga] == resp_df$item_id[gb]
  }
  parts[[2]] <- tibble(resp_a = ga, resp_b = gb, stratum = "diff_guaranteed")

  remaining <- remaining - 300  # ~600

  # 3. Fill remaining with stratified diff-item pairs
  same_cat_target  <- round(remaining * 0.75)
  between_target   <- remaining - same_cat_target

  per_cat_sc <- same_cat_target %/% N_CATEGORIES
  for (cat in seq_len(N_CATEGORIES)) {
    parts[[length(parts) + 1]] <- sample_diff_item_within_cat(resp_df, cat, per_cat_sc)
  }

  cat_combos <- combn(N_CATEGORIES, 2)
  per_cp_bt <- ceiling(between_target / ncol(cat_combos))
  for (cp in seq_len(ncol(cat_combos))) {
    ra <- resp_df$resp_id[resp_df$category == cat_combos[1, cp]]
    rb <- resp_df$resp_id[resp_df$category == cat_combos[2, cp]]
    parts[[length(parts) + 1]] <- tibble(
      resp_a  = sample(ra, per_cp_bt, replace = TRUE),
      resp_b  = sample(rb, per_cp_bt, replace = TRUE),
      stratum = "diff_between_cat"
    )
  }

  bind_rows(parts) %>% mutate(pair_id = row_number())
}

# =========================================================================
# Section 5: Likert Observation (with configurable pathologies)
# =========================================================================
observe_likert <- function(cell_mean, sigma2_eps, a_m,
                           kappa_ct, sigma_anchor,
                           nonlinear_gamma = 1.0, n_scale_points = 5) {
  NR <- dim(cell_mean)[1]; K <- dim(cell_mean)[2]; M <- dim(cell_mean)[3]
  n_reps_total <- N_TEMPS * N_REPS

  all_y <- numeric(0)
  all_idx <- integer(0)

  for (m in seq_len(M)) {
    n_obs <- NR * K * n_reps_total
    v_m   <- rep(as.vector(cell_mean[, , m]), each = n_reps_total)

    # 1. DIF: judge-specific slope
    scaled <- 3 + a_m[m] * (v_m - 3)

    # 2. Generation noise + anchoring noise
    eps    <- rnorm(n_obs, 0, sqrt(sigma2_eps))
    anchor <- if (sigma_anchor > 0) rnorm(n_obs, 0, sigma_anchor) else 0
    latent <- scaled + eps + anchor

    # 3. Central tendency compression
    compressed <- 3 + kappa_ct * (latent - 3)

    # 4. Non-linear scale use: power transform of deviations
    #    gamma < 1 compresses large deviations (extremes under-used)
    if (nonlinear_gamma != 1.0) {
      dev <- compressed - 3
      compressed <- 3 + sign(dev) * abs(dev)^nonlinear_gamma
    }

    # 5. Discretization to n_scale_points on [1, 5]
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

# =========================================================================
# Section 6: Pairwise Tournament Observation
# =========================================================================
observe_pairwise <- function(cell_mean, pairs_df, sigma2_eps, a_m,
                             rho = 1.0,
                             n_prompts = N_PROMPTS, n_judges = N_JUDGES,
                             n_temps = N_TEMPS, n_reps = N_REPS) {
  n_pairs  <- nrow(pairs_df)
  noise_sd <- rho * sqrt(sigma2_eps)

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

# =========================================================================
# Section 7: Bradley-Terry via MM Algorithm
# =========================================================================
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
  converged <- FALSE
  delta <- Inf

  for (iter in seq_len(max_iter)) {
    pi_old <- pi

    sum_pi  <- pi[resp_a] + pi[resp_b]
    contrib <- n_per_pair / sum_pi
    D       <- as.vector(crossprod(A, contrib))

    pi <- ifelse(D > 0, W / D, 1e-10)
    pi <- pi / sum(pi) * n_responses

    delta <- max(abs(log(pmax(pi, 1e-20)) - log(pmax(pi_old, 1e-20))))
    if (delta < tol) { converged <- TRUE; break }
  }

  scores <- log(pmax(pi, 1e-20))
  scores <- scores - mean(scores)
  list(scores = scores, converged = converged, n_iter = iter,
       final_delta = delta)
}

# =========================================================================
# Section 8: Metrics
# =========================================================================
compute_metrics <- function(theta_true, theta_hat, label) {
  n <- length(theta_true)

  # Handle degenerate case (zero variance from extreme discretization)
  if (sd(theta_hat) < 1e-12) {
    return(tibble(method = label, tau = 0, pearson_r = 0, auc = 0.5,
                  rank_mse = 1/3, top10 = 0.10, top20 = 0.20))
  }

  tau_val <- cor(theta_true, theta_hat, method = "kendall")
  r_val   <- cor(theta_true, theta_hat, method = "pearson")

  # AUC (concordance probability): P(correct pairwise ordering)
  n_sample <- min(5000, choose(n, 2))
  idx1 <- sample(n, n_sample, replace = TRUE)
  idx2 <- sample(n, n_sample, replace = TRUE)
  valid <- idx1 != idx2 & theta_true[idx1] != theta_true[idx2]
  if (sum(valid) > 0) {
    correct <- (theta_hat[idx1[valid]] > theta_hat[idx2[valid]]) ==
      (theta_true[idx1[valid]] > theta_true[idx2[valid]])
    auc_val <- mean(correct)
  } else {
    auc_val <- 0.5
  }

  # Rank MSE (normalized by n^2 so scale-free)
  rank_true <- rank(theta_true)
  rank_hat  <- rank(theta_hat)
  rank_mse  <- mean((rank_true - rank_hat)^2) / n^2

  # Top-k recall: fraction of true top-k recovered in estimated top-k
  top_k_recall <- function(k_frac) {
    k <- round(n * k_frac)
    true_topk <- order(theta_true, decreasing = TRUE)[seq_len(k)]
    hat_topk  <- order(theta_hat, decreasing = TRUE)[seq_len(k)]
    length(intersect(true_topk, hat_topk)) / k
  }

  tibble(
    method = label, tau = tau_val, pearson_r = r_val, auc = auc_val,
    rank_mse = rank_mse, top10 = top_k_recall(0.10), top20 = top_k_recall(0.20)
  )
}

# =========================================================================
# Section 9: Full Round-Robin Pair Set (pre-generated, reused across sims)
# =========================================================================
cat("Generating full round-robin pair set...\n")
rr_combos <- combn(N_RESPONSES, 2)
full_rr_pairs <- tibble(
  resp_a  = rr_combos[1, ],
  resp_b  = rr_combos[2, ],
  stratum = "full_rr",
  pair_id = seq_len(ncol(rr_combos))
)
rm(rr_combos)
cat(sprintf("Full RR: %d pairs (degree=%d per response)\n",
            nrow(full_rr_pairs), N_RESPONSES - 1))

# =========================================================================
# Section 10: Experiment 1 -- Ideal Conditions (Peysakhovich Baseline)
# =========================================================================
cat("\n========== Experiment 1: Ideal Conditions ==========\n")

exp1_results <- vector("list", N_SIM)

for (sim in seq_len(N_SIM)) {
  if (sim %% max(1, N_SIM %/% 4) == 0) cat(sprintf("  sim %d/%d\n", sim, N_SIM))
  set.seed(42 + sim)

  resp_df   <- generate_response_pool()
  cell_mean <- generate_cell_means(resp_df)
  pairs     <- sample_tournament_pairs(resp_df)
  pairs_eq  <- sample_cost_eq_pairs(resp_df)

  a_m <- rep(1, N_JUDGES)

  # Likert
  theta_likert <- observe_likert(cell_mean, VC$sigma2_eps, a_m,
                                 kappa_ct = 1.0, sigma_anchor = 0)

  # BT tournament (full)
  pw <- observe_pairwise(cell_mean, pairs, VC$sigma2_eps, a_m)
  bt <- fit_bt_mm(pw$resp_a, pw$resp_b, pw$wins_a, pw$n_per_pair)

  # BT cost-equated
  pw_eq <- observe_pairwise(cell_mean, pairs_eq, VC$sigma2_eps, a_m,
                            n_prompts = COST_EQ_N_PROMPTS,
                            n_judges  = COST_EQ_N_JUDGES,
                            n_temps   = COST_EQ_N_TEMPS,
                            n_reps    = COST_EQ_N_REPS)
  bt_eq <- fit_bt_mm(pw_eq$resp_a, pw_eq$resp_b,
                     pw_eq$wins_a, pw_eq$n_per_pair)

  # BT full round-robin (theoretical ceiling)
  pw_rr <- observe_pairwise(cell_mean, full_rr_pairs, VC$sigma2_eps, a_m,
                            n_prompts = FULL_RR_N_PROMPTS,
                            n_judges  = FULL_RR_N_JUDGES,
                            n_temps   = FULL_RR_N_TEMPS,
                            n_reps    = FULL_RR_N_REPS)
  bt_rr <- fit_bt_mm(pw_rr$resp_a, pw_rr$resp_b,
                     pw_rr$wins_a, pw_rr$n_per_pair)

  m_lik  <- compute_metrics(resp_df$theta, theta_likert, "likert_mean")
  m_bt   <- compute_metrics(resp_df$theta, bt$scores, "bt_tournament")
  m_bte  <- compute_metrics(resp_df$theta, bt_eq$scores, "bt_cost_eq")
  m_btrr <- compute_metrics(resp_df$theta, bt_rr$scores, "bt_full_rr")

  exp1_results[[sim]] <- bind_rows(m_lik, m_bt, m_bte, m_btrr) %>%
    mutate(
      sim_id = sim,
      kappa_ct = 1.0, sigma_anchor = 0, sigma_slope = 0,
      bt_converged = bt$converged,
      bt_eq_converged = bt_eq$converged,
      bt_rr_converged = bt_rr$converged
    )
}

exp1_df <- bind_rows(exp1_results)

cat("\n--- Experiment 1 Summary ---\n")
exp1_summary <- exp1_df %>%
  group_by(method) %>%
  summarize(
    tau_mean = mean(tau), tau_se = sd(tau) / sqrt(n()),
    r_mean   = mean(pearson_r),
    auc_mean = mean(auc),
    rmse_mean = mean(rank_mse),
    top10_mean = mean(top10),
    .groups  = "drop"
  )
print(exp1_summary, width = 120)

cat(sprintf("\nBT convergence: tournament=%.0f%%, cost-eq=%.0f%%, full-rr=%.0f%%\n",
            mean(exp1_df$bt_converged[exp1_df$method == "bt_tournament"]) * 100,
            mean(exp1_df$bt_eq_converged[exp1_df$method == "bt_cost_eq"]) * 100,
            mean(exp1_df$bt_rr_converged[exp1_df$method == "bt_full_rr"]) * 100))

# =========================================================================
# Section 10: Experiment 2 -- Pathology Sweep
# =========================================================================
cat("\n========== Experiment 2: Pathology Sweep ==========\n")

# Optimization: BT varies only with sigma_slope (compression/anchoring
# affect Likert only). Compute BT once per (sim, slope), reuse for Likert.

exp2_results <- list()
exp2_idx <- 0

for (sim in seq_len(N_SIM)) {
  if (sim %% max(1, N_SIM %/% 4) == 0) cat(sprintf("  sim %d/%d\n", sim, N_SIM))
  set.seed(42 + sim)

  resp_df   <- generate_response_pool()
  cell_mean <- generate_cell_means(resp_df)
  pairs     <- sample_tournament_pairs(resp_df)
  pairs_eq  <- sample_cost_eq_pairs(resp_df)

  # Pre-compute BT per sigma_slope level
  bt_by_slope <- list()
  for (s_idx in seq_along(SIGMA_SLOPE_LEVELS)) {
    sigma_slope <- SIGMA_SLOPE_LEVELS[s_idx]
    a_m <- if (sigma_slope > 0) rnorm(N_JUDGES, 1, sigma_slope) else rep(1, N_JUDGES)

    pw    <- observe_pairwise(cell_mean, pairs, VC$sigma2_eps, a_m)
    pw_eq <- observe_pairwise(cell_mean, pairs_eq, VC$sigma2_eps, a_m,
                              n_prompts = COST_EQ_N_PROMPTS,
                              n_judges  = COST_EQ_N_JUDGES,
                              n_temps   = COST_EQ_N_TEMPS,
                              n_reps    = COST_EQ_N_REPS)
    pw_rr <- observe_pairwise(cell_mean, full_rr_pairs, VC$sigma2_eps, a_m,
                              n_prompts = FULL_RR_N_PROMPTS,
                              n_judges  = FULL_RR_N_JUDGES,
                              n_temps   = FULL_RR_N_TEMPS,
                              n_reps    = FULL_RR_N_REPS)

    bt    <- fit_bt_mm(pw$resp_a, pw$resp_b, pw$wins_a, pw$n_per_pair)
    bt_eq <- fit_bt_mm(pw_eq$resp_a, pw_eq$resp_b,
                       pw_eq$wins_a, pw_eq$n_per_pair)
    bt_rr <- fit_bt_mm(pw_rr$resp_a, pw_rr$resp_b,
                       pw_rr$wins_a, pw_rr$n_per_pair)

    bt_by_slope[[s_idx]] <- list(
      a_m = a_m,
      bt_scores    = bt$scores,
      bt_eq_scores = bt_eq$scores,
      bt_rr_scores = bt_rr$scores,
      bt_conv      = bt$converged,
      bt_eq_conv   = bt_eq$converged,
      bt_rr_conv   = bt_rr$converged
    )
  }

  # Likert: varies with all three pathology parameters
  for (kappa_ct in KAPPA_LEVELS) {
    for (sigma_anchor in SIGMA_ANCHOR_LEVELS) {
      for (s_idx in seq_along(SIGMA_SLOPE_LEVELS)) {
        sigma_slope <- SIGMA_SLOPE_LEVELS[s_idx]
        bt_s <- bt_by_slope[[s_idx]]

        theta_likert <- observe_likert(cell_mean, VC$sigma2_eps, bt_s$a_m,
                                       kappa_ct, sigma_anchor)

        m_lik  <- compute_metrics(resp_df$theta, theta_likert, "likert_mean")
        m_bt   <- compute_metrics(resp_df$theta, bt_s$bt_scores, "bt_tournament")
        m_bte  <- compute_metrics(resp_df$theta, bt_s$bt_eq_scores, "bt_cost_eq")
        m_btrr <- compute_metrics(resp_df$theta, bt_s$bt_rr_scores, "bt_full_rr")

        row <- bind_rows(m_lik, m_bt, m_bte, m_btrr) %>%
          mutate(
            sim_id = sim,
            kappa_ct = kappa_ct,
            sigma_anchor = sigma_anchor,
            sigma_slope = sigma_slope,
            bt_converged = bt_s$bt_conv,
            bt_eq_converged = bt_s$bt_eq_conv,
            bt_rr_converged = bt_s$bt_rr_conv
          )

        exp2_idx <- exp2_idx + 1
        exp2_results[[exp2_idx]] <- row
      }
    }
  }

  # Checkpoint
  if (sim %% max(1, N_SIM %/% 10) == 0 || sim == N_SIM) {
    exp2_df_partial <- bind_rows(exp2_results)
    write_csv(exp1_df, file.path(out_dir, "sim_scoring_crossover.csv"))
    write_csv(exp2_df_partial, file.path(out_dir, "sim_scoring_empirical.csv"))
    cat(sprintf("  [checkpoint] sim %d: %d rows\n", sim, nrow(exp2_df_partial)))
  }
}

exp2_df <- bind_rows(exp2_results)
write_csv(exp1_df, file.path(out_dir, "sim_scoring_crossover.csv"))
write_csv(exp2_df, file.path(out_dir, "sim_scoring_empirical.csv"))
cat("\nSaved: sim_scoring_crossover.csv, sim_scoring_empirical.csv\n")

# =========================================================================
# Section 11: Summary Tables
# =========================================================================
cat("\n========== Experiment 2 Summary ==========\n")

exp2_summary <- exp2_df %>%
  group_by(kappa_ct, sigma_anchor, sigma_slope, method) %>%
  summarize(
    tau_mean = mean(tau), tau_se = sd(tau) / sqrt(n()),
    r_mean   = mean(pearson_r),
    auc_mean = mean(auc),
    .groups  = "drop"
  )

# Crossover: Likert vs BT tournament
crossover <- exp2_summary %>%
  filter(method %in% c("likert_mean", "bt_tournament")) %>%
  select(kappa_ct, sigma_anchor, sigma_slope, method, tau_mean) %>%
  pivot_wider(names_from = method, values_from = tau_mean) %>%
  mutate(bt_advantage = bt_tournament - likert_mean)

cat("\nBT tournament advantage (tau_BT - tau_Likert):\n")
print(crossover, width = 120, n = 70)

bt_wins <- crossover %>% filter(bt_advantage > 0)
cat(sprintf("\nTournament: BT wins %d/%d, Likert wins %d/%d\n",
            nrow(bt_wins), nrow(crossover),
            nrow(crossover) - nrow(bt_wins), nrow(crossover)))

# Cost-equated crossover
crossover_eq <- exp2_summary %>%
  filter(method %in% c("likert_mean", "bt_cost_eq")) %>%
  select(kappa_ct, sigma_anchor, sigma_slope, method, tau_mean) %>%
  pivot_wider(names_from = method, values_from = tau_mean) %>%
  mutate(bt_eq_advantage = bt_cost_eq - likert_mean)

bt_eq_wins <- crossover_eq %>% filter(bt_eq_advantage > 0)
cat(sprintf("Cost-equated: BT wins %d/%d, Likert wins %d/%d\n",
            nrow(bt_eq_wins), nrow(crossover_eq),
            nrow(crossover_eq) - nrow(bt_eq_wins), nrow(crossover_eq)))

# BT convergence
bt_conv <- exp2_df %>%
  filter(method == "bt_tournament") %>%
  group_by(sigma_slope) %>%
  summarize(
    tournament = mean(bt_converged),
    cost_eq    = mean(bt_eq_converged),
    .groups = "drop"
  )
cat("\nBT convergence by sigma_slope:\n")
print(bt_conv)

# =========================================================================
# Section 12: Figures
# =========================================================================
cat("\n========== Generating Figures ==========\n")

scoring_colors <- c(
  "Likert mean"         = "#2166ac",
  "BT (full RR)"        = "#4daf4a",
  "BT (tournament)"     = "#e08040",
  "BT (cost-equated)"   = "#b2182b"
)
scoring_linetypes <- c(
  "Likert mean"         = "solid",
  "BT (full RR)"        = "dotted",
  "BT (tournament)"     = "solid",
  "BT (cost-equated)"   = "dashed"
)
scoring_shapes <- c(
  "Likert mean"         = 16,
  "BT (full RR)"        = 18,
  "BT (tournament)"     = 17,
  "BT (cost-equated)"   = 15
)

method_labels <- c(
  "likert_mean"   = "Likert mean",
  "bt_full_rr"    = "BT (full RR)",
  "bt_tournament" = "BT (tournament)",
  "bt_cost_eq"    = "BT (cost-equated)"
)
method_order <- c("Likert mean", "BT (full RR)", "BT (tournament)", "BT (cost-equated)")

# --- Figure 1: Ideal Conditions ---
fig1_data <- exp1_summary %>%
  mutate(method_label = factor(recode(method, !!!method_labels),
                               levels = method_order))

p1 <- ggplot(fig1_data, aes(x = method_label, y = tau_mean,
                              fill = method_label)) +
  geom_col(alpha = 0.8, width = 0.6) +
  geom_errorbar(aes(ymin = tau_mean - 1.96 * tau_se,
                     ymax = tau_mean + 1.96 * tau_se),
                width = 0.15, linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.3f", tau_mean)), vjust = -0.8, size = 3.2) +
  scale_fill_manual(values = scoring_colors, guide = "none") +
  coord_cartesian(ylim = c(
    min(fig1_data$tau_mean) - 0.05,
    max(fig1_data$tau_mean) + 0.03)) +
  labs(
    title = "Ideal Conditions: Likert vs. Pairwise Tournament",
    subtitle = sprintf(
      paste0("No pathologies. %d responses, tournament=%d pairs, ",
             "cost-eq=%d pairs (%d evals). N_SIM=%d."),
      N_RESPONSES, TOTAL_PAIRS, COST_EQ_TOTAL_PAIRS, COST_EQ_EVALS, N_SIM),
    x = NULL,
    y = expression("Kendall " * tau * " with true quality")
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(size = 9)
  )

ggsave(file.path(fig_dir, "sim_scoring_crossover.pdf"), p1,
       width = 7, height = 5)
ggsave(file.path(fig_dir, "sim_scoring_crossover.png"), p1,
       width = 7, height = 5, dpi = 300)
cat("Saved: sim_scoring_crossover.pdf/png\n")

# --- Figure 2: Pathology Sweep ---
fig2_data <- exp2_df %>%
  mutate(
    method_label = factor(recode(method, !!!method_labels),
                          levels = method_order),
    kappa_label  = sprintf("kappa = %.1f", kappa_ct),
    anchor_label = sprintf("anchor = %.1f", sigma_anchor)
  ) %>%
  group_by(sigma_slope, kappa_ct, sigma_anchor,
           kappa_label, anchor_label, method_label) %>%
  summarize(
    tau_mean = mean(tau), tau_se = sd(tau) / sqrt(n()),
    .groups = "drop"
  )

fig2_data$kappa_label <- factor(fig2_data$kappa_label,
                                levels = sprintf("kappa = %.1f", rev(KAPPA_LEVELS)))
fig2_data$anchor_label <- factor(fig2_data$anchor_label,
                                 levels = sprintf("anchor = %.1f", SIGMA_ANCHOR_LEVELS))

p2 <- ggplot(fig2_data, aes(x = sigma_slope, y = tau_mean,
                              color = method_label, linetype = method_label,
                              shape = method_label)) +
  geom_ribbon(aes(ymin = tau_mean - tau_se, ymax = tau_mean + tau_se,
                  fill = method_label), alpha = 0.10, color = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_grid(kappa_label ~ anchor_label) +
  scale_color_manual(values = scoring_colors, name = NULL) +
  scale_fill_manual(values = scoring_colors, guide = "none") +
  scale_linetype_manual(values = scoring_linetypes, name = NULL) +
  scale_shape_manual(values = scoring_shapes, name = NULL) +
  scale_x_continuous(breaks = SIGMA_SLOPE_LEVELS) +
  labs(
    title = "Scoring Method Recovery Under Judge Pathologies",
    subtitle = sprintf(
      paste0("Item-level tournament (%d responses). ",
             "Rows: compression. Cols: anchoring. N_SIM=%d."),
      N_RESPONSES, n_distinct(exp2_df$sim_id)),
    x = expression("Judge scale-use heterogeneity (" * sigma[slope] * ")"),
    y = expression("Kendall " * tau * " with true quality")
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    strip.text = element_text(size = 9, face = "bold")
  ) +
  guides(color = guide_legend(nrow = 1),
         linetype = guide_legend(nrow = 1),
         shape = guide_legend(nrow = 1))

ggsave(file.path(fig_dir, "sim_scoring_empirical.pdf"), p2,
       width = 11, height = 11)
ggsave(file.path(fig_dir, "sim_scoring_empirical.png"), p2,
       width = 11, height = 11, dpi = 300)
cat("Saved: sim_scoring_empirical.pdf/png\n")

# =========================================================================
# Section 13: Experiment 3 -- Non-linear Scale Use + Discretization
# =========================================================================
cat("\n========== Experiment 3: Non-linear + Discretization Sweep ==========\n")

# BT is immune to non-linear scale use and discretization (ordinal comparisons).
# Only Likert varies. Sweep gamma × n_scale_points × kappa (compression interacts
# with discretization). sigma_slope=0, sigma_anchor=0 to isolate these effects.

exp3_results <- list()
exp3_idx <- 0

for (sim in seq_len(N_SIM)) {
  if (sim %% max(1, N_SIM %/% 4) == 0) cat(sprintf("  sim %d/%d\n", sim, N_SIM))
  set.seed(42 + sim)

  resp_df   <- generate_response_pool()
  cell_mean <- generate_cell_means(resp_df)
  pairs     <- sample_tournament_pairs(resp_df)
  pairs_eq  <- sample_cost_eq_pairs(resp_df)

  a_m <- rep(1, N_JUDGES)  # no DIF

  # BT scores don't vary across exp3 conditions (no slope variation)
  pw    <- observe_pairwise(cell_mean, pairs, VC$sigma2_eps, a_m)
  bt    <- fit_bt_mm(pw$resp_a, pw$resp_b, pw$wins_a, pw$n_per_pair)
  pw_eq <- observe_pairwise(cell_mean, pairs_eq, VC$sigma2_eps, a_m,
                            n_prompts = COST_EQ_N_PROMPTS,
                            n_judges  = COST_EQ_N_JUDGES,
                            n_temps   = COST_EQ_N_TEMPS,
                            n_reps    = COST_EQ_N_REPS)
  bt_eq <- fit_bt_mm(pw_eq$resp_a, pw_eq$resp_b,
                     pw_eq$wins_a, pw_eq$n_per_pair)
  pw_rr <- observe_pairwise(cell_mean, full_rr_pairs, VC$sigma2_eps, a_m,
                            n_prompts = FULL_RR_N_PROMPTS,
                            n_judges  = FULL_RR_N_JUDGES,
                            n_temps   = FULL_RR_N_TEMPS,
                            n_reps    = FULL_RR_N_REPS)
  bt_rr <- fit_bt_mm(pw_rr$resp_a, pw_rr$resp_b,
                     pw_rr$wins_a, pw_rr$n_per_pair)

  m_bt   <- compute_metrics(resp_df$theta, bt$scores, "bt_tournament")
  m_bte  <- compute_metrics(resp_df$theta, bt_eq$scores, "bt_cost_eq")
  m_btrr <- compute_metrics(resp_df$theta, bt_rr$scores, "bt_full_rr")

  for (kappa_ct in KAPPA_LEVELS) {
    for (gamma in NONLINEAR_GAMMA_LEVELS) {
      for (n_sp in SCALE_POINTS_LEVELS) {
        theta_likert <- observe_likert(cell_mean, VC$sigma2_eps, a_m,
                                       kappa_ct = kappa_ct, sigma_anchor = 0,
                                       nonlinear_gamma = gamma,
                                       n_scale_points = n_sp)
        m_lik <- compute_metrics(resp_df$theta, theta_likert, "likert_mean")

        row <- bind_rows(m_lik, m_bt, m_bte, m_btrr) %>%
          mutate(
            sim_id = sim,
            kappa_ct = kappa_ct,
            nonlinear_gamma = gamma,
            n_scale_points = n_sp,
            bt_converged = bt$converged,
            bt_eq_converged = bt_eq$converged,
            bt_rr_converged = bt_rr$converged
          )

        exp3_idx <- exp3_idx + 1
        exp3_results[[exp3_idx]] <- row
      }
    }
  }

  # Checkpoint
  if (sim %% max(1, N_SIM %/% 10) == 0 || sim == N_SIM) {
    exp3_df_partial <- bind_rows(exp3_results)
    write_csv(exp3_df_partial, file.path(out_dir, "sim_scoring_exp3.csv"))
    cat(sprintf("  [checkpoint] sim %d: %d rows\n", sim, nrow(exp3_df_partial)))
  }
}

exp3_df <- bind_rows(exp3_results)
write_csv(exp3_df, file.path(out_dir, "sim_scoring_exp3.csv"))
cat("Saved: sim_scoring_exp3.csv\n")

cat("\n========== Experiment 3 Summary ==========\n")
exp3_summary <- exp3_df %>%
  group_by(kappa_ct, nonlinear_gamma, n_scale_points, method) %>%
  summarize(
    tau_mean = mean(tau), tau_se = sd(tau) / sqrt(n()),
    auc_mean = mean(auc),
    rmse_mean = mean(rank_mse),
    .groups = "drop"
  )

# Show crossover points
exp3_cross <- exp3_summary %>%
  filter(method %in% c("likert_mean", "bt_tournament")) %>%
  select(kappa_ct, nonlinear_gamma, n_scale_points, method, tau_mean) %>%
  pivot_wider(names_from = method, values_from = tau_mean) %>%
  mutate(bt_advantage = bt_tournament - likert_mean)
cat("\nBT advantage by (kappa, gamma, scale_points):\n")
print(exp3_cross, width = 120, n = 50)

# --- Figure 3: Discretization + Non-linear ---
fig3_data <- exp3_df %>%
  mutate(
    method_label = factor(recode(method, !!!method_labels),
                          levels = method_order),
    kappa_label  = sprintf("kappa = %.1f", kappa_ct),
    gamma_label  = sprintf("gamma = %.1f", nonlinear_gamma)
  ) %>%
  group_by(n_scale_points, kappa_ct, nonlinear_gamma,
           kappa_label, gamma_label, method_label) %>%
  summarize(
    tau_mean = mean(tau), tau_se = sd(tau) / sqrt(n()),
    .groups = "drop"
  )

fig3_data$kappa_label <- factor(fig3_data$kappa_label,
                                levels = sprintf("kappa = %.1f", rev(KAPPA_LEVELS)))
fig3_data$gamma_label <- factor(fig3_data$gamma_label,
                                levels = sprintf("gamma = %.1f", NONLINEAR_GAMMA_LEVELS))

p3 <- ggplot(fig3_data, aes(x = n_scale_points, y = tau_mean,
                              color = method_label, linetype = method_label,
                              shape = method_label)) +
  geom_ribbon(aes(ymin = tau_mean - tau_se, ymax = tau_mean + tau_se,
                  fill = method_label), alpha = 0.10, color = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_grid(kappa_label ~ gamma_label) +
  scale_color_manual(values = scoring_colors, name = NULL) +
  scale_fill_manual(values = scoring_colors, guide = "none") +
  scale_linetype_manual(values = scoring_linetypes, name = NULL) +
  scale_shape_manual(values = scoring_shapes, name = NULL) +
  scale_x_continuous(breaks = SCALE_POINTS_LEVELS,
                     labels = c("3", "5", "7", "100")) +
  labs(
    title = "Scoring Method Recovery Under Scale Pathologies",
    subtitle = sprintf(
      paste0("Rows: compression. Cols: non-linear scale use. ",
             "x-axis: scale points (100 = continuous). N_SIM=%d."),
      N_SIM),
    x = "Number of scale points",
    y = expression("Kendall " * tau * " with true quality")
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    strip.text = element_text(size = 9, face = "bold")
  ) +
  guides(color = guide_legend(nrow = 1),
         linetype = guide_legend(nrow = 1),
         shape = guide_legend(nrow = 1))

ggsave(file.path(fig_dir, "sim_scoring_exp3.pdf"), p3,
       width = 10, height = 10)
ggsave(file.path(fig_dir, "sim_scoring_exp3.png"), p3,
       width = 10, height = 10, dpi = 300)
cat("Saved: sim_scoring_exp3.pdf/png\n")

# =========================================================================
# Section 14: Experiment 4 -- Full 5-Way Pathology Sweep
# =========================================================================
cat("\n========== Experiment 4: Full 5-Way Pathology Sweep ==========\n")

# All 5 pathologies crossed: kappa × anchor × slope × gamma × scale_points
# 4 × 4 × 4 × 3 × 4 = 768 conditions.
# BT varies only with sigma_slope → 4 BT fits per sim, reused across
# the remaining 4 × 4 × 3 × 4 = 192 Likert conditions per slope level.

cat(sprintf("Exp 4: %d conditions per sim\n", n_conditions_exp4))

exp4_results <- list()
exp4_idx <- 0

for (sim in seq_len(N_SIM)) {
  if (sim %% max(1, N_SIM %/% 4) == 0) cat(sprintf("  sim %d/%d\n", sim, N_SIM))
  set.seed(42 + sim)

  resp_df   <- generate_response_pool()
  cell_mean <- generate_cell_means(resp_df)
  pairs     <- sample_tournament_pairs(resp_df)
  pairs_eq  <- sample_cost_eq_pairs(resp_df)

  # Pre-compute BT per sigma_slope level (same pattern as Exp 2)
  bt_by_slope <- list()
  for (s_idx in seq_along(SIGMA_SLOPE_LEVELS)) {
    sigma_slope <- SIGMA_SLOPE_LEVELS[s_idx]
    a_m <- if (sigma_slope > 0) rnorm(N_JUDGES, 1, sigma_slope) else rep(1, N_JUDGES)

    pw    <- observe_pairwise(cell_mean, pairs, VC$sigma2_eps, a_m)
    pw_eq <- observe_pairwise(cell_mean, pairs_eq, VC$sigma2_eps, a_m,
                              n_prompts = COST_EQ_N_PROMPTS,
                              n_judges  = COST_EQ_N_JUDGES,
                              n_temps   = COST_EQ_N_TEMPS,
                              n_reps    = COST_EQ_N_REPS)
    pw_rr <- observe_pairwise(cell_mean, full_rr_pairs, VC$sigma2_eps, a_m,
                              n_prompts = FULL_RR_N_PROMPTS,
                              n_judges  = FULL_RR_N_JUDGES,
                              n_temps   = FULL_RR_N_TEMPS,
                              n_reps    = FULL_RR_N_REPS)

    bt    <- fit_bt_mm(pw$resp_a, pw$resp_b, pw$wins_a, pw$n_per_pair)
    bt_eq <- fit_bt_mm(pw_eq$resp_a, pw_eq$resp_b,
                       pw_eq$wins_a, pw_eq$n_per_pair)
    bt_rr <- fit_bt_mm(pw_rr$resp_a, pw_rr$resp_b,
                       pw_rr$wins_a, pw_rr$n_per_pair)

    bt_by_slope[[s_idx]] <- list(
      a_m = a_m,
      bt_scores    = bt$scores,
      bt_eq_scores = bt_eq$scores,
      bt_rr_scores = bt_rr$scores,
      bt_conv      = bt$converged,
      bt_eq_conv   = bt_eq$converged,
      bt_rr_conv   = bt_rr$converged
    )
  }

  # Sweep all 5 pathologies (Likert varies with all; BT keyed by slope)
  for (s_idx in seq_along(SIGMA_SLOPE_LEVELS)) {
    sigma_slope <- SIGMA_SLOPE_LEVELS[s_idx]
    bt_s <- bt_by_slope[[s_idx]]

    m_bt   <- compute_metrics(resp_df$theta, bt_s$bt_scores, "bt_tournament")
    m_bte  <- compute_metrics(resp_df$theta, bt_s$bt_eq_scores, "bt_cost_eq")
    m_btrr <- compute_metrics(resp_df$theta, bt_s$bt_rr_scores, "bt_full_rr")

    for (kappa_ct in KAPPA_LEVELS) {
      for (sigma_anchor in SIGMA_ANCHOR_LEVELS) {
        for (gamma in NONLINEAR_GAMMA_LEVELS) {
          for (n_sp in SCALE_POINTS_LEVELS) {
            theta_likert <- observe_likert(cell_mean, VC$sigma2_eps, bt_s$a_m,
                                           kappa_ct, sigma_anchor,
                                           nonlinear_gamma = gamma,
                                           n_scale_points = n_sp)
            m_lik <- compute_metrics(resp_df$theta, theta_likert, "likert_mean")

            row <- bind_rows(m_lik, m_bt, m_bte, m_btrr) %>%
              mutate(
                sim_id = sim,
                kappa_ct = kappa_ct,
                sigma_anchor = sigma_anchor,
                sigma_slope = sigma_slope,
                nonlinear_gamma = gamma,
                n_scale_points = n_sp,
                bt_converged = bt_s$bt_conv,
                bt_eq_converged = bt_s$bt_eq_conv,
                bt_rr_converged = bt_s$bt_rr_conv
              )

            exp4_idx <- exp4_idx + 1
            exp4_results[[exp4_idx]] <- row
          }
        }
      }
    }
  }

  # Checkpoint
  if (sim %% max(1, N_SIM %/% 10) == 0 || sim == N_SIM) {
    exp4_df_partial <- bind_rows(exp4_results)
    write_csv(exp4_df_partial, file.path(out_dir, "sim_scoring_exp4.csv"))
    cat(sprintf("  [checkpoint] sim %d: %d rows\n", sim, nrow(exp4_df_partial)))
  }
}

exp4_df <- bind_rows(exp4_results)
write_csv(exp4_df, file.path(out_dir, "sim_scoring_exp4.csv"))
cat("Saved: sim_scoring_exp4.csv\n")

cat("\n========== Experiment 4 Summary ==========\n")

# BT win rate across all conditions
exp4_cross <- exp4_df %>%
  filter(method %in% c("likert_mean", "bt_tournament")) %>%
  group_by(kappa_ct, sigma_anchor, sigma_slope,
           nonlinear_gamma, n_scale_points, method) %>%
  summarize(tau_mean = mean(tau), .groups = "drop") %>%
  pivot_wider(names_from = method, values_from = tau_mean) %>%
  mutate(bt_advantage = bt_tournament - likert_mean)

bt4_wins <- sum(exp4_cross$bt_advantage > 0, na.rm = TRUE)
cat(sprintf("Tournament BT wins %d/%d conditions (%.0f%%)\n",
            bt4_wins, nrow(exp4_cross),
            bt4_wins / nrow(exp4_cross) * 100))

# Same for cost-equated
exp4_cross_eq <- exp4_df %>%
  filter(method %in% c("likert_mean", "bt_cost_eq")) %>%
  group_by(kappa_ct, sigma_anchor, sigma_slope,
           nonlinear_gamma, n_scale_points, method) %>%
  summarize(tau_mean = mean(tau), .groups = "drop") %>%
  pivot_wider(names_from = method, values_from = tau_mean) %>%
  mutate(bt_advantage = bt_cost_eq - likert_mean)

bt4_eq_wins <- sum(exp4_cross_eq$bt_advantage > 0, na.rm = TRUE)
cat(sprintf("Cost-equated BT wins %d/%d conditions (%.0f%%)\n",
            bt4_eq_wins, nrow(exp4_cross_eq),
            bt4_eq_wins / nrow(exp4_cross_eq) * 100))

# Full RR crossover
exp4_cross_rr <- exp4_df %>%
  filter(method %in% c("likert_mean", "bt_full_rr")) %>%
  group_by(kappa_ct, sigma_anchor, sigma_slope,
           nonlinear_gamma, n_scale_points, method) %>%
  summarize(tau_mean = mean(tau), .groups = "drop") %>%
  pivot_wider(names_from = method, values_from = tau_mean) %>%
  mutate(bt_advantage = bt_full_rr - likert_mean)

bt4_rr_wins <- sum(exp4_cross_rr$bt_advantage > 0, na.rm = TRUE)
cat(sprintf("Full RR BT wins %d/%d conditions (%.0f%%)\n",
            bt4_rr_wins, nrow(exp4_cross_rr),
            bt4_rr_wins / nrow(exp4_cross_rr) * 100))

# --- Figure 4: Marginal Effects of Each Pathology on BT Advantage ---
# For each pathology dimension, average BT advantage over all other dimensions.

# Compute per-sim × condition BT advantages (3 arms)
exp4_wide <- exp4_df %>%
  select(sim_id, method, tau, kappa_ct, sigma_anchor, sigma_slope,
         nonlinear_gamma, n_scale_points) %>%
  pivot_wider(names_from = method, values_from = tau) %>%
  mutate(
    bt_rr_adv = bt_full_rr - likert_mean,
    bt_adv    = bt_tournament - likert_mean,
    bt_eq_adv = bt_cost_eq - likert_mean
  )

# Helper: compute marginal for one grouping variable
compute_marginal <- function(data, group_col, pathology_name,
                             fmt = "%.1f") {
  data %>%
    group_by(level = .data[[group_col]]) %>%
    summarize(
      bt_rr_adv_mean = mean(bt_rr_adv), bt_rr_adv_se = sd(bt_rr_adv) / sqrt(n()),
      bt_adv_mean = mean(bt_adv), bt_adv_se = sd(bt_adv) / sqrt(n()),
      bt_eq_adv_mean = mean(bt_eq_adv), bt_eq_adv_se = sd(bt_eq_adv) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(pathology = pathology_name,
           level_label = if (fmt == "int") as.character(as.integer(level))
                         else sprintf(fmt, level))
}

marginal_df <- bind_rows(
  compute_marginal(exp4_wide, "kappa_ct", "Compression (kappa)"),
  compute_marginal(exp4_wide, "sigma_anchor", "Anchoring noise (sigma_anchor)"),
  compute_marginal(exp4_wide, "sigma_slope", "DIF / scale-use hetero. (sigma_slope)"),
  compute_marginal(exp4_wide, "nonlinear_gamma", "Non-linear scale use (gamma)"),
  compute_marginal(exp4_wide, "n_scale_points", "Discretization (scale points)", "int")
)

# Pivot to long: one row per (pathology, level, arm)
arm_specs <- list(
  list(mean_col = "bt_rr_adv_mean", se_col = "bt_rr_adv_se", label = "BT (full RR)"),
  list(mean_col = "bt_adv_mean",    se_col = "bt_adv_se",    label = "BT (tournament)"),
  list(mean_col = "bt_eq_adv_mean", se_col = "bt_eq_adv_se", label = "BT (cost-equated)")
)
fig4_long <- bind_rows(lapply(arm_specs, function(spec) {
  marginal_df %>%
    transmute(pathology, level, level_label,
              arm = spec$label,
              advantage = .data[[spec$mean_col]],
              se = .data[[spec$se_col]])
}))
fig4_long$arm <- factor(fig4_long$arm,
  levels = c("BT (full RR)", "BT (tournament)", "BT (cost-equated)"))

fig4_long$pathology <- factor(fig4_long$pathology,
  levels = c("Compression (kappa)", "Anchoring noise (sigma_anchor)",
             "DIF / scale-use hetero. (sigma_slope)",
             "Non-linear scale use (gamma)",
             "Discretization (scale points)"))

p4 <- ggplot(fig4_long, aes(x = level_label, y = advantage,
                              color = arm, shape = arm)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_pointrange(aes(ymin = advantage - 1.96 * se,
                       ymax = advantage + 1.96 * se),
                  position = position_dodge(width = 0.5), size = 0.4) +
  facet_wrap(~ pathology, scales = "free_x", nrow = 1) +
  scale_color_manual(values = c("BT (full RR)" = "#4daf4a",
                                "BT (tournament)" = "#e08040",
                                "BT (cost-equated)" = "#b2182b"),
                     name = NULL) +
  scale_shape_manual(values = c("BT (full RR)" = 18,
                                "BT (tournament)" = 17,
                                "BT (cost-equated)" = 15),
                     name = NULL) +
  labs(
    title = "Marginal BT Advantage Over Likert Across All Pathologies",
    subtitle = sprintf(
      paste0("Each panel: one pathology dimension, averaged over the other four. ",
             "Dashed line: Likert = BT. N_SIM=%d."),
      N_SIM),
    x = "Pathology level",
    y = expression(Delta * tau ~ "(BT" - "Likert)")
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    strip.text = element_text(size = 8, face = "bold"),
    axis.text.x = element_text(size = 8)
  )

ggsave(file.path(fig_dir, "sim_scoring_exp4.pdf"), p4,
       width = 14, height = 5)
ggsave(file.path(fig_dir, "sim_scoring_exp4.png"), p4,
       width = 14, height = 5, dpi = 300)
cat("Saved: sim_scoring_exp4.pdf/png\n")

cat("\n=== 06_sim_scoring_recovery.R complete ===\n")
