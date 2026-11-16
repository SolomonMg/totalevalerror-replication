# 04f_sim_dstudy_validation.R
# Monte Carlo simulation: D-study projection validation under misspecification.
# Tests whether D-study projections (extrapolating from a G-study to larger sample sizes)
# are accurate, and how they degrade under realistic misspecification scenarios.

library(tidyverse)
library(lme4)

set.seed(42)

N_SIM <- 1000

# --- Baseline DGP parameters (same as other sims) ---
TRUE_PARAMS <- list(
  mu = 0.5,
  sigma2_category = 0.03,
  sigma2_item = 0.05,
  sigma2_prompt = 0.04,
  sigma2_ip = 0.02,
  sigma2_it = 0.015,
  sigma2_pt = 0.005,
  sigma2_gen = 0.06,
  tau = c(0, -0.03, 0.05),
  n_categories = 5
)

# G-study design (what we observe)
G_ITEMS <- 50
G_PROMPTS <- 4
G_TEMPS <- 3
G_REPS <- 5
G_CATS <- TRUE_PARAMS$n_categories

cat("=== 04f_sim_dstudy_validation.R: D-Study Projection Validation ===\n")
cat(sprintf("G-study design: N=%d, C=%d, K=%d, L=%d, R=%d, N_SIM=%d\n",
            G_ITEMS, G_CATS, G_PROMPTS, G_TEMPS, G_REPS, N_SIM))

# =========================================================================
# Helper: generate data from the DGP (correctly specified)
# =========================================================================
simulate_data <- function(n_items, n_prompts, n_temps, n_reps, params) {
  n_cats <- params$n_categories
  items_per_cat <- ceiling(n_items / n_cats)

  gamma_cat <- rnorm(n_cats, 0, sqrt(params$sigma2_category))
  item_cat <- rep(seq_len(n_cats), each = items_per_cat)[seq_len(n_items)]
  delta_item <- rnorm(n_items, 0, sqrt(params$sigma2_item))
  alpha <- gamma_cat[item_cat] + delta_item

  beta <- rnorm(n_prompts, 0, sqrt(params$sigma2_prompt))

  ab <- matrix(rnorm(n_items * n_prompts, 0, sqrt(params$sigma2_ip)),
               nrow = n_items, ncol = n_prompts)
  at <- matrix(rnorm(n_items * n_temps, 0, sqrt(params$sigma2_it)),
               nrow = n_items, ncol = n_temps)
  bt <- matrix(rnorm(n_prompts * n_temps, 0, sqrt(params$sigma2_pt)),
               nrow = n_prompts, ncol = n_temps)

  rows <- vector("list", n_items * n_prompts * n_temps * n_reps)
  idx <- 0L
  for (i in seq_len(n_items)) {
    for (j in seq_len(n_prompts)) {
      for (k in seq_len(n_temps)) {
        for (r in seq_len(n_reps)) {
          idx <- idx + 1L
          y <- params$mu + alpha[i] + beta[j] + params$tau[k] +
            ab[i, j] + at[i, k] + bt[j, k] +
            rnorm(1, 0, sqrt(params$sigma2_gen))
          rows[[idx]] <- list(
            item_id = paste0("item_", i),
            category = paste0("cat_", item_cat[i]),
            variant_id = paste0("v_", j),
            temperature = paste0("t_", k),
            replication = r,
            outcome = y
          )
        }
      }
    }
  }
  bind_rows(rows) %>%
    mutate(
      item_id = as.factor(item_id),
      category = as.factor(category),
      variant_id = as.factor(variant_id),
      temperature = as.factor(temperature)
    )
}

# =========================================================================
# Helper: generate misspecified data
# =========================================================================

# Scenario 2: Correlated random effects — items with extreme alpha have larger interactions
simulate_data_correlated <- function(n_items, n_prompts, n_temps, n_reps, params, rho) {
  n_cats <- params$n_categories
  items_per_cat <- ceiling(n_items / n_cats)

  gamma_cat <- rnorm(n_cats, 0, sqrt(params$sigma2_category))
  item_cat <- rep(seq_len(n_cats), each = items_per_cat)[seq_len(n_items)]
  delta_item <- rnorm(n_items, 0, sqrt(params$sigma2_item))
  alpha <- gamma_cat[item_cat] + delta_item

  beta <- rnorm(n_prompts, 0, sqrt(params$sigma2_prompt))

  # Interaction variance scales with |alpha_i|
  sd_alpha <- sqrt(params$sigma2_category + params$sigma2_item)
  ip_scale <- 1 + rho * abs(alpha) / sd_alpha
  ab <- matrix(NA, n_items, n_prompts)
  for (i in seq_len(n_items)) {
    ab[i, ] <- rnorm(n_prompts, 0, sqrt(params$sigma2_ip * ip_scale[i]))
  }

  at <- matrix(rnorm(n_items * n_temps, 0, sqrt(params$sigma2_it)),
               nrow = n_items, ncol = n_temps)
  bt <- matrix(rnorm(n_prompts * n_temps, 0, sqrt(params$sigma2_pt)),
               nrow = n_prompts, ncol = n_temps)

  rows <- vector("list", n_items * n_prompts * n_temps * n_reps)
  idx <- 0L
  for (i in seq_len(n_items)) {
    for (j in seq_len(n_prompts)) {
      for (k in seq_len(n_temps)) {
        for (r in seq_len(n_reps)) {
          idx <- idx + 1L
          y <- params$mu + alpha[i] + beta[j] + params$tau[k] +
            ab[i, j] + at[i, k] + bt[j, k] +
            rnorm(1, 0, sqrt(params$sigma2_gen))
          rows[[idx]] <- list(
            item_id = paste0("item_", i),
            category = paste0("cat_", item_cat[i]),
            variant_id = paste0("v_", j),
            temperature = paste0("t_", k),
            replication = r,
            outcome = y
          )
        }
      }
    }
  }
  bind_rows(rows) %>%
    mutate(item_id = as.factor(item_id), category = as.factor(category),
           variant_id = as.factor(variant_id), temperature = as.factor(temperature))
}

# Scenario 3: Non-exchangeable prompts — two populations of prompt quality
simulate_data_nonexch_prompts <- function(n_items, n_prompts, n_temps, n_reps, params,
                                          variance_ratio = 4) {
  n_cats <- params$n_categories
  items_per_cat <- ceiling(n_items / n_cats)

  gamma_cat <- rnorm(n_cats, 0, sqrt(params$sigma2_category))
  item_cat <- rep(seq_len(n_cats), each = items_per_cat)[seq_len(n_items)]
  delta_item <- rnorm(n_items, 0, sqrt(params$sigma2_item))
  alpha <- gamma_cat[item_cat] + delta_item

  # Two populations: first half low-variance, second half high-variance
  n_low <- n_prompts %/% 2
  n_high <- n_prompts - n_low
  beta <- c(
    rnorm(n_low, 0, sqrt(params$sigma2_prompt)),
    rnorm(n_high, 0, sqrt(params$sigma2_prompt * variance_ratio))
  )

  ab <- matrix(rnorm(n_items * n_prompts, 0, sqrt(params$sigma2_ip)),
               nrow = n_items, ncol = n_prompts)
  at <- matrix(rnorm(n_items * n_temps, 0, sqrt(params$sigma2_it)),
               nrow = n_items, ncol = n_temps)
  bt <- matrix(rnorm(n_prompts * n_temps, 0, sqrt(params$sigma2_pt)),
               nrow = n_prompts, ncol = n_temps)

  rows <- vector("list", n_items * n_prompts * n_temps * n_reps)
  idx <- 0L
  for (i in seq_len(n_items)) {
    for (j in seq_len(n_prompts)) {
      for (k in seq_len(n_temps)) {
        for (r in seq_len(n_reps)) {
          idx <- idx + 1L
          y <- params$mu + alpha[i] + beta[j] + params$tau[k] +
            ab[i, j] + at[i, k] + bt[j, k] +
            rnorm(1, 0, sqrt(params$sigma2_gen))
          rows[[idx]] <- list(
            item_id = paste0("item_", i),
            category = paste0("cat_", item_cat[i]),
            variant_id = paste0("v_", j),
            temperature = paste0("t_", k),
            replication = r,
            outcome = y
          )
        }
      }
    }
  }
  bind_rows(rows) %>%
    mutate(item_id = as.factor(item_id), category = as.factor(category),
           variant_id = as.factor(variant_id), temperature = as.factor(temperature))
}

# Scenario 4: Non-Gaussian random effects and residuals
# Preserves category/item nesting structure but draws from non-Gaussian distributions
simulate_data_nongaussian <- function(n_items, n_prompts, n_temps, n_reps, params, df_t = 5) {
  n_cats <- params$n_categories
  items_per_cat <- ceiling(n_items / n_cats)

  # Category effects: shifted log-normal (skewed), rescaled to target SD
  # Use POPULATION SD to avoid inflating mean variance with small n
  # (dividing by sample sd(5 values) inflates Var(mean) by (n-1)/(n-3) = 2×)
  pop_sd_lognormal <- sqrt((exp(0.25) - 1) * exp(0.25))  # SD of exp(N(0,0.5)) - exp(0.125)
  gamma_raw <- exp(rnorm(n_cats, 0, 0.5)) - exp(0.5^2 / 2)
  gamma_cat <- gamma_raw * sqrt(params$sigma2_category) / pop_sd_lognormal

  item_cat <- rep(seq_len(n_cats), each = items_per_cat)[seq_len(n_items)]

  # Item effects within category: also skewed
  delta_raw <- exp(rnorm(n_items, 0, 0.5)) - exp(0.5^2 / 2)
  delta_item <- delta_raw * sqrt(params$sigma2_item) / pop_sd_lognormal

  alpha <- gamma_cat[item_cat] + delta_item

  beta <- rnorm(n_prompts, 0, sqrt(params$sigma2_prompt))

  ab <- matrix(rnorm(n_items * n_prompts, 0, sqrt(params$sigma2_ip)),
               nrow = n_items, ncol = n_prompts)
  at <- matrix(rnorm(n_items * n_temps, 0, sqrt(params$sigma2_it)),
               nrow = n_items, ncol = n_temps)
  bt <- matrix(rnorm(n_prompts * n_temps, 0, sqrt(params$sigma2_pt)),
               nrow = n_prompts, ncol = n_temps)

  rows <- vector("list", n_items * n_prompts * n_temps * n_reps)
  idx <- 0L
  for (i in seq_len(n_items)) {
    for (j in seq_len(n_prompts)) {
      for (k in seq_len(n_temps)) {
        for (r in seq_len(n_reps)) {
          idx <- idx + 1L
          # t-distributed residual, scaled to match target variance
          eps <- rt(1, df = df_t) * sqrt(params$sigma2_gen * (df_t - 2) / df_t)
          y <- params$mu + alpha[i] + beta[j] + params$tau[k] +
            ab[i, j] + at[i, k] + bt[j, k] + eps
          rows[[idx]] <- list(
            item_id = paste0("item_", i),
            category = paste0("cat_", item_cat[i]),
            variant_id = paste0("v_", j),
            temperature = paste0("t_", k),
            replication = r,
            outcome = y
          )
        }
      }
    }
  }
  bind_rows(rows) %>%
    mutate(item_id = as.factor(item_id), category = as.factor(category),
           variant_id = as.factor(variant_id), temperature = as.factor(temperature))
}

# Scenario 5: Heterogeneous interactions by category
simulate_data_hetero_category <- function(n_items, n_prompts, n_temps, n_reps, params,
                                          easy_mult = 0.25, hard_mult = 4) {
  n_cats <- params$n_categories
  items_per_cat <- ceiling(n_items / n_cats)

  gamma_cat <- rnorm(n_cats, 0, sqrt(params$sigma2_category))
  item_cat <- rep(seq_len(n_cats), each = items_per_cat)[seq_len(n_items)]
  delta_item <- rnorm(n_items, 0, sqrt(params$sigma2_item))
  alpha <- gamma_cat[item_cat] + delta_item

  beta <- rnorm(n_prompts, 0, sqrt(params$sigma2_prompt))

  # Per-category interaction variance multiplier
  # Categories 1-2: easy (low interaction), 3: medium, 4-5: hard (high interaction)
  cat_mult <- c(rep(easy_mult, 2), 1, rep(hard_mult, 2))

  ab <- matrix(NA, n_items, n_prompts)
  for (i in seq_len(n_items)) {
    mult <- cat_mult[item_cat[i]]
    ab[i, ] <- rnorm(n_prompts, 0, sqrt(params$sigma2_ip * mult))
  }

  at <- matrix(rnorm(n_items * n_temps, 0, sqrt(params$sigma2_it)),
               nrow = n_items, ncol = n_temps)
  bt <- matrix(rnorm(n_prompts * n_temps, 0, sqrt(params$sigma2_pt)),
               nrow = n_prompts, ncol = n_temps)

  rows <- vector("list", n_items * n_prompts * n_temps * n_reps)
  idx <- 0L
  for (i in seq_len(n_items)) {
    for (j in seq_len(n_prompts)) {
      for (k in seq_len(n_temps)) {
        for (r in seq_len(n_reps)) {
          idx <- idx + 1L
          y <- params$mu + alpha[i] + beta[j] + params$tau[k] +
            ab[i, j] + at[i, k] + bt[j, k] +
            rnorm(1, 0, sqrt(params$sigma2_gen))
          rows[[idx]] <- list(
            item_id = paste0("item_", i),
            category = paste0("cat_", item_cat[i]),
            variant_id = paste0("v_", j),
            temperature = paste0("t_", k),
            replication = r,
            outcome = y
          )
        }
      }
    }
  }
  bind_rows(rows) %>%
    mutate(item_id = as.factor(item_id), category = as.factor(category),
           variant_id = as.factor(variant_id), temperature = as.factor(temperature))
}


# =========================================================================
# Fit model and extract variance components
# =========================================================================
fit_and_extract <- function(df) {
  mod <- tryCatch(
    lmer(
      outcome ~ temperature +
        (1 | category) + (1 | item_id) + (1 | variant_id) +
        (1 | item_id:variant_id) + (1 | item_id:temperature) +
        (1 | variant_id:temperature),
      data = df, REML = TRUE,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10000))
    ),
    error = function(e) NULL
  )
  if (is.null(mod)) return(NULL)

  vc <- as.data.frame(VarCorr(mod))
  tibble(component = vc$grp, variance = vc$vcov)
}

# =========================================================================
# D-study projection formula
# =========================================================================
compute_dstudy_var <- function(vc, n_items_d, n_prompts_d, n_reps_d, n_temps_d, fix_temp) {
  s2 <- setNames(vc$variance, vc$component)
  s2_cat <- s2["category"]
  s2_item <- s2["item_id"]
  s2_prompt <- s2["variant_id"]
  s2_ip <- s2["item_id:variant_id"]
  s2_it <- s2["item_id:temperature"]
  s2_pt <- s2["variant_id:temperature"]
  s2_gen <- s2["Residual"]

  n_cats <- G_CATS

  if (fix_temp) {
    s2_cat/n_cats + s2_item/n_items_d + s2_prompt/n_prompts_d +
      s2_ip/(n_items_d * n_prompts_d) + s2_it/n_items_d + s2_pt/n_prompts_d +
      s2_gen/(n_items_d * n_prompts_d * n_reps_d)
  } else {
    s2_cat/n_cats + s2_item/n_items_d + s2_prompt/n_prompts_d +
      s2_ip/(n_items_d * n_prompts_d) + s2_it/(n_items_d * n_temps_d) +
      s2_pt/(n_prompts_d * n_temps_d) + s2_gen/(n_items_d * n_prompts_d * n_temps_d * n_reps_d)
  }
}

# =========================================================================
# Compute "actual" Var(theta-hat) from repeated simulation at target design
# =========================================================================
compute_actual_var <- function(sim_func, n_items_d, n_prompts_d, n_reps_d, n_temps_d,
                               params, n_mc = 5000, ...) {
  theta_hats <- numeric(n_mc)
  for (m in seq_len(n_mc)) {
    df <- sim_func(n_items_d, n_prompts_d, n_temps_d, n_reps_d, params, ...)
    theta_hats[m] <- mean(df$outcome)
  }
  var(theta_hats)
}

# =========================================================================
# Make true variance components tibble (for D-study projection from truth)
# =========================================================================
make_true_vc <- function(params) {
  tibble(
    component = c("category", "item_id", "variant_id", "item_id:variant_id",
                   "item_id:temperature", "variant_id:temperature", "Residual"),
    variance = c(params$sigma2_category, params$sigma2_item, params$sigma2_prompt,
                 params$sigma2_ip, params$sigma2_it, params$sigma2_pt, params$sigma2_gen)
  )
}

true_vc <- make_true_vc(TRUE_PARAMS)

# =========================================================================
# D-study target designs (extrapolation from G-study)
# =========================================================================
dstudy_targets <- tribble(
  ~target_name, ~n_items_d, ~n_prompts_d, ~n_reps_d, ~fix_temp, ~extrap_ratio,
  "K'=6 (1.5x)",   G_ITEMS,           6,    G_REPS,     FALSE,          1.5,
  "K'=8 (2x)",     G_ITEMS,           8,    G_REPS,     FALSE,          2.0,
  "K'=12 (3x)",    G_ITEMS,          12,    G_REPS,     FALSE,          3.0,
  "K'=20 (5x)",    G_ITEMS,          20,    G_REPS,     FALSE,          5.0,
  "R'=10 (2x)",    G_ITEMS,   G_PROMPTS,       10,     FALSE,          2.0,
  "R'=20 (4x)",    G_ITEMS,   G_PROMPTS,       20,     FALSE,          4.0,
  "K'=8 + R'=10",  G_ITEMS,           8,       10,     FALSE,          2.0,
  "Fix temp",      G_ITEMS,   G_PROMPTS,    G_REPS,      TRUE,          1.0,
)

# True D-study projections at each target (from known parameters)
true_projections <- dstudy_targets %>%
  rowwise() %>%
  mutate(true_projected_var = compute_dstudy_var(true_vc, n_items_d, n_prompts_d,
                                                   n_reps_d, G_TEMPS, fix_temp)) %>%
  ungroup()

cat("\nTrue D-study projections:\n")
print(true_projections %>% select(target_name, true_projected_var, extrap_ratio))


# #########################################################################
# SCENARIO 1: Correct specification — projection accuracy vs extrapolation
# #########################################################################
cat("\n\n========== SCENARIO 1: Correct specification ==========\n")

sc1_results <- list()

for (sim in seq_len(N_SIM)) {
  if (sim %% 200 == 0) cat(sprintf("  Sc1: sim %d/%d\n", sim, N_SIM))
  set.seed(42 + sim)

  # Generate G-study data and estimate variance components
  df_g <- simulate_data(G_ITEMS, G_PROMPTS, G_TEMPS, G_REPS, TRUE_PARAMS)
  vc_est <- fit_and_extract(df_g)
  if (is.null(vc_est)) next

  # Project to each target design
  for (t_idx in seq_len(nrow(dstudy_targets))) {
    tgt <- dstudy_targets[t_idx, ]
    projected_var <- compute_dstudy_var(vc_est, tgt$n_items_d, tgt$n_prompts_d,
                                        tgt$n_reps_d, G_TEMPS, tgt$fix_temp)
    sc1_results[[length(sc1_results) + 1]] <- list(
      sim_id = sim,
      target = tgt$target_name,
      extrap_ratio = tgt$extrap_ratio,
      projected_var = projected_var,
      true_projected_var = true_projections$true_projected_var[t_idx]
    )
  }
}

sc1_df <- bind_rows(sc1_results)

# Save per-simulation Sc1 data for scatter/ribbon figures
sc1_per_sim <- sc1_df %>%
  mutate(rel_bias = (projected_var - true_projected_var) / true_projected_var)
write_csv(sc1_per_sim, file.path("data/processed", "sim_dstudy_sc1_per_sim.csv"))
cat("Saved per-sim Sc1 data: data/processed/sim_dstudy_sc1_per_sim.csv\n")

sc1_summary <- sc1_df %>%
  group_by(target, extrap_ratio) %>%
  summarize(
    mean_projected = mean(projected_var, na.rm = TRUE),
    actual_var = first(true_projected_var),
    rel_bias = mean((projected_var - true_projected_var) / true_projected_var, na.rm = TRUE),
    rmse = sqrt(mean((projected_var - true_projected_var)^2, na.rm = TRUE)),
    rel_rmse = rmse / first(true_projected_var),
    .groups = "drop"
  )

cat("\nScenario 1 — Correct specification, projection accuracy:\n")
print(sc1_summary)


# #########################################################################
# SCENARIO 2: Correlated random effects
# #########################################################################
cat("\n\n========== SCENARIO 2: Correlated random effects ==========\n")

rho_levels <- c(0, 0.5, 1.0, 2.0)

# Compute actual Var(theta-hat) at K'=8 under each correlated DGP
tgt_idx_sc2 <- which(dstudy_targets$target_name == "K'=8 (2x)")
tgt_sc2 <- dstudy_targets[tgt_idx_sc2, ]

cat("  Computing Monte Carlo reference variances for each rho...\n")
sc2_actual_vars <- numeric(length(rho_levels))
names(sc2_actual_vars) <- as.character(rho_levels)
for (ri in seq_along(rho_levels)) {
  set.seed(99999 + ri)
  sc2_actual_vars[ri] <- compute_actual_var(
    simulate_data_correlated,
    tgt_sc2$n_items_d, tgt_sc2$n_prompts_d, tgt_sc2$n_reps_d, G_TEMPS,
    TRUE_PARAMS, n_mc = 5000, rho = rho_levels[ri]
  )
  cat(sprintf("    rho=%.1f: actual Var(theta-hat) = %.6f\n",
              rho_levels[ri], sc2_actual_vars[ri]))
}

sc2_results <- list()

for (rho in rho_levels) {
  cat(sprintf("  rho = %.1f\n", rho))
  for (sim in seq_len(N_SIM)) {
    if (sim %% 200 == 0) cat(sprintf("    Sc2(rho=%.1f): sim %d/%d\n", rho, sim, N_SIM))
    set.seed(42 + sim + round(rho * 1000))

    df_g <- simulate_data_correlated(G_ITEMS, G_PROMPTS, G_TEMPS, G_REPS, TRUE_PARAMS, rho)
    vc_est <- fit_and_extract(df_g)
    if (is.null(vc_est)) next

    # Project to K'=8 (2x extrapolation on prompts)
    projected_var <- compute_dstudy_var(vc_est, tgt_sc2$n_items_d, tgt_sc2$n_prompts_d,
                                        tgt_sc2$n_reps_d, G_TEMPS, tgt_sc2$fix_temp)

    sc2_results[[length(sc2_results) + 1]] <- list(
      sim_id = sim,
      rho = rho,
      projected_var = projected_var,
      actual_var = sc2_actual_vars[as.character(rho)],
      est_sigma2_ip = vc_est$variance[vc_est$component == "item_id:variant_id"]
    )
  }
}

sc2_df <- bind_rows(sc2_results)

sc2_summary <- sc2_df %>%
  group_by(rho) %>%
  summarize(
    mean_projected = mean(projected_var, na.rm = TRUE),
    actual_var = first(actual_var),
    rel_bias = mean((projected_var - actual_var) / actual_var, na.rm = TRUE),
    mean_est_sigma2_ip = mean(est_sigma2_ip, na.rm = TRUE),
    true_sigma2_ip = TRUE_PARAMS$sigma2_ip,
    .groups = "drop"
  )

cat("\nScenario 2 — Correlated random effects:\n")
cat("(rel_bias is vs Monte Carlo actual variance, not baseline DGP)\n")
print(sc2_summary)


# #########################################################################
# SCENARIO 3: Non-exchangeable prompts
# #########################################################################
cat("\n\n========== SCENARIO 3: Non-exchangeable prompts ==========\n")

variance_ratios <- c(1, 2, 4, 8)

# Compute actual Var(theta-hat) at K'=8 under each non-exchangeable DGP
tgt_idx_sc3 <- which(dstudy_targets$target_name == "K'=8 (2x)")
tgt_sc3 <- dstudy_targets[tgt_idx_sc3, ]

cat("  Computing Monte Carlo reference variances for each variance_ratio...\n")
sc3_actual_vars <- numeric(length(variance_ratios))
names(sc3_actual_vars) <- as.character(variance_ratios)
for (vi in seq_along(variance_ratios)) {
  set.seed(88888 + vi)
  sc3_actual_vars[vi] <- compute_actual_var(
    simulate_data_nonexch_prompts,
    tgt_sc3$n_items_d, tgt_sc3$n_prompts_d, tgt_sc3$n_reps_d, G_TEMPS,
    TRUE_PARAMS, n_mc = 5000, variance_ratio = variance_ratios[vi]
  )
  cat(sprintf("    ratio=%d: actual Var(theta-hat) = %.6f\n",
              variance_ratios[vi], sc3_actual_vars[vi]))
}

sc3_results <- list()

for (vr in variance_ratios) {
  cat(sprintf("  variance_ratio = %d\n", vr))
  for (sim in seq_len(N_SIM)) {
    if (sim %% 200 == 0) cat(sprintf("    Sc3(vr=%d): sim %d/%d\n", vr, sim, N_SIM))
    set.seed(42 + sim + vr * 1000)

    df_g <- simulate_data_nonexch_prompts(G_ITEMS, G_PROMPTS, G_TEMPS, G_REPS,
                                           TRUE_PARAMS, variance_ratio = vr)
    vc_est <- fit_and_extract(df_g)
    if (is.null(vc_est)) next

    # Project to K'=8 — adding 4 more prompts
    projected_var <- compute_dstudy_var(vc_est, tgt_sc3$n_items_d, tgt_sc3$n_prompts_d,
                                        tgt_sc3$n_reps_d, G_TEMPS, tgt_sc3$fix_temp)

    sc3_results[[length(sc3_results) + 1]] <- list(
      sim_id = sim,
      variance_ratio = vr,
      projected_var = projected_var,
      actual_var = sc3_actual_vars[as.character(vr)],
      est_sigma2_prompt = vc_est$variance[vc_est$component == "variant_id"]
    )
  }
}

sc3_df <- bind_rows(sc3_results)

sc3_summary <- sc3_df %>%
  group_by(variance_ratio) %>%
  summarize(
    mean_projected = mean(projected_var, na.rm = TRUE),
    actual_var = first(actual_var),
    rel_bias = mean((projected_var - actual_var) / actual_var, na.rm = TRUE),
    mean_est_sigma2_prompt = mean(est_sigma2_prompt, na.rm = TRUE),
    true_sigma2_prompt = TRUE_PARAMS$sigma2_prompt,
    .groups = "drop"
  )

cat("\nScenario 3 — Non-exchangeable prompts:\n")
cat("(rel_bias is vs Monte Carlo actual variance, not baseline DGP)\n")
print(sc3_summary)


# #########################################################################
# SCENARIO 4: Non-Gaussian effects (skewed items, fat-tailed residuals)
# #########################################################################
cat("\n\n========== SCENARIO 4: Non-Gaussian random effects ==========\n")

df_t_levels <- c(100, 10, 5, 3)  # 100 ≈ Gaussian, 3 = heavy tails

# Compute actual Var(theta-hat) at K'=8 under each non-Gaussian DGP
tgt_idx_sc4 <- which(dstudy_targets$target_name == "K'=8 (2x)")
tgt_sc4 <- dstudy_targets[tgt_idx_sc4, ]

cat("  Computing Monte Carlo reference variances for each df_t...\n")
sc4_actual_vars <- numeric(length(df_t_levels))
names(sc4_actual_vars) <- as.character(df_t_levels)
for (di in seq_along(df_t_levels)) {
  set.seed(77777 + di)
  sc4_actual_vars[di] <- compute_actual_var(
    simulate_data_nongaussian,
    tgt_sc4$n_items_d, tgt_sc4$n_prompts_d, tgt_sc4$n_reps_d, G_TEMPS,
    TRUE_PARAMS, n_mc = 5000, df_t = df_t_levels[di]
  )
  cat(sprintf("    df=%d: actual Var(theta-hat) = %.6f\n",
              df_t_levels[di], sc4_actual_vars[di]))
}

sc4_results <- list()

for (df_t in df_t_levels) {
  cat(sprintf("  df_t = %d\n", df_t))
  for (sim in seq_len(N_SIM)) {
    if (sim %% 200 == 0) cat(sprintf("    Sc4(df=%d): sim %d/%d\n", df_t, sim, N_SIM))
    set.seed(42 + sim + df_t * 100)

    df_g <- simulate_data_nongaussian(G_ITEMS, G_PROMPTS, G_TEMPS, G_REPS,
                                       TRUE_PARAMS, df_t = df_t)
    vc_est <- fit_and_extract(df_g)
    if (is.null(vc_est)) next

    # Project to K'=8
    projected_var <- compute_dstudy_var(vc_est, tgt_sc4$n_items_d, tgt_sc4$n_prompts_d,
                                        tgt_sc4$n_reps_d, G_TEMPS, tgt_sc4$fix_temp)

    sc4_results[[length(sc4_results) + 1]] <- list(
      sim_id = sim,
      df_t = df_t,
      projected_var = projected_var,
      actual_var = sc4_actual_vars[as.character(df_t)]
    )
  }
}

sc4_df <- bind_rows(sc4_results)

sc4_summary <- sc4_df %>%
  group_by(df_t) %>%
  summarize(
    mean_projected = mean(projected_var, na.rm = TRUE),
    actual_var = first(actual_var),
    rel_bias = mean((projected_var - actual_var) / actual_var, na.rm = TRUE),
    rel_rmse = sqrt(mean(((projected_var - actual_var) / actual_var)^2,
                         na.rm = TRUE)),
    .groups = "drop"
  )

cat("\nScenario 4 — Non-Gaussian (skewed items + t-distributed residuals):\n")
cat("(rel_bias is vs Monte Carlo actual variance, not baseline DGP)\n")
print(sc4_summary)


# #########################################################################
# SCENARIO 5: Heterogeneous interactions by category
# #########################################################################
cat("\n\n========== SCENARIO 5: Heterogeneous interactions by category ==========\n")

# Compute actual Var(theta-hat) at K'=8 under heterogeneous category DGP
tgt_idx_sc5 <- which(dstudy_targets$target_name == "K'=8 (2x)")
tgt_sc5 <- dstudy_targets[tgt_idx_sc5, ]

cat("  Computing Monte Carlo reference variance...\n")
set.seed(66666)
sc5_actual_var <- compute_actual_var(
  simulate_data_hetero_category,
  tgt_sc5$n_items_d, tgt_sc5$n_prompts_d, tgt_sc5$n_reps_d, G_TEMPS,
  TRUE_PARAMS, n_mc = 5000
)
cat(sprintf("    actual Var(theta-hat) = %.6f\n", sc5_actual_var))

sc5_results <- list()

for (sim in seq_len(N_SIM)) {
  if (sim %% 200 == 0) cat(sprintf("  Sc5: sim %d/%d\n", sim, N_SIM))
  set.seed(42 + sim + 50000)

  df_g <- simulate_data_hetero_category(G_ITEMS, G_PROMPTS, G_TEMPS, G_REPS, TRUE_PARAMS)
  vc_est <- fit_and_extract(df_g)
  if (is.null(vc_est)) next

  # Project to K'=8
  projected_var <- compute_dstudy_var(vc_est, tgt_sc5$n_items_d, tgt_sc5$n_prompts_d,
                                      tgt_sc5$n_reps_d, G_TEMPS, tgt_sc5$fix_temp)

  sc5_results[[length(sc5_results) + 1]] <- list(
    sim_id = sim,
    projected_var = projected_var,
    actual_var = sc5_actual_var,
    est_sigma2_ip = vc_est$variance[vc_est$component == "item_id:variant_id"]
  )
}

sc5_df <- bind_rows(sc5_results)

sc5_summary <- sc5_df %>%
  summarize(
    mean_projected = mean(projected_var, na.rm = TRUE),
    actual_var = first(actual_var),
    rel_bias = mean((projected_var - actual_var) / actual_var, na.rm = TRUE),
    mean_est_sigma2_ip = mean(est_sigma2_ip, na.rm = TRUE),
    pooled_true_sigma2_ip = TRUE_PARAMS$sigma2_ip,
    .groups = "drop"
  )

cat("\nScenario 5 — Heterogeneous interactions by category:\n")
cat("(rel_bias is vs Monte Carlo actual variance, not baseline DGP)\n")
print(sc5_summary)


# #########################################################################
# DIRECTIONAL CORRECTNESS across all scenarios
# #########################################################################
cat("\n\n========== Directional correctness: 'Add prompts' vs 'Add reps' ==========\n")

# For each scenario, check: does the D-study correctly identify whether adding
# prompts or adding reps reduces variance more?

check_directional <- function(vc_est) {
  # Compare K'=8 vs R'=10
  proj_more_prompts <- compute_dstudy_var(vc_est, G_ITEMS, 8, G_REPS, G_TEMPS, FALSE)
  proj_more_reps <- compute_dstudy_var(vc_est, G_ITEMS, G_PROMPTS, 10, G_TEMPS, FALSE)
  # Return: which has lower projected variance?
  ifelse(proj_more_prompts < proj_more_reps, "prompts", "reps")
}

# True answer (from known parameters)
true_proj_prompts <- compute_dstudy_var(true_vc, G_ITEMS, 8, G_REPS, G_TEMPS, FALSE)
true_proj_reps <- compute_dstudy_var(true_vc, G_ITEMS, G_PROMPTS, 10, G_TEMPS, FALSE)
true_best <- ifelse(true_proj_prompts < true_proj_reps, "prompts", "reps")

cat(sprintf("True best intervention: %s (prompts: %.5f, reps: %.5f)\n",
            true_best, true_proj_prompts, true_proj_reps))

# Check directional accuracy for Scenario 1 (correctly specified)
sc1_directional <- sc1_df %>%
  filter(target %in% c("K'=8 (2x)", "R'=10 (2x)")) %>%
  pivot_wider(names_from = target, values_from = projected_var, id_cols = sim_id) %>%
  mutate(
    est_best = ifelse(`K'=8 (2x)` < `R'=10 (2x)`, "prompts", "reps"),
    correct = est_best == true_best
  )

cat(sprintf("\nSc1 directional accuracy: %.1f%% (%d/%d)\n",
            100 * mean(sc1_directional$correct, na.rm = TRUE),
            sum(sc1_directional$correct, na.rm = TRUE),
            sum(!is.na(sc1_directional$correct))))

# Rank preservation (Kendall's tau) across all targets for Scenario 1
sc1_rank <- sc1_df %>%
  group_by(sim_id) %>%
  summarize(
    tau = cor(projected_var, true_projected_var, method = "kendall"),
    .groups = "drop"
  )

cat(sprintf("Sc1 rank preservation (Kendall's tau): mean=%.3f, sd=%.3f\n",
            mean(sc1_rank$tau, na.rm = TRUE), sd(sc1_rank$tau, na.rm = TRUE)))


# =========================================================================
# Save all results
# =========================================================================
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(sc1_summary, file.path(output_dir, "sim_dstudy_sc1_correct.csv"))
write_csv(sc2_summary, file.path(output_dir, "sim_dstudy_sc2_correlated.csv"))
write_csv(sc3_summary, file.path(output_dir, "sim_dstudy_sc3_nonexch.csv"))
write_csv(sc4_summary, file.path(output_dir, "sim_dstudy_sc4_nongaussian.csv"))
write_csv(sc5_summary, file.path(output_dir, "sim_dstudy_sc5_hetero.csv"))

cat("\nSaved all scenario summaries to data/processed/sim_dstudy_sc*.csv\n")


# =========================================================================
# Figures
# =========================================================================
fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Figure 1: Projection bias by extrapolation ratio (Scenario 1)
p1 <- sc1_summary %>%
  ggplot(aes(x = extrap_ratio, y = rel_bias * 100)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 3) +
  geom_line() +
  geom_text(aes(label = target), hjust = -0.1, vjust = -0.5, size = 3) +
  scale_x_continuous(breaks = c(1, 1.5, 2, 3, 4, 5)) +
  labs(
    title = "D-Study Projection Bias Under Correct Specification",
    subtitle = sprintf("Relative bias of projected Var(theta-hat) vs truth (%d simulations)", N_SIM),
    x = "Extrapolation ratio (target / G-study sample size)",
    y = "Relative bias (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

ggsave(file.path(fig_dir, "sim_dstudy_bias_by_extrapolation.pdf"), p1, width = 9, height = 5)
ggsave(file.path(fig_dir, "sim_dstudy_bias_by_extrapolation.png"), p1, width = 9, height = 5, dpi = 300)
cat("Saved: sim_dstudy_bias_by_extrapolation.pdf/png\n")

# Figure 2: Misspecification comparison — all scenarios at K'=8
misspec_comparison <- bind_rows(
  sc1_summary %>%
    filter(target == "K'=8 (2x)") %>%
    mutate(scenario = "Correct spec"),
  sc2_summary %>%
    mutate(scenario = paste0("Correlated RE (rho=", rho, ")"),
           target = "K'=8 (2x)"),
  sc3_summary %>%
    mutate(scenario = paste0("Non-exch prompts (ratio=", variance_ratio, ")"),
           target = "K'=8 (2x)"),
  sc4_summary %>%
    mutate(scenario = paste0("Non-Gaussian (df=", df_t, ")"),
           target = "K'=8 (2x)"),
  sc5_summary %>%
    mutate(scenario = "Hetero category interactions",
           target = "K'=8 (2x)")
)

# Keep only the most informative levels
misspec_plot <- misspec_comparison %>%
  filter(scenario %in% c(
    "Correct spec",
    "Correlated RE (rho=0)", "Correlated RE (rho=1)", "Correlated RE (rho=2)",
    "Non-exch prompts (ratio=1)", "Non-exch prompts (ratio=4)", "Non-exch prompts (ratio=8)",
    "Non-Gaussian (df=100)", "Non-Gaussian (df=5)", "Non-Gaussian (df=3)",
    "Hetero category interactions"
  )) %>%
  mutate(scenario = fct_reorder(scenario, rel_bias))

p2 <- misspec_plot %>%
  ggplot(aes(x = rel_bias * 100, y = scenario)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 3, color = "steelblue") +
  labs(
    title = "D-Study Projection Bias Under Misspecification",
    subtitle = "Relative bias at K'=8 (2× prompt extrapolation)",
    x = "Relative bias (%)",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

ggsave(file.path(fig_dir, "sim_dstudy_misspecification.pdf"), p2, width = 9, height = 6)
ggsave(file.path(fig_dir, "sim_dstudy_misspecification.png"), p2, width = 9, height = 6, dpi = 300)
cat("Saved: sim_dstudy_misspecification.pdf/png\n")

# =========================================================================
# Convergence and singularity diagnostics
# =========================================================================
cat("\n--- Convergence / Singularity Diagnostics ---\n")

diag_04f <- sc1_df %>%
  summarize(
    sc1_n_sims = n_distinct(sim_id),
    sc1_convergence_rate = n_distinct(sim_id) / N_SIM
  )

cat("\nSc1 convergence:\n")
print(diag_04f)

diag_04f_all <- tibble(
  scenario = c("Sc1", "Sc2", "Sc3", "Sc4", "Sc5"),
  n_sims = c(
    n_distinct(sc1_df$sim_id),
    n_distinct(sc2_df$sim_id),
    n_distinct(sc3_df$sim_id),
    n_distinct(sc4_df$sim_id),
    n_distinct(sc5_df$sim_id)
  ),
  convergence_rate = n_sims / N_SIM
)

cat("\nAll scenario convergence:\n")
print(diag_04f_all)

write_csv(diag_04f_all, file.path(output_dir, "sim_diagnostics_04f.csv"))
cat("Saved: sim_diagnostics_04f.csv\n")

cat("\n=== 04f_sim_dstudy_validation.R complete ===\n")
