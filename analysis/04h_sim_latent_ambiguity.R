# 04h_sim_latent_ambiguity.R
# Monte Carlo simulation: latent item ambiguity robustness.
#
# Tests whether TEE variance decomposition and D-study projections are robust
# when a latent ambiguity trait z_i simultaneously induces:
#   - correlated random effects (ambiguous items have larger item x prompt
#     and item x judge interactions)
#   - sparse 3-way interactions (only ambiguous items show item x prompt x judge)
#
# Sweeps gamma in {0, 0.5, 1, 2} where gamma controls ambiguity strength.
# Evaluates: (a) CI coverage, (b) component recovery, (c) D-study ranking.
#
# Usage:
#   Rscript analysis/04h_sim_latent_ambiguity.R              # full (~45 min)
#   Rscript analysis/04h_sim_latent_ambiguity.R --nsim 5     # smoke (~2 min)

library(tidyverse)
library(lme4)

set.seed(42)

N_SIM <- 1000

args <- commandArgs(trailingOnly = TRUE)
if ("--nsim" %in% args) {
  idx <- which(args == "--nsim")
  N_SIM <- as.integer(args[idx + 1])
  cat(sprintf("Using N_SIM = %d (from --nsim flag)\n", N_SIM))
}

pop_var <- function(x) mean((x - mean(x))^2)

# --- DGP parameters (extended with judge terms, matching 04g) ---
TRUE_PARAMS <- list(
  mu = 0.5,
  sigma2_category = 0.03,
  sigma2_item = 0.05,
  sigma2_prompt = 0.04,
  sigma2_ip = 0.02,
  sigma2_it = 0.015,
  sigma2_pt = 0.005,
  sigma2_im = 0.025,
  sigma2_pm = 0.005,
  sigma2_gen = 0.06,
  tau = c(0, -0.03),
  lambda = c(0, 0.03, -0.02),
  n_categories = 5
)

# Design (kept small for tractable lmer: 30*3*2*3*3 = 1620 rows/fit)
N_ITEMS  <- 30
N_CATS   <- TRUE_PARAMS$n_categories
N_PROMPTS <- 3
N_TEMPS  <- 2
N_JUDGES <- 3
N_REPS   <- 3

# Ambiguity sweep
GAMMA_LEVELS <- c(0, 0.5, 1, 2)
GAMMA_LABELS <- c("None (0)", "Mild (0.5)", "Moderate (1)", "Strong (2)")

# 3-way interaction base variance (absorbed into residual in the 2-way model)
SIGMA2_3WAY_BASE <- 0.01

cat("=== 04h_sim_latent_ambiguity.R: Latent Ambiguity Robustness ===\n")
cat(sprintf("Design: N=%d, C=%d, V=%d, H=%d, M=%d, R=%d, N_SIM=%d\n",
            N_ITEMS, N_CATS, N_PROMPTS, N_TEMPS, N_JUDGES, N_REPS, N_SIM))
cat(sprintf("Gamma levels: %s\n", paste(GAMMA_LEVELS, collapse = ", ")))

# =========================================================================
# 1. DGP: latent ambiguity
# =========================================================================
simulate_data_ambiguity <- function(n_items, n_prompts, n_temps, n_judges,
                                     n_reps, params, gamma, sigma2_3way_base) {
  n_cats <- params$n_categories
  items_per_cat <- ceiling(n_items / n_cats)

  # Latent ambiguity per item
  z <- rnorm(n_items)
  ambiguity_scale <- 1 + gamma * z^2
  is_ambiguous <- z > 1  # top ~16%

  # Random effects
  gamma_cat <- rnorm(n_cats, 0, sqrt(params$sigma2_category))
  item_cat <- rep(seq_len(n_cats), each = items_per_cat)[seq_len(n_items)]
  delta_item <- rnorm(n_items, 0, sqrt(params$sigma2_item))
  alpha <- gamma_cat[item_cat] + delta_item

  beta <- rnorm(n_prompts, 0, sqrt(params$sigma2_prompt))

  # Ambiguity-scaled interactions
  ip <- matrix(0, n_items, n_prompts)
  im <- matrix(0, n_items, n_judges)
  for (i in seq_len(n_items)) {
    ip[i, ] <- rnorm(n_prompts, 0, sqrt(params$sigma2_ip * ambiguity_scale[i]))
    im[i, ] <- rnorm(n_judges, 0, sqrt(params$sigma2_im * ambiguity_scale[i]))
  }

  it <- matrix(rnorm(n_items * n_temps, 0, sqrt(params$sigma2_it)),
               nrow = n_items, ncol = n_temps)
  pt <- matrix(rnorm(n_prompts * n_temps, 0, sqrt(params$sigma2_pt)),
               nrow = n_prompts, ncol = n_temps)
  pm <- matrix(rnorm(n_prompts * n_judges, 0, sqrt(params$sigma2_pm)),
               nrow = n_prompts, ncol = n_judges)

  # Sparse 3-way interaction: item x prompt x judge, only for ambiguous items
  ipj <- array(0, dim = c(n_items, n_prompts, n_judges))
  for (i in seq_len(n_items)) {
    if (is_ambiguous[i]) {
      ipj[i, , ] <- rnorm(n_prompts * n_judges, 0, sqrt(sigma2_3way_base))
    }
  }

  # Generate observations
  total <- n_items * n_prompts * n_temps * n_judges * n_reps
  rows <- vector("list", total)
  idx <- 0L
  for (i in seq_len(n_items)) {
    for (v in seq_len(n_prompts)) {
      for (h in seq_len(n_temps)) {
        for (m in seq_len(n_judges)) {
          for (r in seq_len(n_reps)) {
            idx <- idx + 1L
            y <- params$mu + alpha[i] + beta[v] + params$tau[h] + params$lambda[m] +
              ip[i, v] + it[i, h] + pt[v, h] + im[i, m] + pm[v, m] +
              ipj[i, v, m] +
              rnorm(1, 0, sqrt(params$sigma2_gen))
            rows[[idx]] <- list(
              item_id = paste0("item_", i),
              category = paste0("cat_", item_cat[i]),
              variant_id = paste0("v_", v),
              temperature = paste0("t_", h),
              judge_id = paste0("j_", m),
              replication = r,
              outcome = y
            )
          }
        }
      }
    }
  }
  bind_rows(rows) %>%
    mutate(
      item_id = as.factor(item_id),
      category = as.factor(category),
      variant_id = as.factor(variant_id),
      temperature = as.factor(temperature),
      judge_id = as.factor(judge_id)
    )
}

# =========================================================================
# 2. Fit model and extract variance components
# =========================================================================
fit_and_extract <- function(df) {
  mod <- tryCatch(
    lmer(
      outcome ~ temperature + judge_id +
        (1 | category) + (1 | item_id) + (1 | variant_id) +
        (1 | item_id:variant_id) + (1 | item_id:temperature) +
        (1 | variant_id:temperature) +
        (1 | item_id:judge_id) + (1 | variant_id:judge_id),
      data = df, REML = TRUE,
      control = lmerControl(optimizer = "bobyqa",
                            optCtrl = list(maxfun = 20000),
                            calc.derivs = FALSE)
    ),
    error = function(e) NULL
  )
  if (is.null(mod)) return(NULL)

  vc <- as.data.frame(VarCorr(mod))
  tibble(component = vc$grp, variance = vc$vcov)
}

# =========================================================================
# 3. D-study projection (multi-judge, fixed temperature)
# =========================================================================
compute_dstudy_var <- function(vc, n_items_d, n_prompts_d, n_reps_d,
                                n_judges_d, n_temps_d = 1) {
  s2 <- setNames(vc$variance, vc$component)
  g <- function(nm) ifelse(is.na(s2[nm]), 0, s2[nm])

  N <- n_items_d; V <- n_prompts_d; R <- n_reps_d; M <- n_judges_d

  g("category") / N_CATS +
    g("item_id") / N +
    g("variant_id") / V +
    g("item_id:variant_id") / (N * V) +
    g("item_id:temperature") / N +
    g("variant_id:temperature") / V +
    g("item_id:judge_id") / (N * M) +
    g("variant_id:judge_id") / (V * M) +
    g("Residual") / (N * V * M * R)
}

# D-study interventions (same as 04f pattern)
DSTUDY_INTERVENTIONS <- list(
  baseline = list(N = N_ITEMS, V = N_PROMPTS, M = N_JUDGES, R = N_REPS),
  double_items = list(N = N_ITEMS * 2, V = N_PROMPTS, M = N_JUDGES, R = N_REPS),
  double_prompts = list(N = N_ITEMS, V = N_PROMPTS * 2, M = N_JUDGES, R = N_REPS),
  double_judges = list(N = N_ITEMS, V = N_PROMPTS, M = N_JUDGES * 2, R = N_REPS),
  double_reps = list(N = N_ITEMS, V = N_PROMPTS, M = N_JUDGES, R = N_REPS * 2),
  single_judge = list(N = N_ITEMS, V = N_PROMPTS, M = 1, R = N_REPS)
)

compute_intervention_ranks <- function(vc) {
  vars <- sapply(DSTUDY_INTERVENTIONS, function(d) {
    compute_dstudy_var(vc, d$N, d$V, d$R, d$M)
  })
  # Return % change from baseline
  baseline <- vars["baseline"]
  pct_change <- (vars - baseline) / baseline * 100
  pct_change
}

# True marginal variance components (integrating over z_i)
# E[1 + gamma * z^2] = 1 + gamma (since z ~ N(0,1), E[z^2] = 1)
# P(z > 1) ≈ 0.159
compute_true_marginal <- function(params, gamma, sigma2_3way_base) {
  p_ambig <- pnorm(1, lower.tail = FALSE)
  tibble(
    component = c("category", "item_id", "variant_id",
                   "item_id:variant_id", "item_id:temperature",
                   "variant_id:temperature",
                   "item_id:judge_id", "variant_id:judge_id",
                   "Residual"),
    variance = c(
      params$sigma2_category,
      params$sigma2_item,
      params$sigma2_prompt,
      params$sigma2_ip * (1 + gamma),       # marginal over z
      params$sigma2_it,
      params$sigma2_pt,
      params$sigma2_im * (1 + gamma),       # marginal over z
      params$sigma2_pm,
      params$sigma2_gen + p_ambig * sigma2_3way_base  # 3-way absorbed into residual
    )
  )
}

# =========================================================================
# 4. Main simulation loop
# =========================================================================
cat("\nRunning simulation...\n")

results <- list()
sim_idx <- 0L

for (g_idx in seq_along(GAMMA_LEVELS)) {
  gamma <- GAMMA_LEVELS[g_idx]
  gamma_lab <- GAMMA_LABELS[g_idx]
  cat(sprintf("\n--- Gamma = %.1f (%s) ---\n", gamma, gamma_lab))

  true_vc <- compute_true_marginal(TRUE_PARAMS, gamma, SIGMA2_3WAY_BASE)
  true_interventions <- compute_intervention_ranks(true_vc)
  true_dstudy_baseline <- compute_dstudy_var(true_vc, N_ITEMS, N_PROMPTS, N_REPS, N_JUDGES)

  for (sim in seq_len(N_SIM)) {
    if (sim %% 100 == 0) cat(sprintf("  sim %d/%d\n", sim, N_SIM))

    df <- simulate_data_ambiguity(N_ITEMS, N_PROMPTS, N_TEMPS, N_JUDGES,
                                   N_REPS, TRUE_PARAMS, gamma, SIGMA2_3WAY_BASE)

    vc <- fit_and_extract(df)
    if (is.null(vc)) next

    # (a) CI coverage: does the D-study SE cover the true theta?
    theta_hat <- mean(df$outcome)
    dstudy_var <- compute_dstudy_var(vc, N_ITEMS, N_PROMPTS, N_REPS, N_JUDGES)
    dstudy_se <- sqrt(max(dstudy_var, 1e-12))
    ci_lo <- theta_hat - 1.96 * dstudy_se
    ci_hi <- theta_hat + 1.96 * dstudy_se
    true_theta <- TRUE_PARAMS$mu  # E[Y] = mu (tau and lambda average to ~0)
    covers <- (ci_lo <= true_theta) & (true_theta <= ci_hi)

    # (b) Component recovery: bias vs true marginal
    recovery <- vc %>%
      left_join(true_vc, by = "component", suffix = c("_est", "_true")) %>%
      mutate(bias = variance_est - variance_true,
             rel_bias = ifelse(variance_true > 1e-6, bias / variance_true, NA))

    # (c) D-study intervention ranking
    est_interventions <- compute_intervention_ranks(vc)

    sim_idx <- sim_idx + 1L
    results[[sim_idx]] <- list(
      gamma = gamma,
      gamma_label = gamma_lab,
      sim = sim,
      covers = covers,
      theta_hat = theta_hat,
      dstudy_se = dstudy_se,
      recovery = recovery,
      est_interventions = est_interventions,
      true_interventions = true_interventions
    )
  }
}

cat("\nSimulation complete. Processing results...\n")

# =========================================================================
# 5. Aggregate results
# =========================================================================

# (a) CI coverage by gamma
coverage_df <- map_dfr(results, function(r) {
  tibble(gamma = r$gamma, gamma_label = r$gamma_label,
         sim = r$sim, covers = r$covers,
         theta_hat = r$theta_hat, dstudy_se = r$dstudy_se)
})

coverage_summary <- coverage_df %>%
  group_by(gamma, gamma_label) %>%
  summarize(
    coverage = mean(covers),
    mean_se = mean(dstudy_se),
    n_sims = n(),
    .groups = "drop"
  )

cat("\n=== CI Coverage by Gamma ===\n")
print(coverage_summary)

# (b) Component recovery by gamma
recovery_df <- map_dfr(results, function(r) {
  r$recovery %>%
    mutate(gamma = r$gamma, gamma_label = r$gamma_label, sim = r$sim)
})

recovery_summary <- recovery_df %>%
  filter(!is.na(rel_bias)) %>%
  group_by(gamma, gamma_label, component) %>%
  summarize(
    mean_rel_bias = mean(rel_bias) * 100,
    rmse = sqrt(mean(bias^2)),
    n_sims = n(),
    .groups = "drop"
  )

cat("\n=== Component Recovery (% relative bias) ===\n")
recovery_wide <- recovery_summary %>%
  select(gamma, component, mean_rel_bias) %>%
  pivot_wider(names_from = gamma, values_from = mean_rel_bias)
print(recovery_wide, n = 20)

# Max absolute relative bias per gamma (only components with true variance > 0.01)
substantive_components <- c("category", "item_id", "variant_id",
                             "item_id:variant_id", "item_id:judge_id", "Residual")
max_bias <- recovery_summary %>%
  filter(component %in% substantive_components) %>%
  group_by(gamma, gamma_label) %>%
  summarize(max_abs_rel_bias = max(abs(mean_rel_bias)), .groups = "drop")
cat("\n=== Max |Relative Bias| by Gamma (substantive components) ===\n")
print(max_bias)

# (c) D-study intervention ranking: Kendall's tau
dstudy_df <- map_dfr(results, function(r) {
  tibble(
    gamma = r$gamma, gamma_label = r$gamma_label, sim = r$sim,
    intervention = names(r$est_interventions),
    est_pct = as.numeric(r$est_interventions),
    true_pct = as.numeric(r$true_interventions)
  )
})

# Use rank correlation on the non-baseline interventions only
# (baseline is always 0% change, creating a trivial tie)
tau_summary <- dstudy_df %>%
  filter(intervention != "baseline") %>%
  group_by(gamma, gamma_label, sim) %>%
  summarize(
    tau = cor(rank(est_pct), rank(true_pct), method = "spearman"),
    .groups = "drop"
  ) %>%
  group_by(gamma, gamma_label) %>%
  summarize(
    mean_tau = mean(tau, na.rm = TRUE),
    sd_tau = sd(tau, na.rm = TRUE),
    n_sims = n(),
    .groups = "drop"
  )

cat("\n=== D-Study Rank Concordance (Kendall's tau) ===\n")
print(tau_summary)

# =========================================================================
# 6. Save results
# =========================================================================
out_dir <- "data/processed"

write_csv(coverage_summary, file.path(out_dir, "sim_latent_ambiguity_coverage.csv"))
write_csv(recovery_summary, file.path(out_dir, "sim_latent_ambiguity_recovery.csv"))
write_csv(tau_summary, file.path(out_dir, "sim_latent_ambiguity_dstudy.csv"))

# Combined summary table for SI
si_table <- coverage_summary %>%
  select(gamma, gamma_label, coverage) %>%
  left_join(max_bias %>% select(gamma, max_abs_rel_bias), by = "gamma") %>%
  left_join(tau_summary %>% select(gamma, mean_tau), by = "gamma")

write_csv(si_table, file.path(out_dir, "sim_latent_ambiguity_si_table.csv"))

cat("\n=== SI Summary Table ===\n")
print(si_table)

cat(sprintf("\nSaved to %s/sim_latent_ambiguity_*.csv\n", out_dir))

# =========================================================================
# 7. Figures
# =========================================================================
fig_dir <- "figures"

# Panel (a): CI coverage vs gamma
p_cov <- ggplot(coverage_summary, aes(x = gamma, y = coverage * 100)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "grey50") +
  scale_y_continuous(limits = c(80, 100)) +
  labs(x = expression(gamma ~ "(ambiguity strength)"),
       y = "95% CI coverage (%)",
       subtitle = "(a) CI coverage") +
  theme_minimal(base_size = 12)

# Panel (b): Component relative bias vs gamma
key_components <- c("item_id:variant_id", "item_id:judge_id", "Residual",
                     "item_id", "category")
p_bias <- recovery_summary %>%
  filter(component %in% key_components) %>%
  mutate(component = factor(component, levels = key_components)) %>%
  ggplot(aes(x = gamma, y = mean_rel_bias, color = component)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = expression(gamma ~ "(ambiguity strength)"),
       y = "Relative bias (%)",
       subtitle = "(b) Component recovery",
       color = "Component") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", legend.text = element_text(size = 8))

# Panel (c): D-study Kendall's tau vs gamma
p_tau <- ggplot(tau_summary, aes(x = gamma, y = mean_tau)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_tau - sd_tau, ymax = mean_tau + sd_tau),
                width = 0.1) +
  scale_y_continuous(limits = c(0.5, 1)) +
  labs(x = expression(gamma ~ "(ambiguity strength)"),
       y = expression("Kendall's " * tau),
       subtitle = "(c) D-study intervention ranking") +
  theme_minimal(base_size = 12)

# Combine
library(patchwork)

p_combined <- p_cov + p_bias + p_tau +
  plot_layout(ncol = 3, widths = c(1, 1.3, 1)) +
  plot_annotation(
    title = "TEE robustness under latent item ambiguity",
    subtitle = sprintf(
      "Ambiguity z_i scales item×prompt and item×judge interactions; sparse 3-way for z > 1. N_sim = %d",
      N_SIM
    ),
    theme = theme(plot.title = element_text(size = 14, face = "bold"))
  )

ggsave(file.path(fig_dir, "sim_latent_ambiguity.pdf"), p_combined,
       width = 14, height = 5)
ggsave(file.path(fig_dir, "sim_latent_ambiguity.png"), p_combined,
       width = 14, height = 5, dpi = 150)

cat(sprintf("Figures saved to %s/sim_latent_ambiguity.{pdf,png}\n", fig_dir))
cat("\nDone.\n")
