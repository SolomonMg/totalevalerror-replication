# 04_simulation.R
# Monte Carlo simulation to study asymptotic properties of TEE variance
# component estimators under the DGP from the paper.
#
# Generates data from a known DGP with known variance components,
# fits the lme4 model, and evaluates:
#   1. Bias of REML estimators at varying sample sizes
#   2. RMSE convergence
#   3. CI coverage (nominal 95%)
#   4. Sensitivity to assumption violations (non-normality, imbalance)

library(tidyverse)
library(lme4)

set.seed(42)

# --- True DGP parameters ---
TRUE_PARAMS <- list(
  mu = 0.5,              # Grand mean (binary outcome centered)
  sigma2_category = 0.03, # Between-category variance
  sigma2_item = 0.05,    # Within-category item variance (was 0.08 total)
  sigma2_prompt = 0.04,  # Prompt variant variance
  sigma2_ip = 0.02,      # Item x prompt interaction
  sigma2_it = 0.015,     # Item x temperature interaction
  sigma2_pt = 0.005,     # Prompt x temperature interaction
  sigma2_gen = 0.06,     # Generation stochasticity (residual)
  tau = c(0, -0.03, 0.05), # Temperature fixed effects (3 levels)
  n_categories = 5       # Number of categories for nesting
)

# Population variance (denominator n, not n-1) — appropriate for fixed factor levels
pop_var <- function(x) mean((x - mean(x))^2)

cat("=== 04_simulation.R: Asymptotic Properties ===\n")
cat("\nTrue variance components:\n")
str(TRUE_PARAMS)


# --- Simulation function ---
simulate_data <- function(n_items, n_prompts, n_temps, n_reps, params) {
  n_cats <- params$n_categories
  items_per_cat <- ceiling(n_items / n_cats)

  # Generate category random effects
  gamma_cat <- rnorm(n_cats, 0, sqrt(params$sigma2_category))

  # Assign items to categories and generate within-category random effects
  item_cat <- rep(seq_len(n_cats), each = items_per_cat)[seq_len(n_items)]
  delta_item <- rnorm(n_items, 0, sqrt(params$sigma2_item))
  # Composite item effect: alpha_i = gamma_{c(i)} + delta_{i|c}
  alpha <- gamma_cat[item_cat] + delta_item

  beta <- rnorm(n_prompts, 0, sqrt(params$sigma2_prompt))

  # Interaction effects (use composite alpha for interactions)
  ab <- matrix(rnorm(n_items * n_prompts, 0, sqrt(params$sigma2_ip)),
               nrow = n_items, ncol = n_prompts)
  at <- matrix(rnorm(n_items * n_temps, 0, sqrt(params$sigma2_it)),
               nrow = n_items, ncol = n_temps)
  bt <- matrix(rnorm(n_prompts * n_temps, 0, sqrt(params$sigma2_pt)),
               nrow = n_prompts, ncol = n_temps)

  # Generate observations
  rows <- list()
  for (i in seq_len(n_items)) {
    for (j in seq_len(n_prompts)) {
      for (k in seq_len(n_temps)) {
        for (r in seq_len(n_reps)) {
          y <- params$mu + alpha[i] + beta[j] + params$tau[k] +
            ab[i, j] + at[i, k] + bt[j, k] +
            rnorm(1, 0, sqrt(params$sigma2_gen))
          rows[[length(rows) + 1]] <- list(
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


# --- Fit model and extract variance components ---
fit_and_extract <- function(df) {
  mod <- tryCatch(
    lmer(
      outcome ~ temperature +
        (1 | category) +
        (1 | item_id) + (1 | variant_id) +
        (1 | item_id:variant_id) +
        (1 | item_id:temperature) +
        (1 | variant_id:temperature),
      data = df, REML = TRUE,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10000))
    ),
    error = function(e) NULL
  )

  if (is.null(mod)) {
    return(tibble(
      component = c("category", "item", "prompt", "item:prompt",
                     "item:temp", "prompt:temp", "residual"),
      variance = rep(NA_real_, 7),
      converged = FALSE
    ))
  }

  vc <- as.data.frame(VarCorr(mod))
  singular <- isSingular(mod)

  # Temperature fixed effect sensitivity
  tc <- fixef(mod)
  temp_coefs <- tc[grepl("temperature", names(tc))]
  all_temp <- c(0, temp_coefs)
  sigma2_temp_hat <- pop_var(all_temp)

  tibble(
    component = c(vc$grp, "temp_fixed"),
    variance = c(vc$vcov, sigma2_temp_hat),
    converged = TRUE,
    singular = singular
  )
}


# =========================================================================
# Simulation 1: Bias and RMSE across sample sizes
# =========================================================================
cat("\n--- Simulation 1: Convergence across sample sizes ---\n")

sample_configs <- tribble(
  ~n_items, ~n_prompts, ~n_reps, ~label,
  10, 3, 3, "Small (10i, 3p, 3r)",
  20, 3, 5, "Pilot-like (20i, 3p, 5r)",
  30, 3, 5, "Pilot (30i, 3p, 5r)",
  50, 4, 5, "Medium (50i, 4p, 5r)",
  100, 4, 8, "Full (100i, 4p, 8r)",
  200, 5, 10, "Large (200i, 5p, 10r)",
)

N_SIM <- 1000  # Number of simulation replicates per config
n_temps <- 3

sim_results <- list()

for (cfg_idx in seq_len(nrow(sample_configs))) {
  cfg <- sample_configs[cfg_idx, ]
  cat(sprintf("  Config %d/%d: %s ...\n", cfg_idx, nrow(sample_configs), cfg$label))

  reps <- map_dfr(seq_len(N_SIM), function(sim) {
    set.seed(42 + sim)
    df <- simulate_data(cfg$n_items, cfg$n_prompts, n_temps, cfg$n_reps, TRUE_PARAMS)
    result <- fit_and_extract(df)
    result$sim_id <- sim
    result$config <- cfg$label
    result$n_items <- cfg$n_items
    result$n_prompts <- cfg$n_prompts
    result$n_reps <- cfg$n_reps
    result
  })

  sim_results[[cfg_idx]] <- reps
}

all_results <- bind_rows(sim_results)

# Map estimated components to true values
true_map <- c(
  "category" = TRUE_PARAMS$sigma2_category,
  "item_id" = TRUE_PARAMS$sigma2_item,
  "variant_id" = TRUE_PARAMS$sigma2_prompt,
  "item_id:variant_id" = TRUE_PARAMS$sigma2_ip,
  "item_id:temperature" = TRUE_PARAMS$sigma2_it,
  "variant_id:temperature" = TRUE_PARAMS$sigma2_pt,
  "Residual" = TRUE_PARAMS$sigma2_gen,
  "temp_fixed" = pop_var(TRUE_PARAMS$tau)
)

sim_summary <- all_results %>%
  filter(converged) %>%
  mutate(true_value = true_map[component]) %>%
  filter(!is.na(true_value)) %>%
  group_by(config, n_items, n_prompts, n_reps, component) %>%
  summarize(
    true_value = first(true_value),
    mean_est = mean(variance, na.rm = TRUE),
    bias = mean(variance - true_value, na.rm = TRUE),
    rel_bias_pct = 100 * mean((variance - true_value) / true_value, na.rm = TRUE),
    rmse = sqrt(mean((variance - true_value)^2, na.rm = TRUE)),
    convergence_rate = mean(!is.na(variance)),
    n_sims = n(),
    .groups = "drop"
  )

cat("\nBias and RMSE summary:\n")
print(sim_summary %>%
        select(config, n_items, component, true_value, mean_est, rel_bias_pct, rmse) %>%
        arrange(component, n_items))


# =========================================================================
# Simulation 2: CI Coverage
# =========================================================================
cat("\n--- Simulation 2: CI Coverage (profile likelihood) ---\n")
cat("  Using pilot-like config (30 items, 3 prompts, 5 reps), 100 sims\n")

N_SIM_CI <- 100
coverage_results <- list()

for (sim in seq_len(N_SIM_CI)) {
  if (sim %% 20 == 0) cat(sprintf("  Sim %d/%d\n", sim, N_SIM_CI))
  set.seed(42 + sim)
  df <- simulate_data(30, 3, n_temps, 5, TRUE_PARAMS)

  mod <- tryCatch(
    lmer(
      outcome ~ temperature +
        (1 | category) +
        (1 | item_id) + (1 | variant_id) +
        (1 | item_id:variant_id) +
        (1 | item_id:temperature) +
        (1 | variant_id:temperature),
      data = df, REML = TRUE,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10000))
    ),
    error = function(e) NULL
  )

  if (is.null(mod)) next

  ci <- tryCatch(
    confint(mod, method = "profile", oldNames = FALSE, quiet = TRUE),
    error = function(e) NULL
  )

  if (is.null(ci)) next

  ci_df <- as.data.frame(ci) %>%
    rownames_to_column("param") %>%
    rename(ci_lower = `2.5 %`, ci_upper = `97.5 %`)

  # Check coverage for SD parameters (square to get variance)
  sd_rows <- ci_df %>% filter(grepl("^sd_", param) | param == "sigma")

  # Map param names to component names
  param_map <- c(
    "sd_category.(Intercept)" = "category",
    "sd_item_id.(Intercept)" = "item_id",
    "sd_variant_id.(Intercept)" = "variant_id",
    "sd_item_id:variant_id.(Intercept)" = "item_id:variant_id",
    "sd_item_id:temperature.(Intercept)" = "item_id:temperature",
    "sd_variant_id:temperature.(Intercept)" = "variant_id:temperature",
    "sigma" = "Residual"
  )

  for (row_idx in seq_len(nrow(sd_rows))) {
    p <- sd_rows$param[row_idx]
    comp <- param_map[p]
    if (is.na(comp)) next
    true_sd <- sqrt(true_map[comp])
    covers <- sd_rows$ci_lower[row_idx] <= true_sd & true_sd <= sd_rows$ci_upper[row_idx]
    coverage_results[[length(coverage_results) + 1]] <- list(
      sim = sim, component = comp, covers = covers,
      ci_lower = sd_rows$ci_lower[row_idx], ci_upper = sd_rows$ci_upper[row_idx],
      true_sd = true_sd
    )
  }
}

if (length(coverage_results) > 0) {
  coverage_df <- bind_rows(coverage_results)
  coverage_summary <- coverage_df %>%
    group_by(component) %>%
    summarize(
      coverage = mean(covers, na.rm = TRUE),
      n_sims = n(),
      mean_width = mean(ci_upper - ci_lower, na.rm = TRUE),
      .groups = "drop"
    )

  cat("\n95% CI Coverage (nominal = 0.95):\n")
  print(coverage_summary)
} else {
  cat("  No successful CI computations\n")
}


# =========================================================================
# Save simulation results
# =========================================================================
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(sim_summary, file.path(output_dir, "sim_bias_rmse.csv"))
cat("\nSaved simulation results to data/processed/sim_bias_rmse.csv\n")

if (length(coverage_results) > 0) {
  write_csv(coverage_summary, file.path(output_dir, "sim_coverage.csv"))
  cat("Saved CI coverage results to data/processed/sim_coverage.csv\n")
}


# =========================================================================
# Simulation figures
# =========================================================================
cat("\n--- Generating simulation figures ---\n")
fig_dir <- "figures"

# Bias vs sample size
tle_labels <- c(
  "category" = "Between-category",
  "item_id" = "Within-category item",
  "variant_id" = "Prompt",
  "item_id:variant_id" = "Item x Prompt",
  "item_id:temperature" = "Item x Temp",
  "variant_id:temperature" = "Prompt x Temp",
  "Residual" = "Generation",
  "temp_fixed" = "Temperature (fixed)"
)

p_bias <- sim_summary %>%
  filter(component != "temp_fixed") %>%
  mutate(
    component_label = tle_labels[component],
    n_total = n_items * n_prompts * 3 * n_reps
  ) %>%
  ggplot(aes(x = n_total, y = rel_bias_pct, color = component_label)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_log10(labels = scales::comma) +
  labs(
    title = "REML Estimator Bias vs. Total Observations",
    subtitle = sprintf("Based on %d Monte Carlo replicates per configuration", N_SIM),
    x = "Total observations (log scale)",
    y = "Relative bias (%)",
    color = "Component"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(file.path(fig_dir, "sim_bias_convergence.pdf"), p_bias,
       width = 8, height = 5)
ggsave(file.path(fig_dir, "sim_bias_convergence.png"), p_bias,
       width = 8, height = 5, dpi = 300)
cat("  Saved sim_bias_convergence.pdf/png\n")

# RMSE vs sample size
p_rmse <- sim_summary %>%
  filter(component != "temp_fixed") %>%
  mutate(
    component_label = tle_labels[component],
    n_total = n_items * n_prompts * 3 * n_reps
  ) %>%
  ggplot(aes(x = n_total, y = rmse, color = component_label)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  scale_x_log10(labels = scales::comma) +
  scale_y_log10() +
  labs(
    title = "RMSE of Variance Component Estimators vs. Sample Size",
    subtitle = sprintf("Based on %d Monte Carlo replicates per configuration", N_SIM),
    x = "Total observations (log scale)",
    y = "RMSE (log scale)",
    color = "Component"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(file.path(fig_dir, "sim_rmse_convergence.pdf"), p_rmse,
       width = 8, height = 5)
ggsave(file.path(fig_dir, "sim_rmse_convergence.png"), p_rmse,
       width = 8, height = 5, dpi = 300)
cat("  Saved sim_rmse_convergence.pdf/png\n")

# =========================================================================
# Convergence and singularity diagnostics
# =========================================================================
cat("\n--- Convergence / Singularity Diagnostics ---\n")

diag_04 <- all_results %>%
  group_by(config, n_items) %>%
  summarize(
    n_sims = n_distinct(sim_id),
    n_converged = sum(converged, na.rm = TRUE) / n_distinct(component),
    convergence_rate = n_converged / n_sims,
    n_singular = sum(singular & converged, na.rm = TRUE) / n_distinct(component),
    singular_rate = n_singular / max(n_converged, 1),
    .groups = "drop"
  )

cat("\nConvergence summary:\n")
print(diag_04)

# Per-component summary stats
comp_stats_04 <- all_results %>%
  filter(converged) %>%
  group_by(config, component) %>%
  summarize(
    mean_var = mean(variance, na.rm = TRUE),
    sd_var = sd(variance, na.rm = TRUE),
    pct_boundary = 100 * mean(variance < 1e-10, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(diag_04, file.path(output_dir, "sim_diagnostics_04.csv"))
write_csv(comp_stats_04, file.path(output_dir, "sim_component_stats_04.csv"))
cat("Saved: sim_diagnostics_04.csv, sim_component_stats_04.csv\n")

cat("\n=== Simulation complete ===\n")
