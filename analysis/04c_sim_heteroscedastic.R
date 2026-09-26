# 04c_sim_heteroscedastic.R
# Monte Carlo simulation: heteroscedastic recovery.
# Tests whether (a) per-temperature sigma2_eps_k can be recovered, and
# (b) homoscedastic misspecification biases other components.

library(tidyverse)
library(lme4)
library(glmmTMB)

set.seed(42)

N_SIM <- 1000

# --- True DGP parameters (heteroscedastic residuals) ---
TRUE_PARAMS <- list(
  mu = 0.5,
  sigma2_category = 0.03,
  sigma2_item = 0.05,
  sigma2_prompt = 0.04,
  sigma2_ip = 0.02,
  sigma2_it = 0.015,
  sigma2_pt = 0.005,
  tau = c(0, -0.03, 0.05),
  n_categories = 5,
  sigma2_gen_k = c(0.005, 0.04, 0.08)  # Per-temperature residuals
)

# Pooled (weighted) average for reference
sigma2_gen_pooled <- mean(TRUE_PARAMS$sigma2_gen_k)

# Design
n_items <- 50
n_prompts <- 4
n_temps <- 3
n_reps <- 5
n_cats <- TRUE_PARAMS$n_categories

cat("=== 04c_sim_heteroscedastic.R: Heteroscedastic Recovery ===\n")
cat(sprintf("Design: N=%d, C=%d, K=%d, L=%d, R=%d, N_SIM=%d\n",
            n_items, n_cats, n_prompts, n_temps, n_reps, N_SIM))
cat(sprintf("True sigma2_gen_k: %s\n", paste(TRUE_PARAMS$sigma2_gen_k, collapse = ", ")))
cat(sprintf("Pooled average: %.4f\n", sigma2_gen_pooled))

# --- Simulation function with heteroscedastic residuals ---
simulate_data_het <- function(n_items, n_prompts, n_temps, n_reps, params) {
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
            rnorm(1, 0, sqrt(params$sigma2_gen_k[k]))
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

# --- True component map (for non-residual components) ---
true_map <- c(
  "category" = TRUE_PARAMS$sigma2_category,
  "item_id" = TRUE_PARAMS$sigma2_item,
  "variant_id" = TRUE_PARAMS$sigma2_prompt,
  "item_id:variant_id" = TRUE_PARAMS$sigma2_ip,
  "item_id:temperature" = TRUE_PARAMS$sigma2_it,
  "variant_id:temperature" = TRUE_PARAMS$sigma2_pt
)

# =========================================================================
# Main simulation loop
# =========================================================================
results_homo <- list()
results_glmmtmb <- list()
results_strat <- list()

for (sim in seq_len(N_SIM)) {
  if (sim %% 20 == 0) cat(sprintf("  Sim %d/%d\n", sim, N_SIM))
  set.seed(42 + sim)
  df <- simulate_data_het(n_items, n_prompts, n_temps, n_reps, TRUE_PARAMS)

  # --- Model 1: Homoscedastic lmer ---
  mod_homo <- tryCatch(
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

  if (!is.null(mod_homo)) {
    vc_homo <- as.data.frame(VarCorr(mod_homo))
    for (row_idx in seq_len(nrow(vc_homo))) {
      results_homo[[length(results_homo) + 1]] <- list(
        model_type = "homoscedastic_lmer",
        component = vc_homo$grp[row_idx],
        est_value = vc_homo$vcov[row_idx],
        sim_id = sim
      )
    }
    # Store the single pooled residual for each temperature slot
    sigma2_pooled <- sigma(mod_homo)^2
    for (k in seq_len(n_temps)) {
      results_homo[[length(results_homo) + 1]] <- list(
        model_type = "homoscedastic_lmer",
        component = paste0("sigma2_eps_t_", k),
        est_value = sigma2_pooled,
        sim_id = sim
      )
    }
  }

  # --- Model 2: Heteroscedastic glmmTMB ---
  mod_tmb <- tryCatch(
    glmmTMB(
      outcome ~ temperature +
        (1 | category) + (1 | item_id) + (1 | variant_id) +
        (1 | item_id:variant_id) + (1 | item_id:temperature) +
        (1 | variant_id:temperature),
      dispformula = ~temperature,
      data = df, REML = TRUE
    ),
    error = function(e) NULL
  )

  if (!is.null(mod_tmb)) {
    # glmmTMB VarCorr returns a list with $cond component; extract manually
    vc_tmb_raw <- VarCorr(mod_tmb)$cond
    for (grp_name in names(vc_tmb_raw)) {
      vc_val <- as.numeric(vc_tmb_raw[[grp_name]])  # 1x1 matrix -> scalar
      results_glmmtmb[[length(results_glmmtmb) + 1]] <- list(
        model_type = "heteroscedastic_glmmTMB",
        component = grp_name,
        est_value = vc_val,
        sim_id = sim
      )
    }
    # Extract per-temperature residual variances from dispersion model
    disp_coefs <- fixef(mod_tmb)$disp
    # disp_coefs: (Intercept) = log(sigma2) for reference level; temperaturet_2, temperaturet_3 are offsets
    log_sigma2_base <- disp_coefs[1]
    for (k in seq_len(n_temps)) {
      if (k == 1) {
        log_sigma2_k <- log_sigma2_base
      } else {
        offset_name <- paste0("temperaturet_", k)
        log_sigma2_k <- log_sigma2_base + disp_coefs[offset_name]
      }
      results_glmmtmb[[length(results_glmmtmb) + 1]] <- list(
        model_type = "heteroscedastic_glmmTMB",
        component = paste0("sigma2_eps_t_", k),
        est_value = exp(log_sigma2_k),
        sim_id = sim
      )
    }
  }

  # --- Model 3: Temperature-stratified lmer ---
  for (k in seq_len(n_temps)) {
    df_k <- df %>% filter(temperature == paste0("t_", k))
    mod_k <- tryCatch(
      lmer(
        outcome ~ (1 | category) + (1 | item_id) + (1 | variant_id) +
          (1 | item_id:variant_id),
        data = df_k, REML = TRUE,
        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10000))
      ),
      error = function(e) NULL
    )

    if (!is.null(mod_k)) {
      vc_k <- as.data.frame(VarCorr(mod_k))
      for (row_idx in seq_len(nrow(vc_k))) {
        results_strat[[length(results_strat) + 1]] <- list(
          model_type = "stratified_lmer",
          component = paste0(vc_k$grp[row_idx], "_t_", k),
          est_value = vc_k$vcov[row_idx],
          sim_id = sim
        )
      }
      results_strat[[length(results_strat) + 1]] <- list(
        model_type = "stratified_lmer",
        component = paste0("sigma2_eps_t_", k),
        est_value = sigma(mod_k)^2,
        sim_id = sim
      )
    }
  }
}

# =========================================================================
# Aggregate: Per-temperature residual recovery
# =========================================================================
cat("\n=== Per-Temperature Residual Recovery ===\n")

all_results <- bind_rows(
  bind_rows(results_homo),
  bind_rows(results_glmmtmb),
  bind_rows(results_strat)
)

# Focus on sigma2_eps_t_k
eps_results <- all_results %>%
  filter(grepl("^sigma2_eps_t_", component)) %>%
  mutate(
    temp_idx = as.integer(str_extract(component, "\\d+$")),
    true_value = TRUE_PARAMS$sigma2_gen_k[temp_idx]
  )

eps_summary <- eps_results %>%
  group_by(model_type, component, temp_idx) %>%
  summarize(
    true_value = first(true_value),
    mean_est = mean(est_value, na.rm = TRUE),
    bias = mean(est_value - true_value, na.rm = TRUE),
    rmse = sqrt(mean((est_value - true_value)^2, na.rm = TRUE)),
    n_sims = n(),
    .groups = "drop"
  )

cat("\nPer-temperature residual estimates:\n")
print(eps_summary %>% select(model_type, component, true_value, mean_est, bias, rmse))

# =========================================================================
# Aggregate: Non-residual component bias
# =========================================================================
cat("\n=== Non-Residual Component Bias ===\n")

nonres_results <- all_results %>%
  filter(component %in% names(true_map)) %>%
  mutate(true_value = true_map[component])

nonres_summary <- nonres_results %>%
  group_by(model_type, component) %>%
  summarize(
    true_value = first(true_value),
    mean_est = mean(est_value, na.rm = TRUE),
    bias = mean(est_value - true_value, na.rm = TRUE),
    rel_bias_pct = 100 * mean((est_value - true_value) / true_value, na.rm = TRUE),
    rmse = sqrt(mean((est_value - true_value)^2, na.rm = TRUE)),
    n_sims = n(),
    .groups = "drop"
  )

cat("\nNon-residual component bias:\n")
print(nonres_summary %>% select(model_type, component, true_value, mean_est, rel_bias_pct))

# =========================================================================
# D-study projection error
# =========================================================================
cat("\n=== D-Study Projection Error ===\n")

# Compare "fix temperature at t_1 (k=0)" recommendation
# Truth: sigma2_eps_1 = 0.005 (very low), so fixing at low temp is huge win
# Homoscedastic: uses pooled ~0.042, underestimates the benefit

dstudy_results <- list()

for (sim in seq_len(N_SIM)) {
  # Extract homoscedastic pooled residual for this sim
  homo_pooled <- all_results %>%
    filter(model_type == "homoscedastic_lmer",
           component == "sigma2_eps_t_1",
           sim_id == sim) %>%
    pull(est_value)
  if (length(homo_pooled) == 0) next

  # Extract glmmTMB per-k residuals
  tmb_eps <- all_results %>%
    filter(model_type == "heteroscedastic_glmmTMB",
           grepl("^sigma2_eps_t_", component),
           sim_id == sim)

  for (k in seq_len(n_temps)) {
    # True variance when fixing at temperature k
    true_fix_var <- TRUE_PARAMS$sigma2_gen_k[k] / (n_items * n_prompts * n_reps)
    # Add non-residual components
    true_fix_total <- TRUE_PARAMS$sigma2_category / n_cats +
      TRUE_PARAMS$sigma2_item / n_items +
      TRUE_PARAMS$sigma2_prompt / n_prompts +
      TRUE_PARAMS$sigma2_ip / (n_items * n_prompts) +
      TRUE_PARAMS$sigma2_it / n_items +
      TRUE_PARAMS$sigma2_pt / n_prompts +
      true_fix_var

    # Homoscedastic prediction (uses pooled sigma2 for all k)
    homo_fix_total <- TRUE_PARAMS$sigma2_category / n_cats +
      TRUE_PARAMS$sigma2_item / n_items +
      TRUE_PARAMS$sigma2_prompt / n_prompts +
      TRUE_PARAMS$sigma2_ip / (n_items * n_prompts) +
      TRUE_PARAMS$sigma2_it / n_items +
      TRUE_PARAMS$sigma2_pt / n_prompts +
      homo_pooled / (n_items * n_prompts * n_reps)

    dstudy_results[[length(dstudy_results) + 1]] <- list(
      model_type = "homoscedastic_lmer",
      fix_temp_k = k,
      predicted_var = homo_fix_total,
      actual_var = true_fix_total,
      sim_id = sim
    )

    # glmmTMB prediction
    tmb_eps_k <- tmb_eps %>% filter(component == paste0("sigma2_eps_t_", k)) %>% pull(est_value)
    if (length(tmb_eps_k) > 0) {
      tmb_fix_total <- TRUE_PARAMS$sigma2_category / n_cats +
        TRUE_PARAMS$sigma2_item / n_items +
        TRUE_PARAMS$sigma2_prompt / n_prompts +
        TRUE_PARAMS$sigma2_ip / (n_items * n_prompts) +
        TRUE_PARAMS$sigma2_it / n_items +
        TRUE_PARAMS$sigma2_pt / n_prompts +
        tmb_eps_k / (n_items * n_prompts * n_reps)

      dstudy_results[[length(dstudy_results) + 1]] <- list(
        model_type = "heteroscedastic_glmmTMB",
        fix_temp_k = k,
        predicted_var = tmb_fix_total,
        actual_var = true_fix_total,
        sim_id = sim
      )
    }
  }
}

dstudy_df <- bind_rows(dstudy_results)

dstudy_summary <- dstudy_df %>%
  group_by(model_type, fix_temp_k) %>%
  summarize(
    mean_predicted_var = mean(predicted_var, na.rm = TRUE),
    mean_actual_var = mean(actual_var, na.rm = TRUE),
    mean_prediction_error = mean(predicted_var - actual_var, na.rm = TRUE),
    mean_rel_error_pct = 100 * mean((predicted_var - actual_var) / actual_var, na.rm = TRUE),
    n_sims = n(),
    .groups = "drop"
  )

cat("\nD-study prediction error (fix temp at k):\n")
print(dstudy_summary)

# =========================================================================
# Save outputs
# =========================================================================
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Combine component-level results
full_summary <- bind_rows(
  eps_summary %>% rename(rel_bias_pct = bias) %>%
    mutate(rel_bias_pct = 100 * rel_bias_pct / true_value) %>%
    select(model_type, component, true_value, mean_est, bias = rel_bias_pct, rmse),
  nonres_summary %>% select(model_type, component, true_value, mean_est, bias = rel_bias_pct, rmse)
)

write_csv(full_summary, file.path(output_dir, "sim_heteroscedastic.csv"))
cat("\nSaved:", file.path(output_dir, "sim_heteroscedastic.csv"), "\n")

write_csv(dstudy_summary, file.path(output_dir, "sim_het_dstudy.csv"))
cat("Saved:", file.path(output_dir, "sim_het_dstudy.csv"), "\n")

# =========================================================================
# Figures
# =========================================================================
fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

temp_labels <- c("1" = "t=0 (greedy)", "2" = "t=0.7", "3" = "t=1.0")

# Figure 1: Per-temperature residual recovery
p_recovery <- eps_summary %>%
  mutate(
    temp_label = temp_labels[as.character(temp_idx)],
    model_label = case_when(
      model_type == "homoscedastic_lmer" ~ "Homoscedastic lmer",
      model_type == "heteroscedastic_glmmTMB" ~ "Heteroscedastic glmmTMB",
      model_type == "stratified_lmer" ~ "Stratified lmer"
    )
  ) %>%
  ggplot(aes(x = temp_label)) +
  geom_col(aes(y = mean_est, fill = model_label),
           position = position_dodge(width = 0.7), width = 0.6) +
  geom_point(aes(y = true_value), shape = 4, size = 3, stroke = 1.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = expression("Recovery of Per-Temperature" ~ sigma[epsilon * ",k"]^2),
    subtitle = sprintf("X marks = true values (N_SIM=%d)", N_SIM),
    x = "Temperature level",
    y = expression(hat(sigma)[epsilon * ",k"]^2),
    fill = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "sim_het_recovery.pdf"), p_recovery, width = 8, height = 5)
ggsave(file.path(fig_dir, "sim_het_recovery.png"), p_recovery, width = 8, height = 5, dpi = 300)
cat("Saved: sim_het_recovery.pdf/png\n")

# Figure 2: D-study prediction error
p_dstudy_err <- dstudy_summary %>%
  mutate(
    temp_label = temp_labels[as.character(fix_temp_k)],
    model_label = case_when(
      model_type == "homoscedastic_lmer" ~ "Homoscedastic lmer",
      model_type == "heteroscedastic_glmmTMB" ~ "Heteroscedastic glmmTMB"
    )
  ) %>%
  ggplot(aes(x = temp_label, y = mean_rel_error_pct, fill = model_label)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "D-Study Prediction Error: Fix Temperature Scenario",
    subtitle = sprintf("Relative error in predicted total variance (N_SIM=%d)", N_SIM),
    x = "Temperature fixed at",
    y = "Relative prediction error (%)",
    fill = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "sim_het_dstudy_error.pdf"), p_dstudy_err, width = 8, height = 5)
ggsave(file.path(fig_dir, "sim_het_dstudy_error.png"), p_dstudy_err, width = 8, height = 5, dpi = 300)
cat("Saved: sim_het_dstudy_error.pdf/png\n")

# =========================================================================
# Convergence and singularity diagnostics
# =========================================================================
cat("\n--- Convergence / Singularity Diagnostics ---\n")

diag_04c <- all_results %>%
  group_by(model_type) %>%
  summarize(
    n_sims = n_distinct(sim_id),
    convergence_rate = n_sims / N_SIM,
    .groups = "drop"
  )

comp_stats_04c <- all_results %>%
  group_by(model_type, component) %>%
  summarize(
    mean_est = mean(est_value, na.rm = TRUE),
    sd_est = sd(est_value, na.rm = TRUE),
    pct_boundary = 100 * mean(est_value < 1e-10, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nConvergence summary:\n")
print(diag_04c)

write_csv(diag_04c, file.path(output_dir, "sim_diagnostics_04c.csv"))
write_csv(comp_stats_04c, file.path(output_dir, "sim_component_stats_04c.csv"))
cat("Saved: sim_diagnostics_04c.csv, sim_component_stats_04c.csv\n")

cat("\n=== 04c_sim_heteroscedastic.R complete ===\n")
