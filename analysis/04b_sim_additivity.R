# 04b_sim_additivity.R
# Monte Carlo simulation: additivity violation robustness.
# Quantifies when the 2-way lmer breaks down under 3-way interaction violations.
# Adds (alpha*beta*tau)_{ijk} ~ N(0, sigma2_3way) to the DGP.

library(tidyverse)
library(lme4)

set.seed(42)

N_SIM <- 1000

# --- True DGP parameters (same as 04_simulation.R) ---
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

# --- 3-way interaction levels ---
sigma2_3way_levels <- c(0, 0.004, 0.008, 0.012, 0.020)
level_labels <- c("None", "Small (20%)", "Moderate (40%)", "Large (60%)", "Dominant (100%)")

# Design
n_items <- 50
n_prompts <- 4
n_temps <- 3
n_reps <- 5
n_cats <- TRUE_PARAMS$n_categories

cat("=== 04b_sim_additivity.R: Additivity Violation Robustness ===\n")
cat(sprintf("Design: N=%d, C=%d, K=%d, L=%d, R=%d, N_SIM=%d\n",
            n_items, n_cats, n_prompts, n_temps, n_reps, N_SIM))
cat(sprintf("3-way levels: %s\n", paste(sigma2_3way_levels, collapse = ", ")))

# --- Simulation function with 3-way interaction ---
simulate_data_3way <- function(n_items, n_prompts, n_temps, n_reps, params, sigma2_3way) {
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

  # 3-way interaction: item x prompt x temperature
  abt <- array(rnorm(n_items * n_prompts * n_temps, 0, sqrt(sigma2_3way)),
               dim = c(n_items, n_prompts, n_temps))

  rows <- vector("list", n_items * n_prompts * n_temps * n_reps)
  idx <- 0L
  for (i in seq_len(n_items)) {
    for (j in seq_len(n_prompts)) {
      for (k in seq_len(n_temps)) {
        for (r in seq_len(n_reps)) {
          idx <- idx + 1L
          y <- params$mu + alpha[i] + beta[j] + params$tau[k] +
            ab[i, j] + at[i, k] + bt[j, k] + abt[i, j, k] +
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
      component = c("category", "item_id", "variant_id", "item_id:variant_id",
                     "item_id:temperature", "variant_id:temperature", "Residual"),
      variance = rep(NA_real_, 7),
      converged = FALSE
    ))
  }

  vc <- as.data.frame(VarCorr(mod))
  tibble(
    component = vc$grp,
    variance = vc$vcov,
    converged = TRUE,
    singular = isSingular(mod)
  )
}

# --- D-study function ---
compute_dstudy_var <- function(vc_est, n_items_d, n_prompts_d, n_reps_d, n_temps_d, fix_temp) {
  s2 <- setNames(vc_est$variance, vc_est$component)
  s2_cat <- s2["category"]
  s2_item <- s2["item_id"]
  s2_prompt <- s2["variant_id"]
  s2_ip <- s2["item_id:variant_id"]
  s2_it <- s2["item_id:temperature"]
  s2_pt <- s2["variant_id:temperature"]
  s2_gen <- s2["Residual"]

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

# --- D-study scenarios ---
dstudy_scenarios <- tribble(
  ~scenario, ~n_items_d, ~n_prompts_d, ~n_reps_d, ~fix_temp,
  "+5 replications",     n_items,     n_prompts, n_reps + 5, FALSE,
  "+2 prompt variants",  n_items, n_prompts + 2,     n_reps, FALSE,
  "Fix temperature",     n_items,     n_prompts,     n_reps,  TRUE,
  "+5 reps & fix temp",  n_items,     n_prompts, n_reps + 5,  TRUE,
  "Double items",    n_items * 2,     n_prompts,     n_reps, FALSE,
  "Baseline (avg temp)", n_items,     n_prompts,     n_reps, FALSE,
)

# True D-study ranking (computed from true params)
true_vc <- tibble(
  component = c("category", "item_id", "variant_id", "item_id:variant_id",
                 "item_id:temperature", "variant_id:temperature", "Residual"),
  variance = c(TRUE_PARAMS$sigma2_category, TRUE_PARAMS$sigma2_item,
               TRUE_PARAMS$sigma2_prompt, TRUE_PARAMS$sigma2_ip,
               TRUE_PARAMS$sigma2_it, TRUE_PARAMS$sigma2_pt, TRUE_PARAMS$sigma2_gen)
)

# --- True map for bias computation ---
true_map <- c(
  "category" = TRUE_PARAMS$sigma2_category,
  "item_id" = TRUE_PARAMS$sigma2_item,
  "variant_id" = TRUE_PARAMS$sigma2_prompt,
  "item_id:variant_id" = TRUE_PARAMS$sigma2_ip,
  "item_id:temperature" = TRUE_PARAMS$sigma2_it,
  "variant_id:temperature" = TRUE_PARAMS$sigma2_pt,
  "Residual" = TRUE_PARAMS$sigma2_gen
)

# =========================================================================
# Main simulation loop
# =========================================================================
all_bias_results <- list()
all_dstudy_results <- list()

for (lev_idx in seq_along(sigma2_3way_levels)) {
  s2_3way <- sigma2_3way_levels[lev_idx]
  lev_label <- level_labels[lev_idx]
  cat(sprintf("\n--- Level %d/%d: sigma2_3way = %.3f (%s) ---\n",
              lev_idx, length(sigma2_3way_levels), s2_3way, lev_label))

  # True residual should absorb 3-way: effective sigma2_gen = sigma2_gen + sigma2_3way
  # (since lmer has no 3-way term, it goes into residual)
  effective_true_map <- true_map
  effective_true_map["Residual"] <- TRUE_PARAMS$sigma2_gen + s2_3way

  # Compute true D-study rankings with effective residual
  effective_true_vc <- true_vc
  effective_true_vc$variance[effective_true_vc$component == "Residual"] <-
    TRUE_PARAMS$sigma2_gen + s2_3way

  true_dstudy_vars <- dstudy_scenarios %>%
    rowwise() %>%
    mutate(true_var = compute_dstudy_var(effective_true_vc, n_items_d, n_prompts_d,
                                          n_reps_d, n_temps, fix_temp)) %>%
    ungroup()
  true_dstudy_vars <- true_dstudy_vars %>%
    mutate(true_rank = rank(true_var))

  sim_bias <- list()
  sim_dstudy <- list()

  for (sim in seq_len(N_SIM)) {
    if (sim %% 50 == 0) cat(sprintf("  Sim %d/%d\n", sim, N_SIM))
    set.seed(42 + sim)

    df <- simulate_data_3way(n_items, n_prompts, n_temps, n_reps, TRUE_PARAMS, s2_3way)
    result <- fit_and_extract(df)

    if (!result$converged[1]) next

    # Bias for each component
    for (comp in names(effective_true_map)) {
      est_val <- result$variance[result$component == comp]
      if (length(est_val) == 0) next
      sim_bias[[length(sim_bias) + 1]] <- list(
        sigma2_3way = s2_3way,
        level_label = lev_label,
        component = comp,
        true_value = effective_true_map[comp],
        est_value = est_val,
        sim_id = sim
      )
    }

    # D-study ranking
    est_dstudy_vars <- dstudy_scenarios %>%
      rowwise() %>%
      mutate(est_var = compute_dstudy_var(result, n_items_d, n_prompts_d,
                                           n_reps_d, n_temps, fix_temp)) %>%
      ungroup()
    est_dstudy_vars <- est_dstudy_vars %>%
      mutate(est_rank = rank(est_var))

    # Kendall's tau between estimated and true rankings
    kt <- cor(est_dstudy_vars$est_rank, true_dstudy_vars$true_rank, method = "kendall")

    sim_dstudy[[length(sim_dstudy) + 1]] <- list(
      sigma2_3way = s2_3way,
      level_label = lev_label,
      kendall_tau = kt,
      sim_id = sim
    )
  }

  all_bias_results[[lev_idx]] <- bind_rows(sim_bias)
  all_dstudy_results[[lev_idx]] <- bind_rows(sim_dstudy)
}

# =========================================================================
# Aggregate bias results
# =========================================================================
bias_df <- bind_rows(all_bias_results)

bias_summary <- bias_df %>%
  group_by(sigma2_3way, level_label, component) %>%
  summarize(
    true_value = first(true_value),
    mean_est = mean(est_value, na.rm = TRUE),
    bias = mean(est_value - true_value, na.rm = TRUE),
    rel_bias_pct = 100 * mean((est_value - true_value) / true_value, na.rm = TRUE),
    rmse = sqrt(mean((est_value - true_value)^2, na.rm = TRUE)),
    n_sims = n(),
    .groups = "drop"
  )

cat("\n=== Bias Summary ===\n")
print(bias_summary %>% select(level_label, component, true_value, mean_est, rel_bias_pct, rmse))

# =========================================================================
# Aggregate D-study results
# =========================================================================
dstudy_df <- bind_rows(all_dstudy_results)

dstudy_summary <- dstudy_df %>%
  group_by(sigma2_3way, level_label) %>%
  summarize(
    mean_kendall_tau = mean(kendall_tau, na.rm = TRUE),
    sd_kendall_tau = sd(kendall_tau, na.rm = TRUE),
    n_sims = n(),
    .groups = "drop"
  )

cat("\n=== D-Study Rank Preservation (Kendall's tau) ===\n")
print(dstudy_summary)

# =========================================================================
# Save outputs
# =========================================================================
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(bias_summary, file.path(output_dir, "sim_additivity.csv"))
cat("\nSaved:", file.path(output_dir, "sim_additivity.csv"), "\n")

write_csv(dstudy_summary, file.path(output_dir, "sim_additivity_dstudy.csv"))
cat("Saved:", file.path(output_dir, "sim_additivity_dstudy.csv"), "\n")

# =========================================================================
# Figures
# =========================================================================
fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

tle_labels <- c(
  "category" = "Between-category",
  "item_id" = "Within-category item",
  "variant_id" = "Prompt",
  "item_id:variant_id" = "Item x Prompt",
  "item_id:temperature" = "Item x Temp",
  "variant_id:temperature" = "Prompt x Temp",
  "Residual" = "Generation (residual)"
)

# Figure 1: Bias by component across 3-way levels
bias_summary <- bias_summary %>%
  mutate(
    component_label = tle_labels[component],
    level_label = factor(level_label, levels = level_labels)
  )

p_bias <- ggplot(bias_summary, aes(x = level_label, y = rel_bias_pct,
                                    fill = component_label)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  facet_wrap(~component_label, scales = "free_y", ncol = 2) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Bias Under Additivity Violations",
    subtitle = sprintf("3-way item x prompt x temp interaction (N_SIM=%d)", N_SIM),
    x = expression(sigma[3 * way]^2 ~ "level"),
    y = "Relative bias (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "sim_additivity_bias.pdf"), p_bias, width = 9, height = 7)
ggsave(file.path(fig_dir, "sim_additivity_bias.png"), p_bias, width = 9, height = 7, dpi = 300)
cat("Saved: sim_additivity_bias.pdf/png\n")

# Figure 2: D-study rank preservation
dstudy_summary <- dstudy_summary %>%
  mutate(level_label = factor(level_label, levels = level_labels))

p_dstudy <- ggplot(dstudy_summary, aes(x = level_label, y = mean_kendall_tau)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_kendall_tau - 1.96 * sd_kendall_tau / sqrt(n_sims),
                     ymax = mean_kendall_tau + 1.96 * sd_kendall_tau / sqrt(n_sims)),
                width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  ylim(0, 1.1) +
  labs(
    title = "D-Study Rank Preservation Under Additivity Violations",
    subtitle = sprintf("Kendall's tau: estimated vs. true scenario ranking (N_SIM=%d)", N_SIM),
    x = expression(sigma[3 * way]^2 ~ "level"),
    y = expression("Kendall's" ~ tau)
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "sim_additivity_dstudy_rank.pdf"), p_dstudy, width = 7, height = 5)
ggsave(file.path(fig_dir, "sim_additivity_dstudy_rank.png"), p_dstudy, width = 7, height = 5, dpi = 300)
cat("Saved: sim_additivity_dstudy_rank.pdf/png\n")

# =========================================================================
# Convergence and singularity diagnostics
# =========================================================================
cat("\n--- Convergence / Singularity Diagnostics ---\n")

diag_04b <- bias_df %>%
  group_by(sigma2_3way, level_label) %>%
  summarize(
    n_sims = n_distinct(sim_id),
    .groups = "drop"
  )

# Count convergence failures per level (sims that produced no rows)
total_sims_attempted <- N_SIM
diag_04b <- diag_04b %>%
  mutate(
    convergence_rate = n_sims / total_sims_attempted
  )

# Per-component summary stats
comp_stats_04b <- bias_df %>%
  group_by(sigma2_3way, level_label, component) %>%
  summarize(
    mean_est = mean(est_value, na.rm = TRUE),
    sd_est = sd(est_value, na.rm = TRUE),
    pct_boundary = 100 * mean(est_value < 1e-10, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nConvergence summary:\n")
print(diag_04b)

write_csv(diag_04b, file.path(output_dir, "sim_diagnostics_04b.csv"))
write_csv(comp_stats_04b, file.path(output_dir, "sim_component_stats_04b.csv"))
cat("Saved: sim_diagnostics_04b.csv, sim_component_stats_04b.csv\n")

cat("\n=== 04b_sim_additivity.R complete ===\n")
