# 04d_sim_small_k.R
# Monte Carlo simulation: small-K prompt sensitivity precision.
# Quantifies precision of sigma2_beta as a function of K (number of prompt variants).
# Answers: "what is the minimum viable K for D-study projections to be directionally correct?"

library(tidyverse)
library(lme4)

set.seed(42)

N_SIM <- 1000
N_SIM_CI <- 50  # Subset for expensive profile CI computation (skip boundary sigma2_beta=0)

# --- True DGP parameters ---
TRUE_PARAMS_BASE <- list(
  mu = 0.5,
  sigma2_category = 0.03,
  sigma2_item = 0.05,
  sigma2_ip = 0.02,
  sigma2_it = 0.015,
  sigma2_pt = 0.005,
  sigma2_gen = 0.06,
  tau = c(0, -0.03, 0.05),
  n_categories = 5
)

# Sweep parameters
K_levels <- c(2, 3, 4, 5, 7, 10)
sigma2_beta_levels <- c(0, 0.04, 0.10)
beta_labels <- c("0" = "Null (0)", "0.04" = "Moderate (0.04)", "0.1" = "Large (0.10)")

# Design
n_items <- 100
n_temps <- 3
n_reps <- 5
n_cats <- TRUE_PARAMS_BASE$n_categories

cat("=== 04d_sim_small_k.R: Small-K Prompt Sensitivity Precision ===\n")
cat(sprintf("Design: N=%d, C=%d, L=%d, R=%d, N_SIM=%d\n",
            n_items, n_cats, n_temps, n_reps, N_SIM))
cat(sprintf("K levels: %s\n", paste(K_levels, collapse = ", ")))
cat(sprintf("sigma2_beta levels: %s\n", paste(sigma2_beta_levels, collapse = ", ")))

# --- Simulation function ---
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

# --- Fit and extract ---
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
  if (is.null(mod)) return(list(vc = NULL, mod = NULL, converged = FALSE))

  vc <- as.data.frame(VarCorr(mod))
  list(
    vc = tibble(component = vc$grp, variance = vc$vcov),
    mod = mod,
    converged = TRUE
  )
}

# --- D-study directional correctness ---
# Does the model correctly identify whether adding prompts or reps gives more reduction?
compute_dstudy_direction <- function(vc_est, n_items_d, n_prompts_d, n_reps_d, n_temps_d) {
  s2 <- setNames(vc_est$variance, vc_est$component)
  s2_cat <- s2["category"]
  s2_item <- s2["item_id"]
  s2_prompt <- s2["variant_id"]
  s2_ip <- s2["item_id:variant_id"]
  s2_it <- s2["item_id:temperature"]
  s2_pt <- s2["variant_id:temperature"]
  s2_gen <- s2["Residual"]

  # Baseline (avg temp)
  baseline <- s2_cat/n_cats + s2_item/n_items_d + s2_prompt/n_prompts_d +
    s2_ip/(n_items_d * n_prompts_d) + s2_it/(n_items_d * n_temps_d) +
    s2_pt/(n_prompts_d * n_temps_d) + s2_gen/(n_items_d * n_prompts_d * n_temps_d * n_reps_d)

  # +2 prompts
  add_prompts <- s2_cat/n_cats + s2_item/n_items_d + s2_prompt/(n_prompts_d + 2) +
    s2_ip/(n_items_d * (n_prompts_d + 2)) + s2_it/(n_items_d * n_temps_d) +
    s2_pt/((n_prompts_d + 2) * n_temps_d) +
    s2_gen/(n_items_d * (n_prompts_d + 2) * n_temps_d * n_reps_d)

  # +5 reps
  add_reps <- s2_cat/n_cats + s2_item/n_items_d + s2_prompt/n_prompts_d +
    s2_ip/(n_items_d * n_prompts_d) + s2_it/(n_items_d * n_temps_d) +
    s2_pt/(n_prompts_d * n_temps_d) +
    s2_gen/(n_items_d * n_prompts_d * n_temps_d * (n_reps_d + 5))

  # Reduction from each
  red_prompts <- baseline - add_prompts
  red_reps <- baseline - add_reps

  # "prompts help more" or "reps help more"
  list(
    prompts_better = red_prompts > red_reps,
    red_prompts = red_prompts,
    red_reps = red_reps
  )
}

# =========================================================================
# Main simulation loop
# =========================================================================
all_results <- list()
all_ci_results <- list()

total_combos <- length(K_levels) * length(sigma2_beta_levels)
combo_idx <- 0

for (K in K_levels) {
  for (s2_beta in sigma2_beta_levels) {
    combo_idx <- combo_idx + 1
    cat(sprintf("\n--- Combo %d/%d: K=%d, sigma2_beta=%.2f ---\n",
                combo_idx, total_combos, K, s2_beta))

    params <- TRUE_PARAMS_BASE
    params$sigma2_prompt <- s2_beta

    # True direction
    true_vc <- tibble(
      component = c("category", "item_id", "variant_id", "item_id:variant_id",
                     "item_id:temperature", "variant_id:temperature", "Residual"),
      variance = c(params$sigma2_category, params$sigma2_item, params$sigma2_prompt,
                   params$sigma2_ip, params$sigma2_it, params$sigma2_pt, params$sigma2_gen)
    )
    true_dir <- compute_dstudy_direction(true_vc, n_items, K, n_reps, n_temps)

    sim_estimates <- list()
    sim_direction <- list()

    for (sim in seq_len(N_SIM)) {
      if (sim %% 50 == 0) cat(sprintf("  Sim %d/%d\n", sim, N_SIM))
      set.seed(42 + sim)

      df <- simulate_data(n_items, K, n_temps, n_reps, params)
      result <- fit_and_extract(df)

      if (!result$converged) next

      # Extract sigma2_beta estimate
      s2_beta_hat <- result$vc$variance[result$vc$component == "variant_id"]
      if (length(s2_beta_hat) == 0) next

      sim_estimates[[length(sim_estimates) + 1]] <- list(
        K = K, true_sigma2_beta = s2_beta,
        est_sigma2_beta = s2_beta_hat,
        sim_id = sim
      )

      # D-study directional correctness
      est_dir <- compute_dstudy_direction(result$vc, n_items, K, n_reps, n_temps)
      correct <- est_dir$prompts_better == true_dir$prompts_better
      sim_direction[[length(sim_direction) + 1]] <- list(
        K = K, true_sigma2_beta = s2_beta,
        dstudy_correct = correct,
        sim_id = sim
      )

      # CI computation (subset of sims; skip boundary case sigma2_beta=0)
      if (sim <= N_SIM_CI && s2_beta > 0) {
        ci <- tryCatch({
          # Profile only the variant_id SD parameter (much faster than profiling all)
          ci_raw <- confint(result$mod, method = "profile",
                            parm = "sd_variant_id.(Intercept)",
                            oldNames = FALSE, quiet = TRUE)
          ci_df <- as.data.frame(ci_raw) %>%
            rownames_to_column("param") %>%
            rename(ci_lower = `2.5 %`, ci_upper = `97.5 %`)
          sd_row <- ci_df[1, ]  # Only one row since we profiled one param
          true_sd <- sqrt(s2_beta)
          covers <- sd_row$ci_lower <= true_sd & true_sd <= sd_row$ci_upper
          ci_width <- sd_row$ci_upper - sd_row$ci_lower
          list(covers = covers, ci_width = ci_width)
        }, error = function(e) NULL)

        if (!is.null(ci)) {
          all_ci_results[[length(all_ci_results) + 1]] <- list(
            K = K, true_sigma2_beta = s2_beta,
            covers = ci$covers, ci_width = ci$ci_width,
            sim_id = sim
          )
        }
      }
    }

    all_results[[combo_idx]] <- list(
      estimates = bind_rows(sim_estimates),
      direction = bind_rows(sim_direction)
    )
  }
}

# =========================================================================
# Aggregate results
# =========================================================================
cat("\n=== Aggregating results ===\n")

estimates_df <- bind_rows(lapply(all_results, `[[`, "estimates"))
direction_df <- bind_rows(lapply(all_results, `[[`, "direction"))
ci_df <- bind_rows(all_ci_results)

# Main summary
main_summary <- estimates_df %>%
  group_by(K, true_sigma2_beta) %>%
  summarize(
    mean_est = mean(est_sigma2_beta, na.rm = TRUE),
    bias = mean(est_sigma2_beta - true_sigma2_beta, na.rm = TRUE),
    rel_bias_pct = ifelse(
      first(true_sigma2_beta) == 0,
      NA_real_,
      100 * mean((est_sigma2_beta - true_sigma2_beta) / true_sigma2_beta, na.rm = TRUE)
    ),
    rmse = sqrt(mean((est_sigma2_beta - true_sigma2_beta)^2, na.rm = TRUE)),
    n_sims = n(),
    .groups = "drop"
  )

# D-study directional correctness
direction_summary <- direction_df %>%
  group_by(K, true_sigma2_beta) %>%
  summarize(
    dstudy_direction_correct_pct = 100 * mean(dstudy_correct, na.rm = TRUE),
    n_sims = n(),
    .groups = "drop"
  )

# CI coverage
ci_summary <- ci_df %>%
  group_by(K, true_sigma2_beta) %>%
  summarize(
    coverage = mean(covers, na.rm = TRUE),
    ci_width = mean(ci_width, na.rm = TRUE),
    n_sims = n(),
    .groups = "drop"
  )

# Merge all
final_summary <- main_summary %>%
  left_join(direction_summary %>% select(K, true_sigma2_beta, dstudy_direction_correct_pct),
            by = c("K", "true_sigma2_beta")) %>%
  left_join(ci_summary %>% select(K, true_sigma2_beta, coverage, ci_width),
            by = c("K", "true_sigma2_beta"))

cat("\n=== Summary ===\n")
print(final_summary)

# =========================================================================
# Save outputs
# =========================================================================
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(final_summary, file.path(output_dir, "sim_small_k.csv"))
cat("\nSaved:", file.path(output_dir, "sim_small_k.csv"), "\n")

# =========================================================================
# Figures
# =========================================================================
fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Figure 1: RMSE and CI width by K, faceted by sigma2_beta
p_precision <- final_summary %>%
  mutate(beta_label = beta_labels[as.character(true_sigma2_beta)]) %>%
  ggplot(aes(x = K)) +
  geom_line(aes(y = rmse, color = "RMSE"), linewidth = 0.8) +
  geom_point(aes(y = rmse, color = "RMSE"), size = 2) +
  geom_ribbon(aes(ymin = rmse - ci_width/4, ymax = rmse + ci_width/4),
              alpha = 0.15, fill = "steelblue") +
  facet_wrap(~beta_label, scales = "free_y") +
  scale_color_manual(values = c("RMSE" = "steelblue")) +
  labs(
    title = expression("Precision of" ~ hat(sigma)[beta]^2 ~ "as a Function of K"),
    subtitle = sprintf("RMSE with CI width ribbon (N_SIM=%d, CI on %d sims)", N_SIM, N_SIM_CI),
    x = "Number of prompt variants (K)",
    y = expression("RMSE of" ~ hat(sigma)[beta]^2),
    color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "sim_small_k_precision.pdf"), p_precision, width = 10, height = 4)
ggsave(file.path(fig_dir, "sim_small_k_precision.png"), p_precision, width = 10, height = 4, dpi = 300)
cat("Saved: sim_small_k_precision.pdf/png\n")

# Figure 2: D-study directional correctness
p_dstudy <- final_summary %>%
  mutate(beta_label = beta_labels[as.character(true_sigma2_beta)]) %>%
  ggplot(aes(x = K, y = dstudy_direction_correct_pct,
             color = beta_label, group = beta_label)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "gray50") +
  ylim(0, 105) +
  scale_color_brewer(palette = "Set1") +
  labs(
    title = "D-Study Directional Correctness vs. K",
    subtitle = sprintf("Correctly identifies prompts vs. reps as best intervention (N_SIM=%d)", N_SIM),
    x = "Number of prompt variants (K)",
    y = "Directionally correct (%)",
    color = expression(sigma[beta]^2 ~ "level")
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "sim_small_k_dstudy.pdf"), p_dstudy, width = 7, height = 5)
ggsave(file.path(fig_dir, "sim_small_k_dstudy.png"), p_dstudy, width = 7, height = 5, dpi = 300)
cat("Saved: sim_small_k_dstudy.pdf/png\n")

# =========================================================================
# Convergence and singularity diagnostics
# =========================================================================
cat("\n--- Convergence / Singularity Diagnostics ---\n")

diag_04d <- estimates_df %>%
  group_by(K, true_sigma2_beta) %>%
  summarize(
    n_sims = n(),
    convergence_rate = n() / N_SIM,
    .groups = "drop"
  )

cat("\nConvergence summary:\n")
print(diag_04d)

write_csv(diag_04d, file.path(output_dir, "sim_diagnostics_04d.csv"))
cat("Saved: sim_diagnostics_04d.csv\n")

# =========================================================================
# Knife-Edge Directional Test
# =========================================================================
# Tests directional accuracy near the decision boundary where "+2 prompts"
# and "+5 reps" give approximately equal variance reduction.
#
# Key insight: with the default DGP parameters (non-zero interaction terms),
# adding prompts is ALWAYS more effective than adding reps regardless of
# sigma2_beta, because interactions (sigma2_ip, sigma2_pt) also benefit from
# more prompts. To create a meaningful knife-edge, we use a modified DGP
# with zero prompt-involving interactions and a smaller design (N=20, R=3).
# This isolates the pure sigma2_beta/K vs sigma2_eps/(NKLR) tradeoff.

cat("\n=== Knife-Edge Directional Test ===\n")
cat("Tests directional accuracy near the decision boundary.\n")
cat("DGP: zero prompt interactions, N=20, L=3, R=3\n")

KE_N_ITEMS <- 20
KE_N_TEMPS <- 3
KE_N_REPS <- 3
KE_N_CATS <- 5

KE_PARAMS_BASE <- list(
  mu = 0.5,
  sigma2_category = 0.03,
  sigma2_item = 0.05,
  sigma2_ip = 0,       # Zero interaction for clean knife-edge
  sigma2_it = 0,
  sigma2_pt = 0,
  sigma2_gen = 0.06,
  tau = c(0, -0.03, 0.05),
  n_categories = KE_N_CATS
)

# Analytical boundary: sigma2_beta* where Red(+2 prompts) = Red(+5 reps)
# With zero interactions:
#   Red_prompts = 2/(K(K+2)) * [sigma2_beta + sigma2_eps/(NLR)]
#   Red_reps    = 5*sigma2_eps / (N*K*L*R*(R+5))
# Solving: sigma2_beta* = sigma2_eps/(NLR) * [5(K+2)/(2(R+5)) - 1]
compute_boundary_sigma2_beta <- function(K, sigma2_eps, N, L, R) {
  sigma2_eps / (N * L * R) * (5 * (K + 2) / (2 * (R + 5)) - 1)
}

multipliers <- c(0.25, 0.5, 0.8, 1.0, 1.2, 2.0, 4.0)

# Print boundary values
for (K in K_levels) {
  s2b_star <- compute_boundary_sigma2_beta(K, 0.06, KE_N_ITEMS, KE_N_TEMPS, KE_N_REPS)
  cat(sprintf("  K=%2d: sigma2_beta* = %.6f\n", K, s2b_star))
}

# D-study direction with knife-edge design parameters
compute_dstudy_direction_ke <- function(vc_est, n_items_d, n_prompts_d, n_reps_d, n_temps_d) {
  s2 <- setNames(vc_est$variance, vc_est$component)
  n_cats <- KE_N_CATS

  baseline <- s2["category"]/n_cats + s2["item_id"]/n_items_d +
    s2["variant_id"]/n_prompts_d +
    s2["item_id:variant_id"]/(n_items_d * n_prompts_d) +
    s2["item_id:temperature"]/(n_items_d * n_temps_d) +
    s2["variant_id:temperature"]/(n_prompts_d * n_temps_d) +
    s2["Residual"]/(n_items_d * n_prompts_d * n_temps_d * n_reps_d)

  add_prompts <- s2["category"]/n_cats + s2["item_id"]/n_items_d +
    s2["variant_id"]/(n_prompts_d + 2) +
    s2["item_id:variant_id"]/(n_items_d * (n_prompts_d + 2)) +
    s2["item_id:temperature"]/(n_items_d * n_temps_d) +
    s2["variant_id:temperature"]/((n_prompts_d + 2) * n_temps_d) +
    s2["Residual"]/(n_items_d * (n_prompts_d + 2) * n_temps_d * n_reps_d)

  add_reps <- s2["category"]/n_cats + s2["item_id"]/n_items_d +
    s2["variant_id"]/n_prompts_d +
    s2["item_id:variant_id"]/(n_items_d * n_prompts_d) +
    s2["item_id:temperature"]/(n_items_d * n_temps_d) +
    s2["variant_id:temperature"]/(n_prompts_d * n_temps_d) +
    s2["Residual"]/(n_items_d * n_prompts_d * n_temps_d * (n_reps_d + 5))

  list(prompts_better = unname(baseline - add_prompts) > unname(baseline - add_reps))
}

ke_results <- list()

for (K in K_levels) {
  sigma2_beta_star <- compute_boundary_sigma2_beta(
    K, KE_PARAMS_BASE$sigma2_gen, KE_N_ITEMS, KE_N_TEMPS, KE_N_REPS
  )
  if (sigma2_beta_star <= 0) {
    cat(sprintf("\n  K=%d: boundary sigma2_beta* = %.6f (non-positive, skipping)\n",
                K, sigma2_beta_star))
    next
  }

  cat(sprintf("\n  K=%d: boundary sigma2_beta* = %.6f\n", K, sigma2_beta_star))

  for (mult in multipliers) {
    sigma2_beta_test <- sigma2_beta_star * mult
    params <- KE_PARAMS_BASE
    params$sigma2_prompt <- sigma2_beta_test

    # True direction at these parameters
    true_vc_ke <- tibble(
      component = c("category", "item_id", "variant_id", "item_id:variant_id",
                     "item_id:temperature", "variant_id:temperature", "Residual"),
      variance = c(params$sigma2_category, params$sigma2_item, params$sigma2_prompt,
                   params$sigma2_ip, params$sigma2_it, params$sigma2_pt, params$sigma2_gen)
    )
    true_dir <- compute_dstudy_direction_ke(true_vc_ke, KE_N_ITEMS, K, KE_N_REPS, KE_N_TEMPS)

    n_correct <- 0L
    n_converged <- 0L

    for (sim in seq_len(N_SIM)) {
      if (sim %% 200 == 0) cat(sprintf("    K=%d, mult=%.2f: sim %d/%d\n", K, mult, sim, N_SIM))
      set.seed(42 + sim + K * 100 + round(mult * 1000))

      df <- simulate_data(KE_N_ITEMS, K, KE_N_TEMPS, KE_N_REPS, params)
      result <- fit_and_extract(df)
      if (!result$converged) next
      n_converged <- n_converged + 1L

      est_dir <- compute_dstudy_direction_ke(result$vc, KE_N_ITEMS, K, KE_N_REPS, KE_N_TEMPS)
      if (est_dir$prompts_better == true_dir$prompts_better) n_correct <- n_correct + 1L
    }

    ke_results[[length(ke_results) + 1]] <- list(
      K = K,
      sigma2_beta_star = sigma2_beta_star,
      multiplier = mult,
      sigma2_beta = sigma2_beta_test,
      true_prompts_better = true_dir$prompts_better,
      directional_accuracy_pct = 100 * n_correct / max(n_converged, 1),
      n_converged = n_converged
    )
  }
}

ke_df <- bind_rows(ke_results)

cat("\n=== Knife-Edge Results ===\n")
print(ke_df %>% select(K, multiplier, sigma2_beta, true_prompts_better,
                        directional_accuracy_pct, n_converged))

# Save knife-edge results
write_csv(ke_df, file.path(output_dir, "sim_small_k_knife_edge.csv"))
cat("\nSaved:", file.path(output_dir, "sim_small_k_knife_edge.csv"), "\n")

# Figure: Knife-edge directional accuracy
p_knife <- ke_df %>%
  mutate(K_label = paste0("K = ", K)) %>%
  ggplot(aes(x = multiplier, y = directional_accuracy_pct, color = K_label, group = K_label)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = 1.0, linetype = "dotted", color = "gray70") +
  scale_x_continuous(breaks = multipliers) +
  scale_color_brewer(palette = "Set1") +
  labs(
    title = "Directional Accuracy Near the Decision Boundary",
    subtitle = sprintf("Knife-edge: zero interactions, N=%d, R=%d (%d sims per point)",
                        KE_N_ITEMS, KE_N_REPS, N_SIM),
    x = expression("Multiplier of boundary" ~ sigma[beta]^{"2*"}),
    y = "Directionally correct (%)",
    color = "Prompt variants"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "sim_small_k_knife_edge.pdf"), p_knife, width = 8, height = 5)
ggsave(file.path(fig_dir, "sim_small_k_knife_edge.png"), p_knife, width = 8, height = 5, dpi = 300)
cat("Saved: sim_small_k_knife_edge.pdf/png\n")

cat("\n=== 04d_sim_small_k.R complete ===\n")
