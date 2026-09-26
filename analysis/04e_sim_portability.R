# 04e_sim_portability.R
# Monte Carlo simulation: cross-model D-study portability.
# Quantifies D-study transfer error: if variance components are estimated from
# Model A and applied to predict outcomes for Model B, how wrong is the recommendation?

library(tidyverse)
library(lme4)

set.seed(42)

N_SIM <- 1000

# --- Two model profiles ---
PARAMS_A <- list(  # Frontier, low-variance, prompt-robust
  mu = 0.5,
  sigma2_category = 0.03,
  sigma2_item = 0.03,
  sigma2_prompt = 0.02,
  sigma2_ip = 0.01,
  sigma2_it = 0.01,
  sigma2_pt = 0.003,
  sigma2_gen = 0.04,
  tau = c(0, -0.02, 0.03),
  n_categories = 5
)

PARAMS_B <- list(  # Cheap, high-variance, prompt-sensitive
  mu = 0.45,
  sigma2_category = 0.03,
  sigma2_item = 0.08,
  sigma2_prompt = 0.08,
  sigma2_ip = 0.04,
  sigma2_it = 0.02,
  sigma2_pt = 0.01,
  sigma2_gen = 0.10,
  tau = c(0, -0.05, 0.08),
  n_categories = 5
)

# Design
n_items <- 50
n_prompts <- 4
n_temps <- 3
n_reps <- 5
n_cats <- 5

cat("=== 04e_sim_portability.R: Cross-Model D-Study Portability ===\n")
cat(sprintf("Design: N=%d, C=%d, K=%d, L=%d, R=%d, N_SIM=%d\n",
            n_items, n_cats, n_prompts, n_temps, n_reps, N_SIM))

cat("\nModel A (frontier): sigma2_prompt=0.02, sigma2_gen=0.04\n")
cat("Model B (cheap):    sigma2_prompt=0.08, sigma2_gen=0.10\n")

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
  if (is.null(mod)) return(NULL)

  vc <- as.data.frame(VarCorr(mod))
  tibble(component = vc$grp, variance = vc$vcov)
}

# --- D-study variance computation ---
compute_dstudy_var <- function(vc, n_items_d, n_prompts_d, n_reps_d, n_temps_d, fix_temp) {
  s2 <- setNames(vc$variance, vc$component)
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
  "Baseline (avg temp)",  n_items,     n_prompts,     n_reps, FALSE,
  "+5 replications",      n_items,     n_prompts, n_reps + 5, FALSE,
  "+2 prompt variants",   n_items, n_prompts + 2,     n_reps, FALSE,
  "Fix temperature",      n_items,     n_prompts,     n_reps,  TRUE,
  "+5 reps & fix temp",   n_items,     n_prompts, n_reps + 5,  TRUE,
  "Double items",     n_items * 2,     n_prompts,     n_reps, FALSE,
)

# --- True D-study for each model ---
make_true_vc <- function(params) {
  tibble(
    component = c("category", "item_id", "variant_id", "item_id:variant_id",
                   "item_id:temperature", "variant_id:temperature", "Residual"),
    variance = c(params$sigma2_category, params$sigma2_item, params$sigma2_prompt,
                 params$sigma2_ip, params$sigma2_it, params$sigma2_pt, params$sigma2_gen)
  )
}

true_vc_A <- make_true_vc(PARAMS_A)
true_vc_B <- make_true_vc(PARAMS_B)

compute_all_dstudy <- function(vc) {
  dstudy_scenarios %>%
    rowwise() %>%
    mutate(total_var = compute_dstudy_var(vc, n_items_d, n_prompts_d,
                                           n_reps_d, n_temps, fix_temp)) %>%
    ungroup()
}

true_dstudy_A <- compute_all_dstudy(true_vc_A)
true_dstudy_B <- compute_all_dstudy(true_vc_B)

cat("\nTrue D-study (Model A):\n")
print(true_dstudy_A %>% select(scenario, total_var))
cat("\nTrue D-study (Model B):\n")
print(true_dstudy_B %>% select(scenario, total_var))

# =========================================================================
# Main simulation loop
# =========================================================================
transfer_results <- list()
flip_results <- list()

for (sim in seq_len(N_SIM)) {
  if (sim %% 20 == 0) cat(sprintf("  Sim %d/%d\n", sim, N_SIM))
  set.seed(42 + sim)

  # Fit Model A
  df_A <- simulate_data(n_items, n_prompts, n_temps, n_reps, PARAMS_A)
  vc_A <- fit_and_extract(df_A)
  if (is.null(vc_A)) next

  # Fit Model B
  set.seed(42 + sim + 10000)  # Different seed for B
  df_B <- simulate_data(n_items, n_prompts, n_temps, n_reps, PARAMS_B)
  vc_B <- fit_and_extract(df_B)
  if (is.null(vc_B)) next

  # D-study from estimated components
  est_dstudy_A <- compute_all_dstudy(vc_A)
  est_dstudy_B <- compute_all_dstudy(vc_B)

  # Transfer: A's estimates applied to B's true
  # "Predicted" = what A's D-study says; "Actual" = what B's truth says
  for (sc_idx in seq_len(nrow(dstudy_scenarios))) {
    sc <- dstudy_scenarios$scenario[sc_idx]

    # A -> B transfer
    transfer_results[[length(transfer_results) + 1]] <- list(
      transfer_direction = "A_to_B",
      scenario = sc,
      predicted_var = est_dstudy_A$total_var[sc_idx],
      actual_var = true_dstudy_B$total_var[sc_idx],
      sim_id = sim
    )

    # B -> A transfer
    transfer_results[[length(transfer_results) + 1]] <- list(
      transfer_direction = "B_to_A",
      scenario = sc,
      predicted_var = est_dstudy_B$total_var[sc_idx],
      actual_var = true_dstudy_A$total_var[sc_idx],
      sim_id = sim
    )

    # Same-model (sanity check)
    transfer_results[[length(transfer_results) + 1]] <- list(
      transfer_direction = "A_to_A",
      scenario = sc,
      predicted_var = est_dstudy_A$total_var[sc_idx],
      actual_var = true_dstudy_A$total_var[sc_idx],
      sim_id = sim
    )

    transfer_results[[length(transfer_results) + 1]] <- list(
      transfer_direction = "B_to_B",
      scenario = sc,
      predicted_var = est_dstudy_B$total_var[sc_idx],
      actual_var = true_dstudy_B$total_var[sc_idx],
      sim_id = sim
    )
  }

  # Optimal intervention flip
  # Exclude baseline — rank only non-baseline scenarios by variance reduction
  non_baseline_A <- est_dstudy_A %>% filter(scenario != "Baseline (avg temp)")
  non_baseline_B <- est_dstudy_B %>% filter(scenario != "Baseline (avg temp)")
  true_non_baseline_A <- true_dstudy_A %>% filter(scenario != "Baseline (avg temp)")
  true_non_baseline_B <- true_dstudy_B %>% filter(scenario != "Baseline (avg temp)")

  best_from_A <- non_baseline_A$scenario[which.min(non_baseline_A$total_var)]
  best_from_B <- non_baseline_B$scenario[which.min(non_baseline_B$total_var)]
  true_best_for_A <- true_non_baseline_A$scenario[which.min(true_non_baseline_A$total_var)]
  true_best_for_B <- true_non_baseline_B$scenario[which.min(true_non_baseline_B$total_var)]

  # A's recommendation applied to B: does it match B's true best?
  flip_results[[length(flip_results) + 1]] <- list(
    transfer_direction = "A_to_B",
    est_best = best_from_A,
    true_best = true_best_for_B,
    flipped = best_from_A != true_best_for_B,
    sim_id = sim
  )

  # B's recommendation applied to A
  flip_results[[length(flip_results) + 1]] <- list(
    transfer_direction = "B_to_A",
    est_best = best_from_B,
    true_best = true_best_for_A,
    flipped = best_from_B != true_best_for_A,
    sim_id = sim
  )
}

# =========================================================================
# Aggregate transfer results
# =========================================================================
cat("\n=== Transfer Prediction Error ===\n")

transfer_df <- bind_rows(transfer_results)

transfer_summary <- transfer_df %>%
  group_by(transfer_direction, scenario) %>%
  summarize(
    mean_predicted_var = mean(predicted_var, na.rm = TRUE),
    mean_actual_var = mean(actual_var, na.rm = TRUE),
    mean_rel_error = mean(abs(predicted_var - actual_var) / actual_var, na.rm = TRUE),
    .groups = "drop"
  )

# Rank correlation per sim
rank_cors <- transfer_df %>%
  group_by(transfer_direction, sim_id) %>%
  summarize(
    rank_cor = cor(predicted_var, actual_var, method = "kendall"),
    .groups = "drop"
  ) %>%
  group_by(transfer_direction) %>%
  summarize(
    mean_rank_cor = mean(rank_cor, na.rm = TRUE),
    sd_rank_cor = sd(rank_cor, na.rm = TRUE),
    .groups = "drop"
  )

transfer_summary <- transfer_summary %>%
  left_join(rank_cors %>% select(transfer_direction, rank_correlation = mean_rank_cor),
            by = "transfer_direction")

cat("\nTransfer prediction error:\n")
print(transfer_summary)

cat("\nRank correlations:\n")
print(rank_cors)

# =========================================================================
# Aggregate flip results
# =========================================================================
cat("\n=== Optimal Intervention Flip Rate ===\n")

flip_df <- bind_rows(flip_results)

flip_summary <- flip_df %>%
  group_by(transfer_direction) %>%
  summarize(
    flip_rate = mean(flipped, na.rm = TRUE),
    n_sims = n(),
    .groups = "drop"
  )

# Most common flip pattern
flip_patterns <- flip_df %>%
  filter(flipped) %>%
  group_by(transfer_direction) %>%
  count(est_best, true_best, sort = TRUE) %>%
  slice_head(n = 1) %>%
  mutate(most_common_flip = paste0(est_best, " -> ", true_best)) %>%
  select(transfer_direction, most_common_flip)

flip_summary <- flip_summary %>%
  left_join(flip_patterns, by = "transfer_direction")

cat("\nFlip summary:\n")
print(flip_summary)

# =========================================================================
# Save outputs
# =========================================================================
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(transfer_summary, file.path(output_dir, "sim_portability.csv"))
cat("\nSaved:", file.path(output_dir, "sim_portability.csv"), "\n")

write_csv(flip_summary, file.path(output_dir, "sim_portability_flip.csv"))
cat("Saved:", file.path(output_dir, "sim_portability_flip.csv"), "\n")

# =========================================================================
# Figures
# =========================================================================
fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

direction_labels <- c(
  "A_to_A" = "A (same)",
  "A_to_B" = "A -> B (cross)",
  "B_to_A" = "B -> A (cross)",
  "B_to_B" = "B (same)"
)

# Figure 1: Transfer prediction error by scenario
p_error <- transfer_summary %>%
  filter(transfer_direction %in% c("A_to_B", "B_to_A", "A_to_A", "B_to_B")) %>%
  mutate(
    direction_label = direction_labels[transfer_direction],
    scenario = fct_reorder(scenario, mean_rel_error)
  ) %>%
  ggplot(aes(x = mean_rel_error, y = scenario, fill = direction_label)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_x_continuous(labels = scales::percent) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "D-Study Transfer Prediction Error",
    subtitle = sprintf("Relative |predicted - actual| / actual (N_SIM=%d)", N_SIM),
    x = "Mean relative error",
    y = "D-study scenario",
    fill = "Transfer direction"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "sim_portability_error.pdf"), p_error, width = 9, height = 6)
ggsave(file.path(fig_dir, "sim_portability_error.png"), p_error, width = 9, height = 6, dpi = 300)
cat("Saved: sim_portability_error.pdf/png\n")

# Figure 2: Rank positions — estimated vs true
# Show per-scenario ranking under each transfer
rank_plot_data <- transfer_df %>%
  filter(transfer_direction %in% c("A_to_B", "B_to_A")) %>%
  group_by(transfer_direction, scenario) %>%
  summarize(
    mean_predicted = mean(predicted_var, na.rm = TRUE),
    mean_actual = mean(actual_var, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(transfer_direction) %>%
  mutate(
    est_rank = rank(mean_predicted),
    true_rank = rank(mean_actual)
  ) %>%
  ungroup()

p_ranks <- rank_plot_data %>%
  pivot_longer(cols = c(est_rank, true_rank), names_to = "rank_type", values_to = "rank_val") %>%
  mutate(
    rank_label = ifelse(rank_type == "est_rank", "Estimated", "True"),
    direction_label = direction_labels[transfer_direction]
  ) %>%
  ggplot(aes(x = rank_val, y = scenario, color = rank_label, shape = rank_label)) +
  geom_point(size = 3) +
  geom_line(aes(group = interaction(scenario, transfer_direction)),
            color = "gray70", linewidth = 0.4) +
  facet_wrap(~direction_label) +
  scale_color_manual(values = c("Estimated" = "steelblue", "True" = "firebrick")) +
  labs(
    title = "D-Study Scenario Rankings: Estimated vs. True",
    subtitle = sprintf("Cross-model transfer (N_SIM=%d)", N_SIM),
    x = "Rank (lower = less variance = better)",
    y = "Scenario",
    color = "Ranking",
    shape = "Ranking"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "sim_portability_ranks.pdf"), p_ranks, width = 10, height = 5)
ggsave(file.path(fig_dir, "sim_portability_ranks.png"), p_ranks, width = 10, height = 5, dpi = 300)
cat("Saved: sim_portability_ranks.pdf/png\n")

# =========================================================================
# Convergence and singularity diagnostics
# =========================================================================
cat("\n--- Convergence / Singularity Diagnostics ---\n")

diag_04e <- transfer_df %>%
  filter(transfer_direction %in% c("A_to_A", "B_to_B")) %>%
  group_by(transfer_direction) %>%
  summarize(
    n_sims = n_distinct(sim_id),
    convergence_rate = n_sims / N_SIM,
    .groups = "drop"
  )

cat("\nConvergence summary:\n")
print(diag_04e)

write_csv(diag_04e, file.path(output_dir, "sim_diagnostics_04e.csv"))
cat("Saved: sim_diagnostics_04e.csv\n")

cat("\n=== 04e_sim_portability.R complete ===\n")
