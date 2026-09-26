# 05_manuscript_figures.R
# Publication-quality figures for the Monte Carlo simulation appendix.
# Reads CSVs produced by 04–04f and generates 3 multi-panel figures using patchwork.

library(tidyverse)
library(patchwork)

set.seed(42)

fig_dir <- "figures"
data_dir <- "data/processed"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 05_manuscript_figures.R: Publication Figures ===\n")

# =========================================================================
# Figure 1: REML Convergence Properties (bias + RMSE vs sample size)
# =========================================================================
cat("\n--- Figure 1: REML Convergence Properties ---\n")

sim_bias_rmse <- read_csv(file.path(data_dir, "sim_bias_rmse.csv"), show_col_types = FALSE)

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

plot_data_1 <- sim_bias_rmse %>%
  filter(component != "temp_fixed") %>%
  mutate(
    component_label = tle_labels[component],
    n_total = n_items * n_prompts * 3 * n_reps
  )

# Panel A: Relative bias vs sample size
p1a <- plot_data_1 %>%
  ggplot(aes(x = n_total, y = rel_bias_pct, color = component_label)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_log10(labels = scales::comma) +
  scale_color_brewer(palette = "Set1") +
  labs(
    subtitle = "(A) Relative bias",
    x = "Total observations (log scale)",
    y = "Relative bias (%)",
    color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "none"
  )

# Panel B: RMSE vs sample size
p1b <- plot_data_1 %>%
  ggplot(aes(x = n_total, y = rmse, color = component_label)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.8) +
  scale_x_log10(labels = scales::comma) +
  scale_y_log10() +
  scale_color_brewer(palette = "Set1") +
  labs(
    subtitle = "(B) RMSE",
    x = "Total observations (log scale)",
    y = "RMSE (log scale)",
    color = "Component"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.key.size = unit(0.4, "cm"),
    legend.text = element_text(size = 8)
  )

fig1 <- p1a + p1b +
  plot_annotation(
    title = "REML Estimator Convergence Properties",
    subtitle = "Bias and RMSE across sample sizes (1,000 Monte Carlo simulations per configuration)",
    theme = theme(
      plot.title = element_text(size = 13, face = "bold"),
      plot.subtitle = element_text(size = 10)
    )
  )

ggsave(file.path(fig_dir, "manuscript_fig1_convergence.pdf"), fig1,
       width = 10, height = 4.5)
ggsave(file.path(fig_dir, "manuscript_fig1_convergence.png"), fig1,
       width = 10, height = 4.5, dpi = 300)
cat("Saved: manuscript_fig1_convergence.pdf/png\n")


# =========================================================================
# Figure 2: Small-K Guidance (bias + knife-edge directional accuracy)
# =========================================================================
cat("\n--- Figure 2: Small-K Guidance ---\n")

sim_small_k <- read_csv(file.path(data_dir, "sim_small_k.csv"), show_col_types = FALSE)

beta_labels <- c("0" = "Null (0)", "0.04" = "Moderate (0.04)", "0.1" = "Large (0.10)")

# Panel A: Relative bias of sigma2_beta vs K
p2a <- sim_small_k %>%
  filter(true_sigma2_beta > 0) %>%
  mutate(beta_label = beta_labels[as.character(true_sigma2_beta)]) %>%
  ggplot(aes(x = K, y = rel_bias_pct, color = beta_label, group = beta_label)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_brewer(palette = "Set1") +
  labs(
    subtitle = expression("(A) Relative bias of" ~ hat(sigma)[rho]^2 ~ "vs K"),
    x = "Number of prompt variants (K)",
    y = "Relative bias (%)",
    color = expression(sigma[rho]^2)
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.key.size = unit(0.4, "cm")
  )

# Panel B: Knife-edge directional accuracy
ke_file <- file.path(data_dir, "sim_small_k_knife_edge.csv")
if (file.exists(ke_file)) {
  ke_df <- read_csv(ke_file, show_col_types = FALSE)

  p2b <- ke_df %>%
    mutate(K_label = paste0("K = ", K)) %>%
    ggplot(aes(x = multiplier, y = directional_accuracy_pct,
               color = K_label, group = K_label)) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2) +
    geom_hline(yintercept = 50, linetype = "dashed", color = "gray50") +
    geom_vline(xintercept = 1.0, linetype = "dotted", color = "gray70") +
    scale_color_brewer(palette = "Set1") +
    labs(
      subtitle = "(B) Directional accuracy near decision boundary",
      x = expression("Multiplier of boundary" ~ sigma[rho]^{"2*"}),
      y = "Directionally correct (%)",
      color = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.key.size = unit(0.4, "cm")
    )

  fig2 <- p2a + p2b +
    plot_annotation(
      title = "Small-K Prompt Sensitivity: Bias and Directional Guidance",
      subtitle = "1,000 Monte Carlo replicates per condition",
      theme = theme(
        plot.title = element_text(size = 13, face = "bold"),
        plot.subtitle = element_text(size = 10)
      )
    )
} else {
  cat("  WARNING: knife-edge CSV not found, producing single-panel figure\n")
  fig2 <- p2a +
    plot_annotation(
      title = "Small-K Prompt Sensitivity: Bias",
      theme = theme(plot.title = element_text(size = 13, face = "bold"))
    )
}

ggsave(file.path(fig_dir, "manuscript_fig2_small_k.pdf"), fig2,
       width = 10, height = 4.5)
ggsave(file.path(fig_dir, "manuscript_fig2_small_k.png"), fig2,
       width = 10, height = 4.5, dpi = 300)
cat("Saved: manuscript_fig2_small_k.pdf/png\n")


# =========================================================================
# Figure 3: D-Study Robustness (Sc1 projection + misspecification)
# =========================================================================
cat("\n--- Figure 3: D-Study Robustness ---\n")

# Panel A: Intervention ranking — box plots of projected Var by target design
sc1_file <- file.path(data_dir, "sim_dstudy_sc1_per_sim.csv")
sc1_per_sim <- read_csv(sc1_file, show_col_types = FALSE)

# Short labels ordered by true projected variance (best → worst)
target_order <- sc1_per_sim %>%
  group_by(target) %>%
  summarize(true_var = first(true_projected_var), .groups = "drop") %>%
  arrange(true_var)

short_labels <- c(
  "K'=20 (5x)" = "5x prompts (V'=20)",
  "K'=12 (3x)" = "3x prompts (V'=12)",
  "K'=8 + R'=10" = "2x prompts + 2x reps",
  "K'=8 (2x)" = "2x prompts (V'=8)",
  "K'=6 (1.5x)" = "1.5x prompts (V'=6)",
  "Fix temp" = "Fix temperature",
  "R'=10 (2x)" = "2x replications (R'=10)",
  "R'=20 (4x)" = "4x replications (R'=20)"
)

sc1_plot <- sc1_per_sim %>%
  mutate(
    target_short = short_labels[target],
    target_short = factor(target_short, levels = short_labels[target_order$target])
  )

true_vals <- target_order %>%
  mutate(
    target_short = factor(short_labels[target], levels = short_labels[target_order$target])
  )

# Compute rank correlation summary for annotation
per_sim_tau <- sc1_per_sim %>%
  group_by(sim_id) %>%
  summarize(
    tau = cor(projected_var, true_projected_var, method = "kendall"),
    top1_correct = target[which.min(projected_var)] == target[which.min(true_projected_var)],
    .groups = "drop"
  )
tau_median <- median(per_sim_tau$tau)
top1_pct <- mean(per_sim_tau$top1_correct) * 100
annot_text <- sprintf("Rank tau = %.2f (median); top-1 correct %g%% of sims", tau_median, top1_pct)

# Compute baseline variance (current design: N=50, V=4, H=3, R=5, C=5, avg over temps)
baseline_var <- 0.03 / 5 + 0.05 / 50 + 0.04 / 4 + 0.02 / (50 * 4) +
  0.015 / (50 * 3) + 0.005 / (4 * 3) + 0.06 / (50 * 4 * 3 * 5)

# Convert to % change from baseline
sc1_plot <- sc1_plot %>%
  mutate(pct_change = (projected_var - baseline_var) / baseline_var * 100)
true_vals <- true_vals %>%
  mutate(pct_change = (true_var - baseline_var) / baseline_var * 100)

p3a <- sc1_plot %>%
  ggplot(aes(y = target_short, x = pct_change)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_boxplot(fill = "steelblue", alpha = 0.3, outlier.size = 0.5, outlier.alpha = 0.3) +
  geom_point(data = true_vals, aes(y = target_short, x = pct_change),
             color = "firebrick", size = 2.5, shape = 18) +
  coord_cartesian(xlim = c(-60, 20)) +
  labs(
    subtitle = "(A) Projected variance change from baseline design",
    y = "Design intervention\n(ordered by effectiveness, best at top)",
    x = expression("% change in Var(" * hat(theta) * ") vs. baseline (negative = better)")
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 8)
  )

# Panel B: Misspecification dot plot
sc2 <- read_csv(file.path(data_dir, "sim_dstudy_sc2_correlated.csv"), show_col_types = FALSE)
sc3 <- read_csv(file.path(data_dir, "sim_dstudy_sc3_nonexch.csv"), show_col_types = FALSE)
sc4 <- read_csv(file.path(data_dir, "sim_dstudy_sc4_nongaussian.csv"), show_col_types = FALSE)
sc5 <- read_csv(file.path(data_dir, "sim_dstudy_sc5_hetero.csv"), show_col_types = FALSE)

misspec <- bind_rows(
  sc2 %>%
    filter(rho %in% c(0.5, 1, 2)) %>%
    mutate(scenario = paste0("Hard items more prompt-sensitive (", rho, "x)")),
  sc3 %>%
    filter(variance_ratio %in% c(2, 4, 8)) %>%
    mutate(scenario = paste0("Prompts vary in quality (", variance_ratio, "x range)")),
  sc4 %>%
    filter(df_t %in% c(5, 3)) %>%
    mutate(scenario = paste0("Non-normal scores (df=", df_t, ")")),
  sc5 %>%
    mutate(scenario = "Some categories more prompt-sensitive")
) %>%
  mutate(
    rel_bias_pct = rel_bias * 100,
    scenario = fct_reorder(scenario, rel_bias_pct)
  )

p3b <- misspec %>%
  ggplot(aes(x = rel_bias_pct, y = scenario)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 3, color = "steelblue") +
  labs(
    subtitle = "(B) Projection bias when assumptions are violated",
    x = "Relative bias (%)",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

fig3 <- p3a + p3b +
  plot_annotation(
    title = "D-Study Projection Robustness",
    subtitle = "Intervention rankings are reliable (left); projections robust to misspecification (right)",
    theme = theme(
      plot.title = element_text(size = 13, face = "bold"),
      plot.subtitle = element_text(size = 10)
    )
  )

ggsave(file.path(fig_dir, "manuscript_fig3_dstudy.pdf"), fig3,
       width = 12, height = 4.5)
ggsave(file.path(fig_dir, "manuscript_fig3_dstudy.png"), fig3,
       width = 12, height = 4.5, dpi = 300)
cat("Saved: manuscript_fig3_dstudy.pdf/png\n")

cat("\n=== 05_manuscript_figures.R complete ===\n")
