# 03_figures_mmlu.R
# Generate figures for the MMLU variance decomposition.
# Output: figures/mmlu_variance_forest.pdf, figures/mmlu_per_category_gen_var.pdf
#
# The forest plot shows shares of Var(theta_hat) — the analyst's mean SE
# decomposition — under MMLU's operational factorial. Each component
# sigma^2_k is divided by its design-study denominator (e.g.,
# sigma^2_alpha/N, sigma^2_phi_M/M, sigma^2_rho/(N*V*H*M*R)) so shares
# within the panel sum to 100% of Var(theta_hat). MMLU has a SUT layer
# (system-under-test) but no judge layer; SUT plays the role of M in the
# divisors.

library(tidyverse)
library(scales)

set.seed(42)

fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 03_figures_mmlu.R ===\n")

# --- Theme ---
theme_tle <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 9),
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 10)
  )

tier_colors <- c("Tier 1" = "#2166ac", "Tier 2" = "#b2182b")

# --- sigma^2 expression labels for the y-axis (MMLU uses SUT in place of judge) ---
sigma_labels_mmlu <- c(
  "SUT model (design sensitivity)"   = expression(sigma[lambda]^2 ~ (SUT ~ model)),
  "item x SUT"                       = expression(sigma[alpha*lambda]^2 ~ (item %*% SUT)),
  "prompt x SUT"                     = expression(sigma[phi*lambda]^2 ~ (prompt %*% SUT)),
  "within-category item"             = expression(sigma[delta]^2 ~ (within-cat. ~ item)),
  "between-category"                 = expression(sigma[kappa]^2 ~ (between-cat.)),
  "item x prompt"                    = expression(sigma[alpha*phi]^2 ~ (item %*% prompt)),
  "cell-level (3-way+)"              = expression(sigma[epsilon]^2 ~ (cell-level)),
  "temperature (design sensitivity)" = expression(sigma[tau]^2 ~ (temperature)),
  "item x temperature"               = expression(sigma[alpha*tau]^2 ~ (item %*% temp)),
  "prompt x temperature"             = expression(sigma[phi*tau]^2 ~ (prompt %*% temp)),
  "replicate noise"                  = expression(sigma[rho]^2 ~ (replicate)),
  "prompt"                           = expression(sigma[phi]^2 ~ (prompt))
)

# Compute the divisor for each component under a design (V, M, H, R).
# For MMLU, M = number of SUT models. Convention: averaging over a factor
# adds it to the denominator; fixing a factor (level = 1) leaves it out.
compute_divisor <- function(label, N, V, M, H, R) {
  switch(label,
    "between-category"                 = N,
    "within-category item"             = N,
    "prompt"                           = V,
    "item x prompt"                    = N * V,
    "item x temperature"               = N * H,
    "prompt x temperature"             = V * H,
    "item x SUT"                       = N * M,
    "prompt x SUT"                     = V * M,
    "cell-level (3-way+)"              = N * V * H * M,
    "replicate noise"                  = N * V * H * M * R,
    "SUT model (design sensitivity)"   = M,
    "temperature (design sensitivity)" = H,
    NA_real_
  )
}

# --- Load MMLU data ---
vc_mmlu <- read_csv("data/processed/variance_components_mmlu.csv", show_col_types = FALSE)
tv_mmlu <- read_csv("data/processed/per_category_variance_mmlu.csv", show_col_types = FALSE)

# Map tier labels: model-side -> Tier 1, pipeline-side -> Tier 2
vc_mmlu <- vc_mmlu %>%
  mutate(tier_label = case_when(
    tier == "model-side"    ~ "Tier 1",
    tier == "pipeline-side" ~ "Tier 2",
    TRUE                    ~ tier
  ))

# --- Operational factorial (matched to data/processed/mmlu_clean.csv) ---
mmlu_design <- list(N = 200, V = 4, M = 3, H = 3, R = 8)   # v_3 excluded (structural check)


# =========================================================================
# Figure: Var(theta_hat) Forest Plot — MMLU
# =========================================================================
cat("\nFigure: Var(theta_hat) decomposition (MMLU)...\n")

decomp_mmlu <- vc_mmlu %>%
  filter(variance > 0) %>%
  mutate(
    divisor = sapply(tle_label, compute_divisor,
                     N = mmlu_design$N, V = mmlu_design$V,
                     M = mmlu_design$M, H = mmlu_design$H, R = mmlu_design$R),
    contribution = variance / divisor
  ) %>%
  filter(!is.na(divisor), contribution > 0) %>%
  mutate(
    total_var    = sum(contribution),
    pct_mean_var = 100 * contribution / total_var,
    se_design    = sqrt(total_var)
  )

cat("\nMMLU Var(theta_hat) shares (top 8):\n")
print(decomp_mmlu %>%
        arrange(desc(contribution)) %>%
        select(tle_label, variance, divisor, contribution, pct_mean_var) %>%
        head(8))
cat(sprintf("MMLU: total Var(theta_hat) = %.6g, SE = %.6g\n",
            unique(decomp_mmlu$total_var), unique(decomp_mmlu$se_design)))

decomp_plot <- decomp_mmlu %>%
  mutate(tle_label = fct_reorder(tle_label, contribution))

p_forest <- ggplot(decomp_plot, aes(x = contribution, y = tle_label)) +
  geom_point(aes(color = tier_label), size = 3) +
  geom_segment(aes(x = 0, xend = contribution,
                   y = tle_label, yend = tle_label,
                   color = tier_label), linewidth = 0.5) +
  geom_text(aes(label = ifelse(pct_mean_var >= 0.5,
                               sprintf("%.1f%%", pct_mean_var),
                               "<0.5%")),
            hjust = -0.15, size = 2.9) +
  scale_color_manual(values = tier_colors, name = "Tier") +
  scale_y_discrete(labels = sigma_labels_mmlu) +
  labs(
    title = "Var(theta_hat) Decomposition (MMLU: Accuracy)",
    subtitle = sprintf("Operational design: N=%d items, V=%d prompts, M=%d SUTs, H=%d temps, R=%d reps",
                       mmlu_design$N, mmlu_design$V, mmlu_design$M,
                       mmlu_design$H, mmlu_design$R),
    x = "Contribution to Var(theta_hat)",
    y = NULL
  ) +
  theme_tle +
  expand_limits(x = max(decomp_plot$contribution) * 1.25)

ggsave(file.path(fig_dir, "mmlu_variance_forest.pdf"), p_forest, width = 7, height = 5)
ggsave(file.path(fig_dir, "mmlu_variance_forest.png"), p_forest, width = 7, height = 5, dpi = 300)
cat("  Saved mmlu_variance_forest.pdf/png\n")


# =========================================================================
# Figure: Per-Category Replicate Noise — MMLU
# =========================================================================
cat("\nFigure: Per-category replicate noise (MMLU)...\n")

# Clean up category names for display
tv_plot <- tv_mmlu %>%
  mutate(
    cat_label = str_replace_all(category, "_", " ") %>% str_to_title(),
    cat_label = fct_reorder(cat_label, mean_gen_var)
  )

p_catvar <- ggplot(tv_plot, aes(x = mean_gen_var, y = cat_label)) +
  geom_point(size = 3, color = "#2166ac") +
  geom_errorbarh(
    aes(xmin = pmax(0, mean_gen_var - 1.96 * sd_gen_var / sqrt(n_cells)),
        xmax = mean_gen_var + 1.96 * sd_gen_var / sqrt(n_cells)),
    height = 0.2, color = "#2166ac"
  ) +
  labs(
    title = expression(paste("Per-Category Replicate Noise (", sigma[rho]^2, ") - MMLU")),
    subtitle = "Mean within-cell variance by broad category; error bars = 95% CI",
    x = expression(hat(sigma)[rho]^2),
    y = NULL
  ) +
  theme_tle

ggsave(file.path(fig_dir, "mmlu_per_category_gen_var.pdf"), p_catvar,
       width = 7, height = 4)
ggsave(file.path(fig_dir, "mmlu_per_category_gen_var.png"), p_catvar,
       width = 7, height = 4, dpi = 300)
cat("  Saved mmlu_per_category_gen_var.pdf/png\n")


cat("\n=== All MMLU figures saved to", fig_dir, "===\n")
