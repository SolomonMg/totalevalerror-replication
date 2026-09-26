# 03_figures_safety.R
# Generate figures for the safety benchmark variance decomposition.
# Output: figures/safety_*.{pdf,png}
#
# Forest plots show shares of Var(theta_hat) — the analyst's mean SE
# decomposition — under the operational factorial. Each component sigma^2_k
# is divided by its design-study denominator (e.g., sigma^2_alpha/N,
# sigma^2_lambda/M, sigma^2_rho/(N*V*H*M*R)) so shares within a panel sum
# to 100% of that design's Var(theta_hat). The per-category replicate-noise
# figure shows raw per-category structure (sigma^2_rho stratified by
# hazard category) and is left in raw variance units.

library(tidyverse)
library(scales)
library(patchwork)

set.seed(42)

fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 03_figures_safety.R ===\n")

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

# --- sigma^2 expression labels for the y-axis ---
# Covers ideology (Likert/Pairwise) and safety vocabularies; MMLU's "SUT"
# variants are added separately because they relabel the M facet.
sigma_labels <- c(
  "judge model (design sensitivity)" = expression(sigma[lambda]^2 ~ (judge ~ model)),
  "item x judge"                     = expression(sigma[alpha*lambda]^2 ~ (item %*% judge)),
  "prompt x judge"                   = expression(sigma[phi*lambda]^2 ~ (prompt %*% judge)),
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
  "prompt"                           = expression(sigma[phi]^2 ~ (prompt)),
  "position (design sensitivity)"    = expression(sigma[pi]^2 ~ (position))
)

# Compute the divisor for each component under a design (V, M, H, R, P).
# Convention: averaging over a factor adds it to the denominator; fixing a
# factor (level = 1) leaves it out. M plays the role of judge_model for
# ideology/safety and SUT model for MMLU; P = number of position levels
# (pairwise only).
compute_divisor <- function(label, N, V, M, H, R, P = 1) {
  switch(label,
    "between-category"                 = N,
    "within-category item"             = N,
    "prompt"                           = V,
    "item x prompt"                    = N * V,
    "item x temperature"               = N * H,
    "prompt x temperature"             = V * H,
    "item x judge"                     = N * M,
    "prompt x judge"                   = V * M,
    "item x SUT"                       = N * M,
    "prompt x SUT"                     = V * M,
    "cell-level (3-way+)"              = N * V * H * M,
    "replicate noise"                  = N * V * H * M * R,
    "judge model (design sensitivity)" = M,
    "SUT model (design sensitivity)"   = M,
    "temperature (design sensitivity)" = H,
    "position (design sensitivity)"    = P,
    NA_real_
  )
}

apply_var_theta_decomp <- function(vc, design) {
  vc %>%
    filter(variance > 0) %>%
    mutate(
      divisor = sapply(tle_label, compute_divisor,
                       N = design$N, V = design$V, M = design$M,
                       H = design$H, R = design$R, P = design$P),
      contribution = variance / divisor
    ) %>%
    filter(!is.na(divisor), contribution > 0) %>%
    mutate(
      total_var    = sum(contribution),
      pct_mean_var = 100 * contribution / total_var,
      se_design    = sqrt(total_var)
    )
}

# --- Load safety data ---
vc_safety <- read_csv("data/processed/variance_components_safety.csv", show_col_types = FALSE)
ds_safety <- read_csv("data/processed/dstudy_safety.csv", show_col_types = FALSE)
tv_safety <- read_csv("data/processed/per_category_variance_safety.csv", show_col_types = FALSE)

# --- Operational factorial (matched to data/processed/safety_clean.csv) ---
safety_design <- list(N = 141, V = 5, M = 3, H = 3, R = 8, P = 1)


# =========================================================================
# Figure: Var(theta_hat) Forest Plot — Safety
# =========================================================================
cat("\nFigure: Var(theta_hat) decomposition (Safety)...\n")

decomp_safety <- apply_var_theta_decomp(vc_safety, safety_design)

cat("\nSafety Var(theta_hat) shares (top 8):\n")
print(decomp_safety %>%
        arrange(desc(contribution)) %>%
        select(tle_label, variance, divisor, contribution, pct_mean_var) %>%
        head(8))
cat(sprintf("Safety: total Var(theta_hat) = %.6g, SE = %.6g\n",
            unique(decomp_safety$total_var), unique(decomp_safety$se_design)))

decomp_plot <- decomp_safety %>%
  mutate(tier_label = recode(tier,
                             "model-side"    = "Tier 1",
                             "pipeline-side" = "Tier 2"),
         tle_label = fct_reorder(tle_label, contribution))

p_forest <- ggplot(decomp_plot, aes(x = contribution, y = tle_label)) +
  geom_segment(aes(x = 0, xend = contribution,
                   y = tle_label, yend = tle_label,
                   color = tier_label), linewidth = 0.5) +
  geom_point(aes(color = tier_label), size = 3) +
  geom_text(aes(label = ifelse(pct_mean_var >= 0.5,
                               sprintf("%.1f%%", pct_mean_var),
                               "<0.5%")),
            hjust = -0.15, size = 2.9) +
  scale_color_manual(values = tier_colors, name = "Tier") +
  scale_y_discrete(labels = sigma_labels) +
  labs(
    title = "Var(theta_hat) Decomposition (Safety: SAFE/UNSAFE)",
    subtitle = sprintf("Operational design: N=%d items, V=%d prompts, M=%d judges, H=%d temps, R=%d reps",
                       safety_design$N, safety_design$V, safety_design$M,
                       safety_design$H, safety_design$R),
    x = "Contribution to Var(theta_hat)",
    y = NULL
  ) +
  theme_tle +
  expand_limits(x = max(decomp_plot$contribution) * 1.25)

ggsave(file.path(fig_dir, "safety_variance_forest.pdf"), p_forest, width = 7, height = 5)
ggsave(file.path(fig_dir, "safety_variance_forest.png"), p_forest, width = 7, height = 5, dpi = 300)
cat("  Saved safety_variance_forest.pdf/png\n")


# =========================================================================
# Figure: Three-Way Comparison — Likert vs Pairwise vs Safety (Var(theta_hat) shares)
# =========================================================================
cat("\nFigure: Three-way scoring method comparison (Var(theta_hat) shares)...\n")

# Load ideology data if available
vc_likert_path <- "data/processed/variance_components_likert.csv"
vc_pairwise_path <- "data/processed/variance_components_pairwise.csv"

if (file.exists(vc_likert_path) && file.exists(vc_pairwise_path)) {
  vc_likert   <- read_csv(vc_likert_path, show_col_types = FALSE)
  vc_pairwise <- read_csv(vc_pairwise_path, show_col_types = FALSE)

  scoring_colors3 <- c(
    "Likert (1-5)" = "#2166ac",
    "Pairwise (A/B)" = "#d6604d",
    "Safety (SAFE/UNSAFE)" = "#5aae61"
  )

  # Each scoring method has its own operational design (matches *_clean.csv).
  likert_design   <- list(N = 150, V = 5, M = 3, H = 3, R = 3, P = 1)
  pairwise_design <- list(N = 150, V = 5, M = 3, H = 3, R = 3, P = 2)

  decomp_likert   <- apply_var_theta_decomp(vc_likert,   likert_design)
  decomp_pairwise <- apply_var_theta_decomp(vc_pairwise, pairwise_design)

  vc_all <- bind_rows(decomp_likert, decomp_pairwise, decomp_safety) %>%
    mutate(
      scoring_label = case_when(
        scoring == "likert"   ~ "Likert (1-5)",
        scoring == "pairwise" ~ "Pairwise (A/B)",
        scoring == "safety"   ~ "Safety (SAFE/UNSAFE)"
      ),
      tle_label = fct_reorder(tle_label, pct_mean_var, .fun = max)
    )

  cat("\nThree-way comparison: Var(theta_hat) shares by scoring method (top 5 per method):\n")
  vc_all %>%
    group_by(scoring_label) %>%
    arrange(desc(pct_mean_var), .by_group = TRUE) %>%
    slice_head(n = 5) %>%
    select(scoring_label, tle_label, contribution, pct_mean_var) %>%
    print(n = Inf)

  p_compare <- ggplot(vc_all, aes(x = pct_mean_var, y = tle_label, fill = scoring_label)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85) +
    scale_fill_manual(values = scoring_colors3, name = "Scoring Method") +
    scale_y_discrete(labels = sigma_labels) +
    labs(
      title = "Var(theta_hat) Profiles Across Scoring Methods",
      subtitle = "Same framework, different domains and scoring - a Tier 2 design choice",
      x = "% of Var(theta_hat)",
      y = NULL
    ) +
    theme_tle

  ggsave(file.path(fig_dir, "safety_three_way_comparison.pdf"), p_compare, width = 9, height = 6)
  ggsave(file.path(fig_dir, "safety_three_way_comparison.png"), p_compare, width = 9, height = 6, dpi = 300)
  cat("  Saved safety_three_way_comparison.pdf/png\n")
} else {
  cat("  Skipped: ideology data not found\n")
}


# =========================================================================
# Figure: D-Study Waterfall — Safety
# =========================================================================
cat("\nFigure: D-study waterfall (Safety)...\n")

baseline_var <- ds_safety %>%
  filter(str_detect(scenario, "Baseline.*avg")) %>%
  pull(total_var) %>%
  first()

ds_plot <- ds_safety %>%
  mutate(
    scenario = fct_reorder(scenario, total_var, .desc = TRUE),
    bar_color = ifelse(total_var > baseline_var, "#d6604d", "#4393c3")
  )

p_dstudy <- ggplot(ds_plot, aes(x = total_var, y = scenario)) +
  geom_col(aes(fill = bar_color), alpha = 0.8) +
  scale_fill_identity() +
  geom_text(aes(label = paste0(ifelse(reduction_pct > 0, "+", ""), reduction_pct, "%")),
            hjust = -0.1, size = 3) +
  geom_vline(xintercept = baseline_var, linetype = "dashed", color = "gray50") +
  labs(
    title = "D-Study Projections (Safety)",
    subtitle = "Blue = reduction from baseline (dashed); Red = increase from baseline",
    x = "Total Variance of Mean Estimate",
    y = NULL
  ) +
  theme_tle +
  expand_limits(x = max(ds_plot$total_var) * 1.5)

ggsave(file.path(fig_dir, "safety_dstudy.pdf"), p_dstudy, width = 9, height = 5.5)
ggsave(file.path(fig_dir, "safety_dstudy.png"), p_dstudy, width = 9, height = 5.5, dpi = 300)
cat("  Saved safety_dstudy.pdf/png\n")


# =========================================================================
# Figure: Per-Category Replicate Noise — Safety
# =========================================================================
cat("\nFigure: Per-category replicate noise (Safety)...\n")

# AILuminate category labels
category_labels <- c(
  "cse" = "Child Sexual Exploitation",
  "dfm" = "Defamation",
  "hte" = "Hate Speech",
  "ipv" = "Intimate Partner Violence",
  "iwp" = "Indiscriminate Weapons",
  "ncr" = "Non-Consensual Intimate",
  "prv" = "Privacy Violation",
  "src" = "Sexual Content",
  "ssh" = "Self-Harm/Suicide",
  "spc" = "Specialized Advice",
  "sxc_prn" = "Sex Crimes/Pornography",
  "vcr" = "Violent Crime"
)

tv_plot <- tv_safety %>%
  mutate(
    cat_label = ifelse(category %in% names(category_labels),
                       category_labels[category], category),
    cat_label = fct_reorder(cat_label, mean_gen_var)
  )

p_catvar <- ggplot(tv_plot, aes(x = mean_gen_var, y = cat_label)) +
  geom_point(size = 3, color = "#5aae61") +
  geom_errorbarh(
    aes(xmin = pmax(0, mean_gen_var - 1.96 * sd_gen_var / sqrt(n_cells)),
        xmax = mean_gen_var + 1.96 * sd_gen_var / sqrt(n_cells)),
    height = 0.2, color = "#5aae61"
  ) +
  labs(
    title = expression(paste("Per-Category Replicate Noise (", sigma[rho]^2, ") - Safety")),
    subtitle = "Mean within-cell variance by hazard category; error bars = 95% CI",
    x = expression(hat(sigma)[rho]^2),
    y = NULL
  ) +
  theme_tle

ggsave(file.path(fig_dir, "safety_per_category_gen_var.pdf"), p_catvar,
       width = 8, height = 6)
ggsave(file.path(fig_dir, "safety_per_category_gen_var.png"), p_catvar,
       width = 8, height = 6, dpi = 300)
cat("  Saved safety_per_category_gen_var.pdf/png\n")


# =========================================================================
# Figure: Compound Safety Figure (forest + per-category) for main text
# =========================================================================
cat("\nFigure: Compound safety (forest + per-category)...\n")

p_forest_ab <- p_forest +
  labs(title = "(a) Var(theta_hat) Decomposition (Safety: SAFE/UNSAFE)",
       subtitle = NULL) +
  theme(legend.position = "bottom")

p_catvar_ab <- p_catvar +
  labs(title = expression(paste("(b) Per-Category Replicate Noise (", sigma[rho]^2, ")")),
       subtitle = NULL)

p_safety_compound <- p_forest_ab + p_catvar_ab + plot_layout(ncol = 2, widths = c(1, 1))

ggsave(file.path(fig_dir, "fig_safety_compound.pdf"), p_safety_compound,
       width = 14, height = 5.5)
ggsave(file.path(fig_dir, "fig_safety_compound.png"), p_safety_compound,
       width = 14, height = 5.5, dpi = 300)
cat("  Saved fig_safety_compound.pdf/png\n")


cat("\n=== All safety figures saved to", fig_dir, "===\n")
