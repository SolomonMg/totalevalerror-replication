# 03_figures.R
# Generate publication-quality figures for TEE variance decomposition.
# Produces figures comparing Likert vs Pairwise scoring methods.
# Output: figures/ (PDF + PNG)
#
# Forest plots show shares of Var(theta_hat) — the analyst's mean SE
# decomposition — under the operational factorial. Each component sigma^2_k
# is divided by its design-study denominator (e.g., sigma^2_alpha/N,
# sigma^2_lambda/M, sigma^2_rho/(N*V*H*M*R)) so shares within a panel sum
# to 100% of that design's Var(theta_hat).

library(tidyverse)
library(scales)
library(patchwork)

set.seed(42)

# --- Config ---
fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 03_figures.R ===\n")

# --- Theme ---
theme_tle <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 9),
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 10)
  )

# Color palette: blue = model-side, red = pipeline-side
tier_colors <- c("Tier 1" = "#2166ac", "Tier 2" = "#b2182b")
scoring_colors <- c("Likert (1-5)" = "#2166ac", "Pairwise (A/B)" = "#d6604d")

# --- sigma^2 expression labels for the y-axis ---
sigma_labels <- c(
  "judge model (design sensitivity)" = expression(sigma[lambda]^2 ~ (judge ~ model)),
  "item x judge"                     = expression(sigma[alpha*lambda]^2 ~ (item %*% judge)),
  "prompt x judge"                   = expression(sigma[phi*lambda]^2 ~ (prompt %*% judge)),
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
# factor (level = 1) leaves it out. P = number of position levels (pairwise
# only); position is fixed-effect sensitivity, divisor = P.
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
    "cell-level (3-way+)"              = N * V * H * M,
    "replicate noise"                  = N * V * H * M * R,
    "judge model (design sensitivity)" = M,
    "temperature (design sensitivity)" = H,
    "position (design sensitivity)"    = P,
    NA_real_
  )
}

# Apply the Var(theta_hat) decomposition to a variance-components table.
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

# --- Load data ---
vc_likert   <- read_csv("data/processed/variance_components_likert.csv", show_col_types = FALSE)
vc_pairwise <- read_csv("data/processed/variance_components_pairwise.csv", show_col_types = FALSE)

ds_likert   <- read_csv("data/processed/dstudy_likert.csv", show_col_types = FALSE)
ds_pairwise <- read_csv("data/processed/dstudy_pairwise.csv", show_col_types = FALSE)

tv_likert   <- read_csv("data/processed/per_category_variance_likert.csv", show_col_types = FALSE)
tv_pairwise <- read_csv("data/processed/per_category_variance_pairwise.csv", show_col_types = FALSE)

# --- Operational factorials (matched to data/processed/*_clean.csv) ---
likert_design   <- list(N = 150, V = 5, M = 3, H = 3, R = 3, P = 1)
pairwise_design <- list(N = 150, V = 5, M = 3, H = 3, R = 3, P = 2)

decomp_likert   <- apply_var_theta_decomp(vc_likert,   likert_design)
decomp_pairwise <- apply_var_theta_decomp(vc_pairwise, pairwise_design)


# =========================================================================
# Figure 1a: Var(theta_hat) Forest Plot — Likert
# =========================================================================
cat("\nFigure 1a: Var(theta_hat) decomposition (Likert)...\n")

# Lollipop with sigma^2 expressions on the y-axis and Var(theta_hat) % labels.
make_var_theta_plot <- function(decomp, design, title_suffix, point_color = "#2166ac") {
  d <- decomp %>%
    mutate(tier_label = recode(tier,
                               "model-side"    = "Tier 1",
                               "pipeline-side" = "Tier 2"),
           tle_label  = fct_reorder(tle_label, contribution))

  ggplot(d, aes(x = contribution, y = tle_label)) +
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
      title = paste("Var(theta_hat) Decomposition", title_suffix),
      subtitle = sprintf("Operational design: N=%d, V=%d, M=%d, H=%d, R=%d%s",
                        design$N, design$V, design$M, design$H, design$R,
                        if (design$P > 1) sprintf(", P=%d", design$P) else ""),
      x = "Contribution to Var(theta_hat)",
      y = NULL
    ) +
    theme_tle +
    expand_limits(x = max(d$contribution) * 1.25)
}

cat("\nLikert Var(theta_hat) shares (top 8):\n")
print(decomp_likert %>%
        arrange(desc(contribution)) %>%
        select(tle_label, variance, divisor, contribution, pct_mean_var) %>%
        head(8))
cat(sprintf("Likert: total Var(theta_hat) = %.6g, SE = %.6g\n",
            unique(decomp_likert$total_var), unique(decomp_likert$se_design)))

p1a <- make_var_theta_plot(decomp_likert, likert_design, "(Likert 1-5)")
ggsave(file.path(fig_dir, "fig1a_variance_likert.pdf"), p1a, width = 7, height = 5)
ggsave(file.path(fig_dir, "fig1a_variance_likert.png"), p1a, width = 7, height = 5, dpi = 300)
cat("  Saved fig1a_variance_likert.pdf/png\n")


# =========================================================================
# Figure 1b: Var(theta_hat) Forest Plot — Pairwise
# =========================================================================
cat("\nFigure 1b: Var(theta_hat) decomposition (Pairwise)...\n")

cat("\nPairwise Var(theta_hat) shares (top 8):\n")
print(decomp_pairwise %>%
        arrange(desc(contribution)) %>%
        select(tle_label, variance, divisor, contribution, pct_mean_var) %>%
        head(8))
cat(sprintf("Pairwise: total Var(theta_hat) = %.6g, SE = %.6g\n",
            unique(decomp_pairwise$total_var), unique(decomp_pairwise$se_design)))

p1b <- make_var_theta_plot(decomp_pairwise, pairwise_design, "(Pairwise A/B)")
ggsave(file.path(fig_dir, "fig1b_variance_pairwise.pdf"), p1b, width = 7, height = 5)
ggsave(file.path(fig_dir, "fig1b_variance_pairwise.png"), p1b, width = 7, height = 5, dpi = 300)
cat("  Saved fig1b_variance_pairwise.pdf/png\n")


# =========================================================================
# Figure 2: Side-by-Side Comparison — Likert vs Pairwise (Var(theta_hat) shares)
# =========================================================================
cat("Figure 2: Scoring method comparison (Var(theta_hat) shares)...\n")

vc_both <- bind_rows(decomp_likert, decomp_pairwise) %>%
  mutate(
    scoring_label = case_when(
      scoring == "likert"   ~ "Likert (1-5)",
      scoring == "pairwise" ~ "Pairwise (A/B)"
    ),
    tle_label = fct_reorder(tle_label, pct_mean_var, .fun = max)
  )

p2 <- ggplot(vc_both, aes(x = pct_mean_var, y = tle_label, fill = scoring_label)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.85) +
  scale_fill_manual(values = scoring_colors, name = "Scoring Method") +
  scale_y_discrete(labels = sigma_labels) +
  labs(
    title = "Var(theta_hat) Profiles: Likert vs Pairwise",
    subtitle = "Same domain, different scoring method - a Tier 2 design choice",
    x = "% of Var(theta_hat)",
    y = NULL
  ) +
  theme_tle

ggsave(file.path(fig_dir, "fig2_scoring_comparison.pdf"), p2, width = 8, height = 5.5)
ggsave(file.path(fig_dir, "fig2_scoring_comparison.png"), p2, width = 8, height = 5.5, dpi = 300)
cat("  Saved fig2_scoring_comparison.pdf/png\n")


# =========================================================================
# Figure 3a: D-Study Waterfall — Likert
# =========================================================================
cat("Figure 3a: D-study waterfall (Likert)...\n")

make_waterfall_plot <- function(dstudy_df, title_suffix) {
  baseline_var <- dstudy_df %>%
    filter(str_detect(scenario, "Baseline.*avg")) %>%
    pull(total_var) %>%
    first()

  ds_plot <- dstudy_df %>%
    mutate(scenario = fct_reorder(scenario, total_var, .desc = TRUE))

  ds_plot <- ds_plot %>%
    mutate(change_label = case_when(
      is.na(reduction_pct) ~ "",
      abs(reduction_pct) < 0.5 ~ "~0% change",
      reduction_pct > 0 ~ paste0(reduction_pct, "% reduction"),
      reduction_pct < 0 ~ paste0(abs(reduction_pct), "% increase")
    ))

  ggplot(ds_plot, aes(x = total_var, y = scenario)) +
    geom_col(fill = "#4393c3", alpha = 0.8) +
    geom_text(aes(label = change_label),
              hjust = -0.1, size = 3) +
    geom_vline(xintercept = baseline_var, linetype = "dashed", color = "gray50") +
    labs(
      title = paste("D-Study Projections", title_suffix),
      subtitle = "Dashed line = baseline; labels show % variance reduction",
      x = "Total Variance of Mean Estimate",
      y = NULL
    ) +
    scale_x_continuous(labels = scales::label_number()) +
    theme_tle +
    expand_limits(x = max(ds_plot$total_var) * 1.5)
}

p3a <- make_waterfall_plot(ds_likert, "(Likert)")
ggsave(file.path(fig_dir, "fig3a_dstudy_likert.pdf"), p3a, width = 8, height = 5)
ggsave(file.path(fig_dir, "fig3a_dstudy_likert.png"), p3a, width = 8, height = 5, dpi = 300)
cat("  Saved fig3a_dstudy_likert.pdf/png\n")


# =========================================================================
# Figure 3b: D-Study Waterfall — Pairwise
# =========================================================================
cat("Figure 3b: D-study waterfall (Pairwise)...\n")

p3b <- make_waterfall_plot(ds_pairwise, "(Pairwise)")
ggsave(file.path(fig_dir, "fig3b_dstudy_pairwise.pdf"), p3b, width = 8, height = 5)
ggsave(file.path(fig_dir, "fig3b_dstudy_pairwise.png"), p3b, width = 8, height = 5, dpi = 300)
cat("  Saved fig3b_dstudy_pairwise.pdf/png\n")


# =========================================================================
# Figure 3c: Combined D-Study Waterfall — Likert vs Pairwise
# =========================================================================
cat("Figure 3c: Combined D-study comparison...\n")

p3a_combined <- make_waterfall_plot(ds_likert, "(Likert)") + theme(plot.title = element_text(size = 11, face = "bold"))
p3b_combined <- make_waterfall_plot(ds_pairwise, "(Pairwise)") + theme(plot.title = element_text(size = 11, face = "bold"))
p3c <- p3a_combined + p3b_combined + plot_layout(ncol = 2)

ggsave(file.path(fig_dir, "fig_dstudy_comparison.pdf"), p3c, width = 14, height = 5.5)
ggsave(file.path(fig_dir, "fig_dstudy_comparison.png"), p3c, width = 14, height = 5.5, dpi = 300)
cat("  Saved fig_dstudy_comparison.pdf/png\n")


# =========================================================================
# Figure 4: Per-Category Replicate Noise by Scoring Method
# =========================================================================
cat("Figure 4: Per-category replicate noise...\n")

tv_both <- bind_rows(tv_likert, tv_pairwise) %>%
  mutate(
    scoring_label = case_when(
      scoring == "likert" ~ "Likert (1-5)",
      scoring == "pairwise" ~ "Pairwise (A/B)"
    ),
    category_short = str_replace_all(category, "_", " ") %>%
      str_to_title() %>%
      str_replace("Left Right", "L-R")
  )

if (nrow(tv_both) > 2) {
  p4 <- ggplot(tv_both, aes(x = mean_gen_var, y = category_short, color = scoring_label)) +
    geom_point(size = 3, position = position_dodge(width = 0.4)) +
    geom_errorbarh(
      aes(xmin = pmax(0, mean_gen_var - 1.96 * sd_gen_var / sqrt(n_cells)),
          xmax = mean_gen_var + 1.96 * sd_gen_var / sqrt(n_cells)),
      height = 0.2,
      position = position_dodge(width = 0.4)
    ) +
    scale_color_manual(values = scoring_colors, name = "Scoring Method") +
    labs(
      title = expression(paste("Per-Category Replicate Noise (", sigma[rho]^2, ")")),
      subtitle = "Mean within-cell variance by dimension; error bars = 95% CI",
      x = expression(hat(sigma)[rho]^2),
      y = NULL
    ) +
    theme_tle

  ggsave(file.path(fig_dir, "fig4_per_category_gen_var.pdf"), p4,
         width = 8, height = max(4, n_distinct(tv_both$category) * 0.5 + 2))
  ggsave(file.path(fig_dir, "fig4_per_category_gen_var.png"), p4,
         width = 8, height = max(4, n_distinct(tv_both$category) * 0.5 + 2), dpi = 300)
  cat("  Saved fig4_per_category_gen_var.pdf/png\n")
} else {
  cat("  Skipped: insufficient categories\n")
}

cat("\n=== All figures saved to", fig_dir, "===\n")
