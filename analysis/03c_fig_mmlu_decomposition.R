# 03c_fig_mmlu_decomposition.R
# MMLU G-study figure for the main text (fig:mmlu_combined panel a): each component's
# contribution to Var(theta-hat) at the operational design, as a lollipop. Twin of
# 03b_fig_intro_safety.R (Figure 3a); MMLU has no judge, so the model facet is the SUT.
# Uses the four admissible prompt variants (v_3 excluded by the structural check).
# Reads from pre-computed variance_components_mmlu.csv.
#
# Output: figures/fig_mmlu_decomposition.{pdf,png}
#
# Usage:
#   Rscript analysis/03c_fig_mmlu_decomposition.R

library(tidyverse)

set.seed(42)

fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 03c_fig_mmlu_decomposition.R ===\n")

# Drawn at print size (half-width panel, 2.7 in) so text prints at ~7 pt; the
# caption carries the title and design, so title/subtitle are blank.
theme_tle <- theme_minimal(base_size = 7.5) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 10),
    strip.background = element_rect(fill = "grey95", color = NA)
  )

N_items <- 200

# Map tle_label to short labels with sigma^2 notation (MMLU: SUT in place of judge)
sigma_labels <- c(
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

vc_raw <- read_csv("data/processed/variance_components_mmlu.csv",
                   show_col_types = FALSE) %>%
  filter(variance > 1e-8) %>%
  select(tle_label, variance)

# Single design: the operational factorial actually run on MMLU (four admissible variants).
designs <- tibble(
  V = 4, M = 3, H = 3, R = 8,
  design_label = "Operational factorial (V=4, M=3, H=3, R=8)"
)

# For each design, compute contribution of each component to Var(θ̂)
decomp <- designs %>%
  rowwise() %>%
  mutate(rows = list(
    vc_raw %>%
      mutate(divisor = sapply(tle_label, compute_divisor,
                              N = N_items, V = V, M = M, H = H, R = R)) %>%
      mutate(contribution = variance / divisor)
  )) %>%
  unnest(rows) %>%
  ungroup() %>%
  group_by(design_label) %>%
  mutate(
    total_var = sum(contribution),
    pct_mean_var = 100 * contribution / total_var,
    se_design = sqrt(total_var)
  ) %>%
  ungroup() %>%
  filter(contribution > 0)

cat("\nVar(theta_hat) decomposition by design:\n")
decomp %>%
  group_by(design_label) %>%
  arrange(desc(contribution), .by_group = TRUE) %>%
  mutate(rank = row_number()) %>%
  filter(rank <= 5) %>%
  select(design_label, tle_label, variance, divisor, contribution, pct_mean_var) %>%
  print(n = Inf)

cat("\nSE by design:\n")
decomp %>% distinct(design_label, total_var, se_design) %>% print()

# Order panels by status quo → TEE → full
decomp <- decomp %>%
  mutate(tle_label = fct_reorder(tle_label, contribution))

p <- ggplot(decomp, aes(x = contribution, y = tle_label)) +
  geom_segment(aes(x = 0, xend = contribution, y = tle_label, yend = tle_label),
               color = "#2166ac", linewidth = 0.5) +
  geom_point(color = "#2166ac", size = 1.6) +
  geom_text(aes(label = ifelse(pct_mean_var >= 0.5,
                               sprintf("%.1f%%", pct_mean_var),
                               sprintf("<0.5%%"))),
            hjust = -0.15, size = 2.4) +
  scale_y_discrete(labels = sigma_labels) +
  scale_x_continuous(labels = scales::label_number(), breaks = scales::breaks_pretty(n = 3)) +
  labs(
    title = "MMLU accuracy: Var(theta_hat) decomposition",
    subtitle = sprintf("Operational design: N=%d items, V=%d prompts, M=%d SUTs, H=%d temps, R=%d reps",
                       N_items, designs$V, designs$M, designs$H, designs$R),
    x = "Contribution to Var(theta_hat)",
    y = NULL
  ) +
  theme_tle +
  theme(plot.margin = margin(2, 2, 2, 2, "pt")) +
  expand_limits(x = max(decomp$contribution) * 1.2)

# 2.6 in tall to align with the stacked budget panel (12b) in the same figure.
ggsave(file.path(fig_dir, "fig_mmlu_decomposition.pdf"), p,
       width = 2.7, height = 2.6)
ggsave(file.path(fig_dir, "fig_mmlu_decomposition.png"), p,
       width = 2.7, height = 2.6, dpi = 300)
cat("Saved: fig_mmlu_decomposition.pdf/png\n")
