# 12b_fig_budget_allocation_stacked.R
# Main-text MMLU D-study panel (fig:mmlu_combined panel b): RMSE and 95% CI coverage by
# budget for the naive, standard, and TEE-guided allocations, stacked vertically and drawn
# at print size (half-width, 2.7 in) to sit beside fig_mmlu_decomposition.pdf.
# Plot-only: reads the summary written by 12_budget_allocation_mmlu.R and reuses its
# colours and shapes.
#
# Output: figures/fig_budget_allocation_stacked.{pdf,png}
# Usage:  Rscript analysis/12b_fig_budget_allocation_stacked.R

suppressPackageStartupMessages({ library(tidyverse); library(patchwork) })

summary_df <- read_csv("data/processed/mmlu_budget_allocation_summary.csv", show_col_types = FALSE)

rcol <- c(Naive = "#d6604d", Standard = "#878787", TEE = "#2166ac")
rshp <- c(Naive = 16, Standard = 15, TEE = 17)
theme_panel <- theme_minimal(base_size = 7.5) + theme(panel.grid.minor = element_blank())

p_rmse <- ggplot(summary_df, aes(budget, rmse, color = researcher, shape = researcher)) +
  geom_line(linewidth = 0.5) + geom_point(size = 1.4) +
  scale_color_manual(values = rcol) + scale_shape_manual(values = rshp) +
  labs(x = NULL, y = "RMSE", color = NULL, shape = NULL) +
  theme_panel + theme(axis.text.x = element_blank())

p_cov <- ggplot(summary_df, aes(budget, coverage * 100, color = researcher, shape = researcher)) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 0.5) + geom_point(size = 1.4) +
  scale_color_manual(values = rcol) + scale_shape_manual(values = rshp) +
  labs(x = "Budget (API calls per SUT x temperature cell)", y = "Coverage (%)",
       color = NULL, shape = NULL) +
  theme_panel

fig <- (p_rmse / p_cov) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom", legend.margin = margin(0, 0, 0, 0),
        legend.box.spacing = unit(1, "pt"), plot.margin = margin(2, 2, 2, 2, "pt"))

ggsave("figures/fig_budget_allocation_stacked.pdf", fig, width = 2.7, height = 2.6)
ggsave("figures/fig_budget_allocation_stacked.png", fig, width = 2.7, height = 2.6, dpi = 300)
cat("Saved: figures/fig_budget_allocation_stacked.pdf/png\n")
