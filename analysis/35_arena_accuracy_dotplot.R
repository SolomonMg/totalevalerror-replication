#!/usr/bin/env Rscript
# 35_arena_accuracy_dotplot.R
# Horizontal dotplot with 95% Wilson CIs of AB-only agreement between
# scoring pipelines and Arena human preference, faceted vertically by
# task category. Shorter y-axis labels than fig_arena_pipeline_accuracy.pdf.

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

CATEGORY_ORDER <- c("creative_writing", "persuasion", "factual_qa", "coding")
CATEGORY_LABELS <- c(
  creative_writing = "Creative Writing",
  persuasion       = "Persuasion",
  factual_qa       = "Factual QA",
  coding           = "Coding"
)

CONFIG_ORDER <- c("single_likert", "tee_likert", "single_pairwise", "tee_pairwise")
CONFIG_SHORT <- c(
  single_likert    = "Likert (single)",
  tee_likert       = "Likert (TEE)",
  single_pairwise  = "Pairwise (single)",
  tee_pairwise     = "Pairwise (TEE)"
)

df <- read_csv("data/processed/arena_agreement_summary.csv",
               show_col_types = FALSE) %>%
  filter(category_llm %in% CATEGORY_ORDER) %>%
  mutate(
    category = factor(CATEGORY_LABELS[category_llm],
                      levels = CATEGORY_LABELS[CATEGORY_ORDER]),
    config_short = factor(CONFIG_SHORT[config],
                          levels = rev(CONFIG_SHORT[CONFIG_ORDER]))
  )

p <- ggplot(df, aes(x = acc_ab, y = config_short)) +
  geom_vline(xintercept = 0.5, linetype = "dotted", color = "gray75") +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.0f%%", 100 * acc_ab)),
            hjust = -0.35, vjust = 0.5, size = 3) +
  facet_wrap(~ category, ncol = 1, strip.position = "top") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0.45, 0.95),
                     breaks = seq(0.5, 0.9, 0.1),
                     expand = expansion(mult = c(0.02, 0.08))) +
  labs(
    title = "Scoring pipeline agreement with Arena human preference",
    subtitle = "AB-only battles. Dotted line: chance.",
    x = "Agreement with human vote",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(size = 10)
  )

ggsave("figures/fig_arena_pipeline_accuracy_dot.pdf", p, width = 7, height = 7)
ggsave("figures/fig_arena_pipeline_accuracy_dot.png", p, width = 7, height = 7, dpi = 300)

cat("Saved figures/fig_arena_pipeline_accuracy_dot.{pdf,png}\n\n")
print(df %>% select(config, category_llm, n_battles_ab, acc_ab) %>%
        mutate(across(where(is.numeric), ~ round(., 3))))
