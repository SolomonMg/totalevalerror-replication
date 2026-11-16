#!/usr/bin/env Rscript
# 35b_arena_accuracy_dotplot_overall.R
# Pooled (all categories combined) AB-only accuracy dotplot, one panel,
# parallel to fig_arena_pipeline_accuracy_dot.pdf but flattened across
# categories. Single dot per pipeline.

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

CONFIG_LABEL <- c(
  single_likert    = "Likert",
  tee_likert       = "Likert",
  single_pairwise  = "Pairwise",
  tee_pairwise     = "Pairwise"
)
CONFIG_GROUP <- c(
  single_likert    = "Single judge",
  tee_likert       = "TEE",
  single_pairwise  = "Single judge",
  tee_pairwise     = "TEE"
)

# TEE rows use the matched decisive rate (script 30): TEE calls a winner on as many human-decisive battles as the
# single-judge pipeline, so both are scored on the same number of battles. Single-judge rows equal acc_ab.
df <- read_csv("data/processed/arena_agreement_matched.csv", show_col_types = FALSE) %>%
  filter(category_llm == "Pooled") %>%
  select(scoring, n_called, single = acc_single, tee = acc_tee_matched) %>%
  pivot_longer(c(single, tee), names_to = "agg", values_to = "acc_ab") %>%
  mutate(config = paste0(agg, "_", scoring)) %>%
  mutate(scoring_method = factor(CONFIG_LABEL[config],
                                 levels = c("Pairwise", "Likert")),
         aggregation    = factor(CONFIG_GROUP[config],
                                 levels = c("Single judge", "TEE")))

cat("Pooled AB-only accuracies, matched decisive rate (4,676 battles):\n")
print(df %>% mutate(across(where(is.numeric), ~ round(., 3))))

p <- ggplot(df, aes(x = acc_ab, y = scoring_method)) +
  geom_point(size = 4) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * acc_ab)),
            hjust = -0.30, vjust = 0.5, size = 3.8) +
  facet_wrap(~ aggregation, ncol = 1, strip.position = "left") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0.65, 0.80),
                     breaks = c(0.65, 0.70, 0.75, 0.80),
                     expand = expansion(mult = c(0.02, 0.10))) +
  labs(
    x = "Agreement with Arena human vote\n(AB-only; both pipelines call the same number of battles)",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    strip.placement = "outside",
    strip.text.y.left = element_text(face = "bold", angle = 0, size = 11),
    axis.text.y = element_text(size = 11),
    axis.title.x = element_text(size = 11, color = "gray20")
  )

ggsave("figures/fig_arena_pipeline_accuracy_dot_pooled.pdf", p, width = 6.5, height = 3.2)
ggsave("figures/fig_arena_pipeline_accuracy_dot_pooled.png", p, width = 6.5, height = 3.2, dpi = 300)

cat("\nSaved figures/fig_arena_pipeline_accuracy_dot_pooled.{pdf,png}\n")
