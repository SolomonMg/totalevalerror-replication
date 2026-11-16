#!/usr/bin/env Rscript
# 32_arena_figures.R
# Generate figures for the Arena scoring-pipeline demonstration.

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

theme_tle <- theme_minimal(base_size = 11) + theme(
  panel.grid.minor = element_blank(),
  strip.text = element_text(face = "bold"),
  plot.title = element_text(face = "bold"),
  legend.position = "bottom"
)

CATEGORY_ORDER <- c("creative_writing", "persuasion", "factual_qa", "coding")
CATEGORY_LABELS <- c(
  creative_writing = "Creative Writing",
  persuasion       = "Persuasion",
  factual_qa       = "Factual QA",
  coding           = "Coding"
)

CONFIG_LABELS <- c(
  single_likert    = "Single-judge Likert",
  single_pairwise  = "Single-judge pairwise",
  tee_likert       = "TEE Likert (3 judges x 5 prompts)",
  tee_pairwise     = "TEE pairwise (3 judges x 5 prompts x 2 orders)"
)
CONFIG_COLORS <- c(
  "Single-judge Likert"                             = "#9ecae1",
  "Single-judge pairwise"                           = "#fdbb84",
  "TEE Likert (3 judges x 5 prompts)"               = "#2166ac",
  "TEE pairwise (3 judges x 5 prompts x 2 orders)"  = "#b2182b"
)

summary_df <- read_csv("data/processed/arena_agreement_summary.csv",
                       show_col_types = FALSE) %>%
  filter(category_llm %in% CATEGORY_ORDER) %>%
  mutate(
    category = factor(CATEGORY_LABELS[category_llm], levels = CATEGORY_LABELS[CATEGORY_ORDER]),
    config_label = factor(CONFIG_LABELS[config], levels = CONFIG_LABELS[c(
      "single_likert", "single_pairwise", "tee_likert", "tee_pairwise"
    )])
  )

# ---- Figure 1: agreement accuracy by (config x category) ----
p1 <- ggplot(summary_df, aes(x = acc_ab, y = config_label, fill = config_label)) +
  geom_col(width = 0.7, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * acc_ab)),
            hjust = -0.1, size = 3) +
  facet_wrap(~ category, ncol = 1, strip.position = "top") +
  scale_fill_manual(values = CONFIG_COLORS, guide = "none") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1), expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Scoring pipeline agreement with Arena human preference",
    subtitle = "% of decisive human-labeled battles where pipeline prediction matches",
    x = "Agreement accuracy (AB-only battles)", y = NULL
  ) + theme_tle

ggsave("figures/fig_arena_pipeline_accuracy.pdf", p1, width = 9, height = 8)
ggsave("figures/fig_arena_pipeline_accuracy.png", p1, width = 9, height = 8, dpi = 300)
cat("Saved fig_arena_pipeline_accuracy.pdf/png\n")

# ---- Figure 2: improvement (TEE - single) per category ----
imp_df <- read_csv("data/processed/arena_agreement_improvement.csv",
                   show_col_types = FALSE) %>%
  filter(category_llm %in% CATEGORY_ORDER) %>%
  mutate(category = factor(CATEGORY_LABELS[category_llm],
                             levels = CATEGORY_LABELS[CATEGORY_ORDER])) %>%
  pivot_longer(c(improvement_likert, improvement_pairwise),
               names_to = "method", values_to = "improvement") %>%
  mutate(method = factor(
    recode(method,
           improvement_likert = "Likert (TEE - single)",
           improvement_pairwise = "Pairwise (TEE - single)"),
    levels = c("Likert (TEE - single)", "Pairwise (TEE - single)")
  ))

p2 <- ggplot(imp_df, aes(x = category, y = improvement, fill = method)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.85) +
  geom_hline(yintercept = 0, color = "gray60") +
  geom_text(aes(label = sprintf("%+.1f", 100 * improvement)),
            position = position_dodge(width = 0.7), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("Likert (TEE - single)" = "#2166ac",
                                "Pairwise (TEE - single)" = "#b2182b"),
                    name = NULL) +
  scale_y_continuous(labels = scales::label_number(accuracy = 0.01,
                                                    scale = 100, suffix = " pp")) +
  labs(
    title = "TEE pipeline agreement lift over single-judge baselines",
    subtitle = "Percentage-point improvement in agreement with Arena human preference",
    x = NULL, y = "Accuracy improvement (pp)"
  ) + theme_tle

ggsave("figures/fig_arena_improvement.pdf", p2, width = 9, height = 5)
ggsave("figures/fig_arena_improvement.png", p2, width = 9, height = 5, dpi = 300)
cat("Saved fig_arena_improvement.pdf/png\n")

# ---- Figure 3: projected vs observed improvement (if dstudy available) ----
if (file.exists("data/processed/arena_dstudy_vs_observed.csv")) {
  ds <- read_csv("data/processed/arena_dstudy_vs_observed.csv", show_col_types = FALSE)
  if ("improvement_likert" %in% names(ds) && nrow(ds) >= 3) {
    ds <- ds %>% filter(category %in% CATEGORY_ORDER) %>%
      mutate(category = factor(CATEGORY_LABELS[category],
                                 levels = CATEGORY_LABELS[CATEGORY_ORDER]))

    p3 <- ggplot(ds, aes(x = se_reduction, y = improvement_likert, label = category)) +
      geom_point(size = 4, color = "#2166ac") +
      geom_text(hjust = -0.15, size = 3.5) +
      geom_smooth(method = "lm", se = FALSE, linetype = "dashed", color = "gray60") +
      scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                         expand = expansion(mult = c(0.05, 0.25))) +
      scale_y_continuous(labels = scales::label_number(accuracy = 0.01,
                                                        scale = 100, suffix = " pp")) +
      labs(
        title = "D-study projection predicts observed TEE lift",
        subtitle = "Dev-fit SE reduction (x) vs. test-set Likert accuracy lift (y)",
        x = "Projected single->TEE SE reduction (dev)",
        y = "Observed agreement lift (test)"
      ) + theme_tle

    ggsave("figures/fig_arena_dstudy_vs_observed.pdf", p3, width = 8, height = 5.5)
    ggsave("figures/fig_arena_dstudy_vs_observed.png", p3, width = 8, height = 5.5, dpi = 300)
    cat("Saved fig_arena_dstudy_vs_observed.pdf/png\n")
  }
}
