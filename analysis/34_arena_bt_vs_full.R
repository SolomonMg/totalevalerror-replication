#!/usr/bin/env Rscript
# 34_arena_bt_vs_full.R
# Compare each pipeline-derived BT leaderboard (fit on the 1,500 scored
# battles) against a 'gold' BT leaderboard fit on all 46,304 eligible
# single-turn English battles from the full 100k dataset.
#
# The 1,500-battle human reconstruction is itself a noisy target. Comparing
# pipeline BT to the full-100k BT asks a cleaner question: does the pipeline
# recover the underlying human ranking on this population, or just the
# 1,500-battle resample of it?
#
# Outputs:
#   data/processed/arena_bt_vs_full_corr.csv
#   figures/fig_arena_bt_vs_full.pdf

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

full <- read_csv("data/processed/arena_full_bt.csv", show_col_types = FALSE)
pipe <- read_csv("data/processed/arena_bt_rankings.csv", show_col_types = FALSE) %>%
  pivot_wider(names_from = config, values_from = bt_strength)

cat(sprintf("Full-100k BT models:  %d\n", nrow(full)))
cat(sprintf("Pipeline BT models:   %d\n", nrow(pipe)))

j <- pipe %>% inner_join(full, by = "model")
cat(sprintf("Joined models:        %d\n\n", nrow(j)))

configs <- c("human", "pred_single_likert", "pred_single_pairwise",
             "pred_tee_likert", "pred_tee_pairwise")
labels <- c(
  human                  = "Human (1,500 battles)",
  pred_single_likert     = "Single-judge Likert",
  pred_single_pairwise   = "Single-judge pairwise",
  pred_tee_likert        = "TEE Likert",
  pred_tee_pairwise      = "TEE pairwise"
)

corr_tbl <- map_dfr(configs, function(c) {
  if (!(c %in% names(j))) return(NULL)
  sub <- j %>% filter(!is.na(.data[[c]]), !is.na(bt_strength_full))
  tibble(
    config   = c,
    label    = labels[[c]],
    n_models = nrow(sub),
    spearman = cor(sub[[c]], sub$bt_strength_full, method = "spearman"),
    kendall  = cor(sub[[c]], sub$bt_strength_full, method = "kendall"),
    pearson  = cor(sub[[c]], sub$bt_strength_full, method = "pearson")
  )
})

cat("=== Rank correlation vs full-100k BT (gold) ===\n")
print(corr_tbl %>% mutate(across(where(is.numeric), ~ round(., 3))))

write_csv(corr_tbl, "data/processed/arena_bt_vs_full_corr.csv")

# ---- Figure: scatter of each config vs full-100k BT ----
plot_df <- j %>%
  pivot_longer(cols = any_of(configs), names_to = "config", values_to = "bt") %>%
  filter(!is.na(bt)) %>%
  mutate(label = recode(config, !!!labels))

corr_text <- corr_tbl %>%
  mutate(text = sprintf("Spearman = %.2f\nn = %d", spearman, n_models))

p <- ggplot(plot_df, aes(x = bt_strength_full, y = bt)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  geom_point(alpha = 0.7, size = 2) +
  geom_text(data = corr_text, aes(x = -Inf, y = Inf, label = text),
            hjust = -0.1, vjust = 1.3, size = 3.2, inherit.aes = FALSE) +
  facet_wrap(~ label, ncol = 3) +
  labs(
    title = "Pipeline BT vs full-100k human BT",
    subtitle = sprintf("Gold BT fit on 46,304 eligible battles; pipeline BT fit on 1,500. n = %d models shared.",
                       nrow(j)),
    x = "Full-100k BT strength (gold)",
    y = "Leaderboard BT strength"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

ggsave("figures/fig_arena_bt_vs_full.pdf", p, width = 10, height = 7)
ggsave("figures/fig_arena_bt_vs_full.png", p, width = 10, height = 7, dpi = 300)

cat("\nSaved:\n")
cat("  data/processed/arena_bt_vs_full_corr.csv\n")
cat("  figures/fig_arena_bt_vs_full.{pdf,png}\n")
