#!/usr/bin/env Rscript
# 39_pairwise_tie_threshold_sweep.R
# Sweep the tie-window half-width delta on TEE pairwise's 30-cell aggregation.
# For each delta, compute pipeline tie rate, Overall accuracy, and AB-only
# accuracy across the 4,676 Arena battles. Identify the delta that matches
# Arena's human tie rate (~35%) and report agreement at that threshold.
#
# Outputs:
#   data/processed/arena_tie_threshold_sweep.csv
#   figures/fig_arena_tie_threshold.{pdf,png}

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(patchwork)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)
set.seed(42)

# ---- Load cell-level pairwise + battle metadata ----
pw <- read_csv("data/processed/arena_pairwise_clean.csv", show_col_types = FALSE)
battles <- read_csv("data/processed/arena_battles_scored_input.csv", show_col_types = FALSE) %>%
  select(battle_id, winner)

normalize_winner <- function(w) {
  case_when(
    w == "model_a"                   ~ "a",
    w == "model_b"                   ~ "b",
    w %in% c("tie", "tie (bothbad)") ~ "tie",
    TRUE                              ~ NA_character_
  )
}

# ---- Per-battle a_win_rate over all 30 (variant x judge x order) cells ----
per_battle <- pw %>%
  filter(!is.na(model_a_wins)) %>%
  group_by(battle_id, category_llm) %>%
  summarize(a_win_rate = mean(model_a_wins),
            n_cells = n(),
            .groups = "drop") %>%
  inner_join(battles, by = "battle_id") %>%
  mutate(human = normalize_winner(winner)) %>%
  filter(!is.na(human))

cat(sprintf("Battles with per-battle aggregation: %d\n", nrow(per_battle)))
cat("Distribution of a_win_rate (deciles):\n")
print(round(quantile(per_battle$a_win_rate, probs = seq(0, 1, 0.1)), 3))
cat(sprintf("\nHuman tie rate: %.3f\n",
            mean(per_battle$human == "tie")))

# ---- Sweep tie half-width delta ----
eval_at_delta <- function(d, df) {
  pred <- case_when(
    abs(df$a_win_rate - 0.5) <= d ~ "tie",
    df$a_win_rate > 0.5            ~ "a",
    TRUE                           ~ "b"
  )
  ab_mask <- pred %in% c("a", "b") & df$human %in% c("a", "b")
  tibble(
    delta       = d,
    tie_rate    = mean(pred == "tie"),
    acc_overall = mean(pred == df$human),
    n_ab        = sum(ab_mask),
    acc_ab      = if (sum(ab_mask) > 0) mean(pred[ab_mask] == df$human[ab_mask]) else NA_real_
  )
}

sweep <- map_dfr(c(0.001, seq(0.025, 0.50, by = 0.025)), eval_at_delta, df = per_battle)

cat("\n=== Tie-threshold sweep (full data) ===\n")
print(sweep %>% mutate(across(where(is.numeric), ~ round(., 3))))

# ---- Identify threshold matching Arena's human tie rate ----
human_tie_rate <- mean(per_battle$human == "tie")
match_idx <- which.min(abs(sweep$tie_rate - human_tie_rate))
delta_match <- sweep$delta[match_idx]

cat(sprintf("\nClosest match to human tie rate %.3f: delta = %.3f -> pipeline tie rate %.3f\n",
            human_tie_rate, delta_match, sweep$tie_rate[match_idx]))
cat(sprintf("  Overall accuracy at match: %.3f\n", sweep$acc_overall[match_idx]))
cat(sprintf("  AB-only accuracy at match: %.3f (n_AB = %d)\n",
            sweep$acc_ab[match_idx], sweep$n_ab[match_idx]))
cat(sprintf("\nCurrent threshold delta = 0.05 -> tie rate %.3f, overall %.3f, AB-only %.3f\n",
            sweep$tie_rate[which.min(abs(sweep$delta - 0.05))],
            sweep$acc_overall[which.min(abs(sweep$delta - 0.05))],
            sweep$acc_ab[which.min(abs(sweep$delta - 0.05))]))

# ---- Train/test robustness check (80/20 stratified by category) ----
dev_idx <- per_battle %>% group_by(category_llm) %>%
  slice_sample(prop = 0.80) %>% ungroup() %>% pull(battle_id)
test_idx <- setdiff(per_battle$battle_id, dev_idx)
dev <- per_battle %>% filter(battle_id %in% dev_idx)
test <- per_battle %>% filter(battle_id %in% test_idx)
cat(sprintf("\nDev: %d battles | Test: %d battles\n", nrow(dev), nrow(test)))

# Find delta on dev that matches dev's human tie rate; evaluate on test
dev_tie_rate <- mean(dev$human == "tie")
sweep_dev <- map_dfr(sweep$delta, function(d) {
  pred <- case_when(abs(dev$a_win_rate - 0.5) <= d ~ "tie",
                    dev$a_win_rate > 0.5 ~ "a",
                    TRUE ~ "b")
  tibble(delta = d, tie_rate_dev = mean(pred == "tie"))
})
delta_dev_match <- sweep_dev$delta[which.min(abs(sweep_dev$tie_rate_dev - dev_tie_rate))]

# Evaluate at this delta on test
pred_test <- case_when(abs(test$a_win_rate - 0.5) <= delta_dev_match ~ "tie",
                       test$a_win_rate > 0.5 ~ "a",
                       TRUE ~ "b")
ab_mask <- pred_test %in% c("a","b") & test$human %in% c("a","b")
cat(sprintf("\n=== Train/test robustness ===\n"))
cat(sprintf("Dev-tuned delta = %.3f (matches dev human tie rate %.3f)\n",
            delta_dev_match, dev_tie_rate))
cat(sprintf("Test pipeline tie rate: %.3f (test human tie rate: %.3f)\n",
            mean(pred_test == "tie"), mean(test$human == "tie")))
cat(sprintf("Test Overall accuracy:  %.3f\n", mean(pred_test == test$human)))
cat(sprintf("Test AB-only accuracy:  %.3f (n_AB = %d)\n",
            if (sum(ab_mask) > 0) mean(pred_test[ab_mask] == test$human[ab_mask]) else NA,
            sum(ab_mask)))

write_csv(sweep, "data/processed/arena_tie_threshold_sweep.csv")
cat("\nSaved data/processed/arena_tie_threshold_sweep.csv\n")

# ---- Plot ----
plot_df <- sweep %>%
  pivot_longer(cols = c(acc_overall, acc_ab),
               names_to = "metric", values_to = "accuracy") %>%
  mutate(metric = recode(metric,
                         acc_overall = "Overall (ties counted)",
                         acc_ab      = "AB-only (decisive subset)"))

# Panel A: tie rate vs delta
p_tie <- ggplot(sweep, aes(x = delta, y = tie_rate)) +
  geom_hline(yintercept = human_tie_rate, linetype = "dashed", color = "gray40") +
  annotate("text", x = 0.45, y = human_tie_rate + 0.025,
           label = sprintf("Human tie rate %.2f", human_tie_rate),
           size = 3.4, color = "gray30", hjust = 1) +
  geom_vline(xintercept = 0.05, linetype = "dotted", color = "#D55E00") +
  annotate("text", x = 0.06, y = 0.05, label = "Current\ndelta = 0.05",
           size = 3.0, color = "#D55E00", hjust = 0) +
  geom_vline(xintercept = delta_match, linetype = "dotted", color = "#0072B2") +
  annotate("text", x = delta_match + 0.005, y = human_tie_rate - 0.045,
           label = sprintf("Match\ndelta = %.2f", delta_match),
           size = 3.0, color = "#0072B2", hjust = 0) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.6) +
  scale_x_continuous(breaks = seq(0, 0.5, 0.1),
                     labels = scales::number_format(accuracy = 0.01)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1.0), breaks = seq(0, 1, 0.2)) +
  labs(x = "Tie window half-width  delta",
       y = "Pipeline tie rate",
       subtitle = "Wider window catches more battles as ties; meets human rate at delta ~ 0.18.") +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        plot.subtitle = element_text(color = "gray30"))

# Panel B: accuracy vs pipeline tie rate (the calibration frontier)
p_frontier <- ggplot(plot_df, aes(x = tie_rate, y = accuracy, color = metric)) +
  geom_vline(xintercept = human_tie_rate, linetype = "dashed", color = "gray40") +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2) +
  geom_point(data = plot_df %>% filter(delta == 0.05),
             size = 4, color = "#D55E00", shape = 21, stroke = 1.5, fill = NA) +
  geom_point(data = plot_df %>% filter(abs(delta - delta_match) < 0.001),
             size = 4, color = "#0072B2", shape = 21, stroke = 1.5, fill = NA) +
  scale_color_manual(values = c("Overall (ties counted)"     = "#222222",
                                "AB-only (decisive subset)"  = "#888888")) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1.0), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0.0, 1.0), labels = scales::percent_format(accuracy = 1),
                     breaks = seq(0, 1, 0.2)) +
  labs(x = "Pipeline tie rate (varies with delta)",
       y = "Accuracy vs Arena human winner",
       color = NULL,
       subtitle = "Orange ring: current delta = 0.05.  Blue ring: human-matched delta. Dashed: human tie rate.") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank(),
        plot.subtitle = element_text(color = "gray30"))

p_combined <- (p_tie / p_frontier) +
  plot_annotation(
    title = "TEE pairwise tie-threshold sweep on the 4,676-battle factorial",
    subtitle = sprintf("Same 30-cell aggregation; only the post-aggregation decision rule changes. Calibrating to Arena's %.0f%% tie rate gives Overall accuracy %.2f vs current %.2f.",
                       100 * human_tie_rate,
                       sweep$acc_overall[match_idx],
                       sweep$acc_overall[which.min(abs(sweep$delta - 0.05))]),
    theme = theme(plot.title = element_text(face = "bold"),
                  plot.subtitle = element_text(color = "gray30"))
  ) +
  plot_layout(heights = c(1, 1.1))

ggsave("figures/fig_arena_tie_threshold.pdf", p_combined, width = 8, height = 9)
ggsave("figures/fig_arena_tie_threshold.png", p_combined, width = 8, height = 9, dpi = 300)
cat("\nSaved figures/fig_arena_tie_threshold.{pdf,png}\n")
