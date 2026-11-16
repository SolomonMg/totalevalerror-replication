# 14_propaganda_plots.R
# Publication-quality 4-panel validation figure for propaganda pilot.
# Reads from pre-computed CSVs (no expensive recomputation).
#
# Panels:
#   (A) Aggregation ladder — accuracy improves as TEE-recommended factors are averaged
#   (B) TEE flags unreliable items — accuracy by TEE-estimated reliability stratum
#   (C) TEE-predicted reliability predicts accuracy — no human labels needed
#   (D) Pipeline config spread — 18pp range across 45 configs
#
# Usage:
#   Rscript analysis/14_propaganda_plots.R

library(tidyverse)
library(patchwork)

set.seed(42)

cat("=== 14_propaganda_plots.R ===\n")
cat("Generating propaganda validation figure (4 panels)\n\n")

fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# --- Load data ---
items   <- read_csv("data/processed/propaganda_pilot_item_comparison.csv",
                     show_col_types = FALSE)
configs <- read_csv("data/processed/propaganda_pilot_config_accuracy.csv",
                     show_col_types = FALSE)
clean   <- read_csv("data/processed/propaganda_pilot_clean.csv",
                     show_col_types = FALSE)
human   <- read_csv("data/processed/human_tee_item_disagreement.csv",
                     show_col_types = FALSE)

cat("  Items:", nrow(items), " Configs:", nrow(configs),
    " Clean rows:", nrow(clean), "\n\n")

# --- Shared aesthetics ---
theme_tle <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 11),
    plot.subtitle    = element_text(color = "gray40", size = 8.5),
    legend.position  = "none",
    strip.text       = element_text(face = "bold", size = 10)
  )

# --- Human ground truth (needed by multiple panels) ---
human_gt <- human %>%
  filter(item_id %in% unique(clean$question_ID)) %>%
  transmute(
    question_ID = item_id,
    human_gt = as.integer(mean_fav > 0.5),
    human_gt = ifelse(mean_fav == 0.5 & consensus == "English favored", 0L, human_gt),
    human_mean = mean_fav
  )

# --- Per-item accuracy (fraction of 45 configs that classify correctly) ---
item_acc <- clean %>%
  group_by(question_ID, variant_id, judge_model, temperature) %>%
  summarise(
    config_vote = as.integer(mean(cn_more_favorable, na.rm = TRUE) > 0.5),
    .groups = "drop"
  ) %>%
  left_join(human_gt, by = "question_ID") %>%
  group_by(question_ID) %>%
  summarise(
    accuracy  = mean(config_vote == human_gt, na.rm = TRUE),
    n_configs = n(),
    .groups   = "drop"
  )

# Merge LLM disagreement (TEE's reliability estimate)
item_acc <- item_acc %>%
  left_join(items %>% select(question_ID, llm_disagree, llm_mean), by = "question_ID")


# ===================================================================
# Panel (A): Aggregation ladder
# ===================================================================
cat("--- Panel A: Aggregation ladder ---\n")

# Helper: compute accuracy at a given aggregation level
# Each level groups by progressively fewer factors before majority-voting
acc_at_level <- function(data, group_vars, human_gt_df) {
  data %>%
    group_by(across(all_of(c("question_ID", group_vars)))) %>%
    summarise(llm_vote = as.integer(mean(cn_more_favorable, na.rm = TRUE) > 0.5),
              .groups = "drop") %>%
    left_join(human_gt_df, by = "question_ID") %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(accuracy = mean(llm_vote == human_gt, na.rm = TRUE),
              .groups = "drop")
}

# Level 1: Single config (1 prompt, 1 judge, 1 temp, majority of reps)
l1 <- acc_at_level(clean, c("variant_id", "judge_model", "temperature"), human_gt)

# Level 2: Average over prompts (5 prompts, per judge x temp)
l2 <- acc_at_level(clean, c("judge_model", "temperature"), human_gt)

# Level 3: Average over prompts + judges (per temp)
l3 <- acc_at_level(clean, c("temperature"), human_gt)

# Level 4: Average over everything (TEE-optimal)
l4 <- acc_at_level(clean, character(0), human_gt)

ladder <- bind_rows(
  l1 %>% mutate(level = "1 prompt, 1 judge, 1 temp"),
  l2 %>% mutate(level = "+ 5 prompts"),
  l3 %>% mutate(level = "+ 3 judges"),
  l4 %>% mutate(level = "TEE optimal (all averaged)")
) %>%
  mutate(level = factor(level,
    levels = rev(c("1 prompt, 1 judge, 1 temp", "+ 5 prompts",
                   "+ 3 judges", "TEE optimal (all averaged)"))))

ladder_summary <- ladder %>%
  group_by(level) %>%
  summarise(mean_acc = mean(accuracy), sd_acc = sd(accuracy),
            n = n(), .groups = "drop")

cat("  Aggregation ladder:\n")
for (i in 1:nrow(ladder_summary)) {
  cat(sprintf("    %s: %.1f%% (n=%d configs)\n",
              str_replace_all(ladder_summary$level[i], "\n", " "),
              100 * ladder_summary$mean_acc[i],
              ladder_summary$n[i]))
}

p_a <- ggplot(ladder, aes(x = accuracy, y = level)) +
  geom_jitter(height = 0.15, width = 0, size = 1.8, alpha = 0.4,
              color = "gray50") +
  geom_point(data = ladder_summary, aes(x = mean_acc, y = level),
             inherit.aes = FALSE, size = 3.5, color = "black") +
  scale_x_continuous(labels = scales::percent_format(),
                     breaks = seq(0.6, 0.9, 0.05)) +
  labs(title = "(A) Aggregation improves accuracy",
       subtitle = "Each step averages over an additional pipeline factor",
       x = "Accuracy vs. human majority vote", y = NULL) +
  theme_tle

cat("\n")


# ===================================================================
# Panel (B): TEE flags unreliable items
# ===================================================================
cat("--- Panel B: TEE flags unreliable items ---\n")

# Split items into reliability tertiles by LLM cross-config SD
item_flagged <- item_acc %>%
  mutate(
    tee_stratum = cut(llm_disagree,
                      breaks = quantile(llm_disagree, c(0, 1/3, 2/3, 1),
                                        na.rm = TRUE),
                      labels = c("Low variance\n(reliable)", "Medium", "High variance\n(flagged)"),
                      include.lowest = TRUE)
  )

strat_stats <- item_flagged %>%
  group_by(tee_stratum) %>%
  summarise(
    mean_acc = mean(accuracy, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

cat("  Accuracy by TEE reliability stratum:\n")
for (i in 1:nrow(strat_stats)) {
  cat(sprintf("    %s: %.1f%% (n=%d)\n",
              str_replace_all(strat_stats$tee_stratum[i], "\n", " "),
              100 * strat_stats$mean_acc[i],
              strat_stats$n[i]))
}

p_b <- ggplot(item_flagged, aes(x = accuracy, y = tee_stratum)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray60") +
  geom_jitter(height = 0.15, width = 0, size = 1.8, alpha = 0.4,
              color = "gray50") +
  geom_point(data = strat_stats, aes(x = mean_acc, y = tee_stratum),
             inherit.aes = FALSE, size = 3.5, color = "black") +
  scale_x_continuous(labels = scales::percent_format()) +
  annotate("text", x = 0.52, y = 0.6, label = "Chance", hjust = 0,
           size = 2.8, color = "gray50", fontface = "italic") +
  labs(title = "(B) TEE identifies unreliable items",
       subtitle = "Items split into tertiles by cross-config SD",
       x = "Accuracy vs. human majority vote", y = NULL) +
  theme_tle

cat("\n")


# ===================================================================
# Panel (C): TEE-predicted reliability predicts accuracy
# ===================================================================
cat("--- Panel C: TEE reliability vs accuracy (no human labels) ---\n")

cor_tee <- cor(item_acc$llm_disagree, item_acc$accuracy,
               use = "complete.obs")
cat("  Correlation (LLM disagree vs accuracy): r =",
    round(cor_tee, 3), "\n\n")

p_c <- ggplot(item_acc, aes(x = llm_disagree, y = accuracy)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray60") +
  geom_smooth(method = "loess", se = TRUE, color = "gray40",
              fill = "gray85", linewidth = 0.6, span = 0.8) +
  geom_point(size = 2.5, alpha = 0.5, color = "gray30") +
  scale_y_continuous(labels = scales::percent_format()) +
  annotate("text", x = 0.02, y = 0.52, label = "Chance", hjust = 0,
           size = 2.8, color = "gray50", fontface = "italic") +
  annotate("text", x = 0.35, y = 0.95,
           label = paste0("italic(r) == ", sprintf("%.2f", cor_tee)),
           parse = TRUE, hjust = 0, size = 3.5, color = "gray30") +
  labs(title = "(C) TEE reliability predicts accuracy",
       subtitle = "Cross-config variance predicts classification accuracy",
       x = "LLM cross-config disagreement (SD)",
       y = "Accuracy (fraction of configs correct)") +
  theme_tle


# ===================================================================
# Panel (D): Pipeline choice creates accuracy spread
# ===================================================================
cat("--- Panel D: Pipeline config spread ---\n")

configs <- configs %>%
  mutate(
    judge_label = case_when(
      str_detect(judge_model, "claude")  ~ "Claude Haiku",
      str_detect(judge_model, "gemini")  ~ "Gemini Flash",
      str_detect(judge_model, "gpt")     ~ "GPT-4o",
      TRUE ~ judge_model
    ),
    temperature = factor(temperature)
  )

tee_accuracy <- {
  tee_votes <- clean %>%
    group_by(question_ID) %>%
    summarise(llm_vote = as.integer(mean(cn_more_favorable, na.rm = TRUE) > 0.5),
              .groups = "drop") %>%
    left_join(human_gt, by = "question_ID")
  mean(tee_votes$llm_vote == tee_votes$human_gt, na.rm = TRUE)
}

naive_mean <- mean(configs$accuracy)
acc_range  <- max(configs$accuracy) - min(configs$accuracy)

pct_worse <- mean(configs$accuracy < tee_accuracy)

cat("  TEE-optimal accuracy:", sprintf("%.1f%%", 100 * tee_accuracy), "\n")
cat("  Config accuracy range:", sprintf("%.0f pp", 100 * acc_range), "\n")
cat("  Configs worse than TEE:", sprintf("%.0f%%", 100 * pct_worse), "\n\n")

p_d <- ggplot(configs, aes(x = accuracy, y = judge_label)) +
  # Shade region worse than TEE
  annotate("rect",
           xmin = -Inf, xmax = tee_accuracy,
           ymin = -Inf, ymax = Inf,
           fill = "firebrick", alpha = 0.07) +
  geom_jitter(aes(color = temperature), height = 0.15,
              size = 2.5, alpha = 0.7, width = 0) +
  geom_vline(xintercept = tee_accuracy, linetype = "solid",
             color = "black", linewidth = 0.5) +
  annotate("text", x = tee_accuracy + 0.003, y = 3.4,
           label = sprintf("TEE optimal (%.0f%%)", 100 * tee_accuracy),
           hjust = 0, size = 2.8, fontface = "bold") +
  annotate("text",
           x = (min(configs$accuracy) + tee_accuracy) / 2, y = 0.6,
           label = sprintf("%.0f%% of configs\nperform worse", 100 * pct_worse),
           size = 2.8, color = "firebrick4", fontface = "italic") +
  scale_x_continuous(labels = scales::percent_format(),
                     breaks = seq(0.6, 0.9, 0.04)) +
  scale_color_manual(
    values = c("0" = "#4575b4", "0.7" = "#fee090", "1" = "#d73027"),
    name = "Temperature"
  ) +
  labs(title = sprintf("(D) Pipeline choice creates a %d pp accuracy spread",
                       round(100 * acc_range)),
       subtitle = sprintf("%d configs (5 prompts \u00d7 3 judges \u00d7 3 temps)",
                          nrow(configs)),
       x = "Accuracy vs. human majority vote", y = NULL) +
  theme_tle +
  theme(legend.position = "bottom")


# ===================================================================
# Save summary statistics for traceability
# ===================================================================
propaganda_summary <- tibble(
  metric = c("single_config_mean_acc", "prompt_averaged_acc", "tee_optimal_acc",
             "acc_range_pp", "correlation_disagree_vs_acc", "pct_configs_worse_than_tee",
             "n_items", "n_configs", "n_clean_rows",
             "low_var_acc", "medium_var_acc", "high_var_acc"),
  value = c(naive_mean, ladder_summary$mean_acc[ladder_summary$level == "+ 5 prompts"],
            tee_accuracy, acc_range, cor_tee, pct_worse,
            nrow(item_acc), nrow(configs), nrow(clean),
            strat_stats$mean_acc[1], strat_stats$mean_acc[2], strat_stats$mean_acc[3])
)
write_csv(propaganda_summary, "data/processed/propaganda_validation_summary.csv")
cat("Saved: data/processed/propaganda_validation_summary.csv\n\n")

# ===================================================================
# Combine into 4-panel figure
# ===================================================================
p_combined <- (p_a + p_b) / (p_c + p_d) +
  plot_annotation(
    theme = theme(plot.margin = margin(5, 5, 5, 5))
  )

ggsave(file.path(fig_dir, "propaganda_validation.pdf"),
       p_combined, width = 14, height = 11)
ggsave(file.path(fig_dir, "propaganda_validation.png"),
       p_combined, width = 14, height = 11, dpi = 200)

cat("Saved: figures/propaganda_validation.pdf\n")
cat("Saved: figures/propaganda_validation.png\n")
cat("=== Done ===\n")


p_c <- ggplot(item_acc, aes(x = llm_disagree, y = accuracy)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray60") +
  geom_smooth(method = "loess", se = TRUE, color = "gray40",
              fill = "gray85", alpha = 0.5, linewidth = 0.6, span = 0.8) +
  geom_point(size = 2.5, alpha = 0.25, color = "gray30") +
  scale_y_continuous(labels = scales::percent_format()) +
  annotate("text", x = 0.02, y = 0.52, label = "Chance", hjust = 0,
           size = 2.8, color = "gray50", fontface = "italic") +
  annotate("text", x = 0.35, y = 0.95,
           label = paste0("italic(r) == ", sprintf("%.2f", cor_tee)),
           parse = TRUE, hjust = 0, size = 3.5, color = "gray30") +
  labs(title = "TEE reliability predicts accuracy",
       subtitle = "Variance predicts classification accuracy",
       x = "LLM cross-config disagreement (SD)",
       y = "Accuracy (fraction of configs correct)") +
  theme_tle
ggsave(file.path(fig_dir, "propaganda_validation_c_only.pdf"),
       p_c, width = 5, height = 4)









