#!/usr/bin/env Rscript
# 21_judges_vs_sonnet_baseline.R
# Treat Sonnet's 150-item baseline as ground truth. For items where Sonnet has
# a clear preference (non-TIE), compute per-judge agreement / correlation.
# Tells us how much of each judge's "disagreement" is real content signal
# (aligning with Sonnet) vs position-bias noise (uncorrelated with Sonnet).

suppressPackageStartupMessages(library(tidyverse))

# Sonnet per-item preferences (model_a_wins recoded, averaged over both orderings)
sonnet <- read_csv("data/processed/pairwise_baseline_sonnet_per_item.csv",
                   show_col_types = FALSE)

# Existing 3-judge data
judges <- read_csv("data/processed/pairwise_clean.csv", show_col_types = FALSE) %>%
  filter(!is.na(outcome), !is.na(true_order)) %>%
  mutate(model_a_wins = ifelse(true_order == "original", outcome, 1 - outcome))

cat("=== Sonnet baseline (n=150 items, 300 judgments) ===\n")
cat(sprintf("  Non-TIE items: %d\n",
            sum(sonnet$n_non_tie > 0 & !is.na(sonnet$sonnet_model_a_wins))))
cat(sprintf("  Mean Sonnet model_a_wins (non-TIE only): %.3f\n",
            mean(sonnet$sonnet_model_a_wins, na.rm = TRUE)))

# Per-item Haiku/GPT-4o/Gemini mean model_a_wins
per_item <- judges %>%
  group_by(item_id, judge_model) %>%
  summarize(judge_model_a_wins = mean(model_a_wins, na.rm = TRUE),
            n_obs              = n(),
            .groups = "drop")

# Join to Sonnet
cmp <- per_item %>%
  inner_join(sonnet %>% select(item_id, sonnet_model_a_wins),
             by = "item_id")

# --- Correlation per judge, on items where Sonnet is non-TIE ---
cat("\n--- Correlation with Sonnet (on Sonnet non-TIE items only) ---\n")
non_tie_items <- sonnet %>% filter(!is.na(sonnet_model_a_wins)) %>% pull(item_id)

corr_by_judge <- cmp %>%
  filter(item_id %in% non_tie_items) %>%
  group_by(judge_model) %>%
  summarize(
    n_items   = n(),
    cor_pearson  = cor(judge_model_a_wins, sonnet_model_a_wins),
    cor_spearman = cor(judge_model_a_wins, sonnet_model_a_wins, method = "spearman"),
    .groups = "drop"
  )
print(corr_by_judge)

# --- Agreement rate: does judge's mean cross 0.5 in same direction as Sonnet? ---
cat("\n--- Directional agreement with Sonnet (non-TIE items) ---\n")
agree <- cmp %>%
  filter(item_id %in% non_tie_items) %>%
  mutate(
    sonnet_a_won  = sonnet_model_a_wins > 0.5,
    judge_a_won   = judge_model_a_wins  > 0.5,
    agrees        = sonnet_a_won == judge_a_won
  )

agree_by_judge <- agree %>%
  group_by(judge_model) %>%
  summarize(
    n_items        = n(),
    n_agree        = sum(agrees),
    agreement_rate = mean(agrees),
    .groups = "drop"
  )
print(agree_by_judge)

cat("\n  (Chance agreement = 50%. Lower = anti-aligned; higher = real signal.)\n")

# --- Correlation over ALL 150 items (Sonnet TIE -> 0.5 target) ---
# If Sonnet says TIE, target is 0.5. Judges that say "0.5" on those items are
# aligned; judges that deviate strongly are either content-disagreeing or
# noise-making.
cat("\n--- Correlation with Sonnet over ALL 150 items (TIE -> 0.5) ---\n")
sonnet_all <- sonnet %>%
  mutate(sonnet_a_target = ifelse(is.na(sonnet_model_a_wins), 0.5, sonnet_model_a_wins))

cmp_all <- per_item %>%
  inner_join(sonnet_all %>% select(item_id, sonnet_a_target),
             by = "item_id")

corr_all <- cmp_all %>%
  group_by(judge_model) %>%
  summarize(
    n_items  = n(),
    cor_full = cor(judge_model_a_wins, sonnet_a_target),
    .groups = "drop"
  )
print(corr_all)

# Save
write_csv(agree, "data/processed/judges_vs_sonnet_items.csv")
write_csv(corr_by_judge, "data/processed/judges_vs_sonnet_correlation.csv")

cat("\nWrote:\n")
cat("  data/processed/judges_vs_sonnet_items.csv\n")
cat("  data/processed/judges_vs_sonnet_correlation.csv\n")
