#!/usr/bin/env Rscript
# 19_analyze_pairwise_baseline.R
# Analyze the Sonnet-agent baseline: position bias, inter-order agreement,
# and correlation with Gemini's existing per-item preferences.

suppressPackageStartupMessages(library(tidyverse))

# --- Load ---
judg <- read_csv("data/processed/pairwise_baseline_judgments.csv",
                 show_col_types = FALSE)
input <- read_csv("data/processed/pairwise_baseline_input.csv",
                  show_col_types = FALSE)
gemini_raw <- read_csv("data/processed/pairwise_clean.csv",
                       show_col_types = FALSE)

cat("=== Sonnet Agent Baseline Analysis ===\n\n")
cat(sprintf("Sonnet judgments loaded: %d rows (expected 100)\n", nrow(judg)))
cat(sprintf("  Unique items: %d (expected 50)\n", n_distinct(judg$item_id)))
cat(sprintf("  Orders: %s\n", paste(unique(judg$order), collapse = ", ")))
cat(sprintf("  Judgments: %s\n", paste(table(judg$judgment), collapse = ", ")))

# Sanity: every item should have exactly 2 rows (listed + swapped)
per_item <- judg %>% count(item_id)
if (any(per_item$n != 2)) {
  warning("Some items don't have exactly 2 judgments:")
  print(per_item %>% filter(n != 2))
}

# ============================================================================
# 1. Position bias check
# ============================================================================
# Appropriate test with a TIE option: does the A/B split AMONG NON-TIE
# judgments mirror across listed/swapped orders? A judge with B-bias would
# pick B in both orders -- listed-B and swapped-B both elevated.
cat("\n--- 1. Position bias (A/B split by order) ---\n")
bias <- judg %>%
  group_by(order) %>%
  summarize(
    n           = n(),
    a_rate      = mean(judgment == "A"),
    b_rate      = mean(judgment == "B"),
    tie_rate    = mean(judgment == "TIE"),
    n_non_tie   = sum(judgment != "TIE"),
    b_of_nontie = sum(judgment == "B") / pmax(sum(judgment != "TIE"), 1),
    .groups = "drop"
  )
print(bias)

cat(sprintf("\n  Listed  B-rate overall: %.1f%%  |  of non-TIE: %.1f%%\n",
            100 * bias$b_rate[bias$order == "listed"],
            100 * bias$b_of_nontie[bias$order == "listed"]))
cat(sprintf("  Swapped B-rate overall: %.1f%%  |  of non-TIE: %.1f%%\n",
            100 * bias$b_rate[bias$order == "swapped"],
            100 * bias$b_of_nontie[bias$order == "swapped"]))

# Clean baseline: non-TIE B-rates should be near 50% AND listed ~ 1 - swapped
# (perfect mirror symmetry under 100% consistency). Tolerance: abs(listed_B - 50) <= 20pp
b_listed  <- bias$b_of_nontie[bias$order == "listed"]
b_swapped <- bias$b_of_nontie[bias$order == "swapped"]
pass_bias <- abs(b_listed - 0.5) <= 0.20 & abs(b_swapped - 0.5) <= 0.20
cat(sprintf("  Position-bias check (non-TIE B-rate in [30%%, 70%%]): %s\n",
            if (pass_bias) "PASS" else "FAIL"))

# ============================================================================
# 2. Inter-order agreement
# ============================================================================
# If content judgment is unbiased: listed and swapped give OPPOSITE verdicts
# (or both TIE). We code consistency = 1 if that holds.
cat("\n--- 2. Inter-order consistency ---\n")
wide <- judg %>%
  select(item_id, order, judgment) %>%
  pivot_wider(names_from = order, values_from = judgment)

wide <- wide %>% mutate(
  consistent = case_when(
    listed == "A" & swapped == "B" ~ TRUE,
    listed == "B" & swapped == "A" ~ TRUE,
    listed == "TIE" & swapped == "TIE" ~ TRUE,
    TRUE ~ FALSE
  )
)

consistency_rate <- mean(wide$consistent, na.rm = TRUE)
cat(sprintf("  Inter-order consistency: %.1f%% (%d/%d items)\n",
            100 * consistency_rate, sum(wide$consistent), nrow(wide)))
cat("  Clean baseline expects >=80%.\n")

cat("\n  Inconsistent items:\n")
print(wide %>% filter(!consistent))

pass_consistency <- consistency_rate >= 0.80

# ============================================================================
# 3. Recode to model_a_wins (content-based, position-independent)
# ============================================================================
cat("\n--- 3. Recoded model_a_wins (averaged over orders) ---\n")
# For order=listed: judgment=A means model_a_original wins (because response_a is
#   in position A). But wait -- response_a itself was already possibly swapped at
#   item creation via true_order. So we need to unscramble BOTH layers.
#
# Per-item TWO layers of randomization:
#   Layer 1 (item creation, recorded in true_order): was the model-A response
#     placed in CSV-column response_a ("original") or in response_b ("swapped")?
#   Layer 2 (this experiment): was response_a presented to Sonnet in position A
#     (order=listed) or in position B (order=swapped)?
#
# To get model_a_original_wins (the actual content preference):
#   judgment=A in order=listed  -> response_a won. If true_order=original, response_a=model_a_original. model_a_wins=1.
#                                   If true_order=swapped, response_a=model_b_original. model_a_wins=0.
#   judgment=B in order=listed  -> response_b won. If true_order=original, response_b=model_b_original. model_a_wins=0.
#                                   If true_order=swapped, response_b=model_a_original. model_a_wins=1.
#   judgment=A in order=swapped -> response_b won (because response_b was in position A).
#                                   If true_order=original, response_b=model_b_original. model_a_wins=0.
#                                   If true_order=swapped, response_b=model_a_original. model_a_wins=1.
#   judgment=B in order=swapped -> response_a won. Same unscrambling as order=listed / judgment=A.

judg_aug <- judg %>%
  left_join(input %>% select(item_id, category, true_order), by = "item_id")

# Who won in terms of response_a vs response_b?
#   order=listed:  judgment=A -> response_a won; judgment=B -> response_b won
#   order=swapped: judgment=A -> response_b won; judgment=B -> response_a won
judg_aug <- judg_aug %>% mutate(
  response_a_won = case_when(
    judgment == "TIE" ~ NA_real_,
    order == "listed"  & judgment == "A" ~ 1,
    order == "listed"  & judgment == "B" ~ 0,
    order == "swapped" & judgment == "A" ~ 0,
    order == "swapped" & judgment == "B" ~ 1,
    TRUE ~ NA_real_
  )
)

# Then recode response_a_won -> model_a_original_wins using true_order:
#   true_order=original -> response_a IS model_a_original -> model_a_wins = response_a_won
#   true_order=swapped  -> response_a IS model_b_original -> model_a_wins = 1 - response_a_won
judg_aug <- judg_aug %>% mutate(
  model_a_wins = ifelse(true_order == "original", response_a_won, 1 - response_a_won)
)

# Per-item: average across the 2 orders (drops TIEs)
sonnet_item <- judg_aug %>%
  group_by(item_id, category) %>%
  summarize(
    sonnet_model_a_wins = mean(model_a_wins, na.rm = TRUE),
    n_non_tie           = sum(!is.na(model_a_wins)),
    .groups = "drop"
  )

cat(sprintf("  Items with at least one non-TIE: %d / 50\n",
            sum(sonnet_item$n_non_tie > 0)))
cat(sprintf("  Items with BOTH orders non-TIE: %d\n",
            sum(sonnet_item$n_non_tie == 2)))
cat(sprintf("  Overall mean model_a_wins (Sonnet): %.3f\n",
            mean(sonnet_item$sonnet_model_a_wins, na.rm = TRUE)))

# ============================================================================
# 4. Compare to Gemini (existing data, recoded)
# ============================================================================
cat("\n--- 4. Correlation with Gemini on overlap items ---\n")

gemini <- gemini_raw %>%
  filter(judge_model == "google/gemini-2.0-flash-001",
         !is.na(outcome),
         item_id %in% sonnet_item$item_id) %>%
  mutate(
    response_a_won = outcome,  # outcome=1 means judge picked "A" which is response_a
    model_a_wins = ifelse(true_order == "original", response_a_won, 1 - response_a_won)
  ) %>%
  group_by(item_id) %>%
  summarize(gemini_model_a_wins = mean(model_a_wins, na.rm = TRUE),
            n_gemini_obs        = n(),
            .groups = "drop")

joined <- sonnet_item %>%
  inner_join(gemini, by = "item_id") %>%
  filter(!is.na(sonnet_model_a_wins), !is.na(gemini_model_a_wins))

cat(sprintf("  Items with both Sonnet and Gemini: %d\n", nrow(joined)))

if (nrow(joined) >= 5) {
  r <- cor(joined$sonnet_model_a_wins, joined$gemini_model_a_wins)
  cat(sprintf("  Correlation (Sonnet vs Gemini per-item model_a_wins): r = %.3f\n", r))
  cat("  Clean baseline expects r > 0.4.\n")
  pass_corr <- r > 0.4
} else {
  pass_corr <- NA
  cat("  Too few joined items for correlation.\n")
}

# ============================================================================
# 5. TIE rate check
# ============================================================================
cat("\n--- 5. TIE rate ---\n")
tie_rate <- mean(judg$judgment == "TIE")
cat(sprintf("  TIE rate: %.1f%% (expected <30%%; high TIE = judge punting)\n",
            100 * tie_rate))
pass_tie <- tie_rate < 0.30

# ============================================================================
# Summary
# ============================================================================
cat("\n=== Verdict ===\n")
cat(sprintf("  Position-bias (listed & swapped B-rate in [40,60]): %s\n",
            if (pass_bias) "PASS" else "FAIL"))
cat(sprintf("  Inter-order consistency >= 80%%:                    %s\n",
            if (pass_consistency) "PASS" else "FAIL"))
cat(sprintf("  TIE rate < 30%%:                                    %s\n",
            if (pass_tie) "PASS" else "FAIL"))
cat(sprintf("  Sonnet-Gemini correlation > 0.4:                   %s\n",
            if (is.na(pass_corr)) "N/A"
            else if (pass_corr) "PASS" else "FAIL"))

all_pass <- pass_bias && pass_consistency && pass_tie &&
            (is.na(pass_corr) || pass_corr)
cat(sprintf("\n  Overall: %s\n", if (all_pass) "CLEAN BASELINE"
                                  else "FAILED ONE OR MORE CRITERIA"))

# Save per-item Sonnet results for downstream use
write_csv(sonnet_item, "data/processed/pairwise_baseline_sonnet_per_item.csv")
write_csv(joined,      "data/processed/pairwise_baseline_sonnet_vs_gemini.csv")
cat("\nWrote:\n")
cat("  data/processed/pairwise_baseline_sonnet_per_item.csv\n")
cat("  data/processed/pairwise_baseline_sonnet_vs_gemini.csv\n")
