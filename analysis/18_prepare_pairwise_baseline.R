#!/usr/bin/env Rscript
# 18_prepare_pairwise_baseline.R
# Write ALL 150 pairwise items to the baseline input CSV for the
# Sonnet-agent run. Output feeds the Claude Code Agent which produces
# 300 judgments (150 items x 2 orderings).

suppressPackageStartupMessages(library(tidyverse))

set.seed(42)

items <- read_csv("data/items_pairwise.csv", show_col_types = FALSE)

cat("=== items_pairwise.csv ===\n")
cat(sprintf("  Total items: %d\n", nrow(items)))
cat("  Per dimension:\n")
print(items %>% count(category))

out <- items %>%
  arrange(category, item_id) %>%
  select(item_id, category, response_a, response_b, true_order)

cat(sprintf("\n  true_order distribution: "))
print(out %>% count(true_order))

out_path <- "data/processed/pairwise_baseline_input.csv"
write_csv(out, out_path)
cat(sprintf("\nWrote %s (%d rows)\n", out_path, nrow(out)))
