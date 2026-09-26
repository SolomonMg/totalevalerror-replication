# 02d_mmlu_variant_accuracy.R
# Per-variant MMLU accuracy on the four admissible variants (main text sec:mmlu), plus the
# five-variant view with unanswered responses excluded or scored incorrect (SI tab:mmlu_v5).
# Output: data/processed/mmlu_variant_accuracy.csv
# Usage:  Rscript analysis/02d_mmlu_variant_accuracy.R
suppressPackageStartupMessages(library(tidyverse))
cat("=== 02d_mmlu_variant_accuracy.R ===\n")
d <- read_csv("data/processed/mmlu_clean_v5.csv", col_select = c(variant_id, outcome), show_col_types = FALSE)
adm <- read_csv("data/processed/mmlu_variant_structural_check.csv", show_col_types = FALSE) %>%
  filter(set == "main_text") %>% select(variant_id, admissible = passes)
out <- d %>% group_by(variant_id) %>%
  summarise(n_obs = n(), n_unanswered = sum(is.na(outcome)),
            pct_unanswered = 100 * mean(is.na(outcome)),
            accuracy_pct = 100 * mean(outcome, na.rm = TRUE),
            accuracy_unanswered_incorrect_pct = 100 * mean(coalesce(outcome, 0)), .groups = "drop") %>%
  left_join(adm, by = "variant_id") %>% arrange(accuracy_pct)
print(out)
a <- filter(out, admissible)
cat(sprintf("admissible variants: %.1f%% to %.1f%%\n", min(a$accuracy_pct), max(a$accuracy_pct)))
write_csv(out, "data/processed/mmlu_variant_accuracy.csv")
