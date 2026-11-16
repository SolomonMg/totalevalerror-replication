# 01b_reparse_mmlu_recollections.R
# Re-parse the rebuttal MMLU re-collections (narrow, broad wording, broad plus) with the
# strict parser in analysis/lib_parse_mmlu.R. These CSVs were scored at collection time
# by rebuttal_neurips2026/recollect_driver.py, whose parser took the first a-d letter
# anywhere in the response ("Answer: C" -> A).
#
# Overwrites answer_extracted / outcome and adds parse_rule in:
#   data/processed/mmlu_clean_{narrow,broad_wording,broadplus}.csv
#
# Usage: Rscript analysis/01b_reparse_mmlu_recollections.R   (from project root)

suppressPackageStartupMessages(library(tidyverse))
source("analysis/lib_parse_mmlu.R")

cat("=== 01b_reparse_mmlu_recollections.R ===\n")

choices <- read_csv("data/items_mmlu.csv", show_col_types = FALSE) %>%
  transmute(item_id = as.character(item_id), choice_A, choice_B, choice_C, choice_D)

for (set in c("narrow", "broad_wording", "broadplus")) {
  path <- sprintf("data/processed/mmlu_clean_%s.csv", set)
  d <- read_csv(path, show_col_types = FALSE, col_types = cols(item_id = "c"))
  na_before <- 100 * mean(is.na(d$outcome))
  x <- d %>% select(item_id) %>% left_join(choices, by = "item_id")
  stopifnot(!anyNA(x$choice_A))
  p <- parse_mmlu_strict_df(d$response_raw, x$choice_A, x$choice_B, x$choice_C, x$choice_D)
  d <- d %>%
    mutate(answer_extracted = p$letter,
           parse_rule = p$rule,
           outcome = ifelse(is.na(answer_extracted), NA_integer_,
                            as.integer(answer_extracted == str_to_upper(correct_answer))))
  cat(sprintf("  %-14s n=%d  unanswered %.1f%% -> %.1f%%  accuracy %.1f%%\n", set, nrow(d),
              na_before, 100 * mean(is.na(d$outcome)), 100 * mean(d$outcome, na.rm = TRUE)))
  write_csv(d, path)
}
