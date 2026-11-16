# 01c_mmlu_parse_audit.R
# Draw a stratified audit sample of MMLU responses scored by the strict parser
# (analysis/lib_parse_mmlu.R) for manual checking (SI si:mmlu_parsing).
#
# Sample (seed 42) from data/processed/mmlu_clean_v5.csv: 20 responses per rule for
# R1, R2, R3 (all if fewer), every R3b, R3c, and R4 response up to 20, and 40 unanswered.
# Writes data/processed/mmlu_parse_audit.csv with blank columns audit_letter and auditor;
# the auditor fills audit_letter with the letter the response commits to (NA if none).
# The script never overwrites a filled audit.
#
# Usage: Rscript analysis/01c_mmlu_parse_audit.R

suppressPackageStartupMessages(library(tidyverse))
set.seed(42)

out_path <- "data/processed/mmlu_parse_audit.csv"
if (file.exists(out_path) && any(!is.na(read_csv(out_path, show_col_types = FALSE)$audit_letter))) {
  stop("Audit already filled in ", out_path, "; delete it deliberately to redraw.", call. = FALSE)
}

d <- read_csv("data/processed/mmlu_clean_v5.csv", show_col_types = FALSE) %>%
  mutate(rule = coalesce(parse_rule, "none"), row_id = row_number())
cat("Responses by rule:\n"); print(count(d, rule))

take <- c(R1 = 20, R2 = 20, R3 = 20, R3b = 20, R3c = 20, R4 = 20, none = 40)
audit <- d %>%
  group_by(rule) %>%
  group_modify(~ slice_sample(.x, n = min(nrow(.x), take[[.y$rule]]))) %>%
  ungroup() %>%
  transmute(row_id, rule, item_id, variant_id, sut_short, temperature, replication,
            parsed_letter = answer_extracted, correct_answer, response_raw,
            audit_letter = NA_character_, auditor = NA_character_)
write_csv(audit, out_path)
cat(sprintf("\nWrote %d rows to %s\n", nrow(audit), out_path))
