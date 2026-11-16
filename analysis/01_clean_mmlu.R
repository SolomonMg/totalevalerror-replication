# 01_clean_mmlu.R
# Load MMLU experiment JSONL, parse A/B/C/D answers, score correctness, produce clean CSV.
# This is a SUT-layer experiment (no judge): items are multiple-choice questions
# answered directly by the SUT models.
#
# Answers are parsed with the strict parser in analysis/lib_parse_mmlu.R (SI
# si:mmlu_parsing). The legacy parser is kept only to write the comparison table.
# Prompt variants that fail the deterministic structural check
# (experiments/check_mmlu_variant_structure.py; main-text v_3) are excluded from the
# primary file.
#
# Outputs:
#   data/processed/mmlu_clean.csv            admissible variants only (primary)
#   data/processed/mmlu_clean_v5.csv         all five variants (SI V=5 sensitivity)
#   data/processed/mmlu_parse_comparison.csv legacy vs strict parser, per variant
#
# Usage:
#   Rscript analysis/01_clean_mmlu.R [stage]
#   e.g.: Rscript analysis/01_clean_mmlu.R full

library(tidyverse)
library(jsonlite)
source("analysis/lib_parse_mmlu.R")

set.seed(42)

# --- Config ---
args <- commandArgs(trailingOnly = TRUE)
stage <- if (length(args) >= 1) args[1] else "pilot"

input_path  <- sprintf("data/raw/mmlu_%s.jsonl", stage)
output_path    <- "data/processed/mmlu_clean.csv"
output_v5_path <- "data/processed/mmlu_clean_v5.csv"
compare_path   <- "data/processed/mmlu_parse_comparison.csv"
items_path     <- "data/items_mmlu.csv"
structure_path <- "data/processed/mmlu_variant_structural_check.csv"

cat("=== 01_clean_mmlu.R ===\n")
cat("Stage:", stage, "\n")
cat("Input:", input_path, "\n")

# --- Load JSONL ---
lines <- readLines(input_path)
raw <- map_dfr(lines, function(line) {
  x <- fromJSON(line, flatten = TRUE)
  if (!is.null(x$usage)) {
    x$usage_prompt_tokens    <- x$usage$prompt_tokens %||% 0L
    x$usage_completion_tokens <- x$usage$completion_tokens %||% 0L
  }
  x$usage <- NULL
  x <- map(x, ~ if (is.null(.x) || length(.x) == 0) NA else .x)
  as_tibble_row(x)
})

cat("Loaded", nrow(raw), "rows\n")

# --- Legacy parser (submission version); used only for the comparison table ---
# It guesses from prose: first character of any word ("Based on" -> B) and any
# standalone capital after uppercasing (the article "a" -> A).
parse_abcd_legacy <- function(response) {
  if (is.na(response) || is.null(response)) return(NA_character_)
  resp <- str_trim(response)

  # Try first letter if it is A-D (common for well-behaved models)
  first_char <- str_to_upper(str_sub(resp, 1, 1))
  if (first_char %in% c("A", "B", "C", "D")) return(first_char)

  # Look for standalone A/B/C/D near the start (e.g., "A." or "A)")
  m <- str_match(str_to_upper(resp), "^\\s*([A-D])\\s*[\\)\\.:,]")
  if (!is.na(m[1, 2])) return(m[1, 2])

  # Look for "answer is X" or "answer: X" patterns anywhere
  m <- str_match(str_to_upper(resp), "ANSWER\\s*(?:IS|:)\\s*([A-D])\\b")
  if (!is.na(m[1, 2])) return(m[1, 2])

  # Look for any standalone A-D letter anywhere (last resort)
  m <- str_match(str_to_upper(resp), "\\b([A-D])\\b")
  if (!is.na(m[1, 2])) return(m[1, 2])

  return(NA_character_)
}

# --- Parse (strict) and score correctness ---
choices <- read_csv(items_path, show_col_types = FALSE) %>%
  transmute(item_id = as.character(item_id), choice_A, choice_B, choice_C, choice_D)
raw <- raw %>% mutate(item_id = as.character(item_id)) %>% left_join(choices, by = "item_id")
stopifnot(!anyNA(raw$choice_A))
strict <- parse_mmlu_strict_df(raw$response, raw$choice_A, raw$choice_B, raw$choice_C, raw$choice_D)

df <- raw %>%
  mutate(
    answer_extracted = strict$letter,
    parse_rule = strict$rule,
    answer_legacy = map_chr(response, parse_abcd_legacy),
    outcome = as.integer(str_to_upper(answer_extracted) == str_to_upper(correct_answer)),
    outcome = ifelse(is.na(answer_extracted), NA_integer_, outcome),
    temperature = as.factor(temperature),
    sut_model = model,
    sut_short = str_extract(model, "[^/]+$"),
    item_id = as.character(item_id),
    variant_id = as.character(variant_id),
    category = as.character(category),
    subcategory = as.character(subcategory),
    replication = as.integer(replication),
    response_raw = response
  ) %>%
  select(
    item_id, category, subcategory, variant_id, sut_model, sut_short,
    temperature, replication, response_raw, answer_extracted, parse_rule,
    answer_legacy, correct_answer, outcome,
    elapsed_s, timestamp
  )

# --- Legacy vs strict parser, per variant (SI si:mmlu_parsing) ---
comparison <- df %>%
  mutate(status = case_when(
    is.na(answer_legacy) & is.na(answer_extracted) ~ "still_na",
    is.na(answer_legacy)                           ~ "recovered",
    is.na(answer_extracted)                        ~ "legacy_parsed_strict_na",
    answer_legacy == answer_extracted              ~ "agree",
    TRUE                                           ~ "conflict")) %>%
  group_by(variant_id) %>%
  summarise(
    n = n(),
    agree = sum(status == "agree"),
    conflict = sum(status == "conflict"),
    legacy_parsed_strict_na = sum(status == "legacy_parsed_strict_na"),
    still_na = sum(status == "still_na"),
    recovered = sum(status == "recovered"),
    conflict_strict_matches_key = sum(status == "conflict" & answer_extracted == correct_answer),
    conflict_legacy_matches_key = sum(status == "conflict" & answer_legacy == correct_answer),
    pct_unanswered_legacy = 100 * mean(is.na(answer_legacy)),
    pct_unanswered_strict = 100 * mean(is.na(answer_extracted)),
    .groups = "drop")
cat("\nLegacy vs strict parser:\n"); print(as.data.frame(comparison))
write_csv(comparison, compare_path)

# --- Admissibility: drop variants that fail the structural check ---
admissible <- read_csv(structure_path, show_col_types = FALSE) %>%
  filter(set == "main_text") %>% select(variant_id, passes)
stopifnot(setequal(admissible$variant_id, unique(df$variant_id)))
excluded <- admissible$variant_id[!admissible$passes]
cat("\nExcluded by structural check:", paste(excluded, collapse = ", "), "\n")
df_v5 <- df %>% select(-answer_legacy)
df <- df_v5 %>% filter(!variant_id %in% excluded)

# --- Diagnostics ---
cat("\nParsing diagnostics:\n")
cat("  Total rows:", nrow(df), "\n")
cat("  Parsed answers:", sum(!is.na(df$answer_extracted)), "\n")
cat("  Parse failures:", sum(is.na(df$answer_extracted)), "\n")
cat("  Parse rate:", round(mean(!is.na(df$answer_extracted)) * 100, 1), "%\n")

cat("\nScoring diagnostics:\n")
cat("  Correct (=1):", sum(df$outcome == 1, na.rm = TRUE), "\n")
cat("  Incorrect (=0):", sum(df$outcome == 0, na.rm = TRUE), "\n")
cat("  Overall accuracy:", round(mean(df$outcome, na.rm = TRUE) * 100, 1), "%\n")

parse_rate <- mean(!is.na(df$answer_extracted))
if (parse_rate < 0.90) {
  stop(sprintf("MMLU parse rate %.1f%% is below 90%% floor (MMLU is multi-choice free-text, noisier than JSON judges).", 100 * parse_rate), call. = FALSE)
}
if (mean(is.na(df$answer_extracted)) > 0.1) {
  warning("More than 10% parse failures. Check response format.")
  cat("\nSample unparseable responses:\n")
  df %>%
    filter(is.na(answer_extracted)) %>%
    slice_head(n = 10) %>%
    pull(response_raw) %>%
    walk(~ cat("  >", str_trunc(.x, 120), "\n"))
}

cat("\nDesign summary:\n")
cat("  Items:", n_distinct(df$item_id), "\n")
cat("  Categories:", n_distinct(df$category), "\n")
cat("  Subcategories:", n_distinct(df$subcategory), "\n")
cat("  Prompt variants:", n_distinct(df$variant_id), "\n")
cat("  SUT models:", n_distinct(df$sut_model), "\n")
cat("  Temperatures:", paste(levels(df$temperature), collapse = ", "), "\n")
cat("  Replications per cell:", max(df$replication) + 1, "\n")

# Per-model accuracy
cat("\nAccuracy by model:\n")
df %>%
  group_by(sut_short) %>%
  summarize(
    n = n(),
    parsed = sum(!is.na(outcome)),
    parse_rate = round(mean(!is.na(answer_extracted)) * 100, 1),
    accuracy = round(mean(outcome, na.rm = TRUE) * 100, 1),
    .groups = "drop"
  ) %>%
  print()

# Per-temperature accuracy
cat("\nAccuracy by temperature:\n")
df %>%
  group_by(temperature) %>%
  summarize(
    n = n(),
    accuracy = round(mean(outcome, na.rm = TRUE) * 100, 1),
    .groups = "drop"
  ) %>%
  print()

# Per-category accuracy
cat("\nAccuracy by category:\n")
df %>%
  group_by(category) %>%
  summarize(
    n = n(),
    accuracy = round(mean(outcome, na.rm = TRUE) * 100, 1),
    .groups = "drop"
  ) %>%
  print()

# --- Save ---
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_csv(df, output_path)
write_csv(df_v5, output_v5_path)
cat("\nSaved to:", output_path, "(", n_distinct(df$variant_id), "variants ) and", output_v5_path, "\n")
