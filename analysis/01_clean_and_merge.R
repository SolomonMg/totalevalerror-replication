# 01_clean_and_merge.R
# Load JSONL demo results, parse scored outcomes, merge with item metadata.
# Supports both Likert (1-5) and Pairwise (A/B) scoring methods.
#
# Usage:
#   Rscript analysis/01_clean_and_merge.R [stage] [scoring]
#   e.g.: Rscript analysis/01_clean_and_merge.R full likert

library(tidyverse)
library(jsonlite)

set.seed(42)

# --- Config ---
args <- commandArgs(trailingOnly = TRUE)
stage   <- if (length(args) >= 1) args[1] else "pilot"
scoring <- if (length(args) >= 2) args[2] else "likert"

input_path  <- sprintf("data/raw/%s_%s.jsonl", scoring, stage)
output_path <- sprintf("data/processed/%s_clean.csv", scoring)

cat("=== 01_clean_and_merge.R ===\n")
cat("Stage:", stage, "\n")
cat("Scoring:", scoring, "\n")
cat("Input:", input_path, "\n")

# --- Load JSONL ---
lines <- readLines(input_path)
raw <- map_dfr(lines, function(line) {
  x <- fromJSON(line, flatten = TRUE)
  # Flatten nested 'usage' to avoid row duplication from length-2 list
  if (!is.null(x$usage)) {
    x$usage_prompt_tokens    <- x$usage$prompt_tokens %||% 0L
    x$usage_completion_tokens <- x$usage$completion_tokens %||% 0L
  }
  x$usage <- NULL
  # Replace NULL/empty elements with NA to avoid as_tibble_row errors
  x <- map(x, ~ if (is.null(.x) || length(.x) == 0) NA else .x)
  as_tibble_row(x)
})

cat("Loaded", nrow(raw), "rows\n")

# --- Parse scored outcome ---
parse_likert <- function(response) {
  if (is.na(response) || is.null(response)) return(NA_real_)
  resp <- str_trim(response)

  # Try to extract a number 1-5
  nums <- str_extract(resp, "\\d+\\.?\\d*")
  if (!is.na(nums)) {
    val <- as.numeric(nums)
    if (val >= 1 && val <= 5) return(val)
    # Out of range
    return(NA_real_)
  }
  return(NA_real_)
}

parse_pairwise <- function(response) {
  if (is.na(response) || is.null(response)) return(NA_real_)
  resp <- str_to_upper(str_trim(response))

  # Direct match

  if (str_detect(resp, "^A\\b")) return(1)
  if (str_detect(resp, "^B\\b")) return(0)

  # Look for "Response A" or "Response B" anywhere
  if (str_detect(resp, "RESPONSE\\s*A")) return(1)
  if (str_detect(resp, "RESPONSE\\s*B")) return(0)

  return(NA_real_)
}

# Select parser based on scoring method
parse_fn <- if (scoring == "likert") parse_likert else parse_pairwise

df <- raw %>%
  mutate(
    outcome = map_dbl(response, parse_fn),
    temperature = as.factor(temperature),
    judge_model = model,
    judge_short = str_extract(model, "[^/]+$"),
    item_id = as.character(item_id),
    variant_id = as.character(variant_id),
    category = as.character(category),
    replication = as.integer(replication)
  ) %>%
  select(
    item_id, category, variant_id, judge_model, judge_short,
    temperature, replication, outcome, response, scoring,
    elapsed_s, stage, timestamp
  )

# --- Merge true_order for pairwise (position bias covariate) ---
if (scoring == "pairwise") {
  items <- read_csv("data/items_pairwise.csv", show_col_types = FALSE) %>%
    select(item_id, true_order) %>%
    distinct()
  df <- df %>% left_join(items, by = "item_id")
  cat("\nMerged true_order for pairwise items\n")
  cat("  true_order distribution:", table(df$true_order), "\n")
}

# --- Diagnostics ---
cat("\nParsing diagnostics:\n")
cat("  Total rows:", nrow(df), "\n")
cat("  Parsed outcomes:", sum(!is.na(df$outcome)), "\n")
cat("  Parse failures:", sum(is.na(df$outcome)), "\n")
cat("  Parse rate:", round(mean(!is.na(df$outcome)) * 100, 1), "%\n")

if (scoring == "likert") {
  cat("  Outcome range:", range(df$outcome, na.rm = TRUE), "\n")
  cat("  Outcome mean:", round(mean(df$outcome, na.rm = TRUE), 2), "\n")
} else {
  cat("  A (=1) rate:", round(mean(df$outcome == 1, na.rm = TRUE) * 100, 1), "%\n")
  cat("  B (=0) rate:", round(mean(df$outcome == 0, na.rm = TRUE) * 100, 1), "%\n")
}

parse_rate <- mean(!is.na(df$outcome))
if (parse_rate < 0.95) {
  stop(sprintf("Parse rate %.1f%% is below 95%% floor. Cannot proceed -- fix the parser or rerun the source data.", 100 * parse_rate), call. = FALSE)
}
if (mean(is.na(df$outcome)) > 0.1) {
  warning("More than 10% parse failures. Check response format.")
  cat("\nSample unparseable responses:\n")
  df %>%
    filter(is.na(outcome)) %>%
    slice_head(n = 5) %>%
    pull(response) %>%
    walk(~ cat("  >", str_trunc(.x, 80), "\n"))
}

cat("\nDesign summary:\n")
cat("  Items:", n_distinct(df$item_id), "\n")
cat("  Categories:", n_distinct(df$category), "\n")
cat("  Prompt variants:", n_distinct(df$variant_id), "\n")
cat("  Judge models:", n_distinct(df$judge_model), "\n")
cat("  Temperatures:", paste(levels(df$temperature), collapse = ", "), "\n")
cat("  Replications per cell:", max(df$replication) + 1, "\n")

# --- Save ---
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_csv(df, output_path)
cat("\nSaved to:", output_path, "\n")
