# 01_clean_safety_pairwise.R
# Clean pairwise safety JSONL into analysis-ready CSV.
# Parses A/B responses, records counterbalancing, assigns pair-level category.
#
# Usage:
#   Rscript analysis/01_clean_safety_pairwise.R [stage]
#   e.g.: Rscript analysis/01_clean_safety_pairwise.R pilot

library(tidyverse)
library(jsonlite)

set.seed(42)

args <- commandArgs(trailingOnly = TRUE)
stage <- if (length(args) >= 1) args[1] else "pilot"

input_path  <- sprintf("data/raw/safety_pairwise_%s.jsonl", stage)
output_path <- "data/processed/safety_pairwise_clean.csv"

cat("=== 01_clean_safety_pairwise.R ===\n")
cat("Stage:", stage, "\n")
cat("Input:", input_path, "\n")

# --- Load JSONL ---
lines <- readLines(input_path)
raw <- map_dfr(lines, function(line) {
  x <- fromJSON(line, flatten = TRUE)
  if (!is.null(x$usage)) {
    x$usage_prompt_tokens     <- x$usage$prompt_tokens %||% 0L
    x$usage_completion_tokens <- x$usage$completion_tokens %||% 0L
  }
  x$usage <- NULL
  x <- map(x, ~ if (is.null(.x) || length(.x) == 0) NA else .x)
  as_tibble_row(x)
})

cat("Loaded", nrow(raw), "rows\n")

# --- Parse A/B response ---
parse_pairwise <- function(response) {
  if (is.na(response) || is.null(response)) return(NA_real_)
  resp <- str_to_upper(str_trim(response))
  if (str_detect(resp, "^A\\b")) return(1)
  if (str_detect(resp, "^B\\b")) return(0)
  if (str_detect(resp, "RESPONSE\\s*A")) return(1)
  if (str_detect(resp, "RESPONSE\\s*B")) return(0)
  return(NA_real_)
}

# --- Build pair-level category ---
# For same-category pairs: use the shared category
# For cross-category pairs: alphabetically sorted "catA_x_catB"
make_pair_category <- function(cat_a, cat_b) {
  ifelse(cat_a == cat_b, cat_a,
         paste(pmin(cat_a, cat_b), pmax(cat_a, cat_b), sep = "_x_"))
}

df <- raw %>%
  mutate(
    outcome = map_dbl(response, parse_pairwise),
    temperature = as.factor(temperature),
    judge_model = model,
    judge_short = str_extract(model, "[^/]+$"),
    # pair_id is the "item" in the decomposition
    item_id = as.character(pair_id),
    variant_id = as.character(variant_id),
    category = make_pair_category(category_a, category_b),
    replication = as.integer(replication),
    # Position bias covariate
    true_order = ifelse(swapped, "swapped", "original"),
    scoring = "pairwise_safety"
  ) %>%
  select(
    item_id, category, variant_id, judge_model, judge_short,
    temperature, replication, outcome, response, scoring,
    elapsed_s, timestamp, true_order,
    # Keep original item IDs for BT reconstruction if needed
    item_id_a, item_id_b, presented_a, presented_b, swapped
  )

# --- Diagnostics ---
cat("\nParsing diagnostics:\n")
cat("  Total rows:", nrow(df), "\n")
cat("  Parsed outcomes:", sum(!is.na(df$outcome)), "\n")
cat("  Parse failures:", sum(is.na(df$outcome)), "\n")
cat("  Parse rate:", round(mean(!is.na(df$outcome)) * 100, 1), "%\n")
cat("  A (=1) rate:", round(mean(df$outcome == 1, na.rm = TRUE) * 100, 1), "%\n")
cat("  B (=0) rate:", round(mean(df$outcome == 0, na.rm = TRUE) * 100, 1), "%\n")

parse_rate <- mean(!is.na(df$outcome))
if (parse_rate < 0.95) {
  stop(sprintf("Safety pairwise parse rate %.1f%% is below 95%% floor.", 100 * parse_rate), call. = FALSE)
}

cat("\nPosition bias:\n")
cat("  true_order distribution:\n")
print(table(df$true_order))
# Check if position bias exists
pos_bias <- df %>%
  filter(!is.na(outcome)) %>%
  group_by(true_order) %>%
  summarise(mean_outcome = mean(outcome), n = n())
print(pos_bias)

cat("\nDesign summary:\n")
cat("  Pairs (items):", n_distinct(df$item_id), "\n")
cat("  Categories:", n_distinct(df$category), "\n")
cat("    Same-category:", sum(!str_detect(unique(df$category), "_x_")), "\n")
cat("    Cross-category:", sum(str_detect(unique(df$category), "_x_")), "\n")
cat("  Prompt variants:", n_distinct(df$variant_id), "\n")
cat("  Judge models:", n_distinct(df$judge_model), "\n")
cat("  Temperatures:", paste(levels(df$temperature), collapse = ", "), "\n")
cat("  Replications per cell:", max(df$replication) + 1, "\n")

# --- Save ---
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_csv(df, output_path)
cat("\nSaved", nrow(df), "rows to:", output_path, "\n")
