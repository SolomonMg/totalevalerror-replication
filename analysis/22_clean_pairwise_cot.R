#!/usr/bin/env Rscript
# 22_clean_pairwise_cot.R
# Parse the CoT pairwise JSONL output into a clean CSV:
#   - Extract "judgment" (A/B) from each JSON response
#   - Recode to model_a_wins using true_order from items CSV (content-based outcome)
#   - Match the column schema of pairwise_clean.csv so downstream scripts work

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
})

raw_path  <- "data/raw/pairwise_cot.jsonl"
items_path <- "data/items_pairwise.csv"
out_path  <- "data/processed/pairwise_cot_clean.csv"

cat(sprintf("=== 22_clean_pairwise_cot.R ===\n"))
cat(sprintf("Reading %s\n", raw_path))

lines <- readLines(raw_path)
cat(sprintf("  %d raw records\n", length(lines)))

# --- Parse each JSONL line + extract judgment from the response field ---
parse_judgment <- function(response) {
  if (is.null(response) || is.na(response) || response == "") return(NA_character_)
  j <- tryCatch(fromJSON(response, simplifyVector = TRUE), error = function(e) NULL)
  if (is.null(j) || is.null(j$judgment)) {
    # Fallback: substring search on the raw response
    resp_up <- str_to_upper(response)
    if (str_detect(resp_up, "\"JUDGMENT\"\\s*:\\s*\"A\"")) return("A")
    if (str_detect(resp_up, "\"JUDGMENT\"\\s*:\\s*\"B\"")) return("B")
    return(NA_character_)
  }
  verdict <- str_to_upper(str_trim(as.character(j$judgment)))
  if (verdict %in% c("A", "B")) return(verdict)
  return(NA_character_)
}

records <- map(lines, ~ fromJSON(.x, simplifyVector = TRUE))

df <- tibble(
  item_id      = map_chr(records, ~ as.character(.x$item_id)),
  category     = map_chr(records, ~ as.character(.x$category %||% "unknown")),
  variant_id   = map_chr(records, ~ as.character(.x$variant_id)),
  judge_model  = map_chr(records, ~ as.character(.x$model)),
  temperature  = map_dbl(records, ~ as.numeric(.x$temperature)),
  replication  = map_int(records, ~ as.integer(.x$replication)),
  response     = map_chr(records, ~ as.character(.x$response %||% NA_character_)),
  judgment     = map_chr(records, ~ parse_judgment(.x$response %||% NA_character_))
)

cat(sprintf("  Parsed %d / %d records\n",
            sum(!is.na(df$judgment)), nrow(df)))
cat(sprintf("  Parse rate: %.1f%%\n",
            100 * mean(!is.na(df$judgment))))
parse_rate <- mean(!is.na(df$judgment))
if (parse_rate < 0.95) {
  stop(sprintf("Pairwise CoT parse rate %.1f%% is below 95%% floor.", 100 * parse_rate), call. = FALSE)
}

# --- Merge true_order ---
items <- read_csv(items_path, show_col_types = FALSE) %>%
  select(item_id, true_order) %>% distinct()
df <- df %>% left_join(items, by = "item_id")

# --- Recode to model_a_wins (content-based, position-independent) ---
# outcome = 1 if judge said "A" (i.e., response_a won in the prompt)
# model_a_wins = 1 if the original-model-A response won
#   true_order == "original": response_a == model_a_original -> model_a_wins = outcome
#   true_order == "swapped":  response_a == model_b_original -> model_a_wins = 1 - outcome
df <- df %>% mutate(
  outcome      = case_when(judgment == "A" ~ 1, judgment == "B" ~ 0, TRUE ~ NA_real_),
  model_a_wins = case_when(
    is.na(outcome) ~ NA_real_,
    true_order == "original" ~ outcome,
    true_order == "swapped"  ~ 1 - outcome,
    TRUE ~ NA_real_
  ),
  judge_short  = str_extract(judge_model, "[^/]+$"),
  scoring      = "pairwise_cot"
)

# --- Diagnostics ---
cat("\n=== Per-judge means (position-based vs recoded) ===\n")
df %>% filter(!is.na(outcome)) %>%
  group_by(judge_model) %>%
  summarize(
    n = n(),
    a_rate    = mean(outcome),
    b_rate    = 1 - mean(outcome),
    recoded   = mean(model_a_wins, na.rm = TRUE),
    .groups = "drop"
  ) %>% print()

# --- Write ---
write_csv(df, out_path)
cat(sprintf("\nWrote %s (%d rows)\n", out_path, nrow(df)))
