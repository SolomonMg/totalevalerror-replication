#!/usr/bin/env Rscript
# 24_clean_likert_cot.R
# Parse the CoT Likert JSONL output into a clean CSV matching the likert
# pipeline's column schema.

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
})

raw_path  <- "data/raw/likert_cot.jsonl"
out_path  <- "data/processed/likert_cot_clean.csv"

cat("=== 24_clean_likert_cot.R ===\n")
lines <- readLines(raw_path)
cat(sprintf("  %d raw records\n", length(lines)))

parse_score <- function(response) {
  if (is.null(response) || is.na(response) || response == "") return(NA_real_)
  j <- tryCatch(fromJSON(response, simplifyVector = TRUE), error = function(e) NULL)
  if (is.null(j) || is.null(j$score)) {
    # Fallback: substring search for "score":N
    m <- str_match(response, '"score"\\s*:\\s*(\\d+)')
    if (!is.na(m[1, 2])) return(as.numeric(m[1, 2]))
    return(NA_real_)
  }
  s <- suppressWarnings(as.numeric(j$score))
  if (is.na(s) || s < 1 || s > 5) return(NA_real_)
  return(s)
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
  outcome      = map_dbl(records, ~ parse_score(.x$response %||% NA_character_))
) %>%
  mutate(
    judge_short = str_extract(judge_model, "[^/]+$"),
    scoring     = "likert"  # canonical name for downstream scripts
  )

cat(sprintf("  Parsed %d / %d records (%.1f%%)\n",
            sum(!is.na(df$outcome)), nrow(df),
            100 * mean(!is.na(df$outcome))))
parse_rate <- mean(!is.na(df$outcome))
if (parse_rate < 0.95) {
  stop(sprintf("Likert CoT parse rate %.1f%% is below 95%% floor.", 100 * parse_rate), call. = FALSE)
}

cat("\n=== Per-judge score summary ===\n")
df %>% filter(!is.na(outcome)) %>%
  group_by(judge_model) %>%
  summarize(
    n         = n(),
    mean_score = mean(outcome),
    sd_score   = sd(outcome),
    .groups = "drop"
  ) %>% print()

cat("\n=== Score distribution ===\n")
print(df %>% filter(!is.na(outcome)) %>% count(outcome))

write_csv(df, out_path)
cat(sprintf("\nWrote %s (%d rows)\n", out_path, nrow(df)))
