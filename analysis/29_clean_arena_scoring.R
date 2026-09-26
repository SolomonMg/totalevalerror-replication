#!/usr/bin/env Rscript
# 29_clean_arena_scoring.R
# Parse Arena Likert + pairwise JSONL outputs into a clean per-battle CSV
# suitable for pipeline-comparison analysis (analysis/30).
#
# Output: data/processed/arena_scoring_clean.csv with one row per
#   (battle_id, pipeline_config_id, judge, variant, ...)
# Fields:
#   battle_id, category_llm, winner (human truth)
#   mode: "likert" | "pairwise_forced"
#   judge_model, variant_id
#   response_side (likert only): "a" | "b"
#   order (pairwise only): "listed" | "swapped"
#   score (likert 1-5)
#   judgment (pairwise A/B)
#   model_a_wins (pairwise recoded: 1 if response_a won, 0 if response_b won)

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

parse_likert_score <- function(response) {
  if (is.null(response) || is.na(response) || response == "") return(NA_real_)
  j <- tryCatch(fromJSON(response, simplifyVector = TRUE), error = function(e) NULL)
  if (!is.null(j) && !is.null(j$score)) {
    s <- suppressWarnings(as.numeric(j$score))
    if (!is.na(s) && s >= 1 && s <= 5) return(s)
  }
  m <- str_match(response, '"score"\\s*:\\s*(\\d+)')
  if (!is.na(m[1, 2])) {
    s <- as.numeric(m[1, 2])
    if (s >= 1 && s <= 5) return(s)   # same 1 - 5 range check as the JSON path
  }
  NA_real_
}

parse_pairwise <- function(response) {
  if (is.null(response) || is.na(response) || response == "") return(NA_character_)
  j <- tryCatch(fromJSON(response, simplifyVector = TRUE), error = function(e) NULL)
  if (!is.null(j) && !is.null(j$judgment)) {
    raw <- str_trim(as.character(j$judgment))
    v <- str_to_upper(raw)
    if (v %in% c("A", "B")) return(v)
    if (v == "TIE") return("TIE")
    if (v %in% c("BOTH_BAD", "BOTH BAD", "BOTHBAD")) return("BOTH_BAD")
  }
  up <- str_to_upper(response)
  if (str_detect(up, '"JUDGMENT"\\s*:\\s*"A"'))                       return("A")
  if (str_detect(up, '"JUDGMENT"\\s*:\\s*"B"'))                       return("B")
  if (str_detect(up, '"JUDGMENT"\\s*:\\s*"TIE"'))                     return("TIE")
  if (str_detect(up, '"JUDGMENT"\\s*:\\s*"BOTH[_ ]?BAD"'))            return("BOTH_BAD")
  NA_character_
}

read_jsonl <- function(path) {
  lines <- readLines(path)
  map(lines, ~ fromJSON(.x, simplifyVector = TRUE))
}

# ---- Likert ----
cat("=== Cleaning Likert Arena scores ===\n")
lik_raw <- read_jsonl("data/raw/arena_likert.jsonl")
lik <- tibble(
  battle_id     = map_chr(lik_raw, ~ as.character(.x$meta$battle_id)),
  category_llm  = map_chr(lik_raw, ~ as.character(.x$meta$category_llm)),
  winner        = map_chr(lik_raw, ~ as.character(.x$meta$winner)),
  response_side = map_chr(lik_raw, ~ as.character(.x$meta$response_side)),
  variant_id    = map_int(lik_raw, ~ as.integer(.x$meta$variant_id)),
  judge_model   = map_chr(lik_raw, ~ as.character(.x$model)),
  response      = map_chr(lik_raw, ~ as.character(.x$response %||% NA_character_)),
  score         = map_dbl(lik_raw, ~ parse_likert_score(.x$response %||% NA_character_))
) %>% mutate(mode = "likert")

parse_rate_lik <- mean(!is.na(lik$score))
cat(sprintf("  Likert rows: %d, parse rate: %.1f%%\n",
            nrow(lik), 100 * parse_rate_lik))
if (parse_rate_lik < 0.95) {
  stop(sprintf("Arena Likert parse rate %.1f%% below 95%% floor", 100 * parse_rate_lik))
}

# ---- Pairwise (rating-anchored ternary: A / B / tie) ----
# Note: parse_pairwise also recognizes BOTH_BAD for forward compatibility with
# the minimal Arena-vote experiment (data/raw/archive/); the production ternary
# data has no BOTH_BAD entries.
cat("\n=== Cleaning Pairwise (ternary) Arena scores ===\n")
pw_raw <- read_jsonl("data/raw/arena_pairwise_ternary.jsonl")
pw <- tibble(
  battle_id    = map_chr(pw_raw, ~ as.character(.x$meta$battle_id)),
  category_llm = map_chr(pw_raw, ~ as.character(.x$meta$category_llm)),
  winner       = map_chr(pw_raw, ~ as.character(.x$meta$winner)),
  order        = map_chr(pw_raw, ~ as.character(.x$meta$order)),
  variant_id   = map_int(pw_raw, ~ as.integer(.x$meta$variant_id)),
  judge_model  = map_chr(pw_raw, ~ as.character(.x$model)),
  response     = map_chr(pw_raw, ~ as.character(.x$response %||% NA_character_)),
  judgment     = map_chr(pw_raw, ~ parse_pairwise(.x$response %||% NA_character_))
) %>% mutate(mode = "pairwise_ternary")

parse_rate_pw <- mean(!is.na(pw$judgment))
cat(sprintf("  Pairwise rows: %d, parse rate: %.1f%%\n",
            nrow(pw), 100 * parse_rate_pw))
cat(sprintf("  Judgment distribution: A=%d, B=%d, TIE=%d, BOTH_BAD=%d\n",
            sum(pw$judgment == "A", na.rm = TRUE),
            sum(pw$judgment == "B", na.rm = TRUE),
            sum(pw$judgment == "TIE", na.rm = TRUE),
            sum(pw$judgment == "BOTH_BAD", na.rm = TRUE)))
if (parse_rate_pw < 0.95) {
  stop(sprintf("Arena pairwise parse rate %.1f%% below 95%% floor", 100 * parse_rate_pw))
}

# Recode pairwise: model_a_wins = 1 if response_a won, 0 if response_b won,
# 0.5 if TIE or BOTH_BAD (both forms of indifference for BT).
# order=listed:  A -> response_a won; B -> response_b won; TIE/BOTH_BAD -> 0.5
# order=swapped: A -> response_b won; B -> response_a won; TIE/BOTH_BAD -> 0.5
pw <- pw %>% mutate(
  model_a_wins = case_when(
    is.na(judgment) ~ NA_real_,
    judgment %in% c("TIE", "BOTH_BAD") ~ 0.5,
    order == "listed"  & judgment == "A" ~ 1,
    order == "listed"  & judgment == "B" ~ 0,
    order == "swapped" & judgment == "A" ~ 0,
    order == "swapped" & judgment == "B" ~ 1,
    TRUE ~ NA_real_
  )
)

# ---- Position bias diagnostic ----
cat("\n=== Pairwise position-bias check (per judge, non-tie/non-bothbad A-rate) ===\n")
print(pw %>% filter(!is.na(judgment), judgment %in% c("A", "B")) %>%
        group_by(judge_model, order) %>%
        summarize(a_rate = mean(judgment == "A"), n = n(), .groups = "drop") %>%
        pivot_wider(names_from = order, values_from = c(a_rate, n)))
cat("\n(a_rate_listed ~= 1 - a_rate_swapped indicates balanced position bias)\n")

cat("\n=== Pairwise tie-rate per judge (TIE + BOTH_BAD combined) ===\n")
print(pw %>% filter(!is.na(judgment)) %>%
        group_by(judge_model) %>%
        summarize(tie_rate    = mean(judgment %in% c("TIE", "BOTH_BAD")),
                  pure_tie    = mean(judgment == "TIE"),
                  bothbad_rate = mean(judgment == "BOTH_BAD"),
                  n = n(), .groups = "drop"))

# ---- Write long-format CSVs ----
# Keep both modes in separate files to mirror existing _clean.csv convention.
lik_out <- lik %>% select(battle_id, category_llm, winner, mode,
                          response_side, variant_id, judge_model, score)
write_csv(lik_out, "data/processed/arena_likert_clean.csv")
cat(sprintf("\nWrote data/processed/arena_likert_clean.csv (%d rows)\n", nrow(lik_out)))

pw_out <- pw %>% select(battle_id, category_llm, winner, mode,
                        order, variant_id, judge_model, judgment, model_a_wins)
write_csv(pw_out, "data/processed/arena_pairwise_clean.csv")
cat(sprintf("Wrote data/processed/arena_pairwise_clean.csv (%d rows)\n", nrow(pw_out)))

# ---- Category distribution ----
cat("\n=== Category distribution ===\n")
cat_dist <- lik_out %>% distinct(battle_id, category_llm) %>%
  count(category_llm) %>% arrange(desc(n))
print(cat_dist)
