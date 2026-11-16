#!/usr/bin/env Rscript
# 26_prep_anchor_validation.R
# Build scoring input CSVs for the anchor-persona validation study.
#
# Source: 1,201 anchor responses from the aligned_to_whom study1_ideology
# project -- 6 personas x ~49 prompts x 5 dimensions, each response produced
# by prompting an LLM to respond as a specific ideological persona.
#
# Outputs (for the matched CoT scoring pipelines):
#   data/processed/anchor_likert_input.csv     - one row per (persona, prompt)
#   data/processed/anchor_pairwise_input.csv   - one row per (persona_A, persona_B, prompt)
#
# Scope: economic_left_right and social_left_right only. These have the
# cleanest ground-truth anchor rankings; the other dimensions are fuzzier.

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
})

set.seed(42)

ANCHOR_FILE <- path.expand("~/workspace/aligned_to_whom/study1_ideology/responses/anchor_responses_en.jsonl")
KEEP_DIMS <- c("economic_left_right", "social_left_right")
N_PROMPTS_PAIRWISE <- 20  # sample 20 prompts per dim for pairwise (C(6,2)=15 pairs each)

cat("=== 26_prep_anchor_validation.R ===\n")

lines <- readLines(ANCHOR_FILE)
recs <- map(lines, ~ fromJSON(.x, simplifyVector = TRUE))

safe_chr <- function(x) if (is.null(x)) NA_character_ else as.character(x)

raw <- tibble(
  prompt_id      = map_chr(recs, ~ safe_chr(.x$prompt_id)),
  dimension      = map_chr(recs, ~ safe_chr(.x$dimension)),
  subdimension   = map_chr(recs, ~ safe_chr(.x$subdimension)),
  prompt_text    = map_chr(recs, ~ safe_chr(.x$prompt_text)),
  anchor_label   = map_chr(recs, ~ safe_chr(.x$anchor_label)),
  anchor_profile = map_chr(recs, ~ safe_chr(.x$anchor_profile)),
  response_text  = map_chr(recs, ~ safe_chr(.x$response_text))
) %>% filter(!is.na(response_text) & response_text != "")

cat(sprintf("Loaded %d anchor responses\n", nrow(raw)))
cat("Distinct anchors:", paste(sort(unique(raw$anchor_label)), collapse = ", "), "\n")
cat("Distinct dimensions:", paste(sort(unique(raw$dimension)), collapse = ", "), "\n\n")

df <- raw %>%
  filter(dimension %in% KEEP_DIMS) %>%
  select(prompt_id, dimension, subdimension, prompt_text,
         anchor_label, anchor_profile, response_text) %>%
  distinct(prompt_id, anchor_label, .keep_all = TRUE)

cat(sprintf("After filter: %d responses across %d dims\n", nrow(df), n_distinct(df$dimension)))
print(df %>% count(dimension, anchor_label) %>% pivot_wider(names_from = dimension, values_from = n))

# -------- Likert input --------
# One scoring call per (prompt_id, anchor_label). We keep dimension + anchor_profile
# as metadata so the downstream parser can re-attach them.
likert_input <- df %>%
  mutate(item_id = sprintf("%s__%s", prompt_id, anchor_profile)) %>%
  select(item_id, prompt_id, dimension, anchor_label, anchor_profile, response_text)

write_csv(likert_input, "data/processed/anchor_likert_input.csv")
cat(sprintf("\nWrote anchor_likert_input.csv (%d rows)\n", nrow(likert_input)))

# -------- Pairwise input --------
# For a random sample of N_PROMPTS_PAIRWISE prompts per dimension, form all
# C(6,2) = 15 persona pairs. Record (response_a, response_b) and the ordered
# persona labels so we can recode later.
#
# The pairwise runner will do both orderings (listed + swapped), so we only
# emit one direction here.
anchors <- sort(unique(df$anchor_label))
pairs <- combn(anchors, 2, simplify = FALSE)

pair_rows <- map_dfr(KEEP_DIMS, function(dim) {
  prompts_in_dim <- unique(df$prompt_id[df$dimension == dim])
  sampled <- sample(prompts_in_dim, min(N_PROMPTS_PAIRWISE, length(prompts_in_dim)))
  map_dfr(sampled, function(pid) {
    map_dfr(pairs, function(p) {
      ra <- df$response_text[df$prompt_id == pid & df$anchor_label == p[1]]
      rb <- df$response_text[df$prompt_id == pid & df$anchor_label == p[2]]
      if (length(ra) == 0 || length(rb) == 0) return(NULL)
      tibble(
        item_id = sprintf("%s__%s_vs_%s",
                          pid,
                          str_replace_all(p[1], " ", ""),
                          str_replace_all(p[2], " ", "")),
        prompt_id = pid,
        dimension = dim,
        anchor_a = p[1],
        anchor_b = p[2],
        response_a = ra[1],
        response_b = rb[1]
      )
    })
  })
})

write_csv(pair_rows, "data/processed/anchor_pairwise_input.csv")
cat(sprintf("Wrote anchor_pairwise_input.csv (%d rows)\n", nrow(pair_rows)))
cat(sprintf("  %d pairs per dim, %d dims, %d total\n",
            N_PROMPTS_PAIRWISE * length(pairs), length(KEEP_DIMS), nrow(pair_rows)))
