#!/usr/bin/env Rscript
# 17b_tier_disagreement.R
# Absolute judge disagreement in the tier-matched safety analysis.
#
# 17_sensitivity_tier_matched.R reports variance shares, which cannot say whether
# frontier judges disagree more in absolute terms. This script computes, per tier:
#   - each judge's safe rate
#   - mean pairwise item-level disagreement: |rate_j - rate_k| per item, in points,
#     averaged over items and over the tier's three judge pairs
#   - the spread of judge safe rates (SD of the three judge means, in points)
# with item bootstrap 95% CIs for each tier and for the between-tier contrasts,
# plus how concentrated the disagreement is: the items with any closed-frontier
# disagreement, and those items' share of each tier's total disagreement.
#
# Outputs:
#   data/processed/sensitivity_tier_judge_rates.csv
#   data/processed/sensitivity_tier_disagreement.csv

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
})

set.seed(42)
N_BOOT <- 10000L

raw_path <- "data/raw/safety_sensitivity_pilot.jsonl"
out_dir  <- "data/processed"

tier_map <- tribble(
  ~judge_short,             ~tier,
  "gpt-4o",                 "original",
  "gemini-2.0-flash-001",   "original",
  "claude-haiku-4.5",       "original",
  "gpt-5.4",                "closed_frontier",
  "gemini-3.1-pro-preview", "closed_frontier",
  "claude-opus-4-6",        "closed_frontier",
  "gpt-oss-120b",           "open_weight",
  "gemma-4-31b-it",         "open_weight",
  "deepseek-v3.2",          "open_weight"
)

# Same parsing rule as 17_sensitivity_tier_matched.R
df <- map_dfr(readLines(raw_path), function(line) {
  x <- fromJSON(line, flatten = TRUE)
  tibble(model = x$model, item_id = as.character(x$item_id),
         response = if (is.null(x$response)) NA_character_ else as.character(x$response))
}) %>%
  mutate(
    response_clean = str_to_upper(str_trim(response)),
    outcome = case_when(
      str_starts(response_clean, "UNSAFE") ~ 0L,
      str_starts(response_clean, "SAFE")   ~ 1L,
      TRUE ~ NA_integer_
    ),
    judge_short = str_extract(model, "[^/]+$")
  ) %>%
  filter(!is.na(outcome)) %>%
  inner_join(tier_map, by = "judge_short")

judge_rates <- df %>%
  group_by(tier, judge_short) %>%
  summarize(n_valid = n(), n_safe = sum(outcome), safe_pct = 100 * mean(outcome),
            .groups = "drop")
write_csv(judge_rates, file.path(out_dir, "sensitivity_tier_judge_rates.csv"))
print(judge_rates)

# Item x judge safe rates, in points
item_rates <- df %>%
  group_by(tier, judge_short, item_id) %>%
  summarize(rate = 100 * mean(outcome), .groups = "drop")
items <- sort(unique(item_rates$item_id))
stopifnot(length(items) == 24L)

rate_matrix <- function(tier_name) {
  item_rates %>%
    filter(tier == tier_name) %>%
    select(item_id, judge_short, rate) %>%
    pivot_wider(names_from = judge_short, values_from = rate) %>%
    arrange(item_id) %>%
    column_to_rownames("item_id") %>%
    as.matrix()
}
tiers <- c("original", "closed_frontier", "open_weight")
mats  <- set_names(map(tiers, rate_matrix), tiers)
stopifnot(all(map_lgl(mats, ~ !anyNA(.x) && nrow(.x) == 24L && ncol(.x) == 3L)))

pair_diffs <- function(m) {
  prs <- combn(colnames(m), 2)
  out <- apply(prs, 2, function(p) abs(m[, p[1]] - m[, p[2]]))
  colnames(out) <- apply(prs, 2, paste, collapse = " vs ")
  out
}
pair_mats <- map(mats, pair_diffs)

disagreement <- function(pm, idx) mean(pm[idx, , drop = FALSE])
judge_spread <- function(m, idx)  sd(colMeans(m[idx, , drop = FALSE]))

all_idx  <- seq_along(items)
boot_idx <- replicate(N_BOOT, sample(all_idx, replace = TRUE))

stat_table <- function(point, draws, level, label) {
  tibble(level = level, label = label, estimate = point,
         ci_lo = unname(quantile(draws, 0.025)),
         ci_hi = unname(quantile(draws, 0.975)))
}

dis_draws <- map(pair_mats, function(pm) apply(boot_idx, 2, function(i) disagreement(pm, i)))
spr_draws <- map(mats,      function(m)  apply(boot_idx, 2, function(i) judge_spread(m, i)))
dis_point <- map_dbl(pair_mats, disagreement, idx = all_idx)
spr_point <- map_dbl(mats,      judge_spread, idx = all_idx)

contrast <- function(point, draws, a, b, level) {
  stat_table(point[[a]] - point[[b]], draws[[a]] - draws[[b]], level, paste(a, "minus", b))
}

results <- bind_rows(
  map(tiers, ~ stat_table(dis_point[[.x]], dis_draws[[.x]], "tier_disagreement", .x)),
  contrast(dis_point, dis_draws, "closed_frontier", "original", "disagreement_contrast"),
  contrast(dis_point, dis_draws, "closed_frontier", "open_weight", "disagreement_contrast"),
  map(tiers, ~ stat_table(spr_point[[.x]], spr_draws[[.x]], "tier_judge_spread", .x)),
  contrast(spr_point, spr_draws, "closed_frontier", "original", "judge_spread_contrast"),
  contrast(spr_point, spr_draws, "closed_frontier", "open_weight", "judge_spread_contrast"),
  map_dfr(pair_mats, function(pm) {
    tibble(level = "pair", label = colnames(pm), estimate = colMeans(pm),
           ci_lo = NA_real_, ci_hi = NA_real_)
  })
)
# Concentration: the items on which closed-frontier judges disagree at all
item_means <- map(pair_mats, rowMeans)
focal <- names(which(item_means$closed_frontier > 0))
concentration <- bind_rows(
  map_dfr(tiers, ~ tibble(level = "items_with_disagreement", label = .x,
                          estimate = sum(item_means[[.x]] > 0),
                          ci_lo = NA_real_, ci_hi = NA_real_)),
  map_dfr(tiers, ~ tibble(level = "focal_item_share", label = .x,
                          estimate = 100 * sum(item_means[[.x]][focal]) / sum(item_means[[.x]]),
                          ci_lo = NA_real_, ci_hi = NA_real_))
)
results <- bind_rows(results, concentration)
write_csv(results, file.path(out_dir, "sensitivity_tier_disagreement.csv"))
print(results, n = Inf)
cat("\nWrote sensitivity_tier_judge_rates.csv, sensitivity_tier_disagreement.csv\n")
