#!/usr/bin/env Rscript
# 28_discriminability_by_distance.R
# Discriminability-by-distance analysis on the anchor validation data.
#
# For each anchor pair (a, b) with known ground-truth ideological distance,
# measure how well each scoring method (Likert, pairwise forced, pairwise TIE)
# identifies the more-conservative anchor.
#
# Metrics per pair:
#   Likert:          d-prime-style discriminability:
#                    (mean_a - mean_b) / pooled_SE, sign-corrected for ground truth
#   Pairwise forced: fraction of judgments that correctly picked the
#                    more-conservative anchor (chance = 0.5)
#   Pairwise TIE:    same as forced, with TIEs scored as 0.5
#
# Expected pattern: all methods near ceiling at distance 5 (poles); the
# interesting story is what happens at distance 1 (adjacent anchors).

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(ggplot2)
  library(patchwork)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

# -------- Ground truth (same as 27) --------
GROUND_TRUTH <- list(
  economic_left_right = c(
    "Market Liberal"            = 1,
    "Libertarian"               = 2,
    "Authoritarian Nationalist" = 3,
    "Centrist"                  = 4,
    "Social Democrat"           = 5,
    "Green Left"                = 6
  ),
  social_left_right = c(
    "Authoritarian Nationalist" = 1,
    "Market Liberal"            = 2,
    "Centrist"                  = 3,
    "Social Democrat"           = 4,
    "Libertarian"               = 5,
    "Green Left"                = 6
  )
)

parse_likert_score <- function(response) {
  if (is.null(response) || is.na(response) || response == "") return(NA_real_)
  j <- tryCatch(fromJSON(response, simplifyVector = TRUE), error = function(e) NULL)
  if (!is.null(j) && !is.null(j$score)) {
    s <- suppressWarnings(as.numeric(j$score))
    if (!is.na(s) && s >= 1 && s <= 5) return(s)
  }
  m <- str_match(response, '"score"\\s*:\\s*(\\d+)')
  if (!is.na(m[1, 2])) return(as.numeric(m[1, 2]))
  NA_real_
}

parse_pairwise <- function(response) {
  if (is.null(response) || is.na(response) || response == "") return(NA_character_)
  j <- tryCatch(fromJSON(response, simplifyVector = TRUE), error = function(e) NULL)
  if (!is.null(j) && !is.null(j$judgment)) {
    v <- str_to_upper(str_trim(as.character(j$judgment)))
    if (v %in% c("A", "B", "TIE")) return(v)
  }
  up <- str_to_upper(response)
  if (str_detect(up, '"JUDGMENT"\\s*:\\s*"A"'))   return("A")
  if (str_detect(up, '"JUDGMENT"\\s*:\\s*"B"'))   return("B")
  if (str_detect(up, '"JUDGMENT"\\s*:\\s*"TIE"')) return("TIE")
  NA_character_
}

read_jsonl <- function(path) map(readLines(path), ~ fromJSON(.x, simplifyVector = TRUE))

# -------- Likert: per-anchor-per-dimension means and SEs --------
cat("=== Likert discriminability ===\n")
lik_raw <- read_jsonl("data/raw/anchor_likert.jsonl")
lik <- tibble(
  prompt_id    = map_chr(lik_raw, ~ as.character(.x$meta$prompt_id)),
  dimension    = map_chr(lik_raw, ~ as.character(.x$meta$dimension)),
  anchor_label = map_chr(lik_raw, ~ as.character(.x$meta$anchor_label)),
  judge_model  = map_chr(lik_raw, ~ as.character(.x$model)),
  score        = map_dbl(lik_raw, ~ parse_likert_score(.x$response %||% NA_character_))
) %>% filter(!is.na(score))

# Per-(dim, anchor) mean + SE from the 3-judge × ~49-prompt sample
lik_stats <- lik %>%
  group_by(dimension, anchor_label) %>%
  summarize(mean_s = mean(score), sd_s = sd(score), n = n(), .groups = "drop") %>%
  mutate(se = sd_s / sqrt(n))

# For each pair (conservative, progressive), compute signed d-prime style:
#   (mean_conservative - mean_progressive) / pooled_SE
# Positive = method correctly orders pair; magnitude = effect size.
pair_likert <- map_dfr(names(GROUND_TRUTH), function(dim) {
  ranks <- GROUND_TRUTH[[dim]]
  anchors <- names(ranks)
  combn(anchors, 2, simplify = FALSE) %>%
    map_dfr(function(p) {
      # rank 1 = most conservative
      cons <- if (ranks[p[1]] < ranks[p[2]]) p[1] else p[2]
      prog <- if (ranks[p[1]] < ranks[p[2]]) p[2] else p[1]
      gt_dist <- abs(ranks[p[1]] - ranks[p[2]])
      rc <- lik_stats %>% filter(dimension == dim, anchor_label == cons)
      rp <- lik_stats %>% filter(dimension == dim, anchor_label == prog)
      pooled_se <- sqrt(rc$se^2 + rp$se^2)
      tibble(
        dimension = dim, cons_anchor = cons, prog_anchor = prog,
        gt_distance = as.integer(gt_dist),
        diff = rc$mean_s - rp$mean_s,
        d_prime = (rc$mean_s - rp$mean_s) / pooled_se,
        correct_direction = rc$mean_s > rp$mean_s
      )
    })
})
print(pair_likert %>% arrange(gt_distance, dimension) %>%
        mutate(across(where(is.numeric), ~ round(., 3))))

# -------- Pairwise: accuracy per pair --------
analyze_pw <- function(path, name, tie_is_uncertain = FALSE) {
  cat(sprintf("\n=== Pairwise (%s) discriminability ===\n", name))
  raw <- read_jsonl(path)
  pw <- tibble(
    prompt_id   = map_chr(raw, ~ as.character(.x$meta$prompt_id)),
    dimension   = map_chr(raw, ~ as.character(.x$meta$dimension)),
    anchor_a    = map_chr(raw, ~ as.character(.x$meta$anchor_a)),
    anchor_b    = map_chr(raw, ~ as.character(.x$meta$anchor_b)),
    order       = map_chr(raw, ~ as.character(.x$meta$order)),
    judge_model = map_chr(raw, ~ as.character(.x$model)),
    judgment    = map_chr(raw, ~ parse_pairwise(.x$response %||% NA_character_))
  ) %>% filter(!is.na(judgment))

  # Who won in terms of anchor identity (not position)
  pw <- pw %>% mutate(
    winner = case_when(
      judgment == "TIE" ~ "TIE",
      order == "listed"  & judgment == "A" ~ anchor_a,
      order == "listed"  & judgment == "B" ~ anchor_b,
      order == "swapped" & judgment == "A" ~ anchor_b,
      order == "swapped" & judgment == "B" ~ anchor_a,
      TRUE ~ NA_character_
    )
  )

  # For each unordered pair, compute accuracy (= fraction of judgments that
  # correctly picked the more-conservative anchor).
  map_dfr(names(GROUND_TRUTH), function(dim) {
    ranks <- GROUND_TRUTH[[dim]]
    anchors <- names(ranks)
    combn(anchors, 2, simplify = FALSE) %>%
      map_dfr(function(p) {
        cons <- if (ranks[p[1]] < ranks[p[2]]) p[1] else p[2]
        prog <- if (ranks[p[1]] < ranks[p[2]]) p[2] else p[1]
        gt_dist <- abs(ranks[p[1]] - ranks[p[2]])

        # Filter to this pair (regardless of which side was anchor_a vs anchor_b)
        pair_rows <- pw %>% filter(
          dimension == dim,
          (anchor_a == cons & anchor_b == prog) |
          (anchor_a == prog & anchor_b == cons)
        )
        n_total <- nrow(pair_rows)
        n_cons <- sum(pair_rows$winner == cons, na.rm = TRUE)
        n_prog <- sum(pair_rows$winner == prog, na.rm = TRUE)
        n_tie  <- sum(pair_rows$winner == "TIE", na.rm = TRUE)

        # Two accuracy definitions:
        # (1) Strict: fraction where judge picked the more-conservative anchor
        #     (ties count as incorrect).
        # (2) TIE-as-uncertain: ties count as 0.5 correct.
        accuracy_strict <- n_cons / n_total
        accuracy_tie_half <- (n_cons + 0.5 * n_tie) / n_total

        tibble(
          dimension = dim, cons_anchor = cons, prog_anchor = prog,
          gt_distance = as.integer(gt_dist),
          n_total = n_total, n_cons = n_cons, n_prog = n_prog, n_tie = n_tie,
          accuracy = if (tie_is_uncertain) accuracy_tie_half else accuracy_strict
        )
      })
  })
}

pair_pw_forced <- analyze_pw("data/raw/anchor_pairwise_forced.jsonl", "forced", FALSE)
pair_pw_tie    <- analyze_pw("data/raw/anchor_pairwise_tie.jsonl",    "tie",    TRUE)

# -------- Merge into long format --------
cat("\n\n=== Discriminability by distance (all methods) ===\n")
long <- bind_rows(
  pair_likert %>% mutate(
    method = "Likert",
    accuracy = as.numeric(correct_direction),   # 0 or 1 per pair at this granularity
    metric_label = sprintf("d' = %.2f", d_prime)
  ) %>% select(dimension, cons_anchor, prog_anchor, gt_distance,
               method, accuracy, d_prime),
  pair_pw_forced %>% mutate(method = "Pairwise forced", d_prime = NA_real_) %>%
    select(dimension, cons_anchor, prog_anchor, gt_distance, method, accuracy, d_prime),
  pair_pw_tie %>% mutate(method = "Pairwise TIE", d_prime = NA_real_) %>%
    select(dimension, cons_anchor, prog_anchor, gt_distance, method, accuracy, d_prime)
)

# Per-distance summary
summary_by_dist <- long %>%
  group_by(method, gt_distance) %>%
  summarize(
    n_pairs = n(),
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    mean_d_prime = mean(d_prime, na.rm = TRUE),
    .groups = "drop"
  )
cat("\nPer-distance summary (accuracy = fraction pairs with correct ordering):\n")
print(summary_by_dist %>% mutate(across(where(is.numeric), ~ round(., 3))))

# Save
write_csv(long, "data/processed/anchor_discriminability_pairs.csv")
write_csv(summary_by_dist, "data/processed/anchor_discriminability_summary.csv")

# -------- Figure --------
# Two panels: (left) pairwise accuracy vs distance; (right) Likert d-prime vs distance.
fig_dir <- "figures"
theme_tle <- theme_minimal(base_size = 11) + theme(
  panel.grid.minor = element_blank(),
  strip.text = element_text(face = "bold"),
  plot.title = element_text(face = "bold")
)

# Left panel: pairwise accuracy (strict + tie-half) vs distance
pw_long <- bind_rows(
  pair_pw_forced %>% mutate(method = "Pairwise forced"),
  pair_pw_tie    %>% mutate(method = "Pairwise TIE (half-credit)")
)

p1 <- ggplot(pw_long, aes(x = gt_distance, y = accuracy,
                           color = method, shape = dimension)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray60") +
  geom_jitter(width = 0.1, height = 0, size = 2.5, alpha = 0.8) +
  stat_summary(fun = mean, geom = "line", aes(group = method),
               linewidth = 0.9, alpha = 0.8) +
  scale_color_manual(values = c("Pairwise forced" = "#c0392b",
                                  "Pairwise TIE (half-credit)" = "#e67e22")) +
  scale_x_continuous(breaks = 1:5) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0.4, 1.02)) +
  labs(
    title = "Pairwise accuracy by ground-truth distance",
    subtitle = "Fraction of judgments ordering the pair correctly",
    x = "Ground-truth rank distance between anchors",
    y = "Accuracy",
    color = "Method", shape = "Dimension"
  ) + theme_tle

# Right panel: Likert d-prime vs distance
p2 <- ggplot(pair_likert, aes(x = gt_distance, y = d_prime, shape = dimension)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray60") +
  geom_jitter(width = 0.1, height = 0, size = 2.5, alpha = 0.8, color = "#2166ac") +
  stat_summary(fun = mean, geom = "line",
               linewidth = 0.9, alpha = 0.8, color = "#2166ac") +
  scale_x_continuous(breaks = 1:5) +
  labs(
    title = "Likert d' by ground-truth distance",
    subtitle = "Signed effect size: (mean_cons - mean_prog) / pooled SE",
    x = "Ground-truth rank distance between anchors",
    y = "d' (higher = more discriminable)",
    shape = "Dimension"
  ) + theme_tle

p_combined <- p1 + p2 + plot_layout(ncol = 2)

ggsave(file.path(fig_dir, "fig_anchor_discriminability.pdf"),
       p_combined, width = 12, height = 5)
ggsave(file.path(fig_dir, "fig_anchor_discriminability.png"),
       p_combined, width = 12, height = 5, dpi = 300)
cat("\nSaved figure: figures/fig_anchor_discriminability.pdf/png\n")

cat("\n=== Files written ===\n")
cat("  data/processed/anchor_discriminability_pairs.csv\n")
cat("  data/processed/anchor_discriminability_summary.csv\n")
cat("  figures/fig_anchor_discriminability.pdf/png\n")
