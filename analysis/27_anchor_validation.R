#!/usr/bin/env Rscript
# 27_anchor_validation.R
# Validate scoring method (Likert vs pairwise) against anchor-persona ground truth.
#
# Anchor ground truth ordering (most -> least conservative on each dimension):
#
#   economic_left_right:
#     Market Liberal > Libertarian > Authoritarian Nationalist > Centrist >
#     Social Democrat > Green Left
#
#   social_left_right:
#     Authoritarian Nationalist > Market Liberal > Centrist > Social Democrat
#     > Libertarian > Green Left
#
# Outputs:
#   data/processed/anchor_validation_summary.csv   -- tau per method per dimension
#   figures/fig_anchor_validation.pdf/.png          -- bar chart or rank plot

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(BradleyTerry2)  # may be available; if not, use PlackettLuce
})

# -------- Ground truth rankings --------
# Rank 1 = most conservative; rank 6 = most progressive.
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

ANCHORS <- names(GROUND_TRUTH$economic_left_right)

# -------- Parse JSONL helpers --------
parse_likert_score <- function(response) {
  if (is.null(response) || is.na(response) || response == "") return(NA_real_)
  j <- tryCatch(fromJSON(response, simplifyVector = TRUE), error = function(e) NULL)
  if (!is.null(j) && !is.null(j$score)) {
    s <- suppressWarnings(as.numeric(j$score))
    if (!is.na(s) && s >= 1 && s <= 5) return(s)
  }
  m <- str_match(response, '"score"\\s*:\\s*(\\d+)')
  if (!is.na(m[1, 2])) return(as.numeric(m[1, 2]))
  return(NA_real_)
}

parse_pairwise_judgment <- function(response) {
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
  return(NA_character_)
}

read_jsonl <- function(path) {
  lines <- readLines(path)
  map(lines, ~ fromJSON(.x, simplifyVector = TRUE))
}

# -------- Likert analysis --------
cat("=== Likert ===\n")
lik_raw <- read_jsonl("data/raw/anchor_likert.jsonl")
lik <- tibble(
  prompt_id      = map_chr(lik_raw, ~ as.character(.x$meta$prompt_id)),
  dimension      = map_chr(lik_raw, ~ as.character(.x$meta$dimension)),
  anchor_label   = map_chr(lik_raw, ~ as.character(.x$meta$anchor_label)),
  judge_model    = map_chr(lik_raw, ~ as.character(.x$model)),
  response       = map_chr(lik_raw, ~ as.character(.x$response %||% NA_character_)),
  score          = map_dbl(lik_raw, ~ parse_likert_score(.x$response %||% NA_character_))
)

cat(sprintf("  Rows: %d | Parsed: %.1f%%\n",
            nrow(lik), 100 * mean(!is.na(lik$score))))

# Per-anchor mean score per dimension (averaged over judges + prompts)
lik_summary <- lik %>%
  filter(!is.na(score)) %>%
  group_by(dimension, anchor_label) %>%
  summarize(mean_score = mean(score), n = n(), .groups = "drop")

cat("\n  Likert means by (dimension, anchor):\n")
print(lik_summary %>%
        pivot_wider(id_cols = anchor_label,
                    names_from = dimension, values_from = mean_score) %>%
        mutate(across(-anchor_label, ~ round(., 2))))

# Likert ranking per dimension
likert_ranks <- lik_summary %>%
  group_by(dimension) %>%
  mutate(likert_rank = rank(-mean_score)) %>%  # higher score -> more conservative -> lower rank number
  ungroup() %>%
  select(dimension, anchor_label, likert_rank, likert_score = mean_score)

# -------- Pairwise analysis (TIE + forced) --------
analyze_pairwise <- function(path, allow_tie) {
  cat(sprintf("\n=== Pairwise (%s) ===\n", path))
  raw <- read_jsonl(path)
  pw <- tibble(
    prompt_id     = map_chr(raw, ~ as.character(.x$meta$prompt_id)),
    dimension     = map_chr(raw, ~ as.character(.x$meta$dimension)),
    anchor_a      = map_chr(raw, ~ as.character(.x$meta$anchor_a)),
    anchor_b      = map_chr(raw, ~ as.character(.x$meta$anchor_b)),
    order         = map_chr(raw, ~ as.character(.x$meta$order)),
    judge_model   = map_chr(raw, ~ as.character(.x$model)),
    response      = map_chr(raw, ~ as.character(.x$response %||% NA_character_)),
    judgment      = map_chr(raw, ~ parse_pairwise_judgment(.x$response %||% NA_character_))
  )
  cat(sprintf("  Rows: %d | Parsed: %.1f%%\n",
              nrow(pw), 100 * mean(!is.na(pw$judgment))))
  cat("  Judgment distribution:\n")
  print(pw %>% count(judgment))

  # Recode "which physical side won" to "which anchor won"
  # order=listed:  judgment=A -> anchor_a won; judgment=B -> anchor_b won
  # order=swapped: judgment=A -> anchor_b won; judgment=B -> anchor_a won
  pw <- pw %>% mutate(
    winner = case_when(
      judgment == "TIE" ~ "TIE",
      order == "listed"  & judgment == "A" ~ anchor_a,
      order == "listed"  & judgment == "B" ~ anchor_b,
      order == "swapped" & judgment == "A" ~ anchor_b,
      order == "swapped" & judgment == "B" ~ anchor_a,
      TRUE ~ NA_character_
    )
  ) %>% filter(!is.na(winner))

  # For BT fitting, aggregate: counts of A-wins, B-wins, ties per unordered pair per dimension
  agg <- pw %>%
    mutate(
      p1 = pmin(anchor_a, anchor_b),
      p2 = pmax(anchor_a, anchor_b)
    ) %>%
    group_by(dimension, p1, p2) %>%
    summarize(
      n_p1_wins = sum(winner == p1),
      n_p2_wins = sum(winner == p2),
      n_ties    = sum(winner == "TIE"),
      .groups = "drop"
    )

  # Fit BT per dimension using BradleyTerry2's BTm. Ties are split 0.5/0.5
  # if allow_tie=TRUE and we're not fitting Davidson; otherwise dropped.
  dims <- unique(agg$dimension)
  out <- map_dfr(dims, function(dim) {
    a <- agg %>% filter(dimension == dim)
    # Construct the win/loss binomial table.
    # BTm wants: a data.frame with player1/player2 columns + win counts.
    # Simplest: expand ties as 0.5 wins each (half-win rule).
    rows <- a %>% rowwise() %>%
      mutate(
        wins1 = n_p1_wins + 0.5 * ifelse(allow_tie, n_ties, 0),
        wins2 = n_p2_wins + 0.5 * ifelse(allow_tie, n_ties, 0)
      ) %>% ungroup()

    # Pseudo-count framework: create long-form for BTm
    # BTm fits a logit model. With half-wins, round to nearest 0.5 and supply
    # as win counts. BradleyTerry2 accepts non-integer via BTabilities when
    # using the Bradley-Terry likelihood directly; simpler route: use the
    # Minorization-Maximization MLE manually.
    #
    # For this validation we use a direct MLE: pi_i such that
    #   P(i beats j) = pi_i / (pi_i + pi_j)
    # Log-likelihood: sum_ij wins_ij * log(pi_i / (pi_i + pi_j))
    # Solve via MM algorithm on the symmetric count matrix.

    players <- sort(unique(c(rows$p1, rows$p2)))
    K <- length(players)
    W <- matrix(0, K, K, dimnames = list(players, players))
    for (i in seq_len(nrow(rows))) {
      W[rows$p1[i], rows$p2[i]] <- rows$wins1[i]
      W[rows$p2[i], rows$p1[i]] <- rows$wins2[i]
    }

    N <- W + t(W)  # total comparisons between each pair
    w <- rowSums(W)  # total wins per player

    # MM algorithm (Hunter 2004)
    pi <- rep(1, K); names(pi) <- players
    for (it in 1:2000) {
      pi_new <- w
      for (i in seq_len(K)) {
        denom_sum <- 0
        for (j in seq_len(K)) {
          if (i == j || N[i, j] == 0) next
          denom_sum <- denom_sum + N[i, j] / (pi[i] + pi[j])
        }
        pi_new[i] <- w[i] / denom_sum
      }
      pi_new <- pi_new / sum(pi_new) * K  # normalize
      if (max(abs(pi_new - pi)) < 1e-8) break
      pi <- pi_new
    }
    tibble(
      dimension = dim,
      anchor_label = names(pi),
      bt_strength = as.numeric(pi)
    )
  })

  out %>% group_by(dimension) %>%
    mutate(bt_rank = rank(-bt_strength)) %>% ungroup()
}

pw_tie_ranks <- analyze_pairwise("data/raw/anchor_pairwise_tie.jsonl",
                                 allow_tie = TRUE) %>%
  rename(pw_tie_rank = bt_rank, pw_tie_strength = bt_strength)

pw_forced_ranks <- analyze_pairwise("data/raw/anchor_pairwise_forced.jsonl",
                                    allow_tie = FALSE) %>%
  rename(pw_forced_rank = bt_rank, pw_forced_strength = bt_strength)

# -------- Merge + Spearman tau vs. ground truth --------
gt_df <- map_dfr(names(GROUND_TRUTH), ~ tibble(
  dimension = .x,
  anchor_label = names(GROUND_TRUTH[[.x]]),
  gt_rank = as.integer(GROUND_TRUTH[[.x]])
))

all_ranks <- gt_df %>%
  left_join(likert_ranks,   by = c("dimension", "anchor_label")) %>%
  left_join(pw_tie_ranks,   by = c("dimension", "anchor_label")) %>%
  left_join(pw_forced_ranks, by = c("dimension", "anchor_label"))

cat("\n\n=== Rank comparison ===\n")
print(all_ranks %>% arrange(dimension, gt_rank))

# Spearman tau (= Spearman rho for ranks; use cor() with method="spearman")
tau_summary <- all_ranks %>%
  group_by(dimension) %>%
  summarize(
    tau_likert     = cor(gt_rank, likert_rank,     method = "spearman"),
    tau_pairwise_tie    = cor(gt_rank, pw_tie_rank,    method = "spearman"),
    tau_pairwise_forced = cor(gt_rank, pw_forced_rank, method = "spearman"),
    .groups = "drop"
  )

cat("\n=== Spearman rho vs ground truth (per dimension) ===\n")
print(tau_summary %>% mutate(across(where(is.numeric), ~ round(., 3))))

# Pooled tau across both dims
all_pooled <- all_ranks %>% select(-dimension)
tau_pool <- tibble(
  tau_likert          = cor(all_pooled$gt_rank, all_pooled$likert_rank, method = "spearman"),
  tau_pairwise_tie    = cor(all_pooled$gt_rank, all_pooled$pw_tie_rank, method = "spearman"),
  tau_pairwise_forced = cor(all_pooled$gt_rank, all_pooled$pw_forced_rank, method = "spearman")
)
cat("\n=== Pooled Spearman rho (both dims, 12 anchor-dimension cells) ===\n")
print(tau_pool %>% mutate(across(everything(), ~ round(., 3))))

write_csv(all_ranks,    "data/processed/anchor_validation_ranks.csv")
write_csv(tau_summary,  "data/processed/anchor_validation_tau.csv")
cat("\nSaved:\n")
cat("  data/processed/anchor_validation_ranks.csv\n")
cat("  data/processed/anchor_validation_tau.csv\n")
