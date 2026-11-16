#!/usr/bin/env Rscript
# 36_arena_bt_dotplot.R
# Dotplot of Spearman correlation between each scoring pipeline's
# Bradley-Terry leaderboard and the same-sample human BT (fit on the
# same 4,676 battles' human votes). Overall and per category, with
# Fisher z 95% CIs and Williams' dependent-correlation tests for
# TEE pairwise vs each other pipeline.
#
# Human BT is the natural reference: asks whether pipeline X matches
# what the humans concluded on the same battles. The full-100k 'gold'
# comparison (analysis/34_arena_bt_vs_full.R) is a secondary check.

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

# ---- BT MM fit ----
fit_bt_mm <- function(p1, p2, wins1, wins2, n_iter = 2000, tol = 1e-7) {
  players <- sort(unique(c(p1, p2)))
  K <- length(players)
  if (K < 2) return(NULL)
  W <- matrix(0, K, K, dimnames = list(players, players))
  for (i in seq_along(p1)) {
    W[p1[i], p2[i]] <- W[p1[i], p2[i]] + wins1[i]
    W[p2[i], p1[i]] <- W[p2[i], p1[i]] + wins2[i]
  }
  N <- W + t(W)
  w <- rowSums(W)
  pi <- rep(1, K); names(pi) <- players
  for (iter in seq_len(n_iter)) {
    pi_new <- w
    for (i in seq_len(K)) {
      denom_sum <- 0
      for (j in seq_len(K)) {
        if (i == j || N[i, j] == 0) next
        denom_sum <- denom_sum + N[i, j] / (pi[i] + pi[j])
      }
      pi_new[i] <- if (denom_sum > 0) w[i] / denom_sum else pi[i]
    }
    pi_new <- pi_new / mean(pi_new)
    if (max(abs(pi_new - pi)) < tol) break
    pi <- pi_new
  }
  pi
}

aggregate_pair_wins <- function(df_sub, winner_col) {
  df_sub %>%
    mutate(p1 = pmin(model_a, model_b),
           p2 = pmax(model_a, model_b),
           model_a_wins = case_when(
             .data[[winner_col]] == "a" ~ 1,
             .data[[winner_col]] == "b" ~ 0,
             .data[[winner_col]] == "tie" ~ 0.5,
             TRUE ~ NA_real_
           ),
           p1_wins = if_else(model_a == p1, model_a_wins, 1 - model_a_wins)) %>%
    filter(!is.na(p1_wins)) %>%
    group_by(p1, p2) %>%
    summarize(wins1 = sum(p1_wins), wins2 = sum(1 - p1_wins), .groups = "drop")
}

# ---- Fisher z CI ----
fisher_ci <- function(r, n, conf = 0.95) {
  z <- atanh(r)
  se <- 1 / sqrt(n - 3)
  zcrit <- qnorm(1 - (1 - conf) / 2)
  list(lo = tanh(z - zcrit * se), hi = tanh(z + zcrit * se))
}

# ---- Williams' t-test for H0: corr(X, Y1) = corr(X, Y2), paired on n ----
williams_test <- function(r12, r13, r23, n) {
  # r12 = corr(X, Y1); r13 = corr(X, Y2); r23 = corr(Y1, Y2). Compares r12 vs r13.
  detR <- 1 - r12^2 - r13^2 - r23^2 + 2 * r12 * r13 * r23
  r_avg <- (r12 + r13) / 2
  num <- (r12 - r13) * sqrt((n - 1) * (1 + r23))
  den <- sqrt(2 * detR * (n - 1) / (n - 3) + r_avg^2 * (1 - r23)^3)
  t_stat <- num / den
  p <- 2 * pt(-abs(t_stat), df = n - 3)
  list(t = t_stat, p = p, df = n - 3)
}

# ---- Load data ----
preds <- read_csv("data/processed/arena_battle_predictions.csv", show_col_types = FALSE)
battles <- read_csv("data/processed/arena_battles_scored_input.csv", show_col_types = FALSE) %>%
  select(battle_id, model_a, model_b, human_winner = winner, category_llm)

preds_wide <- preds %>%
  select(battle_id, config, pred_winner) %>%
  pivot_wider(names_from = config, values_from = pred_winner)

df <- battles %>%
  mutate(human = case_when(
    human_winner == "model_a" ~ "a",
    human_winner == "model_b" ~ "b",
    human_winner %in% c("tie", "tie (bothbad)") ~ "tie",
    TRUE ~ NA_character_
  )) %>%
  inner_join(preds_wide, by = "battle_id")

# Target: same-sample human BT (column "human" when we fit BT on the human winner)
# Approaches compared: 4 pipelines only. Human is the reference, not a row.
APPROACHES <- c(
  "Likert (single)"    = "single_likert",
  "Likert (TEE)"       = "tee_likert",
  "Pairwise (single)"  = "single_pairwise",
  "Pairwise (TEE)"     = "tee_pairwise"
)
TARGET_SRC <- "human"  # column containing the target (human) BT

# ---- Fit BT for pipelines + human (target) on a subset, correlate with human ----
# Fit the human BT on the same battles (same MIN_BATTLES filter) as the target.
fit_per_approach <- function(df_sub, min_battles) {
  counts <- bind_rows(
    df_sub %>% select(m = model_a),
    df_sub %>% select(m = model_b)
  ) %>% count(m)
  keep <- counts$m[counts$n >= min_battles]
  if (length(keep) < 10) return(NULL)
  df_k <- df_sub %>% filter(model_a %in% keep, model_b %in% keep)

  # Fit pipelines + the human reference in a single pass
  srcs <- c(APPROACHES, human = TARGET_SRC)  # names(APPROACHES) appended with "human"
  map_dfr(names(srcs), function(label) {
    col <- srcs[[label]]
    pair_w <- aggregate_pair_wins(df_k, col)
    if (nrow(pair_w) == 0) return(NULL)
    pi <- fit_bt_mm(pair_w$p1, pair_w$p2, pair_w$wins1, pair_w$wins2)
    if (is.null(pi)) return(NULL)
    tibble(approach = label, model = names(pi), bt = as.numeric(pi))
  })
}

compute_corrs <- function(df_sub, min_battles, cat_label) {
  bt <- fit_per_approach(df_sub, min_battles)
  if (is.null(bt)) return(NULL)
  bt_wide <- bt %>% pivot_wider(names_from = approach, values_from = bt)
  if (!("human" %in% names(bt_wide))) return(NULL)

  map_dfr(names(APPROACHES), function(lab) {
    if (!(lab %in% names(bt_wide))) return(NULL)
    vals <- bt_wide[[lab]]
    ok <- !is.na(vals) & !is.na(bt_wide[["human"]])
    if (sum(ok) < 10) return(NULL)
    rho <- cor(vals[ok], bt_wide[["human"]][ok], method = "spearman")
    ci <- fisher_ci(rho, sum(ok))
    tibble(category = cat_label, approach = lab, n = sum(ok),
           rho = rho, lo = ci$lo, hi = ci$hi)
  })
}

categories <- list(
  list(label = "Overall",          data = df, min = 40),
  list(label = "Creative Writing", data = df %>% filter(category_llm == "creative_writing"), min = 20),
  list(label = "Persuasion",       data = df %>% filter(category_llm == "persuasion"),       min = 20),
  list(label = "Coding",           data = df %>% filter(category_llm == "coding"),           min = 20),
  list(label = "Factual QA",       data = df %>% filter(category_llm == "factual_qa"),       min = 20)
)

all_corrs <- map_dfr(categories, function(c) compute_corrs(c$data, c$min, c$label))

# ---- Williams' test: Pairwise (TEE) vs each other pipeline (Overall), vs human target ----
overall_bt <- fit_per_approach(df, 40)
overall_wide <- overall_bt %>% pivot_wider(names_from = approach, values_from = bt)
n_overall <- sum(!is.na(overall_wide[["human"]]))

tee_pw <- overall_wide[["Pairwise (TEE)"]]
human_v <- overall_wide[["human"]]

williams_results <- map_dfr(
  c("Likert (single)", "Likert (TEE)", "Pairwise (single)"),
  function(other) {
    vals <- overall_wide[[other]]
    ok <- !is.na(tee_pw) & !is.na(human_v) & !is.na(vals)
    r12 <- cor(tee_pw[ok], human_v[ok], method = "spearman")
    r13 <- cor(vals[ok],   human_v[ok], method = "spearman")
    r23 <- cor(tee_pw[ok], vals[ok],    method = "spearman")
    tt <- williams_test(r12, r13, r23, sum(ok))
    tibble(comparison = paste0("Pairwise (TEE) vs ", other),
           n = sum(ok), r_pw = r12, r_other = r13, r23 = r23,
           t = tt$t, df = tt$df, p = tt$p)
  }
)

cat("\n=== Overall correlations with same-sample human BT (Fisher 95% CI) ===\n")
print(all_corrs %>% filter(category == "Overall") %>%
        mutate(across(where(is.numeric), ~ round(., 3))))

cat("\n=== Williams' dependent-correlation test: Pairwise (TEE) vs other pipelines (Overall, n = ",
    n_overall, ") ===\n", sep = "")
print(williams_results %>% mutate(across(where(is.numeric), ~ round(., 4))))

cat("\n=== Per-category correlations ===\n")
print(all_corrs %>% mutate(across(where(is.numeric), ~ round(., 3))))

# ---- Save ----
write_csv(all_corrs, "data/processed/arena_bt_human_corr_bycategory.csv")
write_csv(williams_results, "data/processed/arena_bt_williams_overall.csv")

# ---- Plot ----
approach_order <- c("Likert (single)", "Likert (TEE)",
                    "Pairwise (single)", "Pairwise (TEE)")
category_order <- c("Overall", "Creative Writing", "Persuasion", "Coding", "Factual QA")

plot_df <- all_corrs %>%
  mutate(approach = factor(approach, levels = approach_order),
         category = factor(category, levels = category_order))

p <- ggplot(plot_df, aes(x = rho, y = approach)) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "gray75") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.8) +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.2f", rho)), hjust = -0.5, size = 3) +
  facet_wrap(~ category, ncol = 1, strip.position = "top") +
  scale_x_continuous(limits = c(-0.2, 1.1),
                     breaks = seq(-0.2, 1, 0.2),
                     expand = expansion(mult = c(0.01, 0.02))) +
  labs(
    title = "Leaderboard reconstruction: Spearman rho vs same-sample human BT",
    subtitle = "Each pipeline's 4,676-battle BT correlated with the human-vote BT on the same battles. Error bars: Fisher z 95% CI.",
    x = "Spearman rho vs human BT",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

ggsave("figures/fig_arena_bt_dotplot.pdf", p, width = 8, height = 10)
ggsave("figures/fig_arena_bt_dotplot.png", p, width = 8, height = 10, dpi = 300)

cat("\nSaved:\n")
cat("  data/processed/arena_bt_human_corr_bycategory.csv\n")
cat("  data/processed/arena_bt_williams_overall.csv\n")
cat("  figures/fig_arena_bt_dotplot.{pdf,png}\n")
