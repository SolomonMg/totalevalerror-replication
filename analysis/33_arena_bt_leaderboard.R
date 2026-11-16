#!/usr/bin/env Rscript
# 33_arena_bt_leaderboard.R
# Reconstruct a BT model leaderboard from each pipeline configuration's
# per-battle verdicts, then compare the ranking to the leaderboard derived
# from human votes on the same 1,500 battles.
#
# Directly addresses "does TEE-guided scoring reproduce Arena's leaderboard?"
# — the scientific output Arena actually produces, not per-battle agreement.
#
# Outputs:
#   data/processed/arena_bt_rankings.csv      — one row per (model, config)
#   data/processed/arena_bt_ranking_corr.csv  — per-config Spearman/Kendall vs human
#   figures/fig_arena_bt_leaderboard.pdf      — scatter: pipeline-BT vs human-BT

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

MIN_BATTLES_PER_MODEL <- 40  # drop rarely-seen models; unstable BT otherwise

# ---- Load battle-level predictions + model identities ----
preds <- read_csv("data/processed/arena_battle_predictions.csv", show_col_types = FALSE)
battles <- read_csv("data/processed/arena_battles_scored_input.csv", show_col_types = FALSE) %>%
  select(battle_id, model_a, model_b, human_winner = winner)

# Per-battle predictions are one row per (battle, config). Pivot to wide: one
# row per battle with columns for each config's prediction.
preds_wide <- preds %>%
  select(battle_id, config, pred_winner) %>%
  pivot_wider(names_from = config, values_from = pred_winner) %>%
  rename_with(~ sub("^", "pred_", .x), -battle_id)

# Normalize human winner to {a, b, tie}
normalize_winner <- function(w) case_when(
  w == "model_a" ~ "a",
  w == "model_b" ~ "b",
  w %in% c("tie", "tie (bothbad)") ~ "tie",
  TRUE ~ NA_character_
)

df <- battles %>%
  mutate(human = normalize_winner(human_winner)) %>%
  inner_join(preds_wide, by = "battle_id")

cat(sprintf("Joined battles: %d\n", nrow(df)))

# ---- Filter to models with enough battles ----
model_counts <- bind_rows(
  df %>% select(model = model_a),
  df %>% select(model = model_b)
) %>% count(model)
keep_models <- model_counts %>% filter(n >= MIN_BATTLES_PER_MODEL) %>% pull(model)
cat(sprintf("Models with >= %d battles: %d / %d\n",
            MIN_BATTLES_PER_MODEL, length(keep_models), nrow(model_counts)))

df_f <- df %>% filter(model_a %in% keep_models, model_b %in% keep_models)
cat(sprintf("Battles between retained models: %d\n", nrow(df_f)))

# ---- BT fit via MM algorithm (no external package needed) ----
# Takes a data frame with columns p1, p2, wins1, wins2 (counts) and returns
# a named numeric vector of BT strengths (pi_k), normalized to mean 1.
fit_bt_mm <- function(p1, p2, wins1, wins2, n_iter = 5000, tol = 1e-8) {
  players <- sort(unique(c(p1, p2)))
  K <- length(players)
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
    pi_new <- pi_new / mean(pi_new)  # normalize
    if (max(abs(pi_new - pi)) < tol) break
    pi <- pi_new
  }
  pi
}

# Build per-pair win counts from a winner column.
# winner_col_val should be one of: "a", "b", "tie". Ties split 0.5/0.5.
aggregate_pair_wins <- function(df_f, winner_col) {
  df_f %>%
    mutate(
      p1 = pmin(model_a, model_b),
      p2 = pmax(model_a, model_b),
      model_a_wins = case_when(
        .data[[winner_col]] == "a" ~ 1,
        .data[[winner_col]] == "b" ~ 0,
        .data[[winner_col]] == "tie" ~ 0.5,
        TRUE ~ NA_real_
      ),
      p1_wins = if_else(model_a == p1, model_a_wins, 1 - model_a_wins)
    ) %>%
    filter(!is.na(p1_wins)) %>%
    group_by(p1, p2) %>%
    summarize(
      wins1 = sum(p1_wins),
      wins2 = sum(1 - p1_wins),
      .groups = "drop"
    )
}

# ---- Fit BT for human + each pipeline config ----
configs <- c("human", "pred_single_likert", "pred_single_pairwise",
             "pred_tee_likert", "pred_tee_pairwise")

bt_results <- map_dfr(configs, function(col) {
  pair_w <- aggregate_pair_wins(df_f, col)
  if (nrow(pair_w) == 0) return(NULL)
  pi <- fit_bt_mm(pair_w$p1, pair_w$p2, pair_w$wins1, pair_w$wins2)
  tibble(model = names(pi), bt_strength = as.numeric(pi), config = col)
})

cat("\n=== BT leaderboard (top 10) per config ===\n")
bt_wide <- bt_results %>%
  pivot_wider(names_from = config, values_from = bt_strength)
bt_ranked <- bt_wide %>%
  mutate(rank_human = rank(-human)) %>%
  arrange(rank_human)

print(bt_ranked %>% head(15) %>%
        mutate(across(where(is.numeric), ~ round(., 3))))

# ---- Rank correlations ----
cat("\n=== Rank correlation between pipeline BT and human BT ===\n")
corrs <- map_dfr(c("pred_single_likert", "pred_single_pairwise",
                   "pred_tee_likert",    "pred_tee_pairwise"), function(c) {
  joined <- bt_wide %>% filter(!is.na(human), !is.na(.data[[c]]))
  sr <- cor(joined$human, joined[[c]], method = "spearman")
  kd <- cor(joined$human, joined[[c]], method = "kendall")
  pr <- cor(joined$human, joined[[c]], method = "pearson")
  tibble(config = c, n_models = nrow(joined),
         spearman = sr, kendall = kd, pearson = pr)
})
print(corrs %>% mutate(across(where(is.numeric), ~ round(., 3))))

# ---- Save ----
write_csv(bt_results, "data/processed/arena_bt_rankings.csv")
write_csv(corrs,      "data/processed/arena_bt_ranking_corr.csv")

# ---- Figure: scatter of pipeline-BT vs human-BT ----
plot_df <- bt_wide %>%
  pivot_longer(starts_with("pred_"), names_to = "config", values_to = "pipeline_bt") %>%
  filter(!is.na(human), !is.na(pipeline_bt)) %>%
  mutate(
    config_label = recode(config,
      "pred_single_likert"    = "Single-judge Likert",
      "pred_single_pairwise"  = "Single-judge pairwise",
      "pred_tee_likert"       = "TEE Likert",
      "pred_tee_pairwise"     = "TEE pairwise"
    )
  )

theme_tle <- theme_minimal(base_size = 11) + theme(
  panel.grid.minor = element_blank(),
  strip.text = element_text(face = "bold"),
  plot.title = element_text(face = "bold")
)

corr_text <- corrs %>% mutate(
  config_label = recode(config,
    "pred_single_likert"   = "Single-judge Likert",
    "pred_single_pairwise" = "Single-judge pairwise",
    "pred_tee_likert"      = "TEE Likert",
    "pred_tee_pairwise"    = "TEE pairwise"
  ),
  label = sprintf("Spearman rho = %.2f\nn = %d models", spearman, n_models)
)

p <- ggplot(plot_df, aes(x = human, y = pipeline_bt)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  geom_point(alpha = 0.7, size = 2) +
  geom_text(data = corr_text, aes(x = -Inf, y = Inf, label = label),
            hjust = -0.1, vjust = 1.4, size = 3.2, inherit.aes = FALSE) +
  facet_wrap(~ config_label, ncol = 2) +
  labs(
    title = "BT leaderboard reconstruction: pipeline vs Arena humans",
    subtitle = sprintf("Same 1,500 battles, %d models (>= %d battles each), MM algorithm. Dashed line y=x.",
                       length(keep_models), MIN_BATTLES_PER_MODEL),
    x = "Human-derived BT strength",
    y = "Pipeline-derived BT strength"
  ) + theme_tle

ggsave("figures/fig_arena_bt_leaderboard.pdf", p, width = 9, height = 8)
ggsave("figures/fig_arena_bt_leaderboard.png", p, width = 9, height = 8, dpi = 300)

cat("\nSaved:\n")
cat("  data/processed/arena_bt_rankings.csv\n")
cat("  data/processed/arena_bt_ranking_corr.csv\n")
cat("  figures/fig_arena_bt_leaderboard.pdf/png\n")

# ============================================================================
# Per-category leaderboard reconstruction
# ============================================================================
cat("\n=== Per-category BT ranking correlations ===\n")

df_cat <- df %>%
  left_join(read_csv("data/processed/arena_battles_scored_input.csv",
                     show_col_types = FALSE) %>%
              select(battle_id, category_llm), by = "battle_id") %>%
  filter(model_a %in% keep_models, model_b %in% keep_models)

TARGET_CATS <- c("creative_writing", "persuasion", "coding", "factual_qa")

per_cat_corr <- map_dfr(TARGET_CATS, function(cat) {
  sub <- df_cat %>% filter(category_llm == cat)
  cat_counts <- bind_rows(sub %>% select(m = model_a), sub %>% select(m = model_b)) %>%
    count(m) %>% filter(n >= 10)  # lower floor per-category due to smaller N
  sub <- sub %>% filter(model_a %in% cat_counts$m, model_b %in% cat_counts$m)
  if (nrow(sub) < 50) {
    return(tibble(category = cat, config = NA_character_, n_models = 0,
                  n_battles = nrow(sub), spearman = NA_real_))
  }
  map_dfr(configs, function(col) {
    pair_w <- aggregate_pair_wins(sub, col)
    if (nrow(pair_w) == 0) return(NULL)
    pi <- fit_bt_mm(pair_w$p1, pair_w$p2, pair_w$wins1, pair_w$wins2)
    tibble(category = cat, config = col,
           n_models = length(pi),
           n_battles = nrow(sub),
           model = names(pi),
           bt = as.numeric(pi))
  })
})

per_cat_wide <- per_cat_corr %>%
  filter(!is.na(model)) %>%
  pivot_wider(id_cols = c(category, n_battles, model), names_from = config, values_from = bt)

per_cat_summary <- per_cat_wide %>%
  group_by(category, n_battles) %>%
  summarize(
    n_models = sum(!is.na(human)),
    rho_single_likert   = cor(human, pred_single_likert,   method = "spearman", use = "p"),
    rho_single_pairwise = cor(human, pred_single_pairwise, method = "spearman", use = "p"),
    rho_tee_likert      = cor(human, pred_tee_likert,      method = "spearman", use = "p"),
    rho_tee_pairwise    = cor(human, pred_tee_pairwise,    method = "spearman", use = "p"),
    .groups = "drop"
  )

print(per_cat_summary %>% mutate(across(where(is.numeric), ~ round(., 3))))

write_csv(per_cat_summary, "data/processed/arena_bt_ranking_corr_per_category.csv")
cat("\nSaved: data/processed/arena_bt_ranking_corr_per_category.csv\n")
