# 16_pairwise_vs_binary_recovery.R
# Empirical test: does pairwise BT recover item safety rankings better than
# binary classification at AILuminate's ~6% unsafe prevalence?
#
# Ground truth: per-item mean safe rate from full conventional study (360 obs/item)
# Binary: subsampled single-config slices (budget-matched to pairwise)
# Pairwise: BT scores fitted on pilot data
#
# Outputs:
#   data/processed/pairwise_vs_binary_recovery.csv
#   figures/fig_pairwise_vs_binary_recovery.pdf
#
# Usage:
#   Rscript analysis/16_pairwise_vs_binary_recovery.R

library(tidyverse)
library(BradleyTerry2)

set.seed(42)

cat("=== 16_pairwise_vs_binary_recovery.R ===\n")

# =========================================================================
# 1. Ground truth: full conventional study consensus
# =========================================================================
cat("\n--- Loading conventional study data ---\n")
conv <- read_csv("data/processed/safety_clean.csv", show_col_types = FALSE) %>%
  filter(!is.na(outcome))

ground_truth <- conv %>%
  group_by(item_id) %>%
  summarise(
    gt_safe_rate = mean(outcome),
    gt_n = n(),
    .groups = "drop"
  )

cat("Conventional study:", n_distinct(conv$item_id), "items,",
    nrow(conv), "observations\n")
cat("Mean safe rate:", round(mean(ground_truth$gt_safe_rate), 3), "\n")
cat("Unsafe prevalence:", round(1 - mean(ground_truth$gt_safe_rate), 3), "\n")

# =========================================================================
# 2. Pairwise BT scores
# =========================================================================
cat("\n--- Fitting Bradley-Terry on pairwise pilot ---\n")
pw <- read_csv("data/processed/safety_pairwise_clean.csv", show_col_types = FALSE) %>%
  filter(!is.na(outcome))

cat("Pairwise pilot:", nrow(pw), "valid comparisons\n")

# Recover original item IDs from pair data
# outcome=1 means "A is less safe", outcome=0 means "B is less safe"
# For BT: item chosen as "less safe" = "winner" in the unsafety ranking
# We want unsafety scores (higher = less safe)
pw_wins <- pw %>%
  mutate(
    # If A chosen as less safe (outcome=1): presented_a wins unsafety contest
    # If B chosen as less safe (outcome=0): presented_b wins unsafety contest
    winner = ifelse(outcome == 1, presented_a, presented_b),
    loser  = ifelse(outcome == 1, presented_b, presented_a)
  ) %>%
  filter(!is.na(winner), !is.na(loser))

cat("Valid win/loss records:", nrow(pw_wins), "\n")

# Get all unique items in pairwise data
pw_items <- sort(unique(c(pw_wins$winner, pw_wins$loser)))
cat("Unique items in pairwise:", length(pw_items), "\n")

# Aggregate to win counts per pair
bt_data <- pw_wins %>%
  group_by(winner, loser) %>%
  summarise(n_wins = n(), .groups = "drop")

# Also need reverse direction
bt_data_rev <- pw_wins %>%
  group_by(winner = loser, loser = winner) %>%
  summarise(n_wins = n(), .groups = "drop")

# Build contest matrix for BradleyTerry2
# Need a data frame with player1, player2, win1, win2
contests <- bt_data %>%
  rename(player1 = winner, player2 = loser, win1 = n_wins) %>%
  left_join(
    bt_data_rev %>% rename(player1 = winner, player2 = loser, win2 = n_wins),
    by = c("player1", "player2")
  ) %>%
  mutate(win2 = replace_na(win2, 0))

# Fit BT model
bt_mod <- BTm(
  outcome = cbind(win1, win2),
  player1 = factor(player1, levels = pw_items),
  player2 = factor(player2, levels = pw_items),
  data = contests
)

# Extract BT ability scores (higher = more unsafe)
bt_abilities <- BTabilities(bt_mod)
bt_scores <- tibble(
  item_id = rownames(bt_abilities),
  bt_unsafety = as.numeric(bt_abilities[, "ability"]),
  bt_se = as.numeric(bt_abilities[, "s.e."])
)

cat("BT model fitted:", nrow(bt_scores), "items with scores\n")
cat("BT score range:", round(range(bt_scores$bt_unsafety), 3), "\n")

# =========================================================================
# 3. Merge and compute pairwise recovery
# =========================================================================
cat("\n--- Computing pairwise recovery ---\n")

# Items in both studies
overlap_all <- bt_scores %>%
  inner_join(ground_truth, by = "item_id") %>%
  mutate(bt_safety = -bt_unsafety)

cat("Overlapping items (all):", nrow(overlap_all), "\n")
cat("  At ceiling (gt=1.0):", sum(overlap_all$gt_safe_rate == 1.0), "\n")
cat("  With variation (gt<1.0):", sum(overlap_all$gt_safe_rate < 1.0), "\n")

# Exclude ceiling items — ground truth has no variation to recover
overlap <- overlap_all %>% filter(gt_safe_rate < 1.0)
cat("\nUsing", nrow(overlap), "non-ceiling items for recovery comparison\n")

# Pairwise tau
pw_tau <- cor(overlap$bt_safety, overlap$gt_safe_rate, method = "kendall")
cat("Pairwise BT vs ground truth (Kendall tau):", round(pw_tau, 4), "\n")

# =========================================================================
# 4. Binary subsampling: budget-matched single-config slices
# =========================================================================
cat("\n--- Binary subsampling (1000 replicates) ---\n")

N_BOOT <- 1000
# Items that overlap with pairwise
overlap_items <- overlap$item_id

# Total pairwise calls used
pw_total_calls <- nrow(pw)

# Budget-matched binary: same total calls as pairwise, distributed across items
calls_per_item_matched <- ceiling(pw_total_calls / length(overlap_items))
cat("Budget-matched binary: ~", calls_per_item_matched, "obs/item\n")

binary_taus <- numeric(N_BOOT)
for (i in seq_len(N_BOOT)) {
  bpi <- calls_per_item_matched
  slice <- conv %>%
    filter(item_id %in% overlap_items) %>%
    group_by(item_id) %>%
    slice_sample(n = bpi) %>%
    summarise(bin_safe_rate = mean(outcome), .groups = "drop")

  merged <- slice %>%
    inner_join(ground_truth %>% filter(item_id %in% overlap_items),
               by = "item_id")

  binary_taus[i] <- cor(merged$bin_safe_rate, merged$gt_safe_rate, method = "kendall")
}

# Also report all-items tau for reference
pw_tau_all <- cor(overlap_all$bt_safety, overlap_all$gt_safe_rate, method = "kendall")
cat("(For reference, all-items tau including ceiling:", round(pw_tau_all, 4), ")\n")

cat("Binary subsampled tau: mean =", round(mean(binary_taus), 4),
    ", sd =", round(sd(binary_taus), 4), "\n")
cat("Pairwise BT tau:", round(pw_tau, 4), "\n")
cat("BT advantage:", round(pw_tau - mean(binary_taus), 4), "\n")
cat("BT percentile in binary distribution:",
    round(mean(pw_tau > binary_taus) * 100, 1), "%\n")

# =========================================================================
# 5. Budget sweep: binary recovery at multiple budget levels
# =========================================================================
cat("\n--- Budget sweep ---\n")

# Budget levels: from 1 obs/item to full study
budget_per_item <- c(1, 2, 3, 5, 10, 20, 50, 100, 200, 360)
budget_results <- list()

for (bpi in budget_per_item) {
  taus <- numeric(min(N_BOOT, 200))
  # Max available per item in the conventional data for overlap items
  max_per_item <- conv %>%
    filter(item_id %in% overlap_items) %>%
    count(item_id) %>%
    pull(n) %>%
    min()
  actual_bpi <- min(bpi, max_per_item)

  for (i in seq_along(taus)) {
    slice <- conv %>%
      filter(item_id %in% overlap_items) %>%
      group_by(item_id) %>%
      slice_sample(n = actual_bpi) %>%
      summarise(bin_safe_rate = mean(outcome), .groups = "drop")

    merged <- slice %>%
      inner_join(ground_truth %>% filter(item_id %in% overlap_items),
                 by = "item_id")

    tau_i <- tryCatch(
      cor(merged$bin_safe_rate, merged$gt_safe_rate, method = "kendall"),
      warning = function(w) NA_real_
    )
    taus[i] <- tau_i
  }

  taus_clean <- taus[!is.na(taus)]
  total_calls <- bpi * length(overlap_items)  # cost for non-ceiling items only
  budget_results[[length(budget_results) + 1]] <- tibble(
    method = "Binary classification",
    budget_per_item = bpi,
    total_calls = total_calls,
    tau_mean = mean(taus_clean),
    tau_sd = sd(taus_clean),
    tau_lower = quantile(taus_clean, 0.025),
    tau_upper = quantile(taus_clean, 0.975),
    n_valid = length(taus_clean)
  )

  cat(sprintf("  Binary @ %d obs/item (total=%d): tau=%.4f (%.4f-%.4f) [%d valid]\n",
              bpi, total_calls, mean(taus_clean),
              quantile(taus_clean, 0.025), quantile(taus_clean, 0.975),
              length(taus_clean)))
}

budget_df <- bind_rows(budget_results)

# Add pairwise BT as a point
bt_point <- tibble(
  method = "BT (pairwise)",
  budget_per_item = NA,
  total_calls = pw_total_calls,
  tau_mean = pw_tau,
  tau_sd = NA,
  tau_lower = NA,
  tau_upper = NA
)

all_results <- bind_rows(budget_df, bt_point)

# =========================================================================
# 6. Save results
# =========================================================================
out_path <- "data/processed/pairwise_vs_binary_recovery.csv"
write_csv(all_results, out_path)
cat("\nResults saved to:", out_path, "\n")

# Save per-item scores for inspection
item_scores_path <- "data/processed/pairwise_bt_item_scores.csv"
write_csv(overlap, item_scores_path)
cat("Per-item scores saved to:", item_scores_path, "\n")

# =========================================================================
# 7. Figure: Cost-efficiency frontier
# =========================================================================
cat("\n--- Generating figure ---\n")

theme_set(theme_bw(base_size = 11))

p <- ggplot() +
  # Binary: ribbon + line
  geom_ribbon(
    data = budget_df,
    aes(x = total_calls, ymin = tau_lower, ymax = tau_upper),
    fill = "#4575b4", alpha = 0.2
  ) +
  geom_line(
    data = budget_df,
    aes(x = total_calls, y = tau_mean, colour = "Binary classification"),
    linewidth = 0.9
  ) +
  geom_point(
    data = budget_df,
    aes(x = total_calls, y = tau_mean, colour = "Binary classification"),
    size = 2
  ) +
  # Pairwise BT: single point
  geom_point(
    data = bt_point,
    aes(x = total_calls, y = tau_mean, colour = "BT (pairwise)"),
    size = 4, shape = 17
  ) +
  geom_hline(
    yintercept = pw_tau,
    linetype = "dashed", colour = "#d73027", alpha = 0.5
  ) +
  scale_colour_manual(
    values = c("Binary classification" = "#4575b4", "BT (pairwise)" = "#d73027"),
    name = NULL
  ) +
  scale_x_log10(
    labels = scales::comma
  ) +
  labs(
    x = "Total API calls",
    y = expression("Rank recovery (" * tau * " vs. full-study consensus)"),
    caption = sprintf(
      paste0(
        "%d overlapping items between conventional and pairwise studies.\n",
        "Ground truth = per-item mean from full conventional study (360 obs/item).\n",
        "Binary: 200 bootstrap subsamples per budget level. ",
        "Pairwise: %s total calls, BT fit on pilot data."
      ),
      nrow(overlap),
      format(pw_total_calls, big.mark = ",")
    )
  ) +
  theme(
    legend.position = c(0.7, 0.25),
    legend.background = element_rect(fill = "white", colour = "grey80")
  )

fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

pdf_path <- file.path(fig_dir, "fig_pairwise_vs_binary_recovery.pdf")
png_path <- file.path(fig_dir, "fig_pairwise_vs_binary_recovery.png")

ggsave(pdf_path, p, width = 5, height = 4)
ggsave(png_path, p, width = 5, height = 4, dpi = 150)

cat("Saved:", pdf_path, "\n")
cat("Saved:", png_path, "\n")

cat("\n=== Done ===\n")
