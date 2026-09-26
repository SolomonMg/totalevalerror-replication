###############################################################################
## Script: 11_ground_truth_validation.R
## Purpose: Validate TEE pipeline optimization against external ground truth.
##
## The "aligned_to_whom" project estimated Bradley-Terry ideology positions
## for the same 4 models on the same 40 prompts, using pairwise comparisons
## with 6 anchor personas as reference points. These BT positions serve as
## an independent external benchmark (different scoring method, different
## judge models, different aggregation).
##
## This script tests whether TEE-optimal pipeline configurations (more
## prompts, more judges, averaged temperature) produce Likert-based ideology
## estimates that better recover the BT ranking than naive configurations.
##
## IMPORTANT: The TEE Likert scale measures a generic left-right dimension
## (1=progressive, 5=conservative). The BT data has dimension-specific
## positions. For economic_left_right and social_left_right, the directions
## align (higher BT lambda = more right = higher expected Likert). For
## authoritarian_libertarian, the BT dimension is orthogonal to left-right:
## libertarian positions (high BT lambda) are rated as progressive (low
## Likert score) on items like surveillance and free speech. This script
## accounts for this by sign-flipping the auth_lib dimension and by
## reporting dimension-specific results alongside the aggregate.
##
## Data In:
##   1) TEE Likert data: data/processed/likert_clean.csv
##   2) TEE item metadata: data/items_likert.csv
##   3) BT results: /Users/ns/workspace/aligned_to_whom/study1_ideology/
##      analysis/bt_results.RData (all_scores with model positions)
##
## Data Out:
##   1) data/processed/ground_truth_validation.csv
##   2) figures/ground_truth_validation.pdf
##   3) figures/ground_truth_validation.png
###############################################################################

library(tidyverse)
library(patchwork)

set.seed(42)

cat("=== Ground Truth Validation: TEE vs BT Ideology Positions ===\n\n")

# -------------------------------------------------------------------------
# 1. Load BT ground truth
# -------------------------------------------------------------------------
aligned_path <- Sys.getenv("ALIGNED_TO_WHOM_PATH", unset = "~/workspace/aligned_to_whom")
load(file.path(aligned_path, "study1_ideology/analysis/bt_results.RData"))

bt_models <- all_scores %>%
  filter(!is_anchor) %>%
  select(player, dimension, lambda, se) %>%
  as.data.frame()
names(bt_models) <- c("model", "dimension", "bt_lambda", "bt_se")

# Keep only models present in TEE (exclude gpt-4o)
tee_model_names <- c("claude-opus-4.5", "gpt-5.1", "deepseek-chat-v3.1", "grok-4.1-fast")
bt_models <- bt_models %>% filter(model %in% tee_model_names)

cat("BT ground truth (model positions by dimension):\n")
bt_models %>% arrange(dimension, bt_lambda) %>% print()

# -------------------------------------------------------------------------
# 2. Load TEE Likert data + item metadata
# -------------------------------------------------------------------------
likert <- read.csv("data/processed/likert_clean.csv", stringsAsFactors = FALSE)
items_meta <- read.csv("data/items_likert.csv", stringsAsFactors = FALSE) %>%
  select(item_id, evaluated_model, prompt_id) %>%
  distinct()

likert <- likert %>% left_join(items_meta, by = "item_id")

cat("\nTEE data:", nrow(likert), "observations,",
    length(unique(likert$item_id)), "items\n")

# Map TEE categories to BT dimensions
# economic_left_right -> economic_left_right (direct mapping, same sign)
# social_left_right -> social_left_right (direct mapping, same sign)
# authoritarian_libertarian -> authoritarian_libertarian (SIGN FLIP needed)
# cross_cutting, populist_elitist -> left_right (general dimension)
dim_map <- c(
  "economic_left_right"       = "economic_left_right",
  "social_left_right"         = "social_left_right",
  "authoritarian_libertarian" = "authoritarian_libertarian",
  "cross_cutting"             = "left_right",
  "populist_elitist"          = "left_right"
)

# Sign of the expected correlation: +1 for dimensions where higher BT lambda =
# higher expected Likert; -1 where the mapping is inverted
# auth_lib: high BT lambda = libertarian, but TEE Likert rates libertarian
# positions as progressive (low score) -> negative expected correlation
sign_map <- c(
  "economic_left_right"       = 1,
  "social_left_right"         = 1,
  "authoritarian_libertarian" = -1,
  "left_right"                = 1
)

likert$bt_dimension <- dim_map[likert$category]
stopifnot(sum(is.na(likert$bt_dimension)) == 0)

# -------------------------------------------------------------------------
# 3. Define pipeline configurations
# -------------------------------------------------------------------------
configs <- tribble(
  ~config_name, ~config_label, ~n_variants, ~n_judges, ~temp_strategy, ~n_reps,
  "naive_minimal",   "Naive: 1V, 1J, T=0, 1R",          1, 1, "fix_0",   1,
  "naive_3rep",      "Naive: 1V, 1J, T=0, 3R",          1, 1, "fix_0",   3,
  "naive_8rep",      "Naive: 1V, 1J, T=0, 8R",          1, 1, "fix_0",   8,
  "v3_1j_0t",        "+Variants: 3V, 1J, T=0, 8R",      3, 1, "fix_0",   8,
  "v5_1j_0t",        "+Variants: 5V, 1J, T=0, 8R",      5, 1, "fix_0",   8,
  "v5_3j_0t",        "+Judges: 5V, 3J, T=0, 8R",        5, 3, "fix_0",   8,
  "v5_3j_avg",       "TEE-optimal: 5V, 3J, avg(T), 8R", 5, 3, "avg",     8,
  "v5_3j_07",        "+Temp: 5V, 3J, T=0.7, 8R",        5, 3, "fix_07",  8,
)

judge_order <- c("openai/gpt-4o", "google/gemini-2.0-flash-001", "anthropic/claude-haiku-4.5")

# -------------------------------------------------------------------------
# 4. Helper: compute model x dimension estimates under a config
# -------------------------------------------------------------------------
compute_estimates <- function(data, config) {
  d <- data %>% filter(variant_id < config$n_variants)
  d <- d %>% filter(judge_model %in% judge_order[1:config$n_judges])

  if (config$temp_strategy == "fix_0") {
    d <- d %>% filter(temperature == 0)
  } else if (config$temp_strategy == "fix_07") {
    d <- d %>% filter(temperature == 0.7)
  }

  d <- d %>% filter(replication < config$n_reps)

  # Aggregate to model x bt_dimension level (not model x category),
  # so categories sharing a BT dimension (e.g. cross_cutting and
  # populist_elitist -> left_right) are properly pooled.
  d %>%
    group_by(evaluated_model, bt_dimension) %>%
    summarize(
      mean_score = mean(outcome, na.rm = TRUE),
      sd_score   = sd(outcome, na.rm = TRUE),
      n_obs      = n(),
      .groups    = "drop"
    ) %>%
    rename(model = evaluated_model)
}

# -------------------------------------------------------------------------
# 5. Compute sign-adjusted correlation metrics for each config
# -------------------------------------------------------------------------
# For dimension-specific analysis: within each dimension, correlate
# mean_score with bt_lambda. For auth_lib, the expected sign is negative.
# We report both raw and sign-adjusted correlations.

compute_metrics <- function(est, bt_ref, sign_map) {
  est_bt <- est %>%
    inner_join(bt_ref, by = c("model", "bt_dimension" = "dimension"))

  if (nrow(est_bt) == 0) return(NULL)

  # Dimension-specific
  by_dim <- est_bt %>%
    group_by(bt_dimension) %>%
    summarize(
      n_models     = n(),
      r_raw        = cor(mean_score, bt_lambda, method = "pearson"),
      tau_raw      = cor(mean_score, bt_lambda, method = "kendall"),
      expected_sign = sign_map[first(bt_dimension)],
      r_adjusted   = r_raw * expected_sign,
      tau_adjusted = tau_raw * expected_sign,
      .groups = "drop"
    )

  # Overall (sign-adjusted): flip bt_lambda sign for auth_lib before pooling
  est_bt$bt_lambda_adj <- est_bt$bt_lambda * sign_map[est_bt$bt_dimension]

  overall <- est_bt %>%
    summarize(
      bt_dimension  = "overall_adjusted",
      n_models      = n(),
      r_raw         = cor(mean_score, bt_lambda),
      tau_raw       = cor(mean_score, bt_lambda, method = "kendall"),
      expected_sign = NA_real_,
      r_adjusted    = cor(mean_score, bt_lambda_adj),
      tau_adjusted  = cor(mean_score, bt_lambda_adj, method = "kendall")
    )

  # Overall raw (no sign adjustment, for comparison)
  overall_raw <- est_bt %>%
    summarize(
      bt_dimension  = "overall_raw",
      n_models      = n(),
      r_raw         = cor(mean_score, bt_lambda),
      tau_raw       = cor(mean_score, bt_lambda, method = "kendall"),
      expected_sign = NA_real_,
      r_adjusted    = NA_real_,
      tau_adjusted  = NA_real_
    )

  bind_rows(by_dim, overall, overall_raw)
}

results <- list()
for (i in seq_len(nrow(configs))) {
  cfg <- configs[i, ]
  est <- compute_estimates(likert, cfg)
  metrics <- compute_metrics(est, bt_models, sign_map)
  if (is.null(metrics)) next

  metrics <- metrics %>%
    mutate(
      config_name   = cfg$config_name,
      config_label  = cfg$config_label,
      n_variants    = cfg$n_variants,
      n_judges      = cfg$n_judges,
      temp_strategy = cfg$temp_strategy,
      n_reps        = cfg$n_reps,
      n_obs_total   = sum(est$n_obs)
    )
  results[[i]] <- metrics
}

results_df <- bind_rows(results)

cat("\n=== Overall sign-adjusted correlation by config ===\n")
results_df %>%
  filter(bt_dimension == "overall_adjusted") %>%
  select(config_label, r_adjusted, tau_adjusted, n_obs_total) %>%
  arrange(desc(r_adjusted)) %>%
  print(n = 20)

cat("\n=== Dimension-specific (TEE-optimal vs Naive) ===\n")
results_df %>%
  filter(config_name %in% c("naive_minimal", "v5_3j_avg"),
         !grepl("overall", bt_dimension)) %>%
  select(config_label, bt_dimension, r_raw, tau_raw, expected_sign) %>%
  arrange(bt_dimension, config_label) %>%
  print(n = 20)

# -------------------------------------------------------------------------
# 6. Rank agreement per dimension
# -------------------------------------------------------------------------
cat("\n=== Rank agreement per dimension ===\n")

rank_analysis <- function(data, config, bt_ref) {
  est <- compute_estimates(data, config) %>%
    inner_join(bt_ref, by = c("model", "bt_dimension" = "dimension"))

  est %>%
    group_by(bt_dimension) %>%
    summarize(
      ranking_likert = paste(model[order(mean_score)], collapse = " < "),
      ranking_bt     = paste(model[order(bt_lambda)], collapse = " < "),
      exact_match    = identical(model[order(mean_score)], model[order(bt_lambda)]),
      tau            = cor(mean_score, bt_lambda, method = "kendall"),
      .groups = "drop"
    )
}

naive_cfg <- list(n_variants = 1, n_judges = 1, temp_strategy = "fix_0", n_reps = 1)
optimal_cfg <- list(n_variants = 5, n_judges = 3, temp_strategy = "avg", n_reps = 8)

cat("\nNaive (1V, 1J, T=0, 1R):\n")
rank_analysis(likert, naive_cfg, bt_models) %>% as.data.frame() %>% print()

cat("\nTEE-optimal (5V, 3J, avg(T), 8R):\n")
rank_analysis(likert, optimal_cfg, bt_models) %>% as.data.frame() %>% print()

# -------------------------------------------------------------------------
# 7. Bootstrap test of correlation difference (sign-adjusted)
# -------------------------------------------------------------------------
cat("\n=== Bootstrap: TEE-optimal vs Naive (sign-adjusted r) ===\n")

naive_est_bt <- compute_estimates(likert, naive_cfg) %>%
  inner_join(bt_models, by = c("model", "bt_dimension" = "dimension")) %>%
  mutate(bt_lambda_adj = bt_lambda * sign_map[bt_dimension])

opt_est_bt <- compute_estimates(likert, optimal_cfg) %>%
  inner_join(bt_models, by = c("model", "bt_dimension" = "dimension")) %>%
  mutate(bt_lambda_adj = bt_lambda * sign_map[bt_dimension])

r_naive_obs <- cor(naive_est_bt$mean_score, naive_est_bt$bt_lambda_adj)
r_opt_obs   <- cor(opt_est_bt$mean_score, opt_est_bt$bt_lambda_adj)

cat(sprintf("  Naive r (sign-adj)   = %.3f\n", r_naive_obs))
cat(sprintf("  Optimal r (sign-adj) = %.3f\n", r_opt_obs))
cat(sprintf("  Diff                 = %.3f\n", r_opt_obs - r_naive_obs))

# Item-level bootstrap
items_unique <- unique(likert$item_id)
n_boot <- 2000
boot_diffs <- numeric(n_boot)

for (b in seq_len(n_boot)) {
  boot_items <- sample(items_unique, replace = TRUE)
  boot_freq <- data.frame(item_id = names(table(boot_items)),
                          weight = as.numeric(table(boot_items)),
                          stringsAsFactors = FALSE)

  # Naive
  naive_sub <- likert %>%
    filter(variant_id == 0, judge_model == judge_order[1],
           temperature == 0, replication == 0) %>%
    inner_join(boot_freq, by = "item_id") %>%
    group_by(evaluated_model, bt_dimension) %>%
    summarize(mean_score = weighted.mean(outcome, weight, na.rm = TRUE),
              .groups = "drop") %>%
    rename(model = evaluated_model) %>%
    inner_join(bt_models, by = c("model", "bt_dimension" = "dimension")) %>%
    mutate(bt_lambda_adj = bt_lambda * sign_map[bt_dimension])

  # Optimal
  opt_sub <- likert %>%
    inner_join(boot_freq, by = "item_id") %>%
    group_by(evaluated_model, bt_dimension) %>%
    summarize(mean_score = weighted.mean(outcome, weight, na.rm = TRUE),
              .groups = "drop") %>%
    rename(model = evaluated_model) %>%
    inner_join(bt_models, by = c("model", "bt_dimension" = "dimension")) %>%
    mutate(bt_lambda_adj = bt_lambda * sign_map[bt_dimension])

  if (nrow(naive_sub) >= 3 && nrow(opt_sub) >= 3) {
    boot_diffs[b] <- cor(opt_sub$mean_score, opt_sub$bt_lambda_adj) -
                     cor(naive_sub$mean_score, naive_sub$bt_lambda_adj)
  } else {
    boot_diffs[b] <- NA
  }
}

boot_diffs <- boot_diffs[!is.na(boot_diffs)]
ci <- quantile(boot_diffs, c(0.025, 0.975))
p_val <- mean(boot_diffs <= 0)

cat(sprintf("  Bootstrap 95%% CI: [%.3f, %.3f]\n", ci[1], ci[2]))
cat(sprintf("  p-value (one-sided, optimal > naive): %.3f\n", p_val))
cat(sprintf("  n_boot = %d\n", length(boot_diffs)))

# -------------------------------------------------------------------------
# 8. Stability analysis: SD of correlation across random pipeline draws
# -------------------------------------------------------------------------
cat("\n=== Stability: correlation variance under random single-draw pipelines ===\n")

set.seed(42)
n_draws <- 1000
naive_corrs <- numeric(n_draws)

for (d in seq_len(n_draws)) {
  sub <- likert %>%
    filter(
      variant_id == sample(0:4, 1),
      judge_model == sample(judge_order, 1),
      temperature == sample(c(0, 0.7, 1), 1),
      replication == sample(0:7, 1)
    )

  est <- sub %>%
    group_by(evaluated_model, bt_dimension) %>%
    summarize(mean_score = mean(outcome, na.rm = TRUE), .groups = "drop") %>%
    rename(model = evaluated_model) %>%
    inner_join(bt_models, by = c("model", "bt_dimension" = "dimension")) %>%
    mutate(bt_lambda_adj = bt_lambda * sign_map[bt_dimension])

  naive_corrs[d] <- if (nrow(est) >= 3) cor(est$mean_score, est$bt_lambda_adj) else NA
}

naive_corrs <- naive_corrs[!is.na(naive_corrs)]

cat(sprintf("  Random single-draw: mean r = %.3f, SD = %.3f, 95%% range: [%.3f, %.3f]\n",
            mean(naive_corrs), sd(naive_corrs),
            quantile(naive_corrs, 0.025), quantile(naive_corrs, 0.975)))
cat(sprintf("  TEE-optimal: r = %.3f (deterministic)\n", r_opt_obs))
cat(sprintf("  Percentile of TEE-optimal in naive distribution: %.1f%%\n",
            100 * mean(naive_corrs <= r_opt_obs)))

# -------------------------------------------------------------------------
# 9. Save results
# -------------------------------------------------------------------------
write.csv(results_df, "data/processed/ground_truth_validation.csv", row.names = FALSE)
cat("\nSaved data/processed/ground_truth_validation.csv\n")

# Also save stability draws
write.csv(
  data.frame(draw = seq_along(naive_corrs), r_sign_adj = naive_corrs),
  "data/processed/ground_truth_stability_draws.csv", row.names = FALSE
)

# -------------------------------------------------------------------------
# 10. Visualization
# -------------------------------------------------------------------------

# Panel A: Sign-adjusted correlation by pipeline config
plot_a_data <- results_df %>%
  filter(bt_dimension == "overall_adjusted") %>%
  mutate(config_label = fct_reorder(config_label, r_adjusted))

p_a <- ggplot(plot_a_data, aes(x = r_adjusted, y = config_label)) +
  geom_point(size = 3, color = "steelblue") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  scale_x_continuous(limits = c(0, NA)) +
  labs(
    x = "Sign-adjusted Pearson r with BT positions",
    y = NULL,
    title = "A. Overall accuracy by pipeline configuration"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(size = 11, face = "bold"),
    axis.text.y = element_text(size = 9)
  )

# Panel B: Dimension-specific scatter (TEE-optimal only)
opt_est <- compute_estimates(likert, optimal_cfg) %>%
  inner_join(bt_models, by = c("model", "bt_dimension" = "dimension")) %>%
  mutate(
    bt_lambda_adj = bt_lambda * sign_map[bt_dimension],
    dim_label = recode(bt_dimension,
      "economic_left_right" = "Economic L-R",
      "social_left_right" = "Social L-R",
      "authoritarian_libertarian" = "Auth-Lib (flipped)",
      "left_right" = "General L-R"
    )
  )

p_b <- ggplot(opt_est, aes(x = bt_lambda_adj, y = mean_score,
                            color = model, shape = dim_label)) +
  geom_point(size = 3.5, alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, color = "grey40",
              linewidth = 0.5, linetype = "dashed") +
  labs(
    x = "BT position (sign-adjusted: higher = more conservative)",
    y = "TEE-optimal mean Likert (1=progressive, 5=conservative)",
    title = "B. TEE-optimal vs BT (all dimensions)",
    color = "Evaluated model", shape = "Dimension"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(size = 11, face = "bold"),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.text = element_text(size = 8)
  ) +
  guides(color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))

# Panel C: Stability histogram
stab_df <- data.frame(r = naive_corrs)

p_c <- ggplot(stab_df, aes(x = r)) +
  geom_histogram(bins = 40, fill = "steelblue", alpha = 0.6, color = "white") +
  geom_vline(xintercept = r_opt_obs, color = "red", linewidth = 1) +
  annotate("text", x = r_opt_obs + 0.01, y = Inf,
           label = sprintf("TEE-optimal: r = %.3f", r_opt_obs),
           vjust = 2, hjust = 0, color = "red", size = 3.5) +
  labs(
    x = "Sign-adjusted r with BT positions",
    y = sprintf("Count (%d random draws)", length(naive_corrs)),
    title = "C. Stability: random single-draw pipeline vs TEE-optimal"
  ) +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(size = 11, face = "bold"))

# Panel D: Dimension-specific rank agreement (naive vs optimal)
naive_ranks <- rank_analysis(likert, naive_cfg, bt_models) %>%
  mutate(config = "Naive")
opt_ranks <- rank_analysis(likert, optimal_cfg, bt_models) %>%
  mutate(config = "TEE-optimal")
rank_df <- bind_rows(naive_ranks, opt_ranks)

p_d <- ggplot(rank_df, aes(x = tau, y = bt_dimension, color = config)) +
  geom_point(size = 3, position = position_dodge(width = 0.3)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  scale_x_continuous(limits = c(-1, 1)) +
  labs(
    x = "Kendall's tau (rank agreement with BT)",
    y = NULL,
    title = "D. Rank agreement by dimension",
    color = "Config"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(size = 11, face = "bold"),
    legend.position = "bottom"
  )

# Combine
combined <- (p_a | p_b) / (p_c | p_d) +
  plot_annotation(
    title = "Validation: TEE Pipeline Optimization vs External BT Ideology Benchmark",
    subtitle = paste0(
      "BT positions from pairwise comparisons with anchor personas (separate study, ",
      "same prompts + models). ",
      "Auth-Lib sign-flipped (libertarian = progressive on Likert scale)."
    ),
    theme = theme(
      plot.title = element_text(size = 13, face = "bold"),
      plot.subtitle = element_text(size = 9, color = "grey30")
    )
  )

ggsave("figures/ground_truth_validation.pdf", combined,
       width = 13, height = 10)
cat("Saved figures/ground_truth_validation.pdf\n")

ggsave("figures/ground_truth_validation.png", combined,
       width = 13, height = 10, dpi = 200)
cat("Saved figures/ground_truth_validation.png\n")

# -------------------------------------------------------------------------
# 11. Summary table for paper
# -------------------------------------------------------------------------
cat("\n==================== SUMMARY FOR PAPER ====================\n")
cat("\nDimension-specific rank agreement (TEE-optimal, Kendall's tau):\n")
opt_ranks %>% select(bt_dimension, tau, exact_match, ranking_likert, ranking_bt) %>%
  as.data.frame() %>% print()

cat(sprintf("\nOverall sign-adjusted r: %.3f (TEE-optimal) vs %.3f (naive mean, SD=%.3f)\n",
            r_opt_obs, mean(naive_corrs), sd(naive_corrs)))
cat(sprintf("TEE-optimal is at the %.1fth percentile of naive draws\n",
            100 * mean(naive_corrs <= r_opt_obs)))

# Count exact rank matches
cat(sprintf("\nExact rank match: %d/4 dimensions (TEE-optimal) vs %d/4 (naive)\n",
            sum(opt_ranks$exact_match), sum(naive_ranks$exact_match)))

cat("\n=== Done ===\n")
