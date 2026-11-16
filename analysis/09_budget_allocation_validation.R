#!/usr/bin/env Rscript
# =============================================================================
# 09_budget_allocation_validation.R
#
# Ground-truth validation: does TEE-informed budget allocation beat naive
# allocation on MMLU exact-match data (where ground truth exists)?
#
# Subsamples from the existing 72K-call MMLU dataset. No new API calls.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

set.seed(42)

# --- CLI args ----------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
N_SIM <- if ("--nsim" %in% args) {
  as.integer(args[which(args == "--nsim") + 1])
} else {
  1000L
}
cat(sprintf("Budget allocation validation: N_SIM = %d\n", N_SIM))

# --- Load data ---------------------------------------------------------------
mmlu <- read_csv("data/processed/mmlu_clean.csv", show_col_types = FALSE) %>%
  filter(!is.na(outcome))

vc_raw <- read_csv("data/processed/variance_components_mmlu.csv",
                    show_col_types = FALSE)

cat(sprintf("MMLU data: %d observations, %d items, %d variants, %d SUTs, %d temps\n",
            nrow(mmlu),
            n_distinct(mmlu$item_id),
            n_distinct(mmlu$variant_id),
            n_distinct(mmlu$sut_model),
            n_distinct(mmlu$temperature)))

# --- Extract variance components ---------------------------------------------
get_vc <- function(comp) {
  val <- vc_raw %>% filter(component == comp) %>% pull(variance)
  if (length(val) == 0) 0 else val
}

s2_cat   <- get_vc("category")
s2_item  <- get_vc("item_id")
s2_prompt <- get_vc("variant_id")
s2_ip    <- get_vc("item_id:variant_id")
s2_is    <- get_vc("item_id:sut_model")
s2_ps    <- get_vc("variant_id:sut_model")
s2_it    <- get_vc("item_id:temperature")
s2_pt    <- get_vc("variant_id:temperature")
s2_gen   <- get_vc("Residual")

n_cats <- n_distinct(mmlu$category)

cat(sprintf("Variance components loaded. Total = %.5f\n",
            s2_cat + s2_item + s2_prompt + s2_ip + s2_is + s2_ps +
              s2_it + s2_pt + s2_gen))

# --- D-study formula (fixed SUT, fixed temp) ---------------------------------
dstudy_var <- function(N, K, R) {
  s2_cat / n_cats + s2_item / N + s2_prompt / K +
    s2_ip / (N * K) +
    s2_is / N + s2_ps / K +
    s2_it / N + s2_pt / K +
    s2_gen / (N * K * R)
}

# --- Theme -------------------------------------------------------------------
theme_tle <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray40", size = 9),
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 10)
  )

strategy_colors <- c(
  "Naive: 1 prompt, max reps"     = "#b2182b",
  "Naive: 1 prompt, R=5"          = "#d6604d",
  "TEE: max prompts, R=1"         = "#2166ac",
  "TEE: max items, R=1"           = "#4393c3"
)

# --- Define allocation strategies --------------------------------------------
get_allocations <- function(B) {
  # All items available
  N_max <- 200L
  V_max <- 5L
  R_max <- 8L

  tibble::tribble(
    ~strategy,                        ~N,                         ~V,  ~R,
    "Naive: 1 prompt, max reps",      min(N_max, B),              1L,  max(1L, as.integer(floor(B / min(N_max, B)))),
    "Naive: 1 prompt, R=5",           min(N_max, as.integer(floor(B / 5))), 1L, 5L,
    "TEE: max prompts, R=1",          min(N_max, as.integer(floor(B / V_max))), V_max, 1L,
    "TEE: max items, R=1",            min(N_max, B),              1L,  1L
  ) %>%
    # Ensure R doesn't exceed available
    mutate(R = pmin(R, R_max),
           # Actual budget
           actual_B = N * V * R)
}

# --- Subsampling function ----------------------------------------------------
subsample_data <- function(df, items_all, variants_all, N, V, R, seed) {
  set.seed(seed)

  # Sample items stratified by category
  n_per_cat <- N %/% n_distinct(items_all$category)
  sampled_items <- items_all %>%
    group_by(category) %>%
    mutate(.row_order = sample(n())) %>%
    filter(.row_order <= n_per_cat) %>%
    ungroup() %>%
    pull(item_id)

  # If we need more items than categories allow evenly, top up
  if (length(sampled_items) < N) {
    remaining <- setdiff(items_all$item_id, sampled_items)
    extra <- sample(remaining, min(N - length(sampled_items), length(remaining)))
    sampled_items <- c(sampled_items, extra)
  }

  # Sample variants
  sampled_variants <- sample(variants_all, min(V, length(variants_all)))

  # Sample replications
  sampled_reps <- sample(0:7, min(R, 8L))

  df %>%
    filter(item_id %in% sampled_items,
           variant_id %in% sampled_variants,
           replication %in% sampled_reps)
}

# --- Experiment A: MSE and CI Coverage at Fixed Budget -----------------------
cat("\n=== Experiment A: Budget Allocation Comparison ===\n")

# Use all SUT x temp combinations
sut_temp_combos <- mmlu %>%
  distinct(sut_model, temperature)

items_info <- mmlu %>% distinct(item_id, category)
variants_all <- sort(unique(mmlu$variant_id))

budgets <- c(100L, 200L, 400L, 600L, 1000L)

csv_path <- "data/processed/budget_allocation_results.csv"

if (file.exists(csv_path)) {
  cat(sprintf("Loading cached results from %s\n", csv_path))
  results <- read_csv(csv_path, show_col_types = FALSE)
} else {
  results <- list()
  total_iters <- nrow(sut_temp_combos) * length(budgets) * 4 * N_SIM
  cat(sprintf("Total iterations: %s\n", format(total_iters, big.mark = ",")))

  pb <- txtProgressBar(min = 0, max = nrow(sut_temp_combos) * length(budgets),
                       style = 3)
  pb_i <- 0

  for (st_idx in seq_len(nrow(sut_temp_combos))) {
    sut_name <- sut_temp_combos$sut_model[st_idx]
    temp_val <- sut_temp_combos$temperature[st_idx]

    # Filter data for this SUT x temp
    df_st <- mmlu %>%
      filter(sut_model == sut_name, temperature == temp_val)

    # Oracle mean for this SUT x temp
    oracle_mean <- mean(df_st$outcome)

    for (B in budgets) {
      allocs <- get_allocations(B)

      for (a_idx in seq_len(nrow(allocs))) {
        strategy <- allocs$strategy[a_idx]
        N_alloc  <- allocs$N[a_idx]
        V_alloc  <- allocs$V[a_idx]
        R_alloc  <- allocs$R[a_idx]

        # Skip if allocation is degenerate
        if (N_alloc < 4 || V_alloc < 1 || R_alloc < 1) next

        sim_results <- numeric(N_SIM)
        se_naive    <- numeric(N_SIM)
        se_tee      <- numeric(N_SIM)

        tee_se_val <- sqrt(dstudy_var(N_alloc, V_alloc, R_alloc))

        for (s in seq_len(N_SIM)) {
          sub <- subsample_data(df_st, items_info, variants_all,
                                N_alloc, V_alloc, R_alloc,
                                seed = 42L + s)

          if (nrow(sub) < 10) next

          theta_hat <- mean(sub$outcome)
          sim_results[s] <- theta_hat

          # Naive SE: treat each item mean as an observation
          item_means <- sub %>%
            group_by(item_id) %>%
            summarise(m = mean(outcome), .groups = "drop")
          se_naive[s] <- sd(item_means$m) / sqrt(nrow(item_means))
          se_tee[s]   <- tee_se_val
        }

        # Compute metrics
        valid <- sim_results != 0 | se_naive != 0  # filter any skipped
        theta_hats <- sim_results[valid]
        se_n <- se_naive[valid]
        se_t <- se_tee[valid]

        # MSE
        mse <- mean((theta_hats - oracle_mean)^2)

        # CI coverage
        ci_naive_covers <- mean(abs(theta_hats - oracle_mean) <= 1.96 * se_n)
        ci_tee_covers   <- mean(abs(theta_hats - oracle_mean) <= 1.96 * se_t)

        # CI width
        ci_naive_width <- mean(2 * 1.96 * se_n)
        ci_tee_width   <- mean(2 * 1.96 * se_t)

        results[[length(results) + 1]] <- tibble(
          sut_model = sut_name,
          temperature = temp_val,
          budget = B,
          strategy = strategy,
          N_items = N_alloc,
          V_variants = V_alloc,
          R_reps = R_alloc,
          actual_budget = N_alloc * V_alloc * R_alloc,
          oracle_mean = oracle_mean,
          mean_estimate = mean(theta_hats),
          rmse = sqrt(mse),
          bias = mean(theta_hats) - oracle_mean,
          ci_coverage_naive = ci_naive_covers,
          ci_coverage_tee = ci_tee_covers,
          ci_width_naive = ci_naive_width,
          ci_width_tee = ci_tee_width,
          n_valid = sum(valid)
        )
      }

      pb_i <- pb_i + 1
      setTxtProgressBar(pb, pb_i)
    }
  }
  close(pb)

  results <- bind_rows(results)
  write_csv(results, csv_path)
  cat(sprintf("Results saved to %s (%d rows)\n", csv_path, nrow(results)))
}

# --- Summary table -----------------------------------------------------------
cat("\n=== Summary: RMSE by Strategy x Budget (averaged over SUT x Temp) ===\n")
summary_tbl <- results %>%
  group_by(strategy, budget) %>%
  summarise(
    mean_rmse = mean(rmse),
    mean_coverage_naive = mean(ci_coverage_naive),
    mean_coverage_tee = mean(ci_coverage_tee),
    mean_width_naive = mean(ci_width_naive),
    mean_width_tee = mean(ci_width_tee),
    .groups = "drop"
  )

print(summary_tbl, n = 40)

# --- Relative RMSE improvement -----------------------------------------------
cat("\n=== TEE-Optimal vs Naive-MaxReps RMSE ratio ===\n")
rmse_comparison <- summary_tbl %>%
  select(strategy, budget, mean_rmse) %>%
  pivot_wider(names_from = strategy, values_from = mean_rmse)

if ("TEE: max prompts, R=1" %in% names(rmse_comparison) &&
    "Naive: 1 prompt, max reps" %in% names(rmse_comparison)) {
  rmse_comparison <- rmse_comparison %>%
    mutate(rmse_reduction_pct = (1 - `TEE: max prompts, R=1` /
                                   `Naive: 1 prompt, max reps`) * 100)
  print(rmse_comparison)
}

# --- Figure 1: Two-panel main result -----------------------------------------
cat("\n=== Generating figures ===\n")

# Panel A: RMSE vs budget
p_rmse <- summary_tbl %>%
  ggplot(aes(x = budget, y = mean_rmse, color = strategy, shape = strategy)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  scale_x_log10(labels = comma) +
  scale_color_manual(values = strategy_colors) +
  labs(
    title = "RMSE vs. budget",
    subtitle = "TEE-informed allocation reduces estimation error",
    x = "Budget (API calls per SUT-temperature cell)",
    y = "RMSE (vs. oracle mean)",
    color = NULL, shape = NULL
  ) +
  theme_tle +
  theme(legend.position = "none")

# Panel B: CI coverage vs budget
coverage_long <- summary_tbl %>%
  select(strategy, budget, naive = mean_coverage_naive,
         TEE = mean_coverage_tee) %>%
  pivot_longer(cols = c(naive, TEE), names_to = "SE_type",
               values_to = "coverage")

p_coverage <- summary_tbl %>%
  ggplot(aes(x = budget, y = mean_coverage_naive,
             color = strategy, shape = strategy)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "gray50") +
  scale_x_log10(labels = comma) +
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  scale_color_manual(values = strategy_colors) +
  labs(
    title = "CI coverage (naive SE)",
    subtitle = "Naive CIs miss the oracle mean; multi-prompt designs improve coverage",
    x = "Budget (API calls per SUT-temperature cell)",
    y = "95% CI coverage of oracle mean",
    color = NULL, shape = NULL
  ) +
  theme_tle

fig_main <- p_rmse + p_coverage +
  plot_layout(widths = c(1, 1)) +
  plot_annotation(
    title = "Budget allocation validation on MMLU (ground-truth accuracy)",
    subtitle = sprintf("4 strategies × 5 budgets × 9 SUT-temperature cells × %s Monte Carlo replicates",
                       format(N_SIM, big.mark = ",")),
    theme = theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(color = "gray40", size = 10)
    )
  )

ggsave("figures/fig_budget_allocation.pdf", fig_main,
       width = 11, height = 5, device = cairo_pdf)
ggsave("figures/fig_budget_allocation.png", fig_main,
       width = 11, height = 5, dpi = 300)
cat("Saved figures/fig_budget_allocation.{pdf,png}\n")

# --- TEE vs Naive coverage comparison ----------------------------------------
# Show that TEE-calibrated CIs maintain coverage while naive CIs don't
p_coverage_compare <- summary_tbl %>%
  select(strategy, budget, Naive = mean_coverage_naive,
         TEE = mean_coverage_tee) %>%
  pivot_longer(cols = c(Naive, TEE), names_to = "SE_method",
               values_to = "coverage") %>%
  ggplot(aes(x = budget, y = coverage, color = strategy,
             linetype = SE_method)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "gray50") +
  scale_x_log10(labels = comma) +
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  scale_color_manual(values = strategy_colors) +
  scale_linetype_manual(values = c("Naive" = "dashed", "TEE" = "solid")) +
  facet_wrap(~ strategy, nrow = 1) +
  labs(
    title = "CI coverage: naive SE vs. TEE SE by strategy",
    subtitle = "Solid = TEE-calibrated SE, Dashed = naive (item-mean) SE. Horizontal line = 95%.",
    x = "Budget", y = "Coverage",
    color = NULL, linetype = "SE method"
  ) +
  theme_tle +
  theme(legend.position = "bottom")

ggsave("figures/fig_budget_coverage_detail.pdf", p_coverage_compare,
       width = 12, height = 4.5, device = cairo_pdf)
ggsave("figures/fig_budget_coverage_detail.png", p_coverage_compare,
       width = 12, height = 4.5, dpi = 300)
cat("Saved figures/fig_budget_coverage_detail.{pdf,png}\n")

cat("\n=== Done ===\n")
