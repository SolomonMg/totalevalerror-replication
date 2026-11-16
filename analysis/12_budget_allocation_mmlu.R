# 12_budget_allocation_mmlu.R
# Budget allocation simulation: Naive vs Standard vs TEE on MMLU data.
#
# Demonstrates three evaluation strategies:
#   Naive:    K=1, R=3, N=floor(B/3)  — common practice with replication waste
#   Standard: K=1, R=1, N=min(B, pool) — typical leaderboard single-shot
#   TEE:      D-study-optimized K/R/N  — TEE-informed evaluation
#
# Separates two effects:
#   Allocation:   Naive → Standard (don't waste budget on reps)
#   Optimization: Standard → TEE (D-study SE + multi-prompt at high budgets)
#
# Design:
#   - 30 holdout items for TEE pilot (lmer fit per SUT)
#   - 170 pool items for all researchers to sample from
#   - Oracle = pool mean (170 items) per SUT x temp cell
#   - Budget B = calls per SUT x temp cell
#
# Usage:
#   Rscript analysis/12_budget_allocation_mmlu.R              # full run (~2 min)
#   Rscript analysis/12_budget_allocation_mmlu.R --nsim 10    # smoke test

library(tidyverse)
library(lme4)
library(patchwork)

set.seed(42)

# --- Parse args ---
args <- commandArgs(trailingOnly = TRUE)
N_SIM <- 1000
N_CORES <- max(1, parallel::detectCores() - 1)
if ("--nsim" %in% args) {
  idx <- which(args == "--nsim")
  N_SIM <- as.integer(args[idx + 1])
}
if ("--cores" %in% args) {
  idx <- which(args == "--cores")
  N_CORES <- as.integer(args[idx + 1])
}
cat(sprintf("=== 12_budget_allocation_mmlu.R ===\nN_SIM=%d, cores=%d\n", N_SIM, N_CORES))

# --- Load data ---
df <- read_csv("data/processed/mmlu_clean.csv", show_col_types = FALSE) %>%
  filter(!is.na(outcome))

all_items    <- sort(unique(df$item_id))
all_variants <- sort(unique(df$variant_id))
V <- length(all_variants)

cat(sprintf("Data: %d rows, %d items, %d variants, %d SUTs, %d temps\n",
            nrow(df), length(all_items), V,
            n_distinct(df$sut_model), n_distinct(df$temperature)))

# --- Holdout: 30 items for TEE pilot ---
holdout_items <- sort(sample(all_items, 30))
pool_items    <- setdiff(all_items, holdout_items)
n_pool        <- length(pool_items)
cat(sprintf("Holdout: %d items, Pool: %d items\n", length(holdout_items), n_pool))

# --- Oracle: pool mean per SUT x temp cell (170 items only) ---
oracle_df <- df %>%
  filter(item_id %in% pool_items) %>%
  group_by(sut_model, temperature) %>%
  summarize(oracle_mean = mean(outcome), .groups = "drop")
cat("\nOracle means (pool items only):\n")
print(as.data.frame(oracle_df))

# --- Pilot: fit lmer per SUT on holdout items ---
# 30 items x 5 prompts x 3 temps x 8 reps = 3600 obs per SUT.
pilot_df <- df %>% filter(item_id %in% holdout_items)

cat("\n--- Pilot fits ---\n")
pilot_vcs <- list()
for (sut in sort(unique(df$sut_model))) {
  cat(sprintf("  %s: ", sut))
  psut <- pilot_df %>%
    filter(sut_model == sut) %>%
    mutate(temp_f = factor(temperature))

  # Add the cell-level (3-way+) random effect to separate s2_eps_cell from
  # within-cell replicate noise (s2_rho_rep / Residual).
  fit <- suppressWarnings(lmer(
    outcome ~ temp_f +
      (1 | item_id) + (1 | variant_id) +
      (1 | item_id:variant_id) +
      (1 | item_id:temp_f) + (1 | variant_id:temp_f) +
      (1 | item_id:variant_id:temp_f),  # cell-level (3-way+) for fixed SUT
    data = psut, REML = TRUE,
    control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
  ))

  vc <- as.data.frame(VarCorr(fit))
  s2 <- setNames(vc$vcov, vc$grp)
  pilot_vcs[[sut]] <- s2

  cat(sprintf("item=%.4f, variant=%.4f, iv=%.4f, it=%.4f, vt=%.4f, eps_cell=%.4f, rho_rep=%.4f\n",
              s2["item_id"], s2["variant_id"],
              s2["item_id:variant_id"],
              ifelse(is.na(s2["item_id:temp_f"]), 0, s2["item_id:temp_f"]),
              ifelse(is.na(s2["variant_id:temp_f"]), 0, s2["variant_id:temp_f"]),
              ifelse(is.na(s2["item_id:variant_id:temp_f"]), 0, s2["item_id:variant_id:temp_f"]),
              s2["Residual"]))
}

# --- TEE SE (Case 1: fixed SUT, fixed temp) ---
# Two residual components from Phase 3 split:
#   s2_eps_cell (= cell-level 3-way+) is constant within a cell, varies across
#     cells -> averages over (N items x K prompts) but NOT reducible by R.
#   s2_rho_rep (= Residual) is call-to-call within a fixed cell -> reducible by R.
tee_se <- function(vcs, N, K = 1, R = 1) {
  s2_i        <- vcs["item_id"]
  s2_v        <- vcs["variant_id"]
  s2_iv       <- vcs["item_id:variant_id"]
  s2_it       <- ifelse(is.na(vcs["item_id:temp_f"]), 0, vcs["item_id:temp_f"])
  s2_vt       <- ifelse(is.na(vcs["variant_id:temp_f"]), 0, vcs["variant_id:temp_f"])
  s2_eps_cell <- ifelse(is.na(vcs["item_id:variant_id:temp_f"]), 0,
                        vcs["item_id:variant_id:temp_f"])
  s2_rho_rep  <- vcs["Residual"]
  unname(sqrt(
    (s2_i + s2_it) / N +
      (s2_v + s2_vt) / K +
      s2_iv / (N * K) +
      s2_eps_cell / (N * K) +        # cell-level, no /R
      s2_rho_rep  / (N * K * R)      # replicate, with /R
  ))
}

# --- D-study optimal allocation ---
# Grid search over K ∈ 1..V_max, R ∈ 1..R_max to minimize D-study variance
# given total budget B and item pool size n_pool.
optimal_allocation <- function(vcs, B, n_pool, V_max = 5L, R_max = 8L) {
  s2_i        <- vcs["item_id"]
  s2_v        <- vcs["variant_id"]
  s2_iv       <- vcs["item_id:variant_id"]
  s2_it       <- ifelse(is.na(vcs["item_id:temp_f"]), 0, vcs["item_id:temp_f"])
  s2_vt       <- ifelse(is.na(vcs["variant_id:temp_f"]), 0, vcs["variant_id:temp_f"])
  s2_eps_cell <- ifelse(is.na(vcs["item_id:variant_id:temp_f"]), 0,
                        vcs["item_id:variant_id:temp_f"])
  s2_rho_rep  <- vcs["Residual"]

  best_var <- Inf
  best <- list(K = 1L, R = 1L, N = 1L)

  for (K in seq_len(V_max)) {
    for (R in seq_len(R_max)) {
      N <- min(floor(B / (K * R)), n_pool)
      if (N < 1L) next
      v <- (s2_i + s2_it) / N +
        (s2_v + s2_vt) / K +
        s2_iv / (N * K) +
        s2_eps_cell / (N * K) +        # cell-level, no /R
        s2_rho_rep  / (N * K * R)      # replicate, with /R
      if (v < best_var) {
        best_var <- v
        best <- list(K = as.integer(K), R = as.integer(R), N = as.integer(N))
      }
    }
  }
  best
}

# --- Print allocation table ---
cat("\nAllocation comparison:\n")
BUDGET_LEVELS <- c(50, 100, 200, 400, 700, 1000)
for (B in BUDGET_LEVELS) {
  N_naive <- min(floor(B / 3), n_pool)
  N_std   <- min(B, n_pool)
  cat(sprintf("  B=%4d: Naive N=%3d K=1 R=3 (%4d calls) | Standard N=%3d K=1 R=1 (%4d calls)",
              B, N_naive, N_naive * 3, N_std, N_std))
  # Show TEE allocation for each SUT
  for (sut in sort(names(pilot_vcs))) {
    opt <- optimal_allocation(pilot_vcs[[sut]], B, n_pool)
    sut_short <- sub("^.*/(.*)", "\\1", sut)
    cat(sprintf(" | TEE(%s) N=%d K=%d R=%d", sut_short, opt$N, opt$K, opt$R))
  }
  cat("\n")
}

# --- CSV caching ---
csv_path <- "data/processed/mmlu_budget_allocation.csv"

if (file.exists(csv_path)) {
  cat("\nLoading cached results from", csv_path, "\n")
  results_df <- read_csv(csv_path, show_col_types = FALSE)

} else {
  cat("\n--- Running simulation ---\n")

  cells <- df %>%
    distinct(sut_model, temperature) %>%
    arrange(sut_model, temperature)

  # Pre-build 3D arrays per cell: [item, variant, rep] for fast indexing
  pool_df <- df %>% filter(item_id %in% pool_items)
  cell_arrays <- list()
  for (ci in seq_len(nrow(cells))) {
    sut <- cells$sut_model[ci]; temp <- cells$temperature[ci]
    key <- paste(sut, temp, sep = "___")
    cd <- pool_df %>% filter(sut_model == sut, temperature == temp)
    arr <- array(NA_real_, dim = c(n_pool, V, 8))
    idx <- cbind(match(cd$item_id, pool_items),
                 match(cd$variant_id, all_variants),
                 cd$replication + 1L)
    arr[idx] <- cd$outcome
    cell_arrays[[key]] <- arr
  }

  all_results <- list()
  for (ci in seq_len(nrow(cells))) {
    sut  <- cells$sut_model[ci]
    temp <- cells$temperature[ci]
    key  <- paste(sut, temp, sep = "___")
    arr  <- cell_arrays[[key]]
    vcs  <- pilot_vcs[[sut]]
    oracle_val <- oracle_df %>%
      filter(sut_model == sut, temperature == temp) %>%
      pull(oracle_mean)

    # Pre-compute optimal allocations for TEE at each budget level
    tee_allocs <- lapply(BUDGET_LEVELS, function(B) {
      optimal_allocation(vcs, B, n_pool, V_max = V)  # cap K at the variants in the data
    })
    names(tee_allocs) <- as.character(BUDGET_LEVELS)

    cat(sprintf("  %s @ T=%s (oracle=%.4f): ", sut, temp, oracle_val))

    run_one <- function(sim_id) {
      set.seed(42 + sim_id + ci * 100000)
      out <- vector("list", length(BUDGET_LEVELS))

      for (bi in seq_along(BUDGET_LEVELS)) {
        B <- BUDGET_LEVELS[bi]

        # --- Naive: floor(B/3) items, 1 prompt, 3 reps ---
        N_n  <- min(floor(B / 3), n_pool)
        R_n  <- 3L
        pn   <- sample.int(V, 1)
        items_n <- sample.int(n_pool, N_n)
        naive_means <- vapply(items_n, function(ii) {
          reps <- arr[ii, pn, ]
          reps <- reps[!is.na(reps)]
          nr <- length(reps)
          if (nr == 0) return(NA_real_)
          mean(reps[sample.int(nr, min(R_n, nr))])
        }, numeric(1))
        naive_means <- naive_means[!is.na(naive_means)]
        N_n_eff <- length(naive_means)
        theta_n <- mean(naive_means)
        se_n    <- if (N_n_eff > 1) sd(naive_means) / sqrt(N_n_eff) else NA_real_

        # --- Standard: min(B, n_pool) items, 1 prompt, 1 rep ---
        N_s  <- min(B, n_pool)
        ps   <- sample.int(V, 1)
        items_s <- sample.int(n_pool, N_s)
        std_vals <- vapply(items_s, function(ii) {
          reps <- arr[ii, ps, ]
          reps <- reps[!is.na(reps)]
          nr <- length(reps)
          if (nr == 0) return(NA_real_)
          reps[sample.int(nr, 1)]
        }, numeric(1))
        std_vals <- std_vals[!is.na(std_vals)]
        N_s_eff <- length(std_vals)
        theta_s <- mean(std_vals)
        se_s    <- if (N_s_eff > 1) sd(std_vals) / sqrt(N_s_eff) else NA_real_

        # --- TEE: D-study-optimized K/R/N ---
        alloc <- tee_allocs[[as.character(B)]]
        K_t <- alloc$K; R_t <- alloc$R; N_t <- alloc$N

        # Sample K prompts and N items
        prompt_idx <- sample.int(V, K_t)
        items_t    <- sample.int(n_pool, N_t)

        tee_vals <- vapply(items_t, function(ii) {
          vals <- numeric(0)
          for (kk in prompt_idx) {
            reps <- arr[ii, kk, ]
            reps <- reps[!is.na(reps)]
            if (length(reps) > 0) {
              vals <- c(vals, reps[sample.int(length(reps), min(R_t, length(reps)))])
            }
          }
          if (length(vals) == 0) NA_real_ else mean(vals)
        }, numeric(1))
        tee_vals <- tee_vals[!is.na(tee_vals)]
        N_t_eff  <- length(tee_vals)
        theta_t  <- mean(tee_vals)
        se_t     <- tee_se(vcs, N_t_eff, K_t, R_t)

        # Error + coverage
        err_n <- theta_n - oracle_val
        err_s <- theta_s - oracle_val
        err_t <- theta_t - oracle_val
        cov_n <- abs(err_n) <= 1.96 * se_n
        cov_s <- abs(err_s) <= 1.96 * se_s
        cov_t <- abs(err_t) <= 1.96 * se_t

        out[[bi]] <- tibble(
          sim_id      = sim_id,
          sut_model   = sut,
          temperature = temp,
          budget      = B,
          researcher  = c("Naive", "Standard", "TEE"),
          theta_hat   = c(theta_n, theta_s, theta_t),
          oracle_mean = oracle_val,
          se          = c(se_n, se_s, se_t),
          error       = c(err_n, err_s, err_t),
          covers      = c(cov_n, cov_s, cov_t),
          N           = c(N_n_eff, N_s_eff, N_t_eff),
          K           = c(1L, 1L, K_t),
          R           = c(R_n, 1L, R_t)
        )
      }
      bind_rows(out)
    }

    t0 <- Sys.time()
    cell_res <- parallel::mclapply(seq_len(N_SIM), run_one, mc.cores = N_CORES)
    cell_res <- Filter(is.data.frame, cell_res)
    all_results <- c(all_results, cell_res)
    elapsed <- round(difftime(Sys.time(), t0, units = "secs"), 1)
    cat(sprintf("%d sims in %s sec\n", N_SIM, elapsed))
  }

  results_df <- bind_rows(all_results)
  write_csv(results_df, csv_path)
  cat(sprintf("\nSaved: %s (%d rows)\n", csv_path, nrow(results_df)))
}

# --- Summary ---
summary_df <- results_df %>%
  group_by(researcher, budget) %>%
  summarize(
    rmse       = sqrt(mean(error^2)),
    mean_bias  = mean(error),
    coverage   = mean(covers),
    mean_se    = mean(se),
    mean_N     = mean(N),
    mean_K     = mean(K),
    mean_R     = mean(R),
    n_sims     = n(),
    .groups    = "drop"
  )

cat("\n=== Summary ===\n")
print(summary_df, n = 30)
write_csv(summary_df, "data/processed/mmlu_budget_allocation_summary.csv")

# --- Figure ---
rcol <- c(Naive = "#d6604d", Standard = "#878787", TEE = "#2166ac")
rshp <- c(Naive = 16, Standard = 15, TEE = 17)

p_rmse <- summary_df %>%
  ggplot(aes(budget, rmse, color = researcher, shape = researcher)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  scale_color_manual(values = rcol) +
  scale_shape_manual(values = rshp) +
  labs(subtitle = "(a) Estimation accuracy",
       x = "Budget (API calls per SUT × temperature cell)",
       y = "RMSE vs. oracle accuracy",
       color = NULL, shape = NULL) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

p_cov <- summary_df %>%
  ggplot(aes(budget, coverage * 100, color = researcher, shape = researcher)) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  scale_color_manual(values = rcol) +
  scale_shape_manual(values = rshp) +
  labs(subtitle = "(b) CI coverage of oracle",
       x = "Budget (API calls per SUT × temperature cell)",
       y = "95% CI coverage (%)",
       color = NULL, shape = NULL) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

fig <- (p_rmse | p_cov) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom",
        plot.margin = margin(2, 2, 2, 2, "pt"))

ggsave("figures/fig_budget_allocation.pdf", fig, width = 10, height = 5)
ggsave("figures/fig_budget_allocation.png", fig, width = 10, height = 5, dpi = 300)
cat("Saved: figures/fig_budget_allocation.pdf/png\n")

cat("\n=== 12_budget_allocation_mmlu.R complete ===\n")
