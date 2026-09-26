# 04g2_sim_underestimation_balanced.R
# Variance underestimation simulation, ε / ρ split DGP.
#
# DGP refactor (Phase 1): The previous version lumped two distinct sources of
# variance into a single residual term σ²_ε:
#   1. Cell-level idiosyncrasy (constant within a cell, varies across cells)
#   2. Within-cell replicate noise (varies r-to-r within a cell)
#
# This script generates from a DGP that splits them explicitly:
#   y_{ivhm}^{(r)} = μ + α_i + φ_v + τ_h + λ_m
#                   + (αφ)_{iv} + (ατ)_{ih} + (φτ)_{vh}
#                   + (αλ)_{im} + (φλ)_{vm}
#                   + ε_{ivhm}                    (cell-level, no r index)
#                   + ρ_{ivhm}^{(r)}              (replicate-level)
#
# σ²_ε is invariant to averaging within a cell (D-study formula uses /N'V'H'M',
# NOT /R). Only σ²_ρ shrinks with replicates.
#
# Notation note: ρ is now the replicate-noise SD (free up the symbol from old
# "prompt sensitivity"). The prompt random effect is φ_v in this version.
#
# Scenarios (progressively richer lmer fits applied to the same factorial data):
#   A: V=1, H=1, M=1, R=1                naive s/sqrt(N)
#   B: V=V_full, H=1, M=1, R=R_full      lmer (1|item)+(1|variant)
#   C: V=V_full, H=1, M=M_full, R=R_full lmer + (1|judge_id)
#   D: V=V_full, H=H_full, M=M_full      lmer + (1|temp_id)
#   E: same data as D                     lmer + (1|variant:judge) + (1|variant:temp)
#                                            (= old "saturated" — lumps σ²_ε
#                                             into Residual, divides by /R)
#   F: same data as D                     lmer + (1|item:variant:judge:temp)
#                                            (= cell-level RE; σ²_ε / σ²_ρ split)
#
# Scenario E lumps cell-level (ε) and replicate (ρ) noise into Residual and
# divides by /(NVMHR), under-counting the ε contribution by a factor (R-1)/R.
# Scenario F adds the cell-level RE and recovers the correct decomposition.
#
# Output paths use the _v2 suffix.
#
# Usage:
#   Rscript analysis/04g2_sim_underestimation_balanced.R            # full run
#   Rscript analysis/04g2_sim_underestimation_balanced.R --nsim 5   # smoke test

library(tidyverse)
library(lme4)
library(patchwork)

# --- Parse command-line args ---
args <- commandArgs(trailingOnly = TRUE)
N_SIM <- 500
N_CORES <- max(1, min(parallel::detectCores() - 2, 8))
if ("--nsim" %in% args) {
  idx <- which(args == "--nsim")
  N_SIM <- as.integer(args[idx + 1])
  cat(sprintf("Using N_SIM = %d (from --nsim flag)\n", N_SIM))
}
if ("--cores" %in% args) {
  idx <- which(args == "--cores")
  N_CORES <- as.integer(args[idx + 1])
}
cat(sprintf("Parallel workers: %d\n", N_CORES))

# Population variance (denominator n, not n-1)
pop_var <- function(x) mean((x - mean(x))^2)

# --- DGP parameters (ε / ρ split) ---
# σ²_ε is moderate-to-large relative to per-item terms so scenarios E and F
# diverge in coverage (E lumps ε into Residual and over-shrinks by /R; F
# captures the cell-level RE and divides only by /N'V'H'M').
#
# σ²_φ, σ²_φτ, σ²_φλ kept moderate so scenarios B, C, D each capture a
# meaningful slice of pipeline variance (the "Tier 2" components researchers
# typically miss). Item-side terms σ²_αφ, σ²_ατ, σ²_αλ kept small to keep the
# Residual in scenarios B-E mostly attributable to the ε / ρ mixture.
TRUE_PARAMS <- list(
  mu = 0.5,
  sigma2_alpha = 0.10,         # item RE
  sigma2_phi   = 0.04,         # prompt RE (renamed from old ρ)
  sigma2_aphi  = 0.04,         # item × prompt
  sigma2_atau  = 0.03,         # item × temp
  sigma2_phitau = 0.02,        # prompt × temp
  sigma2_alam  = 0.04,         # item × judge
  sigma2_philam = 0.02,        # prompt × judge
  sigma2_taulam = 0.02,        # temp × judge (design-side 2-way)
  # σ²_ε comparable in scale to the other variance components so panel (b)
  # remains an illustration of which components each scenario captures
  # (rather than being dominated by one bar). The E vs F coverage gap in
  # panel (a) is correspondingly small (~0.1 pp) — F's value is its
  # diagnostic localization of σ²_ε, not raw coverage at this design.
  sigma2_eps_cell = 0.04,      # cell-level idiosyncrasy (constant in r)
  sigma2_rho_rep  = 0.10,      # within-cell replicate noise
  tau    = c(-0.20, 0, 0.20),  # temp design (pop_var ≈ 0.027)
  lambda = c(-0.20, 0, 0.20),  # judge design (pop_var ≈ 0.027)
  n_categories = 5
)

cat("=== 04g2_sim_underestimation_balanced.R: ε / ρ split DGP ===\n")
cat(sprintf("N_SIM = %d\n", N_SIM))
cat("True variance components:\n")
cat(sprintf("  σ²_α (item)         = %.4f\n", TRUE_PARAMS$sigma2_alpha))
cat(sprintf("  σ²_φ (prompt)       = %.4f\n", TRUE_PARAMS$sigma2_phi))
cat(sprintf("  σ²_αφ (item:prompt) = %.4f\n", TRUE_PARAMS$sigma2_aphi))
cat(sprintf("  σ²_ατ (item:temp)   = %.4f\n", TRUE_PARAMS$sigma2_atau))
cat(sprintf("  σ²_φτ (prompt:temp) = %.4f\n", TRUE_PARAMS$sigma2_phitau))
cat(sprintf("  σ²_αλ (item:judge)  = %.4f\n", TRUE_PARAMS$sigma2_alam))
cat(sprintf("  σ²_φλ (prompt:judge)= %.4f\n", TRUE_PARAMS$sigma2_philam))
cat(sprintf("  σ²_τλ (temp:judge)  = %.4f\n", TRUE_PARAMS$sigma2_taulam))
cat(sprintf("  σ²_ε (cell)         = %.4f\n", TRUE_PARAMS$sigma2_eps_cell))
cat(sprintf("  σ²_ρ (replicate)    = %.4f\n", TRUE_PARAMS$sigma2_rho_rep))
cat(sprintf("  pop_var(τ) (temp)   = %.4f\n", pop_var(TRUE_PARAMS$tau)))
cat(sprintf("  pop_var(λ) (judge)  = %.4f\n", pop_var(TRUE_PARAMS$lambda)))

# =========================================================================
# DGP helpers (ε / ρ split)
# =========================================================================

# Draw population random effects + cell-level ε array.
# pop$eps is a 4D array indexed (item, variant, temp, judge) — drawn once,
# constant across replications within a cell.
draw_population <- function(n_items, n_prompts, n_temps, n_judges, params) {
  alpha <- rnorm(n_items, 0, sqrt(params$sigma2_alpha))
  phi   <- rnorm(n_prompts, 0, sqrt(params$sigma2_phi))
  list(
    alpha = alpha, phi = phi,
    aphi  = matrix(rnorm(n_items * n_prompts, 0, sqrt(params$sigma2_aphi)),
                    nrow = n_items, ncol = n_prompts),
    atau  = matrix(rnorm(n_items * n_temps, 0, sqrt(params$sigma2_atau)),
                    nrow = n_items, ncol = n_temps),
    phitau = matrix(rnorm(n_prompts * n_temps, 0, sqrt(params$sigma2_phitau)),
                    nrow = n_prompts, ncol = n_temps),
    alam  = matrix(rnorm(n_items * n_judges, 0, sqrt(params$sigma2_alam)),
                    nrow = n_items, ncol = n_judges),
    philam = matrix(rnorm(n_prompts * n_judges, 0, sqrt(params$sigma2_philam)),
                    nrow = n_prompts, ncol = n_judges),
    taulam = matrix(rnorm(n_temps * n_judges, 0, sqrt(params$sigma2_taulam)),
                    nrow = n_temps, ncol = n_judges),
    eps_cell = array(rnorm(n_items * n_prompts * n_temps * n_judges,
                            0, sqrt(params$sigma2_eps_cell)),
                      dim = c(n_items, n_prompts, n_temps, n_judges))
  )
}

# Generate observations for a slice of the factorial.
# Vectorized: rep() patterns reproduce the (i, v, h, m, r) lexicographic order
# so a single rnorm(total) call consumes RNG state in the same sequence as a
# 5-deep nested loop would.
generate_slice <- function(pop, params, v_set, h_set, m_set, n_reps) {
  n_items <- length(pop$alpha)
  V <- length(v_set); H <- length(h_set); M <- length(m_set); R <- n_reps
  total <- n_items * V * H * M * R

  item_id    <- rep(seq_len(n_items),                         each = V * H * M * R)
  variant_id <- rep(rep(v_set,         each = H * M * R),     times = n_items)
  temp_id    <- rep(rep(h_set,         each = M * R),         times = n_items * V)
  judge_id   <- rep(rep(m_set,         each = R),             times = n_items * V * H)
  rep_id     <- rep(seq_len(R),                               times = n_items * V * H * M)

  # Cell-level ε is constant across r — same indexed lookup for each rep.
  eps_idx <- cbind(item_id, variant_id, temp_id, judge_id)

  outcome <- params$mu +
    pop$alpha[item_id] + pop$phi[variant_id] +
    params$tau[temp_id] + params$lambda[judge_id] +
    pop$aphi[cbind(item_id, variant_id)] +
    pop$atau[cbind(item_id, temp_id)] +
    pop$phitau[cbind(variant_id, temp_id)] +
    pop$alam[cbind(item_id, judge_id)] +
    pop$philam[cbind(variant_id, judge_id)] +
    pop$taulam[cbind(temp_id, judge_id)] +
    pop$eps_cell[eps_idx] +
    rnorm(total, 0, sqrt(params$sigma2_rho_rep))

  tibble(item_id = as.factor(item_id), variant_id = as.factor(variant_id),
         temp_id = as.factor(temp_id), judge_id = as.factor(judge_id),
         rep_id = rep_id, outcome = outcome)
}


# =========================================================================
# Experiment 1: CI Coverage across scenarios A–F
# =========================================================================
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(output_dir, "sim_underestimation_v2.csv")
csv_summary_path <- file.path(output_dir, "sim_underestimation_summary_v2.csv")

if (file.exists(csv_path) && file.exists(csv_summary_path)) {
  cat("\nLoading existing simulation results from CSV...\n")
  exp1_df <- read_csv(csv_path, show_col_types = FALSE)
  exp1_summary <- read_csv(csv_summary_path, show_col_types = FALSE)
  N_levels <- sort(unique(exp1_df$n_items))
} else {
  cat("\n\n========== Experiment 1: CI Coverage ==========\n")

  N_levels_all <- c(25, 50, 100, 200, 400)
  V_full <- 4
  H_full <- 3
  M_full <- 3
  R_full <- 8

  # Oracle SE² for the GRAND-MEAN estimand under full DGP.
  # Used as a fallback when a scenario's lmer fit fails (rare).
  oracle_var_full <- function(N, V, H, M, R, p = TRUE_PARAMS) {
    p$sigma2_alpha / N +
      p$sigma2_phi / V +
      pop_var(p$tau) / H +
      pop_var(p$lambda) / M +
      p$sigma2_aphi / (N * V) +
      p$sigma2_atau / (N * H) +
      p$sigma2_alam / (N * M) +
      p$sigma2_phitau / (V * H) +
      p$sigma2_philam / (V * M) +
      p$sigma2_taulam / (H * M) +
      p$sigma2_eps_cell / (N * V * H * M) +
      p$sigma2_rho_rep / (N * V * H * M * R)
  }

  # Oracle SE² per scenario (used as fallback). Each formula reflects what the
  # scenario's model captures, NOT the true Var(grand mean). Mismatch drives
  # coverage collapse in A–E.
  oracle_var_scenario <- function(scen, N, V, H, M, R, p = TRUE_PARAMS) {
    switch(scen,
      # A: sd(y)/sqrt(N) at 1 obs/item at fixed (v_c, h_c, m_c).
      #    Var(y_i) = σ²_α + σ²_αφ + σ²_αλ + σ²_ατ + σ²_ε + σ²_ρ
      "A" = (p$sigma2_alpha + p$sigma2_aphi + p$sigma2_alam +
             p$sigma2_atau + p$sigma2_eps_cell + p$sigma2_rho_rep) / N,
      # B: lmer (1|item)+(1|variant) on N*V*R data at fixed h_c, m_c.
      "B" = (p$sigma2_alpha + p$sigma2_alam + p$sigma2_atau) / N +
            (p$sigma2_phi + p$sigma2_philam + p$sigma2_phitau) / V +
            (p$sigma2_aphi + p$sigma2_eps_cell) / (N * V) +
            p$sigma2_rho_rep / (N * V * R),
      # C: + (1|judge), V=full, M=full at fixed h_c. Data N*V*M*R.
      "C" = (p$sigma2_alpha + p$sigma2_atau) / N +
            (p$sigma2_phi + p$sigma2_phitau) / V +
            pop_var(p$lambda) / M +
            (p$sigma2_aphi + p$sigma2_alam + p$sigma2_philam +
             p$sigma2_eps_cell) / (N * V * M) +
            p$sigma2_rho_rep / (N * V * M * R),
      # D: + (1|temp), V=full, M=full, H=full. Data N*V*M*H*R.
      "D" = p$sigma2_alpha / N + p$sigma2_phi / V +
            pop_var(p$lambda) / M + pop_var(p$tau) / H +
            (p$sigma2_aphi + p$sigma2_alam + p$sigma2_atau +
             p$sigma2_phitau + p$sigma2_philam + p$sigma2_taulam +
             p$sigma2_eps_cell) / (N * V * M * H) +
            p$sigma2_rho_rep / (N * V * M * H * R),
      # E: D + 2 prompt-side two-way interactions (variant:judge, variant:temp).
      #    Lumps item-side interactions AND σ²_ε into Residual, divides by
      #    /(NVHMR). Under-counts each lumped term by a factor close to its
      #    natural divisor over R.
      "E" = p$sigma2_alpha / N + p$sigma2_phi / V +
            pop_var(p$lambda) / M + pop_var(p$tau) / H +
            p$sigma2_phitau / (V * H) + p$sigma2_philam / (V * M) +
            p$sigma2_taulam / (H * M) +
            (p$sigma2_aphi + p$sigma2_atau + p$sigma2_alam +
             p$sigma2_eps_cell + p$sigma2_rho_rep) / (N * V * H * M * R),
      # F: E + item-side interactions + cell-level RE. Matches True Var.
      "F" = oracle_var_full(N, V, H, M, R, p)
    )
  }

  # Fit lmer with timeout; return intercept SE or NA on failure.
  # Singular fits (zero variance estimate at boundary) are still valid and
  # must not fall back — only hard errors and timeouts should.
  # Timeout is generous (300 s) because scenario F's lmer with the cell-level
  # RE has 10 random-effect groups and N*V*H*M cell levels; under parallel
  # contention each fit can take 1–3 minutes at large N.
  fit_lmer_se <- function(formula, data, timeout_s = 300) {
    mod <- tryCatch(
      suppressMessages(suppressWarnings(
        R.utils::withTimeout(
          lmer(formula, data = data, REML = TRUE,
               control = lmerControl(optimizer = "bobyqa",
                                      calc.derivs = FALSE,
                                      check.conv.singular = .makeCC("ignore", tol = 1e-4),
                                      optCtrl = list(maxfun = 50000))),
          timeout = timeout_s, onTimeout = "silent"
        )
      )),
      error = function(e) NULL
    )
    if (is.null(mod)) return(list(se = NA_real_, mod = NULL))
    list(se = sqrt(as.numeric(vcov(mod)[1, 1])), mod = mod)
  }

  # Per-sim worker function (called in parallel)
  run_one_sim <- function(sim, n_items, V_full, H_full, M_full, R_full,
                          TRUE_PARAMS, oracle_se) {
    set.seed(42 + sim + n_items * 10000)

    pop <- draw_population(n_items, V_full, H_full, M_full, TRUE_PARAMS)

    # Estimand is the GRAND MEAN over the full DGP. Each scenario samples a
    # subset of the factorial and fits a progressively richer lmer.
    v_chosen <- sample(V_full, 1)
    h_chosen <- sample(H_full, 1)
    m_chosen <- sample(M_full, 1)
    theta_true <- TRUE_PARAMS$mu + mean(TRUE_PARAMS$tau) + mean(TRUE_PARAMS$lambda)

    # --- Scenario A: 1 obs per item at fixed (v_c, h_c, m_c), naive SE ---
    df_A <- generate_slice(pop, TRUE_PARAMS, v_chosen, h_chosen, m_chosen, 1)
    theta_A <- mean(df_A$outcome)
    se_A <- sd(df_A$outcome) / sqrt(n_items)
    s2_eps_A <- NA_real_
    s2_rho_A <- NA_real_

    # --- Scenario B: V=full at fixed (h_c, m_c), lmer (1|item)+(1|variant) ---
    df_B <- generate_slice(pop, TRUE_PARAMS, seq_len(V_full), h_chosen, m_chosen, R_full)
    theta_B <- mean(df_B$outcome)
    fit_B <- fit_lmer_se(
      outcome ~ (1 | item_id) + (1 | variant_id),
      df_B
    )
    se_B <- if (is.na(fit_B$se)) oracle_se["B"] else fit_B$se

    # --- Scenario C: V=full, M=full at fixed h_c, lmer + (1|judge) ---
    df_C <- generate_slice(pop, TRUE_PARAMS, seq_len(V_full), h_chosen, seq_len(M_full), R_full)
    theta_C <- mean(df_C$outcome)
    fit_C <- fit_lmer_se(
      outcome ~ (1 | item_id) + (1 | variant_id) + (1 | judge_id),
      df_C
    )
    se_C <- if (is.na(fit_C$se)) oracle_se["C"] else fit_C$se

    # --- Scenario D: full V, M, H, lmer + (1|temp) ---
    df_D <- generate_slice(pop, TRUE_PARAMS, seq_len(V_full), seq_len(H_full), seq_len(M_full), R_full)
    theta_D <- mean(df_D$outcome)
    fit_D <- fit_lmer_se(
      outcome ~ (1 | item_id) + (1 | variant_id) + (1 | judge_id) + (1 | temp_id),
      df_D
    )
    se_D <- if (is.na(fit_D$se)) oracle_se["D"] else fit_D$se

    # --- Scenario E: D + 2 prompt-side interactions ---
    # The previous "saturated" model. Only variant:judge and variant:temp
    # fitted explicitly; item-side interactions and cell-level ε get lumped
    # into Residual. The intercept SE then divides Residual by /(NVMHR),
    # under-counting the contributions of item:variant, item:judge,
    # item:temp, and σ²_ε.
    theta_E <- mean(df_D$outcome)
    fit_E <- fit_lmer_se(
      outcome ~ (1 | item_id) + (1 | variant_id) + (1 | judge_id) + (1 | temp_id) +
        (1 | variant_id:judge_id) + (1 | variant_id:temp_id) +
        (1 | temp_id:judge_id),
      df_D
    )
    se_E <- if (is.na(fit_E$se)) oracle_se["E"] else fit_E$se

    # --- Scenario F: cell-mean lmer (decomposes σ²_ε from σ²_ρ analytically) ---
    # Rather than fit (1|item:variant:judge:temp) on full data (slow, 10 RE
    # groups × NVHM levels), aggregate to cell means and fit lmer there.
    # The Residual on cell means is σ²_ε + σ²_ρ/R; the cell-mean intercept
    # variance includes Residual/(NVHM) = σ²_ε/(NVHM) + σ²_ρ/(NVHMR), which
    # matches the correct F D-study formula. σ²_ρ is recovered separately
    # from the within-cell sample variance (not needed for SE; only for the
    # decomposition diagnostic in the output).
    df_F <- df_D %>%
      group_by(item_id, variant_id, temp_id, judge_id) %>%
      summarize(outcome = mean(outcome), .groups = "drop")
    theta_F <- mean(df_F$outcome)
    fit_F <- fit_lmer_se(
      outcome ~ (1 | item_id) + (1 | variant_id) + (1 | judge_id) + (1 | temp_id) +
        (1 | item_id:variant_id) + (1 | item_id:judge_id) + (1 | item_id:temp_id) +
        (1 | variant_id:judge_id) + (1 | variant_id:temp_id) +
        (1 | temp_id:judge_id),
      df_F
    )
    se_F <- if (is.na(fit_F$se)) oracle_se["F"] else fit_F$se

    # Extract σ²_ε and σ²_ρ recovery (validation diagnostics).
    # Cell-mean fit residual ≈ σ²_ε + σ²_ρ / R. Compute σ²_ρ separately as
    # the average within-cell sample variance of df_D, then back out σ²_ε.
    s2_eps_F <- NA_real_
    s2_rho_F <- NA_real_
    if (!is.null(fit_F$mod)) {
      vc_F <- as.data.frame(VarCorr(fit_F$mod))
      res_idx <- which(vc_F$grp == "Residual")
      s2_resid_F_cellmean <- if (length(res_idx) == 1) vc_F$vcov[res_idx] else NA_real_
      # σ²_ρ from within-cell variance
      s2_rho_F <- df_D %>%
        group_by(item_id, variant_id, temp_id, judge_id) %>%
        summarize(v = var(outcome), .groups = "drop") %>%
        pull(v) %>% mean(na.rm = TRUE)
      # σ²_ε from cell-mean residual minus σ²_ρ/R
      s2_eps_F <- max(0, s2_resid_F_cellmean - s2_rho_F / R_full)
    }
    # Residual estimate from scenario E (lumped ε + ρ on full data)
    s2_resid_E <- NA_real_
    if (!is.null(fit_E$mod)) {
      vc_E <- as.data.frame(VarCorr(fit_E$mod))
      res_idx  <- which(vc_E$grp == "Residual")
      if (length(res_idx) == 1) s2_resid_E <- vc_E$vcov[res_idx]
    }

    tibble(
      scenario = c("A", "B", "C", "D", "E", "F"),
      label = c("1 obs/item, naive SE",
                "+ (1|variant): captures phi",
                "+ (1|judge): captures lambda",
                "+ (1|temp): captures tau",
                "+ interactions: lumps eps into residual",
                "+ (1|item:variant:judge:temp): splits eps/rho"),
      theta_hat = unname(c(theta_A, theta_B, theta_C, theta_D, theta_E, theta_F)),
      se = unname(c(se_A, se_B, se_C, se_D, se_E, se_F)),
      ci_lo = theta_hat - 1.96 * se,
      ci_hi = theta_hat + 1.96 * se,
      covers = as.logical(ci_lo <= theta_true & theta_true <= ci_hi),
      se_ratio = se / oracle_se["F"],
      s2_eps_hat = c(NA_real_, NA_real_, NA_real_, NA_real_, NA_real_, s2_eps_F),
      s2_rho_hat = c(NA_real_, NA_real_, NA_real_, NA_real_, NA_real_, s2_rho_F),
      s2_resid_E = c(NA_real_, NA_real_, NA_real_, NA_real_, s2_resid_E, NA_real_),
      n_items = n_items,
      sim_id = sim,
      theta_true = theta_true
    )
  }

  # Run simulation in parallel, with per-N checkpointing.
  exp1_results <- list()

  for (n_items in N_levels_all) {
    cell_csv <- file.path(output_dir,
                          sprintf("sim_underestimation_v2_cell_N%d.csv", n_items))

    if (file.exists(cell_csv)) {
      cat(sprintf("\n  N = %d items: loading cached cell CSV...\n", n_items))
      exp1_results[[length(exp1_results) + 1]] <- read_csv(cell_csv, show_col_types = FALSE)
      next
    }

    cat(sprintf("\n  N = %d items (%d sims, %d workers)...\n", n_items, N_SIM, N_CORES))
    t0 <- Sys.time()

    oracle_se <- c(
      A = sqrt(oracle_var_scenario("A", n_items, V_full, H_full, M_full, R_full)),
      B = sqrt(oracle_var_scenario("B", n_items, V_full, H_full, M_full, R_full)),
      C = sqrt(oracle_var_scenario("C", n_items, V_full, H_full, M_full, R_full)),
      D = sqrt(oracle_var_scenario("D", n_items, V_full, H_full, M_full, R_full)),
      E = sqrt(oracle_var_scenario("E", n_items, V_full, H_full, M_full, R_full)),
      "F" = sqrt(oracle_var_scenario("F", n_items, V_full, H_full, M_full, R_full))
    )

    # mc.preschedule = FALSE: fork-per-task so each sim's memory returns to OS
    # when the task ends, instead of accumulating across N_SIM/N_CORES tasks.
    results_n <- parallel::mclapply(seq_len(N_SIM), function(sim) {
      run_one_sim(sim, n_items, V_full, H_full, M_full, R_full,
                  TRUE_PARAMS, oracle_se)
    }, mc.cores = N_CORES, mc.preschedule = FALSE)
    results_n <- Filter(is.data.frame, results_n)
    cell_df <- bind_rows(results_n)

    write_csv(cell_df, cell_csv)
    exp1_results[[length(exp1_results) + 1]] <- cell_df

    elapsed <- round(difftime(Sys.time(), t0, units = "secs"), 1)
    cat(sprintf("    done in %s sec (saved %s)\n", elapsed, basename(cell_csv)))
  }

  exp1_df <- bind_rows(exp1_results)
  N_levels <- N_levels_all
  cat(sprintf("  exp1_df: %d rows, %d cols\n", nrow(exp1_df), ncol(exp1_df)))
  cat("  Columns:", paste(names(exp1_df), collapse=", "), "\n")

  # Summary: coverage and SE ratio by scenario and N
  exp1_summary <- exp1_df %>%
    filter(!is.na(covers)) %>%
    group_by(scenario, label, n_items) %>%
    summarize(
      coverage = mean(covers, na.rm = TRUE),
      mean_se_ratio = mean(se_ratio, na.rm = TRUE),
      median_se_ratio = median(se_ratio, na.rm = TRUE),
      mean_se = mean(se, na.rm = TRUE),
      mean_s2_eps_hat = mean(s2_eps_hat, na.rm = TRUE),
      mean_s2_rho_hat = mean(s2_rho_hat, na.rm = TRUE),
      mean_s2_resid_E = mean(s2_resid_E, na.rm = TRUE),
      n_sims = n(),
      .groups = "drop"
    )

  cat("\n\nExperiment 1 — Coverage summary:\n")
  print(exp1_summary %>% select(scenario, n_items, coverage, mean_se_ratio, n_sims) %>%
          arrange(scenario, n_items))

  # Save
  write_csv(exp1_df, csv_path)
  write_csv(exp1_summary, csv_summary_path)
  cat("\nSaved: sim_underestimation_v2.csv, sim_underestimation_summary_v2.csv\n")
}


# =========================================================================
# Experiment 2: "20 Researchers" illustration
# =========================================================================
csv_20r_path <- file.path(output_dir, "sim_underestimation_20researchers_v2.csv")

if (file.exists(csv_20r_path)) {
  cat("\nLoading existing 20-researchers data from CSV...\n")
  researchers_df <- read_csv(csv_20r_path, show_col_types = FALSE)
  theta_overall <- TRUE_PARAMS$mu
  se_tle_v1 <- researchers_df$se_tle_v1[1]
} else {
  cat("\n\n========== Experiment 2: 20 Researchers ==========\n")

  set.seed(12345)
  N_researchers <- 20
  N_items_20 <- 100
  V_pop <- 5
  H_pop <- 3
  M_pop <- 3
  R_pop <- 8

  pop_20 <- draw_population(N_items_20, V_pop, H_pop, M_pop, TRUE_PARAMS)
  theta_overall <- TRUE_PARAMS$mu

  # Full factorial for TEE model
  df_pop <- generate_slice(pop_20, TRUE_PARAMS,
                           seq_len(V_pop), seq_len(H_pop), seq_len(M_pop), R_pop)

  researcher_results <- list()
  for (r_id in seq_len(N_researchers)) {
    v_r <- sample(V_pop, 1)
    h_r <- sample(H_pop, 1)
    m_r <- sample(M_pop, 1)
    theta_r_true <- TRUE_PARAMS$mu + TRUE_PARAMS$tau[h_r] + TRUE_PARAMS$lambda[m_r]

    df_r <- generate_slice(pop_20, TRUE_PARAMS, v_r, h_r, m_r, R_pop)
    item_means <- df_r %>%
      group_by(item_id) %>%
      summarize(y_bar = mean(outcome), .groups = "drop")

    theta_hat <- mean(item_means$y_bar)
    se_naive <- sd(item_means$y_bar) / sqrt(N_items_20)

    researcher_results[[r_id]] <- tibble(
      researcher_id = r_id,
      v_chosen = v_r, h_chosen = h_r, m_chosen = m_r,
      theta_hat = theta_hat,
      se_naive = se_naive,
      ci_lo = theta_hat - 1.96 * se_naive,
      ci_hi = theta_hat + 1.96 * se_naive,
      theta_true_specific = theta_r_true,
      covers_specific = (theta_hat - 1.96 * se_naive <= theta_r_true) &
        (theta_r_true <= theta_hat + 1.96 * se_naive),
      covers_overall = (theta_hat - 1.96 * se_naive <= theta_overall) &
        (theta_overall <= theta_hat + 1.96 * se_naive)
    )
  }

  researchers_df <- bind_rows(researcher_results)

  # Full TEE model (fit on full pop data with cell-level RE)
  mod_full <- tryCatch(
    lmer(outcome ~ temp_id + judge_id +
           (1 | item_id) + (1 | variant_id) +
           (1 | item_id:variant_id) + (1 | item_id:temp_id) +
           (1 | variant_id:temp_id) + (1 | item_id:judge_id) +
           (1 | variant_id:judge_id) +
           (1 | item_id:variant_id:judge_id:temp_id),
         data = df_pop, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa",
                                calc.derivs = FALSE,
                                optCtrl = list(maxfun = 50000))),
    error = function(e) NULL
  )

  if (!is.null(mod_full)) {
    vc_full <- as.data.frame(VarCorr(mod_full))
    s2_full <- setNames(vc_full$vcov, vc_full$grp)
    var_tle_v1 <- (s2_full["item_id"] %||% 0) / N_items_20 +
      (s2_full["variant_id"] %||% 0) / 1 +
      (s2_full["item_id:variant_id"] %||% 0) / (N_items_20 * 1) +
      (s2_full["item_id:judge_id"] %||% 0) / (N_items_20 * 1) +
      (s2_full["variant_id:judge_id"] %||% 0) / (1 * 1) +
      (s2_full["item_id:variant_id:judge_id:temp_id"] %||% 0) / (N_items_20 * 1) +
      (s2_full["Residual"] %||% 0) / (N_items_20 * 1 * 1 * R_pop)
    se_tle_v1 <- sqrt(max(var_tle_v1, 1e-12))
  } else {
    se_tle_v1 <- NA
  }

  researchers_df$se_tle_v1 <- se_tle_v1

  cat(sprintf("\n20 Researchers: %d/%d naive CIs cover true theta_overall\n",
              sum(researchers_df$covers_overall), N_researchers))
  cat(sprintf("TEE SE (V=1 estimand): %.4f\n", se_tle_v1))
  cat(sprintf("Mean naive SE: %.4f\n", mean(researchers_df$se_naive)))
  cat(sprintf("Ratio (naive/TEE): %.2f\n", mean(researchers_df$se_naive) / se_tle_v1))

  write_csv(researchers_df, csv_20r_path)
  cat("Saved: sim_underestimation_20researchers_v2.csv\n")
}

theta_overall <- TRUE_PARAMS$mu
N_researchers <- nrow(researchers_df)


# =========================================================================
# Figures (always regenerated from CSV)
# =========================================================================
cat("\n--- Generating figures ---\n")
fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# --- Panel A: "20 Researchers" CI plot ---
researchers_plot <- researchers_df %>%
  arrange(theta_hat) %>%
  mutate(
    researcher_id = factor(researcher_id, levels = researcher_id),
    covers_label = ifelse(covers_overall, "Covers true mean", "Misses true mean")
  )

p_20 <- ggplot(researchers_plot, aes(y = researcher_id)) +
  geom_vline(xintercept = theta_overall, linetype = "solid", color = "firebrick",
             linewidth = 0.8) +
  geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi, color = covers_label),
                width = 0.3, linewidth = 0.6, orientation = "y") +
  geom_point(aes(x = theta_hat, color = covers_label), size = 1.5) +
  annotate("rect",
           xmin = theta_overall - 1.96 * se_tle_v1,
           xmax = theta_overall + 1.96 * se_tle_v1,
           ymin = 0.2, ymax = N_researchers + 0.8,
           fill = "steelblue", alpha = 0.12) +
  annotate("text",
           x = theta_overall - 1.96 * se_tle_v1 + 0.005,
           y = N_researchers * 0.85,
           label = "TEE 95% CI\n(accounts for\nprompt & judge\nvariance)",
           size = 2.8, hjust = 0, color = "steelblue") +
  scale_color_manual(values = c("Covers true mean" = "gray50",
                                "Misses true mean" = "firebrick")) +
  labs(
    subtitle = "(A) 20 researchers, same items, different pipeline choices",
    x = expression("Estimated mean " * hat(theta)),
    y = "Researcher",
    color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position = "bottom",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  )

# --- Panel B: Coverage vs N for scenarios A–F (log x-axis) ---
scenario_labels <- c(
  "A" = "A: naive",
  "B" = "B: +(1|variant)",
  "C" = "C: +(1|judge)",
  "D" = "D: +(1|temp)",
  "E" = "E: +interactions",
  "F" = "F: +(1|item:variant:judge:temp)"
)

# Component colors — each scenario line in (a) matches the color of the
# component it introduces in (b).
col_prompt <- "#E07B00"   # orange (phi — introduced by B)
col_judge  <- "#C0392B"   # red (lambda — introduced by C)
col_temp   <- "#7A7A7A"   # gray (tau — introduced by D)
col_inter  <- "#6A5ACD"   # slate blue (interactions — introduced by E)
col_cell   <- "#2E8B57"   # sea green (cell-level eps — introduced by F)

scenario_colors <- c(
  "A: naive"                       = "black",
  "B: +(1|variant)"                = col_prompt,
  "C: +(1|judge)"                  = col_judge,
  "D: +(1|temp)"                   = col_temp,
  "E: +interactions"               = col_inter,
  "F: +(1|item:variant:judge:temp)" = col_cell
)

p_cov <- exp1_summary %>%
  filter(!is.na(coverage)) %>%
  mutate(scenario_label = scenario_labels[scenario]) %>%
  ggplot(aes(x = n_items, y = coverage * 100, color = scenario_label,
             group = scenario_label)) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  scale_color_manual(values = scenario_colors) +
  scale_x_log10(breaks = sort(unique(exp1_summary$n_items)),
                labels = scales::comma) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(
    subtitle = "(B) CI coverage vs. number of items",
    x = "Number of items (N, log scale)",
    y = "95% CI coverage (%)",
    color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.key.size = unit(0.4, "cm"),
    legend.text = element_text(size = 8)
  ) +
  guides(color = guide_legend(ncol = 2))

# --- Main text figure: 2 panels ---
fig_main <- p_20 + p_cov +
  plot_layout(widths = c(1, 1.2)) +
  plot_annotation(
    title = "Naive Standard Errors Underestimate Pipeline Uncertainty",
    subtitle = paste0("Researchers using one prompt, one judge, and one temperature report CIs that are too narrow. ",
                      "Coverage worsens as N grows."),
    theme = theme(
      plot.title = element_text(size = 13, face = "bold"),
      plot.subtitle = element_text(size = 9.5)
    )
  )

ggsave(file.path(fig_dir, "fig_underestimation_v2.pdf"), fig_main,
       width = 11, height = 5.5)
ggsave(file.path(fig_dir, "fig_underestimation_v2.png"), fig_main,
       width = 11, height = 5.5, dpi = 300)
cat("Saved: fig_underestimation_v2.pdf/png\n")

# --- PNAS main text figure: coverage (a) + variance components captured (b) ---
# Panel (b) shows components captured per scenario; each step adds one more.

p <- TRUE_PARAMS
sigma2_tau_design <- pop_var(p$tau)
sigma2_lambda_design <- pop_var(p$lambda)

blend_hex <- function(a, b) {
  ra <- col2rgb(a); rb <- col2rgb(b)
  rgb((ra[1]+rb[1])/2, (ra[2]+rb[2])/2, (ra[3]+rb[3])/2, maxColorValue = 255)
}

# Each row = (scenario, component) with sigma2 nonzero iff that scenario's
# lmer captures the component. Bars stack to show progressive accumulation.
captured_per_scenario <- tribble(
  ~scenario, ~component,        ~family,  ~sigma2,                ~label,
  # A: naive sd/sqrt(N) — item-side only, no design-side components
  "A", "prompt",        "prompt", 0,                              "phi",
  "A", "temperature",   "temp",   0,                              "tau",
  "A", "judge",         "judge",  0,                              "lambda",
  "A", "item:prompt",   "i_x_p",  0,                              "alpha*phi",
  "A", "item:temp",     "i_x_t",  0,                              "alpha*tau",
  "A", "item:judge",    "i_x_j",  0,                              "alpha*lambda",
  "A", "prompt:temp",   "p_x_t",  0,                              "phi*tau",
  "A", "prompt:judge",  "p_x_j",  0,                              "phi*lambda",
  "A", "temp:judge",    "t_x_j",  0,                              "tau*lambda",
  "A", "cell",          "cell",   0,                              "epsilon",
  # B: lmer (1|item)+(1|variant) — adds phi
  "B", "prompt",        "prompt", p$sigma2_phi,                   "phi",
  "B", "temperature",   "temp",   0,                              "tau",
  "B", "judge",         "judge",  0,                              "lambda",
  "B", "item:prompt",   "i_x_p",  0,                              "alpha*phi",
  "B", "item:temp",     "i_x_t",  0,                              "alpha*tau",
  "B", "item:judge",    "i_x_j",  0,                              "alpha*lambda",
  "B", "prompt:temp",   "p_x_t",  0,                              "phi*tau",
  "B", "prompt:judge",  "p_x_j",  0,                              "phi*lambda",
  "B", "temp:judge",    "t_x_j",  0,                              "tau*lambda",
  "B", "cell",          "cell",   0,                              "epsilon",
  # C: + (1|judge) — adds lambda
  "C", "prompt",        "prompt", p$sigma2_phi,                   "phi",
  "C", "temperature",   "temp",   0,                              "tau",
  "C", "judge",         "judge",  sigma2_lambda_design,           "lambda",
  "C", "item:prompt",   "i_x_p",  0,                              "alpha*phi",
  "C", "item:temp",     "i_x_t",  0,                              "alpha*tau",
  "C", "item:judge",    "i_x_j",  0,                              "alpha*lambda",
  "C", "prompt:temp",   "p_x_t",  0,                              "phi*tau",
  "C", "prompt:judge",  "p_x_j",  0,                              "phi*lambda",
  "C", "temp:judge",    "t_x_j",  0,                              "tau*lambda",
  "C", "cell",          "cell",   0,                              "epsilon",
  # D: + (1|temp) — adds tau
  "D", "prompt",        "prompt", p$sigma2_phi,                   "phi",
  "D", "temperature",   "temp",   sigma2_tau_design,              "tau",
  "D", "judge",         "judge",  sigma2_lambda_design,           "lambda",
  "D", "item:prompt",   "i_x_p",  0,                              "alpha*phi",
  "D", "item:temp",     "i_x_t",  0,                              "alpha*tau",
  "D", "item:judge",    "i_x_j",  0,                              "alpha*lambda",
  "D", "prompt:temp",   "p_x_t",  0,                              "phi*tau",
  "D", "prompt:judge",  "p_x_j",  0,                              "phi*lambda",
  "D", "temp:judge",    "t_x_j",  0,                              "tau*lambda",
  "D", "cell",          "cell",   0,                              "epsilon",
  # E: + 2 prompt-side interactions (variant:judge, variant:temp).
  #    item-side interactions and σ²_ε still go to Residual (so cell + item-*
  #    interactions are 0).
  "E", "prompt",        "prompt", p$sigma2_phi,                   "phi",
  "E", "temperature",   "temp",   sigma2_tau_design,              "tau",
  "E", "judge",         "judge",  sigma2_lambda_design,           "lambda",
  "E", "item:prompt",   "i_x_p",  0,                              "alpha*phi",
  "E", "item:temp",     "i_x_t",  0,                              "alpha*tau",
  "E", "item:judge",    "i_x_j",  0,                              "alpha*lambda",
  "E", "prompt:temp",   "p_x_t",  p$sigma2_phitau,                "phi*tau",
  "E", "prompt:judge",  "p_x_j",  p$sigma2_philam,                "phi*lambda",
  "E", "temp:judge",    "t_x_j",  p$sigma2_taulam,                "tau*lambda",
  "E", "cell",          "cell",   0,                              "epsilon",
  # F: E + item-side interactions + (1|item:variant:judge:temp).
  #    All variance components captured.
  "F", "prompt",        "prompt", p$sigma2_phi,                   "phi",
  "F", "temperature",   "temp",   sigma2_tau_design,              "tau",
  "F", "judge",         "judge",  sigma2_lambda_design,           "lambda",
  "F", "item:prompt",   "i_x_p",  p$sigma2_aphi,                  "alpha*phi",
  "F", "item:temp",     "i_x_t",  p$sigma2_atau,                  "alpha*tau",
  "F", "item:judge",    "i_x_j",  p$sigma2_alam,                  "alpha*lambda",
  "F", "prompt:temp",   "p_x_t",  p$sigma2_phitau,                "phi*tau",
  "F", "prompt:judge",  "p_x_j",  p$sigma2_philam,                "phi*lambda",
  "F", "temp:judge",    "t_x_j",  p$sigma2_taulam,                "tau*lambda",
  "F", "cell",          "cell",   p$sigma2_eps_cell,              "epsilon"
)

family_palette <- c(
  prompt = col_prompt,
  temp   = col_temp,
  judge  = col_judge,
  i_x_p  = blend_hex(col_inter, col_prompt),
  i_x_t  = blend_hex(col_inter, col_temp),
  i_x_j  = blend_hex(col_inter, col_judge),
  p_x_t  = blend_hex(col_prompt, col_temp),
  p_x_j  = blend_hex(col_prompt, col_judge),
  t_x_j  = blend_hex(col_temp, col_judge),
  cell   = col_cell
)


plot_df <- captured_per_scenario %>%
  mutate(
    scenario = factor(scenario, levels = rev(c("A", "B", "C", "D", "E", "F"))),
    family = factor(family, levels = c("prompt", "temp", "judge",
                                        "i_x_p", "i_x_t", "i_x_j",
                                        "p_x_t", "p_x_j", "t_x_j", "cell")),
    component = factor(component, levels = c("prompt", "temperature", "judge",
                                              "item:prompt", "item:temp", "item:judge",
                                              "prompt:temp", "prompt:judge", "temp:judge",
                                              "cell"))
  ) %>%
  arrange(scenario, component)

scenario_totals <- plot_df %>%
  group_by(scenario) %>%
  summarize(total = sum(sigma2, na.rm = TRUE), .groups = "drop")
x_max <- max(scenario_totals$total, na.rm = TRUE)
label_threshold <- x_max * 0.06

p_variance <- ggplot(plot_df,
    aes(y = scenario, x = sigma2, fill = family, group = component)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.3,
           position = position_stack(reverse = TRUE)) +
  geom_text(
    data = plot_df %>% filter(sigma2 > 0),
    aes(label = label),
    position = position_stack(reverse = TRUE, vjust = 0.5),
    parse = TRUE, size = 2.4, colour = "white", fontface = "bold",
    show.legend = FALSE
  ) +
  # Label for scenario A (bar has zero length — only item-side captured)
  annotate("text", x = 0, y = 6,
           label = "'no design-side components (' * alpha * ', ' * rho * ' only)'",
           parse = TRUE,
    hjust = 0, vjust = 0.5, size = 2.5, fontface = "italic",
    colour = "grey30") +
  scale_fill_manual(values = family_palette, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.02)),
                     labels = NULL, breaks = NULL,
                     limits = c(0, x_max * 1.05)) +
  labs(
    x = "Variance accounted for (components modelled as random effects)",
    y = "Scenario",
    subtitle = "(b) Each step captures one more variance component in the SE"
  ) +
  theme_minimal(base_size = 9) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.text.y = element_text(face = "bold", size = 10),
    axis.title.y = element_text(size = 9)
  )

p_cov_pnas <- p_cov +
  labs(subtitle = "(a) 95% CI coverage vs. number of items") +
  guides(color = guide_legend(nrow = 2)) +
  theme(legend.text = element_text(size = 7),
        legend.key.size = unit(0.3, "cm"),
        legend.spacing.x = unit(4, "pt"))

fig_pnas <- p_cov_pnas / p_variance +
  plot_layout(heights = c(1.3, 1.0))

ggsave(file.path(fig_dir, "fig_underestimation_pnas_v2.pdf"), fig_pnas,
       width = 5.5, height = 5.8)
ggsave(file.path(fig_dir, "fig_underestimation_pnas_v2.png"), fig_pnas,
       width = 5.5, height = 5.8, dpi = 300)
cat("Saved: fig_underestimation_pnas_v2.pdf/png\n")

# --- Slide layout: same two panels side-by-side (landscape aspect) ---
# Drawn at print size: the main text includes this at 0.9\linewidth = 4.95 in, so text prints at ~7 pt.
p_cov_slide <- p_cov +
  labs(subtitle = "(a) 95% CI coverage vs. number of items") +
  guides(color = guide_legend(nrow = 2)) +
  theme_minimal(base_size = 7.5) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 6.5),
    legend.key.size = unit(0.3, "cm"),
    plot.subtitle = element_text(size = 8, face = "bold")
  )

p_variance_slide <- ggplot(plot_df,
    aes(y = scenario, x = sigma2, fill = family, group = component)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.3,
           position = position_stack(reverse = TRUE)) +
  geom_text(
    data = plot_df %>% filter(sigma2 > 0),
    aes(label = label),
    position = position_stack(reverse = TRUE, vjust = 0.5),
    parse = TRUE, size = 2.2, colour = "white", fontface = "bold",
    show.legend = FALSE
  ) +
  annotate("text", x = 0, y = 6,
           label = "'no design-side components (' * alpha * ', ' * rho * ' only)'",
           parse = TRUE,
    hjust = 0, vjust = 0.5, size = 2.1, fontface = "italic",
    colour = "grey30") +
  scale_fill_manual(values = family_palette, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.02)),
                     labels = NULL, breaks = NULL,
                     limits = c(0, x_max * 1.05)) +
  labs(
    x = "Variance accounted for (components in the model)",
    y = "Scenario",
    subtitle = "(b) Components each model captures"
  ) +
  theme_minimal(base_size = 7.5) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.text.y = element_text(face = "bold", size = 8),
    axis.title.y = element_text(size = 7.5),
    axis.title.x = element_text(size = 7),
    plot.subtitle = element_text(size = 8, face = "bold")
  )

# thinner marks for the print-size panel
p_cov_slide$layers[[2]]$aes_params$linewidth <- 0.5
p_cov_slide$layers[[3]]$aes_params$size <- 1.2
fig_slide <- p_cov_slide + p_variance_slide +
  plot_layout(ncol = 2, widths = c(1, 1), guides = "collect") &
  theme(plot.margin = margin(2, 2, 2, 2, "pt"), legend.position = "bottom")

ggsave(file.path(fig_dir, "fig_underestimation_slide_v2.pdf"), fig_slide,
       width = 4.95, height = 2.35)
ggsave(file.path(fig_dir, "fig_underestimation_slide_v2.png"), fig_slide,
       width = 4.95, height = 2.35, dpi = 300)
cat("Saved: fig_underestimation_slide_v2.pdf/png\n")

# --- Appendix figure: SE ratio + coverage decomposition ---
p_se_ratio <- exp1_summary %>%
  filter(scenario %in% c("A", "B", "C", "D", "E"), !is.na(mean_se_ratio)) %>%
  mutate(scenario_label = scenario_labels[scenario]) %>%
  ggplot(aes(x = n_items, y = mean_se_ratio, color = scenario_label,
             group = scenario_label)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  scale_color_manual(values = scenario_colors) +
  scale_x_log10(breaks = sort(unique(exp1_summary$n_items)),
                labels = scales::comma) +
  labs(
    subtitle = "(A) SE ratio (scenario / oracle F) by scenario",
    x = "Number of items (N, log scale)",
    y = "SE ratio (scenario / oracle F)",
    color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.key.size = unit(0.4, "cm"),
    legend.text = element_text(size = 8)
  )

# Coverage by N, faceted by scenario
p_cov_facet <- exp1_df %>%
  filter(!is.na(covers)) %>%
  group_by(scenario, n_items) %>%
  summarize(coverage = mean(covers, na.rm = TRUE), .groups = "drop") %>%
  mutate(scenario_label = scenario_labels[scenario]) %>%
  ggplot(aes(x = n_items, y = coverage * 100)) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 0.7, color = "steelblue") +
  geom_point(size = 2, color = "steelblue") +
  facet_wrap(~ scenario_label, nrow = 1) +
  scale_x_log10(breaks = sort(unique(exp1_df$n_items)),
                labels = scales::comma) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(
    subtitle = "(B) Coverage by scenario and N",
    x = "N items (log scale)",
    y = "Coverage (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 7)
  )

fig_appendix <- p_se_ratio / p_cov_facet +
  plot_annotation(
    title = "Underestimation Detail: SE Ratios and Coverage Decomposition",
    subtitle = paste0(n_distinct(exp1_df$sim_id), " Monte Carlo replicates per (scenario, N) combination"),
    theme = theme(
      plot.title = element_text(size = 13, face = "bold"),
      plot.subtitle = element_text(size = 10)
    )
  )

ggsave(file.path(fig_dir, "manuscript_fig_underestimation_detail_v2.pdf"), fig_appendix,
       width = 10, height = 7)
ggsave(file.path(fig_dir, "manuscript_fig_underestimation_detail_v2.png"), fig_appendix,
       width = 10, height = 7, dpi = 300)
cat("Saved: manuscript_fig_underestimation_detail_v2.pdf/png\n")


# =========================================================================
# Diagnostics
# =========================================================================
cat("\n--- Diagnostics ---\n")

diag <- exp1_df %>%
  group_by(scenario, n_items) %>%
  summarize(
    n_valid = sum(!is.na(theta_hat)),
    n_total = n(),
    convergence_pct = 100 * n_valid / n_total,
    .groups = "drop"
  )
cat("\nConvergence rates:\n")
print(diag)

# Validate σ²_ε recovery in scenario F
if ("mean_s2_eps_hat" %in% names(exp1_summary)) {
  cat("\n--- σ²_ε / σ²_ρ recovery (scenario F) ---\n")
  recov <- exp1_summary %>%
    filter(scenario == "F") %>%
    select(n_items, mean_s2_eps_hat, mean_s2_rho_hat) %>%
    mutate(
      true_eps = TRUE_PARAMS$sigma2_eps_cell,
      true_rho = TRUE_PARAMS$sigma2_rho_rep,
      eps_pct_err = 100 * (mean_s2_eps_hat - true_eps) / true_eps,
      rho_pct_err = 100 * (mean_s2_rho_hat - true_rho) / true_rho
    )
  print(recov)
}

cat("\n--- Coverage by scenario × N ---\n")
print(exp1_summary %>%
        select(scenario, n_items, coverage) %>%
        pivot_wider(names_from = n_items, values_from = coverage,
                    names_prefix = "N=") %>%
        arrange(scenario))

cat("\n=== 04g2_sim_underestimation_balanced.R complete ===\n")
