# 04g_sim_underestimation.R
# Monte Carlo simulation: consequences of omitting pipeline factors.
#
# Demonstrates that naive SE estimates (single prompt, single judge, single temp)
# dramatically underestimate true uncertainty, and that coverage WORSENS with N.
# This is the paper's central claim — current practice underestimates uncertainty —
# and this simulation provides direct evidence.
#
# Two experiments:
#   1. CI coverage across 5 scenarios of progressive sophistication (A–E)
#   2. "20 Researchers" illustration: same population, different arbitrary choices
#
# Usage:
#   Rscript analysis/04g_sim_underestimation.R            # full run (~30 min)
#   Rscript analysis/04g_sim_underestimation.R --nsim 5   # smoke test (~2 min)

library(tidyverse)
library(lme4)
library(patchwork)
# --- Parse command-line args ---
args <- commandArgs(trailingOnly = TRUE)
N_SIM <- 1000
N_CORES <- max(1, parallel::detectCores() - 1)
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

# --- Extended DGP parameters (adds judge/model to existing params) ---
TRUE_PARAMS <- list(
  mu = 0.5,
  sigma2_category = 0.03,
  sigma2_item = 0.05,
  sigma2_prompt = 0.04,
  sigma2_ip = 0.02,
  sigma2_it = 0.015,
  sigma2_pt = 0.005,
  sigma2_im = 0.025,            # item x judge (new)
  sigma2_pm = 0.005,            # prompt x judge (new)
  sigma2_gen = 0.06,
  tau = c(0, -0.03, 0.05),     # temperature fixed effects (3 levels)
  lambda = c(0, 0.03, -0.02),  # judge fixed effects (3 levels)
  n_categories = 5
)

cat("=== 04g_sim_underestimation.R: Variance Underestimation ===\n")
cat(sprintf("N_SIM = %d\n", N_SIM))

# =========================================================================
# DGP helpers
# =========================================================================

# Draw population random effects (no observations yet — cheap at any N)
draw_population <- function(n_items, n_prompts, n_temps, n_judges, params) {
  n_cats <- params$n_categories
  items_per_cat <- ceiling(n_items / n_cats)
  gamma_cat <- rnorm(n_cats, 0, sqrt(params$sigma2_category))
  item_cat <- rep(seq_len(n_cats), each = items_per_cat)[seq_len(n_items)]
  delta_item <- rnorm(n_items, 0, sqrt(params$sigma2_item))
  alpha <- gamma_cat[item_cat] + delta_item
  beta <- rnorm(n_prompts, 0, sqrt(params$sigma2_prompt))
  list(
    alpha = alpha, beta = beta, item_cat = item_cat,
    ip = matrix(rnorm(n_items * n_prompts, 0, sqrt(params$sigma2_ip)),
                nrow = n_items, ncol = n_prompts),
    it = matrix(rnorm(n_items * n_temps, 0, sqrt(params$sigma2_it)),
                nrow = n_items, ncol = n_temps),
    pt = matrix(rnorm(n_prompts * n_temps, 0, sqrt(params$sigma2_pt)),
                nrow = n_prompts, ncol = n_temps),
    im = matrix(rnorm(n_items * n_judges, 0, sqrt(params$sigma2_im)),
                nrow = n_items, ncol = n_judges),
    pm = matrix(rnorm(n_prompts * n_judges, 0, sqrt(params$sigma2_pm)),
                nrow = n_prompts, ncol = n_judges)
  )
}

# Generate observations for a slice of the factorial (only requested v/h/m levels)
generate_slice <- function(pop, params, v_set, h_set, m_set, n_reps) {
  n_items <- length(pop$alpha)
  total <- n_items * length(v_set) * length(h_set) * length(m_set) * n_reps
  item_id <- integer(total)
  variant_id <- integer(total)
  temp_id <- integer(total)
  judge_id <- integer(total)
  rep_id <- integer(total)
  category <- integer(total)
  outcome <- numeric(total)

  idx <- 0L
  for (i in seq_along(pop$alpha)) {
    for (v in v_set) {
      for (h in h_set) {
        for (m in m_set) {
          for (r in seq_len(n_reps)) {
            idx <- idx + 1L
            item_id[idx] <- i
            variant_id[idx] <- v
            temp_id[idx] <- h
            judge_id[idx] <- m
            rep_id[idx] <- r
            category[idx] <- pop$item_cat[i]
            outcome[idx] <- params$mu + pop$alpha[i] + pop$beta[v] +
              params$tau[h] + params$lambda[m] +
              pop$ip[i, v] + pop$it[i, h] + pop$pt[v, h] +
              pop$im[i, m] + pop$pm[v, m] +
              rnorm(1, 0, sqrt(params$sigma2_gen))
          }
        }
      }
    }
  }

  tibble(item_id = as.factor(item_id), variant_id = as.factor(variant_id),
         temp_id = as.factor(temp_id), judge_id = as.factor(judge_id),
         rep_id = rep_id, category = as.factor(category), outcome = outcome)
}


# =========================================================================
# Experiment 1: CI Coverage across scenarios A–E
# =========================================================================
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(output_dir, "sim_underestimation.csv")
csv_summary_path <- file.path(output_dir, "sim_underestimation_summary.csv")

if (file.exists(csv_path) && file.exists(csv_summary_path)) {
  cat("\nLoading existing simulation results from CSV...\n")
  exp1_df <- read_csv(csv_path, show_col_types = FALSE)
  exp1_summary <- read_csv(csv_summary_path, show_col_types = FALSE)
  N_levels <- sort(unique(exp1_df$n_items))
} else {
  cat("\n\n========== Experiment 1: CI Coverage ==========\n")

  N_levels_all <- c(20, 50, 100, 200, 500, 1000, 2000)
  N_levels_full_lmer <- c(20, 50, 100, 200)  # E uses lmer at small N, oracle otherwise
  V_full <- 3
  H_full <- 3
  M_full <- 3
  R_full <- 5

  # Oracle D-study SE from true params (used for E at large N, and for se_ratio)
  oracle_var_E <- function(N, V, H, M, R, p = TRUE_PARAMS) {
    (p$sigma2_category + p$sigma2_item) / N +
      p$sigma2_prompt / V +
      p$sigma2_ip / (N * V) +
      p$sigma2_it / (N * H) +
      p$sigma2_pt / (V * H) +
      p$sigma2_im / (N * M) +
      p$sigma2_pm / (V * M) +
      p$sigma2_gen / (N * V * H * M * R)
  }

  # --- Per-sim worker function (called in parallel) ---
  run_one_sim <- function(sim, n_items, V_full, H_full, M_full, R_full,
                          TRUE_PARAMS, var_E_oracle, run_full_lmer, se_E_oracle) {
    set.seed(42 + sim + n_items * 10000)

    pop <- draw_population(n_items, V_full, H_full, M_full, TRUE_PARAMS)

    v_chosen <- sample(V_full, 1)
    h_chosen <- sample(H_full, 1)
    m_chosen <- sample(M_full, 1)
    theta_true <- TRUE_PARAMS$mu + TRUE_PARAMS$tau[h_chosen] + TRUE_PARAMS$lambda[m_chosen]

    # --- Scenario A: (V=1, M=1, H=1, R=1) ---
    df_A <- generate_slice(pop, TRUE_PARAMS, v_chosen, h_chosen, m_chosen, 1)
    theta_A <- mean(df_A$outcome)
    se_A <- sd(df_A$outcome) / sqrt(n_items)

    # --- Scenario B: (V=1, M=1, H=1, R=5) ---
    df_B <- generate_slice(pop, TRUE_PARAMS, v_chosen, h_chosen, m_chosen, R_full)
    item_means_B <- df_B %>%
      group_by(item_id) %>%
      summarize(y_bar = mean(outcome), .groups = "drop")
    theta_B <- mean(item_means_B$y_bar)
    se_B <- sd(item_means_B$y_bar) / sqrt(n_items)

    # --- Scenario C: lmer with main effects only (V=3, M=1, H=1, R=5) ---
    # Same data as D but the item x variant interaction is dropped; its
    # variance gets absorbed into the residual.
    # At large N, use oracle D-study SE (REML converges to truth anyway;
    # fitting lmer at N > 200 becomes prohibitive for the item:variant term).
    df_C <- generate_slice(pop, TRUE_PARAMS, seq_len(V_full), h_chosen, m_chosen, R_full)
    theta_C <- mean(df_C$outcome)
    p <- TRUE_PARAMS
    if (run_full_lmer) {
      mod_C <- tryCatch(
        lmer(outcome ~ (1 | item_id) + (1 | variant_id),
             data = df_C, REML = TRUE,
             control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE, optCtrl = list(maxfun = 10000))),
        error = function(e) NULL
      )
      if (!is.null(mod_C)) {
        vc_C <- as.data.frame(VarCorr(mod_C))
        s2 <- setNames(vc_C$vcov, vc_C$grp)
        var_C <- (s2["item_id"] %||% 0) / n_items +
          (s2["variant_id"] %||% 0) / V_full +
          (s2["Residual"] %||% 0) / (n_items * V_full * R_full)
        theta_C <- fixef(mod_C)[1]
        se_C <- sqrt(max(var_C, 1e-12))
      } else {
        se_C <- NA
      }
    } else {
      # Oracle D-study SE: main effects only (item:variant absorbed into residual,
      # along with ATH/ALH/PTH/PLH at the fixed (h_c, m_c))
      var_C <- (p$sigma2_category + p$sigma2_item +
                p$sigma2_it + p$sigma2_im) / n_items +
               (p$sigma2_prompt +
                p$sigma2_pt + p$sigma2_pm) / V_full +
               (p$sigma2_ip + p$sigma2_gen) / (n_items * V_full * R_full)
      se_C <- sqrt(var_C)
    }

    # --- Scenario D: + item x prompt interaction (V=3, M=1, H=1, R=5) ---
    theta_D <- mean(df_C$outcome)
    if (run_full_lmer) {
      mod_D <- tryCatch(
        lmer(outcome ~ (1 | item_id) + (1 | variant_id) + (1 | item_id:variant_id),
             data = df_C, REML = TRUE,
             control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE, optCtrl = list(maxfun = 10000))),
        error = function(e) NULL
      )
      if (!is.null(mod_D)) {
        vc_D <- as.data.frame(VarCorr(mod_D))
        s2 <- setNames(vc_D$vcov, vc_D$grp)
        var_D <- (s2["item_id"] %||% 0) / n_items +
          (s2["variant_id"] %||% 0) / V_full +
          (s2["item_id:variant_id"] %||% 0) / (n_items * V_full) +
          (s2["Residual"] %||% 0) / (n_items * V_full * R_full)
        theta_D <- fixef(mod_D)[1]
        se_D <- sqrt(max(var_D, 1e-12))
      } else {
        se_D <- NA
      }
    } else {
      # Oracle D-study SE: explicit item x prompt interaction
      var_D <- (p$sigma2_category + p$sigma2_item +
                p$sigma2_it + p$sigma2_im) / n_items +
               (p$sigma2_prompt +
                p$sigma2_pt + p$sigma2_pm) / V_full +
               p$sigma2_ip / (n_items * V_full) +
               p$sigma2_gen / (n_items * V_full * R_full)
      se_D <- sqrt(var_D)
    }

    # --- Scenario E: Full TEE (V=3, M=3, H=3, R=5) ---
    theta_E <- NA; se_E <- NA; var_E <- var_E_oracle
    if (run_full_lmer) {
      df_full <- generate_slice(pop, TRUE_PARAMS,
                                seq_len(V_full), seq_len(H_full), seq_len(M_full), R_full)
      mod_E <- tryCatch(
        lmer(outcome ~ temp_id + judge_id +
               (1 | item_id) + (1 | variant_id) +
               (1 | item_id:variant_id) + (1 | item_id:temp_id) +
               (1 | variant_id:temp_id) + (1 | item_id:judge_id) +
               (1 | variant_id:judge_id),
             data = df_full, REML = TRUE,
             control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE, optCtrl = list(maxfun = 20000))),
        error = function(e) NULL
      )
      if (!is.null(mod_E)) {
        vc_E <- as.data.frame(VarCorr(mod_E))
        s2_E <- setNames(vc_E$vcov, vc_E$grp)
        var_E <- (s2_E["item_id"] %||% 0) / n_items +
          (s2_E["variant_id"] %||% 0) / V_full +
          (s2_E["item_id:variant_id"] %||% 0) / (n_items * V_full) +
          (s2_E["item_id:temp_id"] %||% 0) / (n_items * H_full) +
          (s2_E["variant_id:temp_id"] %||% 0) / (V_full * H_full) +
          (s2_E["item_id:judge_id"] %||% 0) / (n_items * M_full) +
          (s2_E["variant_id:judge_id"] %||% 0) / (V_full * M_full) +
          (s2_E["Residual"] %||% 0) / (n_items * V_full * H_full * M_full * R_full)
        fe_E <- fixef(mod_E)
        theta_E <- fe_E[1]
        h_name <- paste0("temp_id", h_chosen)
        m_name <- paste0("judge_id", m_chosen)
        if (h_name %in% names(fe_E)) theta_E <- theta_E + fe_E[h_name]
        if (m_name %in% names(fe_E)) theta_E <- theta_E + fe_E[m_name]
        se_E <- sqrt(max(var_E, 1e-12))
      }
    } else {
      # Oracle E at large N: mimic lmer by computing grand mean + fixed-effect
      # deviations from full factorial data. Use oracle D-study SE.
      df_full <- generate_slice(pop, TRUE_PARAMS,
                                seq_len(V_full), seq_len(H_full), seq_len(M_full), R_full)
      grand_mean <- mean(df_full$outcome)
      temp_means <- tapply(df_full$outcome, df_full$temp_id, mean)
      judge_means <- tapply(df_full$outcome, df_full$judge_id, mean)
      tau_hat <- as.numeric(temp_means - grand_mean)
      lambda_hat <- as.numeric(judge_means - grand_mean)
      theta_E <- as.numeric(grand_mean + tau_hat[h_chosen] + lambda_hat[m_chosen])
      se_E <- se_E_oracle
    }

    tibble(
      scenario = c("A", "B", "C", "D", "E"),
      label = c("Standard benchmark (V=1, M=1, R=1)",
                "With replications (V=1, M=1, R=5)",
                "lmer main effects only (V=3, M=1, R=5)",
                "lmer + item x prompt interaction (V=3, M=1, R=5)",
                "Full TEE (multi-judge)"),
      theta_hat = unname(c(theta_A, theta_B, theta_C, theta_D, theta_E)),
      se = unname(c(se_A, se_B, se_C, se_D, se_E)),
      ci_lo = theta_hat - 1.96 * se,
      ci_hi = theta_hat + 1.96 * se,
      covers = as.logical(ci_lo <= theta_true & theta_true <= ci_hi),
      se_ratio = se / sqrt(max(var_E, 1e-12)),
      n_items = n_items,
      sim_id = sim,
      theta_true = theta_true
    )
  }

  # --- Run simulation in parallel ---
  exp1_results <- list()

  for (n_items in N_levels_all) {
    cat(sprintf("\n  N = %d items (%d sims, %d workers)...\n", n_items, N_SIM, N_CORES))
    t0 <- Sys.time()
    run_full_lmer <- n_items %in% N_levels_full_lmer

    var_E_oracle <- oracle_var_E(n_items, V_full, H_full, M_full, R_full)
    se_E_oracle <- sqrt(var_E_oracle)

    results_n <- parallel::mclapply(seq_len(N_SIM), function(sim) {
      run_one_sim(sim, n_items, V_full, H_full, M_full, R_full,
                  TRUE_PARAMS, var_E_oracle, run_full_lmer, se_E_oracle)
    }, mc.cores = N_CORES)
    results_n <- Filter(is.data.frame, results_n)
    exp1_results <- c(exp1_results, results_n)
    elapsed <- round(difftime(Sys.time(), t0, units = "secs"), 1)
    cat(sprintf("    done in %s sec\n", elapsed))
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
      n_sims = n(),
      .groups = "drop"
    )

  cat("\n\nExperiment 1 — Coverage summary:\n")
  print(exp1_summary %>% select(scenario, n_items, coverage, mean_se_ratio, n_sims) %>%
          arrange(scenario, n_items))

  # Save
  write_csv(exp1_df, csv_path)
  write_csv(exp1_summary, csv_summary_path)
  cat("\nSaved: sim_underestimation.csv, sim_underestimation_summary.csv\n")
}


# =========================================================================
# Experiment 2: "20 Researchers" illustration
# =========================================================================
csv_20r_path <- file.path(output_dir, "sim_underestimation_20researchers.csv")

if (file.exists(csv_20r_path)) {
  cat("\nLoading existing 20-researchers data from CSV...\n")
  researchers_df <- read_csv(csv_20r_path, show_col_types = FALSE)
  # Recompute TEE SE from known params (deterministic given the CSV)
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
  R_pop <- 5

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

  # Full TEE model
  mod_full <- tryCatch(
    lmer(outcome ~ temp_id + judge_id +
           (1 | item_id) + (1 | variant_id) +
           (1 | item_id:variant_id) + (1 | item_id:temp_id) +
           (1 | variant_id:temp_id) + (1 | item_id:judge_id) +
           (1 | variant_id:judge_id),
         data = df_pop, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE, optCtrl = list(maxfun = 20000))),
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
  cat("Saved: sim_underestimation_20researchers.csv\n")
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

# --- Panel B: Coverage vs N for scenarios A–E (log x-axis) ---
scenario_labels <- c(
  "A" = "A: 1 obs/item",
  "B" = "B: + reps",
  "C" = "C: lmer main effects",
  "D" = "D: + item x prompt",
  "E" = "E: full TEE"
)

p_cov <- exp1_summary %>%
  filter(!is.na(coverage)) %>%
  mutate(scenario_label = scenario_labels[scenario]) %>%
  ggplot(aes(x = n_items, y = coverage * 100, color = scenario_label,
             group = scenario_label)) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  scale_color_brewer(palette = "Set1") +
  scale_x_log10(breaks = c(20, 50, 100, 200, 500, 1000, 2000),
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

ggsave(file.path(fig_dir, "fig_underestimation.pdf"), fig_main,
       width = 11, height = 5.5)
ggsave(file.path(fig_dir, "fig_underestimation.png"), fig_main,
       width = 11, height = 5.5, dpi = 300)
cat("Saved: fig_underestimation.pdf/png\n")

# --- PNAS main text figure: variance budget (a) + coverage (b) ---
# Panel (a) shows which true variance components each scenario's SE captures.
# No lmer fits needed here — the captured/blind classification is deterministic
# from each scenario's estimator.

true_components <- tribble(
  ~component,       ~family,   ~sigma2,                   ~label,
  "item",           "item",    TRUE_PARAMS$sigma2_item,   "alpha",
  "category",       "item",    TRUE_PARAMS$sigma2_category, "gamma",
  "residual",       "item",    TRUE_PARAMS$sigma2_gen,    "epsilon",
  "prompt",         "prompt",  TRUE_PARAMS$sigma2_prompt, "rho",
  "item:prompt",    "i_x_p",   TRUE_PARAMS$sigma2_ip,     "alpha*rho",
  "prompt:temp",    "p_x_t",   TRUE_PARAMS$sigma2_pt,     "rho*tau",
  "item:temp",      "i_x_t",   TRUE_PARAMS$sigma2_it,     "alpha*tau",
  "item:judge",     "i_x_j",   TRUE_PARAMS$sigma2_im,     "alpha*lambda",
  "prompt:judge",   "p_x_j",   TRUE_PARAMS$sigma2_pm,     "rho*lambda"
)

# Captured/blind classification by scenario.
# Rule: a component is "captured" iff the scenario's design has multiple
# levels of every factor in that component AND the estimator properly
# models it. A design with V=1 cannot identify any prompt-related component.
status_map <- tribble(
  ~scenario, ~component,     ~status,
  # A: V=1, M=1, H=1, R=1 — only item and category identifiable
  "A", "item",         "captured",
  "A", "category",     "captured",
  "A", "residual",     "blind",    # R=1, not separable from item
  "A", "item:prompt",  "blind",    # V=1
  "A", "item:temp",    "blind",    # H=1
  "A", "item:judge",   "blind",    # M=1
  "A", "prompt",       "blind",    # V=1
  "A", "prompt:temp",  "blind",    # V=1, H=1
  "A", "prompt:judge", "blind",    # V=1, M=1
  # B: + replications (V=1, M=1, H=1, R=5) — separates residual
  "B", "item",         "captured",
  "B", "category",     "captured",
  "B", "residual",     "captured", # R>1 now separates
  "B", "item:prompt",  "blind",    # V=1
  "B", "item:temp",    "blind",    # H=1
  "B", "item:judge",   "blind",    # M=1
  "B", "prompt",       "blind",    # V=1
  "B", "prompt:temp",  "blind",
  "B", "prompt:judge", "blind",
  # C: lmer main effects only (V=3, M=1, H=1, R=5) — prompt captured,
  # item x prompt interaction absorbed into residual
  "C", "item",         "captured",
  "C", "category",     "captured",
  "C", "residual",     "captured",
  "C", "item:prompt",  "blind",    # no interaction term in the model
  "C", "item:temp",    "blind",    # H=1
  "C", "item:judge",   "blind",    # M=1
  "C", "prompt",       "captured", # properly modeled as random effect
  "C", "prompt:temp",  "blind",
  "C", "prompt:judge", "blind",
  # D: crossed lmer single judge (V=3, M=1, H=1, R=5) — captures prompt family
  "D", "item",         "captured",
  "D", "category",     "captured",
  "D", "residual",     "captured",
  "D", "item:prompt",  "captured", # V>1, properly modeled
  "D", "item:temp",    "blind",    # H=1
  "D", "item:judge",   "blind",    # M=1
  "D", "prompt",       "captured",
  "D", "prompt:temp",  "blind",    # H=1
  "D", "prompt:judge", "blind",    # M=1
  # E: full TEE (V=3, M=3, H=3, R=5) — captures everything
  "E", "item",         "captured",
  "E", "category",     "captured",
  "E", "residual",     "captured",
  "E", "item:prompt",  "captured",
  "E", "item:temp",    "captured",
  "E", "item:judge",   "captured",
  "E", "prompt",       "captured",
  "E", "prompt:temp",  "captured",
  "E", "prompt:judge", "captured"
)

component_order <- c("item", "category", "residual",
                     "prompt", "item:prompt",
                     "prompt:temp", "item:temp",
                     "item:judge", "prompt:judge")

plot_df <- status_map %>%
  left_join(true_components, by = "component") %>%
  mutate(
    scenario = factor(scenario, levels = rev(c("A", "B", "C", "D", "E"))),
    family = factor(family, levels = c("item", "prompt",
                                        "i_x_p", "i_x_t", "i_x_j",
                                        "p_x_t", "p_x_j")),
    status = factor(status, levels = c("captured", "blind")),
    component = factor(component, levels = component_order)
  ) %>%
  arrange(scenario, component)

# Base family colors for main effects + residual
col_item   <- "#3B6FB6"   # blue (item, category, residual)
col_prompt <- "#E07B00"   # orange (prompt main effect)
col_temp   <- "#7A7A7A"   # gray (temp placeholder)
col_judge  <- "#C0392B"   # red  (judge placeholder)

# Helper: blend two hex colors (simple RGB average)
blend_hex <- function(a, b) {
  ra <- col2rgb(a); rb <- col2rgb(b)
  rgb((ra[1]+rb[1])/2, (ra[2]+rb[2])/2, (ra[3]+rb[3])/2, maxColorValue = 255)
}

# Interaction colors = blends of parent colors
family_palette <- c(
  item    = col_item,
  prompt  = col_prompt,
  i_x_p   = blend_hex(col_item,   col_prompt),   # item x prompt
  i_x_t   = blend_hex(col_item,   col_temp),     # item x temp
  i_x_j   = blend_hex(col_item,   col_judge),    # item x judge (purple)
  p_x_t   = blend_hex(col_prompt, col_temp),     # prompt x temp
  p_x_j   = blend_hex(col_prompt, col_judge)     # prompt x judge
)

p_variance <- ggplot(plot_df,
    aes(y = scenario, x = sigma2, fill = family, alpha = status,
        group = component)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.3,
           position = position_stack(reverse = TRUE)) +
  # Labels inside each segment (Greek symbols matching paper notation).
  # Only label segments wide enough to accommodate text.
  geom_text(
    data = plot_df %>% filter(sigma2 >= 0.015),
    aes(label = label),
    position = position_stack(reverse = TRUE, vjust = 0.5),
    parse = TRUE, size = 2.7, colour = "white", fontface = "bold",
    show.legend = FALSE
  ) +
  scale_fill_manual(values = family_palette, guide = "none") +
  scale_alpha_manual(values = c(captured = 1.0, blind = 0.18), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.02)),
                     labels = NULL, breaks = NULL) +
  labs(
    x = expression("True variance components (summing to " * sigma^2 * ")"),
    y = "Scenario",
    subtitle = "(b) Variance budget: which components each SE accounts for"
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
  guides(color = guide_legend(nrow = 1)) +
  theme(legend.text = element_text(size = 7),
        legend.key.size = unit(0.3, "cm"),
        legend.spacing.x = unit(4, "pt"))

fig_pnas <- p_cov_pnas / p_variance +
  plot_layout(heights = c(1.3, 1.0))

ggsave(file.path(fig_dir, "fig_underestimation_pnas.pdf"), fig_pnas,
       width = 5.5, height = 5.5)
ggsave(file.path(fig_dir, "fig_underestimation_pnas.png"), fig_pnas,
       width = 5.5, height = 5.5, dpi = 300)
cat("Saved: fig_underestimation_pnas.pdf/png\n")

# --- Appendix figure: SE ratio + coverage decomposition ---
p_se_ratio <- exp1_summary %>%
  filter(scenario %in% c("A", "B", "C", "D"), !is.na(mean_se_ratio)) %>%
  mutate(scenario_label = scenario_labels[scenario]) %>%
  ggplot(aes(x = n_items, y = mean_se_ratio, color = scenario_label,
             group = scenario_label)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  scale_color_brewer(palette = "Set1") +
  scale_x_log10(breaks = c(20, 50, 100, 200, 500, 1000, 2000),
                labels = scales::comma) +
  labs(
    subtitle = "(A) SE ratio (naive / TEE) by scenario",
    x = "Number of items (N, log scale)",
    y = "SE ratio (naive / full TEE)",
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
  scale_x_log10(breaks = c(50, 500, 2000), labels = scales::comma) +
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

ggsave(file.path(fig_dir, "manuscript_fig_underestimation_detail.pdf"), fig_appendix,
       width = 10, height = 7)
ggsave(file.path(fig_dir, "manuscript_fig_underestimation_detail.png"), fig_appendix,
       width = 10, height = 7, dpi = 300)
cat("Saved: manuscript_fig_underestimation_detail.pdf/png\n")


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

cat("\n=== 04g_sim_underestimation.R complete ===\n")
