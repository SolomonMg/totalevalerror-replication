#!/usr/bin/env Rscript
# 37_arena_coverage_study.R
# Coverage study: naive vs TEE CI coverage of scoring-family mean agreement.
#
# Population: all 4,676 Chatbot Arena battles scored under the full
# (V=5 variants x J=3 judges) Likert factorial. For each (battle, variant,
# judge) cell, compute agreement with the human winner (binary). Population
# truth p* = grand mean of agreement across all ~70k cells.
#
# Naive design (single-judge, single-variant):
#   Pick one (v, j) combo uniformly at random, sample n_b battles, report
#   p_hat and binomial Wald CI. This is what a practitioner who picked
#   "gpt-oss-120b, variant 2" would write up.
#
# TEE design (full factorial):
#   Use all 5 x 3 = 15 cells on the same n_b battles, fit crossed-RE lmer
#   agree ~ 1 + (1|battle) + (1|variant) + (1|judge), extract fixed-effect
#   SE of the intercept, construct Wald CI.
#
# For each of K Monte Carlo replicates and each n_b, record whether the
# CI covers p*. TEE CI should hit nominal 95%; naive CI should fall short
# because judge and variant variance are omitted.

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(ggplot2)
  library(patchwork)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)
set.seed(42)

CACHE_PATH <- "data/processed/arena_coverage_sim.csv"
K_MC <- 300
N_B_SEQ <- c(100, 250, 500, 1000, 2000)

# ---- Build per-cell Likert agreement ----
# Each (battle, variant, judge) cell predicts a winner by sign(score_a - score_b);
# tie if equal. agree = 1{pred == human winner}. "tie (bothbad)" collapses to "tie".

build_likert_cells <- function() {
  likert <- read_csv("data/processed/arena_likert_clean.csv", show_col_types = FALSE)
  cells <- likert %>%
    mutate(human = if_else(winner == "tie (bothbad)", "tie", winner)) %>%
    select(battle_id, category_llm, human, variant_id, judge_model,
           response_side, score) %>%
    pivot_wider(names_from = response_side, values_from = score) %>%
    mutate(pred = case_when(a > b ~ "model_a",
                            b > a ~ "model_b",
                            TRUE ~ "tie"),
           agree = as.integer(pred == human)) %>%
    select(battle_id, category_llm, variant_id, judge_model, agree)
  cells
}

# ---- Monte Carlo driver ----

run_mc_one <- function(cells, n_b, K, pop) {
  battles_all <- unique(cells$battle_id)
  variants <- sort(unique(cells$variant_id))
  judges <- sort(unique(cells$judge_model))
  V <- length(variants); J <- length(judges)

  # Pre-index cells by battle for fast subsetting
  cells_dt <- cells %>% mutate(key_bv = paste(battle_id, variant_id, judge_model, sep = "|"))

  cov_n  <- integer(K); cov_t  <- integer(K)
  w_n    <- numeric(K); w_t    <- numeric(K)
  se_n   <- numeric(K); se_t   <- numeric(K)
  p_n    <- numeric(K); p_t    <- numeric(K)

  for (k in seq_len(K)) {
    b_samp <- sample(battles_all, n_b, replace = FALSE)
    sub <- cells %>% filter(battle_id %in% b_samp)

    # Naive: random (v0, j0)
    v0 <- sample(variants, 1)
    j0 <- sample(judges, 1)
    naive_d <- sub %>% filter(variant_id == v0, judge_model == j0)
    p_hat   <- mean(naive_d$agree)
    n_naive <- nrow(naive_d)
    se_naive <- sqrt(p_hat * (1 - p_hat) / n_naive)
    ci_lo_n  <- p_hat - 1.96 * se_naive
    ci_hi_n  <- p_hat + 1.96 * se_naive
    cov_n[k] <- as.integer(pop >= ci_lo_n & pop <= ci_hi_n)
    w_n[k]   <- ci_hi_n - ci_lo_n
    se_n[k]  <- se_naive
    p_n[k]   <- p_hat

    # TEE: fit crossed-RE lmer on all cells for the sampled battles
    fit <- tryCatch(
      lmer(agree ~ 1 + (1 | battle_id) + (1 | variant_id) + (1 | judge_model),
           data = sub, REML = TRUE,
           control = lmerControl(check.nobs.vs.nlev = "ignore",
                                 check.nobs.vs.nRE  = "ignore",
                                 calc.derivs = FALSE,
                                 optimizer = "bobyqa")),
      error = function(e) NULL,
      warning = function(w) suppressWarnings(
        lmer(agree ~ 1 + (1 | battle_id) + (1 | variant_id) + (1 | judge_model),
             data = sub, REML = TRUE,
             control = lmerControl(check.nobs.vs.nlev = "ignore",
                                   check.nobs.vs.nRE  = "ignore",
                                   calc.derivs = FALSE,
                                   optimizer = "bobyqa"))
      )
    )
    if (is.null(fit)) {
      cov_t[k] <- NA_integer_; w_t[k] <- NA_real_
      se_t[k]  <- NA_real_;    p_t[k] <- NA_real_
    } else {
      int   <- fixef(fit)[1]
      se_tee <- sqrt(vcov(fit)[1, 1])
      ci_lo_t <- int - 1.96 * se_tee
      ci_hi_t <- int + 1.96 * se_tee
      cov_t[k] <- as.integer(pop >= ci_lo_t & pop <= ci_hi_t)
      w_t[k]   <- ci_hi_t - ci_lo_t
      se_t[k]  <- se_tee
      p_t[k]   <- int
    }
  }

  tibble(
    n_b = n_b,
    coverage_naive = mean(cov_n, na.rm = TRUE),
    coverage_tee   = mean(cov_t, na.rm = TRUE),
    width_naive    = mean(w_n, na.rm = TRUE),
    width_tee      = mean(w_t, na.rm = TRUE),
    se_naive       = mean(se_n, na.rm = TRUE),
    se_tee         = mean(se_t, na.rm = TRUE),
    phat_naive     = mean(p_n, na.rm = TRUE),
    phat_tee       = mean(p_t, na.rm = TRUE),
    K_used_naive   = sum(!is.na(cov_n)),
    K_used_tee     = sum(!is.na(cov_t))
  )
}

# ---- Main ----

if (!file.exists(CACHE_PATH)) {
  cat("Building per-cell agreement matrix...\n")
  cells_lik <- build_likert_cells()
  pop_lik <- mean(cells_lik$agree)
  cat(sprintf("Population p* (Likert, overall) = %.4f\n", pop_lik))
  cat(sprintf("Cells: %d battles x %d variants x %d judges = %d rows\n",
              length(unique(cells_lik$battle_id)),
              length(unique(cells_lik$variant_id)),
              length(unique(cells_lik$judge_model)),
              nrow(cells_lik)))

  cat(sprintf("\nRunning Monte Carlo (K=%d per n_b)...\n", K_MC))
  t0 <- Sys.time()
  res_list <- list()
  for (n_b in N_B_SEQ) {
    cat(sprintf("  n_b = %4d ...", n_b))
    tb0 <- Sys.time()
    r <- run_mc_one(cells_lik, n_b, K = K_MC, pop = pop_lik)
    res_list[[as.character(n_b)]] <- r
    cat(sprintf(" done in %.1fs (naive cov=%.3f, tee cov=%.3f)\n",
                as.numeric(Sys.time() - tb0, units = "secs"),
                r$coverage_naive, r$coverage_tee))
  }
  res <- bind_rows(res_list) %>%
    mutate(mode = "likert", pop_truth = pop_lik) %>%
    select(mode, n_b, pop_truth, everything())
  write_csv(res, CACHE_PATH)
  cat(sprintf("\nTotal elapsed: %.1fs\nSaved %s\n",
              as.numeric(Sys.time() - t0, units = "secs"), CACHE_PATH))
} else {
  cat("Using cached", CACHE_PATH, "\n")
  res <- read_csv(CACHE_PATH, show_col_types = FALSE)
}

cat("\n=== Summary ===\n")
print(res %>% mutate(across(where(is.numeric), ~ round(., 4))))

# ---- Plot: coverage + half-width ----

# MC SE for a binomial proportion
mcse <- function(p, K) sqrt(p * (1 - p) / K)

long_cov <- res %>%
  select(n_b, K = K_used_naive, K_t = K_used_tee, coverage_naive, coverage_tee) %>%
  pivot_longer(cols = c(coverage_naive, coverage_tee),
               names_to = "design", values_to = "coverage") %>%
  mutate(design = recode(design,
                         coverage_naive = "Naive (single judge x variant)",
                         coverage_tee   = "TEE (factorial)"),
         K_used = if_else(design == "TEE (factorial)", K_t, K),
         mcse = mcse(coverage, K_used),
         lo = pmax(0, coverage - 1.96 * mcse),
         hi = pmin(1, coverage + 1.96 * mcse))

long_w <- res %>%
  select(n_b, width_naive, width_tee) %>%
  pivot_longer(-n_b, names_to = "design", values_to = "width") %>%
  mutate(design = recode(design,
                         width_naive = "Naive (single judge x variant)",
                         width_tee   = "TEE (factorial)"),
         halfw = width / 2)

col_map <- c("Naive (single judge x variant)" = "#D55E00",
             "TEE (factorial)" = "#0072B2")

# Drawn at print size (Figure 5 at 0.75\linewidth = 4.1 in) so text prints at ~6 - 7 pt.
# Coverage labels sit above TEE points and below naive points so the two series never
# collide; the half-width panel carries no value labels (the series are too close at this
# size). The caption carries the title.
lab_vjust <- function(design) if_else(design == "TEE (factorial)", -0.9, 1.9)

p_cov <- ggplot(long_cov, aes(x = n_b, y = coverage, color = design, group = design)) +
  annotate("rect", xmin = min(N_B_SEQ) * 0.85, xmax = max(N_B_SEQ) * 1.15,
           ymin = 0.95 - 0.02, ymax = 0.95 + 0.02, fill = "gray85", alpha = 0.6) +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "gray40") +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = design), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.6) +
  geom_text(aes(label = sprintf("%.2f", coverage), vjust = lab_vjust(design)), hjust = 0.5, size = 2.2, show.legend = FALSE) +
  scale_x_log10(breaks = N_B_SEQ, expand = expansion(mult = c(0.1, 0.1))) +
  scale_y_continuous(limits = c(0.70, 1.06), breaks = seq(0.7, 1.0, 0.1)) +
  scale_color_manual(values = col_map) + scale_fill_manual(values = col_map, guide = "none") +
  labs(x = "Matches sampled (n_m)", y = "95% CI coverage of p*", color = NULL) +
  theme_minimal(base_size = 7.5) + theme(legend.position = "bottom", panel.grid.minor = element_blank())

p_w <- ggplot(long_w, aes(x = n_b, y = halfw, color = design, group = design)) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.6) +
  scale_x_log10(breaks = N_B_SEQ, expand = expansion(mult = c(0.1, 0.1))) +
  scale_color_manual(values = col_map) +
  labs(x = "Matches sampled (n_m)", y = "Mean 95% CI half-width", color = NULL) +
  theme_minimal(base_size = 7.5) + theme(legend.position = "bottom", panel.grid.minor = element_blank())

p_combined <- (p_cov | p_w) + plot_layout(guides = "collect", widths = c(1, 1)) +
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")") &
  theme(legend.position = "bottom", plot.tag = element_text(face = "bold", size = 8),
        legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(1, "pt"),
        plot.margin = margin(2, 2, 2, 2, "pt"))

ggsave("figures/fig_arena_coverage.pdf", p_combined, width = 4.1, height = 2.0)
ggsave("figures/fig_arena_coverage.png", p_combined, width = 4.1, height = 2.0, dpi = 300)
cat("\nSaved figures/fig_arena_coverage.{pdf,png}\n")
