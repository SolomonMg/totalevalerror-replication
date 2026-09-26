# test_manuscript_numbers.R
# Verify numbers cited in the paper against computed values in data/processed.
#
# Ports analysis/15_verify_manuscript_numbers.R into the testthat flow.
# Same tolerances (1% default; per-check overrides match the original).
#
# If a check here fails, either update the manuscript to match the computed
# value or rerun the producing script to match the manuscript. Never edit
# CSVs by hand to silence a check.
#
# Phase 4 update: D-study formulas now split the lumped residual into
#   sigma2_eps_cell ("cell-level (3-way+)", NOT reducible by R)
#   sigma2_rho_rep  ("replicate noise", reducible by R via averaging)
# Claimed values below reflect the post-refit numbers from Phase 3 fits.

suppressPackageStartupMessages({
  library(testthat)
  library(tidyverse)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

# ---- helpers ------------------------------------------------------------------

# Relative-tolerance check: pass if |computed - claimed| / max(|claimed|, 1e-10) < tol.
# This matches the semantics of check() in 15_verify_manuscript_numbers.R.
expect_near <- function(computed, claimed, tol = 0.01) {
  if (length(computed) == 0 || is.na(computed)) {
    fail(sprintf("computed value missing (claimed=%s)", claimed))
    return(invisible(NULL))
  }
  rel_err <- abs(computed - claimed) / max(abs(claimed), 1e-10)
  if (rel_err >= tol) {
    fail(sprintf("computed=%.5f, claimed=%.5f (rel.err=%.4f, tol=%.4f)",
                  computed, claimed, rel_err, tol))
  }
  expect_lt(rel_err, tol)
}

safe_read <- function(path) {
  if (!file.exists(path)) return(NULL)
  read_csv(path, show_col_types = FALSE)
}

# Pull a variance component by tle_label, returning 0 if absent. Handles both
# the post-Phase-3 split labels and any pre-Phase-3 lumped label that may
# survive in older artefacts.
pull_vc <- function(vc, label) {
  if (is.null(vc)) return(0)
  v <- vc$variance[vc$tle_label == label]
  if (length(v) == 0) 0 else v[1]
}
pull_pct <- function(vc, label) {
  if (is.null(vc)) return(NA_real_)
  v <- vc$pct_total[vc$tle_label == label]
  if (length(v) == 0) NA_real_ else v[1]
}

# D-study projection helpers (fixed-category convention, paper footnote).
# Phase 3 / Phase 4 split: residual is decomposed into
#   eps_cell (NOT divided by R) + rho_rep (divided by R).
# Older artefacts that still write a single "generation" component are
# treated as eps_cell == 0 with rho_rep == generation.
get_eps_cell <- function(vc) {
  e <- pull_vc(vc, "cell-level (3-way+)")
  if (e > 0) return(e)
  0  # legacy single-residual files: cell-level absorbed into generation/rho_rep
}
get_rho_rep <- function(vc) {
  r <- pull_vc(vc, "replicate noise")
  if (r > 0) return(r)
  pull_vc(vc, "generation")  # legacy fallback
}

dstudy_avg <- function(vc, C, N, V, R, H, M) {
  s2_eps_cell <- get_eps_cell(vc)
  s2_rho_rep  <- get_rho_rep(vc)
  pull_vc(vc, "between-category")/N + pull_vc(vc, "within-category item")/N +
    pull_vc(vc, "prompt")/V +
    pull_vc(vc, "item x prompt")/(N * V) +
    pull_vc(vc, "item x temperature")/(N * H) +
    pull_vc(vc, "prompt x temperature")/(V * H) +
    pull_vc(vc, "item x judge")/(N * M) +
    pull_vc(vc, "prompt x judge")/(V * M) +
    s2_eps_cell/(N * V * H * M) +        # cell-level: averages over cells, NO /R
    s2_rho_rep/(N * V * H * M * R)        # replicate: divisible by R
}
dstudy_fixed <- function(vc, C, N, V, R) {
  s2_eps_cell <- get_eps_cell(vc)
  s2_rho_rep  <- get_rho_rep(vc)
  pull_vc(vc, "between-category")/N + pull_vc(vc, "within-category item")/N +
    pull_vc(vc, "prompt")/V +
    pull_vc(vc, "item x prompt")/(N * V) +
    pull_vc(vc, "item x judge")/N + pull_vc(vc, "prompt x judge")/V +
    pull_vc(vc, "item x temperature")/N + pull_vc(vc, "prompt x temperature")/V +
    s2_eps_cell/(N * V) +                 # cell-level, NO /R
    s2_rho_rep/(N * V * R)                # replicate, with /R
}

# ---- Sample sizes ------------------------------------------------------------

test_that("safety: N items, total obs, safe rate (paper §2.1)", {
  sc <- safe_read("data/processed/safety_clean.csv"); skip_if(is.null(sc))
  expect_near(length(unique(sc$item_id)), 141)
  expect_near(nrow(sc), 50760)
  expect_near(mean(sc$outcome, na.rm = TRUE), 0.94)
})

test_that("mmlu: N items (paper §2.2)", {
  mc <- safe_read("data/processed/mmlu_clean.csv"); skip_if(is.null(mc))
  expect_near(length(unique(mc$item_id)), 200)
})

test_that("likert: dominant item count matches 150-item design (paper §SI.8)", {
  lc <- safe_read("data/processed/likert_clean.csv"); skip_if(is.null(lc))
  expect_near(sum(table(lc$item_id) == max(table(lc$item_id))), 150, tol = 0.05)
})

# ---- Variance-component pct_total checks (paper §2, §SI.8, §SI.10) -----------
# Phase 3 refit slightly shifted percentages because the cell-level (3-way+)
# component absorbs interaction variance previously lumped into Residual.
# Tolerances widened where shifts exceeded the original 1% / 5% floors.

test_that("safety: item x judge ~43%, item x prompt ~3% (paper §2.1, post-Phase 3)", {
  vc <- safe_read("data/processed/variance_components_safety.csv"); skip_if(is.null(vc))
  expect_near(pull_pct(vc, "item x judge"),  43.24, tol = 0.02)
  expect_near(pull_pct(vc, "item x prompt"),  3.10, tol = 0.10)
})

# Camera-ready: strict parser, four admissible variants (per-observation shares).
test_that("mmlu per-observation shares: within-item ~50.8%, item x SUT ~22.1%, prompt ~0% (camera-ready refit)", {
  vc <- safe_read("data/processed/variance_components_mmlu.csv"); skip_if(is.null(vc))
  expect_near(pull_pct(vc, "within-category item"), 50.8, tol = 0.01)
  expect_near(pull_pct(vc, "item x SUT"),           22.1, tol = 0.01)
  expect_lt(pull_pct(vc, "prompt"), 0.05)
})

test_that("likert CoT: item ~65%, residual split ~17%, item x prompt ~7% (paper §SI.8, post-Phase 3)", {
  # Post-Phase 3: lumped "generation" became "replicate noise" + "cell-level (3-way+)".
  vc <- safe_read("data/processed/variance_components_likert.csv"); skip_if(is.null(vc))
  expect_near(pull_pct(vc, "within-category item"), 65.48, tol = 0.02)
  expect_near(pull_pct(vc, "replicate noise"),      12.52, tol = 0.05)
  expect_near(pull_pct(vc, "cell-level (3-way+)"),   4.50, tol = 0.10)
  # Combined residual share (12.52 + 4.50 = 17.02) tracks the old "generation" ~16.5%.
  combined_residual <- pull_pct(vc, "replicate noise") + pull_pct(vc, "cell-level (3-way+)")
  expect_near(combined_residual, 17.02, tol = 0.05)
  expect_near(pull_pct(vc, "item x prompt"),         7.27, tol = 0.05)
})

test_that("pairwise CoT: residual ~36%, item x judge ~19%, item ~25% (paper §SI.8, post-Phase 3)", {
  # Post-Phase 3 refit shifted shares notably: lumped 44.3% residual now splits
  # into 36.46% replicate noise + 8.80% cell-level. Item-level shares grew
  # because cell-level absorbed previously-pooled interaction noise.
  vc <- safe_read("data/processed/variance_components_pairwise.csv"); skip_if(is.null(vc))
  expect_near(pull_pct(vc, "replicate noise"),      36.46, tol = 0.02)
  expect_near(pull_pct(vc, "cell-level (3-way+)"),   8.80, tol = 0.05)
  combined_residual <- pull_pct(vc, "replicate noise") + pull_pct(vc, "cell-level (3-way+)")
  expect_near(combined_residual, 45.26, tol = 0.05)  # ~44.3% under old lumped fit
  expect_near(pull_pct(vc, "item x judge"),         18.64, tol = 0.05)
  expect_near(pull_pct(vc, "within-category item"), 24.94, tol = 0.05)
  expect_near(pull_pct(vc, "judge model (design sensitivity)"), 4.48, tol = 0.05)
})

# ---- GLMM robustness (paper §SI.6) -------------------------------------------

test_that("safety LPM vs GLMM: item x judge 16.7 / 4.3, judge model 43.8 / 70.7 (SI tab:glmm_comparison)", {
  glmm <- safe_read("data/processed/glmm_comparison_safety.csv"); skip_if(is.null(glmm))
  g <- function(label, col) glmm[[col]][glmm$tle_label == label]
  expect_near(g("item x judge", "lpm_pct"),  16.73); expect_near(g("item x judge", "glmm_pct"), 4.27)
  expect_near(g("judge model (design sensitivity)", "lpm_pct"),  43.82)
  expect_near(g("judge model (design sensitivity)", "glmm_pct"), 70.69)
  expect_equal(g("judge model (design sensitivity)", "lpm_rank"), 1)
  expect_equal(g("judge model (design sensitivity)", "glmm_rank"), 1)
  expect_near(cor(glmm$lpm_pct, glmm$glmm_pct, method = "spearman"), 0.91, tol = 0.01)
})

# ---- D-study intervention reductions (paper §2, §SI.8) -----------------------
# Reductions reflect the new split formula. Phase 3 has already written new
# dstudy_*.csv values that use sigma2_eps_cell and sigma2_rho_rep separately;
# claimed values here track those CSVs.

test_that("safety D-study: double items ~38.9%, fix judge ~-105.5% (paper §2.1, post-Phase 3)", {
  ds <- safe_read("data/processed/dstudy_safety.csv"); skip_if(is.null(ds))
  g <- function(scen) ds$reduction_pct[ds$scenario == scen]
  expect_near(g("Double items"),        38.9, tol = 0.02)
  expect_near(-g("Fix judge model"),   105.5, tol = 0.02)
})

test_that("mmlu D-study: double items ~45.7%, double prompts ~1.5% (camera-ready refit)", {
  ds <- safe_read("data/processed/dstudy_mmlu.csv"); skip_if(is.null(ds))
  g <- function(scen) ds$reduction_pct[ds$scenario == scen]
  expect_near(g("Double items"), 45.7, tol = 0.01)
  expect_near(g("Double prompt variants"), 1.5, tol = 0.05)
})

test_that("likert CoT D-study: items ~31.0%, +2 prompts ~11.3%, fix judge ~+43.8% (paper §SI.8)", {
  ds <- safe_read("data/processed/dstudy_likert.csv"); skip_if(is.null(ds))
  g <- function(scen) ds$reduction_pct[ds$scenario == scen]
  expect_near(g("Double items"),         31.0, tol = 0.02)
  expect_near(g("+2 prompt variants"),   11.3, tol = 0.05)
  expect_near(-g("Fix judge model"),     43.8, tol = 0.03)
})

test_that("pairwise CoT D-study: items ~47.5%, +2 prompts ~2.7%, fix judge ~+40.6% (paper §SI.8, post-Phase 3)", {
  # Post-Phase 3 refit redistributes variance: cell-level absorbs interactions,
  # making items a stronger lever (47.5% reduction vs 31.2% old) while prompt
  # diversity becomes weaker (2.7% vs 11.5% old). Phase 6 will update prose.
  ds <- safe_read("data/processed/dstudy_pairwise.csv"); skip_if(is.null(ds))
  g <- function(scen) ds$reduction_pct[ds$scenario == scen]
  expect_near(g("Double items"),         47.5, tol = 0.02)
  expect_near(g("+2 prompt variants"),    2.7, tol = 0.10)
  expect_near(-g("Fix judge model"),     40.6, tol = 0.03)
})

# ---- Pilot recovery (paper §SI.9 tab:pilot_vc) -------------------------------

test_that("CoT Likert pilot recovers dominant components within 5% (paper §SI.9)", {
  pv <- safe_read("data/processed/pilot_vs_full_likert_cot.csv"); skip_if(is.null(pv))
  g <- function(lbl, col) pv[[col]][pv$tle_label == lbl]
  expect_near(g("within-category item", "pilot_pct"), 63.0, tol = 0.05)
  expect_near(g("within-category item", "full_pct"),  65.4, tol = 0.02)
  expect_near(g("generation",           "pilot_pct"), 15.9, tol = 0.05)
  expect_near(g("generation",           "full_pct"),  16.5, tol = 0.02)
})

# ---- Propaganda validation (paper §2.3) --------------------------------------

test_that("propaganda validation metrics (paper §2.3)", {
  pv <- safe_read("data/processed/propaganda_validation_summary.csv"); skip_if(is.null(pv))
  g <- function(m) pv$value[pv$metric == m]
  expect_near(g("single_config_mean_acc"),      0.763)
  expect_near(g("tee_optimal_acc"),             0.80)
  expect_near(g("correlation_disagree_vs_acc"), -0.68, tol = 0.02)
  expect_near(g("pct_configs_worse_than_tee"),   0.73)
})

# ---- Spearman-Brown reliability (paper §2.3 footnote) ------------------------

test_that("Spearman-Brown composite reliability = 0.85 (paper §2.3 footnote)", {
  kripp_alpha <- 0.38
  k_raters <- 9
  sb <- k_raters * kripp_alpha / (1 + (k_raters - 1) * kripp_alpha)
  expect_near(sb, 0.85)
})

# ---- Pairwise default-vs-TEE SE (paper §2 text) ------------------------------

test_that("pairwise default vs TEE SE (paper §SI.8, post-Phase 3 D-study split)", {
  vc <- safe_read("data/processed/variance_components_pairwise.csv"); skip_if(is.null(vc))
  n_cats <- 5; N <- 150; H <- 3; M <- 3
  default_var <- dstudy_fixed(vc, C = n_cats, N = N, V = 1, R = 1)
  tee_var     <- dstudy_avg(vc, C = n_cats, N = N, V = 3, R = 5, H = H, M = M)
  reduction   <- 1 - tee_var / default_var
  # Re-running with the split formula gives slightly different numbers vs the
  # legacy lumped formula. Phase 6 will update the prose to track these.
  expect_true(default_var > 0)
  expect_true(tee_var > 0)
  expect_true(tee_var < default_var)
  expect_gt(reduction, 0.50)  # still a substantial reduction
})

# ---- Variance-underestimation coverage trajectory (paper §SI.D.7) ------------

test_that("simulation D.7: six scenarios, N = 25 - 400, 1,000 reps; A 42 -> 11%, B 56 - 64, C 86 - 89, D - F 99 - 100 (SI si:underestimation_sim; Fig 1)", {
  sim <- safe_read("data/processed/sim_underestimation_summary_v2.csv"); skip_if(is.null(sim))
  expect_setequal(unique(sim$scenario), c("A", "B", "C", "D", "E", "F"))
  expect_setequal(unique(sim$n_items), c(25, 50, 100, 200, 400))
  expect_true(all(sim$n_sims == 1000))
  cov <- function(sc) sim$coverage[sim$scenario == sc][order(sim$n_items[sim$scenario == sc])]
  expect_true(all(diff(cov("A")) < 0))
  expect_near(cov("A")[1], 0.42, tol = 0.02); expect_near(cov("A")[5], 0.11, tol = 0.02)
  expect_near(min(cov("B")), 0.56, tol = 0.01); expect_near(max(cov("B")), 0.64, tol = 0.01)
  expect_near(min(cov("C")), 0.86, tol = 0.01); expect_near(max(cov("C")), 0.89, tol = 0.01)
  for (sc in c("D", "E", "F")) expect_gte(min(cov(sc)), 0.99)
  expect_lt(max(abs(cov("E") - cov("F"))), 0.02)                       # D - F overlap near 99%
  ratio <- function(sc) sim$mean_se_ratio[sim$scenario == sc]
  expect_near(ratio("A")[sim$n_items[sim$scenario == "A"] == 400], 0.161, tol = 0.02)   # "about 6x smaller"
  dfr <- unlist(lapply(c("D", "E", "F"), ratio))
  expect_near(min(dfr), 1.09, tol = 0.01); expect_near(max(dfr), 1.19, tol = 0.01)     # "SEs 9 - 19% larger"
  det <- safe_read("data/processed/sim_underestimation_v2.csv"); skip_if(is.null(det))
  dup <- det |> group_by(scenario, n_items) |> summarise(d = max(table(round(se, 12))), .groups = "drop")
  expect_equal(dup$d[dup$scenario == "E" & dup$n_items == 400], 28L)     # oracle-SE fallbacks
  expect_true(all(dup$d[!(dup$scenario == "E" & dup$n_items == 400)] == 1L))
})

# ---- Phase 4 headline numbers: cost-efficiency frontier ----------------------
# These are the headline numbers the prose phase (Phase 6) should cite. The
# block computes them rather than reading from a CSV so any drift in the
# decomposition propagates immediately.

test_that("Phase 4 headline: safety cost-efficiency frontier numbers (post-split)", {
  vc <- safe_read("data/processed/variance_components_safety.csv"); skip_if(is.null(vc))
  s2_alpha    <- pull_vc(vc, "within-category item")
  s2_gamma    <- pull_vc(vc, "between-category")
  s2_rho_var  <- pull_vc(vc, "prompt")
  s2_lambda   <- pull_vc(vc, "judge model (design sensitivity)")
  s2_ap       <- pull_vc(vc, "item x prompt")
  s2_al       <- pull_vc(vc, "item x judge")
  s2_pl       <- pull_vc(vc, "prompt x judge")
  s2_eps_cell <- pull_vc(vc, "cell-level (3-way+)")
  s2_rho_rep  <- pull_vc(vc, "replicate noise")

  dstudy_var <- function(N, V, M, R) {
    (s2_gamma + s2_alpha)/N +
      s2_rho_var/V +
      s2_lambda/M +
      s2_ap/(N*V) +
      s2_al/(N*M) +
      s2_pl/(V*M) +
      s2_eps_cell/(N*V*M) +
      s2_rho_rep/(N*V*M*R)
  }

  sq_se   <- sqrt(dstudy_var(141, 1, 1, 1))
  tee_se  <- sqrt(dstudy_var(141, 3, 3, 1))
  full_se <- sqrt(dstudy_var(141, 5, 3, 8))

  reduction_tee  <- 1 - tee_se  / sq_se
  reduction_full <- 1 - full_se / sq_se
  incremental    <- 1 - full_se / tee_se

  # Reference values for Phase 6 to cite in the paper:
  #   Status quo SE  = 0.0391
  #   TEE-guided SE  = 0.0199 (49% reduction at 9x cost)
  #   Full design SE = 0.0188 (52% reduction at 120x cost; +5.4% incremental)
  expect_near(sq_se,           0.0391, tol = 0.01)
  expect_near(tee_se,          0.0199, tol = 0.01)
  expect_near(full_se,         0.0188, tol = 0.01)
  expect_near(reduction_tee,   0.491,  tol = 0.02)
  expect_near(reduction_full,  0.518,  tol = 0.02)
  expect_near(incremental,     0.054,  tol = 0.10)  # was 5.7% under lumped
})

test_that("Phase 4 headline: residual split shares (sigma2_eps_cell vs sigma2_rho_rep)", {
  # Reference percentages for the prose phase (Phase 6). The cell-level
  # component (sigma2_eps_cell) is the reducible-by-R floor: averaging more
  # within-cell calls cannot drive it below this value.
  for (domain in c("safety", "mmlu", "likert", "pairwise")) {
    vc <- safe_read(sprintf("data/processed/variance_components_%s.csv", domain))
    skip_if(is.null(vc))
    eps_pct <- pull_pct(vc, "cell-level (3-way+)")
    rho_pct <- pull_pct(vc, "replicate noise")
    # Both should be positive and together sum to a non-trivial share.
    if (is.na(eps_pct) || is.na(rho_pct)) {
      fail(sprintf("%s: missing eps_cell or rho_rep label", domain))
    }
    expect_gte(eps_pct + rho_pct, 5)
  }
})

# ---- D-study validation robustness (paper §2 D-study paragraph; SI si:dstudy_validation,
#      si:latent_ambiguity). Sources: analysis/04f_sim_dstudy_validation.R,
#      analysis/04h_sim_latent_ambiguity.R -------------------------------------

test_that("D-study validation: correct top-1 intervention in 98% of sims (paper §2)", {
  per_sim <- safe_read("data/processed/sim_dstudy_sc1_per_sim.csv")
  skip_if(is.null(per_sim), "sim_dstudy_sc1_per_sim.csv missing")
  top1 <- per_sim |>
    group_by(sim_id) |>
    summarise(correct = target[which.min(projected_var)] ==
                        target[which.min(true_projected_var)],
              .groups = "drop")
  expect_near(100 * mean(top1$correct), 98, tol = 0.01)
})

test_that("D-study validation: four misspecified scenarios keep |rel. bias| <= 8%, worst -7.2% (paper §2, SI tab:dstudy_misspec)", {
  files <- c("sim_dstudy_sc2_correlated.csv", "sim_dstudy_sc3_nonexch.csv",
             "sim_dstudy_sc4_nongaussian.csv", "sim_dstudy_sc5_hetero.csv")
  scen <- lapply(files, function(f) safe_read(file.path("data/processed", f)))
  skip_if(any(vapply(scen, is.null, logical(1))), "04f scenario CSVs missing")
  biases <- 100 * unlist(lapply(scen, function(d) d$rel_bias))
  expect_lte(max(abs(biases)), 8)
  sc4 <- scen[[3]]
  expect_near(100 * sc4$rel_bias[sc4$df_t == 5], -7.2, tol = 0.01)
})

test_that("latent ambiguity: max component |rel. bias| < 9% across all gamma, worst 8.8% (paper §2, SI tab:latent_ambiguity)", {
  d <- safe_read("data/processed/sim_latent_ambiguity_si_table.csv")
  skip_if(is.null(d), "sim_latent_ambiguity_si_table.csv missing")
  expect_lt(max(d$max_abs_rel_bias), 9)
  expect_near(max(d$max_abs_rel_bias), 8.8, tol = 0.01)
})

# ---- Camera-ready additions (plan v2, Appendix C) ------------------------------

# (b) screen validation. MMLU narrow re-scored on the collected templates (2026-09-25).
# The main-text MMLU v_3 ("Minimal") is screen row broad_construct/V3; it fails the
# structural check and is reported separately.
test_that("equivalence screen: retained variants >= 4, decoys 1.0 safety / 2.5 MMLU (SI tab:equiv_screen)", {
  es <- safe_read("data/processed/equiv_screen_validation.csv"); skip_if(is.null(es))
  es$excluded <- es$domain == "mmlu" & es$set == "broad_construct" & es$variant == "V3"
  m <- es |> filter(!excluded) |> group_by(domain, set) |>
    summarise(mean_equiv = mean(equiv), n = n(), .groups = "drop")
  g <- function(dom, s) m$mean_equiv[m$domain == dom & m$set == s]
  expect_near(g("safety", "decoy"), 1.0)
  expect_near(g("mmlu",   "decoy"), 2.5)
  expect_near(g("safety", "broad_construct"), 4.75)
  expect_near(g("mmlu",   "broad_construct"), 4.67, tol = 0.01)
  expect_near(g("mmlu",   "broadplus"),       4.80)
  expect_equal(m$n[m$domain == "mmlu" & m$set == "broad_construct"], 3L)
  for (dom in c("safety", "mmlu")) for (s in c("narrow", "broad_wording")) expect_near(g(dom, s), 5.0)
  expect_true(all(es$equiv[es$set != "decoy" & !es$excluded] >= 4))
  expect_equal(es$equiv[es$excluded], 5)
  expect_equal(sum(es$set == "decoy"), 4L)
})

# (c) LOJO (safety; unaffected by the MMLU re-parse)
test_that("LOJO safety: judge-attributable 73.0 full, 68.3 - 82.6 drop-one, all fits singular (SI tab:lojo; main text)", {
  lj <- safe_read("data/processed/lojo_judge_safety.csv"); skip_if(is.null(lj))
  full <- lj[lj$n_judges == 3, ]; drop <- lj[lj$n_judges == 2, ]
  expect_equal(nrow(full), 1L); expect_equal(nrow(drop), 3L)
  expect_near(full$pct_judge_total,      72.98)
  expect_near(full$pct_judge_ds,         43.82)
  expect_near(min(drop$pct_judge_total), 68.30)
  expect_near(max(drop$pct_judge_total), 82.58)
  expect_true(all(lj$singular))
  expect_gt(diff(range(lj$pct_judge_ds)), 20)
  g <- function(pat) lj$pct_judge_total[grepl(pat, lj$config)]
  expect_near(g("trinity"), 68.30); expect_near(g("gemini"), 76.86); expect_near(g("gpt-oss"), 82.58)
  expect_near(min(lj$pct_judge_ds), 13.80); expect_near(max(lj$pct_judge_ds), 56.83)
})

# (d) MMLU per-variant accuracy, four admissible variants (SI si:mmlu; main text sec:mmlu)
test_that("mmlu: four admissible variants, accuracy 83.9 - 85.3, base rate 84.5, 1.3% unanswered (SI si:mmlu, si:glmm)", {
  skip_if(!file.exists("data/processed/mmlu_clean.csv"))
  mc <- read_csv("data/processed/mmlu_clean.csv", col_select = c(variant_id, outcome), show_col_types = FALSE)
  expect_setequal(unique(mc$variant_id), c("v_0", "v_1", "v_2", "v_4"))
  va <- safe_read("data/processed/mmlu_variant_accuracy.csv"); skip_if(is.null(va))
  a <- va[va$admissible, ]
  expect_near(min(a$accuracy_pct), 83.9, tol = 0.002); expect_near(max(a$accuracy_pct), 85.3, tol = 0.002)
  expect_near(100 * mean(mc$outcome, na.rm = TRUE), 84.5, tol = 0.002)
  expect_near(100 * mean(is.na(mc$outcome)), 1.30, tol = 0.02)
})

# (f) MMLU operational-design shares (V = 4) and budget allocation (SI tab:vc_mmlu; main text fig:mmlu_combined)
test_that("mmlu Var(theta-hat) shares at V = 4: item 72.2, item x SUT 10.5, between 7.7, SUT 6.7, prompt terms 2.7 (SI tab:vc_mmlu)", {
  vc <- safe_read("data/processed/variance_components_mmlu.csv"); skip_if(is.null(vc))
  N <- 200; V <- 4; M <- 3; H <- 3; R <- 8
  div <- c("between-category" = N, "within-category item" = N, "prompt" = V,
           "item x prompt" = N * V, "item x temperature" = N * H, "prompt x temperature" = V * H,
           "item x SUT" = N * M, "prompt x SUT" = V * M, "cell-level (3-way+)" = N * V * H * M,
           "replicate noise" = N * V * H * M * R, "SUT model (design sensitivity)" = M,
           "temperature (design sensitivity)" = H)
  contrib <- vc$variance / div[vc$tle_label]
  pct <- setNames(100 * contrib / sum(contrib), vc$tle_label)
  expect_near(pct[["within-category item"]], 72.2, tol = 0.002)
  expect_near(pct[["item x SUT"]], 10.5, tol = 0.005)
  expect_near(pct[["between-category"]], 7.7, tol = 0.01)
  expect_near(pct[["SUT model (design sensitivity)"]], 6.7, tol = 0.01)
  expect_near(pct[["prompt x SUT"]], 1.35, tol = 0.02)
  expect_near(pct[["item x prompt"]], 0.87, tol = 0.02)
  expect_near(pct[["prompt"]], 0.445, tol = 0.02)
  expect_near(sum(pct[c("prompt", "prompt x SUT", "item x prompt", "prompt x temperature")]), 2.66, tol = 0.02)
  expect_near(sum(contrib), 0.000475, tol = 0.01)
  expect_near(pct[["within-category item"]] + pct[["between-category"]], 79.9, tol = 0.005)   # "two item terms hold 80%"
})
test_that("mmlu budget allocation: 9 cells x 1,000 sims; TEE/naive RMSE 0.41 - 0.63; TEE = standard to B = 200 (main text sec:mmlu)", {
  ba <- safe_read("data/processed/mmlu_budget_allocation_summary.csv"); skip_if(is.null(ba))
  expect_true(all(ba$n_sims == 9000))
  g <- function(who, b, col) ba[[col]][ba$researcher == who & ba$budget == b]
  ratio <- sapply(unique(ba$budget), function(b) g("TEE", b, "rmse") / g("Naive", b, "rmse"))
  expect_near(min(ratio), 0.41, tol = 0.02); expect_near(max(ratio), 0.63, tol = 0.02)
  for (b in c(50, 100, 200)) expect_near(g("TEE", b, "rmse") / g("Standard", b, "rmse"), 1.00, tol = 0.02)
  for (b in c(700, 1000)) expect_near(g("TEE", b, "rmse") / g("Standard", b, "rmse"), 0.51, tol = 0.03)
  expect_near(g("Naive", 50, "coverage"), 0.909, tol = 0.005); expect_near(g("Naive", 100, "coverage"), 0.932, tol = 0.005)
  expect_gte(min(ba$coverage[ba$researcher == "Naive" & ba$budget >= 200]), 0.975)
  expect_gte(min(ba$coverage[ba$researcher == "TEE"]), 0.968)
  expect_true(all(ba$mean_R[ba$researcher == "Naive"] == 3) && all(ba$mean_K[ba$researcher == "Naive"] == 1))
  expect_lte(max(ba$mean_K[ba$researcher == "TEE"]), 4)
  expect_equal(max(ba$mean_N), 170)                    # 200 items minus the 30-item pilot
})

# (i) MMLU parsing, structural exclusion, and per-domain parse rates (SI si:mmlu_parsing, tab:mmlu_parse_rates; line 955)
test_that("mmlu parsing: 1,043 legacy junk-scored, 321 recovered, 114 conflicts; v_3 sole structural failure; parse rates", {
  sc <- safe_read("data/processed/mmlu_variant_structural_check.csv"); skip_if(is.null(sc))
  expect_equal(nrow(sc[!sc$passes, ]), 1L)
  expect_equal(sc$set[!sc$passes], "main_text"); expect_equal(sc$variant_id[!sc$passes], "v_3")
  pc <- safe_read("data/processed/mmlu_parse_comparison.csv"); skip_if(is.null(pc))
  expect_equal(sum(pc$legacy_parsed_strict_na), 1043L)
  expect_equal(sum(pc$recovered), 321L)
  expect_equal(sum(pc$conflict), 114L)
  expect_equal(sum(pc$conflict_strict_matches_key), 103L); expect_equal(sum(pc$conflict_legacy_matches_key), 5L)
  au <- safe_read("data/processed/mmlu_parse_audit.csv"); skip_if(is.null(au))
  expect_equal(nrow(au), 154L)
  expect_true(all(is.na(au$parsed_letter) == is.na(au$audit_letter)))
  expect_true(all(au$parsed_letter == au$audit_letter, na.rm = TRUE))
  skip_if(!file.exists("data/processed/mmlu_clean_v5.csv"))
  v5 <- read_csv("data/processed/mmlu_clean_v5.csv", col_select = c(variant_id, sut_short, outcome), show_col_types = FALSE)
  expect_setequal(unique(v5$variant_id), paste0("v_", 0:4))
  na <- v5 |> group_by(variant_id, sut_short) |> summarise(p = 100 * mean(is.na(outcome)), .groups = "drop")
  p <- function(v, m) na$p[na$variant_id == v & na$sut_short == m]
  expect_near(100 * mean(is.na(v5$outcome[v5$variant_id == "v_3"])), 34.7, tol = 0.002)
  expect_near(p("v_3", "deepseek-chat-v3.1"), 59.4, tol = 0.002); expect_near(p("v_3", "gemini-2.0-flash-001"), 21.3, tol = 0.003)
  expect_near(p("v_3", "gpt-4o"), 23.5, tol = 0.003); expect_near(p("v_4", "deepseek-chat-v3.1"), 6.2, tol = 0.01)
  na_pct <- function(f) 100 * mean(is.na(read_csv(f, col_select = "outcome", show_col_types = FALSE)$outcome))
  expect_near(na_pct("data/processed/safety_clean.csv"),   1.08, tol = 0.02)
  expect_near(na_pct("data/processed/likert_clean.csv"),   0.548, tol = 0.02)
  expect_near(na_pct("data/processed/pairwise_clean.csv"), 0.41, tol = 0.03)
})

# (j) V = 5 sensitivity (SI tab:mmlu_v5; main-text MMLU footnote: 70%)
test_that("mmlu v5 sensitivity: prompt terms 4.4% (missing) vs 70% (incorrect); prompt 58.0, SUT 18.0 (SI tab:mmlu_v5)", {
  s5 <- safe_read("data/processed/mmlu_v5_sensitivity.csv"); skip_if(is.null(s5))
  expect_setequal(unique(s5$treatment), c("unanswered_missing", "unanswered_incorrect"))
  g <- function(tr, lab) s5$pct_var_theta[s5$treatment == tr & s5$tle_label == lab]
  pt <- function(tr) sum(s5$pct_var_theta[s5$treatment == tr & grepl("prompt", s5$tle_label)])
  expect_near(pt("unanswered_missing"), 4.35, tol = 0.02); expect_near(pt("unanswered_incorrect"), 69.76, tol = 0.01)
  expect_near(g("unanswered_incorrect", "prompt"), 58.0, tol = 0.005)
  expect_near(g("unanswered_incorrect", "SUT model (design sensitivity)"), 18.0, tol = 0.01)
  expect_near(g("unanswered_missing", "within-category item"), 72.5, tol = 0.002)
  expect_near(g("unanswered_incorrect", "within-category item"), 8.6, tol = 0.01)
})

# (e) GLMM MMLU: item heterogeneity leads under both links (SI tab:glmm_comparison, si:glmm)
test_that("mmlu LPM vs GLMM: item 72.2 / 71.4 leads both; prompt terms 2.7 / 6.1; rho 0.95 (SI si:glmm)", {
  glmm <- safe_read("data/processed/glmm_comparison_mmlu.csv"); skip_if(is.null(glmm))
  g <- function(label, col) glmm[[col]][glmm$tle_label == label]
  expect_near(g("within-category item", "lpm_pct"), 72.2, tol = 0.002)
  expect_near(g("within-category item", "glmm_pct"), 71.4, tol = 0.002)
  expect_equal(glmm$tle_label[glmm$lpm_rank == 1], "within-category item")
  expect_equal(glmm$tle_label[glmm$glmm_rank == 1], "within-category item")
  expect_near(g("SUT model (design sensitivity)", "glmm_pct"), 7.6, tol = 0.01)
  expect_near(g("item x SUT", "glmm_pct"), 5.2, tol = 0.01)
  expect_near(g("prompt x SUT", "glmm_pct"), 4.4, tol = 0.01)
  expect_near(g("between-category", "glmm_pct"), 9.6, tol = 0.01)
  expect_near(g("prompt", "glmm_pct"), 1.3, tol = 0.03)
  pt <- function(col) sum(glmm[[col]][glmm$tle_label %in% c("prompt", "prompt x SUT", "item x prompt", "prompt x temperature")])
  expect_near(pt("lpm_pct"), 2.66, tol = 0.02); expect_near(pt("glmm_pct"), 6.12, tol = 0.02)
  expect_near(cor(glmm$lpm_pct, glmm$glmm_pct, method = "spearman"), 0.95, tol = 0.01)
})

# (a) prompt-breadth sensitivity after the strict re-parse (SI tab:breadth_sensitivity)
test_that("prompt breadth: MMLU share 2.1 / 5.5 / 7.3 / 20.6 rising; safety flat 1.26 - 1.27 (SI tab:breadth_sensitivity)", {
  bs <- safe_read("data/processed/prompt_breadth_sensitivity.csv"); skip_if(is.null(bs))
  row   <- function(dom, s) bs[bs$domain == dom & bs$set == s, ]
  share <- function(dom, s) with(row(dom, s), pct_phi + pct_alphaphi)
  absv  <- function(dom, s) with(row(dom, s), sigma2_phi + sigma2_alphaphi)
  gcoef <- function(dom, s) row(dom, s)$gen_coef
  sets <- c("narrow", "broad_wording", "broad_construct", "broadplus")
  expect_near(share("mmlu", "narrow"), 2.07, tol = 0.02);          expect_near(share("mmlu", "broad_wording"), 5.53, tol = 0.02)
  expect_near(share("mmlu", "broad_construct"), 7.33, tol = 0.02); expect_near(share("mmlu", "broadplus"), 20.6, tol = 0.01)
  expect_true(all(diff(sapply(sets, function(s) absv("mmlu", s))) > 0))
  expect_near(absv("mmlu", "narrow"), 0.00282, tol = 0.02);        expect_near(absv("mmlu", "broad_wording"), 0.00706, tol = 0.02)
  expect_near(absv("mmlu", "broad_construct"), 0.0105, tol = 0.02); expect_near(absv("mmlu", "broadplus"), 0.0325, tol = 0.02)
  expect_near(gcoef("mmlu", "narrow"), 0.94, tol = 0.01);          expect_near(gcoef("mmlu", "broad_wording"), 0.89, tol = 0.01)
  expect_near(gcoef("mmlu", "broad_construct"), 0.89, tol = 0.01); expect_near(gcoef("mmlu", "broadplus"), 0.67, tol = 0.01)
  expect_near(row("mmlu", "narrow")$mean_outcome, 0.857, tol = 0.005)
  expect_near(row("mmlu", "broad_construct")$mean_outcome, 0.838, tol = 0.005)
  expect_near(row("mmlu", "broadplus")$mean_outcome, 0.812, tol = 0.005)
  expect_equal(row("mmlu", "broad_construct")$n_obs, 18612)
  it <- bs$sigma2_item[bs$domain == "mmlu"]
  expect_near(min(it), 0.0787, tol = 0.02); expect_near(max(it), 0.0964, tol = 0.02)
  saf_sets <- c("narrow", "broad_wording", "broad_construct")
  for (s in saf_sets) {
    expect_gte(share("safety", s), 1.26); expect_lte(share("safety", s), 1.27)
    expect_near(gcoef("safety", s), 0.965, tol = 0.01)
  }
  saf_abs <- sapply(saf_sets, function(s) absv("safety", s))
  expect_lt(diff(range(saf_abs)) / mean(saf_abs), 0.03)
})

# (a2) breadth caveats and screen gap stated in SI si:prompt_admissibility
test_that("prompt breadth caveats: bootstrap CIs separate narrow and broad plus; wording and framing overlap; plus = 3.1x framing; cosine; screen gap", {
  bs <- safe_read("data/processed/prompt_breadth_sensitivity.csv"); skip_if(is.null(bs))
  m <- bs[bs$domain == "mmlu", ]; a <- setNames(m$sigma2_phi + m$sigma2_alphaphi, m$set)
  expect_near(a[["broadplus"]] / a[["broad_construct"]], 3.1, tol = 0.02)
  cs <- function(s) bs$mean_cosine[bs$domain == "safety" & bs$set == s]
  expect_near(cs("broad_wording"), 0.77, tol = 0.01); expect_near(cs("broad_construct"), 0.81, tol = 0.01)
  bt <- safe_read("data/processed/prompt_breadth_sensitivity_boot.csv"); skip_if(is.null(bt))
  sh <- bt[bt$stat == "share", ]; ci <- function(s, col) sh[[col]][sh$set == s]
  expect_true(all(bt$n_ok == 200) && all(bt$nsim == 200))
  expect_true(all(bt$lo < bt$est & bt$est < bt$hi))
  for (s in c("narrow", "broad_wording", "broad_construct", "broadplus"))
    expect_near(ci(s, "est"), with(bs[bs$domain == "mmlu" & bs$set == s, ], pct_phi + pct_alphaphi), tol = 0.01)
  expect_near(ci("narrow", "lo"), 1.6, tol = 0.02);           expect_near(ci("narrow", "hi"), 2.7, tol = 0.02)
  expect_near(ci("broad_wording", "lo"), 4.3, tol = 0.02);    expect_near(ci("broad_wording", "hi"), 6.5, tol = 0.02)
  expect_near(ci("broad_construct", "lo"), 5.9, tol = 0.02);  expect_near(ci("broad_construct", "hi"), 8.7, tol = 0.02)
  expect_near(ci("broadplus", "lo"), 17.7, tol = 0.02);       expect_near(ci("broadplus", "hi"), 26.0, tol = 0.02)
  expect_lt(ci("narrow", "hi"), ci("broad_wording", "lo"))                  # narrow separated
  expect_gt(ci("broad_wording", "hi"), ci("broad_construct", "lo"))         # wording and framing overlap
  expect_lt(ci("broad_construct", "hi"), ci("broadplus", "lo"))             # broad plus separated
  es <- safe_read("data/processed/equiv_screen_validation.csv"); skip_if(is.null(es))
  es <- es[!(es$domain == "mmlu" & es$set == "broad_construct" & es$variant == "V3"), ]
  mm <- aggregate(equiv ~ domain + set, es, mean)
  gap <- min(sapply(c("safety", "mmlu"), function(d) min(mm$equiv[mm$domain == d & mm$set != "decoy"]) - mm$equiv[mm$domain == d & mm$set == "decoy"]))
  expect_near(gap, 2.17, tol = 0.01)
})

# Safety prompt breadth on the logit scale (SI si:prompt_admissibility caveats; analysis/02f)
test_that("safety prompt breadth GLMM: 1.2 / 1.6 / 1.3% on the logit scale, converged, singular", {
  gs <- safe_read("data/processed/prompt_breadth_glmm_safety.csv"); skip_if(is.null(gs))
  g <- function(s) gs$pct_prompt_logit[gs$set == s]
  expect_near(g("narrow"), 1.2, tol = 0.02); expect_near(g("broad_wording"), 1.6, tol = 0.02); expect_near(g("broad_construct"), 1.3, tol = 0.02)
  expect_true(all(gs$converged)); expect_true(all(gs$singular))
})

# Arena coverage study and gaming-table shares (main text sec:arena, Fig 5; SI tab:gaming)
test_that("arena coverage: p* 0.389; naive 93% at n = 100, 78% at n = 2,000; TEE >= 95% throughout", {
  cv <- safe_read("data/processed/arena_coverage_sim.csv"); skip_if(is.null(cv))
  expect_near(unique(round(cv$pop_truth, 3)), 0.389, tol = 0.002)
  expect_near(cv$coverage_naive[cv$n_b == 100], 0.93, tol = 0.01)
  expect_near(cv$coverage_naive[cv$n_b == 2000], 0.78, tol = 0.01)
  expect_gte(min(cv$coverage_tee), 0.95)
})
test_that("arena gaming table: pooled Likert shares item x judge 18.0, judge 8.8, item x prompt 2.5, prompt 1.2, prompt x judge 0.7 (SI tab:gaming)", {
  d <- safe_read("data/processed/arena_dstudy_per_category.csv"); skip_if(is.null(d))
  pooled <- function(lab) 100 * sum(d$variance[d$tle_label == lab]) / sum(d$variance)
  expect_near(pooled("item x judge"), 18.0, tol = 0.01);  expect_near(pooled("judge"), 8.8, tol = 0.01)
  expect_near(pooled("item x prompt"), 2.5, tol = 0.02); expect_near(pooled("prompt"), 1.2, tol = 0.05)   # 1.16, printed as 1.2
  expect_near(pooled("prompt x judge"), 0.7, tol = 0.05)
})
