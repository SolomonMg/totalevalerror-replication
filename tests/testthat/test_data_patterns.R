# test_data_patterns.R
# Statistical-sanity tests: does the data look like it should?
#
# Three families:
#   (1) Range/bound checks on outcome columns
#   (2) Design-balance checks (full factorial with no holes)
#   (3) Distribution sanity (means, spreads, dominant components)
#   (4) Cross-domain relationship invariants

suppressPackageStartupMessages({
  library(testthat)
  library(tidyverse)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

# ============================================================================
# (1) Range/bound checks on outcome columns
# ============================================================================

test_that("Likert outcomes lie in [1, 5]", {
  skip_if_not(file.exists("data/processed/likert_clean.csv"))
  df <- read_csv("data/processed/likert_clean.csv", show_col_types = FALSE)
  vals <- df$outcome[!is.na(df$outcome)]
  if (any(vals < 1 | vals > 5)) {
    bad <- vals[vals < 1 | vals > 5]
    fail(sprintf("Likert outcome out of [1,5]: %d violations, e.g. %s",
                  length(bad), paste(head(unique(bad), 5), collapse = ", ")))
  }
  expect_true(all(vals >= 1 & vals <= 5))
})

test_that("Pairwise recoded outcomes are in {0, 1}", {
  skip_if_not(file.exists("data/processed/pairwise_clean.csv"))
  df <- read_csv("data/processed/pairwise_clean.csv", show_col_types = FALSE)
  if ("model_a_wins" %in% names(df)) {
    vals <- df$model_a_wins[!is.na(df$model_a_wins)]
    if (!all(vals %in% c(0, 1))) {
      fail("pairwise model_a_wins has values outside {0, 1}")
    }
    expect_true(all(vals %in% c(0, 1)))
  }
})

test_that("Safety outcomes are in {0, 1}", {
  skip_if_not(file.exists("data/processed/safety_clean.csv"))
  df <- read_csv("data/processed/safety_clean.csv", show_col_types = FALSE)
  vals <- df$outcome[!is.na(df$outcome)]
  if (!all(vals %in% c(0, 1))) {
    fail("safety outcome has values outside {0, 1}")
  }
  expect_true(all(vals %in% c(0, 1)))
})

test_that("temperatures are in {0, 0.7, 1.0}", {
  for (f in c("data/processed/likert_clean.csv",
              "data/processed/pairwise_clean.csv",
              "data/processed/safety_clean.csv",
              "data/processed/mmlu_clean.csv")) {
    if (!file.exists(f)) next
    df <- read_csv(f, show_col_types = FALSE)
    temps <- sort(unique(as.numeric(df$temperature)))
    expected <- c(0, 0.7, 1.0)
    if (!isTRUE(all.equal(temps, expected))) {
      fail(sprintf("%s has temperatures %s (expected %s)",
                    basename(f), paste(temps, collapse = ", "),
                    paste(expected, collapse = ", ")))
    }
    expect_equal(temps, expected)
  }
})

# ============================================================================
# (2) Design-balance checks
# ============================================================================

test_that("ideology CoT pipelines have balanced full factorial (150 x 5 x 3 x 3 x 3)", {
  for (f in c("data/processed/likert_clean.csv",
              "data/processed/pairwise_clean.csv")) {
    skip_if_not(file.exists(f))
    df <- read_csv(f, show_col_types = FALSE)
    n_cells <- df %>%
      count(item_id, variant_id, judge_model, temperature) %>%
      pull(n)
    if (!all(n_cells == 3)) {
      tab <- table(n_cells)
      fail(sprintf("%s: cells do not all have 3 reps. Counts: %s",
                    basename(f), paste(names(tab), "=", tab, collapse = ", ")))
    }
    expect_true(all(n_cells == 3))
    # Expected total cells: 150 * 5 * 3 * 3 = 6750
    expect_equal(length(n_cells), 150 * 5 * 3 * 3)
  }
})

test_that("no duplicate (item, variant, judge, temp, rep) combinations", {
  for (f in c("data/processed/likert_clean.csv",
              "data/processed/pairwise_clean.csv",
              "data/processed/safety_clean.csv",
              "data/processed/mmlu_clean.csv")) {
    if (!file.exists(f)) next
    df <- read_csv(f, show_col_types = FALSE)
    facet <- if ("judge_model" %in% names(df)) "judge_model" else "sut_model"
    if (!(facet %in% names(df))) next
    combo_n <- df %>%
      count(item_id, variant_id, !!sym(facet), temperature, replication) %>%
      pull(n)
    if (any(combo_n > 1)) {
      fail(sprintf("%s has %d duplicate cells",
                    basename(f), sum(combo_n > 1)))
    }
    expect_true(all(combo_n == 1))
  }
})

# ============================================================================
# (3) Distribution sanity
# ============================================================================

test_that("Likert outcome mean is plausibly ideological, not stuck at an extreme", {
  skip_if_not(file.exists("data/processed/likert_clean.csv"))
  df <- read_csv("data/processed/likert_clean.csv", show_col_types = FALSE)
  m <- mean(df$outcome, na.rm = TRUE)
  # On a 1-5 scale rating conservatism of ideologically-varied responses,
  # a mean in [2, 4] is plausible. Outside that, the data or judges have an issue.
  if (m < 2 || m > 4) {
    fail(sprintf("Likert mean = %.2f (expected in [2, 4])", m))
  }
  expect_true(m >= 2 && m <= 4)
})

test_that("pairwise recoded per-judge means are within 0.15 of each other (low position bias)", {
  # Raw outcome means can diverge (position bias); recoded means should NOT.
  # This test catches the Haiku-91%-B bug directly.
  skip_if_not(file.exists("data/processed/pairwise_clean.csv"))
  df <- read_csv("data/processed/pairwise_clean.csv", show_col_types = FALSE)
  if (!"model_a_wins" %in% names(df)) skip("model_a_wins column not present")
  means <- df %>%
    group_by(judge_model) %>%
    summarize(m = mean(model_a_wins, na.rm = TRUE), .groups = "drop")
  spread <- diff(range(means$m))
  if (spread > 0.15) {
    fail(sprintf("pairwise recoded per-judge means span %.3f (> 0.15). Means: %s",
                  spread,
                  paste(sprintf("%s=%.2f", means$judge_model, means$m),
                        collapse = "; ")))
  }
  expect_lte(spread, 0.15)
})

test_that("safety safe-rate is close to 0.94 (AILuminate benchmark property)", {
  skip_if_not(file.exists("data/processed/safety_clean.csv"))
  df <- read_csv("data/processed/safety_clean.csv", show_col_types = FALSE)
  sr <- mean(df$outcome, na.rm = TRUE)
  if (sr < 0.80 || sr > 0.99) {
    fail(sprintf("safety safe rate = %.3f (expected ~0.94, tolerance [0.80, 0.99])", sr))
  }
  expect_true(sr >= 0.80 && sr <= 0.99)
})

test_that("MMLU overall accuracy is in a plausible range for frontier models", {
  skip_if_not(file.exists("data/processed/mmlu_clean.csv"))
  df <- read_csv("data/processed/mmlu_clean.csv", show_col_types = FALSE)
  acc <- mean(df$outcome, na.rm = TRUE)
  if (acc < 0.60 || acc > 0.95) {
    fail(sprintf("MMLU accuracy = %.3f (expected in [0.60, 0.95] for frontier SUTs)", acc))
  }
  expect_true(acc >= 0.60 && acc <= 0.95)
})

# ============================================================================
# (4) Variance profile expectations (domain-specific)
# ============================================================================

test_that("Likert CoT: within-category item is the dominant component", {
  skip_if_not(file.exists("data/processed/variance_components_likert.csv"))
  vc <- read_csv("data/processed/variance_components_likert.csv", show_col_types = FALSE)
  top <- vc %>% arrange(desc(pct_total)) %>% slice(1) %>% pull(tle_label)
  if (top != "within-category item") {
    fail(sprintf("Likert dominant component is '%s' (expected 'within-category item')", top))
  }
  expect_equal(top, "within-category item")
})

test_that("Pairwise CoT: replicate noise is the dominant component", {
  # Phase 3 split: lumped "generation" became "replicate noise" (within-cell)
  # plus "cell-level (3-way+)". Replicate noise still dominates pairwise CoT.
  skip_if_not(file.exists("data/processed/variance_components_pairwise.csv"))
  vc <- read_csv("data/processed/variance_components_pairwise.csv", show_col_types = FALSE)
  top <- vc %>% arrange(desc(pct_total)) %>% slice(1) %>% pull(tle_label)
  if (top != "replicate noise") {
    fail(sprintf("Pairwise dominant component is '%s' (expected 'replicate noise' for CoT pipeline)", top))
  }
  expect_equal(top, "replicate noise")
})

test_that("Safety: item x judge is the dominant component", {
  skip_if_not(file.exists("data/processed/variance_components_safety.csv"))
  vc <- read_csv("data/processed/variance_components_safety.csv", show_col_types = FALSE)
  top <- vc %>% arrange(desc(pct_total)) %>% slice(1) %>% pull(tle_label)
  if (top != "item x judge") {
    fail(sprintf("Safety dominant component is '%s' (expected 'item x judge')", top))
  }
  expect_equal(top, "item x judge")
})

test_that("MMLU: within-category item dominates; prompt is negligible", {
  skip_if_not(file.exists("data/processed/variance_components_mmlu.csv"))
  vc <- read_csv("data/processed/variance_components_mmlu.csv", show_col_types = FALSE)
  top <- vc %>% arrange(desc(pct_total)) %>% slice(1) %>% pull(tle_label)
  if (top != "within-category item") {
    fail(sprintf("MMLU dominant component is '%s' (expected 'within-category item')", top))
  }
  prompt_pct <- vc$pct_total[vc$tle_label == "prompt"]
  if (length(prompt_pct) == 1 && prompt_pct > 5) {
    fail(sprintf("MMLU prompt share = %.2f%% (expected < 5%% for verifiable-answer benchmark)", prompt_pct))
  }
  expect_equal(top, "within-category item")
})

# ============================================================================
# (5) D-study internal consistency
# ============================================================================

test_that("D-study baseline total_var equals sum of items from VC decomposition (approximately)", {
  # The "Baseline (avg temp & judge)" total_var should be reconstructible
  # from the variance components using the standard D-study formula.
  # We don't reproduce the full formula here; just sanity-check that the
  # baseline total_var is smaller than the largest VC (since D-study averages).
  pairs <- list(
    list(domain = "likert", N = 150, V = 5, H = 3, M = 3, R = 3),
    list(domain = "pairwise", N = 150, V = 5, H = 3, M = 3, R = 3),
    list(domain = "safety", N = 141, V = 5, H = 3, M = 3, R = 8),
    list(domain = "mmlu", N = 200, V = 5, H = 3, M = 3, R = 8)
  )
  for (p in pairs) {
    vc_path <- sprintf("data/processed/variance_components_%s.csv", p$domain)
    ds_path <- sprintf("data/processed/dstudy_%s.csv", p$domain)
    if (!file.exists(vc_path) || !file.exists(ds_path)) next
    vc <- read_csv(vc_path, show_col_types = FALSE)
    ds <- read_csv(ds_path, show_col_types = FALSE)
    baseline <- ds$total_var[str_detect(ds$scenario, "Baseline.*avg")]
    if (length(baseline) != 1) next
    # Baseline variance should be positive and smaller than any single variance component
    # (since D-study divides every component by at least N).
    if (baseline <= 0) {
      fail(sprintf("%s: baseline D-study variance is non-positive (%.5f)",
                    p$domain, baseline))
    }
    max_vc <- max(vc$variance)
    if (baseline > max_vc) {
      fail(sprintf("%s: baseline D-study variance %.5f exceeds largest VC %.5f (D-study should reduce)",
                    p$domain, baseline, max_vc))
    }
    expect_true(baseline > 0)
    expect_lt(baseline, max_vc)
  }
})

test_that("+5 reps reduction is smaller than Double items reduction", {
  # Replications are cheap-but-weak (only help residual). Items are the dominant lever.
  # Any D-study where +5 reps saves MORE than doubling items is broken.
  DS_FILES <- list.files("data/processed", "^dstudy_[^_]*\\.csv$", full.names = TRUE)
  DS_FILES <- DS_FILES[!str_detect(DS_FILES, "deprecated|_cot\\.csv")]
  for (f in DS_FILES) {
    ds <- read_csv(f, show_col_types = FALSE)
    if (!all(c("Double items", "+5 replications") %in% ds$scenario)) next
    r_items <- ds$reduction_pct[ds$scenario == "Double items"]
    r_reps  <- ds$reduction_pct[ds$scenario == "+5 replications"]
    if (r_reps > r_items) {
      fail(sprintf("%s: +5 reps (%.2f%%) beats Double items (%.2f%%) -- broken D-study",
                    basename(f), r_reps, r_items))
    }
    expect_gt(r_items, r_reps)
  }
})
