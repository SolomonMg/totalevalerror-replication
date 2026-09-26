# test_invariants.R
# Numerical invariants on D-study projections and decomposition arithmetic.

suppressPackageStartupMessages({
  library(testthat)
  library(tidyverse)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

DS_FILES <- list.files("data/processed", "^dstudy_[^_]*\\.csv$", full.names = TRUE)
DS_FILES <- DS_FILES[!str_detect(DS_FILES, "deprecated|_cot\\.csv")]

test_that("adding items always reduces variance", {
  for (f in DS_FILES) {
    ds <- read_csv(f, show_col_types = FALSE)
    if (!"Double items" %in% ds$scenario) next
    r <- ds$reduction_pct[ds$scenario == "Double items"]
    if (r <= 0) fail(sprintf("%s: Double items reduction = %.2f", basename(f), r))
    expect_gt(r, 0)
  }
})

test_that("adding prompt variants reduces variance (unless prompt-side is zero)", {
  for (f in DS_FILES) {
    ds <- read_csv(f, show_col_types = FALSE)
    if (!"+2 prompt variants" %in% ds$scenario) next
    r <- ds$reduction_pct[ds$scenario == "+2 prompt variants"]
    if (r < -0.01) fail(sprintf("%s: +2 prompt variants reduction = %.2f", basename(f), r))
    expect_gte(r, -0.01)
  }
})

test_that("fixing judge model never reduces variance", {
  for (f in DS_FILES) {
    ds <- read_csv(f, show_col_types = FALSE)
    if (!"Fix judge model" %in% ds$scenario) next
    r <- ds$reduction_pct[ds$scenario == "Fix judge model"]
    if (r > 0.01) fail(sprintf("%s: Fix judge model reduction = %.2f (should be <= 0)", basename(f), r))
    expect_lte(r, 0.01)
  }
})

test_that("fixing temperature never reduces variance", {
  for (f in DS_FILES) {
    ds <- read_csv(f, show_col_types = FALSE)
    if (!"Fix temperature" %in% ds$scenario) next
    r <- ds$reduction_pct[ds$scenario == "Fix temperature"]
    if (r > 0.01) fail(sprintf("%s: Fix temperature reduction = %.2f (should be <= 0)", basename(f), r))
    expect_lte(r, 0.01)
  }
})

test_that("+5 replications reduces variance (even if only slightly)", {
  for (f in DS_FILES) {
    ds <- read_csv(f, show_col_types = FALSE)
    if (!"+5 replications" %in% ds$scenario) next
    r <- ds$reduction_pct[ds$scenario == "+5 replications"]
    if (r < -0.01) fail(sprintf("%s: +5 replications reduction = %.2f", basename(f), r))
    expect_gte(r, -0.01)
  }
})

test_that("baseline reduction is exactly 0", {
  for (f in DS_FILES) {
    ds <- read_csv(f, show_col_types = FALSE)
    baseline_row <- ds[str_detect(ds$scenario, "Baseline.*avg"), ]
    if (nrow(baseline_row) == 0) next
    if (abs(baseline_row$reduction_pct) > 0.1) {
      fail(sprintf("%s: baseline reduction = %.2f (should be 0)", basename(f), baseline_row$reduction_pct))
    }
    expect_equal(baseline_row$reduction_pct, 0, tolerance = 0.1)
  }
})

test_that("pct_total matches variance / sum(variance) to 2 decimal places", {
  VC_FILES <- list.files("data/processed", "^variance_components_[^_]*\\.csv$",
                          full.names = TRUE)
  VC_FILES <- VC_FILES[!str_detect(VC_FILES, "deprecated|_recoded|_cot\\.csv")]
  for (f in VC_FILES) {
    vc <- read_csv(f, show_col_types = FALSE)
    total <- sum(vc$variance)
    expected <- round(100 * vc$variance / total, 2)
    diffs <- abs(expected - vc$pct_total)
    if (any(diffs >= 0.05)) {
      fail(sprintf("%s: pct_total inconsistent with variance (max diff %.3f)",
                    basename(f), max(diffs)))
    }
    expect_true(all(diffs < 0.05))
  }
})
