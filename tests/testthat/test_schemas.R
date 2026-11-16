# test_schemas.R
# Schema-integrity tests for data/processed CSVs. Catches label drift,
# negative variances, pct columns that don't sum to 100, and missing
# required components.
#
# Run: Rscript tests/run_tests.R

suppressPackageStartupMessages({
  library(testthat)
  library(tidyverse)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

# Canonical variance-component labels (as written by 02/23/25 decomposition scripts).
# Phase 3 split the lumped residual into two distinct components:
#   "cell-level (3-way+)" — sigma2_eps_cell, NOT reducible by R
#   "replicate noise"     — sigma2_rho_rep, reducible by R via averaging
# Older CoT-only artifacts still write "generation" (lumped); kept here for compatibility.
CANONICAL_VC_LABELS <- c(
  "within-category item",
  "between-category",
  "generation",            # pre-Phase 3 lumped residual (legacy CoT artifacts)
  "replicate noise",       # Phase 3: within-cell replicate noise (reducible by R)
  "cell-level (3-way+)",   # Phase 3: cell-level residual (NOT reducible by R)
  "prompt",
  "item x prompt",
  "item x temperature",
  "prompt x temperature",
  "item x judge", "item x SUT",
  "prompt x judge", "prompt x SUT",
  "judge model (design sensitivity)",
  "SUT model (design sensitivity)",
  "temperature (design sensitivity)",
  "position bias (design sensitivity)",  # safety_pairwise
  "position (design sensitivity)"         # pairwise (CoT-pipeline shorthand)
)

VC_FILES <- list.files("data/processed", "^variance_components_.*\\.csv$",
                        full.names = TRUE)
VC_FILES <- VC_FILES[!str_detect(VC_FILES, "deprecated|_recoded|_cot\\.csv")]

test_that("variance_components files exist for all demonstrations", {
  expected <- c("likert", "pairwise", "safety", "mmlu")
  found <- str_extract(basename(VC_FILES), "(?<=variance_components_).*(?=\\.csv)")
  for (e in expected) {
    expect_true(e %in% found)
  }
})

test_that("each VC file has required columns", {
  required_cols <- c("component", "variance", "tle_label", "pct_total", "tier", "scoring")
  for (f in VC_FILES) {
    vc <- read_csv(f, show_col_types = FALSE)
    missing <- setdiff(required_cols, names(vc))
    if (length(missing) > 0) {
      fail(sprintf("%s missing columns: %s",
                    basename(f), paste(missing, collapse = ", ")))
    }
    expect_length(missing, 0)
  }
})

test_that("VC files use canonical tle_labels (no typos / label drift)", {
  for (f in VC_FILES) {
    vc <- read_csv(f, show_col_types = FALSE)
    unknown <- setdiff(vc$tle_label, CANONICAL_VC_LABELS)
    if (length(unknown) > 0) {
      fail(sprintf("%s has unknown labels: %s",
                    basename(f), paste(unknown, collapse = ", ")))
    }
    expect_length(unknown, 0)
  }
})

test_that("variances are non-negative", {
  for (f in VC_FILES) {
    vc <- read_csv(f, show_col_types = FALSE)
    if (any(vc$variance < 0)) {
      fail(sprintf("%s has negative variance components", basename(f)))
    }
    expect_true(all(vc$variance >= 0))
  }
})

test_that("pct_total sums to ~100", {
  for (f in VC_FILES) {
    vc <- read_csv(f, show_col_types = FALSE)
    s <- sum(vc$pct_total)
    if (abs(s - 100) > 2) {
      fail(sprintf("%s: pct_total sums to %.2f", basename(f), s))
    }
    expect_equal(s, 100, tolerance = 2)
  }
})

test_that("dominant component has a plausible share (>= 5%)", {
  for (f in VC_FILES) {
    vc <- read_csv(f, show_col_types = FALSE)
    m <- max(vc$pct_total)
    if (m < 5) {
      fail(sprintf("%s: max component share is only %.2f%%", basename(f), m))
    }
    expect_true(m >= 5)
  }
})

test_that("cleaned ideology data has matched-protocol design", {
  skip_if_not(file.exists("data/processed/pairwise_clean.csv"))
  skip_if_not(file.exists("data/processed/likert_clean.csv"))

  for (f in c("data/processed/pairwise_clean.csv",
              "data/processed/likert_clean.csv")) {
    df <- read_csv(f, show_col_types = FALSE)
    expect_equal(n_distinct(df$item_id), 150)
    expect_equal(n_distinct(df$variant_id), 5)
    expect_equal(n_distinct(df$temperature), 3)
    expect_equal(n_distinct(df$judge_model), 3)
    expect_true(n_distinct(df$replication) >= 3)
  }
})

test_that("ideology pipelines use the matched CoT judge panel", {
  skip_if_not(file.exists("data/processed/likert_clean.csv"))
  expected_judges <- c("deepseek/deepseek-chat-v3.1",
                        "google/gemini-2.0-flash-001",
                        "openai/gpt-oss-120b")
  for (f in c("data/processed/pairwise_clean.csv",
              "data/processed/likert_clean.csv")) {
    df <- read_csv(f, show_col_types = FALSE)
    actual <- sort(unique(df$judge_model))
    expect_equal(actual, expected_judges)
  }
})

test_that("parse rates meet per-dataset floors", {
  # Floors reflect empirical worst-case per domain. MMLU multi-choice
  # parsing is noisier than JSON-output judges, so 90% is the floor.
  checks <- list(
    list(path = "data/processed/likert_clean.csv",   col = "outcome",      floor = 0.95),
    list(path = "data/processed/pairwise_clean.csv", col = "model_a_wins", floor = 0.95),
    list(path = "data/processed/safety_clean.csv",   col = "outcome",      floor = 0.95),
    list(path = "data/processed/mmlu_clean.csv",     col = "outcome",      floor = 0.90)
  )
  for (c in checks) {
    if (!file.exists(c$path)) next
    df <- read_csv(c$path, show_col_types = FALSE)
    if (!(c$col %in% names(df))) next
    rate <- mean(!is.na(df[[c$col]]))
    if (rate < c$floor) {
      fail(sprintf("%s: %s parse rate is %.2f%% (floor %.0f%%)",
                    basename(c$path), c$col, 100 * rate, 100 * c$floor))
    }
    expect_gte(rate, c$floor)
  }
})
