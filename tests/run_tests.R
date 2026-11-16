#!/usr/bin/env Rscript
# run_tests.R
# Entry point for the testthat suite. Runs all tests in tests/testthat/ and
# exits with a non-zero code on any failure.
#
# Usage: Rscript tests/run_tests.R [--strict]
#   --strict also fails on skipped tests (a missing CSV skips silently otherwise).

strict <- "--strict" %in% commandArgs(trailingOnly = TRUE)

suppressPackageStartupMessages({
  library(testthat)
  library(rprojroot)
})

root <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(root)

cat("=== Running tests from", file.path(root, "tests/testthat"), "===\n\n")

results <- test_dir("tests/testthat",
                     reporter = SummaryReporter$new(),
                     stop_on_failure = FALSE)
# results is a `testthat_results` object; coerce to df to tally
df <- as.data.frame(results)
failed <- sum(df$failed, na.rm = TRUE)
errors <- sum(df$error,  na.rm = TRUE)
skipped <- sum(df$skipped, na.rm = TRUE)
if (failed + errors > 0) {
  cat(sprintf("\nTESTS FAILED: %d failures, %d errors\n", failed, errors))
  quit(status = 1)
} else if (strict && skipped > 0) {
  cat(sprintf("\nSTRICT: %d skipped tests\n", skipped))
  print(df[df$skipped, c("file", "test")])
  quit(status = 1)
} else {
  cat("\nAll tests passed.\n")
  quit(status = 0)
}
