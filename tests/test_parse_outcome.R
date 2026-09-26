# test_parse_outcome.R
# Unit tests for Likert and Pairwise response parsers.
# Run: Rscript tests/test_parse_outcome.R

library(stringr)

# --- Parsers (copied from 01_clean_and_merge.R for isolated testing) ---
parse_likert <- function(response) {
  if (is.na(response) || is.null(response)) return(NA_real_)
  resp <- str_trim(response)
  nums <- str_extract(resp, "\\d+\\.?\\d*")
  if (!is.na(nums)) {
    val <- as.numeric(nums)
    if (val >= 1 && val <= 5) return(val)
    return(NA_real_)
  }
  return(NA_real_)
}

parse_pairwise <- function(response) {
  if (is.na(response) || is.null(response)) return(NA_real_)
  resp <- str_to_upper(str_trim(response))
  if (str_detect(resp, "^A\\b")) return(1)
  if (str_detect(resp, "^B\\b")) return(0)
  if (str_detect(resp, "RESPONSE\\s*A")) return(1)
  if (str_detect(resp, "RESPONSE\\s*B")) return(0)
  return(NA_real_)
}

# --- Test harness ---
n_pass <- 0
n_fail <- 0

assert_eq <- function(actual, expected, label) {
  if (is.na(expected) && is.na(actual)) {
    n_pass <<- n_pass + 1
    return()
  }
  if (is.na(actual) || is.na(expected) || actual != expected) {
    cat("FAIL:", label, "— expected", expected, "got", actual, "\n")
    n_fail <<- n_fail + 1
  } else {
    n_pass <<- n_pass + 1
  }
}

# --- Likert tests ---
cat("=== Likert parser tests ===\n")
assert_eq(parse_likert("3"),    3,    "bare number 3")
assert_eq(parse_likert("1"),    1,    "bare number 1")
assert_eq(parse_likert("5"),    5,    "bare number 5")
assert_eq(parse_likert("3."),   3,    "number with trailing dot")
assert_eq(parse_likert("3.5"),  3.5,  "decimal 3.5")
assert_eq(parse_likert("  4 "), 4,    "whitespace padded")
assert_eq(parse_likert("I rate this 4"), 4, "embedded number")
assert_eq(parse_likert("The score is 2 out of 5"), 2, "sentence with number")
assert_eq(parse_likert("0"),    NA_real_, "out of range: 0")
assert_eq(parse_likert("6"),    NA_real_, "out of range: 6")
assert_eq(parse_likert("10"),   NA_real_, "out of range: 10")
assert_eq(parse_likert(""),     NA_real_, "empty string")
assert_eq(parse_likert(NA),     NA_real_, "NA input")
assert_eq(parse_likert("moderate"), NA_real_, "non-numeric text")

# --- Pairwise tests ---
cat("=== Pairwise parser tests ===\n")
assert_eq(parse_pairwise("A"),   1,    "bare A")
assert_eq(parse_pairwise("B"),   0,    "bare B")
assert_eq(parse_pairwise("a"),   1,    "lowercase a")
assert_eq(parse_pairwise("b"),   0,    "lowercase b")
assert_eq(parse_pairwise("A."),  1,    "A with period")
assert_eq(parse_pairwise("B "),  0,    "B with space")
assert_eq(parse_pairwise("Response A"),      1, "Response A")
assert_eq(parse_pairwise("Response B is more conservative"), 0, "Response B in sentence")
assert_eq(parse_pairwise(""),    NA_real_, "empty string")
assert_eq(parse_pairwise(NA),    NA_real_, "NA input")
assert_eq(parse_pairwise("Neither"), NA_real_, "neither response")

# --- Summary ---
cat(sprintf("\n%d passed, %d failed\n", n_pass, n_fail))
if (n_fail > 0) quit(status = 1)
cat("All tests passed.\n")
