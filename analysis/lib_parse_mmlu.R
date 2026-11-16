# lib_parse_mmlu.R
# Strict MMLU answer parser shared by 01_clean_mmlu.R, 01b_reparse_mmlu_recollections.R,
# and 01c_mmlu_parse_audit.R (SI si:mmlu_parsing).
#
# Returns list(letter = "A".."D" or NA, rule = "R1", "R2", "R3", "R3b", "R3c", "R4", or NA). Never guesses from
# prose: no first-character rule and no bare-capital rule (the legacy parser scored
# "Based on ..." as B and the article "a" as A).
# Keep experiments/run_mmlu.py and rebuttal_neurips2026/recollect_driver.py in sync
# with rules R1 - R3.
suppressPackageStartupMessages(library(stringr))

.norm_opt <- function(x) str_squish(str_to_lower(str_replace_all(x, "[^[:alnum:] ]", " ")))

parse_mmlu_strict <- function(r, A, B, C, D) {
  none <- list(letter = NA_character_, rule = NA_character_)
  if (is.na(r)) return(none)
  # U+0120 / U+010A are byte-level BPE space / newline markers leaked by some providers (DeepSeek)
  t <- r |> str_replace_all("\u0120", " ") |> str_replace_all("\u010a", "\n") |>
    str_remove_all("\\*\\*|__|`|#+ ") |> str_trim()
  hit <- function(m, rule) list(letter = str_to_upper(na.omit(m[1, -1])[1]), rule = rule)
  # R1: the whole response is one letter, optionally bracketed or punctuated
  m <- str_match(t, "^[\\(\\[]?([A-Da-d])[\\)\\]]?[\\.:,]?$")
  if (!all(is.na(m[1, -1]))) return(hit(m, "R1"))
  # R2: letter plus delimiter (or line break) at the very start: "C. text", "c) text", "C\n..."
  m <- str_match(t, "^[\\(\\[]?(?:([A-D])(?:[\\)\\]\\.:,]\\s|[\\)\\]]?\\s*\\n)|([a-d])[\\)\\]\\.]\\s)")
  if (!all(is.na(m[1, -1]))) return(hit(m, "R2"))
  # R3: explicit answer statement: "answer is C", "Answer: C.", "correct option is (b)"
  m <- str_match(t, paste0("(?i:\\banswer|\\bcorrect (?:option|choice|letter))\\s*(?i:is|would be|should be)?",
                           "\\s*:?\\s*[\\(\\[]?(?:([A-D])(?![A-Za-z])|([a-d])[\\)\\.])"))
  if (!all(is.na(m[1, -1]))) return(hit(m, "R3"))
  # R3b: a boxed letter: "\boxed{C}", "\boxed{\text{C}}"
  m <- str_match(t, "\\\\boxed\\{\\s*(?:\\\\text\\{)?\\s*\\(?([A-Da-d])\\)?\\s*\\}")
  if (!all(is.na(m[1, -1]))) return(hit(m, "R3b"))
  # R3c: the response ends by naming a letter: "... is D", "... statement is (B)."
  m <- str_match(t, "\\bis:?\\s*\\(?([A-D])\\)?\\.?\\s*$")
  if (!all(is.na(m[1, -1]))) return(hit(m, "R3c"))
  # R4: the response names exactly one option's full text and is not walking through all options
  nt <- .norm_opt(t)
  opts <- c(A = .norm_opt(A), B = .norm_opt(B), C = .norm_opt(C), D = .norm_opt(D))
  found <- names(opts)[nchar(opts) >= 3 & vapply(opts, function(o) grepl(o, nt, fixed = TRUE), logical(1))]
  walk <- str_detect(t, "(?i)each option|let.s (go through|check|evaluate|analy[sz]e) (the|each|all)")
  if (length(found) == 1 && !walk) return(list(letter = found, rule = "R4"))
  none
}

# Vectorized wrapper: returns a tibble with columns letter, rule.
parse_mmlu_strict_df <- function(response, A, B, C, D) {
  res <- purrr::pmap(list(response, A, B, C, D), parse_mmlu_strict)
  tibble::tibble(letter = purrr::map_chr(res, "letter"), rule = purrr::map_chr(res, "rule"))
}
