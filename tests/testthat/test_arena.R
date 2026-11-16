# test_arena.R
# Schema checks on the Arena scoring demonstration outputs.

suppressPackageStartupMessages({
  library(testthat)
  library(tidyverse)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

TARGET_CATEGORIES <- c("creative_writing", "persuasion", "coding", "factual_qa")

test_that("arena_likert_clean.csv has expected schema + parse rate", {
  skip_if_not(file.exists("data/processed/arena_likert_clean.csv"))
  df <- read_csv("data/processed/arena_likert_clean.csv", show_col_types = FALSE)
  required <- c("battle_id", "category_llm", "winner", "mode",
                "response_side", "variant_id", "judge_model", "score")
  expect_true(all(required %in% names(df)))
  expect_true(all(df$score[!is.na(df$score)] >= 1 & df$score[!is.na(df$score)] <= 5))
  rate <- mean(!is.na(df$score))
  if (rate < 0.95) {
    fail(sprintf("Likert parse rate %.1f%% below 95%% floor", 100 * rate))
  }
  expect_gte(rate, 0.95)
})

test_that("arena_pairwise_clean.csv has expected schema + parse rate", {
  skip_if_not(file.exists("data/processed/arena_pairwise_clean.csv"))
  df <- read_csv("data/processed/arena_pairwise_clean.csv", show_col_types = FALSE)
  required <- c("battle_id", "category_llm", "winner", "mode",
                "order", "variant_id", "judge_model", "judgment", "model_a_wins")
  expect_true(all(required %in% names(df)))
  vals <- df$model_a_wins[!is.na(df$model_a_wins)]
  expect_true(all(vals %in% c(0, 0.5, 1)))                 # a tie counts as half a win
  expect_true(all(df$model_a_wins[which(df$judgment == "TIE")] == 0.5))
  rate <- mean(!is.na(df$judgment))
  if (rate < 0.95) fail(sprintf("Pairwise parse rate %.1f%% below 95%% floor", 100 * rate))
  expect_gte(rate, 0.95)
})

test_that("target categories are all represented with enough battles", {
  skip_if_not(file.exists("data/processed/arena_likert_clean.csv"))
  df <- read_csv("data/processed/arena_likert_clean.csv", show_col_types = FALSE)
  n_battles_per_cat <- df %>%
    filter(category_llm %in% TARGET_CATEGORIES) %>%
    distinct(battle_id, category_llm) %>% count(category_llm)
  for (cat in TARGET_CATEGORIES) {
    n <- n_battles_per_cat$n[n_battles_per_cat$category_llm == cat]
    if (length(n) == 0 || n < 100) {
      fail(sprintf("Category %s has %d battles (need >= 100 for stable estimates)",
                    cat, if (length(n) == 0) 0 else n))
    }
    expect_gte(n, 100)
  }
})

test_that("arena agreement summary has an entry for each (config x category)", {
  skip_if_not(file.exists("data/processed/arena_agreement_summary.csv"))
  s <- read_csv("data/processed/arena_agreement_summary.csv", show_col_types = FALSE)
  expected_configs <- c("single_likert", "single_pairwise", "tee_likert", "tee_pairwise")
  expect_true(all(expected_configs %in% s$config))
  for (cat in TARGET_CATEGORIES) {
    n <- sum(s$config %in% expected_configs & s$category_llm == cat)
    if (n != length(expected_configs)) {
      fail(sprintf("Category %s has %d config rows (expected %d)",
                    cat, n, length(expected_configs)))
    }
    expect_equal(n, length(expected_configs))
  }
})

test_that("agreement rates are in a plausible range", {
  skip_if_not(file.exists("data/processed/arena_agreement_summary.csv"))
  s <- read_csv("data/processed/arena_agreement_summary.csv", show_col_types = FALSE)
  # Chatbot Arena inter-LLM-judge-to-human agreement is well-studied;
  # single-judge baselines typically land at 60-75%. If any config falls
  # below 50% or above 95%, something is wrong.
  bad <- s %>% filter(!is.na(acc_ab), acc_ab < 0.50 | acc_ab > 0.95)
  if (nrow(bad) > 0) {
    fail(sprintf("Agreement out of [50%%, 95%%] range: %s",
                  paste(bad$config, bad$category_llm, round(bad$acc_ab, 3),
                        sep = "/", collapse = "; ")))
  }
  expect_true(all(is.na(s$acc_ab) | (s$acc_ab >= 0.50 & s$acc_ab <= 0.95)))
})
