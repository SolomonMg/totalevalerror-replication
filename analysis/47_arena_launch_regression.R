#!/usr/bin/env Rscript
# 47_arena_launch_regression.R
# Do Chatbot Arena scores fall after launch? A test for selection on noise (benchmark hacking) in practice.
#
# If a lab tests several private variants and launches the one that scored best, the launch rating carries the
# selected variant's luck from the private-testing votes. As new votes arrive the rating regresses (winner's curse).
# Without selection, a model's rating should drift up or down by chance only, relative to the rest of the board.
#
# Data: Arena's published leaderboard history (lmarena-ai/leaderboard-dataset, CC-BY-4.0), overall category,
# converted by experiments/arena_leaderboard_to_csv.py. We use the leaderboard Arena displayed by default:
#   era 1  2024-01-09 (Bradley-Terry ratings begin) to 2025-05-15: text (no style control)
#   era 2  2025-05-16 (style control becomes the default) to 2026-10-19: text_style_control
# Arena renamed most models to display names on 2026-10-20 (391 new names, 390 retired), so snapshots stop at
# 2026-10-19.
#
# Launch = a model's first snapshot in the era, at least 30 days after the era starts (so it is a real debut).
# Follow-up = the model's first snapshot at least H days later (H = 28; 14 and 56 as checks), within H + 21 days.
# Change = (rating at follow-up - rating at launch) minus the median change of established models over the same
# two snapshots (present in both, first seen 60+ days before launch, 20,000+ votes at launch). That removes Arena's
# scale shifts, including the 2025-07-23 frequency reweighting; pairs that span it are also dropped as a check.
# Pre-release testing proxy: votes at debut. A model that debuts with many votes collected them while anonymous.
# Implied launch inflation: launch votes n0 are part of the follow-up rating (n1 votes), so a launch inflation I
# shows up as a change of -I (1 - n0/n1). We estimate I by regressing the change on (1 - n0/n1) with no intercept.
#
# Outputs: data/processed/arena_launch_changes.csv (one row per launch and horizon)
#          data/processed/arena_launch_pretest_diff.csv (pretested minus not, H = 28, cutoffs 5k/10k/20k)
#          data/processed/arena_launch_summary.csv (mean change and implied inflation by group, bootstrap 95% CIs)

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
})
PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)
set.seed(42)
N_BOOT <- 4000L
RENAME <- as.Date("2026-10-20")
RW <- as.Date("2025-07-23")
ERAS <- tribble(~era, ~cfg, ~start, ~end,
                "1 (no style control)", "text", as.Date("2024-01-09"), as.Date("2025-05-15"),
                "2 (style control)", "text_style_control", as.Date("2025-05-16"), RENAME - 1)
PRETEST_VOTES <- 10000L

launches_for <- function(era, cfg, start, end, H) {
  d <- fread(sprintf("data/processed/arena_lb_overall_%s.csv", cfg)) %>% as_tibble() %>%
    transmute(model = model_name, org = organization, date = as.Date(leaderboard_publish_date), rating, vote_count,
              hw = (rating_upper - rating_lower) / 2)
  first_seen <- d %>% group_by(model) %>% summarize(first_any = min(date), .groups = "drop")
  d <- d %>% filter(date >= start, date <= end) %>% left_join(first_seen, by = "model")
  snaps <- sort(unique(d$date))
  L <- d %>% filter(date == first_any, date >= start + 30)
  rows <- map_dfr(seq_len(nrow(L)), function(k) {
    l <- L[k, ]
    f <- d %>% filter(model == l$model, date >= l$date + H, date <= l$date + H + 21) %>% slice_min(date, n = 1)
    if (nrow(f) == 0) return(NULL)
    ref0 <- d %>% filter(date == l$date, model != l$model, first_any <= l$date - 60, vote_count >= 20000)
    ref1 <- d %>% filter(date == f$date, model %in% ref0$model)
    ref <- inner_join(ref0 %>% select(model, r0 = rating), ref1 %>% select(model, r1 = rating), by = "model")
    if (nrow(ref) < 10) return(NULL)
    tibble(era = era, horizon = H, model = l$model, org = l$org, t0 = l$date, t1 = f$date, r0 = l$rating, r1 = f$rating,
           n0 = l$vote_count, n1 = f$vote_count, hw0 = l$hw, n_ref = nrow(ref), drift = median(ref$r1 - ref$r0),
           change = (f$rating - l$rating) - median(ref$r1 - ref$r0), dilution = 1 - l$vote_count / f$vote_count,
           spans_reweighting = l$date < RW & f$date >= RW)
  })
  rows
}

ch <- pmap_dfr(ERAS, function(era, cfg, start, end) map_dfr(c(14L, 28L, 56L), ~ launches_for(era, cfg, start, end, .x))) %>%
  mutate(pretested = n0 >= PRETEST_VOTES)
fwrite(ch, "data/processed/arena_launch_changes.csv")

boot_stats <- function(x) {
  est <- c(mean_change = mean(x$change), inflation = -coef(lm(change ~ 0 + dilution, data = x))[[1]])
  b <- replicate(N_BOOT, {
    s <- x[sample.int(nrow(x), replace = TRUE), ]
    c(mean(s$change), -coef(lm(change ~ 0 + dilution, data = s))[[1]])
  })
  tibble(n = nrow(x), mean_change = est[["mean_change"]], mean_lo = quantile(b[1, ], 0.025),
         mean_hi = quantile(b[1, ], 0.975), share_down = mean(x$change < 0), inflation = est[["inflation"]],
         infl_lo = quantile(b[2, ], 0.025), infl_hi = quantile(b[2, ], 0.975), median_n0 = median(x$n0),
         median_dilution = median(x$dilution), median_hw0 = median(x$hw0))
}
groups <- list(
  all = function(x) x,
  `pretested (>= 10k debut votes)` = function(x) filter(x, pretested),
  `not pretested (< 10k debut votes)` = function(x) filter(x, !pretested),
  `all, no pair spans 2025-07-23 reweighting` = function(x) filter(x, !spans_reweighting))
summ <- expand_grid(horizon = c(14L, 28L, 56L), era = c("both", unique(ch$era)), group = names(groups)) %>%
  pmap_dfr(function(horizon, era, group) {
    x <- ch %>% filter(.data$horizon == !!horizon, era == !!era | !!era == "both")
    x <- groups[[group]](x)
    if (nrow(x) < 8) return(NULL)
    bind_cols(tibble(horizon = horizon, era = era, group = group), boot_stats(x))
  })
fwrite(summ, "data/processed/arena_launch_summary.csv")

# Pretested minus not pretested (mean change and implied inflation), H = 28, with the vote cutoff varied as a check
# (the main cutoff, 10k, was set before looking at results; 5k and 20k are reported alongside, not chosen from).
diff_stats <- function(x) {
  f <- function(s) {
    a <- s[s$pre, ]; b <- s[!s$pre, ]
    c(mean(a$change) - mean(b$change),
      -coef(lm(change ~ 0 + dilution, data = a))[[1]] + coef(lm(change ~ 0 + dilution, data = b))[[1]])
  }
  est <- f(x)
  # stratified bootstrap: resample within each group so both groups keep their size
  ia <- which(x$pre); ib <- which(!x$pre)
  b <- replicate(N_BOOT, f(x[c(sample(ia, replace = TRUE), sample(ib, replace = TRUE)), ]))
  tibble(n_pre = length(ia), n_not = length(ib), diff_mean = est[1], diff_lo = quantile(b[1, ], 0.025),
         diff_hi = quantile(b[1, ], 0.975), diff_infl = est[2], dinfl_lo = quantile(b[2, ], 0.025),
         dinfl_hi = quantile(b[2, ], 0.975))
}
dsum <- expand_grid(cutoff = c(5000L, 10000L, 20000L), era = c("both", unique(ch$era))) %>%
  pmap_dfr(function(cutoff, era) {
    x <- ch %>% filter(horizon == 28, era == !!era | !!era == "both") %>% mutate(pre = n0 >= cutoff)
    if (sum(x$pre) < 8) return(NULL)
    bind_cols(tibble(cutoff = cutoff, era = era), diff_stats(x))
  })
fwrite(dsum, "data/processed/arena_launch_pretest_diff.csv")

cat(sprintf("Launches with a follow-up: H=14 %d, H=28 %d, H=56 %d\n", sum(ch$horizon == 14), sum(ch$horizon == 28),
            sum(ch$horizon == 56)))
print(as.data.frame(summ %>% filter(horizon == 28) %>%
  transmute(era, group, n, mean = round(mean_change, 2), lo = round(mean_lo, 2), hi = round(mean_hi, 2),
            down = round(share_down, 2), infl = round(inflation, 1), infl_lo = round(infl_lo, 1),
            infl_hi = round(infl_hi, 1), n0 = median_n0, dil = round(median_dilution, 2))), row.names = FALSE)
cat("\nH = 14 and 56, both eras:\n")
print(as.data.frame(summ %>% filter(horizon != 28, era == "both") %>%
  transmute(horizon, group, n, mean = round(mean_change, 2), lo = round(mean_lo, 2), hi = round(mean_hi, 2),
            infl = round(inflation, 1), infl_lo = round(infl_lo, 1), infl_hi = round(infl_hi, 1))), row.names = FALSE)
cat("\nPretested minus not pretested, H = 28 (debut-vote cutoff varied):\n")
print(as.data.frame(dsum %>% mutate(across(where(is.double), ~ round(.x, 2)))), row.names = FALSE)
cat(sprintf("\nMedian launch 95%% CI half-width (rating points): all %.1f, pretested %.1f, not %.1f\n",
            median(ch$hw0[ch$horizon == 28]), median(ch$hw0[ch$horizon == 28 & ch$pretested]),
            median(ch$hw0[ch$horizon == 28 & !ch$pretested])))
cat("Saved data/processed/arena_launch_{changes,summary,pretest_diff}.csv\n")
