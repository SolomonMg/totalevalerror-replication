# 04g2b: Compare the fixed-level and redrawn-level runs of 04g2.
#
# 04g2 (default) holds the three judge and three temperature levels fixed at
# c(-0.2, 0, 0.2). 04g2 --redraw-design draws them from N(0, pop_var) in every
# replicate, so the estimand is the superpopulation mean mu. This script reads
# both per-replicate CSVs and writes one comparison table for the SI
# (si:underestimation_redraw) and the D.7 ratio sentence.
#
# Usage: Rscript analysis/04g2b_redraw_comparison.R

suppressPackageStartupMessages(library(tidyverse))

data_dir <- "data/processed"
runs <- c(fixed = "sim_underestimation_v2.csv",
          redraw = "sim_underestimation_v2_redraw.csv")

reps <- imap_dfr(runs, function(f, design) {
  read_csv(file.path(data_dir, f), show_col_types = FALSE, guess_max = 1e6) |>
    mutate(design = design, .before = 1)
})

comparison <- reps |>
    group_by(design, scenario, n_items) |>
    summarise(
      n_sims = n(),
      coverage = mean(covers),
      coverage_mcse = sqrt(coverage * (1 - coverage) / n_sims),
      mean_se = mean(se),
      median_se = median(se),
      emp_sd = sd(theta_hat),
      # Median: a few degenerate Scenario E fits return SEs of 3 - 317, which
      # dominate the mean but leave coverage unchanged.
      se_over_sd = median_se / emp_sd,
      # Timed-out fits fall back to the scenario's oracle SE, a constant per
      # (scenario, N); fitted SEs never repeat exactly.
      n_fallback = {
        dup <- se[duplicated(se) | duplicated(se, fromLast = TRUE)]
        if (scenario[1] == "A") 0L else length(dup)
      },
      n_degenerate = sum(se > 1),
      .groups = "drop"
    )

out <- file.path(data_dir, "sim_underestimation_redraw_comparison.csv")
write_csv(comparison, out)
cat("Saved:", out, "\n")
print(comparison |> select(design, scenario, n_items, coverage, se_over_sd, n_fallback, n_degenerate) |>
        pivot_wider(names_from = design, values_from = c(coverage, se_over_sd, n_fallback, n_degenerate)),
      n = Inf, width = Inf)

# Coverage by within-cell SE quartile, Scenarios D - F: shows where misses fall
# when the SE rests on three judge and three temperature levels.
by_quartile <- reps |>
  filter(scenario %in% c("D", "E", "F")) |>
  group_by(design, scenario, n_items) |>
  mutate(se_quartile = ntile(se, 4)) |>
  group_by(design, scenario, se_quartile) |>
  summarise(coverage = mean(covers), n = n(), .groups = "drop")
out_q <- file.path(data_dir, "sim_underestimation_redraw_se_quartile.csv")
write_csv(by_quartile, out_q)
cat("Saved:", out_q, "\n")
print(by_quartile |> select(-n) |> pivot_wider(names_from = se_quartile, values_from = coverage,
                                               names_prefix = "SE_q"))

# LaTeX rows for tab:underestimation_redraw (coverage in percent).
cat("\n% rows for tab:underestimation_redraw\n")
comparison |>
  mutate(cell = sprintf("%.1f", 100 * coverage)) |>
  select(design, scenario, n_items, cell) |>
  pivot_wider(names_from = n_items, values_from = cell) |>
  arrange(scenario, design) |>
  mutate(row = sprintf("%s & %s & %s \\\\", scenario,
                       if_else(design == "fixed", "Fixed", "Redrawn"),
                       paste(`25`, `50`, `100`, `200`, `400`, sep = " & "))) |>
  pull(row) |>
  cat(sep = "\n")
cat("\n")
