#!/usr/bin/env Rscript
# 13d_fig_gaming_by_k.R
# Gaming figure for the paper and poster: expected Elo gain from reporting the best of K
# runs, for the five pipelines in the paper's gaming section. It plots the formula the
# paper states, sigma * E[max of K standard normals] (normal_approx_elo from 13e), so
# K = 27 matches the text (56, 45, 46, 46, 32 Elo). 13e also computes the gain directly
# from the bootstrap fits (gain_elo); those values run up to 3 Elo lower and are kept in
# the CSV as a check, not plotted.
#
# Input:   data/processed/arena_gaming_best_of_k.csv
# Outputs: figures/fig_gaming_by_k.pdf      (three lines: human, single judge, TEE)
#          figures/fig_gaming_by_k_all.pdf  (all five pipelines)

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

K_META <- 27L
curves <- fread("data/processed/arena_gaming_best_of_k.csv") %>%
  as_tibble() %>%
  mutate(gain_direct = gain_elo, gain_elo = normal_approx_elo,
         group = recode(pipeline_key, human = "human", `1_1` = "single",
                        `5_1` = "partial", `1_3` = "partial", `5_3` = "tee"))
K_MAX <- max(curves$K)

cat("Gain at K = 27 (Elo), plotted formula vs direct computation:\n")
print(curves %>% filter(K == K_META) %>% select(pipeline, gain_elo, gain_direct))

cols <- c(human = "grey40", single = "#D55E00", partial = "#E69F00", tee = "#0072B2")

plot_gain <- function(keep, file) {
  d <- curves %>% filter(group %in% keep)
  # One label per colour group; the two partial-averaging lines nearly coincide with the
  # human line, so their labels are pushed apart by a fixed offset when all are shown
  both <- all(c("human", "partial") %in% keep)
  labs_at <- function(k) d %>% filter(K == k) %>% group_by(group) %>%
    summarize(num = paste(unique(sprintf("%.0f", sort(gain_elo))), collapse = "-"),
              gain_elo = mean(gain_elo),
              pipeline = if (n() > 1) "5 prompts or 3 judges alone" else first(pipeline),
              .groups = "drop") %>% mutate(K = k) %>%
    mutate(nudge = if (both) case_when(group == "partial" ~ 2.8, group == "human" ~ -2.8,
                                       TRUE ~ 0) else 0)
  end <- d %>% filter(K == K_META)
  num <- labs_at(K_META) %>%
    mutate(vj = if (both) if_else(group == "human", 1.6, -0.35) else -0.35)
  lab <- labs_at(K_MAX)
  p <- ggplot(d, aes(K, gain_elo, colour = group, group = pipeline)) +
    geom_vline(xintercept = K_META, linetype = "dashed", colour = "grey60", linewidth = 0.4) +
    # bottom right of the panel is empty below the lowest curve
    annotate("text", x = K_META + 0.8, y = max(d$gain_elo) * 0.03, hjust = 0, vjust = 0,
             size = 3.9, colour = "grey30",
             label = "27 versions of\nLlama 4 tested") +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2) +
    geom_point(data = end, size = 3.2) +
    geom_text(data = num, aes(label = num, vjust = vj, group = NULL),
              hjust = 1.35, size = 4.2, show.legend = FALSE) +
    geom_text(data = lab, aes(x = K_MAX + 1.3, y = gain_elo + nudge, label = pipeline,
                              group = NULL),
              hjust = 0, size = 4.2, show.legend = FALSE) +
    scale_colour_manual(values = cols, guide = "none") +
    scale_x_continuous(breaks = c(1, 10, 20, 30, 40)) +
    # coord limits, not scale limits: the end labels sit past K_MAX
    coord_cartesian(xlim = c(1, K_MAX), ylim = c(0, max(d$gain_elo) * 1.1),
                    expand = FALSE, clip = "off") +
    labs(x = "Variants tested, reporting only the best (K)",
         y = "Expected Elo gain") +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(),
          axis.text = element_text(size = 11.5, colour = "grey20"),
          plot.margin = margin(6, 172, 2, 2, "pt"))
  ggsave(file, p, width = 5.4, height = 2.7)
}

plot_gain(c("human", "single", "tee"), "figures/fig_gaming_by_k.pdf")
plot_gain(c("human", "single", "partial", "tee"), "figures/fig_gaming_by_k_all.pdf")
cat("Saved figures/fig_gaming_by_k{,_all}.pdf\n")
