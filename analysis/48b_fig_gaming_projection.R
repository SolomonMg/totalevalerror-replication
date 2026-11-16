#!/usr/bin/env Rscript
# 48b_fig_gaming_projection.R
# Best-of-K gaming gain curves at live-leaderboard vote counts (analysis 48), in the style of 13d's
# fig_gaming_by_k.pdf. Gain(K) = sigma(n) * E[max of K standard normals], sigma(n) from
# arena_gaming_projection.csv; the sample curves are 13e's normal approximation (what fig_gaming_by_k plots).
#
# Outputs: data/processed/arena_gaming_projection_by_k.csv
#          figures/fig_gaming_by_k_projected.pdf      (option B: projected only, same size as fig_gaming_by_k)
#          figures/fig_gaming_by_k_projection_2panel.pdf (option A: sample | projected, shared y-axis)

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(patchwork)
})
PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)

K_META <- 27L
N_LABEL <- "live launch, median"
e_max <- function(K, n_sim = 100000) {   # as in 13e and 48
  set.seed(42); z <- matrix(rnorm(K * n_sim), nrow = K); mean(apply(z, 2, max))
}
LAB <- c(human = "Human voting", `1_1` = "One judge, one prompt", `5_3` = "TEE: 5 prompts × 3 judges")
GRP <- c(human = "human", `1_1` = "single", `5_3` = "tee")

samp <- fread("data/processed/arena_gaming_best_of_k.csv") %>% as_tibble() %>%
  filter(pipeline_key %in% names(LAB)) %>%
  transmute(pipeline_key, K, gain_elo = normal_approx_elo, panel = "sample")
proj_sig <- fread("data/processed/arena_gaming_projection.csv") %>% as_tibble() %>% filter(n_label == N_LABEL)
n_proj <- unique(proj_sig$n); stopifnot(length(n_proj) == 1)
Ks <- sort(unique(samp$K))
em <- tibble(K = Ks, em = map_dbl(Ks, e_max))
proj <- proj_sig %>% filter(pipeline_key %in% names(LAB)) %>% select(pipeline_key, sigma_elo, gain_k27_elo) %>%
  crossing(em) %>% mutate(gain_elo = sigma_elo * em, panel = "projected")
stopifnot(max(abs(proj$gain_elo[proj$K == K_META] - proj$gain_k27_elo[proj$K == K_META])) < 1e-9)
curves <- bind_rows(samp, proj %>% select(pipeline_key, K, gain_elo, panel)) %>%
  mutate(pipeline = LAB[pipeline_key], group = GRP[pipeline_key])
fwrite(curves %>% mutate(n_per_model = if_else(panel == "projected", n_proj, NA_real_)),
       "data/processed/arena_gaming_projection_by_k.csv")
K_MAX <- max(curves$K)
cols <- c(human = "grey40", single = "#D55E00", tee = "#0072B2")

panel_plot <- function(d, ymax, end_labels, title = NULL, llama = TRUE, llama_y = 0.03, spread = FALSE) {
  # spread = TRUE when the human and TEE curves sit close together: number labels go above (human) and below (TEE),
  # and the end labels are pushed apart
  end <- d %>% filter(K == K_META) %>%
    mutate(vj = if (spread) if_else(group == "tee", 1.5, -0.35) else -0.35)
  lab <- d %>% filter(K == K_MAX) %>%
    mutate(nudge = if (spread) case_when(group == "human" ~ 0.045, group == "tee" ~ -0.045, TRUE ~ 0) * ymax else 0)
  p <- ggplot(d, aes(K, gain_elo, colour = group, group = pipeline)) +
    geom_vline(xintercept = K_META, linetype = "dashed", colour = "grey60", linewidth = 0.4)
  if (llama) p <- p + annotate("text", x = K_META + 0.8, y = ymax * llama_y, hjust = 0, vjust = 0, size = 3.6,
                               colour = "grey30", label = "27 versions of\nLlama 4 tested")
  p <- p + geom_line(linewidth = 1.1) + geom_point(size = 2) + geom_point(data = end, size = 3.2) +
    geom_text(data = end, aes(label = sprintf("%.0f", gain_elo), vjust = vj), hjust = 1.35, size = 4.2,
              show.legend = FALSE)
  if (end_labels) p <- p + geom_text(data = lab, aes(x = K_MAX + 1.3, y = gain_elo + nudge, label = pipeline), hjust = 0, size = 4.2,
                                     show.legend = FALSE)
  p + scale_colour_manual(values = cols, guide = "none") +
    scale_x_continuous(breaks = c(1, 10, 20, 30, 40)) +
    coord_cartesian(xlim = c(1, K_MAX), ylim = c(0, ymax), expand = FALSE, clip = "off") +
    labs(x = "Variants tested, reporting only the best (K)", y = "Expected Elo gain", title = title) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(), axis.text = element_text(size = 11.5, colour = "grey20"),
          plot.title = element_text(size = 11.5, colour = "grey20"), plot.title.position = "plot",
          plot.margin = margin(6, if (end_labels) 172 else 12, 2, 2, "pt"))
}

ymax <- max(curves$gain_elo) * 1.1
# Option B: projected only
pB <- panel_plot(filter(curves, panel == "projected"), max(curves$gain_elo[curves$panel == "projected"]) * 1.15,
                 TRUE, sprintf("Projected: %s votes per model (median at launch)", format(n_proj, big.mark = ",")),
                 llama_y = 0.42, spread = TRUE)
ggsave("figures/fig_gaming_by_k_projected.pdf", pB, width = 5.4, height = 2.95)
# R's pdf() device does not embed Helvetica; the poster print order needs every font embedded (needs Ghostscript).
# Ghostscript skips the base-14 fonts unless NeverEmbed is cleared.
embedFonts("figures/fig_gaming_by_k_projected.pdf", options = '-c "<</NeverEmbed [ ]>> setdistillerparams" -f')
# Option A: sample | projected, shared y-axis
pA <- panel_plot(filter(curves, panel == "sample"), ymax, FALSE, "(a) Paper's sample: 163 matches per model") +
  panel_plot(filter(curves, panel == "projected"), ymax, TRUE,
             sprintf("(b) Projected: %s votes per model", format(n_proj, big.mark = ",")), llama = FALSE,
             spread = TRUE) +
  plot_layout(widths = c(1, 1))
ggsave("figures/fig_gaming_by_k_projection_2panel.pdf", pA, width = 9.6, height = 3.0)
print(as.data.frame(curves %>% filter(K == K_META) %>% select(panel, pipeline, gain_elo) %>%
                      mutate(gain_elo = round(gain_elo, 1))), row.names = FALSE)
cat("Saved figures/fig_gaming_by_k_projected.pdf, figures/fig_gaming_by_k_projection_2panel.pdf\n")
