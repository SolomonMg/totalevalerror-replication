#!/usr/bin/env Rscript
# 45b_fig_tbench.R
# Figures for 45_tbench_decomposition.R (Terminal-Bench 2.0, 17 models under Terminus 2, 89 tasks, ~5 trials each).
#   fig_tbench_adjacent_gaps.pdf     gap between neighbouring models (dot plot): paired 95% CI over the tasks
#                                    (thick red line) over the CI that adds settings-matched harness x model
#                                    variance (thin black line); same style as fig_swebench_adjacent_gaps.pdf
#   fig_tbench_adjacent_gaps_lb.pdf  (chosen 2026-11-08) three nested 95% CIs per gap. The legend says, in prose
#                                    and as the TEE variance term, what each interval adds: the leaderboard's
#                                    per-model +/- values (rerun noise only; thick grey), naive CI over tasks
#                                    (red), TEE CI with settings-matched harness x model variance (thin black)
#   Variance of a one-harness gap (N = 89 tasks, R ~ 5 trials; task, harness and task x harness effects cancel):
#     2 s2_phi.lambda + 2 (s2_alpha.lambda + s2_eps) / N + 2 s2_rho / (N R)
#
# Inputs: data/processed/tbench2_{t2_pairs,t2_models,summary}.csv

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
})
PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)
Z <- qnorm(0.975)
theme_fig <- theme_minimal(base_size = 10) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        plot.subtitle = element_text(size = 8.5, colour = "grey25"), plot.title.position = "plot",
        plot.margin = margin(4, 8, 2, 2, "pt"))

models <- fread("data/processed/tbench2_t2_models.csv") %>% as_tibble()
summ <- fread("data/processed/tbench2_summary.csv") %>% as_tibble()
sd_gap <- summ$value[summ$stat == "sd_pair_gap_pts_matched"]
# Soft hyphens (U+00AD): the pdf device draws "-" as a minus sign, and cairo_pdf needs XQuartz (not installed)
hy <- function(x) gsub("-", "\u00ad", x, fixed = TRUE)
lab <- setNames(hy(sprintf("%s (%.1f)", models$model, models$acc_pts)), models$model)
adj <- fread("data/processed/tbench2_t2_pairs.csv") %>% as_tibble() %>% filter(adjacent) %>%
  mutate(pair = paste(lab[model_a], "vs", lab[model_b]), pair = factor(pair, levels = rev(pair)))
stopifnot(nrow(adj) == 16)

base <- function(d) {
  ggplot(d, aes(x = gap_pts, y = pair)) +
    geom_vline(xintercept = 0, colour = "grey70", linewidth = 0.4) +
    geom_linerange(aes(xmin = gap_pts - Z * se_tee_matched_pts, xmax = gap_pts + Z * se_tee_matched_pts),
                   linewidth = 0.5, colour = "grey20") +
    geom_linerange(aes(xmin = gap_pts - Z * se_paired_pts, xmax = gap_pts + Z * se_paired_pts),
                   linewidth = 1.1, colour = "#C62828")
}
sub1 <- paste0("Terminal-Bench 2.0, 17 models under the Terminus 2 harness (accuracy % in parentheses).\n",
               "Red: paired 95% CI over the 89 tasks.\nBlack: 95% CI after adding how much the gap ",
               sprintf("moves with the harness (SD %.1f points).", sd_gap))
p1 <- base(adj) + geom_point(size = 2.2, colour = "black") +
  labs(x = hy("Gap (points), higher-ranked model minus the next"), y = NULL, subtitle = hy(sub1)) + theme_fig
ggsave("figures/fig_tbench_adjacent_gaps.pdf", p1, width = 6.5, height = 4.5)

# ---- Chosen: three nested intervals with a legend ----
sd_mh <- summ$value[summ$stat == "sd_mh_pts_matched"]
KEYS <- c("lb", "naive", "tee")
lab_tee <- list(
  bquote("rerunning the same two models on the same 89 tasks in the same harness (leaderboard \u00b1):" ~
           2 * sigma[rho]^2 / NR),
  bquote("the above, plus which 89 tasks the benchmark happens to include (naive CI):" ~ "+" ~
           2 * (sigma[alpha * lambda]^2 + sigma[epsilon]^2) / N),
  bquote("the above, plus another harness that sends the models the same settings (TEE CI):" ~ "+" ~
           2 * sigma[phi * lambda]^2 * "," ~ sigma[phi * lambda] == .(sprintf("%.1f", sd_mh)) ~ "points"))
lines3 <- function(labels, title = "95% CI for the gap") {
  ggplot(adj, aes(x = gap_pts, y = pair)) +
    geom_vline(xintercept = 0, colour = "grey70", linewidth = 0.4) +
    geom_linerange(aes(xmin = gap_pts - Z * se_tee_matched_pts, xmax = gap_pts + Z * se_tee_matched_pts,
                       colour = "tee", linewidth = "tee"), key_glyph = "path") +
    geom_linerange(aes(xmin = gap_pts - Z * se_paired_pts, xmax = gap_pts + Z * se_paired_pts,
                       colour = "naive", linewidth = "naive"), key_glyph = "path") +
    geom_linerange(aes(xmin = gap_pts - Z * se_trial_pts, xmax = gap_pts + Z * se_trial_pts,
                       colour = "lb", linewidth = "lb"), key_glyph = "path") +
    geom_point(size = 2.2, colour = "black") +
    scale_colour_manual(values = c(lb = "#90A4AE", naive = "#C62828", tee = "grey20"), breaks = KEYS,
                        labels = labels, name = title) +
    scale_linewidth_manual(values = c(lb = 2.6, naive = 1.1, tee = 0.5), breaks = KEYS, labels = labels,
                           name = title) +
    labs(x = hy("Gap (points), higher-ranked model minus the next"), y = NULL,
         subtitle = hy(paste0("Terminal-Bench 2.0, 17 models under the Terminus 2 harness (accuracy % in parentheses);\n",
                              "N = 89 tasks, R = about 5 runs per task and model."))) +
    theme_fig +
    theme(legend.position = "bottom", legend.direction = "vertical", legend.justification = "left",
          legend.location = "plot", legend.title = element_text(size = 8.5, face = "bold"),
          legend.text = element_text(size = 8), legend.key.width = unit(1.1, "cm"),
          legend.key.height = unit(0.4, "cm"), legend.margin = margin(0, 0, 0, 0))
}
ggsave("figures/fig_tbench_adjacent_gaps_lb.pdf",
       lines3(lab_tee, "Each 95% CI for the gap counts the variation (and variance term) from:"),
       width = 6.5, height = 5.6)
cat("Saved figures/fig_tbench_adjacent_gaps{,_lb}.pdf\n")

# ---- Two panels (chosen 2026-11-09): (a) each model's score, in the style of leaderboard bar charts; (b) the gaps ----
# Variance of one model's score under one harness, relative to the other models (the harness main effect moves
# every model equally, so it is left out):
#   s2_phi.lambda + (s2_alpha + s2_alpha.phi + s2_alpha.lambda + s2_eps) / N + s2_rho / (N R)
# Logos: data/raw/external/model_logos, written by experiments/fetch_model_logos.py
library(patchwork)
read_logo <- function(name) {
  f <- file.path("data/raw/external/model_logos", name)
  wh <- scan(paste0(f, ".dim"), quiet = TRUE)
  b <- readBin(paste0(f, ".rgba"), "raw", n = prod(wh) * 4)
  as.raster(aperm(array(as.integer(b) / 255, dim = c(4, wh[1], wh[2])), c(3, 2, 1)))
}
ORG <- c(Claude = "anthropic", GPT = "openai", gpt = "openai", Gemini = "google-color", Grok = "xai",
         Qwen = "alibaba-color", Kimi = "moonshot", GLM = "zai", MiniMax = "minimax-color")
COLS <- c(lb = "#90A4AE", naive = "#C62828", tee = "grey20")
LWD <- c(lb = 2.6, naive = 1.1, tee = 0.5)
sc <- models %>% arrange(rank) %>%
  mutate(i = rank, logo = ORG[str_extract(model, "^[A-Za-z]+")], se_tee_pts = sqrt(se_naive_pts^2 + sd_mh^2))
stopifnot(!anyNA(sc$logo), identical(sc$i, seq_len(17)))
xs <- scale_x_continuous(breaks = sc$i, labels = hy(sc$model), expand = expansion(add = 0.6))
pa <- ggplot(sc, aes(i, acc_pts)) +
  geom_col(fill = "grey93", width = 0.8) +
  geom_linerange(aes(ymin = pmax(0, acc_pts - Z * se_tee_pts), ymax = acc_pts + Z * se_tee_pts),
                 colour = COLS[["tee"]], linewidth = LWD[["tee"]]) +
  geom_linerange(aes(ymin = pmax(0, acc_pts - Z * se_naive_pts), ymax = acc_pts + Z * se_naive_pts),
                 colour = COLS[["naive"]], linewidth = LWD[["naive"]]) +
  geom_linerange(aes(ymin = acc_pts - Z * se_trial_pts, ymax = acc_pts + Z * se_trial_pts),
                 colour = COLS[["lb"]], linewidth = LWD[["lb"]]) +
  geom_point(size = 1.4) +
  geom_text(aes(y = acc_pts + Z * se_tee_pts, label = sprintf("%.1f", acc_pts)), vjust = -0.5, size = 2.5,
            colour = "grey15") +
  xs + scale_y_continuous(expand = c(0, 0)) +
  coord_cartesian(ylim = c(0, max(sc$acc_pts + Z * sc$se_tee_pts) + 5), clip = "off") +
  labs(x = NULL, y = "Accuracy (%)", subtitle = "(a) Each model's score") +
  theme_fig +
  theme(panel.grid.major.x = element_blank(), axis.text.x = element_blank(),
        panel.grid.major.y = element_line(colour = "grey88", linetype = "dashed"), plot.margin = margin(4, 8, 0, 2))
ps <- ggplot(sc, aes(i, 0.5)) + geom_blank() + xs +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) + coord_cartesian(clip = "off") +
  theme_void(base_size = 10) +
  theme(axis.text.x = element_text(angle = 50, hjust = 1, vjust = 1, size = 7.5, margin = margin(t = 2)),
        plot.margin = margin(0, 8, 2, 2))
logos <- map(set_names(unique(sc$logo)), read_logo)
for (k in seq_len(nrow(sc)))
  ps <- ps + annotation_custom(grid::rasterGrob(logos[[sc$logo[k]]], interpolate = TRUE),
                               xmin = k - 0.45, xmax = k + 0.45, ymin = 0.08, ymax = 0.92)
pb <- lines3(lab_tee) + labs(subtitle = "(b) Gap between neighboring models (accuracy % in parentheses)") +
  theme(legend.position = "none")
# One legend for both panels: what each interval counts, and its variance term for a score (a) and a gap (b)
leg <- tibble(key = KEYS, y = 3:1,
              desc = c("reruns of the same model on the same 89 tasks (leaderboard ±)",
                       "+ which 89 tasks the benchmark includes (naive CI)",
                       "+ the harness, as a prompt variant (TEE CI)"),
              ta = c("sigma[rho]^2/NR",
                     "'+' ~ (sigma[alpha]^2 + sigma[alpha*phi]^2 + sigma[alpha*lambda]^2 + sigma[epsilon]^2)/N",
                     "'+' ~ sigma[phi*lambda]^2"),
              tb = c("2*sigma[rho]^2/NR", "'+' ~ 2*(sigma[alpha*lambda]^2 + sigma[epsilon]^2)/N",
                     "'+' ~ 2*sigma[phi*lambda]^2"))
XA <- 0.59; XB <- 0.85
pl <- ggplot(leg) +
  geom_segment(aes(x = 0, xend = 0.04, y = y, yend = y, colour = key, linewidth = key)) +
  scale_colour_manual(values = COLS, guide = "none") + scale_linewidth_manual(values = LWD, guide = "none") +
  geom_text(aes(x = 0.052, y = y, label = desc), hjust = 0, size = 2.65) +
  geom_text(aes(x = XA, y = y, label = ta), parse = TRUE, hjust = 0, size = 2.65) +
  geom_text(aes(x = XB, y = y, label = tb), parse = TRUE, hjust = 0, size = 2.65) +
  annotate("text", x = c(0, XA, XB), y = 4.1, hjust = 0, size = 2.75, fontface = "bold",
           label = c("Each 95% CI counts the variation from:", "Term for a score (a)", "Term for a gap (b)")) +
  annotate("text", x = 0, y = -0.1, hjust = 0, size = 2.55, colour = "grey25", parse = TRUE,
           label = sprintf("'N = 89 tasks, R = about 5 runs per task and model,' ~ hat(sigma)[phi*lambda] == '%.1f points.'",
                           sd_mh)) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) + scale_y_continuous(limits = c(-0.5, 4.5), expand = c(0, 0)) +
  coord_cartesian(clip = "off") + theme_void() + theme(plot.margin = margin(2, 8, 2, 4))
fig2 <- wrap_elements(full = pa / ps + plot_layout(heights = unit(c(1, 0.24), c("null", "in")))) /
  wrap_elements(full = pb) / pl + plot_layout(heights = unit(c(1, 1.15, 0.95), c("null", "null", "in")))
ggsave("figures/fig_tbench_scores_gaps.pdf", fig2, width = 6.5, height = 8)
cat("Saved figures/fig_tbench_scores_gaps.pdf\n")


# Embed all fonts, the base 14 included (R's pdf device embeds none), so print shops and the paper's PDF get them
for (f in c("figures/fig_tbench_adjacent_gaps.pdf", "figures/fig_tbench_adjacent_gaps_lb.pdf",
           "figures/fig_tbench_scores_gaps.pdf")) {
  tmp <- tempfile(fileext = ".pdf")
  embedFonts(f, outfile = tmp, options = paste("-dPDFSETTINGS=/prepress -dEmbedAllFonts=true -dSubsetFonts=true",
                                               "-c \"<</NeverEmbed [ ]>> setdistillerparams\" -f"))
  stopifnot(file.copy(tmp, f, overwrite = TRUE))
}
cat("Embedded fonts in all three figures\n")
