# 16_cost_efficiency_frontier.R
# Cost-Efficiency Frontier: projected SE vs API cost for safety (AILuminate)
#
# Enumerates design configurations (N, V, M, R), computes D-study projected SE
# and API cost, identifies the Pareto frontier, and highlights how far the
# status quo (single-config) sits from efficient designs.
#
# Variance components from data/processed/variance_components_safety.csv
# Temperature held fixed (contributes <0.1% of variance).

library(tidyverse)

set.seed(42)

fig_dir <- "figures"
out_dir <- "data/processed"

cat("=== 16_cost_efficiency_frontier.R ===\n")

# =========================================================================
# 1. Load variance components
# =========================================================================

vc <- read_csv(file.path(out_dir, "variance_components_safety.csv"),
               show_col_types = FALSE)

pull_vc <- function(label) {
  row <- filter(vc, tle_label == label)
  if (nrow(row) == 0) return(0)
  row$variance[1]
}

s2_alpha     <- pull_vc("within-category item")      # item (within-cat)
s2_gamma     <- pull_vc("between-category")           # item (between-cat)
s2_rho_var   <- pull_vc("prompt")                     # prompt main
s2_lambda    <- pull_vc("judge model (design sensitivity)")  # judge main
s2_ap        <- pull_vc("item x prompt")              # item x prompt
s2_al        <- pull_vc("item x judge")               # item x judge
s2_pl        <- pull_vc("prompt x judge")             # prompt x judge
# Two residual components from the Phase 3 split:
# s2_eps_cell is constant within a cell, varies across cells -> NOT reducible by R.
# s2_rho_rep is call-to-call sampling at fixed cell -> reducible by R via averaging.
s2_eps_cell  <- pull_vc("cell-level (3-way+)")        # cell-level residual (3-way+)
s2_rho_rep   <- pull_vc("replicate noise")            # within-cell replicate noise

cat(sprintf("s2_alpha    = %.6f (within-cat item)\n", s2_alpha))
cat(sprintf("s2_gamma    = %.6f (between-cat item)\n", s2_gamma))
cat(sprintf("s2_rho_var  = %.6f (prompt)\n", s2_rho_var))
cat(sprintf("s2_lambda   = %.6f (judge)\n", s2_lambda))
cat(sprintf("s2_ap       = %.6f (item x prompt)\n", s2_ap))
cat(sprintf("s2_al       = %.6f (item x judge)\n", s2_al))
cat(sprintf("s2_pl       = %.6f (prompt x judge)\n", s2_pl))
cat(sprintf("s2_eps_cell = %.6f (cell-level 3-way+, NOT reducible by R)\n", s2_eps_cell))
cat(sprintf("s2_rho_rep  = %.6f (replicate noise, reducible by R)\n", s2_rho_rep))

# =========================================================================
# 2. D-study variance formula (temperature fixed)
# =========================================================================
# Var(theta_hat) = s2_gamma/N + s2_alpha/N + s2_rho_var/V + s2_lambda/M
#                + s2_ap/(N*V) + s2_al/(N*M) + s2_pl/(V*M)
#                + s2_eps_cell/(N*V*M)         [no /R — averaging within a cell can't reduce]
#                + s2_rho_rep/(N*V*M*R)         [the only term with /R]

dstudy_var <- function(N, V, M, R) {
  (s2_gamma + s2_alpha) / N +
    s2_rho_var / V +
    s2_lambda / M +
    s2_ap / (N * V) +
    s2_al / (N * M) +
    s2_pl / (V * M) +
    s2_eps_cell / (N * V * M) +
    s2_rho_rep  / (N * V * M * R)
}

# =========================================================================
# 3. Enumerate designs
# =========================================================================

N_MAX <- 141L  # AILuminate item pool

grid <- expand.grid(
  N = c(20L, 30L, 50L, 75L, 100L, 141L),
  V = c(1L, 2L, 3L, 5L),
  M = c(1L, 2L, 3L),
  R = c(1L, 2L, 3L, 5L, 8L)
) %>%
  mutate(
    cost = N * V * M * R,
    d_var = mapply(dstudy_var, N, V, M, R),
    se = sqrt(d_var)
  )

cat(sprintf("Grid: %d designs\n", nrow(grid)))

# =========================================================================
# 4. Pareto frontier
# =========================================================================
# A design is Pareto-optimal if no other design has both lower cost AND lower SE.

grid <- grid %>% arrange(cost, se)

frontier <- grid %>%
  arrange(cost) %>%
  mutate(frontier = accumulate(se, min) == se) %>%
  filter(frontier)

cat(sprintf("Frontier: %d designs\n", nrow(frontier)))

# =========================================================================
# 5. Key designs to label
# =========================================================================

status_quo <- grid %>% filter(N == N_MAX, V == 1, M == 1, R == 1)
full_design <- grid %>% filter(N == N_MAX, V == 5, M == 3, R == 8)
tee_moderate <- grid %>% filter(N == N_MAX, V == 3, M == 3, R == 1)

labels_df <- bind_rows(
  status_quo %>% mutate(label = "Status quo (V=1, M=1, R=1)",
                        hjust = 0, nudge_x = 0.15, nudge_y = 0.004),
  full_design %>% mutate(label = "Full factorial (V=5, M=3, R=8)",
                         hjust = 1, nudge_x = -0.15, nudge_y = 0.004),
  tee_moderate %>% mutate(label = "TEE-guided (V=3, M=3, R=1)",
                          hjust = 0, nudge_x = 0.15, nudge_y = -0.003)
)

cat("\n=== Key designs ===\n")
for (i in seq_len(nrow(labels_df))) {
  cat(sprintf("  %s: cost=%d, SE=%.4f\n",
              gsub("\n", " ", labels_df$label[i]),
              labels_df$cost[i], labels_df$se[i]))
}

# =========================================================================
# 6. Plot
# =========================================================================

# Drawn at print size (half-width panel, 2.7 in) so text prints at ~7 pt.
theme_set(theme_bw(base_size = 7.5))

# Dashed horizontal line from status quo to the frontier at same SE
# shows "you could get this SE for much less"
sq_frontier_match <- frontier %>% filter(se <= status_quo$se[1]) %>% slice_max(se, n = 1)

# Dashed vertical line from status quo down to frontier at same cost
# shows "you could get much better SE for this cost"
best_at_sq_cost <- frontier %>% filter(cost <= status_quo$cost[1]) %>% slice_min(se, n = 1)

p <- ggplot() +
  # All designs as light points
  geom_point(data = grid, aes(x = cost, y = se),
             colour = "grey80", size = 0.3, alpha = 0.4) +
  # Frontier
  geom_step(data = frontier, aes(x = cost, y = se),
            colour = "#4575b4", linewidth = 0.5, direction = "vh") +
  geom_point(data = frontier, aes(x = cost, y = se),
             colour = "#4575b4", size = 1.0) +
  # Status quo: bigger, distinct shape
  geom_point(data = status_quo, aes(x = cost, y = se),
             colour = "#d73027", size = 2.5, shape = 18) +
  # TEE-guided and full factorial
  geom_point(data = filter(labels_df, label != labels_df$label[1]),
             aes(x = cost, y = se),
             colour = "#d73027", size = 1.6) +
  # Labels via annotate for precise control
  annotate("text", x = status_quo$cost * 2.2, y = status_quo$se + 0.003,
           label = "Status quo", size = 2.2, fontface = "bold", hjust = 0) +
  annotate("text", x = status_quo$cost * 2.2, y = status_quo$se + 0.0005,
           label = "(V=1, M=1, R=1)", size = 1.9, hjust = 0, colour = "grey40") +
  annotate("text", x = tee_moderate$cost, y = 0.032,
           label = "TEE-guided", size = 2.2, fontface = "bold", hjust = 0.5) +
  annotate("text", x = tee_moderate$cost, y = 0.0295,
           label = "(V=3, M=3, R=1)", size = 1.9, hjust = 0.5, colour = "grey40") +
  annotate("text", x = full_design$cost * 1.3, y = 0.041,
           label = "Full factorial", size = 2.2, fontface = "bold", hjust = 1) +
  annotate("text", x = full_design$cost * 1.3, y = 0.0385,
           label = "(V=5, M=3, R=8)", size = 1.9, hjust = 1, colour = "grey40") +
  # Leader lines
  annotate("segment",
           x = status_quo$cost * 2, y = status_quo$se + 0.002,
           xend = status_quo$cost * 1.15, yend = status_quo$se + 0.0005,
           colour = "grey50", linewidth = 0.3) +
  annotate("segment",
           x = tee_moderate$cost, y = 0.028,
           xend = tee_moderate$cost, yend = tee_moderate$se + 0.001,
           colour = "grey50", linewidth = 0.3) +
  annotate("segment",
           x = full_design$cost, y = 0.037,
           xend = full_design$cost, yend = full_design$se + 0.001,
           colour = "grey50", linewidth = 0.3) +
  scale_x_log10(
    labels = scales::comma,
    breaks = c(100, 1000, 10000)
  ) +
  coord_cartesian(xlim = c(15, 30000), ylim = c(0.015, 0.067)) +
  labs(
    x = "API calls (log scale)",
    y = "Projected standard error"
  ) +
  theme(panel.grid.minor = element_blank(),
        plot.margin = margin(2, 2, 2, 2, "pt"))

pdf_path <- file.path(fig_dir, "fig_cost_efficiency.pdf")
png_path <- file.path(fig_dir, "fig_cost_efficiency.png")

ggsave(pdf_path, p, width = 2.7, height = 1.9)
ggsave(png_path, p, width = 2.7, height = 1.9, dpi = 300)

cat("\nSaved:", pdf_path, "\n")
cat("Saved:", png_path, "\n")

# =========================================================================
# 7. Key numbers for manuscript
# =========================================================================

sq_se <- status_quo$se[1]
mod_se <- tee_moderate$se[1]
full_se <- full_design$se[1]

cat("\n=== Manuscript numbers ===\n")
cat(sprintf("Status quo SE: %.4f (cost: %d)\n", sq_se, status_quo$cost[1]))
cat(sprintf("TEE-guided SE: %.4f (cost: %d) — %.0f%% SE reduction at %.1fx cost\n",
            mod_se, tee_moderate$cost[1],
            (1 - mod_se / sq_se) * 100,
            tee_moderate$cost[1] / status_quo$cost[1]))
cat(sprintf("Full design SE: %.4f (cost: %d) — %.0f%% SE reduction at %.1fx cost\n",
            full_se, full_design$cost[1],
            (1 - full_se / sq_se) * 100,
            full_design$cost[1] / status_quo$cost[1]))

# Same-cost comparison: what's the best frontier design at status-quo cost?
sq_cost <- status_quo$cost[1]
best_at_sq_cost <- frontier %>% filter(cost <= sq_cost) %>% slice_min(se, n = 1)
cat(sprintf("\nBest frontier design at ≤%d calls: N=%d, V=%d, M=%d, R=%d → SE=%.4f (%.0f%% reduction)\n",
            sq_cost, best_at_sq_cost$N, best_at_sq_cost$V, best_at_sq_cost$M, best_at_sq_cost$R,
            best_at_sq_cost$se, (1 - best_at_sq_cost$se / sq_se) * 100))
