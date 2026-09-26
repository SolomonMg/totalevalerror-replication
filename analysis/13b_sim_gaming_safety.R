# 13b_sim_gaming_safety.R
# Gaming Surface Simulation calibrated to SAFETY benchmark (AILuminate)
#
# Same logic as 13_sim_gaming.R but uses variance components from the
# safety binary classification decomposition (N=141 items, 0-1 scale).
# Safety is a real benchmark, making the gaming scenario more concrete.
#
# Usage:
#   Rscript analysis/13b_sim_gaming_safety.R
#   Rscript analysis/13b_sim_gaming_safety.R --nsim 5   # smoke test

library(tidyverse)

set.seed(42)

args  <- commandArgs(trailingOnly = TRUE)
N_SIM <- 1000L
if ("--nsim" %in% args) {
  idx <- which(args == "--nsim")
  if (idx < length(args)) N_SIM <- as.integer(args[idx + 1])
}

fig_dir <- "figures"
out_dir <- "data/processed"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 13b_sim_gaming_safety.R: Safety Benchmark Gaming Simulation ===\n")
cat(sprintf("N_SIM = %d\n", N_SIM))

# =========================================================================
# Load safety variance components
# =========================================================================

vc_path <- file.path(out_dir, "variance_components_safety.csv")
if (!file.exists(vc_path)) stop("Missing: ", vc_path)

vc_raw <- read_csv(vc_path, show_col_types = FALSE)

pull_vc <- function(label) {
  row <- filter(vc_raw, tle_label == label)
  if (nrow(row) == 0) {
    warning("Component not found: ", label, " — using 0")
    return(0)
  }
  row$variance[1]
}

sigma2_rho    <- pull_vc("prompt")                           # prompt main: ~0
sigma2_lambda <- pull_vc("judge model (design sensitivity)") # judge main: 0.00046
sigma2_ap     <- pull_vc("item x prompt")                    # item x prompt: 0.0023
sigma2_al     <- pull_vc("item x judge")                     # item x judge: 0.0254
sigma2_pl     <- pull_vc("prompt x judge")                   # prompt x judge: 0.0007
sigma2_eps    <- pull_vc("generation")                       # residual: 0.0163

N_ITEMS <- 141L  # AILuminate benchmark

cat(sprintf("sigma2_rho    = %.6f  (prompt main)\n",      sigma2_rho))
cat(sprintf("sigma2_lambda = %.6f  (judge main)\n",       sigma2_lambda))
cat(sprintf("sigma2_ap     = %.6f  (item x prompt)\n",    sigma2_ap))
cat(sprintf("sigma2_al     = %.6f  (item x judge)\n",     sigma2_al))
cat(sprintf("sigma2_pl     = %.6f  (prompt x judge)\n",   sigma2_pl))
cat(sprintf("sigma2_eps    = %.6f  (residual)\n",         sigma2_eps))
cat(sprintf("N_ITEMS = %d\n", N_ITEMS))

# =========================================================================
# Design grid
# =========================================================================

V_LEVELS <- 1:8
M_LEVELS <- c(1L, 2L, 3L)
K_TRIES  <- 10L

# D-study variance of the benchmark mean under design (V, M, R=1)
dstudy_var <- function(V, M, N = N_ITEMS, R = 1) {
  sigma2_rho / V +
    sigma2_lambda / M +
    sigma2_ap / (N * V) +
    sigma2_al / (N * M) +
    sigma2_pl / (V * M) +
    sigma2_eps / (N * V * M * R)
}

# =========================================================================
# Simulation (parallelized over grid)
# =========================================================================

csv_path <- file.path(out_dir, "sim_gaming_safety.csv")

if (file.exists(csv_path)) {
  cat("Loading cached results from", csv_path, "\n")
  results <- read_csv(csv_path, show_col_types = FALSE)
} else {
  cat("Running simulation (K_TRIES =", K_TRIES, ")...\n")

  grid <- expand.grid(V = V_LEVELS, M = M_LEVELS)
  grid$sigma_pipeline <- sqrt(mapply(dstudy_var, grid$V, grid$M))

  N_CORES <- max(1, parallel::detectCores() - 1)
  cat(sprintf("Parallel workers: %d\n", N_CORES))

  run_one <- function(g) {
    sig <- grid$sigma_pipeline[g]
    max_scores <- replicate(N_SIM, max(rnorm(K_TRIES, 0, sig)))
    data.frame(
      V               = grid$V[g],
      M               = grid$M[g],
      sigma_pipeline  = sig,
      mean_gaming_adv = mean(max_scores),
      sd_gaming_adv   = sd(max_scores),
      K_TRIES         = K_TRIES
    )
  }

  results_list <- parallel::mclapply(seq_len(nrow(grid)), run_one,
                                      mc.cores = N_CORES)
  results <- bind_rows(results_list)
  write_csv(results, csv_path)
  cat("Saved:", csv_path, "\n")
}

# =========================================================================
# Summarise
# =========================================================================

baseline_adv <- results %>% filter(V == 1, M == 1) %>% pull(mean_gaming_adv)

results <- results %>%
  mutate(
    pct_eliminated = (baseline_adv - mean_gaming_adv) / baseline_adv * 100,
    M_label = paste0("M = ", M, " judge", ifelse(M == 1, "", "s"))
  )

cat("\n--- Gaming advantage summary (K=", K_TRIES, ", safety benchmark) ---\n")
print(results %>% select(V, M, sigma_pipeline, mean_gaming_adv, pct_eliminated))

# =========================================================================
# Figure
# =========================================================================

theme_set(theme_bw(base_size = 11))
palette <- c("M = 1 judge"  = "#d73027",
             "M = 2 judges" = "#fc8d59",
             "M = 3 judges" = "#4575b4")

p <- ggplot(results, aes(x = V, y = mean_gaming_adv,
                          colour = M_label, group = M_label)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_x_continuous(breaks = V_LEVELS) +
  labs(
    x = "Number of prompt variants averaged (V)",
    y = "Expected gaming advantage\n(safety score units above true mean)"
  ) +
  theme(legend.position = c(0.72, 0.85),
        legend.background = element_rect(fill = "white", colour = "grey80"))

pdf_path <- file.path(fig_dir, "fig_gaming_surface_safety.pdf")
png_path <- file.path(fig_dir, "fig_gaming_surface_safety.png")

ggsave(pdf_path, p, width = 4, height = 3.5)
ggsave(png_path, p, width = 4, height = 3.5, dpi = 150)

cat("Saved:", pdf_path, "\n")

# =========================================================================
# Key numbers
# =========================================================================

v1m1 <- filter(results, V == 1, M == 1)
v1m3 <- filter(results, V == 1, M == 3)
v5m1 <- filter(results, V == 5, M == 1)
v5m3 <- filter(results, V == 5, M == 3)

cat("\n=== Key numbers (safety benchmark) ===\n")
cat(sprintf("V=1, M=1: sigma=%.4f, gaming_adv=%.4f (baseline)\n",
            v1m1$sigma_pipeline, v1m1$mean_gaming_adv))
cat(sprintf("V=1, M=3: sigma=%.4f, gaming_adv=%.4f  (%.1f%% eliminated)\n",
            v1m3$sigma_pipeline, v1m3$mean_gaming_adv, v1m3$pct_eliminated))
cat(sprintf("V=5, M=1: sigma=%.4f, gaming_adv=%.4f  (%.1f%% eliminated)\n",
            v5m1$sigma_pipeline, v5m1$mean_gaming_adv, v5m1$pct_eliminated))
cat(sprintf("V=5, M=3: sigma=%.4f, gaming_adv=%.4f  (%.1f%% eliminated)\n",
            v5m3$sigma_pipeline, v5m3$mean_gaming_adv, v5m3$pct_eliminated))

cat(sprintf("\nVariance share at V=1, M=1:\n"))
total <- dstudy_var(1, 1)
cat(sprintf("  judge main:     %.1f%%\n", sigma2_lambda / total * 100))
cat(sprintf("  item x judge:   %.1f%%\n", sigma2_al / N_ITEMS / total * 100))
cat(sprintf("  prompt x judge: %.1f%%\n", sigma2_pl / total * 100))
cat(sprintf("  residual:       %.1f%%\n", sigma2_eps / N_ITEMS / total * 100))
cat(sprintf("  prompt main:    %.1f%%\n", sigma2_rho / total * 100))
cat(sprintf("  item x prompt:  %.1f%%\n", sigma2_ap / N_ITEMS / total * 100))
