# 13_sim_gaming.R
# Gaming Surface Simulation: How TEE Designs Shrink the Exploitable Score Surface
#
# Model: A developer submits K_TRIES model variants to a benchmark. The
# benchmark evaluates each variant using V prompt variants averaged over M
# judge models (TEE design). The gaming advantage is the expected score
# inflation from submitting K_TRIES variants and reporting the best, driven
# by measurement noise that the developer can exploit.
#
# When the benchmark averages over V*M configurations, the per-evaluation
# standard error sigma_pipeline(V, M) falls, reducing the gaming advantage.
#
# The gaming advantage for K_TRIES variants at design (V, M) is:
#   E[max_{k=1..K} X_k] - truth  where  X_k ~ N(truth, sigma2_pipeline(V,M))
#   = sigma_pipeline(V,M) * E[max of K_TRIES standard normals]
#
# sigma2_pipeline(V, M) = sigma2_rho/V + sigma2_lambda/M
#                        + sigma2_ap/(N*V) + sigma2_al/(N*M) + sigma2_eps/(N*V)
#
# Variance components loaded from data/processed/variance_components_likert.csv
# (ideology Likert decomposition, N = 150 items).
#
# Usage:
#   Rscript analysis/13_sim_gaming.R               # default N_SIM = 1000
#   Rscript analysis/13_sim_gaming.R --nsim 5       # smoke test (~5 sec)

library(tidyverse)

set.seed(42)

# --- Command-line arguments ---
args  <- commandArgs(trailingOnly = TRUE)
N_SIM <- 1000L
if ("--nsim" %in% args) {
  idx  <- which(args == "--nsim")
  if (idx < length(args)) N_SIM <- as.integer(args[idx + 1])
}

fig_dir <- "figures"
out_dir <- "data/processed"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 13_sim_gaming.R: Gaming Surface Simulation ===\n")
cat(sprintf("N_SIM = %d\n", N_SIM))

# =========================================================================
# Section 1: Load Empirical Variance Components
# =========================================================================

vc_path <- file.path(out_dir, "variance_components_likert.csv")
if (!file.exists(vc_path)) stop("Missing: ", vc_path)

vc_raw <- read_csv(vc_path, show_col_types = FALSE)

pull_vc <- function(label) {
  row <- filter(vc_raw, tle_label == label)
  if (nrow(row) == 0) stop("Component not found: ", label)
  row$variance[1]
}

sigma2_rho    <- pull_vc("prompt")                           # prompt main effect
sigma2_lambda <- pull_vc("judge model (design sensitivity)") # judge main effect
sigma2_ap     <- pull_vc("item x prompt")                    # item x prompt
sigma2_al     <- pull_vc("item x judge")                     # item x judge
sigma2_eps    <- pull_vc("generation")                       # residual

N_ITEMS <- 150L  # ideology full-run design

cat(sprintf("sigma2_rho    = %.5f  (prompt main effect)\n",    sigma2_rho))
cat(sprintf("sigma2_lambda = %.5f  (judge main effect)\n",     sigma2_lambda))
cat(sprintf("sigma2_ap     = %.5f  (item x prompt)\n",         sigma2_ap))
cat(sprintf("sigma2_al     = %.5f  (item x judge)\n",          sigma2_al))
cat(sprintf("sigma2_eps    = %.5f  (residual)\n",              sigma2_eps))
cat(sprintf("N_ITEMS = %d\n", N_ITEMS))

# =========================================================================
# Section 2: Design grid and helper
# =========================================================================

V_LEVELS  <- 1:8
M_LEVELS  <- c(1L, 3L)
K_TRIES   <- 10L  # number of model variants a developer can test

# D-study SE of the benchmark-reported mean under (V, M):
# averages over V prompts, M judges, N items, 1 replication
dstudy_var <- function(V, M, N = N_ITEMS) {
  sigma2_rho / V +
  sigma2_lambda / M +
  sigma2_ap / (N * V) +
  sigma2_al / (N * M) +
  sigma2_eps / (N * V)
}

# =========================================================================
# Section 3: Simulation
# =========================================================================

csv_path <- file.path(out_dir, "sim_gaming.csv")

if (file.exists(csv_path)) {
  cat("Loading cached results from", csv_path, "\n")
  results <- read_csv(csv_path, show_col_types = FALSE)
} else {
  cat("Running simulation (K_TRIES =", K_TRIES, ")...\n")

  grid <- expand.grid(V = V_LEVELS, M = M_LEVELS)
  grid$sigma_pipeline <- sqrt(mapply(dstudy_var, grid$V, grid$M))

  results_list <- vector("list", nrow(grid))

  for (g in seq_len(nrow(grid))) {
    V   <- grid$V[g]
    M   <- grid$M[g]
    sig <- grid$sigma_pipeline[g]

    # Each simulation rep: draw K_TRIES scores ~ N(0, sigma^2), take max
    max_scores <- replicate(N_SIM, {
      scores <- rnorm(K_TRIES, mean = 0, sd = sig)
      max(scores)
    })

    results_list[[g]] <- data.frame(
      V               = V,
      M               = M,
      sigma_pipeline  = sig,
      mean_gaming_adv = mean(max_scores),
      sd_gaming_adv   = sd(max_scores),
      K_TRIES         = K_TRIES
    )
    cat(sprintf("  V=%d, M=%d: sigma=%.4f, gaming_adv=%.4f\n",
                V, M, sig, mean(max_scores)))
  }

  results <- bind_rows(results_list)
  write_csv(results, csv_path)
  cat("Saved:", csv_path, "\n")
}

# =========================================================================
# Section 4: Summarise
# =========================================================================

# Baseline: V=1, M=1
baseline_adv <- results %>% filter(V == 1, M == 1) %>% pull(mean_gaming_adv)

results <- results %>%
  mutate(
    pct_eliminated = (baseline_adv - mean_gaming_adv) / baseline_adv * 100,
    M_label = paste0("M = ", M, " judge", ifelse(M == 1, "", "s"))
  )

cat("\n--- Gaming advantage summary (K_TRIES =", K_TRIES, ") ---\n")
print(results %>% select(V, M, sigma_pipeline, mean_gaming_adv, pct_eliminated))

# =========================================================================
# Section 5: Figures
# =========================================================================

theme_set(theme_bw(base_size = 11))
palette <- c("M = 1 judge" = "#d73027", "M = 3 judges" = "#4575b4")

# Panel A: Gaming advantage (absolute) vs V
p_a <- ggplot(results, aes(x = V, y = mean_gaming_adv,
                            colour = M_label, group = M_label)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  scale_colour_manual(values = palette, name = NULL) +
  scale_x_continuous(breaks = V_LEVELS) +
  labs(
    x = "Number of prompt variants averaged (V)",
    y = "Expected gaming advantage\n(score units above true mean)",
    title = "A. Absolute gaming advantage"
  ) +
  theme(legend.position = c(0.72, 0.85),
        legend.background = element_rect(fill = "white", colour = "grey80"))

p_a <- p_a + labs(title = NULL)
p_combined <- p_a

# Not the paper figure: fig_gaming_surface.{pdf,png} (Figure 4) comes from 13c_sim_gaming_arena.R.
pdf_path <- file.path(fig_dir, "fig_gaming_surface_ideology.pdf")
png_path <- file.path(fig_dir, "fig_gaming_surface_ideology.png")

ggsave(pdf_path, p_combined, width = 4, height = 3.5)
ggsave(png_path, p_combined, width = 4, height = 3.5, dpi = 150)

cat("Saved:", pdf_path, "\n")
cat("Saved:", png_path, "\n")

# =========================================================================
# Section 6: Print key numbers for manuscript
# =========================================================================

v1m1 <- filter(results, V == 1, M == 1)
v1m3 <- filter(results, V == 1, M == 3)
v5m1 <- filter(results, V == 5, M == 1)
v5m3 <- filter(results, V == 5, M == 3)
v8m3 <- filter(results, V == 8, M == 3)

cat("\n=== Key numbers for manuscript ===\n")
cat(sprintf("V=1, M=1: sigma=%.4f, gaming_adv=%.4f (baseline, 0%% eliminated)\n",
            v1m1$sigma_pipeline, v1m1$mean_gaming_adv))
cat(sprintf("V=1, M=3: sigma=%.4f, gaming_adv=%.4f  (%.1f%% eliminated)\n",
            v1m3$sigma_pipeline, v1m3$mean_gaming_adv, v1m3$pct_eliminated))
cat(sprintf("V=5, M=1: sigma=%.4f, gaming_adv=%.4f  (%.1f%% eliminated)\n",
            v5m1$sigma_pipeline, v5m1$mean_gaming_adv, v5m1$pct_eliminated))
cat(sprintf("V=5, M=3: sigma=%.4f, gaming_adv=%.4f  (%.1f%% eliminated)\n",
            v5m3$sigma_pipeline, v5m3$mean_gaming_adv, v5m3$pct_eliminated))
cat(sprintf("V=8, M=3: sigma=%.4f, gaming_adv=%.4f  (%.1f%% eliminated)\n",
            v8m3$sigma_pipeline, v8m3$mean_gaming_adv, v8m3$pct_eliminated))
cat(sprintf("\nJudge main effect share of gaming variance at V=1, M=1: %.1f%%\n",
            sigma2_lambda / dstudy_var(1,1) * 100))
cat(sprintf("Item x judge interaction share of gaming variance at V=1, M=1: %.1f%%\n",
            sigma2_al / N_ITEMS / dstudy_var(1,1) * 100))
