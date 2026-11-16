# 02_variance_decomposition.R
# Fit crossed random effects model and extract TEE variance components.
# Supports both Likert and Pairwise scoring methods.
#
# Usage:
#   Rscript analysis/02_variance_decomposition.R [scoring]
#   e.g.: Rscript analysis/02_variance_decomposition.R likert

library(tidyverse)
library(lme4)

set.seed(42)

# --- Config ---
args <- commandArgs(trailingOnly = TRUE)
scoring <- if (length(args) >= 1) args[1] else "likert"

input_path     <- sprintf("data/processed/%s_clean.csv", scoring)
vc_output_path <- sprintf("data/processed/variance_components_%s.csv", scoring)
ds_output_path <- sprintf("data/processed/dstudy_%s.csv", scoring)

cat("=== 02_variance_decomposition.R ===\n")
cat("Scoring:", scoring, "\n")

# --- Load data ---
df <- read_csv(input_path, show_col_types = FALSE) %>%
  filter(!is.na(outcome)) %>%
  mutate(
    item_id    = as.factor(item_id),
    variant_id = as.factor(variant_id),
    temperature = as.factor(temperature),
    judge_model = as.factor(judge_model),
    category   = as.factor(category)
  )

cat("Data:", nrow(df), "rows,", n_distinct(df$item_id), "items,",
    n_distinct(df$category), "categories\n")

# --- Fit crossed random effects model ---
# Fixed effects: temperature, judge_model (Tier 2 design choices)
#   For pairwise: also true_order (position bias covariate)
# Random effects: category, item, prompt variant, and two-way interactions
# Matches paper DGP (Eq. 1): Y_{ijkm}^{(r)} = mu + gamma_{c(i)} + delta_{i|c}
#   + beta_j + tau_k + phi_m + (alpha*beta)_{ij} + (alpha*tau)_{ik}
#   + (beta*tau)_{jk} + (alpha*phi)_{im} + (beta*phi)_{jm} + epsilon
cat("\nFitting lmer model...\n")

has_position <- "true_order" %in% names(df)
if (has_position) {
  df$true_order <- as.factor(df$true_order)
  cat("  Including true_order as fixed effect (pairwise position bias)\n")
}

fixed_formula <- if (has_position) {
  outcome ~ temperature + judge_model + true_order +
    (1 | category) +
    (1 | item_id) +
    (1 | variant_id) +
    (1 | item_id:variant_id) +
    (1 | item_id:temperature) +
    (1 | variant_id:temperature) +
    (1 | item_id:judge_model) +
    (1 | variant_id:judge_model) +
    (1 | item_id:variant_id:judge_model:temperature)  # cell-level (3-way+)
} else {
  outcome ~ temperature + judge_model +
    (1 | category) +
    (1 | item_id) +
    (1 | variant_id) +
    (1 | item_id:variant_id) +
    (1 | item_id:temperature) +
    (1 | variant_id:temperature) +
    (1 | item_id:judge_model) +
    (1 | variant_id:judge_model) +
    (1 | item_id:variant_id:judge_model:temperature)  # cell-level (3-way+)
}

mod <- lmer(
  fixed_formula,
  data = df,
  REML = TRUE,
  control = lmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 50000)
  )
)

cat("\nModel summary:\n")
print(summary(mod))

# --- Check convergence ---
if (any(grepl("singular", mod@optinfo$conv$lme4$messages))) {
  warning("Singular fit detected — some variance components near zero.")
}

# --- Extract variance components ---
vc <- as.data.frame(VarCorr(mod))
cat("\nVariance components:\n")
print(vc)

# Map to TEE labels
tle_labels <- c(
  "category"                = "between-category",
  "item_id"                 = "within-category item",
  "variant_id"              = "prompt",
  "item_id:variant_id"      = "item x prompt",
  "item_id:temperature"     = "item x temperature",
  "variant_id:temperature"  = "prompt x temperature",
  "item_id:judge_model"     = "item x judge",
  "variant_id:judge_model"  = "prompt x judge",
  "item_id:variant_id:judge_model:temperature" = "cell-level (3-way+)",
  "Residual"                = "replicate noise"
)

vc_clean <- vc %>%
  select(grp, vcov, sdcor) %>%
  rename(component = grp, variance = vcov, sd = sdcor) %>%
  mutate(
    tle_label = tle_labels[component],
    tle_label = ifelse(is.na(tle_label), component, tle_label)
  )

# Add temperature fixed effect sensitivity index
# Reconstruct L cell means (intercept + each contrast), then compute
# population variance (1/L * sum of squared deviations from grand mean).
# This is the quantity that enters the EMS, not sample variance (1/(L-1)).
temp_contrasts <- fixef(mod)[grepl("temperature", names(fixef(mod)))]
grand_mean <- fixef(mod)["(Intercept)"]
temp_cell_means <- c(grand_mean, grand_mean + temp_contrasts)
sigma2_temp <- sum((temp_cell_means - mean(temp_cell_means))^2) / length(temp_cell_means)

vc_clean <- bind_rows(
  vc_clean,
  tibble(
    component = "temperature (fixed)",
    variance = sigma2_temp,
    sd = sqrt(sigma2_temp),
    tle_label = "temperature (design sensitivity)"
  )
)

# Add judge model fixed effect sensitivity index (same population variance)
judge_contrasts <- fixef(mod)[grepl("judge_model", names(fixef(mod)))]
if (length(judge_contrasts) > 0) {
  judge_cell_means <- c(grand_mean, grand_mean + judge_contrasts)
  sigma2_judge <- sum((judge_cell_means - mean(judge_cell_means))^2) / length(judge_cell_means)
  vc_clean <- bind_rows(
    vc_clean,
    tibble(
      component = "judge_model (fixed)",
      variance = sigma2_judge,
      sd = sqrt(sigma2_judge),
      tle_label = "judge model (design sensitivity)"
    )
  )
}

# Add position (true_order) fixed effect sensitivity index (pairwise only)
if (has_position) {
  pos_contrasts <- fixef(mod)[grepl("true_order", names(fixef(mod)))]
  pos_cell_means <- c(grand_mean, grand_mean + pos_contrasts)
  sigma2_position <- sum((pos_cell_means - mean(pos_cell_means))^2) / length(pos_cell_means)
  vc_clean <- bind_rows(
    vc_clean,
    tibble(
      component = "true_order (fixed)",
      variance = sigma2_position,
      sd = sqrt(sigma2_position),
      tle_label = "position (design sensitivity)"
    )
  )
  cat("\nPosition sensitivity index (sigma2_position):", sigma2_position, "\n")
}

# Clamp negative REML estimates before computing proportions
vc_clean <- vc_clean %>%
  mutate(variance = pmax(variance, 0))
total_var <- sum(vc_clean$variance)
vc_clean <- vc_clean %>%
  mutate(
    pct_total = round(100 * variance / total_var, 2),
    tier = case_when(
      tle_label %in% c("replicate noise", "cell-level (3-way+)",
                        "between-category", "within-category item",
                        "prompt", "item x prompt", "item x temperature",
                        "prompt x temperature", "item x judge",
                        "prompt x judge") ~ "model-side",
      TRUE ~ "pipeline-side"
    ),
    scoring = scoring
  ) %>%
  arrange(desc(variance))

cat("\n=== TEE Variance Components ===\n")
print(vc_clean %>% select(tle_label, variance, pct_total, tier))

# Sanity check: cell-level vs replicate noise split
cat(sprintf(
  "\nResidual split sanity check:\n  s2_eps_cell (cell-level) = %.5f (%.2f%% of total)\n  s2_rho_rep  (replicate) = %.5f (%.2f%% of total)\n",
  vc_clean %>% filter(tle_label == "cell-level (3-way+)") %>% pull(variance),
  vc_clean %>% filter(tle_label == "cell-level (3-way+)") %>% pull(pct_total),
  vc_clean %>% filter(tle_label == "replicate noise") %>% pull(variance),
  vc_clean %>% filter(tle_label == "replicate noise") %>% pull(pct_total)
))

# --- Save variance components ---
dir.create(dirname(vc_output_path), recursive = TRUE, showWarnings = FALSE)
write_csv(vc_clean, vc_output_path)
cat("\nVariance components saved to:", vc_output_path, "\n")


# --- D-Study Projections ---
cat("\n=== D-Study Projections ===\n")

# Extract key variance components
s2_cat    <- vc_clean %>% filter(tle_label == "between-category") %>% pull(variance)
s2_item   <- vc_clean %>% filter(tle_label == "within-category item") %>% pull(variance)
s2_prompt <- vc_clean %>% filter(tle_label == "prompt") %>% pull(variance)
s2_ip     <- vc_clean %>% filter(tle_label == "item x prompt") %>% pull(variance)
s2_it     <- vc_clean %>% filter(tle_label == "item x temperature") %>% pull(variance)
s2_pt     <- vc_clean %>% filter(tle_label == "prompt x temperature") %>% pull(variance)
s2_ij     <- vc_clean %>% filter(tle_label == "item x judge") %>% pull(variance)
s2_pj     <- vc_clean %>% filter(tle_label == "prompt x judge") %>% pull(variance)
s2_temp   <- vc_clean %>% filter(tle_label == "temperature (design sensitivity)") %>% pull(variance)
s2_judge  <- vc_clean %>% filter(tle_label == "judge model (design sensitivity)") %>% pull(variance)
s2_temp   <- if (length(s2_temp) == 0) 0 else s2_temp
s2_judge  <- if (length(s2_judge) == 0) 0 else s2_judge
# Two residual components: cell-level (3-way+) and within-cell replicate.
# s2_eps_cell is constant within a cell, varies across cells -> NOT reducible by R.
# s2_rho_rep is call-to-call sampling at fixed cell -> reducible by R via averaging.
s2_eps_cell <- vc_clean %>% filter(tle_label == "cell-level (3-way+)") %>% pull(variance)
s2_rho_rep  <- vc_clean %>% filter(tle_label == "replicate noise")    %>% pull(variance)

# Current design parameters
n_cats    <- n_distinct(df$category)
n_items   <- n_distinct(df$item_id)
n_prompts <- n_distinct(df$variant_id)
n_temps   <- n_distinct(df$temperature)
n_judges  <- n_distinct(df$judge_model)
n_reps    <- max(df$replication) + 1

# D-study helpers
# Case 1: Fixed temperature, fixed judge model
# When judge is fixed, item×judge (s2_ij) still varies across items → averages at 1/N.
# When temp is fixed, item×temp (s2_it) still varies across items → averages at 1/N.
# Same logic for prompt×judge (s2_pj/K) and prompt×temp (s2_pt/K).
dstudy_fixed <- function(C, N, K, R) {
  # Categories are fixed (taxonomy-level), so s2_cat absorbs into s2_item
  # and both scale with N (paper footnote, pnas_main.tex Materials & Methods).
  (s2_cat + s2_item)/N + s2_prompt/K + s2_temp + s2_judge +
    s2_ip/(N * K) +
    s2_ij/N + s2_pj/K +   # judge interaction terms (fixed judge)
    s2_it/N + s2_pt/K +   # temp interaction terms (fixed temp)
    s2_eps_cell/(N * K) +        # cell-level, no /R
    s2_rho_rep/(N * K * R)       # replicate, with /R
}

# Case 2: Averaged over L temperatures and M judge models
dstudy_avg <- function(C, N, K, R, L = n_temps, M = n_judges) {
  (s2_cat + s2_item)/N + s2_prompt/K + s2_temp/L + s2_judge/M +
    s2_ip/(N * K) +
    s2_it/(N * L) + s2_pt/(K * L) +
    s2_ij/(N * M) + s2_pj/(K * M) +
    s2_eps_cell/(N * K * L * M) +        # cell-level, no /R
    s2_rho_rep/(N * K * L * M * R)       # replicate, with /R
}

baseline_var <- dstudy_avg(n_cats, n_items, n_prompts, n_reps)

# D-study scenarios
scenarios <- tribble(
  ~scenario,                                               ~n_cats_d, ~n_items_d, ~n_prompts_d, ~n_reps_d, ~fix_temp, ~fix_judge,
  "Baseline (avg temp & judge)",                           n_cats,    n_items,    n_prompts,    n_reps,    FALSE,     FALSE,
  "Baseline (fix temp & judge)",                           n_cats,    n_items,    n_prompts,    n_reps,    TRUE,      TRUE,
  "+5 replications",                                       n_cats,    n_items,    n_prompts,    n_reps+5,  FALSE,     FALSE,
  "+2 prompt variants",                                    n_cats,    n_items,    n_prompts+2,  n_reps,    FALSE,     FALSE,
  "Fix temperature",                                       n_cats,    n_items,    n_prompts,    n_reps,    TRUE,      FALSE,
  "Fix judge model",                                       n_cats,    n_items,    n_prompts,    n_reps,    FALSE,     TRUE,
  "Fix temp & judge",                                      n_cats,    n_items,    n_prompts,    n_reps,    TRUE,      TRUE,
  "+5 reps & fix temp & judge",                            n_cats,    n_items,    n_prompts,    n_reps+5,  TRUE,      TRUE,
  "Double items",                                          n_cats,    n_items*2,  n_prompts,    n_reps,    FALSE,     FALSE,
  "Maximal: 2x items + 5 prompts + 10 reps + fix design", n_cats,    n_items*2,  5,            10,        TRUE,      TRUE,
)

dstudy <- scenarios %>%
  rowwise() %>%
  mutate(
    total_var = if (fix_temp & fix_judge) {
      dstudy_fixed(n_cats_d, n_items_d, n_prompts_d, n_reps_d)
    } else if (fix_temp & !fix_judge) {
      # Fixed temp, avg over judges
      # Temp is fixed → s2_it/N + s2_pt/K remain (fixed temp interaction)
      # Judge is averaged → s2_ij/(N*M) + s2_pj/(K*M)
      # Cell-level averages over N*K*M cells.
      (s2_cat + s2_item)/n_items_d + s2_prompt/n_prompts_d +
        s2_temp + s2_judge/n_judges +
        s2_ip/(n_items_d * n_prompts_d) +
        s2_it/n_items_d + s2_pt/n_prompts_d +                          # fixed temp
        s2_ij/(n_items_d * n_judges) + s2_pj/(n_prompts_d * n_judges) + # avg judge
        s2_eps_cell/(n_items_d * n_prompts_d * n_judges) +
        s2_rho_rep/(n_items_d * n_prompts_d * n_judges * n_reps_d)
    } else if (!fix_temp & fix_judge) {
      # Avg over temps, fixed judge
      # Judge is fixed → s2_ij/N + s2_pj/K remain (fixed judge interaction)
      # Temp is averaged → s2_it/(N*L) + s2_pt/(K*L)
      # Cell-level averages over N*K*L cells.
      (s2_cat + s2_item)/n_items_d + s2_prompt/n_prompts_d +
        s2_temp/n_temps + s2_judge +
        s2_ip/(n_items_d * n_prompts_d) +
        s2_ij/n_items_d + s2_pj/n_prompts_d +                          # fixed judge
        s2_it/(n_items_d * n_temps) + s2_pt/(n_prompts_d * n_temps) +   # avg temp
        s2_eps_cell/(n_items_d * n_prompts_d * n_temps) +
        s2_rho_rep/(n_items_d * n_prompts_d * n_temps * n_reps_d)
    } else {
      dstudy_avg(n_cats_d, n_items_d, n_prompts_d, n_reps_d)
    }
  ) %>%
  ungroup() %>%
  mutate(
    pct_of_baseline = round(100 * total_var / baseline_var, 1),
    reduction_pct = round(100 * (1 - total_var / baseline_var), 1),
    scoring = scoring
  )

cat("\nD-Study Results:\n")
print(dstudy %>% select(scenario, total_var, pct_of_baseline, reduction_pct))

write_csv(dstudy, ds_output_path)
cat("\nD-study projections saved to:", ds_output_path, "\n")

# --- Per-category generation variance ---
cat("\n=== Per-Category Generation Variance ===\n")

topic_var <- df %>%
  group_by(category, item_id, variant_id, temperature, judge_model) %>%
  summarize(cell_var = var(outcome, na.rm = TRUE), .groups = "drop") %>%
  group_by(category) %>%
  summarize(
    mean_gen_var = mean(cell_var, na.rm = TRUE),
    sd_gen_var = sd(cell_var, na.rm = TRUE),
    n_cells = n(),
    .groups = "drop"
  ) %>%
  mutate(scoring = scoring) %>%
  arrange(desc(mean_gen_var))

cat("\nPer-category generation variance:\n")
print(topic_var)

topic_output_path <- sprintf("data/processed/per_category_variance_%s.csv", scoring)
write_csv(topic_var, topic_output_path)
cat("\nPer-category variance saved to:", topic_output_path, "\n")

# --- Bootstrap CIs for variance components ---
# Parametric bootstrap on random effect SDs; square bounds for variance CIs.
# nsim=200 is a smoke test (~1-3 hours for 54K obs); increase to 1000 for publication.
# Skip if CI file already exists (delete to recompute).
ci_output_path <- sprintf("data/processed/variance_ci_%s.csv", scoring)

boot_nsim <- as.integer(Sys.getenv("TEE_BOOT_NSIM", "200"))
boot_skip <- Sys.getenv("TEE_BOOT_SKIP", "0") == "1"

if (boot_skip) {
  cat("\nTEE_BOOT_SKIP=1: skipping bootstrap CIs (stale file, if any, ignored).\n")
} else if (!file.exists(ci_output_path)) {
  cat(sprintf("\nComputing bootstrap CIs (nsim=%d)...\n", boot_nsim))
  tryCatch({
    # parallel="snow" is cross-platform (multicore fails on Windows).
    # Override via TEE_BOOT_PARALLEL if needed (e.g. "no" to disable).
    boot_par <- Sys.getenv("TEE_BOOT_PARALLEL", "snow")
    boot_ncpus <- max(1, parallel::detectCores() - 1)
    boot_ci <- confint(mod, method = "boot", nsim = boot_nsim,
                       oldNames = FALSE, parallel = boot_par,
                       ncpus = boot_ncpus)
    cat("Bootstrap CIs:\n")
    print(boot_ci)

    # Extract random effect rows (on SD scale) and square for variance CIs
    re_names <- rownames(boot_ci)[grepl("^sd_|^sigma$", rownames(boot_ci))]
    ci_df <- tibble(
      ci_name = re_names,
      ci_lower_sd = boot_ci[re_names, 1],
      ci_upper_sd = boot_ci[re_names, 2]
    ) %>%
      mutate(
        ci_lower_var = ci_lower_sd^2,
        ci_upper_var = ci_upper_sd^2,
        # Map confint names to vc_clean component names
        component = case_when(
          ci_name == "sd_category.(Intercept)"               ~ "category",
          ci_name == "sd_item_id.(Intercept)"                ~ "item_id",
          ci_name == "sd_variant_id.(Intercept)"             ~ "variant_id",
          ci_name == "sd_item_id:variant_id.(Intercept)"     ~ "item_id:variant_id",
          ci_name == "sd_item_id:temperature.(Intercept)"    ~ "item_id:temperature",
          ci_name == "sd_variant_id:temperature.(Intercept)" ~ "variant_id:temperature",
          ci_name == "sd_item_id:judge_model.(Intercept)"    ~ "item_id:judge_model",
          ci_name == "sd_variant_id:judge_model.(Intercept)" ~ "variant_id:judge_model",
          ci_name == "sd_item_id:variant_id:judge_model:temperature.(Intercept)" ~
            "item_id:variant_id:judge_model:temperature",
          ci_name == "sigma"                                 ~ "Residual",
          TRUE ~ ci_name
        )
      ) %>%
      select(component, ci_lower_var, ci_upper_var)

    write_csv(ci_df, ci_output_path)
    cat("Bootstrap CIs saved to:", ci_output_path, "\n")
  }, error = function(e) {
    warning("Bootstrap CIs failed: ", e$message)
  })
} else {
  cat("\nBootstrap CI file exists:", ci_output_path, "(delete to recompute)\n")
}

# --- Merge CIs into variance components if available ---
if (!boot_skip && file.exists(ci_output_path)) {
  ci_df <- read_csv(ci_output_path, show_col_types = FALSE)
  vc_clean <- vc_clean %>%
    left_join(ci_df, by = "component")
  # Re-save with CIs
  write_csv(vc_clean, vc_output_path)
  cat("Variance components updated with CIs.\n")
}
