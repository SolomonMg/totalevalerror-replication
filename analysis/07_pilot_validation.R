# 07_pilot_validation.R
# Validate whether a pilot subset (N=30 items, V=3 variants) can predict
# the full-run variance profile via D-study projections.
#
# Approach:
#   1. Subset full data to pilot-sized sample (30 items, 3 variants)
#   2. Fit same lmer model as 02_variance_decomposition.R
#   3. Use pilot variance components to project to full design (N=150, V=5)
#   4. Compare projected vs actual full-run variance components
#
# Usage:
#   Rscript analysis/07_pilot_validation.R

library(tidyverse)
library(lme4)

set.seed(42)

# --- Config ---
N_PILOT  <- 30   # items in pilot
V_PILOT  <- 3    # prompt variants in pilot
N_FULL   <- 150  # items in full run
V_FULL   <- 5    # prompt variants in full run

tle_labels <- c(
  "category"                = "between-category",
  "item_id"                 = "within-category item",
  "variant_id"              = "prompt",
  "item_id:variant_id"      = "item x prompt",
  "item_id:temperature"     = "item x temperature",
  "variant_id:temperature"  = "prompt x temperature",
  "item_id:judge_model"     = "item x judge",
  "variant_id:judge_model"  = "prompt x judge",
  "Residual"                = "generation"
)

cat("=== 07_pilot_validation.R ===\n")

# --- Helper: fit model and extract variance components ---
fit_and_extract <- function(df, scoring_type) {
  has_position <- "true_order" %in% names(df)
  if (has_position) df$true_order <- as.factor(df$true_order)

  fixed_formula <- if (has_position) {
    outcome ~ temperature + judge_model + true_order +
      (1 | category) +
      (1 | item_id) +
      (1 | variant_id) +
      (1 | item_id:variant_id) +
      (1 | item_id:temperature) +
      (1 | variant_id:temperature) +
      (1 | item_id:judge_model) +
      (1 | variant_id:judge_model)
  } else {
    outcome ~ temperature + judge_model +
      (1 | category) +
      (1 | item_id) +
      (1 | variant_id) +
      (1 | item_id:variant_id) +
      (1 | item_id:temperature) +
      (1 | variant_id:temperature) +
      (1 | item_id:judge_model) +
      (1 | variant_id:judge_model)
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

  vc <- as.data.frame(VarCorr(mod)) %>%
    select(grp, vcov, sdcor) %>%
    rename(component = grp, variance = vcov, sd = sdcor) %>%
    mutate(
      tle_label = tle_labels[component],
      tle_label = ifelse(is.na(tle_label), component, tle_label),
      variance = pmax(variance, 0)
    )

  # Temperature fixed effect sensitivity
  temp_contrasts <- fixef(mod)[grepl("temperature", names(fixef(mod)))]
  grand_mean <- fixef(mod)["(Intercept)"]
  temp_cell_means <- c(grand_mean, grand_mean + temp_contrasts)
  sigma2_temp <- sum((temp_cell_means - mean(temp_cell_means))^2) / length(temp_cell_means)

  vc <- bind_rows(vc, tibble(
    component = "temperature (fixed)",
    variance = sigma2_temp,
    sd = sqrt(sigma2_temp),
    tle_label = "temperature (design sensitivity)"
  ))

  # Judge model fixed effect sensitivity
  judge_contrasts <- fixef(mod)[grepl("judge_model", names(fixef(mod)))]
  if (length(judge_contrasts) > 0) {
    judge_cell_means <- c(grand_mean, grand_mean + judge_contrasts)
    sigma2_judge <- sum((judge_cell_means - mean(judge_cell_means))^2) / length(judge_cell_means)
    vc <- bind_rows(vc, tibble(
      component = "judge_model (fixed)",
      variance = sigma2_judge,
      sd = sqrt(sigma2_judge),
      tle_label = "judge model (design sensitivity)"
    ))
  }

  # Position fixed effect sensitivity (pairwise only)
  if (has_position) {
    pos_contrasts <- fixef(mod)[grepl("true_order", names(fixef(mod)))]
    pos_cell_means <- c(grand_mean, grand_mean + pos_contrasts)
    sigma2_position <- sum((pos_cell_means - mean(pos_cell_means))^2) / length(pos_cell_means)
    vc <- bind_rows(vc, tibble(
      component = "true_order (fixed)",
      variance = sigma2_position,
      sd = sqrt(sigma2_position),
      tle_label = "position (design sensitivity)"
    ))
  }

  vc
}

# --- Helper: extract named variance from vc tibble ---
get_var <- function(vc, label) {
  val <- vc %>% filter(tle_label == label) %>% pull(variance)
  if (length(val) == 0) 0 else val
}

# --- Helper: D-study projection (avg over temps and judges) ---
# Matches dstudy_avg() from 02_variance_decomposition.R.
# C = number of categories (fixed at 5 for both pilot and full).
dstudy_project <- function(vc, C, N, K, R, L, M) {
  s2_cat  <- get_var(vc, "between-category")
  s2_item <- get_var(vc, "within-category item")
  s2_prom <- get_var(vc, "prompt")
  s2_ip   <- get_var(vc, "item x prompt")
  s2_it   <- get_var(vc, "item x temperature")
  s2_pt   <- get_var(vc, "prompt x temperature")
  s2_ij   <- get_var(vc, "item x judge")
  s2_pj   <- get_var(vc, "prompt x judge")
  s2_gen  <- get_var(vc, "generation")

  s2_cat/C + s2_item/N + s2_prom/K +
    s2_ip/(N * K) +
    s2_it/(N * L) + s2_pt/(K * L) +
    s2_ij/(N * M) + s2_pj/(K * M) +
    s2_gen/(N * K * L * M * R)
}

# --- Process each scoring method ---
results_all <- list()

for (scoring in c("likert", "pairwise")) {
  cat("\n--- Scoring:", scoring, "---\n")

  input_path <- sprintf("data/processed/%s_clean.csv", scoring)
  vc_full_path <- sprintf("data/processed/variance_components_%s.csv", scoring)

  if (!file.exists(input_path)) {
    cat("  Skipping — file not found:", input_path, "\n")
    next
  }

  # Load full data
  df_full <- read_csv(input_path, show_col_types = FALSE) %>%
    filter(!is.na(outcome)) %>%
    mutate(
      item_id     = as.factor(item_id),
      variant_id  = as.factor(variant_id),
      temperature = as.factor(temperature),
      judge_model = as.factor(judge_model),
      category    = as.factor(category)
    )

  n_temps  <- n_distinct(df_full$temperature)
  n_judges <- n_distinct(df_full$judge_model)
  n_reps   <- max(as.integer(as.character(df_full$replication))) + 1

  cat("  Full data:", nrow(df_full), "rows,", n_distinct(df_full$item_id), "items,",
      n_distinct(df_full$variant_id), "variants\n")

  # --- Subset to pilot ---
  # Sample 30 items stratified by category (6 per category) to match pilot design
  pilot_items <- df_full %>%
    distinct(item_id, category) %>%
    group_by(category) %>%
    slice_sample(n = N_PILOT / n_distinct(df_full$category)) %>%
    ungroup() %>%
    pull(item_id)

  pilot_variants <- sort(unique(df_full$variant_id))[seq_len(V_PILOT)]

  df_pilot <- df_full %>%
    filter(item_id %in% pilot_items, variant_id %in% pilot_variants) %>%
    droplevels()

  cat("  Pilot subset:", nrow(df_pilot), "rows,", n_distinct(df_pilot$item_id), "items,",
      n_distinct(df_pilot$variant_id), "variants\n")

  # --- Fit pilot model ---
  cat("  Fitting pilot model...\n")
  vc_pilot <- fit_and_extract(df_pilot, scoring)

  # --- Fit full model (or load saved components) ---
  if (file.exists(vc_full_path)) {
    cat("  Loading saved full-run variance components from:", vc_full_path, "\n")
    vc_full <- read_csv(vc_full_path, show_col_types = FALSE)
  } else {
    cat("  Fitting full model...\n")
    vc_full <- fit_and_extract(df_full, scoring)
  }

  # --- Compare individual variance components ---
  comparison <- vc_pilot %>%
    select(tle_label, variance) %>%
    rename(pilot_var = variance) %>%
    left_join(
      vc_full %>% select(tle_label, variance) %>% rename(full_var = variance),
      by = "tle_label"
    ) %>%
    filter(!is.na(full_var)) %>%
    mutate(
      ratio = ifelse(full_var > 0, pilot_var / full_var, NA_real_),
      pct_diff = ifelse(full_var > 0, round(100 * (pilot_var - full_var) / full_var, 1), NA_real_),
      scoring = scoring
    )

  cat("\n  Component-level comparison (pilot vs full):\n")
  comparison %>% select(tle_label, pilot_var, full_var, ratio, pct_diff) %>%
    as.data.frame() %>% print()

  # --- D-study projection: pilot VCs → full design size ---
  n_cats_full <- n_distinct(df_full$category)

  projected_total_var <- dstudy_project(vc_pilot, n_cats_full, N_FULL, V_FULL, n_reps, n_temps, n_judges)
  actual_total_var    <- dstudy_project(vc_full,  n_cats_full, N_FULL, V_FULL, n_reps, n_temps, n_judges)

  # Also compute actual total var at pilot design (for reference)
  pilot_total_var_actual  <- dstudy_project(vc_full,  n_cats_full, N_PILOT, V_PILOT, n_reps, n_temps, n_judges)
  pilot_total_var_pilot   <- dstudy_project(vc_pilot, n_cats_full, N_PILOT, V_PILOT, n_reps, n_temps, n_judges)

  dstudy_comparison <- tibble(
    scoring = scoring,
    scenario = c(
      "Pilot design (pilot VCs)",
      "Pilot design (full VCs)",
      "Full design projected (pilot VCs)",
      "Full design actual (full VCs)"
    ),
    total_var = c(
      pilot_total_var_pilot,
      pilot_total_var_actual,
      projected_total_var,
      actual_total_var
    )
  ) %>%
    mutate(
      ratio_to_actual_full = total_var / actual_total_var,
      pct_diff_from_actual = round(100 * (total_var - actual_total_var) / actual_total_var, 1)
    )

  cat("\n  D-study projection comparison:\n")
  print(dstudy_comparison)

  projection_error <- round(100 * (projected_total_var - actual_total_var) / actual_total_var, 1)
  cat(sprintf("\n  Pilot-projected vs actual full-run total variance: %+.1f%%\n", projection_error))

  results_all[[scoring]] <- list(
    components = comparison,
    dstudy = dstudy_comparison
  )
}

# --- Combine and save ---
components_out <- bind_rows(lapply(results_all, `[[`, "components"))
dstudy_out     <- bind_rows(lapply(results_all, `[[`, "dstudy"))

output_components <- components_out %>%
  select(scoring, tle_label, pilot_var, full_var, ratio, pct_diff)

output_dstudy <- dstudy_out %>%
  select(scoring, scenario, total_var, ratio_to_actual_full, pct_diff_from_actual)

output_path <- "data/processed/pilot_validation.csv"
write_csv(output_components, output_path)
cat("\nComponent comparison saved to:", output_path, "\n")

dstudy_path <- "data/processed/pilot_validation_dstudy.csv"
write_csv(output_dstudy, dstudy_path)
cat("D-study projections saved to:", dstudy_path, "\n")

# --- Summary ---
cat("\n=== SUMMARY ===\n")
for (scoring in names(results_all)) {
  cat(sprintf("\n%s:\n", toupper(scoring)))
  comp <- results_all[[scoring]]$components
  ds   <- results_all[[scoring]]$dstudy

  # Top 3 components by full_var magnitude
  top <- comp %>% filter(!is.na(ratio)) %>% arrange(desc(full_var)) %>% head(3)
  cat("  Top 3 components (ratio = pilot/full):\n")
  for (i in seq_len(nrow(top))) {
    cat(sprintf("    %-30s  ratio=%.3f  (%+.1f%%)\n",
                top$tle_label[i], top$ratio[i], top$pct_diff[i]))
  }

  projected <- ds %>% filter(scenario == "Full design projected (pilot VCs)") %>% pull(total_var)
  actual    <- ds %>% filter(scenario == "Full design actual (full VCs)") %>% pull(total_var)
  cat(sprintf("  D-study projection error: %+.1f%%\n",
              100 * (projected - actual) / actual))
}
