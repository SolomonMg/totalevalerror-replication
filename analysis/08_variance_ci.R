# 08_variance_ci.R
# Compute confidence intervals for variance components.
# Strategy: try profile likelihood first (fast, exact); if it fails
# (common with complex crossed RE), fall back to parametric bootstrap
# on a 30-item pilot subset (~15 min).
#
# Usage:
#   Rscript analysis/08_variance_ci.R

library(tidyverse)
library(lme4)

set.seed(42)

BOOT_NSIM <- 200

cat("=== 08_variance_ci.R ===\n")

# --- Helper: fit model ---
fit_model <- function(df, scoring) {
  has_position <- "true_order" %in% names(df)
  if (has_position) df$true_order <- as.factor(df$true_order)

  fixed_formula <- if (has_position) {
    outcome ~ temperature + judge_model + true_order +
      (1 | category) + (1 | item_id) + (1 | variant_id) +
      (1 | item_id:variant_id) + (1 | item_id:temperature) +
      (1 | variant_id:temperature) + (1 | item_id:judge_model) +
      (1 | variant_id:judge_model)
  } else {
    outcome ~ temperature + judge_model +
      (1 | category) + (1 | item_id) + (1 | variant_id) +
      (1 | item_id:variant_id) + (1 | item_id:temperature) +
      (1 | variant_id:temperature) + (1 | item_id:judge_model) +
      (1 | variant_id:judge_model)
  }

  lmer(fixed_formula, data = df, REML = TRUE,
       control = lmerControl(optimizer = "bobyqa",
                             optCtrl = list(maxfun = 50000)))
}

# --- Helper: extract CIs from confint output ---
extract_ci <- function(ci_raw) {
  re_names <- rownames(ci_raw)[grepl("^sd_|^sigma$", rownames(ci_raw))]
  tibble(
    ci_name = re_names,
    ci_lower_sd = ci_raw[re_names, 1],
    ci_upper_sd = ci_raw[re_names, 2]
  ) %>%
    mutate(
      ci_lower_var = ci_lower_sd^2,
      ci_upper_var = ci_upper_sd^2,
      component = case_when(
        ci_name == "sd_category.(Intercept)"               ~ "category",
        ci_name == "sd_item_id.(Intercept)"                ~ "item_id",
        ci_name == "sd_variant_id.(Intercept)"             ~ "variant_id",
        ci_name == "sd_item_id:variant_id.(Intercept)"     ~ "item_id:variant_id",
        ci_name == "sd_item_id:temperature.(Intercept)"    ~ "item_id:temperature",
        ci_name == "sd_variant_id:temperature.(Intercept)" ~ "variant_id:temperature",
        ci_name == "sd_item_id:judge_model.(Intercept)"    ~ "item_id:judge_model",
        ci_name == "sd_variant_id:judge_model.(Intercept)" ~ "variant_id:judge_model",
        ci_name == "sigma"                                 ~ "Residual",
        TRUE ~ ci_name
      )
    ) %>%
    select(component, ci_lower_var, ci_upper_var)
}

# --- Process each scoring method ---
for (scoring in c("likert", "pairwise")) {
  cat("\n--- Scoring:", scoring, "---\n")

  input_path <- sprintf("data/processed/%s_clean.csv", scoring)
  ci_output_path <- sprintf("data/processed/variance_ci_%s.csv", scoring)

  if (file.exists(ci_output_path)) {
    cat("  CI file already exists:", ci_output_path, "(delete to recompute)\n")
    next
  }

  if (!file.exists(input_path)) {
    cat("  Skipping — file not found:", input_path, "\n")
    next
  }

  df <- read_csv(input_path, show_col_types = FALSE) %>%
    filter(!is.na(outcome)) %>%
    mutate(
      item_id     = as.factor(item_id),
      variant_id  = as.factor(variant_id),
      temperature = as.factor(temperature),
      judge_model = as.factor(judge_model),
      category    = as.factor(category)
    )

  cat("  Full data:", nrow(df), "rows\n")

  # --- Attempt 1: Profile likelihood on full data ---
  cat("  Trying profile likelihood CIs on full data...\n")
  mod_full <- fit_model(df, scoring)

  profile_ok <- FALSE
  tryCatch({
    ci_profile <- confint(mod_full, method = "profile", oldNames = FALSE, quiet = TRUE)
    ci_df <- extract_ci(ci_profile)
    ci_df$method <- "profile"
    ci_df$data_subset <- "full"
    write_csv(ci_df, ci_output_path)
    cat("  Profile CIs succeeded! Saved to:", ci_output_path, "\n")
    profile_ok <- TRUE
  }, error = function(e) {
    cat("  Profile CIs failed:", e$message, "\n")
  })

  if (profile_ok) next

  # --- Attempt 2: Bootstrap on 30-item pilot subset ---
  cat(sprintf("  Falling back to bootstrap (nsim=%d) on 30-item pilot subset...\n", BOOT_NSIM))

  N_PILOT <- 30
  V_PILOT <- 3
  pilot_items <- df %>%
    distinct(item_id, category) %>%
    group_by(category) %>%
    slice_sample(n = N_PILOT / n_distinct(df$category)) %>%
    ungroup() %>%
    pull(item_id)

  pilot_variants <- sort(unique(df$variant_id))[seq_len(V_PILOT)]

  df_pilot <- df %>%
    filter(item_id %in% pilot_items, variant_id %in% pilot_variants) %>%
    droplevels()

  cat("  Pilot subset:", nrow(df_pilot), "rows\n")
  mod_pilot <- fit_model(df_pilot, scoring)

  tryCatch({
    ci_boot <- confint(mod_pilot, method = "boot", nsim = BOOT_NSIM,
                       oldNames = FALSE, parallel = "multicore",
                       ncpus = max(1, parallel::detectCores() - 1))
    ci_df <- extract_ci(ci_boot)
    ci_df$method <- "bootstrap"
    ci_df$data_subset <- sprintf("pilot (N=%d, V=%d)", N_PILOT, V_PILOT)
    write_csv(ci_df, ci_output_path)
    cat("  Bootstrap CIs saved to:", ci_output_path, "\n")
  }, error = function(e) {
    cat("  Bootstrap CIs also failed:", e$message, "\n")
    cat("  No CIs produced for", scoring, "\n")
  })
}

cat("\n=== Done ===\n")
