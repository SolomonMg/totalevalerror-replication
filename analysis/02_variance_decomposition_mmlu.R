# 02_variance_decomposition_mmlu.R
# Fit crossed random effects model for MMLU (binary correct/incorrect) data.
# SUT-layer only: no judge terms. Items are multiple-choice questions answered
# directly by the SUT models.
#
# Extracts TEE variance components and D-study projections.
#
# Usage:
#   Rscript analysis/02_variance_decomposition_mmlu.R

library(tidyverse)
library(lme4)

set.seed(42)

input_path     <- "data/processed/mmlu_clean.csv"
vc_output_path <- "data/processed/variance_components_mmlu.csv"
ds_output_path <- "data/processed/dstudy_mmlu.csv"

cat("=== 02_variance_decomposition_mmlu.R ===\n")

# --- Load data ---
df <- read_csv(input_path, show_col_types = FALSE) %>%
  filter(!is.na(outcome)) %>%
  mutate(
    item_id     = as.factor(item_id),
    variant_id  = as.factor(variant_id),
    temperature = as.factor(temperature),
    sut_model   = as.factor(sut_model),
    category    = as.factor(category),
    subcategory = as.factor(subcategory)
  )

cat("Data:", nrow(df), "rows,", n_distinct(df$item_id), "items,",
    n_distinct(df$category), "categories,",
    n_distinct(df$subcategory), "subcategories\n")

# --- Population variance helper ---
# Fixed-effect sensitivity uses population variance (1/L, not 1/(L-1)).
pop_var <- function(x) mean((x - mean(x))^2)

# --- Fit crossed random effects model ---
# SUT-layer only: no judge facet. Fixed effects are temperature and sut_model
# (Tier 2 design choices). Random effects capture item heterogeneity, prompt
# sensitivity, generation stochasticity, and their interactions.
# Binary outcome (LPM): correct=1, incorrect=0.
cat("\nFitting lmer model...\n")

mod <- lmer(
  outcome ~ temperature + sut_model +
    (1 | category) +
    (1 | item_id) +
    (1 | variant_id) +
    (1 | item_id:variant_id) +
    (1 | item_id:temperature) +
    (1 | variant_id:temperature) +
    (1 | item_id:sut_model) +
    (1 | variant_id:sut_model) +
    (1 | item_id:variant_id:sut_model:temperature),  # cell-level (3-way+)
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
  warning("Singular fit detected -- some variance components near zero.")
}

# --- Extract variance components ---
vc <- as.data.frame(VarCorr(mod))
cat("\nVariance components:\n")
print(vc)

# Map to TEE labels (SUT-layer: no judge, use SUT instead)
tle_labels <- c(
  "category"                = "between-category",
  "item_id"                 = "within-category item",
  "variant_id"              = "prompt",
  "item_id:variant_id"      = "item x prompt",
  "item_id:temperature"     = "item x temperature",
  "variant_id:temperature"  = "prompt x temperature",
  "item_id:sut_model"       = "item x SUT",
  "variant_id:sut_model"    = "prompt x SUT",
  "item_id:variant_id:sut_model:temperature" = "cell-level (3-way+)",
  "Residual"                = "replicate noise"
)

# TEE symbol mapping for reference
tle_symbols <- c(
  "between-category"        = "sigma2_kappa",
  "within-category item"    = "sigma2_delta",
  "prompt"                  = "sigma2_phi",     # was sigma2_rho; rho freed for replicate
  "item x prompt"           = "sigma2_alpha_phi",
  "item x temperature"      = "sigma2_alpha_tau",
  "prompt x temperature"    = "sigma2_phi_tau",
  "item x SUT"              = "sigma2_alpha_lambda",
  "prompt x SUT"            = "sigma2_phi_lambda",
  "cell-level (3-way+)"     = "sigma2_epsilon", # cell-level residual
  "replicate noise"         = "sigma2_rho"      # within-cell replicate
)

vc_clean <- vc %>%
  select(grp, vcov, sdcor) %>%
  rename(component = grp, variance = vcov, sd = sdcor) %>%
  mutate(
    tle_label = tle_labels[component],
    tle_label = ifelse(is.na(tle_label), component, tle_label)
  )

# Add temperature fixed effect sensitivity index
temp_contrasts <- fixef(mod)[grepl("temperature", names(fixef(mod)))]
grand_mean <- fixef(mod)["(Intercept)"]
temp_cell_means <- c(grand_mean, grand_mean + temp_contrasts)
sigma2_temp <- pop_var(temp_cell_means)

vc_clean <- bind_rows(
  vc_clean,
  tibble(
    component = "temperature (fixed)",
    variance = sigma2_temp,
    sd = sqrt(sigma2_temp),
    tle_label = "temperature (design sensitivity)"
  )
)

# Add SUT model fixed effect sensitivity index
sut_contrasts <- fixef(mod)[grepl("sut_model", names(fixef(mod)))]
if (length(sut_contrasts) > 0) {
  sut_cell_means <- c(grand_mean, grand_mean + sut_contrasts)
  sigma2_sut <- pop_var(sut_cell_means)
  vc_clean <- bind_rows(
    vc_clean,
    tibble(
      component = "sut_model (fixed)",
      variance = sigma2_sut,
      sd = sqrt(sigma2_sut),
      tle_label = "SUT model (design sensitivity)"
    )
  )
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
                        "prompt x temperature", "item x SUT",
                        "prompt x SUT") ~ "model-side",
      TRUE ~ "pipeline-side"
    ),
    scoring = "mmlu"
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
s2_is     <- vc_clean %>% filter(tle_label == "item x SUT") %>% pull(variance)
s2_ps     <- vc_clean %>% filter(tle_label == "prompt x SUT") %>% pull(variance)
s2_temp   <- vc_clean %>% filter(tle_label == "temperature (design sensitivity)") %>% pull(variance)
s2_sut    <- vc_clean %>% filter(tle_label == "SUT model (design sensitivity)") %>% pull(variance)
s2_temp   <- if (length(s2_temp) == 0) 0 else s2_temp
s2_sut    <- if (length(s2_sut) == 0) 0 else s2_sut
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
n_suts    <- n_distinct(df$sut_model)
n_reps    <- max(df$replication) + 1

cat("\nCurrent design:\n")
cat("  Categories:", n_cats, "\n")
cat("  Items:", n_items, "\n")
cat("  Prompts:", n_prompts, "\n")
cat("  Temperatures:", n_temps, "\n")
cat("  SUT models:", n_suts, "\n")
cat("  Replications:", n_reps, "\n")

# D-study: Case 1 -- Fixed temperature, fixed SUT model
# When SUT is fixed, item x SUT (s2_is) still varies across items -> averages at 1/N.
# When temp is fixed, item x temp (s2_it) still varies across items -> averages at 1/N.
# Same logic for prompt x SUT (s2_ps/K) and prompt x temp (s2_pt/K).
dstudy_fixed <- function(C, N, K, R) {
  # Categories are fixed (taxonomy-level), so s2_cat absorbs into s2_item
  # and both scale with N (paper footnote, pnas_main.tex Materials & Methods).
  (s2_cat + s2_item)/N + s2_prompt/K + s2_temp + s2_sut +
    s2_ip/(N * K) +
    s2_is/N + s2_ps/K +   # SUT interaction terms (fixed SUT)
    s2_it/N + s2_pt/K +   # temp interaction terms (fixed temp)
    s2_eps_cell/(N * K) +        # cell-level, no /R
    s2_rho_rep/(N * K * R)       # replicate, with /R
}

# D-study: Case 2 -- Averaged over L temperatures and M SUT models
dstudy_avg <- function(C, N, K, R, L = n_temps, M = n_suts) {
  (s2_cat + s2_item)/N + s2_prompt/K + s2_temp/L + s2_sut/M +
    s2_ip/(N * K) +
    s2_it/(N * L) + s2_pt/(K * L) +
    s2_is/(N * M) + s2_ps/(K * M) +
    s2_eps_cell/(N * K * L * M) +        # cell-level, no /R
    s2_rho_rep/(N * K * L * M * R)       # replicate, with /R
}

baseline_var <- dstudy_avg(n_cats, n_items, n_prompts, n_reps)

# D-study scenarios
scenarios <- tribble(
  ~scenario,                                               ~n_cats_d, ~n_items_d, ~n_prompts_d, ~n_reps_d, ~fix_temp, ~fix_sut,
  "Baseline (avg temp & SUT)",                             n_cats,    n_items,    n_prompts,    n_reps,    FALSE,     FALSE,
  "Baseline (fix temp & SUT)",                             n_cats,    n_items,    n_prompts,    n_reps,    TRUE,      TRUE,
  "Double items",                                          n_cats,    n_items*2,  n_prompts,    n_reps,    FALSE,     FALSE,
  "Double replications",                                   n_cats,    n_items,    n_prompts,    n_reps*2,  FALSE,     FALSE,
  "Double prompt variants",                                n_cats,    n_items,    n_prompts*2,  n_reps,    FALSE,     FALSE,
  "Fix temperature",                                       n_cats,    n_items,    n_prompts,    n_reps,    TRUE,      FALSE,
  "Fix SUT model",                                         n_cats,    n_items,    n_prompts,    n_reps,    FALSE,     TRUE,
  "Fix temp & SUT",                                        n_cats,    n_items,    n_prompts,    n_reps,    TRUE,      TRUE,
  "+5 reps & fix temp & SUT",                              n_cats,    n_items,    n_prompts,    n_reps+5,  TRUE,      TRUE,
  "Maximal: 2x items + 5 prompts + 10 reps + fix design", n_cats,    n_items*2,  5,            10,        TRUE,      TRUE,
)

dstudy <- scenarios %>%
  rowwise() %>%
  mutate(
    total_var = if (fix_temp & fix_sut) {
      dstudy_fixed(n_cats_d, n_items_d, n_prompts_d, n_reps_d)
    } else if (fix_temp & !fix_sut) {
      # Fixed temp, avg over SUTs
      # Temp is fixed -> s2_it/N + s2_pt/K remain (fixed temp interaction)
      # SUT is averaged -> s2_is/(N*M) + s2_ps/(K*M)
      # Cell-level averages over N*K*M cells.
      (s2_cat + s2_item)/n_items_d + s2_prompt/n_prompts_d +
        s2_temp + s2_sut/n_suts +
        s2_ip/(n_items_d * n_prompts_d) +
        s2_it/n_items_d + s2_pt/n_prompts_d +                        # fixed temp
        s2_is/(n_items_d * n_suts) + s2_ps/(n_prompts_d * n_suts) +  # avg SUT
        s2_eps_cell/(n_items_d * n_prompts_d * n_suts) +
        s2_rho_rep/(n_items_d * n_prompts_d * n_suts * n_reps_d)
    } else if (!fix_temp & fix_sut) {
      # Avg over temps, fixed SUT
      # SUT is fixed -> s2_is/N + s2_ps/K remain (fixed SUT interaction)
      # Temp is averaged -> s2_it/(N*L) + s2_pt/(K*L)
      # Cell-level averages over N*K*L cells.
      (s2_cat + s2_item)/n_items_d + s2_prompt/n_prompts_d +
        s2_temp/n_temps + s2_sut +
        s2_ip/(n_items_d * n_prompts_d) +
        s2_is/n_items_d + s2_ps/n_prompts_d +                          # fixed SUT
        s2_it/(n_items_d * n_temps) + s2_pt/(n_prompts_d * n_temps) +  # avg temp
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
    scoring = "mmlu"
  )

cat("\nD-Study Results:\n")
print(dstudy %>% select(scenario, total_var, pct_of_baseline, reduction_pct))

write_csv(dstudy, ds_output_path)
cat("\nD-study projections saved to:", ds_output_path, "\n")

# --- Per-category generation variance ---
cat("\n=== Per-Category Generation Variance ===\n")

topic_var <- df %>%
  group_by(category, item_id, variant_id, temperature, sut_model) %>%
  summarize(cell_var = var(outcome, na.rm = TRUE), .groups = "drop") %>%
  group_by(category) %>%
  summarize(
    mean_gen_var = mean(cell_var, na.rm = TRUE),
    sd_gen_var = sd(cell_var, na.rm = TRUE),
    n_cells = n(),
    .groups = "drop"
  ) %>%
  mutate(scoring = "mmlu") %>%
  arrange(desc(mean_gen_var))

cat("\nPer-category generation variance:\n")
print(topic_var)

topic_output_path <- "data/processed/per_category_variance_mmlu.csv"
write_csv(topic_var, topic_output_path)
cat("\nPer-category variance saved to:", topic_output_path, "\n")

# --- Confidence intervals (Wald-based, fast) ---
cat("\nComputing Wald CIs for variance components...\n")
tryCatch({
  ci <- confint(mod, method = "Wald", oldNames = FALSE)
  cat("Wald CIs:\n")
  print(ci)
}, error = function(e) {
  warning("Wald CIs failed: ", e$message)
})
