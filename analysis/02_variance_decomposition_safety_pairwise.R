# 02_variance_decomposition_safety_pairwise.R
# Fit crossed random effects model for pairwise safety data.
# Mirrors 02_variance_decomposition_safety.R but with:
#   - pair_id as the "item" (each pair is a unit of analysis)
#   - true_order as a fixed effect (position bias covariate)
#   - Cross-category pairs handled via combined category labels
#
# Usage:
#   Rscript analysis/02_variance_decomposition_safety_pairwise.R

library(tidyverse)
library(lme4)

set.seed(42)

input_path     <- "data/processed/safety_pairwise_clean.csv"
vc_output_path <- "data/processed/variance_components_safety_pairwise.csv"
ds_output_path <- "data/processed/dstudy_safety_pairwise.csv"

cat("=== 02_variance_decomposition_safety_pairwise.R ===\n")

# --- Load data ---
df <- read_csv(input_path, show_col_types = FALSE) %>%
  filter(!is.na(outcome)) %>%
  mutate(
    item_id     = as.factor(item_id),
    variant_id  = as.factor(variant_id),
    temperature = as.factor(temperature),
    judge_model = as.factor(judge_model),
    category    = as.factor(category),
    true_order  = as.factor(true_order)
  )

cat("Data:", nrow(df), "rows,", n_distinct(df$item_id), "pairs,",
    n_distinct(df$category), "categories\n")
cat("  Judge models:", paste(levels(df$judge_model), collapse = ", "), "\n")
cat("  Temperatures:", paste(levels(df$temperature), collapse = ", "), "\n")
cat("  Prompt variants:", paste(levels(df$variant_id), collapse = ", "), "\n")
cat("  Position bias levels:", paste(levels(df$true_order), collapse = ", "), "\n")

# --- Fit crossed random effects model ---
# Binary outcome (LPM): A chosen = 1, B chosen = 0
# true_order = position bias fixed effect (counterbalancing)
cat("\nFitting lmer model...\n")

# Check if we have multiple temperatures
n_temps <- n_distinct(df$temperature)
n_judges <- n_distinct(df$judge_model)

# Build formula dynamically based on available facets
fixed_terms <- "outcome ~ true_order"
random_terms <- c(
  "(1 | category)",
  "(1 | item_id)",
  "(1 | variant_id)",
  "(1 | item_id:variant_id)"
)

# Build cell-level term to include only varying facets (those with > 1 level)
cell_facets <- c("item_id", "variant_id")

if (n_temps > 1) {
  fixed_terms <- paste(fixed_terms, "+ temperature")
  random_terms <- c(random_terms,
    "(1 | item_id:temperature)",
    "(1 | variant_id:temperature)"
  )
  cell_facets <- c(cell_facets, "temperature")
}

if (n_judges > 1) {
  fixed_terms <- paste(fixed_terms, "+ judge_model")
  random_terms <- c(random_terms,
    "(1 | item_id:judge_model)",
    "(1 | variant_id:judge_model)"
  )
  cell_facets <- c(cell_facets, "judge_model")
}

# Cell-level (3-way+) random effect: only add if more than item:variant (i.e.
# at least one of judge/temp varies). Otherwise the "cell" coincides with
# item:variant and is already in the model.
cell_term_str <- NA_character_
if (length(cell_facets) > 2) {
  cell_term_str <- paste0("(1 | ", paste(cell_facets, collapse = ":"), ")")
  random_terms <- c(random_terms, cell_term_str)
  cat("Cell-level term:", cell_term_str, "\n")
} else {
  cat("Cell-level term skipped: only item and variant vary (no judge/temp facet).\n")
}

formula_str <- paste(fixed_terms, "+", paste(random_terms, collapse = " + "))
cat("Formula:", formula_str, "\n")

mod <- lmer(
  as.formula(formula_str),
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

# Map to TEE labels.
# The cell-level (3-way+) term name varies by which facets are present
# (item:variant:judge_model, item:variant:temperature, or
# item:variant:judge_model:temperature). All map to "cell-level (3-way+)".
tle_labels <- c(
  "category"                                          = "between-category",
  "item_id"                                           = "within-category item",
  "variant_id"                                        = "prompt",
  "item_id:variant_id"                                = "item x prompt",
  "item_id:temperature"                               = "item x temperature",
  "variant_id:temperature"                            = "prompt x temperature",
  "item_id:judge_model"                               = "item x judge",
  "variant_id:judge_model"                            = "prompt x judge",
  "item_id:variant_id:judge_model"                    = "cell-level (3-way+)",
  "item_id:variant_id:temperature"                    = "cell-level (3-way+)",
  "item_id:variant_id:judge_model:temperature"        = "cell-level (3-way+)",
  "Residual"                                          = "replicate noise"
)

vc_clean <- vc %>%
  select(grp, vcov, sdcor) %>%
  rename(component = grp, variance = vcov, sd = sdcor) %>%
  mutate(
    tle_label = tle_labels[component],
    tle_label = ifelse(is.na(tle_label), component, tle_label)
  )

# --- Fixed effect sensitivity indices ---
grand_mean <- fixef(mod)["(Intercept)"]

# Temperature (if multiple)
if (n_temps > 1) {
  temp_contrasts <- fixef(mod)[grepl("temperature", names(fixef(mod)))]
  temp_cell_means <- c(grand_mean, grand_mean + temp_contrasts)
  sigma2_temp <- sum((temp_cell_means - mean(temp_cell_means))^2) / length(temp_cell_means)
  vc_clean <- bind_rows(vc_clean, tibble(
    component = "temperature (fixed)",
    variance = sigma2_temp,
    sd = sqrt(sigma2_temp),
    tle_label = "temperature (design sensitivity)"
  ))
}

# Judge model (if multiple)
if (n_judges > 1) {
  judge_contrasts <- fixef(mod)[grepl("judge_model", names(fixef(mod)))]
  if (length(judge_contrasts) > 0) {
    judge_cell_means <- c(grand_mean, grand_mean + judge_contrasts)
    sigma2_judge <- sum((judge_cell_means - mean(judge_cell_means))^2) / length(judge_cell_means)
    vc_clean <- bind_rows(vc_clean, tibble(
      component = "judge_model (fixed)",
      variance = sigma2_judge,
      sd = sqrt(sigma2_judge),
      tle_label = "judge model (design sensitivity)"
    ))
  }
}

# Position bias sensitivity index
order_contrasts <- fixef(mod)[grepl("true_order", names(fixef(mod)))]
if (length(order_contrasts) > 0) {
  order_cell_means <- c(grand_mean, grand_mean + order_contrasts)
  sigma2_order <- sum((order_cell_means - mean(order_cell_means))^2) / length(order_cell_means)
  vc_clean <- bind_rows(vc_clean, tibble(
    component = "true_order (fixed)",
    variance = sigma2_order,
    sd = sqrt(sigma2_order),
    tle_label = "position bias (design sensitivity)"
  ))
}

# --- Compute proportions ---
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
    scoring = "pairwise_safety"
  ) %>%
  arrange(desc(variance))

cat("\n=== TEE Variance Components ===\n")
print(vc_clean %>% select(tle_label, variance, pct_total, tier))

# Sanity check: cell-level vs replicate noise split
s2_eps_cell_check <- vc_clean %>% filter(tle_label == "cell-level (3-way+)") %>% pull(variance)
s2_rho_rep_check  <- vc_clean %>% filter(tle_label == "replicate noise")    %>% pull(variance)
if (length(s2_eps_cell_check) == 0) s2_eps_cell_check <- 0
pct_eps <- vc_clean %>% filter(tle_label == "cell-level (3-way+)") %>% pull(pct_total)
pct_rho <- vc_clean %>% filter(tle_label == "replicate noise") %>% pull(pct_total)
if (length(pct_eps) == 0) pct_eps <- 0
cat(sprintf(
  "\nResidual split sanity check:\n  s2_eps_cell (cell-level) = %.5f (%.2f%% of total)\n  s2_rho_rep  (replicate) = %.5f (%.2f%% of total)\n",
  s2_eps_cell_check, pct_eps,
  s2_rho_rep_check,  pct_rho
))

# --- Save variance components ---
dir.create(dirname(vc_output_path), recursive = TRUE, showWarnings = FALSE)
write_csv(vc_clean, vc_output_path)
cat("\nVariance components saved to:", vc_output_path, "\n")

# --- D-Study Projections ---
cat("\n=== D-Study Projections ===\n")

s2_cat    <- vc_clean %>% filter(tle_label == "between-category") %>% pull(variance)
s2_item   <- vc_clean %>% filter(tle_label == "within-category item") %>% pull(variance)
s2_prompt <- vc_clean %>% filter(tle_label == "prompt") %>% pull(variance)
s2_ip     <- vc_clean %>% filter(tle_label == "item x prompt") %>% pull(variance)
s2_ij     <- vc_clean %>% filter(tle_label == "item x judge") %>% pull(variance)
s2_pj     <- vc_clean %>% filter(tle_label == "prompt x judge") %>% pull(variance)

# Handle optional components (may not exist if single temp/judge)
s2_it <- vc_clean %>% filter(tle_label == "item x temperature") %>% pull(variance)
s2_pt <- vc_clean %>% filter(tle_label == "prompt x temperature") %>% pull(variance)
s2_temp <- vc_clean %>% filter(tle_label == "temperature (design sensitivity)") %>% pull(variance)
s2_judge <- vc_clean %>% filter(tle_label == "judge model (design sensitivity)") %>% pull(variance)
if (length(s2_it) == 0) s2_it <- 0
if (length(s2_pt) == 0) s2_pt <- 0
if (length(s2_ij) == 0) s2_ij <- 0
if (length(s2_pj) == 0) s2_pj <- 0
if (length(s2_temp) == 0) s2_temp <- 0
if (length(s2_judge) == 0) s2_judge <- 0

# Two residual components: cell-level (3-way+) and within-cell replicate.
# Cell-level term may be absent if neither judge nor temp varies (then "cell"
# coincides with item:variant, which is already in the model).
s2_eps_cell <- vc_clean %>% filter(tle_label == "cell-level (3-way+)") %>% pull(variance)
s2_rho_rep  <- vc_clean %>% filter(tle_label == "replicate noise")    %>% pull(variance)
if (length(s2_eps_cell) == 0) s2_eps_cell <- 0
if (length(s2_rho_rep)  == 0) s2_rho_rep  <- 0

n_cats_actual    <- n_distinct(df$category)
n_items_actual   <- n_distinct(df$item_id)
n_prompts_actual <- n_distinct(df$variant_id)
n_reps_actual    <- max(as.integer(as.character(df$replication))) + 1

# D-study: Case 1 — Fixed temperature, fixed judge model
dstudy_fixed <- function(C, N, K, R) {
  # Categories are fixed (taxonomy-level), so s2_cat absorbs into s2_item
  # and both scale with N (paper footnote, pnas_main.tex Materials & Methods).
  (s2_cat + s2_item)/N + s2_prompt/K + s2_temp + s2_judge +
    s2_ip/(N * K) +
    s2_ij/N + s2_pj/K +
    s2_it/N + s2_pt/K +
    s2_eps_cell/(N * K) +        # cell-level, no /R
    s2_rho_rep/(N * K * R)       # replicate, with /R
}

# D-study: Case 2 — Averaged over temperatures and judges
dstudy_avg <- function(C, N, K, R, L = n_temps, M = n_judges) {
  (s2_cat + s2_item)/N + s2_prompt/K + s2_temp/L + s2_judge/M +
    s2_ip/(N * K) +
    s2_it/(N * L) + s2_pt/(K * L) +
    s2_ij/(N * M) + s2_pj/(K * M) +
    s2_eps_cell/(N * K * L * M) +        # cell-level, no /R
    s2_rho_rep/(N * K * L * M * R)       # replicate, with /R
}

baseline_var <- dstudy_avg(n_cats_actual, n_items_actual, n_prompts_actual, n_reps_actual)

scenarios <- tribble(
  ~scenario,                        ~n_cats_d,      ~n_items_d,      ~n_prompts_d,        ~n_reps_d,      ~fix_temp, ~fix_judge,
  "Baseline (avg temp & judge)",    n_cats_actual,   n_items_actual,  n_prompts_actual,    n_reps_actual,   FALSE,     FALSE,
  "Baseline (fix temp & judge)",    n_cats_actual,   n_items_actual,  n_prompts_actual,    n_reps_actual,   TRUE,      TRUE,
  "Double items",                   n_cats_actual,   n_items_actual*2, n_prompts_actual,   n_reps_actual,   FALSE,     FALSE,
  "+2 prompt variants",             n_cats_actual,   n_items_actual,  n_prompts_actual+2,  n_reps_actual,   FALSE,     FALSE,
  "+5 replications",                n_cats_actual,   n_items_actual,  n_prompts_actual,    n_reps_actual+5, FALSE,     FALSE,
  "Fix judge model",                n_cats_actual,   n_items_actual,  n_prompts_actual,    n_reps_actual,   FALSE,     TRUE,
  "Fix temp & judge",               n_cats_actual,   n_items_actual,  n_prompts_actual,    n_reps_actual,   TRUE,      TRUE,
)

dstudy_results <- scenarios %>%
  rowwise() %>%
  mutate(
    total_var = if (fix_temp & fix_judge) {
      dstudy_fixed(n_cats_d, n_items_d, n_prompts_d, n_reps_d)
    } else if (fix_temp) {
      dstudy_avg(n_cats_d, n_items_d, n_prompts_d, n_reps_d, L = 1, M = n_judges)
    } else if (fix_judge) {
      dstudy_avg(n_cats_d, n_items_d, n_prompts_d, n_reps_d, L = n_temps, M = 1)
    } else {
      dstudy_avg(n_cats_d, n_items_d, n_prompts_d, n_reps_d)
    }
  ) %>%
  ungroup() %>%
  mutate(
    pct_of_baseline = round(100 * total_var / baseline_var, 1),
    reduction_pct = round(100 * (1 - total_var / baseline_var), 1),
    scoring = "pairwise_safety"
  )

cat("\nD-Study results:\n")
print(dstudy_results %>% select(scenario, total_var, pct_of_baseline, reduction_pct))

# --- Save D-study ---
write_csv(dstudy_results, ds_output_path)
cat("\nD-study saved to:", ds_output_path, "\n")

# --- Per-category generation variance ---
cat("\n=== Per-Category Generation Variance ===\n")
per_cat <- df %>%
  filter(!is.na(outcome)) %>%
  group_by(category) %>%
  summarise(
    n = n(),
    mean_outcome = mean(outcome),
    var_outcome = var(outcome),
    .groups = "drop"
  ) %>%
  arrange(desc(var_outcome))

print(per_cat)

per_cat_path <- "data/processed/per_category_variance_safety_pairwise.csv"
write_csv(per_cat, per_cat_path)
cat("\nPer-category variance saved to:", per_cat_path, "\n")

cat("\n=== Done ===\n")
