#!/usr/bin/env Rscript
# 23_decomposition_pairwise_cot.R
# Fit the TEE variance decomposition on the CoT + bias-mitigated pairwise data
# (recoded model_a_wins as outcome). Compute D-study projections and
# per-category residual variance. Outputs match the schema of the naive-prompt
# pipeline so downstream figure code can swap datasets.
#
# Writes:
#   data/processed/variance_components_pairwise_cot.csv
#   data/processed/dstudy_pairwise_cot.csv
#   data/processed/per_category_variance_pairwise_cot.csv
#   data/processed/pairwise_decomp_comparison_full.csv

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
})

df <- read_csv("data/processed/pairwise_cot_clean.csv", show_col_types = FALSE) %>%
  filter(!is.na(model_a_wins), !is.na(true_order))

cat("=== 23_decomposition_pairwise_cot.R ===\n")
cat(sprintf("Rows: %d | judges: %d | items: %d | variants: %d | temps: %d | reps: %d\n",
            nrow(df),
            n_distinct(df$judge_model),
            n_distinct(df$item_id),
            n_distinct(df$variant_id),
            n_distinct(df$temperature),
            n_distinct(df$replication)))

df$temperature <- as.factor(df$temperature)
df$judge_model <- as.factor(df$judge_model)

# ---- Fit ----
cat("\nFitting lmer on model_a_wins ...\n")
formula_cot <- model_a_wins ~ temperature + judge_model +
  (1 | category) +
  (1 | item_id) +
  (1 | variant_id) +
  (1 | item_id:variant_id) +
  (1 | item_id:temperature) +
  (1 | variant_id:temperature) +
  (1 | item_id:judge_model) +
  (1 | variant_id:judge_model)

mod <- lmer(
  formula_cot, data = df, REML = TRUE,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000))
)
if (any(grepl("singular", mod@optinfo$conv$lme4$messages))) {
  warning("Singular fit -- some variance components near zero.")
}

# ---- Variance components (schema matches naive pipeline) ----
vc_raw <- as.data.frame(VarCorr(mod)) %>%
  select(grp, vcov, sdcor) %>%
  rename(component = grp, variance = vcov, sd = sdcor)

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
vc_clean <- vc_raw %>%
  mutate(tle_label = ifelse(component %in% names(tle_labels),
                            tle_labels[component], component))

# Fixed-effect sensitivity indices (population variance, 1/L not 1/(L-1))
fe <- fixef(mod)
temp_contrasts  <- c(0, fe[grepl("^temperature", names(fe))])
judge_contrasts <- c(0, fe[grepl("^judge_model", names(fe))])
sigma2_tau    <- var(temp_contrasts)  * (length(temp_contrasts) - 1)  / length(temp_contrasts)
sigma2_lambda <- var(judge_contrasts) * (length(judge_contrasts) - 1) / length(judge_contrasts)

vc_clean <- bind_rows(
  vc_clean,
  tibble(component = "temperature (fixed)", variance = sigma2_tau, sd = sqrt(sigma2_tau),
         tle_label = "temperature (design sensitivity)"),
  tibble(component = "judge_model (fixed)", variance = sigma2_lambda, sd = sqrt(sigma2_lambda),
         tle_label = "judge model (design sensitivity)")
)

total_var <- sum(vc_clean$variance)
vc_clean <- vc_clean %>%
  mutate(
    pct_total = round(100 * variance / total_var, 2),
    tier = case_when(
      tle_label %in% c("generation", "between-category", "within-category item",
                        "prompt", "item x prompt", "item x temperature",
                        "prompt x temperature", "item x judge",
                        "prompt x judge") ~ "model-side",
      TRUE ~ "pipeline-side"
    ),
    scoring = "pairwise"  # keep name stable for downstream code
  ) %>%
  arrange(desc(variance))

cat("\n=== TEE Variance Components (CoT pairwise) ===\n")
print(vc_clean %>% select(tle_label, variance, pct_total, tier) %>%
        mutate(variance = round(variance, 5)))

vc_path <- "data/processed/variance_components_pairwise_cot.csv"
write_csv(vc_clean, vc_path)
cat("\nSaved:", vc_path, "\n")

# ---- D-study projections ----
cat("\n=== D-Study Projections ===\n")

s2_cat    <- vc_clean %>% filter(tle_label == "between-category") %>% pull(variance)
s2_item   <- vc_clean %>% filter(tle_label == "within-category item") %>% pull(variance)
s2_prompt <- vc_clean %>% filter(tle_label == "prompt") %>% pull(variance)
s2_gen    <- vc_clean %>% filter(tle_label == "generation") %>% pull(variance)
s2_ip     <- vc_clean %>% filter(tle_label == "item x prompt") %>% pull(variance)
s2_it     <- vc_clean %>% filter(tle_label == "item x temperature") %>% pull(variance)
s2_pt     <- vc_clean %>% filter(tle_label == "prompt x temperature") %>% pull(variance)
s2_ij     <- vc_clean %>% filter(tle_label == "item x judge") %>% pull(variance)
s2_pj     <- vc_clean %>% filter(tle_label == "prompt x judge") %>% pull(variance)
s2_temp   <- vc_clean %>% filter(tle_label == "temperature (design sensitivity)") %>% pull(variance)
s2_judge  <- vc_clean %>% filter(tle_label == "judge model (design sensitivity)") %>% pull(variance)
s2_temp   <- if (length(s2_temp) == 0) 0 else s2_temp
s2_judge  <- if (length(s2_judge) == 0) 0 else s2_judge

n_cats    <- n_distinct(df$category)
n_items   <- n_distinct(df$item_id)
n_prompts <- n_distinct(df$variant_id)
n_temps   <- n_distinct(df$temperature)
n_judges  <- n_distinct(df$judge_model)
n_reps    <- max(as.integer(df$replication), na.rm = TRUE) + 1

cat(sprintf("  Design: N=%d, V=%d, H=%d, M=%d, R=%d\n",
            n_items, n_prompts, n_temps, n_judges, n_reps))

# Case: fixed temp & judge
dstudy_fixed <- function(C, N, K, R) {
  (s2_cat + s2_item)/N + s2_prompt/K + s2_temp + s2_judge +
    s2_ip/(N * K) +
    s2_ij/N + s2_pj/K +
    s2_it/N + s2_pt/K +
    s2_gen/(N * K * R)
}

# Case: averaged temp & judge
dstudy_avg <- function(C, N, K, R, L = n_temps, M = n_judges) {
  (s2_cat + s2_item)/N + s2_prompt/K + s2_temp/L + s2_judge/M +
    s2_ip/(N * K) +
    s2_it/(N * L) + s2_pt/(K * L) +
    s2_ij/(N * M) + s2_pj/(K * M) +
    s2_gen/(N * K * L * M * R)
}

baseline_var <- dstudy_avg(n_cats, n_items, n_prompts, n_reps)

scenarios <- tribble(
  ~scenario,                                                ~n_cats_d, ~n_items_d, ~n_prompts_d, ~n_reps_d, ~fix_temp, ~fix_judge,
  "Baseline (avg temp & judge)",                            n_cats,    n_items,    n_prompts,    n_reps,    FALSE,     FALSE,
  "Baseline (fix temp & judge)",                            n_cats,    n_items,    n_prompts,    n_reps,    TRUE,      TRUE,
  "+5 replications",                                        n_cats,    n_items,    n_prompts,    n_reps+5,  FALSE,     FALSE,
  "+2 prompt variants",                                     n_cats,    n_items,    n_prompts+2,  n_reps,    FALSE,     FALSE,
  "Fix temperature",                                        n_cats,    n_items,    n_prompts,    n_reps,    TRUE,      FALSE,
  "Fix judge model",                                        n_cats,    n_items,    n_prompts,    n_reps,    FALSE,     TRUE,
  "Fix temp & judge",                                       n_cats,    n_items,    n_prompts,    n_reps,    TRUE,      TRUE,
  "+5 reps & fix temp & judge",                             n_cats,    n_items,    n_prompts,    n_reps+5,  TRUE,      TRUE,
  "Double items",                                           n_cats,    n_items*2,  n_prompts,    n_reps,    FALSE,     FALSE,
  "Maximal: 2x items + 5 prompts + 10 reps + fix design",   n_cats,    n_items*2,  5,            10,        TRUE,      TRUE,
)

dstudy <- scenarios %>%
  rowwise() %>%
  mutate(
    total_var = if (fix_temp & fix_judge) {
      dstudy_fixed(n_cats_d, n_items_d, n_prompts_d, n_reps_d)
    } else if (fix_temp & !fix_judge) {
      (s2_cat + s2_item)/n_items_d + s2_prompt/n_prompts_d +
        s2_temp + s2_judge/n_judges +
        s2_ip/(n_items_d * n_prompts_d) +
        s2_it/n_items_d + s2_pt/n_prompts_d +
        s2_ij/(n_items_d * n_judges) + s2_pj/(n_prompts_d * n_judges) +
        s2_gen/(n_items_d * n_prompts_d * n_judges * n_reps_d)
    } else if (!fix_temp & fix_judge) {
      (s2_cat + s2_item)/n_items_d + s2_prompt/n_prompts_d +
        s2_temp/n_temps + s2_judge +
        s2_ip/(n_items_d * n_prompts_d) +
        s2_ij/n_items_d + s2_pj/n_prompts_d +
        s2_it/(n_items_d * n_temps) + s2_pt/(n_prompts_d * n_temps) +
        s2_gen/(n_items_d * n_prompts_d * n_temps * n_reps_d)
    } else {
      dstudy_avg(n_cats_d, n_items_d, n_prompts_d, n_reps_d)
    }
  ) %>%
  ungroup() %>%
  mutate(
    pct_of_baseline = round(100 * total_var / baseline_var, 1),
    reduction_pct   = round(100 * (1 - total_var / baseline_var), 1),
    scoring = "pairwise"
  )

print(dstudy %>% select(scenario, total_var, pct_of_baseline, reduction_pct))

ds_path <- "data/processed/dstudy_pairwise_cot.csv"
write_csv(dstudy, ds_path)
cat("\nSaved:", ds_path, "\n")

# ---- Per-category residual variance ----
cat("\n=== Per-Category Generation Variance ===\n")
topic_var <- df %>%
  group_by(category, item_id, variant_id, temperature, judge_model) %>%
  summarize(cell_var = var(model_a_wins, na.rm = TRUE), .groups = "drop") %>%
  group_by(category) %>%
  summarize(
    mean_gen_var = mean(cell_var, na.rm = TRUE),
    sd_gen_var   = sd(cell_var,   na.rm = TRUE),
    n_cells      = n(),
    .groups = "drop"
  ) %>%
  mutate(scoring = "pairwise")

print(topic_var)

tv_path <- "data/processed/per_category_variance_pairwise_cot.csv"
write_csv(topic_var, tv_path)
cat("\nSaved:", tv_path, "\n")

# ---- Comparison with prior decompositions ----
orig <- read_csv("data/processed/variance_components_pairwise.csv",
                 show_col_types = FALSE) %>% filter(!is.na(variance))
recoded_naive <- tryCatch(
  read_csv("data/processed/variance_components_pairwise_recoded.csv", show_col_types = FALSE),
  error = function(e) NULL
)

cmp <- orig %>% select(tle_label, v_original_naive = variance)
if (!is.null(recoded_naive)) {
  cmp <- cmp %>% full_join(recoded_naive %>% select(tle_label, v_recoded_naive = variance),
                           by = "tle_label")
}
cmp <- cmp %>% full_join(vc_clean %>% select(tle_label, v_cot_recoded = variance),
                         by = "tle_label") %>%
  mutate(
    pct_original_naive = 100 * v_original_naive / sum(v_original_naive, na.rm = TRUE),
    pct_recoded_naive  = if ("v_recoded_naive" %in% names(.))
      100 * v_recoded_naive / sum(v_recoded_naive, na.rm = TRUE) else NA_real_,
    pct_cot_recoded    = 100 * v_cot_recoded / sum(v_cot_recoded, na.rm = TRUE)
  ) %>%
  select(tle_label, starts_with("pct_")) %>%
  arrange(desc(coalesce(pct_original_naive, 0)))

cat("\n=== Side-by-side %-of-total variance ===\n")
print(cmp %>% mutate(across(where(is.numeric), ~ round(., 1))))
write_csv(cmp, "data/processed/pairwise_decomp_comparison_full.csv")
cat("\nSaved: data/processed/pairwise_decomp_comparison_full.csv\n")
