# 02b_glmm_robustness.R
# Compare LPM (lmer) and GLMM (glmer, binomial) variance decompositions
# for binary outcome datasets (pairwise and safety) at the Var(θ̂) level.
#
# Both fits include the cell-level random effect
# `(1|item:variant:judge:temperature)` so σ²_ε is separated from σ²_ρ.
# Fixed-effect sensitivity indices (σ²_λ, σ²_τ, σ²_π for pairwise position)
# are computed on each scale as the population variance of the fitted
# fixed-effect cell means.
#
# Each component σ²_k is then divided by its design-study denominator
# (e.g., σ²_λ/M, σ²_αλ/(NM), σ²_ρ/(NVMHR)) to produce shares of
# Var(θ̂) — the analyst's mean-variance decomposition. LPM shares are on
# the probability scale; GLMM shares are on the logit scale (units differ;
# the comparison is about preserved component rankings, not absolute SE).
#
# Reference: Nakagawa & Schielzeth (2010) — logistic GLMM level-1
# variance = π²/3 on the latent (logit) scale.
#
# Usage:
#   Rscript analysis/02b_glmm_robustness.R

library(tidyverse)
library(lme4)

set.seed(42)

cat("=== 02b_glmm_robustness.R ===\n")
cat("LPM vs GLMM Var(θ̂) decomposition for binary outcomes\n\n")

# Population variance helper (1/L, not 1/(L-1))
pop_var <- function(x) mean((x - mean(x))^2)

# --- TEE label mapping (shared) ---
tle_labels <- c(
  "category"                                  = "between-category",
  "item_id"                                   = "within-category item",
  "variant_id"                                = "prompt",
  "item_id:variant_id"                        = "item x prompt",
  "item_id:temperature"                       = "item x temperature",
  "variant_id:temperature"                    = "prompt x temperature",
  "item_id:judge_model"                       = "item x judge",
  "variant_id:judge_model"                    = "prompt x judge",
  "item_id:variant_id:judge_model:temperature" = "cell-level (3-way+)",
  "Residual"                                  = "replicate noise"
)

# Divisor for each component under the operational design (averaged projection)
component_divisor <- function(label, N, V, M, H, R, P = 1) {
  switch(label,
    "between-category"                  = N,
    "within-category item"              = N,
    "prompt"                            = V,
    "item x prompt"                     = N * V,
    "item x temperature"                = N * H,
    "prompt x temperature"              = V * H,
    "item x judge"                      = N * M,
    "prompt x judge"                    = V * M,
    "item x SUT"                        = N * M,
    "prompt x SUT"                      = V * M,
    "cell-level (3-way+)"               = N * V * H * M,
    "replicate noise"                   = N * V * H * M * R,
    "judge model (design sensitivity)"  = M,
    "SUT model (design sensitivity)"    = M,
    "temperature (design sensitivity)"  = H,
    "position (design sensitivity)"     = P,
    NA_real_
  )
}

# Compute fixed-effect sensitivity (population variance of cell means)
fixed_effect_sensitivity <- function(mod, term_prefix) {
  fe <- fixef(mod)
  # Fitted cell means at each level of the term: intercept + contrast (level k) + 0 for ref level
  contrasts_k <- fe[grepl(paste0("^", term_prefix), names(fe))]
  if (length(contrasts_k) == 0) return(0)
  # Cell means: ref level has coefficient 0; non-ref levels have their contrasts.
  # Fixed-effect intercept cancels in pop_var, so use contrasts_k augmented with 0
  cell_means <- c(0, contrasts_k)
  pop_var(cell_means)
}

# ======================================================================
# Helper: fit both LPM and GLMM, extract Var(θ̂) shares
# ======================================================================
fit_and_compare <- function(df, dataset_name, design, has_position = FALSE,
                             layer = "judge") {
  # layer = "judge" → factor name `judge_model`; layer = "sut" → `sut_model`.
  # Variance components map identically; only the column name changes.

  cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
  cat("Dataset:", dataset_name, "\n")
  cat(sprintf("  Operational design: N=%d, V=%d, M=%d, H=%d, R=%d%s\n",
              design$N, design$V, design$M, design$H, design$R,
              if (has_position) sprintf(", P=%d", design$P) else ""))
  cat("  Rows:", nrow(df), "\n")
  cat("  Outcome mean:", round(mean(df$outcome), 3), "\n\n")

  layer_col <- if (layer == "judge") "judge_model" else "sut_model"

  # --- Build formula ---
  fixed_part <- if (has_position) {
    sprintf("outcome ~ temperature + %s + true_order", layer_col)
  } else {
    sprintf("outcome ~ temperature + %s", layer_col)
  }

  random_part <- paste(
    "(1 | category)",
    "(1 | item_id)",
    "(1 | variant_id)",
    "(1 | item_id:variant_id)",
    "(1 | item_id:temperature)",
    "(1 | variant_id:temperature)",
    sprintf("(1 | item_id:%s)", layer_col),
    sprintf("(1 | variant_id:%s)", layer_col),
    sprintf("(1 | item_id:variant_id:%s:temperature)", layer_col),
    sep = " + "
  )

  full_formula <- as.formula(paste(fixed_part, "+", random_part))

  # Adjust tle_labels for SUT layer (rename judge → sut for this dataset)
  local_tle_labels <- tle_labels
  if (layer == "sut") {
    names(local_tle_labels) <- gsub("judge_model", "sut_model", names(local_tle_labels))
    local_tle_labels[grepl("item x judge|prompt x judge|cell-level", local_tle_labels)] <-
      gsub("judge", "SUT", local_tle_labels[grepl("item x judge|prompt x judge|cell-level", local_tle_labels)])
  }

  # ── LPM (lmer) ──────────────────────────────────────────────────────
  cat("  Fitting LPM (lmer)...\n")
  lpm_mod <- lmer(
    full_formula, data = df, REML = TRUE,
    control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000))
  )

  lpm_vc_random <- as.data.frame(VarCorr(lpm_mod)) %>%
    select(grp, vcov) %>%
    rename(component = grp, lpm_variance = vcov) %>%
    mutate(
      tle_label = local_tle_labels[component],
      tle_label = ifelse(is.na(tle_label), component, tle_label),
      lpm_variance = pmax(lpm_variance, 0)
    )

  # Add fixed-effect sensitivity indices (LPM)
  lpm_layer_sens <- fixed_effect_sensitivity(lpm_mod, layer_col)
  lpm_temp_sens  <- fixed_effect_sensitivity(lpm_mod, "temperature")
  layer_sens_label <- if (layer == "judge") "judge model (design sensitivity)" else "SUT model (design sensitivity)"
  lpm_vc <- bind_rows(
    lpm_vc_random,
    tibble(component = paste0(layer_col, "_fixed"), lpm_variance = lpm_layer_sens,
           tle_label = layer_sens_label),
    tibble(component = "temperature_fixed", lpm_variance = lpm_temp_sens,
           tle_label = "temperature (design sensitivity)")
  )
  if (has_position) {
    lpm_pos_sens <- fixed_effect_sensitivity(lpm_mod, "true_order")
    lpm_vc <- bind_rows(lpm_vc, tibble(
      component = "true_order_fixed", lpm_variance = lpm_pos_sens,
      tle_label = "position (design sensitivity)"))
  }

  # ── GLMM (glmer, binomial, logit) ───────────────────────────────────
  cat("  Fitting GLMM (glmer, binomial)...\n")

  glmm_mod <- NULL
  glmm_note <- "full model"

  # Attempt 1: full spec
  glmm_mod <- tryCatch(
    suppressWarnings(glmer(
      full_formula, data = df,
      family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa",
                             optCtrl = list(maxfun = 200000)),
      nAGQ = 0
    )),
    error = function(e) {
      cat("    full model failed:", conditionMessage(e), "\n"); NULL
    }
  )

  if (is.null(glmm_mod)) {
    cat("  *** GLMM fitting failed for", dataset_name, "***\n")
    return(NULL)
  }

  cat("  GLMM fit complete.\n")

  glmm_vc_random <- as.data.frame(VarCorr(glmm_mod)) %>%
    select(grp, vcov) %>%
    rename(component = grp, glmm_variance_logit = vcov) %>%
    mutate(
      tle_label = local_tle_labels[component],
      tle_label = ifelse(is.na(tle_label), component, tle_label),
      glmm_variance_logit = pmax(glmm_variance_logit, 0)
    )

  # Add level-1 logistic residual variance = π²/3 (Nakagawa & Schielzeth 2010)
  glmm_vc_random <- bind_rows(
    glmm_vc_random,
    tibble(component = "Residual",
           glmm_variance_logit = pi^2 / 3,
           tle_label = "replicate noise")
  )

  # Add fixed-effect sensitivity indices (GLMM logit scale)
  glmm_layer_sens <- fixed_effect_sensitivity(glmm_mod, layer_col)
  glmm_temp_sens  <- fixed_effect_sensitivity(glmm_mod, "temperature")
  glmm_vc <- bind_rows(
    glmm_vc_random,
    tibble(component = paste0(layer_col, "_fixed"), glmm_variance_logit = glmm_layer_sens,
           tle_label = layer_sens_label),
    tibble(component = "temperature_fixed", glmm_variance_logit = glmm_temp_sens,
           tle_label = "temperature (design sensitivity)")
  )
  if (has_position) {
    glmm_pos_sens <- fixed_effect_sensitivity(glmm_mod, "true_order")
    glmm_vc <- bind_rows(glmm_vc, tibble(
      component = "true_order_fixed", glmm_variance_logit = glmm_pos_sens,
      tle_label = "position (design sensitivity)"))
  }

  # ── Apply divisors → Var(θ̂) contributions ─────────────────────────
  with_div <- function(df_vc, var_col) {
    df_vc %>%
      mutate(divisor = sapply(tle_label, component_divisor,
                              N = design$N, V = design$V, M = design$M,
                              H = design$H, R = design$R,
                              P = if (!is.null(design$P)) design$P else 1L)) %>%
      mutate(contribution = .data[[var_col]] / divisor) %>%
      mutate(contribution = ifelse(is.na(contribution), 0, contribution))
  }

  lpm_div  <- with_div(lpm_vc,  "lpm_variance")
  glmm_div <- with_div(glmm_vc, "glmm_variance_logit")

  lpm_total  <- sum(lpm_div$contribution,  na.rm = TRUE)
  glmm_total <- sum(glmm_div$contribution, na.rm = TRUE)

  lpm_div  <- lpm_div  %>% mutate(lpm_pct  = round(100 * contribution / lpm_total,  2))
  glmm_div <- glmm_div %>% mutate(glmm_pct = round(100 * contribution / glmm_total, 2))

  # ── Merge and compare ──────────────────────────────────────────────
  comparison <- lpm_div %>%
    select(tle_label, lpm_variance, lpm_contribution = contribution, lpm_pct) %>%
    full_join(
      glmm_div %>% select(tle_label, glmm_variance_logit,
                           glmm_contribution = contribution, glmm_pct),
      by = "tle_label"
    ) %>%
    replace_na(list(
      lpm_variance = 0, lpm_contribution = 0, lpm_pct = 0,
      glmm_variance_logit = 0, glmm_contribution = 0, glmm_pct = 0
    )) %>%
    mutate(
      lpm_rank  = rank(-lpm_pct, ties.method = "min"),
      glmm_rank = rank(-glmm_pct, ties.method = "min"),
      rank_diff = lpm_rank - glmm_rank,
      dataset   = dataset_name,
      glmm_note = glmm_note
    ) %>%
    arrange(lpm_rank)

  # ── Print side-by-side comparison ──────────────────────────────────
  cat("\n  Var(θ̂) decomposition (LPM probability scale, GLMM logit scale):\n\n")
  print_tbl <- comparison %>%
    select(tle_label, lpm_pct, lpm_rank, glmm_pct, glmm_rank, rank_diff) %>%
    rename(Component = tle_label, `LPM %` = lpm_pct, `LPM rank` = lpm_rank,
           `GLMM %` = glmm_pct, `GLMM rank` = glmm_rank, `Rank diff` = rank_diff)
  print(as.data.frame(print_tbl), row.names = FALSE)

  cat(sprintf("\n  LPM  total Var(θ̂) = %.5f, SE = %.4f\n", lpm_total, sqrt(lpm_total)))
  cat(sprintf("  GLMM total Var(θ̂) on logit scale = %.5f\n", glmm_total))

  rho <- cor(comparison$lpm_pct, comparison$glmm_pct, method = "spearman")
  cat(sprintf("\n  Spearman rank correlation (Var(θ̂) shares): rho = %.4f\n", rho))

  return(comparison)
}


# ======================================================================
# Datasets
# ======================================================================
designs <- list(
  pairwise = list(N = 150, V = 5, M = 3, H = 3, R = 3, P = 2),
  safety   = list(N = 141, V = 5, M = 3, H = 3, R = 8),
  mmlu     = list(N = 200, V = 4, M = 3, H = 3, R = 8)   # v_3 excluded (structural check)
)

pairwise_path <- "data/processed/pairwise_clean.csv"
pairwise_result <- NULL
if (file.exists(pairwise_path)) {
  cat("Loading pairwise data from:", pairwise_path, "\n")
  df_pw <- read_csv(pairwise_path, show_col_types = FALSE) %>%
    filter(!is.na(outcome)) %>%
    mutate(item_id = as.factor(item_id), variant_id = as.factor(variant_id),
           temperature = as.factor(temperature), judge_model = as.factor(judge_model),
           category = as.factor(category), true_order = as.factor(true_order))
  pairwise_result <- fit_and_compare(df_pw, "pairwise", designs$pairwise,
                                      has_position = TRUE)
}

safety_path <- "data/processed/safety_clean.csv"
safety_result <- NULL
if (file.exists(safety_path)) {
  cat("Loading safety data from:", safety_path, "\n")
  df_sf <- read_csv(safety_path, show_col_types = FALSE) %>%
    filter(!is.na(outcome)) %>%
    mutate(item_id = as.factor(item_id), variant_id = as.factor(variant_id),
           temperature = as.factor(temperature), judge_model = as.factor(judge_model),
           category = as.factor(category))
  safety_result <- fit_and_compare(df_sf, "safety", designs$safety,
                                    has_position = FALSE)
}

mmlu_path <- "data/processed/mmlu_clean.csv"
mmlu_result <- NULL
if (file.exists(mmlu_path)) {
  cat("Loading mmlu data from:", mmlu_path, "\n")
  df_mm <- read_csv(mmlu_path, show_col_types = FALSE) %>%
    filter(!is.na(outcome)) %>%
    mutate(item_id = as.factor(item_id), variant_id = as.factor(variant_id),
           temperature = as.factor(temperature), sut_model = as.factor(sut_model),
           category = as.factor(category))
  mmlu_result <- fit_and_compare(df_mm, "mmlu", designs$mmlu,
                                  has_position = FALSE, layer = "sut")
}


# ======================================================================
# Save results
# ======================================================================
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

if (!is.null(pairwise_result)) {
  write_csv(pairwise_result, "data/processed/glmm_comparison_pairwise.csv")
  cat("\nSaved: data/processed/glmm_comparison_pairwise.csv\n")
}
if (!is.null(safety_result)) {
  write_csv(safety_result, "data/processed/glmm_comparison_safety.csv")
  cat("Saved: data/processed/glmm_comparison_safety.csv\n")
}
if (!is.null(mmlu_result)) {
  write_csv(mmlu_result, "data/processed/glmm_comparison_mmlu.csv")
  cat("Saved: data/processed/glmm_comparison_mmlu.csv\n")
}

cat("\nDone.\n")
