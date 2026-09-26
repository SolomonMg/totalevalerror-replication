# 17_sensitivity_tier_matched.R
# Tier-matched judge sensitivity analysis for safety classification
#
# Compares item×judge interaction across 3 judge tiers (same items):
#   - Original (mixed-tier): GPT-4o, Gemini 2.0 Flash, Claude Haiku 4.5
#   - Closed frontier: GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6
#   - Open-weight: GPT-oss-120b, Gemma 4 31B, DeepSeek 3.2
#
# Tests whether the item×judge interaction (44% in the full safety run)
# is a capability-tier artifact or genuine measurement disagreement.

library(tidyverse)
library(jsonlite)
library(lme4)

set.seed(42)

cat("=== 17_sensitivity_tier_matched.R ===\n")

# =========================================================================
# 1. Load and clean data
# =========================================================================

raw_path <- "data/raw/safety_sensitivity_pilot.jsonl"
out_dir  <- "data/processed"
fig_dir  <- "figures"

lines <- readLines(raw_path)
df_raw <- map_dfr(lines, ~ fromJSON(.x, flatten = TRUE))

cat(sprintf("Raw rows: %d\n", nrow(df_raw)))

# Parse SAFE/UNSAFE from response
df <- df_raw %>%
  mutate(
    response_clean = str_to_upper(str_trim(response)),
    outcome = case_when(
      str_starts(response_clean, "UNSAFE") ~ 0L,
      str_starts(response_clean, "SAFE")   ~ 1L,
      TRUE ~ NA_integer_
    ),
    judge_short = str_extract(model, "[^/]+$"),
    temperature = as.factor(temperature),
    variant_id  = as.character(variant_id)
  ) %>%
  filter(!is.na(outcome))

cat(sprintf("Valid rows: %d (%.1f%% parse rate)\n",
            nrow(df), nrow(df) / nrow(df_raw) * 100))

# Define tiers
tier_map <- tribble(
  ~judge_short,            ~tier,
  "gpt-4o",                "original",
  "gemini-2.0-flash-001",  "original",
  "claude-haiku-4.5",      "original",
  "gpt-5.4",               "closed_frontier",
  "gemini-3.1-pro-preview","closed_frontier",
  "claude-opus-4-6",       "closed_frontier",
  "gpt-oss-120b",          "open_weight",
  "gemma-4-31b-it",        "open_weight",
  "deepseek-v3.2",         "open_weight",
)

df <- df %>% left_join(tier_map, by = "judge_short")

# Summary
cat("\nPer-model valid responses:\n")
df %>% count(tier, judge_short) %>% print(n = 20)

cat(sprintf("\nOverall safe rate: %.1f%%\n", mean(df$outcome) * 100))
cat("Safe rate by tier:\n")
df %>% group_by(tier) %>%
  summarize(safe_rate = mean(outcome), n = n(), .groups = "drop") %>%
  print()

# =========================================================================
# 2. Variance decomposition per tier
# =========================================================================

run_decomposition <- function(data, tier_name) {
  cat(sprintf("\n--- %s tier (N=%d) ---\n", tier_name, nrow(data)))

  # Check we have 3 judges
  judges <- unique(data$judge_short)
  cat(sprintf("Judges: %s\n", paste(judges, collapse = ", ")))

  fit <- tryCatch(
    lmer(outcome ~ temperature + judge_short +
           (1 | category) +
           (1 | item_id) +
           (1 | variant_id) +
           (1 | item_id:variant_id) +
           (1 | item_id:temperature) +
           (1 | variant_id:temperature) +
           (1 | item_id:judge_short) +
           (1 | variant_id:judge_short),
         data = data, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa",
                               optCtrl = list(maxfun = 50000))),
    error = function(e) {
      cat(sprintf("  lmer FAILED: %s\n", e$message))
      return(NULL)
    }
  )

  if (is.null(fit)) return(NULL)

  # Extract variance components
  vc <- as.data.frame(VarCorr(fit))
  vc$pct <- vc$vcov / sum(vc$vcov) * 100

  # Map to TEE labels
  label_map <- c(
    "item_id:judge_short"    = "item x judge",
    "item_id:variant_id"     = "item x prompt",
    "item_id:temperature"    = "item x temperature",
    "variant_id:judge_short" = "prompt x judge",
    "variant_id:temperature" = "prompt x temperature",
    "item_id"                = "within-category item",
    "category"               = "between-category",
    "variant_id"             = "prompt",
    "Residual"               = "generation"
  )

  vc$tle_label <- label_map[vc$grp]
  vc$tier <- tier_name

  # Fixed-effect sensitivity (judge model)
  fe <- fixef(fit)
  judge_fe <- fe[grep("judge_short", names(fe))]
  if (length(judge_fe) > 0) {
    all_judge_effects <- c(0, judge_fe)  # reference level = 0
    s2_judge <- mean((all_judge_effects - mean(all_judge_effects))^2)
  } else {
    s2_judge <- 0
  }

  judge_row <- data.frame(
    grp = "judge_short (fixed)", var1 = NA, var2 = NA,
    vcov = s2_judge, sdcor = sqrt(s2_judge),
    pct = s2_judge / (sum(vc$vcov) + s2_judge) * 100,
    tle_label = "judge model (design sensitivity)",
    tier = tier_name
  )

  # Recalculate percentages with judge fixed effect
  total_var <- sum(vc$vcov) + s2_judge
  vc$pct <- vc$vcov / total_var * 100
  vc <- bind_rows(vc, judge_row)

  cat("\nVariance components:\n")
  vc %>%
    filter(!is.na(tle_label)) %>%
    select(tle_label, vcov, pct) %>%
    arrange(desc(pct)) %>%
    mutate(vcov = round(vcov, 6), pct = round(pct, 1)) %>%
    as.data.frame() %>%
    print()

  return(vc)
}

tiers <- c("original", "closed_frontier", "open_weight")
results <- list()

for (tier_name in tiers) {
  tier_data <- df %>% filter(tier == tier_name)
  vc <- run_decomposition(tier_data, tier_name)
  if (!is.null(vc)) results[[tier_name]] <- vc
}

# =========================================================================
# 3. Side-by-side comparison
# =========================================================================

if (length(results) > 0) {
  comparison <- bind_rows(results) %>%
    select(tier, tle_label, vcov, pct) %>%
    filter(!is.na(tle_label))

  cat("\n\n=== SIDE-BY-SIDE COMPARISON ===\n")

  wide <- comparison %>%
    select(tier, tle_label, pct) %>%
    pivot_wider(names_from = tier, values_from = pct, values_fill = 0) %>%
    arrange(desc(original))

  print(wide, n = 15)

  # Key number: item×judge across tiers
  ij <- comparison %>% filter(tle_label == "item x judge")
  cat("\n=== KEY FINDING: item x judge interaction ===\n")
  for (i in seq_len(nrow(ij))) {
    cat(sprintf("  %s: %.1f%%\n", ij$tier[i], ij$pct[i]))
  }

  # Save
  write_csv(comparison, file.path(out_dir, "sensitivity_tier_matched.csv"))
  write_csv(wide, file.path(out_dir, "sensitivity_tier_matched_wide.csv"))
  cat("\nSaved: sensitivity_tier_matched.csv, sensitivity_tier_matched_wide.csv\n")

  # =======================================================================
  # 4. D-study contribution shares (THESE are the numbers in Table
  #    tab:tier_sensitivity). The raw component shares above are NOT what
  #    the table reports: the table divides each component by the design
  #    factors it averages over, at the operational design
  #    N=24, V=3, M=3, H=3, R=5 with judges and temperatures averaged, then
  #    takes shares of the resulting Var(theta-hat).
  # =======================================================================
  Nd <- 24; Vd <- 3; Md <- 3; Hd <- 3; Rd <- 5

  dstudy_share <- function(tier_vc) {
    g <- function(lbl) {
      v <- tier_vc$vcov[tier_vc$tle_label == lbl]
      if (length(v) == 0) 0 else v[1]
    }
    contrib <- c(
      "within-category item"             = g("within-category item") / Nd,
      "between-category"                 = g("between-category") / Nd,
      "item x judge"                     = g("item x judge") / (Nd * Md),
      "item x prompt"                    = g("item x prompt") / (Nd * Vd),
      "prompt x judge"                   = g("prompt x judge") / (Vd * Md),
      "item x temperature"               = g("item x temperature") / (Nd * Hd),
      "prompt x temperature"             = g("prompt x temperature") / (Vd * Hd),
      "judge model (design sensitivity)" = g("judge model (design sensitivity)") / Md,
      "prompt"                           = g("prompt") / Vd,
      "residual (pooled)"                = g("generation") / (Nd * Vd * Md * Hd * Rd)
    )
    100 * contrib / sum(contrib)
  }

  dstudy_tbl <- imap_dfr(results, ~ tibble(
    tier = .y, tle_label = names(dstudy_share(.x)), pct_var_theta = dstudy_share(.x)
  ))
  dstudy_wide <- dstudy_tbl %>%
    pivot_wider(names_from = tier, values_from = pct_var_theta, values_fill = 0) %>%
    arrange(desc(closed_frontier))

  cat("\n=== D-STUDY CONTRIBUTION SHARES (Table tab:tier_sensitivity) ===\n")
  print(dstudy_wide, n = 15)
  write_csv(dstudy_wide,
            file.path(out_dir, "sensitivity_tier_matched_dstudy_shares.csv"))
  cat("\nSaved: sensitivity_tier_matched_dstudy_shares.csv (matches the paper table)\n")
}
