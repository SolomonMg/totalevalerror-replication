# 02g_mmlu_v3_sensitivity.R
# SI robustness for the MMLU variant exclusion (SI si:mmlu_parsing, tab:mmlu_v5).
# The primary analysis uses the four admissible prompt variants; v_3 ("Minimal", no
# letter-only instruction) fails the structural check. This script refits the MMLU
# G-study on all five variants, with unanswered responses (a) excluded and (b) scored
# incorrect, and reports each component's share of Var(theta-hat) at the operational
# design with V = 5.
#
# The model formula, TEE labels, and fixed-effect sensitivity (population variance)
# follow analysis/02_variance_decomposition_mmlu.R lines 49 - 145.
#
# Output: data/processed/mmlu_v5_sensitivity.csv (treatment, tle_label, variance, pct_var_theta)
# Usage:  Rscript analysis/02g_mmlu_v3_sensitivity.R

suppressPackageStartupMessages({ library(tidyverse); library(lme4) })
set.seed(42)
cat("=== 02g_mmlu_v3_sensitivity.R ===\n")

d <- read_csv("data/processed/mmlu_clean_v5.csv", show_col_types = FALSE)
stopifnot(n_distinct(d$variant_id) == 5)

tle_labels <- c(
  "category"                                = "between-category",
  "item_id"                                 = "within-category item",
  "variant_id"                              = "prompt",
  "item_id:variant_id"                      = "item x prompt",
  "item_id:temperature"                     = "item x temperature",
  "variant_id:temperature"                  = "prompt x temperature",
  "item_id:sut_model"                       = "item x SUT",
  "variant_id:sut_model"                    = "prompt x SUT",
  "item_id:variant_id:sut_model:temperature" = "cell-level (3-way+)",
  "Residual"                                = "replicate noise"
)
N <- 200; V <- 5; M <- 3; H <- 3; R <- 8
divisor <- c("between-category" = N, "within-category item" = N, "prompt" = V,
             "item x prompt" = N * V, "item x temperature" = N * H, "prompt x temperature" = V * H,
             "item x SUT" = N * M, "prompt x SUT" = V * M, "cell-level (3-way+)" = N * V * H * M,
             "replicate noise" = N * V * H * M * R, "SUT model (design sensitivity)" = M,
             "temperature (design sensitivity)" = H)
pop_var <- function(x) mean((x - mean(x))^2)

fit_shares <- function(df, treatment) {
  df <- df %>% mutate(across(c(item_id, variant_id, temperature, sut_model, category), as.factor))
  mod <- lmer(outcome ~ temperature + sut_model +
                (1 | category) + (1 | item_id) + (1 | variant_id) +
                (1 | item_id:variant_id) + (1 | item_id:temperature) +
                (1 | variant_id:temperature) + (1 | item_id:sut_model) +
                (1 | variant_id:sut_model) + (1 | item_id:variant_id:sut_model:temperature),
              data = df, REML = TRUE,
              control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  vc <- as.data.frame(VarCorr(mod)) %>% transmute(tle_label = tle_labels[grp], variance = pmax(vcov, 0))
  fe <- fixef(mod); gm <- fe["(Intercept)"]
  vc <- bind_rows(vc,
    tibble(tle_label = "temperature (design sensitivity)",
           variance = pop_var(c(gm, gm + fe[grepl("temperature", names(fe))]))),
    tibble(tle_label = "SUT model (design sensitivity)",
           variance = pop_var(c(gm, gm + fe[grepl("sut_model", names(fe))]))))
  contrib <- vc$variance / divisor[vc$tle_label]
  vc %>% mutate(treatment = treatment, n_obs = nrow(df), mean_outcome = mean(df$outcome),
                pct_var_theta = 100 * contrib / sum(contrib), .before = 1)
}

res <- bind_rows(
  fit_shares(filter(d, !is.na(outcome)), "unanswered_missing"),
  fit_shares(mutate(d, outcome = coalesce(outcome, 0)), "unanswered_incorrect")
)
print(res %>% select(treatment, tle_label, pct_var_theta) %>%
        pivot_wider(names_from = treatment, values_from = pct_var_theta) %>%
        arrange(desc(unanswered_missing)), n = 20)
write_csv(res, "data/processed/mmlu_v5_sensitivity.csv")
cat("Saved: data/processed/mmlu_v5_sensitivity.csv\n")
