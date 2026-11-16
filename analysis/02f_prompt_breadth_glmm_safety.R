# 02f_prompt_breadth_glmm_safety.R
# Logit-scale check of the safety prompt-breadth result (SI si:prompt_admissibility,
# caveats). Refits the 02e safety decomposition with glmer(binomial) on the three safety
# variant sets (narrow, broad wording, main-text framing subset to the two re-collected
# judges) and reports the prompt-attributable share of per-observation variance on the
# logit scale. GLMM specification follows analysis/02b_glmm_robustness.R: bobyqa,
# nAGQ = 0, level-1 residual fixed at pi^2/3 (nakagawa2010repeatability).
#
# Output: data/processed/prompt_breadth_glmm_safety.csv
# Usage:  Rscript analysis/02f_prompt_breadth_glmm_safety.R

suppressPackageStartupMessages({ library(tidyverse); library(lme4) })
set.seed(42)
cat("=== 02f_prompt_breadth_glmm_safety.R ===\n")

SAFETY_JUDGES2 <- c("google/gemini-3-flash-preview", "openai/gpt-oss-120b")   # as in 02e
sets <- tribble(
  ~set,              ~path,
  "narrow",          "data/processed/safety_clean_narrow.csv",
  "broad_wording",   "data/processed/safety_clean_broad_wording.csv",
  "broad_construct", "data/processed/safety_clean.csv"
)
form <- outcome ~ temperature + judge_model + (1|category) + (1|item_id) +
  (1|variant_id) + (1|item_id:variant_id) + (1|item_id:temperature) +
  (1|variant_id:temperature) + (1|item_id:judge_model) + (1|variant_id:judge_model) +
  (1|item_id:variant_id:judge_model:temperature)

fit_one <- function(set, path) {
  df <- read_csv(path, show_col_types = FALSE) %>%
    filter(judge_model %in% SAFETY_JUDGES2, !is.na(outcome))
  if (!"category" %in% names(df)) df$category <- "all"
  df <- df %>% mutate(across(c(item_id, variant_id, temperature, category, judge_model), factor))
  cat(sprintf("  %s: %d rows, safe rate %.3f\n", set, nrow(df), mean(df$outcome)))
  warns <- character()
  m <- withCallingHandlers(
    glmer(form, data = df, family = binomial(link = "logit"), nAGQ = 0,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  v <- as.data.frame(VarCorr(m))
  gv <- function(g) { x <- v$vcov[v$grp == g]; if (length(x) == 0) 0 else max(x, 0) }
  total <- sum(pmax(v$vcov, 0)) + pi^2 / 3
  tibble(set = set, n_obs = nrow(df),
         sigma2_phi_logit = gv("variant_id"), sigma2_alphaphi_logit = gv("item_id:variant_id"),
         total_logit = total,
         pct_prompt_logit = 100 * (gv("variant_id") + gv("item_id:variant_id")) / total,
         converged = !any(grepl("converge", warns)), singular = isSingular(m),
         warnings = paste(unique(warns), collapse = " | "))
}

res <- pmap_dfr(sets, fit_one)
print(as.data.frame(res %>% select(-warnings)), digits = 4)
write_csv(res, "data/processed/prompt_breadth_glmm_safety.csv")
cat("Saved: data/processed/prompt_breadth_glmm_safety.csv\n")
