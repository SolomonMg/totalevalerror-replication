# test_variance_decomposition.R
# Integration test: generate synthetic data from known DGP,
# fit lmer, check estimated components within 20% of true values.
# Run: Rscript tests/test_variance_decomposition.R

library(lme4)

set.seed(42)

cat("=== Synthetic DGP Variance Decomposition Test ===\n")

# --- True variance components ---
true_s2_cat    <- 0.10  # between-category
true_s2_item   <- 0.15  # within-category item
true_s2_prompt <- 0.08  # prompt sensitivity
true_s2_ip     <- 0.03  # item × prompt interaction
true_s2_it     <- 0.02  # item × temperature interaction
true_s2_pt     <- 0.01  # prompt × temperature interaction
true_s2_gen    <- 0.20  # generation (residual)

# --- Design ---
n_cats    <- 5
n_per_cat <- 30  # items per category
n_items   <- n_cats * n_per_cat
n_prompts <- 5
n_temps   <- 3
n_reps    <- 8

# --- Generate data ---
cat("Generating synthetic data...\n")

# Category effects
cat_eff <- rnorm(n_cats, 0, sqrt(true_s2_cat))

# Item effects (nested in categories)
item_cat <- rep(1:n_cats, each = n_per_cat)
item_eff <- rnorm(n_items, 0, sqrt(true_s2_item))

# Prompt effects
prompt_eff <- rnorm(n_prompts, 0, sqrt(true_s2_prompt))

# Temperature fixed effects
temp_levels <- c(0.0, 0.7, 1.0)
temp_eff <- c(0, 0.1, -0.05)  # fixed effects

# Judge model fixed effects (2 models)
judge_eff <- c(0, 0.08)

# Interaction effects
ip_eff <- matrix(rnorm(n_items * n_prompts, 0, sqrt(true_s2_ip)), n_items, n_prompts)
it_eff <- matrix(rnorm(n_items * n_temps, 0, sqrt(true_s2_it)), n_items, n_temps)
pt_eff <- matrix(rnorm(n_prompts * n_temps, 0, sqrt(true_s2_pt)), n_prompts, n_temps)

mu <- 3.0  # grand mean

rows <- list()
for (i in 1:n_items) {
  for (j in 1:n_prompts) {
    for (k in 1:n_temps) {
      for (m in 1:2) {  # 2 judge models
        for (r in 1:n_reps) {
          y <- mu + cat_eff[item_cat[i]] + item_eff[i] + prompt_eff[j] +
               temp_eff[k] + judge_eff[m] +
               ip_eff[i, j] + it_eff[i, k] + pt_eff[j, k] +
               rnorm(1, 0, sqrt(true_s2_gen))
          rows[[length(rows) + 1]] <- data.frame(
            item_id     = paste0("item_", i),
            category    = paste0("cat_", item_cat[i]),
            variant_id  = paste0("v_", j),
            temperature = as.factor(temp_levels[k]),
            judge_model = paste0("judge_", m),
            replication = r,
            outcome     = y
          )
        }
      }
    }
  }
}

df <- do.call(rbind, rows)
cat("Generated", nrow(df), "rows\n")

# --- Fit model ---
cat("Fitting lmer...\n")
mod <- lmer(
  outcome ~ temperature + judge_model +
    (1 | category) +
    (1 | item_id) +
    (1 | variant_id) +
    (1 | item_id:variant_id) +
    (1 | item_id:temperature) +
    (1 | variant_id:temperature),
  data = df,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 20000))
)

# --- Extract estimates ---
vc <- as.data.frame(VarCorr(mod))
est <- setNames(vc$vcov, vc$grp)

# --- Check estimates within 20% of truth ---
n_pass <- 0
n_fail <- 0

check <- function(label, estimated, true_val, tolerance = 0.20) {
  rel_error <- abs(estimated - true_val) / true_val
  pass <- rel_error <= tolerance
  status <- if (pass) "PASS" else "FAIL"
  cat(sprintf("  %s %s: est=%.4f true=%.4f rel_err=%.1f%%\n",
              status, label, estimated, true_val, rel_error * 100))
  if (pass) n_pass <<- n_pass + 1 else n_fail <<- n_fail + 1
}

cat("\n=== Checking variance components ===\n")
# Between-category has wider tolerance: only 5 groups → REML underestimates
check("between-category",     est["category"],             true_s2_cat, tolerance = 0.50)
check("within-category item", est["item_id"],              true_s2_item)
check("prompt",               est["variant_id"],           true_s2_prompt)
check("item x prompt",        est["item_id:variant_id"],   true_s2_ip)
check("item x temperature",   est["item_id:temperature"],  true_s2_it)
check("prompt x temperature", est["variant_id:temperature"], true_s2_pt)
check("generation",           est["Residual"],             true_s2_gen)

cat(sprintf("\n%d passed, %d failed\n", n_pass, n_fail))
if (n_fail > 0) quit(status = 1)
cat("All tests passed.\n")
