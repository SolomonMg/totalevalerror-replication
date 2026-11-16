# 02e_prompt_breadth_sensitivity.R
# Prompt-variant breadth sensitivity (SI si:prompt_admissibility, tab:breadth_sensitivity;
# main text sec:gstudy admissibility paragraph). Promoted from the NeurIPS 2026 rebuttal
# (rebuttal_neurips2026/analyze_A.R, "Analysis A").
#
# For each domain (safety, mmlu) x set (narrow, broad-wording, broad-construct[, broadplus]):
#   - fit the variance decomposition, extract sigma^2_phi (variant main) and
#     sigma^2_alphaphi (item x variant)
#   - behavioral measures: mean pairwise Cohen's kappa across variants + the
#     prompt-facet generalizability coefficient (item reliability across variants)
#   - attach the embedding-cosine anchor (the negative finding), when the rebuttal
#     JSONs are present
# Broad-construct = the main-text framing data (MMLU subset to DeepSeek; safety subset
# to the two judges that were re-collected).
#
# Output: data/processed/prompt_breadth_sensitivity.csv
#         data/processed/prompt_breadth_sensitivity_boot.csv   (with --boot N)
# Usage:  Rscript analysis/02e_prompt_breadth_sensitivity.R   (3 - 5 min)
#         Rscript analysis/02e_prompt_breadth_sensitivity.R --boot 200 --cores 8
#         (--boot N adds parametric-bootstrap 95% CIs for the MMLU prompt share)

suppressPackageStartupMessages({ library(tidyverse); library(lme4); library(jsonlite) })
cli <- commandArgs(trailingOnly = TRUE)
n_boot  <- if ("--boot" %in% cli)  as.integer(cli[which(cli == "--boot") + 1])  else 0L
n_cores <- if ("--cores" %in% cli) as.integer(cli[which(cli == "--cores") + 1]) else 8L
setwd(rprojroot::find_root(rprojroot::has_file("run_all.sh")))
HERE <- "rebuttal_neurips2026"

# ---- cosine anchors (negative finding) ----
saf_ax_path <- file.path(HERE, "safety_diversity_axis.json")
mml_ax_path <- file.path(HERE, "mmlu_variant_sets.json")
if (file.exists(saf_ax_path) && file.exists(mml_ax_path)) {
saf_ax <- fromJSON(saf_ax_path)
mml_ax <- fromJSON(mml_ax_path)
cosine <- tibble(
  domain = c("safety","safety","safety","mmlu","mmlu","mmlu"),
  set    = rep(c("narrow","broad_wording","broad_construct"), 2),
  mean_cosine = c(saf_ax$sets$narrow$mean_pairwise_cosine,
                  saf_ax$sets$broad_wording$mean_pairwise_cosine,
                  saf_ax$sets$broad_construct$mean_pairwise_cosine,
                  mml_ax$sets$narrow$mean_pairwise_cosine,
                  mml_ax$sets$broad_wording$mean_pairwise_cosine,
                  mml_ax$sets$broad_construct$mean_pairwise_cosine))
} else {
  cosine <- tibble(domain = character(), set = character(), mean_cosine = numeric())
}

# ---- helpers ----
mean_pairwise_kappa <- function(df) {
  # df: item_id, variant_id, outcome (binary). Aggregate to (item,variant) majority,
  # then mean pairwise Cohen's kappa across variant columns.
  agg <- df %>% filter(!is.na(outcome)) %>%
    group_by(item_id, variant_id) %>%
    summarize(p = mean(outcome), .groups = "drop") %>%
    mutate(cls = as.integer(p >= 0.5)) %>%
    select(item_id, variant_id, cls) %>%
    pivot_wider(names_from = variant_id, values_from = cls) %>%
    select(-item_id)
  vs <- names(agg); ks <- c()
  for (i in seq_along(vs)) for (j in seq_along(vs)) if (i < j) {
    a <- agg[[vs[i]]]; b <- agg[[vs[j]]]; ok <- !is.na(a) & !is.na(b)
    a <- a[ok]; b <- b[ok]
    po <- mean(a == b)
    pa <- mean(a); pb <- mean(b)
    pe <- pa*pb + (1-pa)*(1-pb)
    ks <- c(ks, if (pe < 1) (po - pe)/(1 - pe) else NA)
  }
  mean(ks, na.rm = TRUE)
}

prompt_gen_coef <- function(df) {
  # Item reliability across variants: fit mean_outcome ~ (1|item)+(1|variant) on
  # (item,variant) aggregates; G = s2_item/(s2_item+s2_variant+s2_resid) (single-variant).
  agg <- df %>% filter(!is.na(outcome)) %>%
    group_by(item_id, variant_id) %>% summarize(p = mean(outcome), .groups = "drop") %>%
    mutate(item_id = factor(item_id), variant_id = factor(variant_id))
  m <- tryCatch(lmer(p ~ (1|item_id) + (1|variant_id), data = agg,
                     control = lmerControl(calc.derivs = FALSE)), error = function(e) NULL)
  if (is.null(m)) return(NA)
  v <- as.data.frame(VarCorr(m))
  s2_item <- v$vcov[v$grp=="item_id"]; s2_var <- v$vcov[v$grp=="variant_id"]
  s2_res  <- v$vcov[v$grp=="Residual"]
  s2_item / (s2_item + s2_var + s2_res)
}

fit_decomp <- function(df, domain) {
  df <- df %>% filter(!is.na(outcome)) %>%
    mutate(item_id=factor(item_id), variant_id=factor(variant_id),
           temperature=factor(temperature), category=factor(category))
  if (domain == "safety") {
    df$judge_model <- factor(df$judge_model)
    form <- outcome ~ temperature + judge_model + (1|category) + (1|item_id) +
      (1|variant_id) + (1|item_id:variant_id) + (1|item_id:temperature) +
      (1|variant_id:temperature) + (1|item_id:judge_model) + (1|variant_id:judge_model) +
      (1|item_id:variant_id:judge_model:temperature)
  } else {
    form <- outcome ~ temperature + (1|category) + (1|item_id) + (1|variant_id) +
      (1|item_id:variant_id) + (1|item_id:temperature) + (1|variant_id:temperature) +
      (1|item_id:variant_id:temperature)
  }
  lmer(form, data = df, REML = TRUE,
       control = lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=5e4)))
}

decompose <- function(df, domain) {
  m <- fit_decomp(df, domain)
  v <- as.data.frame(VarCorr(m))
  gv <- function(g){ x <- v$vcov[v$grp==g]; if(length(x)==0) 0 else x }
  tibble(sigma2_phi = gv("variant_id"),
         sigma2_alphaphi = gv("item_id:variant_id"),
         sigma2_item = gv("item_id"),
         total_var = sum(pmax(v$vcov,0)),
         pct_phi = 100*gv("variant_id")/sum(pmax(v$vcov,0)),
         pct_alphaphi = 100*gv("item_id:variant_id")/sum(pmax(v$vcov,0)))
}

SAFETY_JUDGES2 <- c("google/gemini-3-flash-preview", "openai/gpt-oss-120b")  # trinity pulled
prep_set <- function(path, domain, subset_deepseek=FALSE) {
  df <- read_csv(path, show_col_types = FALSE)
  if (domain=="mmlu" && subset_deepseek && "sut_short" %in% names(df))
    df <- df %>% filter(sut_short=="deepseek-chat-v3.1")
  # subset existing framing safety data to the 2 surviving judges for apples-to-apples
  if (domain=="safety" && "judge_model" %in% names(df))
    df <- df %>% filter(judge_model %in% SAFETY_JUDGES2)
  if (!"category" %in% names(df)) df$category <- "all"
  df
}

analyze_set <- function(path, domain, set, subset_deepseek=FALSE) {
  df <- prep_set(path, domain, subset_deepseek)
  dec <- decompose(df, domain)
  bind_cols(tibble(domain=domain, set=set,
                   n_obs=sum(!is.na(df$outcome)),
                   parse_rate=mean(!is.na(df$outcome)),
                   mean_outcome=mean(df$outcome, na.rm=TRUE),
                   kappa=mean_pairwise_kappa(df),
                   gen_coef=prompt_gen_coef(df)), dec)
}

sets <- tribble(
  ~path, ~domain, ~set, ~ds,
  "data/processed/safety_clean_narrow.csv",           "safety","narrow", FALSE,
  "data/processed/safety_clean_broad_wording.csv", "safety","broad_wording", FALSE,
  "data/processed/safety_clean.csv",                  "safety","broad_construct", FALSE,
  "data/processed/mmlu_clean_narrow.csv",             "mmlu","narrow", TRUE,
  "data/processed/mmlu_clean_broad_wording.csv",     "mmlu","broad_wording", TRUE,
  "data/processed/mmlu_clean.csv",                    "mmlu","broad_construct", TRUE,
  "data/processed/mmlu_clean_broadplus.csv",         "mmlu","broadplus", TRUE,
)

res <- pmap_dfr(sets, function(path, domain, set, ds) {
  if (!file.exists(path)) { message("MISSING: ", path); return(tibble()) }
  message("Analyzing ", domain, "/", set)
  analyze_set(path, domain, set, ds)
})

out <- res %>% left_join(cosine, by=c("domain","set")) %>%
  select(domain, set, mean_cosine, kappa, gen_coef, sigma2_phi, pct_phi,
         sigma2_alphaphi, pct_alphaphi, sigma2_item, mean_outcome, parse_rate, n_obs)

out_path <- "data/processed/prompt_breadth_sensitivity.csv"
write_csv(out, out_path)
cat("\n================ ANALYSIS A SUMMARY ================\n")
print(as.data.frame(out), digits=3)
cat("\nSaved ", out_path, "\n")
cat("\nReading order: within a domain, narrow -> broad_wording -> broad_construct.\n")
cat("Expect kappa/gen_coef to DROP and sigma2_phi/alphaphi to RISE as the set broadens;\n")
cat("cosine does NOT track this monotonically (the negative finding).\n")

# ---- optional: parametric-bootstrap CIs for the MMLU prompt share (--boot N) ----
if (n_boot > 0) {
  boot_stat <- function(m) {
    v <- as.data.frame(VarCorr(m)); tot <- sum(pmax(v$vcov, 0))
    gv <- function(g) { x <- v$vcov[v$grp == g]; if (length(x) == 0) 0 else x }
    a <- gv("variant_id") + gv("item_id:variant_id")
    c(share = 100 * a / tot, abs = a)
  }
  boot_rows <- sets %>% filter(domain == "mmlu") %>% pmap_dfr(function(path, domain, set, ds) {
    message("Bootstrapping ", domain, "/", set, " (nsim = ", n_boot, ")")
    m <- fit_decomp(prep_set(path, domain, ds), domain)
    b <- bootMer(m, boot_stat, nsim = n_boot, seed = 42, use.u = FALSE, type = "parametric",
                 parallel = "multicore", ncpus = n_cores)
    ok <- stats::complete.cases(b$t)
    tibble(domain = domain, set = set, stat = c("share", "abs"), est = unname(b$t0),
           lo = apply(b$t[ok, , drop = FALSE], 2, quantile, 0.025),
           hi = apply(b$t[ok, , drop = FALSE], 2, quantile, 0.975),
           nsim = n_boot, n_ok = sum(ok))
  })
  print(as.data.frame(boot_rows), digits = 3)
  write_csv(boot_rows, "data/processed/prompt_breadth_sensitivity_boot.csv")
  cat("Saved data/processed/prompt_breadth_sensitivity_boot.csv\n")
}
