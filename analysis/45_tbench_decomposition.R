#!/usr/bin/env Rscript
# 45_tbench_decomposition.R
# Terminal-Bench 2.0: how much does a model's score move with the harness, once trial noise and tasks are
# separated, and what does that do to a one-harness leaderboard?
#
# Data: the 50 agent x model cells of the Terminal-Bench 2.0 paper's job files (experiments/fetch_terminalbench.py):
# 17 models across six harnesses (Terminus 2, mini-SWE-agent, OpenHands 0.60.0, Claude Code 2.0.31, Codex CLI
# 0.53.0, Gemini CLI 0.11.3), 89 tasks, about 5 trials per task and cell, 22,198 trials, run 2025-10-31 to 11-05.
# The vendor CLIs ran only their own vendor's models. No job file sets temperature or reasoning effort.
#
# 1. Variance components. Linear probability model fit by REML (lme4) on trial outcomes y (0/1):
#      y ~ model + harness + (1 | task) + (1 | task:model) + (1 | task:harness) + (1 | model:harness)
#          + (1 | task:model:harness),   residual = trial-to-trial noise
#    Model and harness are fixed; only the model x harness component (s2_mh) reorders models within a
#    harness. Fit on several sets of cells (SETS):
#      matched (primary)  24 cells where a model's harnesses send the same temperature and reasoning effort or
#                         thinking setting (code reading in experiments/fetch_terminalbench.py): GPT-5, -Codex,
#                         -mini, -nano under Terminus 2, mini-SWE-agent, Codex CLI (temperature not settable,
#                         effort at the API default); Gemini 2.5 Pro and Flash under mini-SWE-agent and Gemini CLI
#                         (temperature 0, dynamic thinking); Claude Sonnet 4.5 and Haiku 4.5 under OpenHands and
#                         Claude Code (API-default temperature, no thinking); Claude Opus 4.1 and Grok 4 under
#                         mini-SWE-agent and OpenHands (temperature 0, no thinking)
#      all (alternative)  all 50 cells: each harness with whatever settings it sends by default
#      sensitivity        matched without OpenHands; effort matched (all cells but OpenHands x GPT-5, -mini,
#                         -nano, Gemini 2.5, which get effort high or fixed thinking budgets); GPT-5 family only;
#                         the three open harnesses; all cells leaving out each harness in turn
#    95% interval for s2_mh from the cell means: regress the cell accuracies on model + harness; the residual
#    mean square RMS on df = cells - rank estimates s2_mh + noise, noise = mean_c (s2_tmh + s2_e mean_t(1/n_ct)) / N
#    (REML values). Interval (df RMS / qchisq(c(.975, .025), df) - noise), as in 42/44. Its point value equals
#    the REML estimate here. (lme4's profile interval is centred on the ML estimate, which ignores the 22 fixed
#    effects and gives 3.0 instead of 3.95 points, so we do not use it.)
# 2. One harness. The 17 models under Terminus 2 (the only harness that ran all of them), ranked by accuracy
#    (mean over tasks of the per-task mean reward, as on the leaderboard). For each pair, three SEs of the gap:
#      trial    sqrt(se_a^2 + se_b^2), se = sqrt(sum_t p_t (1 - p_t) / (n_t - 1)) / N: the leaderboard's
#               interval, which holds the 89 tasks fixed
#      paired   sd_t(D_t) / sqrt(N), D_t = difference in per-task mean reward: tasks as a sample
#      tee      sqrt(paired^2 + 2 s2_mh): adds how much a two-model gap moves from harness to harness
#    Rank intervals as in 43/44: [1 + models significantly above, 1 + models not significantly below].
# 3. Cross-check of 2 s2_mh without the mixed model: for every model pair that shares at least three harnesses,
#    the harness x task matrix of D is decomposed as in 42/44 (two-way ANOVA, s2_v = harness x pair variance).
# 4. 2,000 bootstrap resamples of tasks (seed 42) for the Terminus 2 accuracies, gaps and headline counts.
#    s2_mh is held at its full-sample value, so these intervals omit its uncertainty (see its chi-square interval).
#
# Caveats: each cell is one Harbor job, so a shift of a whole job (API trouble, sandbox failures) is
# indistinguishable from the harness and sits in s2_mh. The leaderboard and the paper's Table 2 disagree on all
# OpenHands 0.60.0 cells (the leaderboard is up to 7.6 points higher); we use the leaderboard's trials. In the
# matched set, output-token caps still differ (Anthropic 4,096 under mini-SWE-agent, 32,000 under Claude Code,
# 64,000 under OpenHands), and Claude Code's thinking setting rests on reading its minified bundle. The matched
# s2_mh comes from 10 models (9 df) and is applied to all 17 models under Terminus 2.
#
# Inputs:  data/processed/tbench2_trials.csv, tbench2_cells.csv
# Outputs: data/processed/tbench2_components.csv  (variance components per harness set)
#          data/processed/tbench2_t2_models.csv    (17 models under Terminus 2: accuracy, per-model SEs, rank intervals)
#          data/processed/tbench2_t2_pairs.csv     (136 pairs under Terminus 2: gap and three SEs)
#          data/processed/tbench2_pair_gaps.csv    (gap under each shared harness, per model pair)
#          data/processed/tbench2_pair_anova.csv   (per-pair harness x pair variance)
#          data/processed/tbench2_model_range.csv  (each model's accuracy range across harnesses)
#          data/processed/tbench2_summary.csv      (headline numbers with 95% intervals)
#          data/processed/tbench2_boot.csv         (bootstrap cache; delete to rerun)

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(lme4)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("run_all.sh"))
setwd(PROJECT_ROOT)
set.seed(42)
Z <- qnorm(0.975)
N_BOOT <- 2000
BOOT_PATH <- "data/processed/tbench2_boot.csv"
REF <- "terminus-2"
HARNESS <- c("terminus-2" = "Terminus 2", "mini-swe-agent" = "mini-SWE-agent", "openhands" = "OpenHands",
             "claude-code" = "Claude Code", "codex" = "Codex CLI", "gemini-cli" = "Gemini CLI")
OPEN <- c("terminus-2", "mini-swe-agent", "openhands")
GPT5 <- c("GPT-5", "GPT-5-Codex", "GPT-5 mini", "GPT-5 nano")
GEMINI <- c("Gemini 2.5 Pro", "Gemini 2.5 Flash")
TEE_SETS <- c("matched", "all")
MODEL <- c(
  "claude-sonnet-4-5-20250929@anthropic" = "Claude Sonnet 4.5", "claude-opus-4-1-20250805@anthropic" = "Claude Opus 4.1",
  "claude-haiku-4-5-20251001@anthropic" = "Claude Haiku 4.5", "gpt-5@openai" = "GPT-5", "gpt-5-codex@openai" = "GPT-5-Codex",
  "gpt-5-mini@openai" = "GPT-5 mini", "gpt-5-nano@openai" = "GPT-5 nano", "gemini-2.5-pro@gemini" = "Gemini 2.5 Pro",
  "gemini-2.5-flash@gemini" = "Gemini 2.5 Flash", "grok-4-0709@xai" = "Grok 4", "grok-code-fast-1@xai" = "Grok Code Fast 1",
  "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8@together_ai" = "Qwen3 Coder 480B",
  "moonshotai/Kimi-K2-Instruct-0905@together_ai" = "Kimi K2 Instruct", "openai/gpt-oss-120b@together_ai" = "gpt-oss-120b",
  "openai/gpt-oss-20b@together_ai" = "gpt-oss-20b", "accounts/fireworks/models/glm-4p6@fireworks_ai" = "GLM 4.6",
  "accounts/fireworks/models/minimax-m2@fireworks_ai" = "MiniMax M2")

d <- fread("data/processed/tbench2_trials.csv")[paper_design == TRUE]
d[, `:=`(model = MODEL[model], harness = agent, task = task_name)]
stopifnot(!anyNA(d$model), all(d$harness %in% names(HARNESS)), uniqueN(d$task) == 89,
          uniqueN(d[, .(model, harness)]) == 50, nrow(d) == 22198)
cells <- fread("data/processed/tbench2_cells.csv")[paper_design == TRUE]
cells[, `:=`(model = MODEL[model], harness = agent)]

# Per cell x task: mean reward, trials, and within-task variance of that mean
ct <- d[, .(p = mean(reward), n = .N), by = .(model, harness, task)]
ct[, v := fifelse(n > 1, p * (1 - p) / (n - 1), NA_real_)]
stopifnot(nrow(ct) == 50 * 89, all(ct$n >= 3))
acc <- ct[, .(acc = mean(p)), by = .(model, harness)]
chk <- merge(acc, cells[, .(model, harness, lb_acc)], by = c("model", "harness"))
stopifnot(max(abs(100 * chk$acc - chk$lb_acc)) < 0.01)

ALL <- unique(d[, .(model, harness)])
MATCHED <- rbind(
  CJ(model = GPT5, harness = c("terminus-2", "mini-swe-agent", "codex")),
  CJ(model = GEMINI, harness = c("mini-swe-agent", "gemini-cli")),
  CJ(model = c("Claude Sonnet 4.5", "Claude Haiku 4.5"), harness = c("openhands", "claude-code")),
  CJ(model = c("Claude Opus 4.1", "Grok 4"), harness = c("mini-swe-agent", "openhands")))
stopifnot(nrow(MATCHED) == 24, nrow(merge(MATCHED, ALL, by = c("model", "harness"))) == 24)
SETS <- c(list(matched = MATCHED, all = ALL, matched_without_openhands = MATCHED[harness != "openhands"],
               effort_matched = ALL[!(harness == "openhands" & model %in% c(GPT5, GEMINI))],
               gpt5_only = MATCHED[model %in% GPT5], open = ALL[harness %in% OPEN]),
          setNames(map(names(HARNESS), ~ ALL[harness != .x]), paste0("all_without_", names(HARNESS))))
stopifnot(nrow(SETS$effort_matched) == 45)

# ---- 1. Variance components ----
fit_set <- function(cs) {
  x <- merge(d, cs, by = c("model", "harness"))
  x[, `:=`(model = factor(model), harness = factor(harness), task = factor(task))]
  lmer(reward ~ model + harness + (1 | task) + (1 | task:model) + (1 | task:harness) + (1 | model:harness) +
         (1 | task:model:harness), data = x, REML = TRUE, control = lmerControl(calc.derivs = FALSE))
}
mh_interval <- function(cs, vc) {
  cl <- merge(ct, cs, by = c("model", "harness"))[, .(a = mean(p), ninv = mean(1 / n)), by = .(model, harness)]
  l <- lm(a ~ model + harness, data = cl)
  df <- l$df.residual; rms <- sum(resid(l)^2) / df
  noise <- mean((vc$vcov[vc$grp == "task:model:harness"] + vc$vcov[vc$grp == "Residual"] * cl$ninv) / 89)
  list(df = df, mom = rms - noise, ci = pmax(0, df * rms / qchisq(c(0.975, 0.025), df) - noise))
}
components <- imap_dfr(SETS, function(cs, nm) {
  vc <- as.data.frame(VarCorr(fit_set(cs)))
  mi <- mh_interval(cs, vc)
  tibble(set = nm, harnesses = paste(intersect(names(HARNESS), cs$harness), collapse = ","), n_cells = nrow(cs),
         n_models = uniqueN(cs$model), n_trials = nrow(merge(d, cs, by = c("model", "harness"))),
         component = vc$grp, var = vc$vcov, sd_pts = 100 * vc$sdcor) %>%
    mutate(df_mh = if_else(component == "model:harness", mi$df, NA_integer_),
           sd_mom_pts = if_else(component == "model:harness", 100 * sqrt(max(mi$mom, 0)), NA_real_),
           sd_lo_pts = if_else(component == "model:harness", 100 * sqrt(mi$ci[1]), NA_real_),
           sd_hi_pts = if_else(component == "model:harness", 100 * sqrt(mi$ci[2]), NA_real_))
})
s2_mh <- components %>% filter(component == "model:harness") %>% select(set, var) %>% deframe()

# ---- 2. Terminus 2: all 17 models ----
P <- dcast(ct[harness == REF], model ~ task, value.var = "p")
V <- dcast(ct[harness == REF], model ~ task, value.var = "v")
stopifnot(identical(P$model, V$model), identical(names(P), names(V)))
Pm <- as.matrix(P[, -1]); rownames(Pm) <- P$model
Vm <- as.matrix(V[, -1]); rownames(Vm) <- V$model
N <- ncol(Pm)
stopifnot(nrow(Pm) == 17, N == 89, !anyNA(Pm), !anyNA(Vm))

pair_mats <- function(M, Vv) {
  C <- cov(t(M)); s <- diag(C); se_trial <- sqrt(colSums(t(Vv))) / ncol(M)
  list(acc = rowMeans(M), se_naive = sqrt(s / ncol(M)), G = outer(rowMeans(M), rowMeans(M), "-"),
       S = sqrt(pmax(outer(s, s, "+") - 2 * C, 0) / ncol(M)),
       St = sqrt(outer(se_trial^2, se_trial^2, "+")), se_trial = se_trial)
}
rank_intervals <- function(G, S, extra_var = 0) {
  sig <- G / sqrt(S^2 + extra_var) > Z
  sig[is.na(sig)] <- FALSE; diag(sig) <- FALSE
  k <- nrow(G)
  tibble(best = 1 + colSums(sig), worst = k - rowSums(sig), ties = k - 1 - colSums(sig) - rowSums(sig))
}

pm <- pair_mats(Pm, Vm)
ord <- order(-pm$acc, rownames(Pm))
Pm <- Pm[ord, ]; Vm <- Vm[ord, ]; pm <- pair_mats(Pm, Vm)
K <- nrow(Pm)
adj_idx <- cbind(1:(K - 1), 2:K)

stats <- function(ii) {
  p <- pair_mats(Pm[, ii], Vm[, ii])
  ri <- c(list(trial = rank_intervals(p$G, p$St), paired = rank_intervals(p$G, p$S)),
          setNames(map(TEE_SETS, ~ rank_intervals(p$G, p$S, 2 * s2_mh[[.x]])), paste0("tee_", TEE_SETS)))
  g <- p$G[adj_idx]
  sig <- c(trial = sum(g / p$St[adj_idx] > Z), paired = sum(g / p$S[adj_idx] > Z),
           setNames(map_dbl(TEE_SETS, ~ sum(g / sqrt(p$S[adj_idx]^2 + 2 * s2_mh[[.x]]) > Z)), paste0("tee_", TEE_SETS)))
  c(setNames(p$acc, paste0("acc|", rownames(Pm))), setNames(g, paste0("gap|", 1:(K - 1))),
    setNames(sig, paste0("n_sig_adj|", names(sig))),
    setNames(map_dbl(ri, ~ .x$ties[1]), paste0("top_ties|", names(ri))),
    setNames(map_dbl(ri, ~ mean(.x$ties)), paste0("mean_ties|", names(ri))),
    setNames(map_dbl(ri, ~ mean(.x$worst - .x$best + 1)), paste0("mean_width|", names(ri))))
}
full <- stats(seq_len(N))
if (file.exists(BOOT_PATH)) {
  boot <- fread(BOOT_PATH)
  stopifnot(nrow(boot) == N_BOOT, identical(names(boot), names(full)))
} else {
  boot <- as.data.table(do.call(rbind, map(seq_len(N_BOOT), ~ stats(sample.int(N, N, replace = TRUE)))))
  fwrite(boot, BOOT_PATH)
}
bci <- function(k) quantile(boot[[k]], c(0.025, 0.975), names = FALSE)

ri_all <- c(list(trial = rank_intervals(pm$G, pm$St), paired = rank_intervals(pm$G, pm$S)),
            setNames(map(TEE_SETS, ~ rank_intervals(pm$G, pm$S, 2 * s2_mh[[.x]])), paste0("tee_", TEE_SETS)))
t2_models <- tibble(model = rownames(Pm), rank = seq_len(K), acc_pts = 100 * pm$acc,
                    se_trial_pts = 100 * pm$se_trial, se_naive_pts = 100 * pm$se_naive) %>%
  mutate(acc_lo = map_dbl(model, ~ 100 * bci(paste0("acc|", .x))[1]),
         acc_hi = map_dbl(model, ~ 100 * bci(paste0("acc|", .x))[2])) %>%
  bind_cols(imap_dfc(ri_all, ~ setNames(.x[, c("best", "worst")], paste0(c("best_", "worst_"), .y))))
stopifnot(isTRUE(all.equal(unname(t2_models$acc_pts),
                           100 * acc[harness == REF][match(t2_models$model, model), acc])))

pairs <- expand_grid(i = seq_len(K), j = seq_len(K)) %>% filter(i < j) %>%
  mutate(model_a = rownames(Pm)[i], model_b = rownames(Pm)[j], adjacent = j == i + 1,
         gap_pts = 100 * pm$G[cbind(i, j)], se_trial_pts = 100 * pm$St[cbind(i, j)],
         se_paired_pts = 100 * pm$S[cbind(i, j)],
         se_tee_matched_pts = 100 * sqrt(pm$S[cbind(i, j)]^2 + 2 * s2_mh[["matched"]]),
         se_tee_all_pts = 100 * sqrt(pm$S[cbind(i, j)]^2 + 2 * s2_mh[["all"]])) %>%
  mutate(gap_lo = NA_real_, gap_hi = NA_real_)
adj_rows <- which(pairs$adjacent)
pairs$gap_lo[adj_rows] <- map_dbl(seq_along(adj_rows), ~ 100 * bci(paste0("gap|", .x))[1])
pairs$gap_hi[adj_rows] <- map_dbl(seq_along(adj_rows), ~ 100 * bci(paste0("gap|", .x))[2])
pairs <- select(pairs, -i, -j)

# ---- 3. Model pairs across shared harnesses ----
Pall <- dcast(ct, model + harness ~ task, value.var = "p")
cellP <- function(m, h) { r <- Pall[model == m & harness == h]; as.numeric(as.matrix(r[, -(1:2)])) }
two_way <- function(D) {
  Vb <- nrow(D); Nb <- ncol(D); g <- mean(D); r <- rowMeans(D); cm <- colMeans(D)
  ss_t <- Nb * sum((r - g)^2); ss_i <- Vb * sum((cm - g)^2)
  ss_e <- sum(D^2) - Vb * Nb * g^2 - ss_t - ss_i
  ms_t <- ss_t / (Vb - 1); ms_e <- ss_e / ((Vb - 1) * (Nb - 1))
  list(delta = g, gap_h = r, s2_v = max(0, (ms_t - ms_e) / Nb), ms_t = ms_t, ms_e = ms_e)
}
hs_of <- split(acc$harness, acc$model)
mods <- sort(unique(acc$model))
pair_list <- combn(mods, 2, simplify = FALSE) %>%
  keep(~ length(intersect(hs_of[[.x[1]]], hs_of[[.x[2]]])) >= 3)
pair_gaps <- map_dfr(pair_list, function(pr) {
  hs <- intersect(hs_of[[pr[1]]], hs_of[[pr[2]]])
  tibble(model_a = pr[1], model_b = pr[2], harness = hs,
         gap_pts = map_dbl(hs, ~ 100 * mean(cellP(pr[1], .x) - cellP(pr[2], .x))))
})
pair_anova <- map_dfr(pair_list, function(pr) {
  hs <- intersect(hs_of[[pr[1]]], hs_of[[pr[2]]])
  D <- do.call(rbind, map(hs, ~ cellP(pr[1], .x) - cellP(pr[2], .x)))
  cp <- two_way(D)
  tibble(model_a = pr[1], model_b = pr[2], n_harness = length(hs), open_only = all(hs %in% OPEN),
         harnesses = paste(hs, collapse = ","), delta_pts = 100 * cp$delta,
         range_pts = 100 * diff(range(cp$gap_h)), sd_v_pts = 100 * sqrt(cp$s2_v))
})
# Same decomposition restricted to the three open harnesses, for pairs that ran under all three
pair_anova_open <- map_dfr(pair_list, function(pr) {
  if (!all(OPEN %in% intersect(hs_of[[pr[1]]], hs_of[[pr[2]]]))) return(NULL)
  cp <- two_way(do.call(rbind, map(OPEN, ~ cellP(pr[1], .x) - cellP(pr[2], .x))))
  tibble(model_a = pr[1], model_b = pr[2], s2_v = cp$s2_v)
})

model_range <- acc %>% as_tibble() %>% group_by(model) %>%
  summarise(n_harness = n(), min_pts = 100 * min(acc), max_pts = 100 * max(acc),
            range_pts = max_pts - min_pts, best = HARNESS[harness[which.max(acc)]],
            worst = HARNESS[harness[which.min(acc)]], .groups = "drop") %>%
  arrange(desc(n_harness), desc(range_pts))

# ---- Summary ----
row <- function(stat, k, label) tibble(stat = stat, value = full[[k]], lo = bci(k)[1], hi = bci(k)[2], note = label)
comp_row <- function(set) components %>% filter(set == !!set, component == "model:harness")
summary_tbl <- bind_rows(
  tibble(stat = "n_trials", value = nrow(d), lo = NA, hi = NA, note = "paper design, 50 cells"),
  map_dfr(TEE_SETS, function(s) {
    cr <- comp_row(s)
    tibble(stat = paste0("sd_mh_pts_", s), value = cr$sd_pts, lo = cr$sd_lo_pts, hi = cr$sd_hi_pts,
           note = paste0("model x harness SD (REML), chi-square 95% CI on ", cr$df_mh, " df; ", cr$n_cells, " cells"))
  }),
  map_dfr(TEE_SETS, function(s) {
    cr <- comp_row(s)
    tibble(stat = paste0("sd_pair_gap_pts_", s), value = sqrt(2) * cr$sd_pts, lo = sqrt(2) * cr$sd_lo_pts,
           hi = sqrt(2) * cr$sd_hi_pts, note = "SD of a two-model gap across harnesses, sqrt(2) x sd_mh")
  }),
  tibble(stat = "sd_v_pairs_mean_pts_open", value = 100 * sqrt(mean(pair_anova_open$s2_v)), lo = NA, hi = NA,
         note = paste0("moment cross-check: sqrt(mean s2_v) over ", nrow(pair_anova_open),
                       " pairs run under all three open harnesses")),
  tibble(stat = "sd_resid_trial_pts", value = components$sd_pts[components$set == "all" &
                                                                  components$component == "Residual"],
         lo = NA, hi = NA, note = "trial-to-trial SD, all six harnesses"),
  imap_dfr(c(trial = "leaderboard (trials only)", paired = "paired over tasks",
             tee_matched = "plus harness, settings matched (primary)", tee_all = "plus harness, all 50 cells"),
           function(lbl, k) bind_rows(
    row(paste0("n_sig_adjacent_", k), paste0("n_sig_adj|", k), paste("of", K - 1, "adjacent pairs,", lbl)),
    row(paste0("top_ties_", k), paste0("top_ties|", k), paste("models tied with #1,", lbl)),
    row(paste0("mean_ties_", k), paste0("mean_ties|", k), lbl),
    row(paste0("mean_rank_width_", k), paste0("mean_width|", k), lbl))))

fwrite(components, "data/processed/tbench2_components.csv")
fwrite(t2_models, "data/processed/tbench2_t2_models.csv")
fwrite(pairs, "data/processed/tbench2_t2_pairs.csv")
fwrite(pair_gaps, "data/processed/tbench2_pair_gaps.csv")
fwrite(pair_anova, "data/processed/tbench2_pair_anova.csv")
fwrite(model_range, "data/processed/tbench2_model_range.csv")
fwrite(summary_tbl, "data/processed/tbench2_summary.csv")

options(width = 200)
print(components %>% filter(set %in% TEE_SETS) %>% mutate(across(where(is.numeric), ~ round(.x, 4))),
      n = Inf)
print(components %>% filter(component == "model:harness") %>%
        select(set, n_cells, n_models, df_mh, sd_pts, sd_mom_pts, sd_lo_pts, sd_hi_pts) %>% mutate(across(where(is.numeric), ~ round(.x, 2))))
print(t2_models %>% mutate(across(where(is.numeric), ~ round(.x, 1))), n = Inf, width = Inf)
print(pairs %>% filter(adjacent) %>% mutate(across(where(is.numeric), ~ round(.x, 2))), n = Inf, width = Inf)
print(model_range %>% mutate(across(where(is.numeric), ~ round(.x, 1))), n = Inf)
print(summary_tbl %>% mutate(across(where(is.numeric), ~ round(.x, 2))), n = Inf, width = Inf)
cat("Saved data/processed/tbench2_{components,t2_models,t2_pairs,pair_gaps,pair_anova,model_range,summary}.csv\n")
