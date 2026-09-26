#!/usr/bin/env bash
# run_all.sh — master orchestration script for the TEE replication pipeline.
#
# Runs the analysis pipeline end-to-end, from raw JSONL (if present) or cleaned
# CSVs through variance decomposition, simulations, validation, and figures.
# Finishes with 15_verify_manuscript_numbers.R to assert all cited numbers
# match the computed outputs.
#
# Usage:
#   ./run_all.sh                # full pipeline (~6-8 hours)
#   ./run_all.sh --fast         # skip long simulations (>1 hour each)
#   ./run_all.sh --smoke        # smoke tests only (~10 min total)
#   ./run_all.sh --verify-only  # skip everything, just run 15_verify
#
# Requirements:
#   - R with tidyverse, lme4, patchwork, jsonlite, BradleyTerry2, glmmTMB,
#     scales, arrow (optional)
#   - Cleaned CSVs in data/processed/ OR raw JSONL in data/raw/
#   - ALIGNED_TO_WHOM_PATH env var (for 11_ground_truth_validation.R)
#
# All R scripts expect cwd = project root.

set -euo pipefail

MODE="full"
case "${1:-}" in
  --fast)        MODE="fast" ;;
  --smoke)       MODE="smoke" ;;
  --verify-only) MODE="verify" ;;
  --help|-h)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "")            MODE="full" ;;
  *)             echo "Unknown flag: $1"; exit 2 ;;
esac

SMOKE_FLAG=""
if [[ "$MODE" == "smoke" ]]; then
  SMOKE_FLAG="--nsim 5"
fi

run() {
  local script="$1"
  echo ""
  echo "=== [$(date '+%H:%M:%S')] $script $SMOKE_FLAG ==="
  Rscript "$script" $SMOKE_FLAG
}

if [[ "$MODE" == "verify" ]]; then
  run analysis/15_verify_manuscript_numbers.R
  exit $?
fi

# -------- Stage 1: Clean raw data (JSONL -> CSV) --------
# Skip if cleaned CSVs already exist.
if [[ ! -f data/processed/safety_clean.csv || "$MODE" == "smoke" ]]; then
  run analysis/24_clean_likert_cot.R          # likert ideology (matched CoT run used in the paper)
  run analysis/22_clean_pairwise_cot.R        # pairwise ideology (matched CoT run)
  cp data/processed/likert_cot_clean.csv data/processed/likert_clean.csv       # downstream scripts read *_clean.csv
  cp data/processed/pairwise_cot_clean.csv data/processed/pairwise_clean.csv
  run analysis/01_clean_safety.R              # safety (AILuminate)
  run analysis/01_clean_safety_pairwise.R     # safety pairwise
  experiments/.venv/bin/python experiments/check_mmlu_variant_structure.py  # MMLU variant admissibility
  Rscript analysis/01_clean_mmlu.R full       # MMLU (strict parser; excludes inadmissible variants); default stage is pilot
  run analysis/01b_reparse_mmlu_recollections.R  # MMLU breadth re-collections, strict parser
fi

# -------- Stage 2: Variance decomposition --------
run analysis/02_variance_decomposition.R              # ideology (Likert + Pairwise)
run analysis/02_variance_decomposition_safety.R       # safety (LPM)
run analysis/02_variance_decomposition_safety_pairwise.R
run analysis/02c_lojo_judge_safety.R                 # leave-one-judge-out (SI si:lojo)
Rscript analysis/02e_prompt_breadth_sensitivity.R --boot 200 --cores 8   # prompt-variant breadth + bootstrap CIs (SI si:prompt_admissibility)
run analysis/02f_prompt_breadth_glmm_safety.R        # safety breadth on the logit scale
run analysis/02_variance_decomposition_mmlu.R         # MMLU
run analysis/02b_glmm_robustness.R                    # GLMM vs LPM
run analysis/02d_mmlu_variant_accuracy.R              # MMLU per-variant accuracy
run analysis/02g_mmlu_v3_sensitivity.R                # MMLU V=5 sensitivity (SI tab:mmlu_v5)

# -------- Stage 3: Simulations (skip in fast mode) --------
if [[ "$MODE" != "fast" ]]; then
  run analysis/04_simulation.R                # REML asymptotics
  run analysis/04b_sim_additivity.R           # additivity robustness
  run analysis/04c_sim_heteroscedastic.R      # heteroscedastic recovery
  run analysis/04d_sim_small_k.R              # small-K prompt sensitivity
  run analysis/04e_sim_portability.R          # cross-model D-study transfer
  run analysis/04f_sim_dstudy_validation.R    # D-study projection validation
  run analysis/04g2_sim_underestimation_balanced.R    # Fig 2: variance underestimation
  run analysis/04h_sim_latent_ambiguity.R     # latent ambiguity robustness
  run analysis/06_sim_scoring_recovery.R      # scoring method recovery (D.6)
fi

# -------- Stage 4: Cross-validation and real-data analyses --------
run analysis/07_pilot_validation.R
run analysis/10_naive_vs_tee_se.R
if [[ -n "${ALIGNED_TO_WHOM_PATH:-}" && -d "${ALIGNED_TO_WHOM_PATH}" ]]; then
  run analysis/11_ground_truth_validation.R
else
  echo "  [SKIP] 11_ground_truth_validation.R: set ALIGNED_TO_WHOM_PATH"
fi
run analysis/12_budget_allocation_mmlu.R
run analysis/12_human_tee_decomposition.R
run analysis/13_propaganda_pilot_analysis.R
run analysis/13_sim_gaming.R
run analysis/13b_sim_gaming_safety.R
run analysis/13c_sim_gaming_arena.R                  # Arena gaming surface: paper Figure 4 and tab:gaming
run analysis/16_cost_efficiency_frontier.R
run analysis/16_pairwise_vs_binary_recovery.R
run analysis/17_sensitivity_tier_matched.R

# -------- Stage 5: Figures --------
run analysis/03_figures.R
run analysis/03_figures_safety.R
run analysis/03_figures_mmlu.R
run analysis/03b_fig_intro_safety.R
run analysis/03c_fig_mmlu_decomposition.R            # main-text MMLU G-study panel
run analysis/12b_fig_budget_allocation_stacked.R     # main-text MMLU D-study panel (after 12)
run analysis/05_manuscript_figures.R
run analysis/05b_scoring_method_figures.R
run analysis/14_propaganda_plots.R
bash analysis/embed_figure_fonts.sh                   # embed fonts in every figure the paper includes

# -------- Stage 6: Verify manuscript numbers --------
run analysis/15_verify_manuscript_numbers.R

echo ""
echo "=== [$(date '+%H:%M:%S')] Pipeline complete ==="
