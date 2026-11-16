#!/usr/bin/env bash
# Run the full post-scoring analysis pipeline for the Arena demonstration.
# Prereqs:
#   - data/raw/arena_likert.jsonl        (~45k rows)
#   - data/raw/arena_pairwise_forced.jsonl (~45k rows)
# Outputs: cleaned CSVs, agreement summaries, figures, and passing test suite.

set -euo pipefail
cd "$(dirname "$0")/.."  # project root

echo "=== 29: Clean Arena scoring JSONLs ==="
Rscript analysis/29_clean_arena_scoring.R

echo
echo "=== 30: Pipeline comparison + bootstrap CIs ==="
Rscript analysis/30_arena_pipeline_comparison.R

echo
echo "=== 31: D-study prediction (dev) vs observed (test) ==="
Rscript analysis/31_arena_dstudy_predicts_improvement.R

echo
echo "=== 32: Figures ==="
Rscript analysis/32_arena_figures.R

echo
echo "=== Tests ==="
Rscript tests/run_tests.R

echo
echo "=== DONE ==="
echo "Outputs:"
ls -la data/processed/arena_*.csv 2>&1 | tail -15
echo
ls -la figures/fig_arena_*.{pdf,png} 2>&1 | tail -10
