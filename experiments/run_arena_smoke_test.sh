#!/usr/bin/env bash
# Smoke-test the Arena scoring pipeline end-to-end.
#
# Stage 1: scoring smoke (5 battles, ~10 min total for both modes, ~$0.01)
# Stage 2: pilot (20 battles, ~5 min per mode, ~$0.60 total)
# Stage 3: cleaning step on smoke output to verify parsers match runner schema
#
# Usage: bash experiments/run_arena_smoke_test.sh [smoke|pilot]

set -euo pipefail

STAGE="${1:-smoke}"
if [[ "$STAGE" != "smoke" && "$STAGE" != "pilot" ]]; then
  echo "Usage: $0 [smoke|pilot]" >&2
  exit 2
fi

cd "$(dirname "$0")/.." # project root
source experiments/.venv/bin/activate

echo "=== Arena ${STAGE} smoke test ==="
echo "-- Running Likert mode"
python experiments/run_arena_scoring.py --mode likert --stage "$STAGE"

echo
echo "-- Running Pairwise forced mode"
python experiments/run_arena_scoring.py --mode pairwise_forced --stage "$STAGE"

echo
echo "-- Verifying Likert JSONL schema"
if [[ "$STAGE" == "full" ]]; then
  LIK_PATH="data/raw/arena_likert.jsonl"
  PW_PATH="data/raw/arena_pairwise_forced.jsonl"
else
  LIK_PATH="data/raw/arena_likert_${STAGE}.jsonl"
  PW_PATH="data/raw/arena_pairwise_forced_${STAGE}.jsonl"
fi

python - <<PY
import json
import sys

for label, path in [("likert", "$LIK_PATH"), ("pairwise", "$PW_PATH")]:
    rows = 0
    sample = None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if sample is None:
                sample = r
            rows += 1
    required_meta = {"battle_id", "category_llm", "winner", "variant_id"}
    if label == "likert":
        required_meta.add("response_side")
    else:
        required_meta.add("order")
    meta_keys = set(sample["meta"].keys())
    missing = required_meta - meta_keys
    if missing:
        print(f"  FAIL {label}: missing meta keys {missing}")
        sys.exit(1)
    print(f"  OK {label}: {rows} rows, meta keys match")
PY

echo
echo "-- Done. Next steps: Rscript analysis/29_clean_arena_scoring.R on these JSONL files"
