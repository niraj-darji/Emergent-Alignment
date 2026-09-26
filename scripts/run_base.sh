#!/usr/bin/env bash
# Machine A (RTX 4090): benchmark the three base models on every benchmark, then
# write results/<run-name>/summary.csv.
#
#   bash scripts/run_base.sh smoke 5    # quick check: 5 samples per benchmark
#   bash scripts/run_base.sh base-v1    # the real run
#
# Needs the judge from scripts/start_judge.sh (machine B) at localhost:8001,
# reached through an SSH tunnel:  ssh -N -L 8001:localhost:8001 <user>@<machine-B>
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUN="${1:-base-v1}"
LIMIT="${2:-}"
MODELS=(qwen3-8b llama3.1-8b gemma3-12b-nf4)   # 8B models in bf16, Gemma in 4-bit NF4

if ! curl -sf http://localhost:8001/v1/models > /dev/null; then
  echo "The judge is not reachable at localhost:8001."
  echo "  1. On machine B:  bash scripts/start_judge.sh"
  echo "  2. On this machine, in another terminal:  ssh -N -L 8001:localhost:8001 <user>@<machine-B>"
  exit 1
fi

python -m benchmarks.run --models "${MODELS[@]}" --benchmarks all --run-name "$RUN" \
  ${LIMIT:+--limit "$LIMIT"}
python -m benchmarks.summarize --run-name "$RUN"
