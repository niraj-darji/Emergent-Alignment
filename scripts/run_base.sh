#!/usr/bin/env bash
# Benchmark the three base models on every benchmark, then write
# results/<run-name>/summary.csv.
#
#   bash scripts/run_base.sh smoke 5    # quick check: 5 samples per benchmark
#   bash scripts/run_base.sh base-v1    # the real run
#
# Needs the judge from scripts/start_judge.sh at localhost:8001:
#   one machine, two GPUs:  CUDA_VISIBLE_DEVICES=1 bash scripts/start_judge.sh
#                           CUDA_VISIBLE_DEVICES=0 bash scripts/run_base.sh base-v1
#   two machines:           start_judge.sh on B, then on A:
#                           ssh -N -L 8001:localhost:8001 <user>@<machine-B>
# If the judge is still starting up, this script waits for it (up to 20 minutes).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source scripts/env.sh

RUN="${1:-base-v1}"
LIMIT="${2:-}"
MODELS=(qwen3-8b llama3.1-8b gemma3-12b-nf4)   # 8B models in bf16, Gemma in 4-bit NF4

WAIT_SECONDS=1200
waited=0
until curl -sf --max-time 5 http://localhost:8001/v1/models > /dev/null; do
  if [ "$waited" -ge "$WAIT_SECONDS" ]; then
    echo "The judge is still not reachable at localhost:8001 after $((WAIT_SECONDS / 60)) minutes."
    echo "  Is scripts/start_judge.sh running, and did it print 'Application startup complete'?"
    echo "  One machine, two GPUs:  CUDA_VISIBLE_DEVICES=1 bash scripts/start_judge.sh  (in another terminal)"
    echo "  Two machines:           start it on B, then here:  ssh -N -L 8001:localhost:8001 <user>@<machine-B>"
    exit 1
  fi
  if [ "$waited" -eq 0 ]; then
    echo "Waiting for the judge at localhost:8001 (it takes a few minutes to start)..."
  fi
  sleep 10
  waited=$((waited + 10))
done
echo "Judge is up."


# --resume: if this run was interrupted, rerunning the same command only redoes
# the benchmarks that didn't finish.
python -m benchmarks.run --models "${MODELS[@]}" --benchmarks all --run-name "$RUN" --resume \
  ${LIMIT:+--limit "$LIMIT"}
python -m benchmarks.summarize --run-name "$RUN"
