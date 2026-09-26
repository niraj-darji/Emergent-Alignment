#!/usr/bin/env bash
# Machine B (RTX 5090): serve the judge, gpt-oss-20b, for ANIMA, MANTA and sycophancy.
# Keep this terminal open while machine A runs benchmarks. Ctrl+C stops the server.
# Settings must match `judge-remote` in configs/models.yaml
# (material/quantization-strategy.md §3 and §12).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source scripts/env.sh

exec vllm serve openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768
