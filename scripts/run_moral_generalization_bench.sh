#!/usr/bin/env bash
# Master Benchmark Runner: Moral Generalization Suite
# Evaluates Hendrycks ETHICS and Anthropic HH Harmless across target model arms.
# Usage: ./scripts/run_moral_generalization_bench.sh [arm: a0|a2|x2_5m|a2_v2|a3_v2|generalization|all] [gpu_id: default 0]

set -euo pipefail

ARM="${1:-all}"
GPU_ID="${2:-0}"
BATCH_SIZE="${3:-4}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"
export HF_HUB_OFFLINE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="/mnt/HDD_2TB/niraj/.triton"
export TORCH_EXTENSIONS_DIR="/mnt/HDD_2TB/niraj/.torch_extensions"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWMSM_ROOT="$(dirname "${SCRIPT_DIR}")"

cd "${AWMSM_ROOT}"

echo "======================================================================"
echo "MORAL GENERALIZATION BENCHMARK SUITE"
echo "Target Arm: ${ARM}"
echo "GPU Device: ${GPU_ID}"
echo "Datasets: Hendrycks ETHICS (5 subtasks) & Anthropic HH Harmless (2.3k test)"
echo "Time: $(date)"
echo "======================================================================"

echo ""
echo ">>> STEP 1/2: Running Hendrycks ETHICS Benchmark..."
./scripts/run_ethics_bench.sh "${ARM}" "${GPU_ID}" "${BATCH_SIZE}"

echo ""
echo ">>> STEP 2/2: Running Anthropic HH Harmless Benchmark..."
./scripts/run_anthropic_hh_bench.sh "${ARM}" "${GPU_ID}"

echo ""
echo "======================================================================"
echo "ALL MORAL GENERALIZATION EVALUATIONS COMPLETE!"
echo "ETHICS results:       results/ethics/"
echo "Anthropic HH results: results/anthropic_hh/"
echo "======================================================================"
