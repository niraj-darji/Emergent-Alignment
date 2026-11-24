#!/usr/bin/env bash
# Anthropic HH Harmless Benchmark Runner
# Usage: ./scripts/run_anthropic_hh_bench.sh [arm: a0|a2|a2_v2|a3_v2|generalization|all] [gpu_id: default 0]

set -euo pipefail

ARM="${1:-generalization}"
GPU_ID="${2:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"
export HF_HUB_OFFLINE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="/mnt/HDD_2TB/niraj/.triton"
export TORCH_EXTENSIONS_DIR="/mnt/HDD_2TB/niraj/.torch_extensions"

PYTHON_BIN="/mnt/HDD_2TB/niraj/venvs/awmsm/bin/python"
LOG_DIR="/mnt/HDD_2TB/niraj/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/eval_anthropic_hh_${ARM}.log"

trap '' HUP

echo "======================================================================" >> "${LOG_FILE}"
echo "Starting Anthropic HH Harmless Benchmark" >> "${LOG_FILE}"
echo "Arm: ${ARM}" >> "${LOG_FILE}"
echo "Data: data/benchmarks/anthropic_hh_harmless/harmless_test.parquet" >> "${LOG_FILE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}" >> "${LOG_FILE}"
echo "Log: ${LOG_FILE}" >> "${LOG_FILE}"
echo "Time: $(date)" >> "${LOG_FILE}"
echo "======================================================================" >> "${LOG_FILE}"

${PYTHON_BIN} -u -m awmsm.eval.bench_anthropic_hh \
    --arm "${ARM}" \
    --device "cuda:${GPU_ID}" >> "${LOG_FILE}" 2>&1

echo "======================================================================" >> "${LOG_FILE}"
echo "Anthropic HH Harmless Complete. Results stored in results/anthropic_hh/" >> "${LOG_FILE}"
echo "======================================================================" >> "${LOG_FILE}"
