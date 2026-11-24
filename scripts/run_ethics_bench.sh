#!/usr/bin/env bash
# Hendrycks ETHICS Benchmark Runner
# Usage: ./scripts/run_ethics_bench.sh [arm: a0|a2|a2_v2|a3_v2|generalization|all] [gpu_id: default 0] [batch_size: default auto]

set -euo pipefail

ARM="${1:-generalization}"
GPU_ID="${2:-0}"
BATCH_SIZE="${3:-auto}"

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

LOG_FILE="${LOG_DIR}/eval_ethics_${ARM}.log"

trap '' HUP

echo "======================================================================" >> "${LOG_FILE}"
echo "Starting Hendrycks ETHICS Benchmark" >> "${LOG_FILE}"
echo "Arm: ${ARM}" >> "${LOG_FILE}"
echo "Subtasks: ethics_cm, ethics_deontology, ethics_justice, ethics_utilitarianism, ethics_virtue" >> "${LOG_FILE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}" >> "${LOG_FILE}"
echo "Log: ${LOG_FILE}" >> "${LOG_FILE}"
echo "Time: $(date)" >> "${LOG_FILE}"
echo "======================================================================" >> "${LOG_FILE}"

${PYTHON_BIN} -u -m awmsm.eval.bench_ethics \
    --arm "${ARM}" \
    --batch-size "${BATCH_SIZE}" >> "${LOG_FILE}" 2>&1

echo "======================================================================" >> "${LOG_FILE}"
echo "Hendrycks ETHICS Complete. Results stored in results/ethics/" >> "${LOG_FILE}"
echo "======================================================================" >> "${LOG_FILE}"
