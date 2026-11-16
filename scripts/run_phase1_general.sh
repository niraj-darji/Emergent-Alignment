#!/usr/bin/env bash
# Phase 1 Runner: General Capabilities & Safety (MMLU, HellaSwag, BBQ, HHH)
# Usage: ./scripts/run_phase1_general.sh [arm: a0|a2|a5|both|all] [gpu_id: default 0]

set -euo pipefail

ARM="${1:-both}"
GPU_ID="${2:-0}"
BATCH_SIZE="${3:-auto}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="/mnt/HDD_2TB/niraj/venvs/awmsm/bin/python"
LOG_DIR="/mnt/HDD_2TB/niraj/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/eval_phase1_${ARM}.log"

trap '' HUP

echo "======================================================================" >> "${LOG_FILE}"
echo "Starting Phase 1 Benchmark: General Capabilities & Safety" >> "${LOG_FILE}"
echo "Arm: ${ARM}" >> "${LOG_FILE}"
echo "Tasks: MMLU, HellaSwag, BBQ, HHH" >> "${LOG_FILE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}" >> "${LOG_FILE}"
echo "Log: ${LOG_FILE}" >> "${LOG_FILE}"
echo "======================================================================" >> "${LOG_FILE}"

${PYTHON_BIN} -u -m awmsm.eval.bench_general \
    --arm "${ARM}" \
    --batch-size "${BATCH_SIZE}" >> "${LOG_FILE}" 2>&1

echo "======================================================================" >> "${LOG_FILE}"
echo "Phase 1 Complete. Results stored in results/phase1/" >> "${LOG_FILE}"
echo "======================================================================" >> "${LOG_FILE}"

