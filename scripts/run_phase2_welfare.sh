#!/usr/bin/env bash
# Phase 2 Runner: Animal Welfare Alignment (ANIMA & MANTA)
# Usage: ./scripts/run_phase2_welfare.sh [arm: a0|a2|a5|both|all] [manta_limit: default 100] [gpu_id: default 0] [num_samples: default 1]

set -euo pipefail

ARM="${1:-both}"
MANTA_LIMIT="${2:-100}"
GPU_ID="${3:-0}"
NUM_SAMPLES="${4:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="/mnt/HDD_2TB/niraj/venvs/awmsm/bin/python"
LOG_DIR="/mnt/HDD_2TB/niraj/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/eval_phase2_${ARM}_n${NUM_SAMPLES}.log"

trap '' HUP

echo "======================================================================" >> "${LOG_FILE}"
echo "Starting Phase 2 Benchmark: Animal Welfare Alignment" >> "${LOG_FILE}"
echo "Arm: ${ARM}" >> "${LOG_FILE}"
echo "Benchmarks: ANIMA (115 scenarios), MANTA (limit: ${MANTA_LIMIT})" >> "${LOG_FILE}"
echo "Samples per item: ${NUM_SAMPLES} (T=0.7 averaged)" >> "${LOG_FILE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}" >> "${LOG_FILE}"
echo "Log: ${LOG_FILE}" >> "${LOG_FILE}"
echo "======================================================================" >> "${LOG_FILE}"

${PYTHON_BIN} -u -m awmsm.eval.bench_welfare \
    --arm "${ARM}" \
    --manta-limit "${MANTA_LIMIT}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size 4 >> "${LOG_FILE}" 2>&1

echo "======================================================================" >> "${LOG_FILE}"
echo "Phase 2 Complete. Results stored in results/phase2/" >> "${LOG_FILE}"
echo "======================================================================" >> "${LOG_FILE}"
