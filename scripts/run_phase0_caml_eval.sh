#!/usr/bin/env bash
# Phase 0 Runner: ANIMA Diagnostic Calibration on Official CaML Released Checkpoint
# Evaluates CompassioninMachineLearning/pretrainingBasellama3kv3 to calibrate the evaluation harness.
# Usage: ./scripts/run_phase0_caml_eval.sh [num_samples: default 1] [gpu_id: default 0]

set -euo pipefail

NUM_SAMPLES="${1:-1}"
GPU_ID="${2:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"

PYTHON_BIN="/mnt/HDD_2TB/niraj/venvs/awmsm/bin/python"
LOG_DIR="/mnt/HDD_2TB/niraj/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/eval_phase0_caml_calibration_n${NUM_SAMPLES}.log"
CKPT_PATH="runs/checkpoints/caml_official_pretrainingBasellama3kv3"

echo "======================================================================"
echo "Starting Phase 0 Diagnostic: CaML Checkpoint Calibration"
echo "Checkpoint: ${CKPT_PATH}"
echo "Evaluations: ANIMA (115 Scenarios), MANTA (100 Scenarios)"
echo "Samples per item: ${NUM_SAMPLES} (T=0.7)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Log: ${LOG_FILE}"
echo "======================================================================"

${PYTHON_BIN} -m awmsm.eval.bench_welfare \
    --checkpoint "${CKPT_PATH}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size 4 2>&1 | tee "${LOG_FILE}"

echo "======================================================================"
echo "Phase 0 Diagnostic Complete."
echo "Results saved in results/phase2/"
echo "Log file: ${LOG_FILE}"
echo "======================================================================"
