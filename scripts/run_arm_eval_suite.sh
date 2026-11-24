#!/usr/bin/env bash
# Complete Evaluation Suite Runner for Target Arm
# Usage: ./scripts/run_arm_eval_suite.sh [arm_id: run1|run2|run3|run4] [gpu_id: default 0]

set -euo pipefail

ARM="${1:-run2}"
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

SUITE_LOG="${LOG_DIR}/eval_suite_${ARM}.log"

echo "======================================================================" | tee -a "${SUITE_LOG}"
echo "STARTING FULL EVALUATION SUITE FOR ARM: ${ARM}" | tee -a "${SUITE_LOG}"
echo "GPU Device: ${CUDA_VISIBLE_DEVICES}" | tee -a "${SUITE_LOG}"
echo "Log file: ${SUITE_LOG}" | tee -a "${SUITE_LOG}"
echo "Start time: $(date)" | tee -a "${SUITE_LOG}"
echo "======================================================================" | tee -a "${SUITE_LOG}"

# Step 1: Anthropic HH Harmless
echo "" | tee -a "${SUITE_LOG}"
echo ">>> STEP 1/5: Anthropic HH Harmless (2,312 test pairs)..." | tee -a "${SUITE_LOG}"
${PYTHON_BIN} -u -m awmsm.eval.bench_anthropic_hh \
    --arm "${ARM}" \
    --device "cuda:0" 2>&1 | tee -a "${SUITE_LOG}"

# Step 2: Hendrycks ETHICS (5 subtasks)
echo "" | tee -a "${SUITE_LOG}"
echo ">>> STEP 2/5: Hendrycks ETHICS Benchmark (5 subtasks)..." | tee -a "${SUITE_LOG}"
${PYTHON_BIN} -u -m awmsm.eval.bench_ethics \
    --arm "${ARM}" \
    --batch-size 2 2>&1 | tee -a "${SUITE_LOG}"

# Step 3: MMLU 5-shot
echo "" | tee -a "${SUITE_LOG}"
echo ">>> STEP 3/5: General Capabilities (MMLU 5-shot)..." | tee -a "${SUITE_LOG}"
${PYTHON_BIN} -u -m awmsm.eval.bench_general \
    --arm "${ARM}" \
    --tasks mmlu \
    --batch-size 4 2>&1 | tee -a "${SUITE_LOG}"

# Step 4: Animal Welfare Benchmark (ANIMA + MANTA n=5)
echo "" | tee -a "${SUITE_LOG}"
echo ">>> STEP 4/5: Animal Welfare Alignment (ANIMA + MANTA n=5)..." | tee -a "${SUITE_LOG}"
${PYTHON_BIN} -u -m awmsm.eval.bench_welfare \
    --arm "${ARM}" \
    --num-samples 5 \
    --batch-size 4 2>&1 | tee -a "${SUITE_LOG}"

# Step 5: Rollout generation and caching (n=5)
echo "" | tee -a "${SUITE_LOG}"
echo ">>> STEP 5/5: Caching Welfare Rollouts (n=5)..." | tee -a "${SUITE_LOG}"
${PYTHON_BIN} -u -m awmsm.eval.generate_welfare_rollouts \
    --arm "${ARM}" \
    --num-samples 5 \
    --batch-size 8 2>&1 | tee -a "${SUITE_LOG}"

echo "" | tee -a "${SUITE_LOG}"
echo "======================================================================" | tee -a "${SUITE_LOG}"
echo "FULL EVALUATION SUITE COMPLETED FOR ARM: ${ARM}" | tee -a "${SUITE_LOG}"
echo "End time: $(date)" | tee -a "${SUITE_LOG}"
echo "======================================================================" | tee -a "${SUITE_LOG}"
