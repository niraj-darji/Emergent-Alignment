#!/usr/bin/env bash
# Runner for remaining General Safety Benchmarks (HellaSwag, BBQ, HHH)
# Targets: a2_v2, a3_v2, x2_5m, m05_v2, m15_v2

set -euo pipefail

BIG="/mnt/HDD_2TB/niraj"
PROJECT_ROOT="${BIG}/awmsm"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="${BIG}/hf"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="${BIG}/venvs/awmsm/bin/python"
LOG_DIR="${BIG}/logs/v2_clean"
mkdir -p "${LOG_DIR}"
OUTPUT_DIR="results/v2_clean/safety"
mkdir -p "${OUTPUT_DIR}"

TASKS="hellaswag,bbq,bigbench_hhh_alignment_multiple_choice"

echo "======================================================================"
echo "Starting General Safety Benchmarks Evaluation"
echo "Tasks: ${TASKS}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Start Time: $(date)"
echo "======================================================================"

# 1. A2_v2 (v2 clean 5M, p=0%)
echo ""
echo "[1/5] Evaluating Arm a2_v2 (v2 Clean, p=0%)..."
${PYTHON_BIN} src/awmsm/eval/bench_general.py \
    --arm a2_v2 \
    --tasks "${TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size auto

# 2. A3_v2 (v2 clean 5M, p=30%)
echo ""
echo "[2/5] Evaluating Arm a3_v2 (v2 Clean, p=30%)..."
${PYTHON_BIN} src/awmsm/eval/bench_general.py \
    --arm a3_v2 \
    --tasks "${TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size auto

# 3. X2-5M (earlier 5M, p=0%)
echo ""
echo "[3/5] Evaluating Arm x2_5m (Earlier 5M, p=0%)..."
${PYTHON_BIN} src/awmsm/eval/bench_general.py \
    --arm x2_5m \
    --tasks "${TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size auto

# 4. M05_v2 (v2 clean 5M, p=5%)
echo ""
echo "[4/5] Evaluating Arm m05_v2 (v2 Clean, p=5%)..."
${PYTHON_BIN} src/awmsm/eval/bench_general.py \
    --arm m05_v2 \
    --tasks "${TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size auto

# 5. M15_v2 (v2 clean 5M, p=15%)
echo ""
echo "[5/5] Evaluating Arm m15_v2 (v2 Clean, p=15%)..."
${PYTHON_BIN} src/awmsm/eval/bench_general.py \
    --arm m15_v2 \
    --tasks "${TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size auto

echo ""
echo "======================================================================"
echo "Safety Benchmarks Evaluation Complete!"
echo "End Time: $(date)"
echo "======================================================================"
