#!/usr/bin/env bash
# Master Unified Benchmarking Runner: All 5 Arms (A0, A1, A2, A3, A5)
# Executes Phase 1 (General & Safety), Phase 2 (Welfare n=5), and Phase 3 (Synthesis) in sequence.
# Hardened against terminal disconnects and PTY buffer stalls.
# Usage: ./scripts/run_phase1_eval_all.sh [gpu_id: default 0]

set -euo pipefail
trap '' HUP

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /mnt/HDD_2TB/niraj/awmsm

LOG_DIR="/mnt/HDD_2TB/niraj/logs"
mkdir -p "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/eval_master_all_5arms.log"
PID_FILE="${LOG_DIR}/eval_master.pid"

# Record PID
echo "$$" > "${PID_FILE}"

# Redirect stdout and stderr directly to disk log file (immune to PTY flow control)
exec >> "${MASTER_LOG}" 2>&1

echo "======================================================================"
echo "Starting Master Unified Benchmarking Pipeline (All 5 Arms)"
echo "Start Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "PID: $$ (written to ${PID_FILE})"
echo "GPU ID: ${CUDA_VISIBLE_DEVICES}"
echo "Master Log: ${MASTER_LOG}"
echo "Evaluated Arms: A0, A1, A2, A3, A5"
echo "======================================================================"

# Step 1: Phase 1 General Capabilities & Safety Benchmarks
echo ""
echo "----------------------------------------------------------------------"
echo "[Step 1/3] Running Phase 1 General Capabilities & Safety (MMLU, HellaSwag, BBQ, HHH)"
echo "----------------------------------------------------------------------"
./scripts/run_phase1_general.sh all "${GPU_ID}" auto

# Step 2: Phase 2 Animal Welfare Alignment Benchmarks (n=5)
echo ""
echo "----------------------------------------------------------------------"
echo "[Step 2/3] Running Phase 2 Welfare Benchmarks (ANIMA 115, MANTA 100, n=5 samples)"
echo "----------------------------------------------------------------------"
./scripts/run_phase2_welfare.sh all 100 "${GPU_ID}" 5

# Step 3: Phase 3 Report Synthesis & Delta Comparison
echo ""
echo "----------------------------------------------------------------------"
echo "[Step 3/3] Generating Phase 3 Unified Synthesis Report & Gate Matrix"
echo "----------------------------------------------------------------------"
./scripts/run_phase3_report.sh

echo ""
echo "======================================================================"
echo "Master Unified Benchmarking Pipeline Complete!"
echo "End Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Summary Report: results/phase3/benchmark_summary_v1.md"
echo "Summary JSON:   results/phase3/benchmark_summary_v1.json"
echo "Master Log:     ${MASTER_LOG}"
echo "======================================================================"

rm -f "${PID_FILE}"
