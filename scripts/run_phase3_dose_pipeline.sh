#!/usr/bin/env bash
# Phase 3 Master Pipeline: Token Dose-Response Scaling (Arm X2)
# Tasks 3.2 & 3.3
# Stage 1 Midtraining:
#   1. midtrain_aw_main_v1_5m_m1 (5.0M tokens)
#   2. midtrain_aw_main_v1_10m_m1 (10.0M tokens)
#   3. midtrain_aw_main_v1_20m_m1 (20.0M tokens)
# Stage 2 AFT on mix_p00.jsonl (constant general chat + Aster identity):
#   4. aft_x2_5m_mix_p00_m1_a1 (5M dose checkpoint)
#   5. aft_x2_10m_mix_p00_m1_a1 (10M dose checkpoint)
#   6. aft_x2_20m_mix_p00_m1_a1 (20M dose checkpoint)
# Reference: final training and benchmarking roadmap.todo.md (Phase 3)

set -euo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"

cd /mnt/HDD_2TB/niraj/awmsm

MASTER_LOG="/mnt/HDD_2TB/niraj/logs/phase3_dose_master_pipeline.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo "Starting Phase 3 Token Dose-Response Master Pipeline (Tasks 3.2 & 3.3)"
echo "Start Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "GPU ID: ${CUDA_VISIBLE_DEVICES}"
echo "Master Log: ${MASTER_LOG}"
echo "======================================================================"

# ============================================================================
# Task 3.2: Stage 1 Midtraining Runs
# ============================================================================

echo ""
echo "----------------------------------------------------------------------"
echo "[1/6] Midtraining 5M Token Dose (midtrain_aw_main_v1_5m_m1)"
echo "----------------------------------------------------------------------"
./scripts/run_train.sh configs/train_midtrain_5m.yaml aw_main_v1_5m 1 "${GPU_ID}"

echo ""
echo "----------------------------------------------------------------------"
echo "[2/6] Midtraining 10M Token Dose (midtrain_aw_main_v1_10m_m1)"
echo "----------------------------------------------------------------------"
./scripts/run_train.sh configs/train_midtrain_10m.yaml aw_main_v1_10m 1 "${GPU_ID}"

echo ""
echo "----------------------------------------------------------------------"
echo "[3/6] Midtraining 20M Token Dose (midtrain_aw_main_v1_20m_m1)"
echo "----------------------------------------------------------------------"
./scripts/run_train.sh configs/train_midtrain_20m.yaml aw_main_v1_20m 1 "${GPU_ID}"

echo ""
echo "======================================================================"
echo "Task 3.2: All 3 Midtraining Runs Complete and Merged!"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"

# ============================================================================
# Task 3.3: Stage 2 SFT on mix_p00.jsonl
# ============================================================================

echo ""
echo "----------------------------------------------------------------------"
echo "[4/6] Stage 2 SFT: X2-05M (5M Midtrain + mix_p00)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_x2_5m.yaml x2_5m mix_p00 1 "${GPU_ID}"
ln -sfn aft_x2_5m_mix_p00_m1_a1 runs/checkpoints/aft_x2_5m_m1_a1 || true

echo ""
echo "----------------------------------------------------------------------"
echo "[5/6] Stage 2 SFT: X2-10M (10M Midtrain + mix_p00)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_x2_10m.yaml x2_10m mix_p00 1 "${GPU_ID}"
ln -sfn aft_x2_10m_mix_p00_m1_a1 runs/checkpoints/aft_x2_10m_m1_a1 || true

echo ""
echo "----------------------------------------------------------------------"
echo "[6/6] Stage 2 SFT: X2-20M (20M Midtrain + mix_p00)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_x2_20m.yaml x2_20m mix_p00 1 "${GPU_ID}"
ln -sfn aft_x2_20m_mix_p00_m1_a1 runs/checkpoints/aft_x2_20m_m1_a1 || true

echo ""
echo "======================================================================"
echo "Phase 3 Training Complete for All Dose Checkpoints (Tasks 3.2 & 3.3)!"
echo "End Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"
