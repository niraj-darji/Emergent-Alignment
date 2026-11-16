#!/usr/bin/env bash
# Phase 2 Master Pipeline: Sequential SFT Training for M-Grid (Data-Mix Efficiency Curve)
# Models trained:
#   1. M-05: Raw Base + mix_p05.jsonl (p = 5%)
#   2. M-15: Raw Base + mix_p15.jsonl (p = 15%)
#   3. M-MSM-05: midtrain_aw_main_v1_m1 + mix_p05.jsonl (p = 5%)
#   4. M-MSM-15: midtrain_aw_main_v1_m1 + mix_p15.jsonl (p = 15%)
# Reference: final training and benchmarking roadmap.todo.md (Phase 2, Tasks 2.1 - 2.4)

set -euo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"

cd /mnt/HDD_2TB/niraj/awmsm

MASTER_LOG="/mnt/HDD_2TB/niraj/logs/phase2_mgrid_master_pipeline.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo "Starting Phase 2 M-Grid Sequential Training Pipeline"
echo "Start Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "GPU ID: ${CUDA_VISIBLE_DEVICES}"
echo "Master Log: ${MASTER_LOG}"
echo "======================================================================"

# Step 1: Arm M-05 (Raw Base + mix_p05)
echo ""
echo "----------------------------------------------------------------------"
echo "[1/4] Training Arm M-05: Raw Llama-3.1-8B + mix_p05 (5% AW Demos)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_m05_p05.yaml m05 mix_p05 1 "${GPU_ID}"
# Link alias for Table 1.2 path convention
ln -sfn aft_m05_mix_p05_m1_a1 runs/checkpoints/aft_m05_raw_m1_a1 || true

# Step 2: Arm M-15 (Raw Base + mix_p15)
echo ""
echo "----------------------------------------------------------------------"
echo "[2/4] Training Arm M-15: Raw Llama-3.1-8B + mix_p15 (15% AW Demos)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_m15_p15.yaml m15 mix_p15 1 "${GPU_ID}"
# Link alias for Table 1.2 path convention
ln -sfn aft_m15_mix_p15_m1_a1 runs/checkpoints/aft_m15_raw_m1_a1 || true

# Step 3: Arm M-MSM-05 (MSM Base + mix_p05)
echo ""
echo "----------------------------------------------------------------------"
echo "[3/4] Training Arm M-MSM-05: MSM aw_main_v1 + mix_p05 (5% AW Demos)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_m_msm05_p05.yaml m_msm05 mix_p05 1 "${GPU_ID}"
# Link alias for Table 1.2 path convention
ln -sfn aft_m_msm05_mix_p05_m1_a1 runs/checkpoints/aft_m05_aw_main_v1_m1_a1 || true

# Step 4: Arm M-MSM-15 (MSM Base + mix_p15)
echo ""
echo "----------------------------------------------------------------------"
echo "[4/4] Training Arm M-MSM-15: MSM aw_main_v1 + mix_p15 (15% AW Demos)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_m_msm15_p15.yaml m_msm15 mix_p15 1 "${GPU_ID}"
# Link alias for Table 1.2 path convention
ln -sfn aft_m_msm15_mix_p15_m1_a1 runs/checkpoints/aft_m15_aw_main_v1_m1_a1 || true

echo ""
echo "======================================================================"
echo "Phase 2 M-Grid Training Complete for All 4 Models!"
echo "End Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"
