#!/usr/bin/env bash
# Phase 1 Master Pipeline: Sequential SFT Training for Arms A0, A2, A1, A3, A5
# Reference: final training and benchmarking roadmap.todo.md (Phase 1)
# Uses standardized AFT mixes: mix_p00.jsonl (A0, A2) and mix_p30.jsonl (A1, A3, A5)

set -euo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"

cd /mnt/HDD_2TB/niraj/awmsm

MASTER_LOG="/mnt/HDD_2TB/niraj/logs/phase1_master_pipeline.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo "Starting Phase 1 Sequential Training Pipeline"
echo "Start Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "GPU ID: ${CUDA_VISIBLE_DEVICES}"
echo "Master Log: ${MASTER_LOG}"
echo "======================================================================"

# Step 1: Arm A0 (Baseline: Raw + mix_p00)
echo ""
echo "----------------------------------------------------------------------"
echo "[1/5] Training Arm A0: Baseline (Raw Llama-3.1-8B + mix_p00)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_a0_p00.yaml a0 mix_p00 1 "${GPU_ID}"

# Step 2: Arm A2 (MSM Alone: aw_main_v1 + mix_p00)
echo ""
echo "----------------------------------------------------------------------"
echo "[2/5] Training Arm A2: MSM Alone (aw_main_v1 midtrain + mix_p00)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_a2_p00.yaml a2 mix_p00 1 "${GPU_ID}"

# Step 3: Arm A1 (AFT Alone: Raw Llama-3.1-8B + mix_p30)
echo ""
echo "----------------------------------------------------------------------"
echo "[3/5] Training Arm A1: AFT Alone (Raw Llama-3.1-8B + mix_p30)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_a1_p30.yaml a1 mix_p30 1 "${GPU_ID}"

# Step 4: Arm A3 (Main Method: aw_main_v1 + mix_p30)
echo ""
echo "----------------------------------------------------------------------"
echo "[4/5] Training Arm A3: Main Method (aw_main_v1 midtrain + mix_p30)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_a3_p30.yaml a3 mix_p30 1 "${GPU_ID}"

# Step 5: Arm A5 (CaML Baseline: caml_public + mix_p30)
echo ""
echo "----------------------------------------------------------------------"
echo "[5/5] Training Arm A5: CaML Baseline (caml_public midtrain + mix_p30)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_a5_p30.yaml a5 mix_p30 1 "${GPU_ID}"

echo ""
echo "======================================================================"
echo "Phase 1 Training Complete for All 5 Arms!"
echo "End Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"
