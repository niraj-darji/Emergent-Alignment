#!/usr/bin/env bash
# Stage 1: Model Spec Midtraining Runner for Clean Qwen27 Corpus (4.64M tokens)
# Reference: awmsm_v2_clean_execution_blueprint.md

set -euo pipefail

cd /mnt/HDD_2TB/niraj/awmsm
mkdir -p /mnt/HDD_2TB/niraj/logs/v2_clean runs/checkpoints_v2_clean

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"

LOG_FILE="/mnt/HDD_2TB/niraj/logs/v2_clean/midtrain_qwen27_clean_m1.log"

echo "======================================================================"
echo "Starting Stage 1 Midtraining: midtrain_qwen27_clean_m1"
echo "Corpus: /mnt/HDD_2TB/niraj/awmsm_v2/out_full/corpus_qwen27_clean.jsonl"
echo "Log: ${LOG_FILE}"
echo "Start Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"

./scripts/run_train.sh configs/v2_clean/train_midtrain_v2_clean.yaml qwen27_clean 1 0 v2_clean/midtrain_qwen27_clean_m1.log

ln -sfn ../checkpoints/midtrain_qwen27_clean_m1 runs/checkpoints_v2_clean/midtrain_qwen27_clean_m1 || true

echo "======================================================================"
echo "Midtraining and Merge Complete for midtrain_qwen27_clean_m1!"
echo "Checkpoint: runs/checkpoints/midtrain_qwen27_clean_m1"
echo "End Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"
