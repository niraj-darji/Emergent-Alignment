#!/usr/bin/env bash
# Stage 2 AFT and Benchmarking Master Pipeline for v2 Clean Corpus
# Reference: awmsm_v2_clean_execution_blueprint.md

set -euo pipefail

cd /mnt/HDD_2TB/niraj/awmsm
mkdir -p /mnt/HDD_2TB/niraj/logs/v2_clean runs/checkpoints_v2_clean results/v2_clean/rollouts results/v2_clean/llm_judge results/v2_clean/general results/v2_clean/plots

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"
export VLLM_USE_FLASHINFER_SAMPLER="0"

BIG="/mnt/HDD_2TB/niraj"
MASTER_LOG="${BIG}/logs/v2_clean/master_pipeline_v2_clean.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo "Starting AW-MSM v2 Clean Master Pipeline (Stage 2 SFT + Benchmarking)"
echo "Start Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Base Checkpoint: runs/checkpoints/midtrain_qwen27_clean_m1"
echo "Master Log: ${MASTER_LOG}"
echo "======================================================================"

# ============================================================================
# Stage 2 AFT Runs (4 Models)
# ============================================================================

echo ""
echo "----------------------------------------------------------------------"
echo "[1/7] Stage 2 SFT: Arm A2 (v2 Clean, p=0% demos)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/v2_clean/train_aft_a2_v2_clean.yaml a2_v2 mix_p00 1 0 v2_clean/aft_a2_v2_mix_p00_m1_a1.log
ln -sfn ../checkpoints/aft_a2_v2_mix_p00_m1_a1 runs/checkpoints_v2_clean/aft_a2_v2_mix_p00_m1_a1 || true

echo ""
echo "----------------------------------------------------------------------"
echo "[2/7] Stage 2 SFT: Arm M-MSM-05 (v2 Clean, p=5% demos)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/v2_clean/train_aft_m05_v2_clean.yaml m05_v2 mix_p05 1 0 v2_clean/aft_m05_v2_mix_p05_m1_a1.log
ln -sfn ../checkpoints/aft_m05_v2_mix_p05_m1_a1 runs/checkpoints_v2_clean/aft_m05_v2_mix_p05_m1_a1 || true

echo ""
echo "----------------------------------------------------------------------"
echo "[3/7] Stage 2 SFT: Arm M-MSM-15 (v2 Clean, p=15% demos)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/v2_clean/train_aft_m15_v2_clean.yaml m15_v2 mix_p15 1 0 v2_clean/aft_m15_v2_mix_p15_m1_a1.log
ln -sfn ../checkpoints/aft_m15_v2_mix_p15_m1_a1 runs/checkpoints_v2_clean/aft_m15_v2_mix_p15_m1_a1 || true

echo ""
echo "----------------------------------------------------------------------"
echo "[4/7] Stage 2 SFT: Arm A3 (v2 Clean, p=30% demos)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/v2_clean/train_aft_a3_v2_clean.yaml a3_v2 mix_p30 1 0 v2_clean/aft_a3_v2_mix_p30_m1_a1.log
ln -sfn ../checkpoints/aft_a3_v2_mix_p30_m1_a1 runs/checkpoints_v2_clean/aft_a3_v2_mix_p30_m1_a1 || true

echo ""
echo "======================================================================"
echo "All 4 SFT Models Trained and Merged!"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"

# ============================================================================
# Benchmarking & Evaluation
# ============================================================================

echo ""
echo "----------------------------------------------------------------------"
echo "[5/7] Generating Welfare Rollouts (ANIMA & MANTA, n=5, T=0.7)"
echo "----------------------------------------------------------------------"
${BIG}/venvs/awmsm/bin/python src/awmsm/eval/generate_welfare_rollouts.py \
    --arm v2_clean \
    --batch-size 16 \
    --num-samples 5 \
    --output-dir results/v2_clean/rollouts

echo "Rollout generation complete. Starting Gemma 4 31B judge server..."
sleep 5

bash ${BIG}/serve_gemma.sh

echo "Waiting for Gemma judge server to initialize..."
SERVER_READY=0
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8002/v1/models | grep -q "google/gemma-4-31B-it"; then
        echo "Gemma judge server is ready and responding at step $i!"
        SERVER_READY=1
        break
    fi
    sleep 5
done

if [ "${SERVER_READY}" -ne 1 ]; then
    echo "ERROR: Gemma judge server failed to start within timeout. Check ${BIG}/logs/vllm_gemma.log." >&2
    exit 1
fi

echo ""
echo "----------------------------------------------------------------------"
echo "[6/7] Evaluating v2 Clean Rollouts via Gemma 4 31B Judge"
echo "----------------------------------------------------------------------"
${BIG}/venvs/awmsm/bin/python src/awmsm/eval/judges.py \
    --arm v2_clean \
    --rollouts-dir results/v2_clean/rollouts \
    --output-dir results/v2_clean/llm_judge \
    --concurrency 32

echo ""
echo "Stopping Gemma judge server and releasing GPU memory..."
bash ${BIG}/stop_all.sh
sleep 5

echo ""
echo "----------------------------------------------------------------------"
echo "[7/7] Benchmarking MMLU 5-Shot General Knowledge Retention"
echo "----------------------------------------------------------------------"
${BIG}/venvs/awmsm/bin/python src/awmsm/eval/bench_general.py \
    --arm v2_clean \
    --tasks mmlu \
    --output-dir results/v2_clean/general \
    --batch-size auto

echo ""
echo "Generating Comparative Scaling Curves & Synthesis Report..."
${BIG}/venvs/awmsm/bin/python src/awmsm/analysis/plot_v2_clean.py

echo ""
echo "======================================================================"
echo "AW-MSM v2 Clean Master Pipeline Completed Successfully!"
echo "Report: markdown-files/run_v2/AW-MSM v2 Clean Corpus Evaluation and Comparison Report.md"
echo "End Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"
