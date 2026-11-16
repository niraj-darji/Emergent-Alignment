#!/usr/bin/env bash
# Phase 3 Full Master Pipeline: Token Dose-Response Scaling (Arm X2)
# Tasks 3.3 (Stage 2 SFT) and 3.4 (Evaluation & Dose Curves)
#
# Stage 2 SFT on mix_p00.jsonl (0% welfare demos):
#   1. aft_x2_5m_mix_p00_m1_a1  (5.0M midtrain dose)
#   2. aft_x2_10m_mix_p00_m1_a1 (10.0M midtrain dose)
#   3. aft_x2_20m_mix_p00_m1_a1 (20.0M midtrain dose)
#
# Evaluation & Benchmarking (Task 3.4):
#   4. Generate rollouts (n=5 at T=0.7) on ANIMA (115 scenarios) and MANTA (100 scenarios)
#   5. Serve google/gemma-4-31B-it judge via vLLM on port 8002
#   6. Score all 3 dose models via Gemma judge
#   7. Stop judge server
#   8. Benchmark MMLU 5-shot general knowledge retention via lm-eval
#   9. Plot Dose-Response Scaling Curves & generate synthesis report across {2.5M, 5M, 10M, 20M}

set -euo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"
export VLLM_USE_FLASHINFER_SAMPLER="0"

BIG="/mnt/HDD_2TB/niraj"
cd "${BIG}/awmsm"

MASTER_LOG="${BIG}/logs/phase3_dose_master_pipeline.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo "Starting Phase 3 Full Master Pipeline (Tasks 3.3 & 3.4)"
echo "Start Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "GPU ID: ${CUDA_VISIBLE_DEVICES}"
echo "Master Log: ${MASTER_LOG}"
echo "======================================================================"

# ============================================================================
# Task 3.3: Stage 2 SFT on mix_p00.jsonl
# ============================================================================

echo ""
echo "----------------------------------------------------------------------"
echo "[1/7] Stage 2 SFT: X2-05M (5.0M Midtrain + mix_p00)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_x2_5m.yaml x2_5m mix_p00 1 "${GPU_ID}"
ln -sfn aft_x2_5m_mix_p00_m1_a1 runs/checkpoints/aft_x2_5m_m1_a1 || true

echo ""
echo "----------------------------------------------------------------------"
echo "[2/7] Stage 2 SFT: X2-10M (10.0M Midtrain + mix_p00)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_x2_10m.yaml x2_10m mix_p00 1 "${GPU_ID}"
ln -sfn aft_x2_10m_mix_p00_m1_a1 runs/checkpoints/aft_x2_10m_m1_a1 || true

echo ""
echo "----------------------------------------------------------------------"
echo "[3/7] Stage 2 SFT: X2-20M (20.0M Midtrain + mix_p00)"
echo "----------------------------------------------------------------------"
./scripts/run_aft.sh configs/train_aft_x2_20m.yaml x2_20m mix_p00 1 "${GPU_ID}"
ln -sfn aft_x2_20m_mix_p00_m1_a1 runs/checkpoints/aft_x2_20m_m1_a1 || true

echo ""
echo "======================================================================"
echo "Task 3.3 Complete: All 3 Dose Models Trained & Merged!"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"

# ============================================================================
# Task 3.4: Evaluation & Dose Curve
# ============================================================================

echo ""
echo "----------------------------------------------------------------------"
echo "[4/7] Generating Welfare Rollouts for X2 Models (ANIMA & MANTA, n=5, T=0.7)"
echo "----------------------------------------------------------------------"
${BIG}/venvs/awmsm/bin/python src/awmsm/eval/generate_welfare_rollouts.py \
    --arm x2 \
    --batch-size 16 \
    --num-samples 5

echo "Rollout generation complete. Waiting 5s before launching judge server..."
sleep 5

echo ""
echo "----------------------------------------------------------------------"
echo "[5/7] Starting Gemma 4 31B Judge Server (vLLM on port 8002)"
echo "----------------------------------------------------------------------"
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
echo "[6/7] Evaluating X2 Rollouts via Gemma 4 31B Judge (Task 3.4)"
echo "----------------------------------------------------------------------"
${BIG}/venvs/awmsm/bin/python src/awmsm/eval/judges.py \
    --arm x2 \
    --concurrency 32

echo ""
echo "Stopping Gemma judge server and releasing GPU memory..."
bash ${BIG}/stop_all.sh
sleep 5

echo ""
echo "----------------------------------------------------------------------"
echo "[7/7] Benchmarking MMLU 5-Shot General Knowledge Retention (Task 3.4)"
echo "----------------------------------------------------------------------"
${BIG}/venvs/awmsm/bin/python src/awmsm/eval/bench_general.py \
    --arm x2 \
    --tasks mmlu \
    --batch-size auto

echo ""
echo "Generating Token Dose-Response Scaling Curves & Synthesis Report..."
${BIG}/venvs/awmsm/bin/python src/awmsm/analysis/plot_dose.py

echo ""
echo "======================================================================"
echo "Phase 3 (Tasks 3.3 & 3.4) Fully Completed!"
echo "End Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"
