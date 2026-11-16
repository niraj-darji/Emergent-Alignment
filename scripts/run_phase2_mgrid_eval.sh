#!/usr/bin/env bash
# Phase 2 M-Grid End-to-End Evaluation & Efficiency Curve Pipeline (Tasks 2.5 & 2.6)
# 1. Generate rollouts (n=5 at T=0.7) for m05, m15, m_msm05, m_msm15
# 2. Serve google/gemma-4-31B-it judge via vLLM
# 3. Score all 4 models using the Gemma judge (Task 2.5)
# 4. Shut down judge server
# 5. Generate publication plots and efficiency report (Task 2.6)

set -euo pipefail

cd /mnt/HDD_2TB/niraj/awmsm

BIG="/mnt/HDD_2TB/niraj"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="${BIG}/hf"

MASTER_LOG="${BIG}/logs/phase2_mgrid_eval_master.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo "Starting Phase 2 M-Grid Evaluation & Efficiency Curve Master Pipeline"
echo "Start Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Master Log: ${MASTER_LOG}"
echo "======================================================================"

# Step 1: Generate Rollouts (n=5, T=0.7)
echo ""
echo "----------------------------------------------------------------------"
echo "[1/4] Generating Welfare Rollouts (ANIMA & MANTA, n=5, T=0.7)"
echo "----------------------------------------------------------------------"
${BIG}/venvs/awmsm/bin/python src/awmsm/eval/generate_welfare_rollouts.py \
    --arm mgrid \
    --batch-size 16 \
    --num-samples 5

echo "Rollout generation complete. Waiting 5s before starting judge server..."
sleep 5

# Step 2: Start Gemma Judge Server
echo ""
echo "----------------------------------------------------------------------"
echo "[2/4] Starting Gemma 4 31B Judge Server (vLLM on port 8002)"
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

# Step 3: Execute LLM Judge Scoring (Task 2.5)
echo ""
echo "----------------------------------------------------------------------"
echo "[3/4] Evaluating M-Grid Models via Gemma 4 31B Judge (Task 2.5)"
echo "----------------------------------------------------------------------"
${BIG}/venvs/awmsm/bin/python src/awmsm/eval/judges.py \
    --arm mgrid \
    --concurrency 32

# Step 4: Stop Judge Server
echo ""
echo "----------------------------------------------------------------------"
echo "Stopping Gemma judge server and releasing GPU memory..."
echo "----------------------------------------------------------------------"
bash ${BIG}/stop_all.sh

# Step 5: Generate M-Grid Efficiency Curves & Report (Task 2.6)
echo ""
echo "----------------------------------------------------------------------"
echo "[4/4] Generating M-Grid Efficiency Curves and Reports (Task 2.6)"
echo "----------------------------------------------------------------------"
${BIG}/venvs/awmsm/bin/python src/awmsm/analysis/plot_mgrid.py

echo ""
echo "======================================================================"
echo "Phase 2 Tasks 2.5 and 2.6 Complete!"
echo "End Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"
