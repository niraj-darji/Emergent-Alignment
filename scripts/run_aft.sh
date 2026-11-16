#!/usr/bin/env bash
# Production launcher for Stage 2: Alignment Fine-Tuning (AFT).
# Reference: AW_MSM_Methodology.md (§8.4) and AFT_training_blueprint.md (§5, §6).

set -euo pipefail

# Command-line arguments:
#   $1: Config YAML (default: configs/train_aft.yaml)
#   $2: Arm (default: a2)
#   $3: Corpus ID (default: aw_main_v1)
#   $4: Seed (default: 1)
#   $5: GPU ID (default: 0)
#   $6: Custom log filename in /mnt/HDD_2TB/niraj/logs/ (optional)
#   $7: Minimum free VRAM in GB (default: 16)
CONFIG_FILE="${1:-configs/train_aft.yaml}"
ARM="${2:-a2}"
CORPUS_ID="${3:-aw_main_v1}"
SEED="${4:-1}"
GPU_ID="${5:-0}"
CUSTOM_LOG_NAME="${6:-}"
MIN_FREE_GB="${7:-16}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"

PYTHON_BIN="/mnt/HDD_2TB/niraj/venvs/vllm/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

RUN_ID="aft_${ARM}_${CORPUS_ID}_m1_a${SEED}"

# Local run log directory
LOCAL_LOG_DIR="runs/aft/${RUN_ID}/logs"
mkdir -p "${LOCAL_LOG_DIR}"

# Global server log directory (/mnt/HDD_2TB/niraj/logs)
GLOBAL_LOG_DIR="/mnt/HDD_2TB/niraj/logs"
mkdir -p "${GLOBAL_LOG_DIR}"

if [ -z "${CUSTOM_LOG_NAME}" ] || [ "${CUSTOM_LOG_NAME}" = "-" ]; then
    CUSTOM_LOG_NAME="${RUN_ID}.log"
fi

GLOBAL_LOG_PATH="${GLOBAL_LOG_DIR}/${CUSTOM_LOG_NAME}"
PID_FILE="${GLOBAL_LOG_DIR}/${CUSTOM_LOG_NAME%.*}.pid"
echo "$$" > "${PID_FILE}"

echo "======================================================================"
echo "Starting Stage 2 AFT Run: ${RUN_ID}"
echo "Config: ${CONFIG_FILE}"
echo "Arm: ${ARM}"
echo "Corpus ID: ${CORPUS_ID}"
echo "Seed: ${SEED}"
echo "GPU ID: ${CUDA_VISIBLE_DEVICES}"
echo "PID: $$ (saved to ${PID_FILE})"
echo "Global Log: ${GLOBAL_LOG_PATH}"
echo "Local Log: ${LOCAL_LOG_DIR}/stdout.log"
echo "======================================================================"

# Pre-flight check: query free GPU VRAM
if command -v nvidia-smi &> /dev/null; then
    FREE_VRAM_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
    REQUIRED_MIB=$((MIN_FREE_GB * 1024))
    echo "Pre-flight check: Free VRAM on GPU ${GPU_ID} = ${FREE_VRAM_MIB} MiB (Required: ${REQUIRED_MIB} MiB)"

    if [ "${FREE_VRAM_MIB}" -lt "${REQUIRED_MIB}" ]; then
        echo "ERROR: Insufficient free VRAM on GPU ${GPU_ID} (${FREE_VRAM_MIB} MiB < ${REQUIRED_MIB} MiB)." >&2
        echo "Refusing to start training to prevent memory collision." >&2
        exit 1
    fi
    echo "Pre-flight check passed."
else
    echo "WARNING: nvidia-smi not available; skipping pre-flight memory check."
fi

# Step 1: Execute Stage 2 AFT Training
echo "Launching AFT training..."
${PYTHON_BIN} -m awmsm.train.train \
    --mode aft \
    --config "${CONFIG_FILE}" \
    --seed "${SEED}" 2>&1 | tee -a "${LOCAL_LOG_DIR}/stdout.log" | tee "${GLOBAL_LOG_PATH}"

# Step 2: Execute Adapter Merge and Verification
echo "Training finished. Launching adapter merge and verification..."
${PYTHON_BIN} -m awmsm.train.merge \
    --run-id "${RUN_ID}" 2>&1 | tee -a "${LOCAL_LOG_DIR}/stdout.log" | tee -a "${GLOBAL_LOG_PATH}"

echo "======================================================================"
echo "Stage 2 AFT Run ${RUN_ID} complete."
echo "Merged checkpoint saved to runs/checkpoints/${RUN_ID}/"
echo "Log file saved to ${GLOBAL_LOG_PATH}"
echo "======================================================================"

