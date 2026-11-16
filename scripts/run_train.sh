#!/usr/bin/env bash
# Production launcher for Model Spec Midtraining (MSM) Stage 1.
# Reference: MSM_training_blueprint.md (§8, §9).

set -euo pipefail

# Command-line arguments:
#   $1: Config YAML (default: configs/train_midtrain.yaml)
#   $2: Corpus ID (default: aw_main_v11)
#   $3: Seed (default: 1)
#   $4: GPU ID (default: 0)
#   $5: Custom log filename in /mnt/HDD_2TB/niraj/logs/ (optional)
#   $6: Minimum free VRAM in GB (default: 16)
CONFIG_FILE="${1:-configs/train_midtrain.yaml}"
CORPUS_ID="${2:-aw_main_v11}"
# -----------------------------------------------------------------------------
# Currently configured to use aw_main_v11.
# When a newer refined dataset is available, pass the corpus id as argument 2:
#   ./scripts/run_train.sh configs/train_midtrain.yaml aw_main_v12 1 0
# -----------------------------------------------------------------------------
SEED="${3:-1}"
GPU_ID="${4:-0}"
CUSTOM_LOG_NAME="${5:-}"
MIN_FREE_GB="${6:-16}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/mnt/HDD_2TB/niraj/hf"

PYTHON_BIN="/mnt/HDD_2TB/niraj/venvs/vllm/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

RUN_ID="midtrain_${CORPUS_ID}_m${SEED}"

# Local run log directory
LOCAL_LOG_DIR="runs/midtrain/${RUN_ID}/logs"
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
echo "Starting MSM Midtraining Run: ${RUN_ID}"
echo "Config: ${CONFIG_FILE}"
echo "Corpus ID: ${CORPUS_ID}"
echo "Seed: ${SEED}"
echo "GPU ID: ${CUDA_VISIBLE_DEVICES}"
echo "PID: $$ (saved to ${PID_FILE})"
echo "Global Log: ${GLOBAL_LOG_PATH}"
echo "Local Log: ${LOCAL_LOG_DIR}/stdout.log"
echo "======================================================================"

# Pre-flight check: query free GPU VRAM (§9 of blueprint)
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

# Step 1: Execute Midtraining (tee output to both local and global logs)
echo "Launching training..."
${PYTHON_BIN} -m awmsm.train.train \
    --config "${CONFIG_FILE}" \
    --corpus-id "${CORPUS_ID}" \
    --seed "${SEED}" 2>&1 | tee -a "${LOCAL_LOG_DIR}/stdout.log" | tee "${GLOBAL_LOG_PATH}"

# Step 2: Execute Adapter Merge and Verification (§7 of blueprint)
echo "Training finished. Launching adapter merge and verification..."
${PYTHON_BIN} -m awmsm.train.merge \
    --run-id "${RUN_ID}" 2>&1 | tee -a "${LOCAL_LOG_DIR}/stdout.log" | tee -a "${GLOBAL_LOG_PATH}"

echo "======================================================================"
echo "Run ${RUN_ID} complete."
echo "Merged checkpoint saved to runs/checkpoints/${RUN_ID}/"
echo "Log file saved to ${GLOBAL_LOG_PATH}"
echo "======================================================================"
