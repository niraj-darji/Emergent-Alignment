# Sourced by the run scripts after activating .venv.
#
# vLLM (through FlashInfer) compiles some GPU kernels the first time it starts,
# which needs nvcc. It looks in $CUDA_HOME first. A system CUDA_HOME can point to a
# toolkit without nvcc (e.g. /usr/local/cuda-12.8), or to a CUDA version that
# doesn't match PyTorch (cu130 here). Use the CUDA 13 toolkit that pip installed
# into the venv instead: it has nvcc and matches PyTorch.
CUDA_PIP="$(python -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))' 2>/dev/null || true)"
if [ -n "$CUDA_PIP" ] && [ -x "$CUDA_PIP/bin/nvcc" ]; then
  export CUDA_HOME="$CUDA_PIP"
  export PATH="$CUDA_HOME/bin:$PATH"
  export FLASHINFER_NVCC="nvcc -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
  if [ ! -d "$CUDA_HOME/lib64" ]; then
    ln -sf lib "$CUDA_HOME/lib64"
  fi
  if [ ! -e "$CUDA_HOME/lib/libcudart.so" ] && [ -e "$CUDA_HOME/lib/libcudart.so.13" ]; then
    ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
  fi
else
  echo "Warning: no nvcc in the venv's CUDA toolkit; using CUDA_HOME=${CUDA_HOME:-<unset>}" >&2
fi

# Automatically load .env if it exists
if [ -f "$(dirname "${BASH_SOURCE[0]}")/../.env" ]; then
  set -a
  source "$(dirname "${BASH_SOURCE[0]}")/../.env"
  set +a
fi

# The project uses HF_TOKEN_VALUE, map it to HF_TOKEN for standard HF tools
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HF_TOKEN_VALUE:-}" ]; then
  export HF_TOKEN="$HF_TOKEN_VALUE"
fi
