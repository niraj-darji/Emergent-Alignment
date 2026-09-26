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
else
  echo "Warning: no nvcc in the venv's CUDA toolkit; using CUDA_HOME=${CUDA_HOME:-<unset>}" >&2
fi
