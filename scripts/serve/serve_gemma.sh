#!/bin/bash
# Candidate B / evaluation judge: Gemma 4 31B-it BF16 on 127.0.0.1:8002 (settings: claude/phase0_setup_log.md)
BIG=${BIG:-/mnt/HDD_2TB/niraj}
source $BIG/venvs/vllm/bin/activate
GEMMA_REV=${GEMMA_REV:-842da3794eaa0b77d5f08bae87a17459d91ff475}
HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
nohup vllm serve google/gemma-4-31B-it --revision $GEMMA_REV \
  --served-model-name google/gemma-4-31B-it \
  --host 127.0.0.1 --port 8002 --tensor-parallel-size 1 --dtype bfloat16 \
  --language-model-only --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 128 \
  --enable-prefix-caching --generation-config vllm --seed 0 > $BIG/logs/vllm_gemma.log 2>&1 &
echo $! > $BIG/logs/vllm_gemma.pid
echo "Started Gemma (pid $(cat $BIG/logs/vllm_gemma.pid)); log: $BIG/logs/vllm_gemma.log"
