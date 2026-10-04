#!/bin/bash
# Candidate A / generator: Qwen3.8-27B-FP8 on 127.0.0.1:8000 (settings: claude/phase0_setup_log.md)
BIG=${BIG:-/mnt/HDD_2TB/niraj}
source $BIG/venvs/vllm/bin/activate
QWEN_REV=${QWEN_REV:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}
HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
nohup vllm serve Qwen/Qwen3.8-27B-FP8 --revision $QWEN_REV \
  --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 \
  --language-model-only --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 256 \
  --enable-prefix-caching --default-chat-template-kwargs '{"enable_thinking": false}' \
  --generation-config vllm --seed 0 > $BIG/logs/vllm_qwen.log 2>&1 &
echo $! > $BIG/logs/vllm_qwen.pid
echo "Started Qwen (pid $(cat $BIG/logs/vllm_qwen.pid)); log: $BIG/logs/vllm_qwen.log"
