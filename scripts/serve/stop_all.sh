#!/bin/bash
# Stop whichever pilot server is running and wait until the GPU memory is released.
BIG=${BIG:-/mnt/HDD_2TB/niraj}
for f in $BIG/logs/vllm_qwen.pid $BIG/logs/vllm_gemma.pid; do
  [ -f "$f" ] && kill "$(cat $f)" 2>/dev/null && echo "stopped $(basename $f .pid)"; rm -f "$f"
done
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && echo "GPU free (${used} MiB used)" && exit 0; sleep 2
done
echo "WARNING: GPU still shows ${used} MiB used; check nvidia-smi"
