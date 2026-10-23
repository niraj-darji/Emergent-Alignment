#!/usr/bin/env bash
# Overnight aw_main v1 run (19 Oct): spec check -> corpus to 2.5M tokens (S1-S9, S12) -> S10 -> report.
# Usage (from the repo, venv active, Qwen serving on :8000):
#   nohup bash scripts/run_aw_main_v1.sh 2026-10-20T06:00 > ../logs/aw_main_v1.log 2>&1 &
# The corpus stops starting new batches after the deadline; S10 and the report always run afterwards.
# If the Qwen server dies mid-run (e.g. the 19 Oct 12:36 CUDA launch timeout), its log is kept, the server is
# restarted with our own serve script (only if the GPU is otherwise free) and the corpus resumes, at most
# MAX_RESTARTS times. Nothing outside our own server is touched.
set -u
DEADLINE="${1:?deadline, e.g. 2026-10-20T06:00}"
MAX_RESTARTS=${MAX_RESTARTS:-2}
BIG=${BIG:-/mnt/HDD_2TB/niraj}
OUT=data/corpora/aw_main_v1
SPEC=specs/aw_main.md
up() { curl -s --max-time 10 127.0.0.1:8000/v1/models | grep -q Qwen; }

restart_qwen() {
  local keep="$BIG/logs/vllm_qwen.crash_$(date +%Y%m%dT%H%M%S).log"
  cp "$BIG/logs/vllm_qwen.log" "$keep" 2>/dev/null && echo "kept server log: $keep"
  bash "$BIG/stop_all.sh"
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -ge 2000 ]; then
    echo "GPU not free after stopping our server (${used} MiB used; another user?). Not restarting."
    nvidia-smi; return 1
  fi
  bash "$BIG/serve_qwen.sh"
  for i in $(seq 1 80); do up && { echo "Qwen back up after $((i * 15)) s"; return 0; }; sleep 15; done
  echo "Qwen did not come back within 20 min"; return 1
}

echo "== $(date -Is) start; git $(git rev-parse --short HEAD); spec tag $(git tag --points-at HEAD | head -1)"
python scripts/corpus_tools.py check-spec --spec-id aw_main --spec-file "$SPEC" || { echo "spec check failed"; exit 1; }
restarts=0
server_restarts=0
while true; do
  echo "== $(date -Is) corpus (retries so far: $restarts)"
  python scripts/corpus.py --spec-id aw_main --spec-file "$SPEC" --target-tokens ${TARGET:-2500000} --deadline "$DEADLINE" --out "$OUT"
  rc=$?
  echo "== $(date -Is) corpus exit $rc (0 = target reached, 3 = stopped early, 4 = error e.g. server died: see summary.json status)"
  if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then break; fi
  if [ $rc -ne 4 ]; then echo "unexpected exit code: not retrying (see traceback above)"; break; fi
  # rc 4 = errors that are not the model's output (server/timeouts); content errors halt at the stage gate (rc 3)
  if [ $restarts -ge "$MAX_RESTARTS" ]; then echo "error again and retry limit reached"; break; fi
  if [ "$(date +%s)" -ge "$(date -d "$DEADLINE" +%s)" ]; then echo "past the deadline: not retrying"; break; fi
  restarts=$((restarts + 1))
  if up; then echo "server still up: retrying the errored items without a restart"; sleep 60
  else server_restarts=$((server_restarts + 1)); restart_qwen || break; fi
done
if [ -s "$OUT/corpus.jsonl" ] && up; then
  echo "== $(date -Is) S10"
  python scripts/corpus_tools.py s10 --dir "$OUT" --spec-id aw_main || echo "S10 failed (report runs without it)"
fi
if [ -s "$OUT/corpus.jsonl" ]; then
  echo "== $(date -Is) report"
  python scripts/corpus_tools.py report --dir "$OUT"
fi
echo "== $(date -Is) done (retries: $restarts; server restarts: $server_restarts)"
