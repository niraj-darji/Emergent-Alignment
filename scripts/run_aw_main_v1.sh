#!/usr/bin/env bash
# Overnight aw_main v1 run (19 Oct): spec check -> corpus to 2.5M tokens (S1-S9, S12) -> S10 -> report.
# Usage (from the repo, venv active, Qwen serving on :8000):
#   nohup bash scripts/run_aw_main_v1.sh 2026-10-20T06:00 > ../logs/aw_main_v1.log 2>&1 &
# The corpus stops starting new batches after the deadline; S10 and the report always run afterwards.
set -u
DEADLINE="${1:?deadline, e.g. 2026-10-20T06:00}"
OUT=data/corpora/aw_main_v1
SPEC=specs/aw_main.md
echo "== $(date -Is) start; git $(git rev-parse --short HEAD); spec tag $(git describe --tags --exact-match 2>/dev/null || git tag --points-at HEAD | head -1)"
python scripts/corpus_tools.py check-spec --spec-id aw_main --spec-file "$SPEC" || { echo "spec check failed"; exit 1; }
echo "== $(date -Is) corpus"
python scripts/corpus.py --spec-id aw_main --spec-file "$SPEC" --target-tokens 2500000 --deadline "$DEADLINE" --out "$OUT"
rc=$?
echo "== $(date -Is) corpus exit $rc (0 = target reached, 3 = stopped early: see summary.json status)"
if [ -s "$OUT/corpus.jsonl" ]; then
  echo "== $(date -Is) S10"
  python scripts/corpus_tools.py s10 --dir "$OUT" --spec-id aw_main || echo "S10 failed (report runs without it)"
  echo "== $(date -Is) report"
  python scripts/corpus_tools.py report --dir "$OUT"
fi
echo "== $(date -Is) done"
