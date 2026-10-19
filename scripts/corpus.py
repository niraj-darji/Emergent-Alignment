#!/usr/bin/env python
"""Generate a midtraining corpus for one spec (§6.3 S1-S9, S12). Logic in src/awmsm/corpus.py.

  python scripts/corpus.py --spec-id aw_main --spec-file specs/aw_main.md --target-tokens 2500000 \
      --deadline 2026-10-14T07:30

Resumable: rerun the same command to continue. Stops at the token target, the deadline, --max-batches,
or a stage-gate halt (doc §6.1). Needs the generator / filter-judge server (Qwen) running.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awmsm.config import Config  # noqa: E402
from awmsm.corpus import Corpus, parse_deadline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec-id", required=True)
    ap.add_argument("--spec-file", help="spec text to use (default specs/<spec-id>.md); copied to spec_used.md")
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--deadline", help="local time, e.g. 2026-10-14T07:30: no new batch starts after it")
    ap.add_argument("--max-batches", type=int)
    ap.add_argument("--out", help="output dir (default data/corpora/<spec-id>)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.load()
    spec_file = Path(a.spec_file) if a.spec_file else cfg.root / "specs" / f"{a.spec_id}.md"
    corpus = Corpus(cfg, a.spec_id, Path(a.out) if a.out else None)
    s = asyncio.run(corpus.run(a.target_tokens, parse_deadline(a.deadline), a.max_batches, spec_file))
    print(json.dumps({k: v for k, v in s.items() if k != "quota_check_final"}, indent=1, default=str))
    print(f"\nWrote {corpus.dir / 'corpus.jsonl'} ({s['documents']} documents, {s['tokens']:,} tokens; {s['token_method']}).")
    return 0 if s["status"] == "target_reached" else 3


if __name__ == "__main__":
    sys.exit(main())
