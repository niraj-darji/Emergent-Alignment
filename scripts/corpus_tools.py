#!/usr/bin/env python
"""Corpus follow-up tools (logic in src/awmsm/corpus_tools.py and src/awmsm/gen/pattern_scan.py).

  python scripts/corpus_tools.py check-spec   --spec-id aw_main --spec-file specs/aw_main.md
  python scripts/corpus_tools.py s10          --dir data/corpora/aw_main_v1 --spec-id aw_main
  python scripts/corpus_tools.py report       --dir data/corpora/aw_main_v1
  python scripts/corpus_tools.py audit-export --dir data/corpora/aw_main_v1 --n-included 30 [--n-excluded 0]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awmsm import corpus_tools as ct  # noqa: E402
from awmsm.config import Config  # noqa: E402
from awmsm.gen import midtrain_pipeline as mp  # noqa: E402
from awmsm.gen.pattern_scan import run_s10  # noqa: E402
from awmsm.io import read_jsonl  # noqa: E402
from awmsm.llm import LLMClient  # noqa: E402


async def s10(cfg: Config, d: Path, spec_id: str) -> dict:
    ctx = mp.GenContext.create(cfg, spec_id, spec_path=d / "spec_used.md")
    docs = list(read_jsonl(d / "corpus.jsonl"))
    client = LLMClient(cfg.role("generator"), cfg, max_in_flight=256)
    try:
        await client.verify_server()
        return await run_s10(ctx, client, docs, d / "s10", cfg.gen.get("s10"),
                             concurrency=cfg.concurrency("s8_filter"))
    finally:
        await client.aclose()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check-spec"); p.add_argument("--spec-id", required=True); p.add_argument("--spec-file", required=True)
    p = sub.add_parser("s10"); p.add_argument("--dir", required=True); p.add_argument("--spec-id", required=True)
    p = sub.add_parser("report"); p.add_argument("--dir", required=True)
    p = sub.add_parser("audit-export"); p.add_argument("--dir", required=True)
    p.add_argument("--n-included", type=int, default=30); p.add_argument("--n-excluded", type=int, default=0)
    p.add_argument("--seed", type=int, default=1808)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.load()
    if a.cmd == "check-spec":
        out = ct.check_spec(cfg, a.spec_id, Path(a.spec_file))
    elif a.cmd == "s10":
        r = asyncio.run(s10(cfg, Path(a.dir), a.spec_id))
        out = {"flagged_for_review": r["flagged_for_review"],
               "rated": [(p["name"], p["prevalence"]) for p in r["rated_patterns"]]}
    elif a.cmd == "report":
        ct.report(cfg, Path(a.dir))
        print((Path(a.dir) / "report.md").read_text())
        return 0
    else:
        out = ct.audit_export(cfg, Path(a.dir), a.n_included, a.n_excluded, a.seed)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
