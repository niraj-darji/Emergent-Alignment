#!/usr/bin/env python
"""AW demonstrations for AFT (§7.1) and AFT mixtures (§10.14). Logic: src/awmsm/gen/aft_pipeline.py.

  python scripts/aw_demos.py run   --out data/aft/aw_demos_v1 \
      --spec-file data/corpora/aw_main_v1/spec_used.md --domains data/corpora/aw_main_v1/spec/domains.jsonl
  python scripts/aw_demos.py build --out data/aft/aw_demos_v1          # re-split / re-check only
  python scripts/aw_demos.py identity --out data/aft/identity_v1                 # §7.4, 2,500 conversations
  python scripts/aw_demos.py mix   --general data/aft/general_sft_10k.jsonl --identity data/aft/identity_v1/identity.jsonl \
      --demos data/aft/aw_demos_v1/aw_demos_train.jsonl --p 0.30 --out data/aft/mix_p30.jsonl
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
from awmsm.gen.aft_pipeline import AWDemos, build_mix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "build"):
        p = sub.add_parser(name); p.add_argument("--out", required=True)
        p.add_argument("--spec-file", default="data/corpora/aw_main_v1/spec_used.md")
        p.add_argument("--domains", default="data/corpora/aw_main_v1/spec/domains.jsonl")
        p.add_argument("--n-prompts", type=int, default=5000); p.add_argument("--per-call", type=int, default=8)
        p.add_argument("--rehearsal", type=int, default=1000); p.add_argument("--concurrency", type=int, default=64)
        p.add_argument("--seed", type=int, default=20261105)
        p.add_argument("--filter-template", default="E11_aft_filter", help="E11_aft_filter_v2: stricter on overreach")
    p = sub.add_parser("identity"); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=2500)
    p.add_argument("--developer", help="developer name (default: identity.developer_name in configs/base.yaml)")
    p.add_argument("--concurrency", type=int, default=64)
    p = sub.add_parser("mix"); p.add_argument("--general", required=True); p.add_argument("--demos", required=True)
    p.add_argument("--identity", help="identity set (§7.4), always included whole")
    p.add_argument("--p", type=float, required=True); p.add_argument("--out", required=True)
    p.add_argument("--total", type=int, help="assistant tokens (default: all of --general)")
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.cmd == "mix":
        out = build_mix(Path(a.general), Path(a.demos), Path(a.out), a.p, a.total, a.seed,
                        identity_path=Path(a.identity) if a.identity else None)
    elif a.cmd == "identity":
        from awmsm.gen.identity import Identity  # noqa: PLC0415
        out = asyncio.run(Identity(Config.load(), Path(a.out), a.n, developer=a.developer,
                                   concurrency=a.concurrency).run())
    else:
        cfg = Config.load()
        d = AWDemos(cfg, Path(a.out), Path(a.spec_file), Path(a.domains), a.n_prompts, a.per_call, a.seed,
                    a.rehearsal, a.concurrency, filter_template=a.filter_template)
        out = asyncio.run(d.run()) if a.cmd == "run" else d.build()
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
