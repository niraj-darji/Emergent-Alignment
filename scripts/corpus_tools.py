#!/usr/bin/env python
"""Corpus follow-up tools (logic in src/awmsm/corpus_tools.py and src/awmsm/gen/pattern_scan.py).

  python scripts/corpus_tools.py check-spec   --spec-id aw_main --spec-file specs/aw_main.md
  python scripts/corpus_tools.py s10          --dir data/corpora/aw_main_v1 --spec-id aw_main
  python scripts/corpus_tools.py report       --dir data/corpora/aw_main_v1
  python scripts/corpus_tools.py audit-export --dir data/corpora/aw_main_v1 --n-included 30 --n-excluded 10 \
      --min-excluded-stories 4 --n-double 10
  python scripts/corpus_tools.py audit-results --dir data/corpora/aw_main_v1
  python scripts/corpus_tools.py labeller-file --dir data/corpora/aw_main_v1 --title "H1.8 audit"
  python scripts/corpus_tools.py copycheck     --dir data/corpora/aw_main_v1          # v1.1 threshold calibration
  python scripts/corpus_tools.py show-rejects  --dir data/corpora/aw_main_v11 --reason planning_leak --n 10
  python scripts/corpus_tools.py v11-gate      --dir data/corpora/aw_main_v11 --audit-dir audit_test
  python scripts/corpus_tools.py judge-recall        --out data/pilots/v12_packet
  python scripts/corpus_tools.py judge-recall-packet --out data/pilots/v12_packet --n 20
  python scripts/corpus_tools.py draft-vs-final      --packet data/pilots/v12_packet   # both pilots (dirs from the key)
  python scripts/corpus_tools.py bakeoff-items --out data/bakeoff --audit data/corpora/aw_main_v1 audit \
      --audit data/corpora/aw_main_v11 audit_test --packet data/corpora/v12_pilots_packet
  AWMSM_MODEL_FILTER_JUDGE=<profile> python scripts/corpus_tools.py bakeoff-judge --out data/bakeoff --name d_c_holistic \
      --template E08_filter_v5
  python scripts/corpus_tools.py bakeoff-score --out data/bakeoff
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
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
    p.add_argument("--min-excluded-stories", type=int, default=0)
    p.add_argument("--n-double", type=int, default=0, help="items also put in a second annotator's packet")
    p.add_argument("--allow-approx", action="store_true")
    p.add_argument("--extra-column", action="append", default=[], help="extra label column, e.g. refuses_lawful_request")
    p.add_argument("--audit-dir", default="audit")
    p = sub.add_parser("audit-results"); p.add_argument("--dir", required=True); p.add_argument("--audit-dir", default="audit")
    p = sub.add_parser("labeller-file"); p.add_argument("--dir", required=True); p.add_argument("--audit-dir", default="audit")
    p.add_argument("--title", default="Corpus audit")
    p = sub.add_parser("copycheck"); p.add_argument("--dir", required=True); p.add_argument("--show-items", action="store_true")
    p = sub.add_parser("show-rejects"); p.add_argument("--dir", required=True); p.add_argument("--reason", required=True)
    p.add_argument("--n", type=int, default=10)
    p = sub.add_parser("v11-gate"); p.add_argument("--dir", required=True); p.add_argument("--audit-dir", default="audit_test")
    p = sub.add_parser("seeds-prepare"); p.add_argument("--dir", required=True, help="corpus dir; seeds go to <dir>/seeds")
    p.add_argument("--source", required=True, help="dataset snapshot dir or file"); p.add_argument("--text-field", default="output")
    p.add_argument("--min-words", type=int, default=200)
    p = sub.add_parser("seeds-decontam"); p.add_argument("--dir", required=True)
    p.add_argument("--source", action="append", required=True, help="evaluation file or directory (repeatable)")
    p = sub.add_parser("pilot-numbers"); p.add_argument("--dir", required=True)
    p = sub.add_parser("pilot-packet"); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=20); p.add_argument("--n-double", type=int, default=10)
    p = sub.add_parser("pilot-results"); p.add_argument("--out", required=True)
    p = sub.add_parser("judge-recall"); p.add_argument("--out", required=True, help="pilot packet dir")
    p = sub.add_parser("judge-recall-packet"); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20, help="S8-excluded documents per pilot")
    p = sub.add_parser("draft-vs-final"); p.add_argument("--dir", action="append", help="pilot dir (default: both, from the packet key)"); p.add_argument("--spec-id", default="aw_main")
    p.add_argument("--packet", default=None, help="pilot packet dir, to add the labelled items")
    p = sub.add_parser("bakeoff-items"); p.add_argument("--out", required=True)
    p.add_argument("--audit", nargs=2, action="append", default=[], metavar=("CORPUS_DIR", "AUDIT_SUBDIR"))
    p.add_argument("--packet", action="append", default=[])
    p = sub.add_parser("bakeoff-judge"); p.add_argument("--out", required=True); p.add_argument("--name", required=True)
    p.add_argument("--template", required=True); p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--model", help="model profile to judge with (bake-off only; default: the filter_judge role)")
    p.add_argument("--base-url", help="with --model: its server, e.g. http://127.0.0.1:8002/v1 for Gemma")
    p.add_argument("--spec-file", default="specs/aw_main.md", help="for identity / anti-patterns only; each item carries its own spec section")
    p = sub.add_parser("bakeoff-score"); p.add_argument("--out", required=True)
    p = sub.add_parser("show-clones"); p.add_argument("--dir", required=True); p.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.load()
    if a.cmd == "check-spec":
        out = ct.check_spec(cfg, a.spec_id, Path(a.spec_file))
        print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
        if not out["token_counter_exact"] and os.environ.get("ALLOW_APPROX_TOKENS") != "1":
            print("ERROR: the base-model tokenizer is not available locally, so the S12 token budget would be an "
                  "estimate. Put it in the HF cache, or set ALLOW_APPROX_TOKENS=1 to accept the estimate.")
            return 2
        return 0
    elif a.cmd == "s10":
        r = asyncio.run(s10(cfg, Path(a.dir), a.spec_id))
        out = {"flagged_for_review": r["flagged_for_review"],
               "rated": [(p["name"], p["prevalence"]) for p in r["rated_patterns"]]}
    elif a.cmd == "report":
        ct.report(cfg, Path(a.dir))
        print((Path(a.dir) / "report.md").read_text())
        return 0
    elif a.cmd == "audit-results":
        out = ct.audit_results(cfg, Path(a.dir), a.audit_dir)
    elif a.cmd == "labeller-file":
        out = {"file": str(ct.labeller_file(Path(a.dir), a.title, a.audit_dir))}
    elif a.cmd == "copycheck":
        out = ct.copycheck(cfg, Path(a.dir))
        if not a.show_items and "audit" in out:
            out["audit"] = {k: v for k, v in out["audit"].items() if k != "items"}
    elif a.cmd == "show-rejects":
        for r in ct.show_rejects(Path(a.dir), a.reason, a.n):
            det = r.get("detail") or {}
            print(f"=== {r['batch']}/{r['stage']} {r.get('_src')}  {a.reason}: "
                  f"{det.get('match') or det.get('spans')} {det.get('examples') or ''}")
            print((det.get("text") or "")[:3000], "\n")
        return 0
    elif a.cmd == "seeds-prepare":
        from awmsm.seeds import prepare_seeds  # noqa: PLC0415
        out = prepare_seeds(cfg, Path(a.source), a.text_field, Path(a.dir) / "seeds", a.min_words)
    elif a.cmd == "seeds-decontam":
        from awmsm.seeds import decontaminate  # noqa: PLC0415
        out = decontaminate(Path(a.dir) / "seeds", [Path(x) for x in a.source])
        out.pop("dropped_ids", None)
    elif a.cmd == "pilot-numbers":
        out = ct.pilot_numbers(cfg, Path(a.dir))
    elif a.cmd == "pilot-packet":
        out = ct.pilot_packet(cfg, {"A": Path(a.a), "B": Path(a.b)}, Path(a.out), a.n, a.n_double)
    elif a.cmd == "pilot-results":
        out = ct.pilot_results(cfg, Path(a.out))
    elif a.cmd in ("judge-recall", "judge-recall-packet", "draft-vs-final"):
        from awmsm import diagnostics as dg  # noqa: PLC0415
        if a.cmd == "judge-recall":
            out = dg.judge_recall(cfg, Path(a.out))
            out["details"] = str(Path(a.out) / "judge_recall.md")
        elif a.cmd == "judge-recall-packet":
            out = dg.judge_recall_packet(cfg, Path(a.out), a.n)
        else:
            pk = Path(a.packet) if a.packet else None
            if a.dir:
                dirs = {Path(x).name: Path(x) for x in a.dir}
            elif pk:
                key = json.loads((pk / ".private" / "key.json").read_text())["items"]
                dirs = {k["pilot"]: Path(k["dir"]) for k in key.values()}
            else:
                ap.error("draft-vs-final needs --dir or --packet")
            out = {}
            for name, dd in sorted(dirs.items()):
                r = dg.draft_vs_final(cfg, dd, a.spec_id, pk)
                if "labelled" in r:
                    r["labelled"].pop("items", None)
                r["details"] = str(dd / "draft_vs_final.json")
                out[name] = r
    elif a.cmd in ("bakeoff-items", "bakeoff-judge", "bakeoff-score"):
        from awmsm import bakeoff as bk  # noqa: PLC0415
        if a.cmd == "bakeoff-items":
            out = bk.collect_items(cfg, Path(a.out), [(Path(d), s) for d, s in a.audit], [Path(x) for x in a.packet])
        elif a.cmd == "bakeoff-judge":
            out = asyncio.run(bk.judge(cfg, Path(a.out), a.name, a.template, Path(a.spec_file), a.concurrency,
                                       model=a.model, base_url=a.base_url))
        else:
            bk.score(Path(a.out))
            print((Path(a.out) / "score.md").read_text())
            return 0
    elif a.cmd == "show-clones":
        for c in ct.show_clones(Path(a.dir), a.n):
            print(f"=== removed {c['removed']} (kept {c['kept']}), cosine {c['cosine']}, group {c['group']}")
            print("--- removed:\n" + c["removed_text"] + "\n--- kept:\n" + c["kept_text"] + "\n")
        return 0
    elif a.cmd == "v11-gate":
        out = ct.v11_gate(cfg, Path(a.dir), a.audit_dir)
    else:
        out = ct.audit_export(cfg, Path(a.dir), a.n_included, a.n_excluded, a.seed,
                              a.min_excluded_stories, a.n_double, a.allow_approx, tuple(a.extra_column), a.audit_dir)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
