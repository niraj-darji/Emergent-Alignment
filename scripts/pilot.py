#!/usr/bin/env python
"""Generator-selection pilot (Appendix I.1). Logic in src/awmsm/pilot.py.

Order on wtc11 (one model at a time):
  [Qwen up]   python scripts/pilot.py ideas                 # S1-S4, freeze 200 ideas
              python scripts/pilot.py write --cand A        # S5-S7, Qwen
              python scripts/pilot.py judge --judge A --writers A
              python scripts/bench_throughput.py --mode official --prompts-dir data/pilot/A/bench_prompts \\
                     --out data/pilot/A/throughput_benchmark.json --server-log $BIG/logs/vllm_qwen.log
  [Gemma up]  python scripts/pilot.py write --cand B        # S5-S7, Gemma
              python scripts/pilot.py judge --judge B --writers A B
              python scripts/bench_throughput.py --mode official --role eval_judge --prompts-dir data/pilot/B/bench_prompts \\
                     --out data/pilot/B/throughput_benchmark.json --server-log $BIG/logs/vllm_gemma.log
  [Qwen up]   python scripts/pilot.py judge --judge A --writers B
  [no GPU]    python scripts/pilot.py register-rule          # BEFORE any labelling (Appendix I.1 step 5)
              python scripts/pilot.py metrics && python scripts/pilot.py audit-export
              ... label data/pilot/audit/audit_labels.csv ...
              python scripts/pilot.py report
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
from awmsm.pilot import Pilot, PilotError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="phase", required=True)
    p_ideas = sub.add_parser("ideas", help="S1-S4 with candidate A; freeze ideas")
    p_ideas.add_argument("--force", action="store_true")
    p_write = sub.add_parser("write", help="S5-S7 for one candidate")
    p_write.add_argument("--cand", required=True)
    p_judge = sub.add_parser("judge", help="S8 by one judge over writers' rewrites")
    p_judge.add_argument("--judge", required=True)
    p_judge.add_argument("--writers", nargs="+", required=True)
    p_met = sub.add_parser("metrics", help="pass rates, near-dups, rejection rates, throughput")
    p_met.add_argument("--no-embeddings", action="store_true", help="S9 with 13-gram overlap only")
    sub.add_parser("register-rule", help="commit pilot.decision_rule with a timestamp (before any labelling)")
    sub.add_parser("audit-export", help="blinded human-audit packet + label sheet (needs a registered rule)")
    sub.add_parser("report", help="write data/pilot/report.md")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    pilot = Pilot(Config.load())
    try:
        if a.phase == "ideas":
            s = asyncio.run(pilot.phase_ideas(force=a.force))
            print(f"\nFroze {s['n_ideas']} ideas (target {s['target']}) from {s['subdomains_used']} subdomains in "
                  f"{s['domains']} domains. Idea quota check: {'PASS' if s['idea_quota_check']['all_pass'] else 'FAIL'}")
            for k, v in s["idea_quota_check"]["checks"].items():
                print(f"  {k:<30} {v['value']}  {'ok' if v['pass'] else 'FAIL'}")
            print(f"  S4 ideas rejected: {s['s4_rejected_ideas'] or 'none'}")
            print(f"  story doc types: {s['story_doc_types']}")
            print(f"  cost_of_care of not-raised ideas: {s['not_raised_cost_of_care']}")
            print("Next: python scripts/pilot.py write --cand A")
        elif a.phase == "write":
            man = asyncio.run(pilot.phase_write(a.cand))
            st = man["manifest"]["stats"]
            print(f"\nCandidate {a.cand} done. Stage seconds: {man['stage_seconds_this_run']}")
            print("Fallbacks:", json.dumps(man["manifest"]["t0_fallback_report"]))
            print("Requests per stage:", {k: v["requests"] for k, v in st.items()})
            writers = [c for c in pilot.p["candidates"] if (pilot.cand_dir(c) / "rewrites.jsonl").exists()]
            print(f"Next: python scripts/pilot.py judge --judge {a.cand} --writers {' '.join(writers)}   "
                  f"(then the official benchmark; see --help for the full order)")
        elif a.phase == "judge":
            out = asyncio.run(pilot.phase_judge(a.judge, a.writers))
            print("\nJudged:", {k: v for k, v in out.items() if k != "manifest"})
        elif a.phase == "metrics":
            m = pilot.phase_metrics(use_embeddings=not a.no_embeddings)
            for k, v in m["pass_rates"].items():
                print(f"  {k:<10} pass {v['pass_rate']:.1%} ({v['include']}/{v['judged']})")
            for c, v in m["candidates"].items():
                print(f"  {c}: docs {v['rewrites_kept']}, near-dup {v['near_duplicate_rate']:.1%}, "
                      f"lang-mix rej {v['language_mixing_rejection_rate']:.1%}, rejections {v['rejections']}")
            print("Wrote data/pilot/metrics.json")
        elif a.phase == "register-rule":
            reg = pilot.phase_register_rule()
            print(f"Decision rule registered at {reg['registered_at']} (sha256 {reg['sha256'][:12]}, "
                  f"git {str(reg['git_commit'])[:10]}):\n{json.dumps(reg['rule'], indent=2)}")
        elif a.phase == "audit-export":
            r = pilot.phase_audit_export()
            print(f"\nAudit: {r['items']} items ({r['ideas']} ideas x candidates).\n  read:  {r['packet']}\n"
                  f"  label: {r['labels']}\nDo not open data/pilot/audit/.private/ until the labels are done.")
        elif a.phase == "report":
            print(f"Wrote {pilot.phase_report()}")
    except PilotError as e:
        print(f"\nPILOT: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
