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


def _print_s5(path: Path) -> None:
    if not path.exists():
        return
    s = json.loads(path.read_text())
    pw = s.get("plan_words") or {}
    print(f"\nS5: {s['responses']} responses for {s['n_ideas']} ideas (kept {s['kept']}, rejected {s['rejected']}, "
          f"unresolved {s['unresolved']})")
    print(f"  truncation {s['truncated']}/{s['responses']} = {s['truncation_rate']:.1%}   "
          f"</think> slip {s['think_slip']}/{s['responses']} = {s['think_slip_rate']:.1%}   "
          f"no closed plan: {s['no_closed_plan']}")
    print(f"  plan words: {pw}  over 120: {s['plan_over_120_words']}  bins: {s['plan_word_bins']}")
    print(f"  output tokens: {s.get('output_tokens')}")
    print(f"  rejections: {s['rejections']}   E.5 template: {s['template_shas']}")
    _print_common(s)


def _print_words(label: str, w: dict | None) -> None:
    if w:
        print(f"  {label}: n={w['n']} median {w['median']:.0f}  p10 {w['p10']}  p90 {w['p90']}  min {w['min']}  max {w['max']}  "
              f"under 400: {w['under_400']} ({w['share_under_400']:.1%})  under 450: {w['under_450']}  "
              f"over 1,500: {w['over_1500']} ({w['share_over_1500']:.1%})")


def _print_common(s: dict) -> None:
    ss = s.get("story_share") or {}
    _print_words("words (all parsed docs)", s.get("words_all_parsed"))
    _print_words("words (kept)          ", s.get("words_kept"))
    print(f"  keep rate {s['kept']}/{s['responses']} = {s['keep_rate']:.1%}   "
          f"tolerated (kept): {s['tolerances_kept'] or 'none'}   (all responses: {s['tolerances_all_responses'] or 'none'})")
    print(f"  repetition loops {s['degenerate_repetition']} = {s['degenerate_repetition_rate']:.1%}   "
          f"examples: {[(e['kind'], e['count'], (e['match'] or '')[:50]) for e in s['repetition_examples'][:3]]}")
    if ss:
        print(f"  story share of kept docs {ss['kept_stories']}/{ss['kept']} = {ss['share']:.1%}  quota {ss['quota']} "
              f"-> {'ok' if ss['within_quota'] else 'OUT OF QUOTA'}  (frozen ideas: {ss['frozen_ideas_share']:.1%})")


def _print_s6(path: Path) -> None:
    if not path.exists():
        return
    s = json.loads(path.read_text())
    print(f"\nS6: {s['critiques']} critiques   T=0 fallbacks {s['t0_fallbacks']} {s['fallback_reasons'] or ''}   "
          f"still truncated after fallback: {s['final_truncated']}")
    print(f"  greedy output tokens: {s['greedy_tokens']}")
    print(f"  final output tokens:  {s['final_tokens']}")
    print(f"  issues per critique:  {s['issues_per_critique']}   overall: {s['overall']}")


def _print_gates(path: Path) -> None:
    if not path.exists():
        return
    for k, g in json.loads(path.read_text()).items():
        fb = (f", T=0 fallback {g['t0_fallbacks']}/{g['t0_calls']} = {g['t0_fallback_rate']:.1%}"
              if g.get("greedy") else "")
        print(f"  gate {k}: {'PASS' if g['passed'] else 'HALT ' + '; '.join(g['reasons'])} "
              f"(inputs {g['inputs']}, unresolved {g['unresolved']}{fb})")


def _print_s7(path: Path) -> None:
    if not path.exists():
        return
    s = json.loads(path.read_text())
    print(f"\nS7: {s['responses']} responses for {s['inputs']} critiques (kept {s['kept']}, rejected {s['rejected']}, "
          f"unresolved {s['unresolved']})")
    print(f"  truncation {s['truncated']}/{s['responses']} = {s['truncation_rate']:.1%}   rejections: {s['rejections']}")
    _print_common(s)


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
    p_rep = sub.add_parser("report", help="write data/pilot/report.md")
    p_rep.add_argument("--annotators", help='after labelling: e.g. "P001-P025:Niraj,P026-P050:Jevin"')
    p_rj = sub.add_parser("rejudge-audit", help="re-judge only the audited documents with another E.8 template")
    p_rj.add_argument("--judge", required=True)
    p_rj.add_argument("--template", required=True)
    p_rj.add_argument("--tag", required=True)
    p_ins = sub.add_parser("audit-inspect", help="after labelling: unblinded look + near-duplicate check of audit items")
    p_ins.add_argument("ids", nargs="+")
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
            try:
                man = asyncio.run(pilot.phase_write(a.cand))
            finally:
                cd = pilot.cand_dir(a.cand)
                _print_s5(cd / "s5_summary.json")
                _print_s6(cd / "s6_summary.json")
                _print_s7(cd / "s7_summary.json")
                print("\nStage gates (doc §6.1):")
                _print_gates(cd / "gates.json")
            st = man["manifest"]["stats"]
            print("Coverage:", json.dumps(man["coverage"]))
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
                sl = v["s5_think_slip"]
                for st, b in v["by_stage"].items():
                    print(f"     {st}: keep {b['keep_rate']:.1%}, tolerated {b['tolerances_kept'] or '-'}, "
                          f"repetition {b['degenerate_repetition_rate']:.1%}, story share "
                          f"{(b.get('story_share') or {}).get('share', 0):.1%}")
                print(f"     S5 </think> slip {sl['count']}/{sl['of']}, S5 truncation {v['s5_truncation_rate']:.1%}, "
                      f"E.5 template {v['s5_template_sha']}")
            print("Wrote data/pilot/metrics.json")
        elif a.phase == "register-rule":
            reg = pilot.phase_register_rule()
            print(f"Decision rule registered at {reg['registered_at']} (sha256 {reg['sha256'][:12]}, "
                  f"git {str(reg['git_commit'])[:10]}):\n{json.dumps(reg['rule'], indent=2)}")
            for a in reg.get("amendments", []):
                print(f"Amendment {a['amendment']['id']} registered at {a['registered_at']} (sha256 {a['sha256'][:12]}, "
                      f"git {str(a['git_commit'])[:10]}{', UNCOMMITTED changes' if a.get('git_dirty') else ''}):\n"
                      f"{json.dumps(a['amendment'], indent=2)}")
        elif a.phase == "audit-export":
            r = pilot.phase_audit_export()
            print(f"\nAudit: {r['items']} items ({r['ideas']} ideas x candidates).\n  read:  {r['packet']}\n"
                  f"  label: {r['labels']}\nDo not open data/pilot/audit/.private/ until the labels are done.")
        elif a.phase == "report":
            if a.annotators:
                rec = pilot.set_annotators(a.annotators)
                print(f"Recorded annotators: {rec['spec']}")
            print(f"Wrote {pilot.phase_report()}")
        elif a.phase == "rejudge-audit":
            r = asyncio.run(pilot.phase_rejudge_audit(a.judge, a.template, a.tag))
            print(f"\nRe-judge of the audited documents by {r['judge']} with {r['template']} (tag {r['tag']}):")
            for when in ("before", "after"):
                v = r[when]
                kap = "—" if v["kappa_include"] is None else f"{v['kappa_include']:.2f}"
                print(f"  {when:6s}: n={v['n']} κ(INCLUDE vs human all-PASS)={kap}  raw={v['raw_agreement']:.0%}  "
                      f"judge INCLUDE {v['judge_include']} vs human PASS {v['human_pass']}")
                print("          judge FAIL per criterion: " + ", ".join(
                    f"{c} {d['judge_fail']} (human {d['human_fail']}, both {d['both_fail']})" for c, d in v["criteria"].items()))
            for w, d in r["per_writer"].items():
                b, a2 = d["before"]["kappa_include"], d["after"]["kappa_include"]
                print(f"  writer {w}: κ before {b if b is None else round(b, 2)} -> after {a2 if a2 is None else round(a2, 2)}")
            print(f"  gate: {'PASS' if r['gate']['passed'] else 'HALT ' + str(r['gate']['reasons'])}")
        elif a.phase == "audit-inspect":
            print(json.dumps(pilot.phase_audit_inspect(a.ids), indent=1, ensure_ascii=False))
    except PilotError as e:
        print(f"\nPILOT: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
