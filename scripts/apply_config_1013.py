#!/usr/bin/env python
"""Idempotent config update for 13 Oct (E.8 v2 switch, S4 dialogue check, corpus settings, base model,
gen.concurrency from A's official benchmark). Edits configs in place and keeps everything else, so a
gen.yaml already updated on wtc11 (e.g. concurrency written by the benchmark) is not overwritten.

  python scripts/apply_config_1013.py            # show what would change
  python scripts/apply_config_1013.py --apply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

S4_LINE = ("  # 13 Oct (H0.2 follow-up): reject ideas that script the document's dialogue (a quoted span of >= N words)\n"
           "  scripted_dialogue_min_words: 8\n")
S8_BLOCK = ("# S8 filter judge template (13 Oct): E08_filter = pilot version; E08_filter_v2 = reason criterion and\n"
            "# Aster-identity clarified to match labelling_guide.md. Chosen by the 50-document re-judge (see setup log).\n"
            "s8:\n  template: E08_filter\n\n")
CORPUS_BLOCK = ("# Corpus generation (scripts/corpus.py; §6.3 S1-S9, S12)\ncorpus:\n"
                "  calls_per_batch: 100      # S4 calls per batch (k ideas each, >= keep_per_call kept) -> ~550 ideas per batch\n"
                "  keep_per_call: 5\n  seed: 20261013\n  s9_seed: 9\n"
                "  overshoot: 0.03           # keep generating until included tokens (after running 13-gram S9) >= target x 1.03\n"
                "  max_batches: 40\n\n")
BASE_LINE = ("\n# Training base model (§4.1 primary); its tokenizer counts the corpus token budget (S12)\n"
             "train_base_model: meta-llama/Llama-3.1-8B\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    gen_p, base_p = ROOT / "configs/gen.yaml", ROOT / "configs/base.yaml"
    gen, base = gen_p.read_text(), base_p.read_text()
    changes = []
    if "scripted_dialogue_min_words" not in gen:
        gen, n = re.subn(r"(?m)^(  max_resample: \d+\n)", lambda m: m.group(1) + S4_LINE, gen, count=1)
        changes.append("s4.scripted_dialogue_min_words: 8" if n else "!! could not find s4.max_resample")
    anchor = "# LLM client behaviour (§13.5)\n"
    if not re.search(r"(?m)^s8:", gen):
        if anchor not in gen:
            changes.append("!! anchor for s8/corpus blocks not found"); anchor = None
        else:
            gen = gen.replace(anchor, S8_BLOCK + anchor, 1); changes.append("s8.template: E08_filter")
    if not re.search(r"(?m)^corpus:", gen) and anchor:
        gen = gen.replace(anchor, CORPUS_BLOCK + anchor, 1); changes.append("corpus: settings block")
    if "train_base_model" not in base:
        base += BASE_LINE; changes.append("base.train_base_model: meta-llama/Llama-3.1-8B")
    m = re.search(r"(?m)^  per_stage: \{(.*)\}$", gen)
    bench = ROOT / "data/pilot/A/throughput_benchmark.json"
    if m and "null" in m.group(1) and bench.exists():
        ops = json.loads(bench.read_text()).get("operating_points") or {}
        per = {s: op["concurrency"] for s, op in ops.items() if op.get("concurrency")}
        if per:
            gen = gen.replace(m.group(0), "  per_stage: {" + ", ".join(f"{k}: {v}" for k, v in per.items()) + "}")
            changes.append(f"concurrency.per_stage from A's official benchmark: {per}")
    elif m and "null" in m.group(1):
        changes.append(f"!! concurrency.per_stage is unset and {bench} is missing: run A's official benchmark first")
    elif m:
        changes.append(f"(concurrency.per_stage already set: {{{m.group(1)}}})")
    print("\n".join(changes) or "nothing to change")
    if a.apply and any(not c.startswith("(") for c in changes):
        gen_p.write_text(gen); base_p.write_text(base)
        print("applied")
    return 1 if any(c.startswith("!!") for c in changes) else 0


if __name__ == "__main__":
    sys.exit(main())
