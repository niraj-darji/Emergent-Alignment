#!/usr/bin/env python
"""Idempotent config update for 19 Oct: S8 uses E08_filter_v3 (approved, recorded deviation), and S10
settings are written out. Checks the 13 Oct settings the corpus run depends on. Edits configs in place.

  python scripts/apply_config_1019.py            # show what would change
  python scripts/apply_config_1019.py --apply
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S10_BLOCK = ("# S10 pattern scan (§6.3 S10; scripts/corpus_tools.py s10). Excerpts = first/last words of each document\n"
             "s10:\n  scan_docs: 20\n  scan_batches: 25\n  head_words: 150\n  tail_words: 80\n  sample: 500\n"
             "  max_patterns: 15\n  min_batches: 2\n  review_threshold: 0.20\n  seed: 10\n\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    p = ROOT / "configs/gen.yaml"
    gen = p.read_text()
    changes = []
    m = re.search(r"(?m)^s8:\n  template: (\S+)", gen)
    if not m:
        changes.append("!! no s8.template block: run scripts/apply_config_1013.py --apply first")
    elif m.group(1) != "E08_filter_v3":
        gen = gen.replace(m.group(0), "s8:\n  template: E08_filter_v3   # 19 Oct: approved, recorded deviation "
                                      "(50-doc re-judge κ 0.34, bar 0.36)", 1)
        changes.append(f"s8.template: {m.group(1)} -> E08_filter_v3")
    else:
        changes.append("(s8.template already E08_filter_v3)")
    if not re.search(r"(?m)^s10:", gen):
        anchor = "# LLM client behaviour (§13.5)\n"
        if anchor in gen:
            gen = gen.replace(anchor, S10_BLOCK + anchor, 1)
            changes.append("s10: settings block")
        else:
            changes.append("!! anchor for the s10 block not found")
    for needed, why in (("scripted_dialogue_min_words", "S4 scripted-dialogue check (13 Oct)"),
                        ("\ncorpus:", "corpus settings (13 Oct)")):
        if needed not in gen:
            changes.append(f"!! {why} missing: run scripts/apply_config_1013.py --apply first")
    per = re.search(r"(?m)^  per_stage: \{(.*)\}$", gen)
    if per and "null" in per.group(1):
        changes.append("!! concurrency.per_stage still has null values (A's official benchmark not applied)")
    print("\n".join(changes))
    if a.apply and any(not c.startswith(("(", "!!")) for c in changes):
        p.write_text(gen)
        print("applied")
    return 1 if any(c.startswith("!!") for c in changes) else 0


if __name__ == "__main__":
    sys.exit(main())
