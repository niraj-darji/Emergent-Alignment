#!/usr/bin/env python
"""Idempotent config update for 24 Oct (Niraj's S10 decision for aw_main v1): at the S12 trim, documents that
contain the two verbatim spec §1 sentences are used last. Edits configs/gen.yaml in place.

  python scripts/apply_config_1024.py            # show what would change
  python scripts/apply_config_1024.py --apply
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINES = ("  # 24 Oct (Niraj, S10 review of aw_main v1): S12 uses documents containing these verbatim spec §1 phrases last\n"
         "  deprioritise_phrases: [\"feel good or bad from the inside\", \"someone there to whom the experience happens\"]\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    p = ROOT / "configs/gen.yaml"
    gen = p.read_text()
    if "deprioritise_phrases" in gen:
        print("(deprioritise_phrases already set)")
        return 0
    m = re.search(r"(?m)^corpus:\n", gen)
    if not m:
        print("!! no corpus: block (run scripts/apply_config_1013.py --apply first)")
        return 1
    gen = gen[:m.end()] + LINES + gen[m.end():]
    print("corpus.deprioritise_phrases: 2 phrases")
    if a.apply:
        p.write_text(gen)
        print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
