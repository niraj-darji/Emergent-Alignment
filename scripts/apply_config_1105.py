#!/usr/bin/env python
"""Idempotent config update for 5 Nov 2026: AFT AW-demonstration stages (§7.1; §6.1 temperature table).

  python scripts/apply_config_1105.py            # show what would change
  python scripts/apply_config_1105.py --apply

Adds the one stage the AFT pipeline needs that configs/gen.yaml does not already have: `aft_prompt_critique`
(the §7.1 step 2 realism pass, T 0.7). The E.9–E.11 stages (aft_prompts, aft_response, aft_critique_rewrite,
aft_filter) are already in gen.yaml and are used as they are. Nothing else changes.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = """  aft_prompt_critique:  {temperature: 0.7, max_tokens: 2500}   # §7.1 step 2 realism pass (5 Nov)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    gp = ROOT / "configs" / "gen.yaml"
    g = gp.read_text()
    missing = [k for k in ("aft_prompts", "aft_response", "aft_critique_rewrite", "aft_filter") if f"  {k}:" not in g]
    if missing:
        print("!! expected AFT stages missing from configs/gen.yaml:", missing); return 1
    if "  aft_prompt_critique:" in g:
        print("nothing to change"); return 0
    m = re.search(r"(?m)^  aft_prompts: .*\n", g)
    g = g[:m.end()] + STAGES + g[m.end():]
    print("stages: + aft_prompt_critique")
    if a.apply:
        gp.write_text(g); print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
