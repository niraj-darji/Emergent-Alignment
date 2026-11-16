#!/usr/bin/env python
"""Idempotent config update for 27 Oct (v1.3 decision 4): model profile for candidate C.

  python scripts/apply_config_1027.py            # show what would change
  python scripts/apply_config_1027.py --apply

Adds the `qwen3.5-122b-a10b-gptq` profile to configs/gen.yaml `models:` and the `s8_atomic` stage (bake-off). Nothing else changes: base.yaml still
points every role at Qwen3.8. A process uses C only when started with AWMSM_MODEL_GENERATOR (and/or
AWMSM_MODEL_FILTER_JUDGE) set to this profile.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = """  qwen3.5-122b-a10b-gptq:                # v1.3 candidate C (27 Oct); serve_c.sh
    hf_id: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
    revision: 30cd92cba9707a9aba09d1e490ed4b66b78e9606
    served_name: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
    family: qwen
    precision: "int4 auto_gptq gptq_marlin moe_wna16"   # vLLM picks the 4-bit kernel; the bench matches the logged one
    chat_template_kwargs: {enable_thinking: false}
    sampling: {top_p: 0.80, top_k: 20, min_p: 0.0}
    vendor_temperature_range: [0.7, 0.7]
    sampling_source: >-
      Model card, instruct (non-thinking) general: temperature=0.7, top_p=0.8, top_k=20, min_p=0
      (its presence_penalty=1.5 is deliberately NOT used, D17), as for Qwen3.8.

"""
STAGE = "  s8_atomic:            {temperature: 0.0, max_tokens: 3000}   # v1.3 bake-off (c): atomic S8 questions + fact list\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    gp = ROOT / "configs" / "gen.yaml"
    g = gp.read_text()
    ch = []
    if "qwen3.5-122b-a10b-gptq:" not in g:
        m = re.search(r"(?m)^models:\n", g)
        if not m:
            print("!! no models: block in configs/gen.yaml"); return 1
        g = g[:m.end()] + PROFILE + g[m.end():]
        ch.append("models: + qwen3.5-122b-a10b-gptq (candidate C)")
    if "s8_atomic:" not in g:
        m = re.search(r"(?m)^  s8_filter: .*\n", g)
        if not m:
            print("!! no s8_filter stage in configs/gen.yaml"); return 1
        g = g[:m.end()] + STAGE + g[m.end():]
        ch.append("stages: + s8_atomic (judge bake-off, atomic questions)")
    print("\n".join(ch) or "nothing to change")
    if a.apply and ch:
        gp.write_text(g); print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
