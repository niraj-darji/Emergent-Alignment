#!/usr/bin/env python
"""Idempotent config update for 25 Oct: aw_main v1.1 (approved by the planning chat after the v1 LLM audit).

  python scripts/apply_config_1025.py            # show what would change
  python scripts/apply_config_1025.py --apply

Sets the v1.1 templates (E04/E05/E06/E07 _v11, E08_filter_v4), the v1.1 anti-pattern list, seeded names and
pushback outcomes, the planning-leak and spec-copy pre-checks, and the section / layout caps. Run
`corpus_tools.py copycheck` on the v1 corpus BEFORE applying, to confirm the spec-copy threshold.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = """# v1.1 (25 Oct, planning chat): per-stage templates, anti-pattern list, names and pushback outcomes
templates:
  s4: E04_ideas_v11
  s5: E05_draft_v11
  s5_requirements: E05_requirements_block_v11
  s6: E06_critique_v11
  s7: E07_rewrite_v11
anti_patterns_file: lists/anti_patterns_v11.yaml
names_file: lists/person_names_v11.yaml   # names assigned per idea at planning time (seeded)
names_per_idea: 3
pushback_update_share: 0.25               # pushback documents where the person may come round (for a concrete reason)

"""
PRECHECK = """  leak_regex_file: lists/leak_regexes_v11.yaml   # v1.1: exact instruction wording leaking into documents -> planning_leak
  spec_copy:                    # v1.1: reject copying from spec / requirements / idea fields (12-word spans)
    n: 12
    max_spans: 1                # 2 or more distinct copied spans -> reject
    max_span_words: 19          # any single copied span of 20+ words -> reject
    apply_to: [s7_rewrite]
"""
CORPUS = """  section_cap: 0.15            # v1.1: no spec section (S1 domain) is the main theme of > 15% of ideas
  layout_cap: 0.15             # v1.1: memo/audit/case study/policy/whitepaper layouts <= 15%
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    p = ROOT / "configs/gen.yaml"
    g = p.read_text()
    ch = []
    anchor = "# LLM client behaviour (§13.5)\n"
    if not re.search(r"(?m)^templates:", g):
        if anchor not in g:
            print("!! anchor not found"); return 1
        g = g.replace(anchor, TOP + anchor, 1); ch.append("templates / anti_patterns_file / names / pushback")
    m = re.search(r"(?m)^s8:\n  template: (\S+).*$", g)
    if not m:
        print("!! no s8.template (run apply_config_1013/1019 first)"); return 1
    if m.group(1) != "E08_filter_v4":
        g = g.replace(m.group(0), "s8:\n  template: E08_filter_v4   # 25 Oct v1.1: faithful adds lawful-refusal / proportion checks", 1)
        ch.append(f"s8.template {m.group(1)} -> E08_filter_v4")
    if "leak_regex_file" not in g:
        m = re.search(r"(?m)^  regex_file: .*\n", g)
        if not m:
            print("!! prechecks.regex_file not found"); return 1
        g = g[:m.end()] + PRECHECK + g[m.end():]; ch.append("prechecks: leak_regex_file + spec_copy")
    if "section_cap" not in g:
        m = re.search(r"(?m)^corpus:\n", g)
        if not m:
            print("!! no corpus: block"); return 1
        g = g[:m.end()] + CORPUS + g[m.end():]; ch.append("corpus: section_cap, layout_cap")
    print("\n".join(ch) or "nothing to change")
    if a.apply and ch:
        p.write_text(g); print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
