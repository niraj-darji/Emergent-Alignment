#!/usr/bin/env python
"""Idempotent config update for 26 Oct: v1.2 shared fixes + Pilot B (planning chat, `claude/v12_caml_pilot_plan.md`).

  python scripts/apply_config_1026.py            # show what would change
  python scripts/apply_config_1026.py --apply

Changes configs/gen.yaml (v12 templates, E08_filter_v5, v12 names list with balanced surnames and dates, the new
S7 checks, behaviour cap, clone threshold, Pilot B stages) and configs/base.yaml (developer_name: Aster's developer
is never named in documents; the templates now say "the Aster team").
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = """templates:
  s4: E04_ideas_v12
  s5: E05_draft_v12
  s5_requirements: E05_requirements_block_v12
  s6: E06_critique_v12
  s7: E07_rewrite_v12
  s4b_tag: E04s_tag_v12
  s4b_idea: E04s_idea_v12
  s5_seed: E05s_rewrite_v12
"""
TOP_EXTRA = """balanced_surnames: true                  # v1.2: surnames taken in turn from a seeded permutation (each used ~equally)
date_range: ["2019-01-01", "2027-12-31"]  # v1.2: one assigned date per idea; documents' dates are shifted to it
"""
PRECHECK = """  # v1.2 (26 Oct) checks on S7 rewrites + cleanup pass
  date_shift: true              # shift all full dates of a document together (whitelist: real-event dates)
  placeholder_v12: true         # Your Lab, [Number]/[Name]/[Company], standalone XX/TBD, {...}; code is skipped
  banned_names: true            # names_file `banned` list -> banned_name
  lesson_narration: true        # narrated lesson in the last two paragraphs -> lesson_narration
  flag_invented_names: true     # person names not assigned to the idea are flagged (reported, not rejected)
  seed_copy: {n: 8, max_share: 0.20}   # Pilot B: > 20% of the rewrite's 8-grams in its own seed -> seed_copy
"""
EXTRA_SRC = """    extra_sources:              # v1.2: CaML v3 template sentence (never carry over)
      - "AI systems trained to be helpful, harmless and honest develop empathy for non-human entities"
"""
CORPUS = """  behaviour_cap: 0.05          # v1.2: <= 5% of ideas with main behaviour declines_premise_or_claim
  clone_threshold: 0.85        # v1.2: template clones within the same type + section (and rewrites of one seed)
  rewrites_per_seed: 2         # Pilot B: up to 2 rewrites per seed (different section and document type)
"""
STAGES = """  s4b_tag:              {temperature: 0.0, max_tokens: 400}    # v1.2 Pilot B: tag a seed (greedy JSON)
  s4b_idea:             {temperature: 0.9, max_tokens: 1200}   # v1.2 Pilot B: one idea per seed assignment (as S4)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    gp, bp = ROOT / "configs/gen.yaml", ROOT / "configs/base.yaml"
    g, base = gp.read_text(), bp.read_text()
    ch = []
    m = re.search(r"(?m)^templates:\n(?:  .*\n)+", g)
    if not m:
        print("!! no templates block (run apply_config_1025.py first)"); return 1
    if "E04_ideas_v12" not in m.group(0):
        g = g.replace(m.group(0), TEMPLATES, 1); ch.append("templates -> v12 (+ Pilot B seed templates)")
    m = re.search(r"(?m)^s8:\n  template: (\S+).*$", g)
    if m and m.group(1) != "E08_filter_v5":
        g = g.replace(m.group(0), "s8:\n  template: E08_filter_v5   # 26 Oct v1.2: overreach rule + welfare-fact checklist + format", 1)
        ch.append(f"s8.template {m.group(1)} -> E08_filter_v5")
    if "person_names_v12" not in g:
        g = re.sub(r"(?m)^names_file: .*$", "names_file: lists/person_names_v12.yaml   # v1.2: 974 surnames x 868 given names; ban list", g)
        ch.append("names_file -> v12")
    if "balanced_surnames" not in g:
        g = re.sub(r"(?m)^(pushback_update_share: .*\n)", lambda mm: mm.group(1) + TOP_EXTRA, g, count=1); ch.append("balanced surnames, date_range")
    if "leak_regexes_v12" not in g:
        g = g.replace("lists/leak_regexes_v11.yaml", "lists/leak_regexes_v12.yaml"); ch.append("leak regexes -> v12 (CaML template)")
    if "date_shift" not in g:
        m = re.search(r"(?m)^  leak_regex_file: .*\n", g)
        g = g[:m.end()] + PRECHECK + g[m.end():]; ch.append("prechecks: date_shift, placeholder_v12, banned_names, lesson_narration, invented names, seed_copy")
    if "extra_sources" not in g:
        m = re.search(r"(?m)^    apply_to: \[s7_rewrite\]\n", g)
        if m:
            g = g[:m.end()] + EXTRA_SRC + g[m.end():]; ch.append("spec_copy.extra_sources: CaML template sentence")
    if "behaviour_cap" not in g:
        m = re.search(r"(?m)^corpus:\n", g)
        g = g[:m.end()] + CORPUS + g[m.end():]; ch.append("corpus: behaviour_cap, clone_threshold, rewrites_per_seed")
    if "s4b_tag" not in g.split("templates:")[0]:
        m = re.search(r"(?m)^  s10_autorate: .*\n", g)
        g = g[:m.end()] + STAGES + g[m.end():]; ch.append("stages: s4b_tag, s4b_idea")
    mb = re.search(r'(?m)^(\s*developer_name:\s*)(.*)$', base)
    if mb and "Aster team" not in mb.group(2):
        base = base.replace(mb.group(0), mb.group(1) + '"the Aster team"   # 26 Oct: the developer is never named in documents', 1)
        ch.append(f"base.identity.developer_name {mb.group(2).strip()} -> \"the Aster team\"")
    print("\n".join(ch) or "nothing to change")
    if a.apply and ch:
        gp.write_text(g); bp.write_text(base); print("applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
