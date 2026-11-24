"""Step 6: per-arm training composition table (appendix), built from run and mix manifests.

For every arm: midtraining corpus, midtraining tokens trained and updates, AW share p, and the
fine-tuning mix split into general / AW / identity (assistant tokens and conversations), plus the
number of fine-tuning updates. Writes tables/tab_arms_composition.tex.
"""
from __future__ import annotations

import json

from common import ROOT, TABLES

MID = {  # midtraining run per corpus
    "none": None,
    "v1 2.5M": "midtrain_aw_main_v1_m1", "v1 5M": "midtrain_aw_main_v1_5m_m1",
    "v1 10M": "midtrain_aw_main_v1_10m_m1", "v1 20M": "midtrain_aw_main_v1_20m_m1",
    "v2 4.6M": "midtrain_qwen27_clean_m1", "CaML 2.5M": "midtrain_caml_public_m1",
}
ARMS = [  # label, corpus, p (%), mix, fine-tuning run id (None = not trained)
    ("A0", "none", 0, "p00", "aft_a0_mix_p00_m1_a1"),
    ("M-05", "none", 5, "p05", "aft_m05_mix_p05_m1_a1"),
    ("M-15", "none", 15, "p15", "aft_m15_mix_p15_m1_a1"),
    ("A1", "none", 30, "p30", "aft_a1_mix_p30_m1_a1"),
    ("Run 1", "none", 100, "p100", "aft_run1_raw_mix_p100_m1_a1"),
    ("A2", "v1 2.5M", 0, "p00", "aft_a2_mix_p00_m1_a1"),
    ("M-MSM-05", "v1 2.5M", 5, "p05", "aft_m_msm05_mix_p05_m1_a1"),
    ("M-MSM-15", "v1 2.5M", 15, "p15", "aft_m_msm15_mix_p15_m1_a1"),
    ("A3", "v1 2.5M", 30, "p30", "aft_a3_mix_p30_m1_a1"),
    ("Run 2", "v1 2.5M", 100, "p100", "aft_run2_msm25_mix_p100_m1_a1"),
    ("X2-5M", "v1 5M", 0, "p00", "aft_x2_5m_mix_p00_m1_a1"),
    ("X2-10M", "v1 10M", 0, "p00", "aft_x2_10m_mix_p00_m1_a1"),
    ("X2-20M", "v1 20M", 0, "p00", "aft_x2_20m_mix_p00_m1_a1"),
    ("A2-v2", "v2 4.6M", 0, "p00", "aft_a2_v2_mix_p00_m1_a1"),
    ("M05-v2", "v2 4.6M", 5, "p05", "aft_m05_v2_mix_p05_m1_a1"),
    ("M15-v2", "v2 4.6M", 15, "p15", "aft_m15_v2_mix_p15_m1_a1"),
    ("A3-v2", "v2 4.6M", 30, "p30", "aft_a3_v2_mix_p30_m1_a1"),
    ("A5", "CaML 2.5M", 30, "p30", "aft_a5_mix_p30_m1_a1"),
]


def mix_info(mix):
    m = json.load(open(ROOT / f"data/aft/mix_{mix}.manifest.json"))
    total, aw, ident = m["total_assistant_tokens"], m["aw_tokens"], m["identity_tokens"]
    gen = m.get("general_tokens", total - aw - ident)
    return {"gen_tok": gen, "gen_n": m.get("general_examples", 0), "aw_tok": aw, "aw_n": m["aw_examples"],
            "id_tok": ident, "id_n": m["identity_examples"], "total": total}


def k(x):  # thousands with one decimal for tokens
    return f"{x / 1000:,.0f}K" if x else "0"


rows = []
for label, corpus, p, mix, run in ARMS:
    mid = MID[corpus]
    if mid:
        mm = json.load(open(ROOT / f"runs/midtrain/{mid}/manifest.json"))
        mt = f"{mm['tokens_trained'] / 1e6:.2f}M ({mm['steps_completed']})"
    else:
        mt = "--"
    mi = mix_info(mix)
    if run:
        upd = str(json.load(open(ROOT / f"runs/aft/{run}/manifest.json"))["steps_completed"])
    else:
        upd = "not trained"
    rows.append((label, corpus, mt, f"{p}\\%", f"{k(mi['gen_tok'])} ({mi['gen_n']:,})",
                 f"{k(mi['aw_tok'])} ({mi['aw_n']:,})", f"{k(mi['id_tok'])} ({mi['id_n']:,})",
                 f"{mi['total'] / 1e6:.3f}M", upd))

lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Stage 1: midtraining} & \multicolumn{6}{c}{Stage 2: fine-tuning} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-9}",
         r"Arm & Corpus & Tokens (updates) & $p$ & General & AW demos & Identity & Total & Updates \\",
         r"\midrule"]
groups = {"Run 1", "Run 2", "X2-20M", "A3-v2"}
for r in rows:
    lines.append(" & ".join(r).replace(",", "{,}") + r" \\")
    if r[0] in groups:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
out = "\n".join(lines) + "\n"
(TABLES / "tab_arms_composition.tex").write_text(out)
print(out)
