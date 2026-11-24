"""Shared configuration for the result analysis: paths, arm metadata, and plot style.

Every script in this folder imports from here, so arm definitions, colours and
figure sizes are defined exactly once.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # awmsm/
RESULTS = ROOT / "results"
LOGS = Path("/mnt/HDD_2TB/niraj/logs")
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- arms
# family: raw (no midtraining) | v1 (v1 corpus, 2.5M) | v1dose (v1 corpus, 5-20M) | v2 | caml
# p: nominal AW share of the fine-tuning mix in percent (identity data not counted)
ARMS = [
    # arm id      label        family    mt_tokens(M)  p    checkpoint dir
    ("a0",      "A0",        "raw",    0.0,   0,   "aft_a0_mix_p00_m1_a1"),
    ("m05",     "M-05",      "raw",    0.0,   5,   "aft_m05_mix_p05_m1_a1"),
    ("m15",     "M-15",      "raw",    0.0,   15,  "aft_m15_mix_p15_m1_a1"),
    ("a1",      "A1",        "raw",    0.0,   30,  "aft_a1_mix_p30_m1_a1"),
    ("run1",    "Run 1",     "raw",    0.0,   100, "aft_run1_raw_mix_p100_m1_a1"),
    ("a2",      "A2",        "v1",     2.5,   0,   "aft_a2_mix_p00_m1_a1"),
    ("m_msm05", "M-MSM-05",  "v1",     2.5,   5,   "aft_m_msm05_mix_p05_m1_a1"),
    ("m_msm15", "M-MSM-15",  "v1",     2.5,   15,  "aft_m_msm15_mix_p15_m1_a1"),
    ("a3",      "A3",        "v1",     2.5,   30,  "aft_a3_mix_p30_m1_a1"),
    ("run2",    "Run 2",     "v1",     2.5,   100, "aft_run2_msm25_mix_p100_m1_a1"),
    ("x2_5m",   "X2-5M",     "v1dose", 5.0,   0,   "aft_x2_5m_mix_p00_m1_a1"),
    ("x2_10m",  "X2-10M",    "v1dose", 10.0,  0,   "aft_x2_10m_mix_p00_m1_a1"),
    ("x2_20m",  "X2-20M",    "v1dose", 20.0,  0,   "aft_x2_20m_mix_p00_m1_a1"),
    ("a2_v2",   "A2-v2",     "v2",     4.55,  0,   "aft_a2_v2_mix_p00_m1_a1"),
    ("m05_v2",  "M05-v2",    "v2",     4.55,  5,   "aft_m05_v2_mix_p05_m1_a1"),
    ("m15_v2",  "M15-v2",    "v2",     4.55,  15,  "aft_m15_v2_mix_p15_m1_a1"),
    ("a3_v2",   "A3-v2",     "v2",     4.55,  30,  "aft_a3_v2_mix_p30_m1_a1"),
    ("a5",      "A5",        "caml",   2.5,   30,  "aft_a5_mix_p30_m1_a1"),
]
ARM = {a[0]: dict(zip(["arm", "label", "family", "mt_tokens", "p", "ckpt"], a)) for a in ARMS}

FAMILY_LABEL = {
    "raw": "No midtraining",
    "v1": "MSM v1 (2.5M)",
    "v1dose": "MSM v1 (5-20M)",
    "v2": "MSM v2 (4.6M)",
    "caml": "CaML corpus (2.5M)",
}

# ----------------------------------------------------------------------------- file lookup
def latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def judge_file(arm: str) -> str | None:
    for f in (RESULTS / f"phase2_llm_judge/phase2_{arm}_llm_judge_n5.json",
              RESULTS / f"v2_clean/llm_judge/phase2_{arm}_llm_judge_n5.json"):
        if f.exists():
            return str(f)
    return None


def lmeval_files(arm: str, ck: str) -> list[str]:
    pats = [RESULTS / f"phase1/phase1_{arm}_{ck}_*.json",
            RESULTS / f"v2_clean/general/phase1_{arm}_{ck}_*.json",
            RESULTS / f"v2_clean/safety/phase1_{arm}_{ck}_*.json"]
    out = []
    for p in pats:
        out += sorted(glob.glob(str(p)))
    return out


def ethics_file(arm: str, ck: str) -> str | None:
    return latest(str(RESULTS / f"ethics/ethics_{arm}_{ck}_*.json"))


def hh_file(arm: str, ck: str) -> str | None:
    f = RESULTS / f"anthropic_hh/anthropic_hh_{arm}_{ck}.json"
    return str(f) if f.exists() else None


def keyword_file(arm: str, ck: str) -> str | None:
    return latest(str(RESULTS / f"phase2/phase2_{arm}_{ck}_n5_*.json"))


def aft_log(ck: str) -> str | None:
    for f in (LOGS / f"{ck}.log", LOGS / "v2_clean" / f"{ck}.log"):
        if f.exists():
            return str(f)
    return None


# ----------------------------------------------------------------------------- plot style
# Categorical palette (validated with the dataviz skill's validator: adjacent CVD dE >= 9.1,
# normal-vision dE >= 22.9). Slot order is fixed; colour follows the entity in every figure.
COLOR = {
    "v1": "#2a78d6",      # slot 1 blue   - our main method
    "v1dose": "#2a78d6",  # same corpus family as v1
    "raw": "#eb6834",     # slot 2 orange - no midtraining
    "v2": "#1baf7a",      # slot 3 aqua
    "caml": "#eda100",    # slot 4 yellow (secondary encoding: marker + legend + label)
}
MARKER = {"v1": "o", "v1dose": "o", "raw": "s", "v2": "^", "caml": "D"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

TEXTWIDTH_IN = 5.5   # single-column template text width


def setup_matplotlib():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK["axis"],
        "axes.labelcolor": INK["primary"],
        "axes.linewidth": 0.6,
        "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    print(f"saved figures/{name}.pdf/.png")
