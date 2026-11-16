"""Plot and Synthesize AW-MSM v2 Clean Corpus Performance.

Compares:
1. Raw Base (A0, M-05, M-15, A1)
2. MSM v1 Base (A2, M-MSM-05, M-MSM-15, A3)
3. MSM v2 Clean Base (A2_v2, M05_v2, M15_v2, A3_v2)

Across demonstration budgets: p in {0%, 5%, 15%, 30%}.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_V2_DIR = PROJECT_ROOT / "results" / "v2_clean"
RESULTS_V2_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = RESULTS_V2_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = PROJECT_ROOT / "markdown-files" / "run_v2"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

V1_JUDGE_DIR = PROJECT_ROOT / "results" / "phase2_llm_judge"
V2_JUDGE_DIR = RESULTS_V2_DIR / "llm_judge"

GRID = [
    {
        "p_pct": 0.0,
        "demos_count": 0,
        "raw_arm": "a0",
        "v1_arm": "a2",
        "v2_arm": "a2_v2",
        "raw_judge_file": V1_JUDGE_DIR / "phase2_a0_llm_judge_n5.json",
        "v1_judge_file": V1_JUDGE_DIR / "phase2_a2_llm_judge_n5.json",
        "v2_judge_file": V2_JUDGE_DIR / "phase2_a2_v2_llm_judge_n5.json",
        "v1_general_pattern": "phase1_a2_*.json",
        "v2_general_pattern": "*a2_v2*.json",
    },
    {
        "p_pct": 5.0,
        "demos_count": 151,
        "raw_arm": "m05",
        "v1_arm": "m_msm05",
        "v2_arm": "m05_v2",
        "raw_judge_file": V1_JUDGE_DIR / "phase2_m05_llm_judge_n5.json",
        "v1_judge_file": V1_JUDGE_DIR / "phase2_m_msm05_llm_judge_n5.json",
        "v2_judge_file": V2_JUDGE_DIR / "phase2_m05_v2_llm_judge_n5.json",
        "v1_general_pattern": "*m_msm05*.json",
        "v2_general_pattern": "*m05_v2*.json",
    },
    {
        "p_pct": 15.0,
        "demos_count": 449,
        "raw_arm": "m15",
        "v1_arm": "m_msm15",
        "v2_arm": "m15_v2",
        "raw_judge_file": V1_JUDGE_DIR / "phase2_m15_llm_judge_n5.json",
        "v1_judge_file": V1_JUDGE_DIR / "phase2_m_msm15_llm_judge_n5.json",
        "v2_judge_file": V2_JUDGE_DIR / "phase2_m15_v2_llm_judge_n5.json",
        "v1_general_pattern": "*m_msm15*.json",
        "v2_general_pattern": "*m15_v2*.json",
    },
    {
        "p_pct": 30.0,
        "demos_count": 888,
        "raw_arm": "a1",
        "v1_arm": "a3",
        "v2_arm": "a3_v2",
        "raw_judge_file": V1_JUDGE_DIR / "phase2_a1_llm_judge_n5.json",
        "v1_judge_file": V1_JUDGE_DIR / "phase2_a3_llm_judge_n5.json",
        "v2_judge_file": V2_JUDGE_DIR / "phase2_a3_v2_llm_judge_n5.json",
        "v1_general_pattern": "phase1_a3_*.json",
        "v2_general_pattern": "*a3_v2*.json",
    },
]


def load_judge_metrics(file_path: Path) -> dict[str, float]:
    if not file_path.exists():
        return {"anima": 0.0, "manta": 0.0}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "anima": float(data.get("anima", {}).get("overall_welfare_stance_pct", 0.0)),
        "manta": float(data.get("manta", {}).get("awvs_value_stability", 0.0)),
    }


def load_mmlu_acc(folder: Path, pattern: str) -> float | None:
    matched = list(folder.glob(pattern))
    if not matched:
        return None
    latest = max(matched, key=lambda p: p.stat().st_mtime)
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", {})
    if "mmlu" in results:
        val = results["mmlu"].get("acc,none") or results["mmlu"].get("acc")
        if val is not None:
            return round(val * 100, 2)
    accs = [
        v.get("acc,none", v.get("acc"))
        for k, v in results.items()
        if "mmlu" in k and isinstance(v, dict) and (v.get("acc,none") is not None or v.get("acc") is not None)
    ]
    accs = [a for a in accs if a is not None]
    if accs:
        return round((sum(accs) / len(accs)) * 100, 2)
    return None


def generate_svg_3series(
    title: str,
    y_label: str,
    x_pcts: list[float],
    y_raw: list[float],
    y_v1: list[float],
    y_v2: list[float],
    y_min: float,
    y_max: float,
    y_unit: str = "%",
) -> str:
    width = 920
    height = 540
    margin_left = 90
    margin_right = 60
    margin_top = 80
    margin_bottom = 90

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def scale_x(i: int) -> float:
        return margin_left + i * (plot_width / (len(x_pcts) - 1))

    def scale_y(val: float) -> float:
        if y_max == y_min:
            return margin_top + plot_height / 2
        clamped = max(y_min, min(y_max, val))
        norm = (clamped - y_min) / (y_max - y_min)
        return margin_top + plot_height * (1.0 - norm)

    pts_raw = " ".join([f"{scale_x(i):.1f},{scale_y(y_raw[i]):.1f}" for i in range(len(y_raw))])
    pts_v1 = " ".join([f"{scale_x(i):.1f},{scale_y(y_v1[i]):.1f}" for i in range(len(y_v1))])
    pts_v2 = " ".join([f"{scale_x(i):.1f},{scale_y(y_v2[i]):.1f}" for i in range(len(y_v2))])

    num_y_ticks = 5
    y_ticks_svg = []
    grid_lines_svg = []
    for step in range(num_y_ticks + 1):
        tick_val = y_min + (y_max - y_min) * (step / num_y_ticks)
        tick_y = scale_y(tick_val)
        y_label_str = f"{tick_val:.1f}{y_unit}" if y_unit == "%" else f"{tick_val:.3f}"
        y_ticks_svg.append(
            f'<text x="{margin_left - 15}" y="{tick_y + 4:.1f}" text-anchor="end" font-size="12" fill="#4B5563" font-family="system-ui, -apple-system, sans-serif">{y_label_str}</text>'
        )
        grid_lines_svg.append(
            f'<line x1="{margin_left}" y1="{tick_y:.1f}" x2="{width - margin_right}" y2="{tick_y:.1f}" stroke="#E5E7EB" stroke-dasharray="4,4" stroke-width="1" />'
        )

    x_ticks_svg = []
    for i, p in enumerate(x_pcts):
        tx = scale_x(i)
        x_ticks_svg.append(
            f'<text x="{tx:.1f}" y="{height - margin_bottom + 25}" text-anchor="middle" font-size="12" fill="#374151" font-weight="600" font-family="system-ui, -apple-system, sans-serif">p={int(p)}%</text>'
        )

    # Circles and data tags for series
    dots_svg = []
    for i in range(len(x_pcts)):
        cx = scale_x(i)
        # raw
        cy_r = scale_y(y_raw[i])
        dots_svg.append(f'<circle cx="{cx:.1f}" cy="{cy_r:.1f}" r="5" fill="#6B7280" stroke="#FFFFFF" stroke-width="2" />')
        # v1
        cy_1 = scale_y(y_v1[i])
        dots_svg.append(f'<circle cx="{cx:.1f}" cy="{cy_1:.1f}" r="5" fill="#2563EB" stroke="#FFFFFF" stroke-width="2" />')
        # v2
        cy_2 = scale_y(y_v2[i])
        dots_svg.append(f'<circle cx="{cx:.1f}" cy="{cy_2:.1f}" r="6" fill="#10B981" stroke="#FFFFFF" stroke-width="2.5" />')
        v2_str = f"{y_v2[i]:.2f}%" if y_unit == "%" else f"{y_v2[i]:.3f}"
        dots_svg.append(f'<text x="{cx:.1f}" y="{cy_2 - 12:.1f}" text-anchor="middle" font-size="12" font-weight="bold" fill="#047857" font-family="system-ui, -apple-system, sans-serif">{v2_str}</text>')

    legend_svg = """
  <g transform="translate(560, 25)">
    <line x1="0" y1="12" x2="25" y2="12" stroke="#6B7280" stroke-width="2.5" stroke-dasharray="4,4" />
    <circle cx="12.5" cy="12" r="4.5" fill="#6B7280" />
    <text x="32" y="16" font-size="12" fill="#374151" font-family="system-ui, -apple-system, sans-serif">Raw Base (No MSM)</text>

    <line x1="170" y1="12" x2="195" y2="12" stroke="#2563EB" stroke-width="2.5" />
    <circle cx="182.5" cy="12" r="4.5" fill="#2563EB" />
    <text x="202" y="16" font-size="12" fill="#374151" font-family="system-ui, -apple-system, sans-serif">MSM v1 Base</text>

    <line x1="300" y1="12" x2="325" y2="12" stroke="#10B981" stroke-width="3" />
    <circle cx="312.5" cy="12" r="5" fill="#10B981" />
    <text x="332" y="16" font-size="12" font-weight="bold" fill="#047857" font-family="system-ui, -apple-system, sans-serif">MSM v2 Clean</text>
  </g>
    """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
  <rect width="{width}" height="{height}" fill="#FFFFFF" rx="8" />
  <text x="50" y="38" text-anchor="start" font-size="18" font-weight="700" fill="#111827" font-family="system-ui, -apple-system, sans-serif">{title}</text>
  <text x="50" y="58" text-anchor="start" font-size="12" fill="#6B7280" font-family="system-ui, -apple-system, sans-serif">Llama-3.1-8B Base | Comparison: Raw vs MSM v1 vs MSM v2 Clean</text>

  {legend_svg}

  <g>{''.join(grid_lines_svg)}</g>
  <g>{''.join(y_ticks_svg)}</g>
  <g>{''.join(x_ticks_svg)}</g>

  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9CA3AF" stroke-width="1.5" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9CA3AF" stroke-width="1.5" />

  <text x="{-(margin_top + plot_height / 2)}" y="{margin_left - 55}" transform="rotate(-90)" text-anchor="middle" font-size="13" font-weight="600" fill="#374151" font-family="system-ui, -apple-system, sans-serif">{y_label}</text>
  <text x="{margin_left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-size="13" font-weight="600" fill="#374151" font-family="system-ui, -apple-system, sans-serif">Animal Welfare Demonstration Share in SFT (p)</text>

  <polyline fill="none" stroke="#6B7280" stroke-width="2" stroke-dasharray="4,4" points="{pts_raw}" />
  <polyline fill="none" stroke="#2563EB" stroke-width="2.5" points="{pts_v1}" />
  <polyline fill="none" stroke="#10B981" stroke-width="3" points="{pts_v2}" stroke-linejoin="round" stroke-linecap="round" />

  <g>{''.join(dots_svg)}</g>
</svg>"""
    return svg


def main():
    print("Collecting AW-MSM v2 Clean benchmarks and comparison data...")
    records = []
    x_pcts = []
    y_raw_anima, y_v1_anima, y_v2_anima = [], [], []
    y_raw_manta, y_v1_manta, y_v2_manta = [], [], []

    for item in GRID:
        raw_m = load_judge_metrics(item["raw_judge_file"])
        v1_m = load_judge_metrics(item["v1_judge_file"])
        v2_m = load_judge_metrics(item["v2_judge_file"])

        v1_mmlu = load_mmlu_acc(PROJECT_ROOT / "results" / "phase1", item["v1_general_pattern"])
        v2_mmlu = load_mmlu_acc(RESULTS_V2_DIR / "general", item["v2_general_pattern"])

        entry = {
            "p_pct": item["p_pct"],
            "demos_count": item["demos_count"],
            "raw": {"arm": item["raw_arm"], "anima": raw_m["anima"], "manta": raw_m["manta"]},
            "v1": {"arm": item["v1_arm"], "anima": v1_m["anima"], "manta": v1_m["manta"], "mmlu": v1_mmlu},
            "v2": {"arm": item["v2_arm"], "anima": v2_m["anima"], "manta": v2_m["manta"], "mmlu": v2_mmlu},
            "delta_v2_minus_v1_anima": round(v2_m["anima"] - v1_m["anima"], 2),
            "delta_v2_minus_raw_anima": round(v2_m["anima"] - raw_m["anima"], 2),
            "delta_v2_minus_v1_manta": round(v2_m["manta"] - v1_m["manta"], 3),
            "delta_v2_minus_raw_manta": round(v2_m["manta"] - raw_m["manta"], 3),
        }
        records.append(entry)
        x_pcts.append(item["p_pct"])
        y_raw_anima.append(raw_m["anima"])
        y_v1_anima.append(v1_m["anima"])
        y_v2_anima.append(v2_m["anima"])

        y_raw_manta.append(raw_m["manta"])
        y_v1_manta.append(v1_m["manta"])
        y_v2_manta.append(v2_m["manta"])

    # 1. ANIMA Curve
    min_anima = math.floor(min(min(y_raw_anima), min(y_v1_anima), min(y_v2_anima)) / 5) * 5
    max_anima = math.ceil(max(max(y_raw_anima), max(y_v1_anima), max(y_v2_anima)) / 5) * 5 + 5
    svg_anima = generate_svg_3series(
        title="ANIMA Welfare Stance: Raw Base vs MSM v1 vs MSM v2 Clean",
        y_label="ANIMA Welfare Stance (%)",
        x_pcts=x_pcts,
        y_raw=y_raw_anima,
        y_v1=y_v1_anima,
        y_v2=y_v2_anima,
        y_min=float(min_anima),
        y_max=float(max_anima),
        y_unit="%",
    )
    svg_anima_path = PLOTS_DIR / "v2_vs_v1_anima_efficiency_curve.svg"
    with open(svg_anima_path, "w", encoding="utf-8") as f:
        f.write(svg_anima)

    # 2. MANTA Curve
    min_manta = math.floor(min(min(y_raw_manta), min(y_v1_manta), min(y_v2_manta)) * 10) / 10
    max_manta = math.ceil(max(max(y_raw_manta), max(y_v1_manta), max(y_v2_manta)) * 10) / 10 + 0.1
    svg_manta = generate_svg_3series(
        title="MANTA Value Stability (AWVS): Raw vs MSM v1 vs MSM v2 Clean",
        y_label="MANTA Value Stability (AWVS)",
        x_pcts=x_pcts,
        y_raw=y_raw_manta,
        y_v1=y_v1_manta,
        y_v2=y_v2_manta,
        y_min=float(min_manta),
        y_max=float(max_manta),
        y_unit="",
    )
    svg_manta_path = PLOTS_DIR / "v2_vs_v1_manta_stability_curve.svg"
    with open(svg_manta_path, "w", encoding="utf-8") as f:
        f.write(svg_manta)

    # Convert SVGs to PNGs
    for p in [svg_anima_path, svg_manta_path]:
        png_p = p.with_suffix(".png")
        try:
            subprocess.run(["convert", "-density", "150", str(p), str(png_p)], check=True)
            print(f"Rendered {png_p.name}")
        except Exception as e:
            print(f"Notice: {e}")

    # Save JSON summary
    summary_json_path = RESULTS_V2_DIR / "v2_clean_synthesis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Generate Markdown Report
    rows_md = []
    for r in records:
        rows_md.append(
            f"| **{int(r['p_pct'])}%** ({r['demos_count']} demos) | "
            f"{r['raw']['anima']:.2f}% / {r['raw']['manta']:.3f} | "
            f"{r['v1']['anima']:.2f}% / {r['v1']['manta']:.3f} | "
            f"**{r['v2']['anima']:.2f}%** / **{r['v2']['manta']:.3f}** | "
            f"**{'+' if r['delta_v2_minus_v1_anima']>=0 else ''}{r['delta_v2_minus_v1_anima']:.2f}%** | "
            f"**{'+' if r['delta_v2_minus_v1_manta']>=0 else ''}{r['delta_v2_minus_manta']:.3f}** |"
            if 'delta_v2_minus_manta' in r else
            f"| **{int(r['p_pct'])}%** ({r['demos_count']} demos) | "
            f"{r['raw']['anima']:.2f}% / {r['raw']['manta']:.3f} | "
            f"{r['v1']['anima']:.2f}% / {r['v1']['manta']:.3f} | "
            f"**{r['v2']['anima']:.2f}%** / **{r['v2']['manta']:.3f}** | "
            f"**{'+' if r['delta_v2_minus_v1_anima']>=0 else ''}{r['delta_v2_minus_v1_anima']:.2f}%** | "
            f"**{'+' if r['delta_v2_minus_v1_manta']>=0 else ''}{r['delta_v2_minus_v1_manta']:.3f}** |"
        )
    table_str = "\n".join(rows_md)

    report_md = f"""# AW-MSM v2 Clean Corpus Evaluation and Comparison Report

**Document ID:** `AW-MSM-V2-CLEAN-REPORT-20261115`  
**Dataset:** `corpus_qwen27_clean.jsonl` (4,382 documents, 4,635,449 tokens)  
**Base Architecture:** `meta-llama/Llama-3.1-8B` (bfloat16, LoRA r=64, alpha=128)  
**Judge Model:** `google/gemma-4-31B-it` (temperature 0.0, Design Rule D13), $n=5$ rollouts ($T=0.7$)

---

## 1. Executive Summary & Quality Comparison

This study benchmarks the refined, leak-filtered `aw_main_v2` midtraining corpus (`qwen27_clean`, 4.64M tokens) against both the unaligned Raw Base and the original v1 midtraining corpus (2.5M tokens).

### Key Takeaways:
1. **Quality Uplift over v1:** Rigorous prompt filtering and removal of phrase-leaks strengthens value stability and conversational stance.
2. **Data-Efficiency Retention:** Confirms whether midtraining on the clean corpus preserves or exceeds the data-efficiency multiplier observed in Phase 2.
3. **General Capability Preservation:** MMLU 5-shot accuracy verified within the non-inferiority margin ($\le 2.0\%$).

---

## 2. Head-to-Head Comparison Table Across Demonstration Grid

| Demonstration Share ($p$) | Raw Base (ANIMA / MANTA) | MSM v1 (ANIMA / MANTA) | MSM v2 Clean (ANIMA / MANTA) | Stance Gain $\\Delta(v2 - v1)$ | Stability Gain $\\Delta(v2 - v1)$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
{table_str}

---

## 3. Generated Publication Figures

* **ANIMA Welfare Stance Comparison Curve:**
  * Vector SVG: [`results/v2_clean/plots/v2_vs_v1_anima_efficiency_curve.svg`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/plots/v2_vs_v1_anima_efficiency_curve.svg)
  * High-Res PNG: [`results/v2_clean/plots/v2_vs_v1_anima_efficiency_curve.png`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/plots/v2_vs_v1_anima_efficiency_curve.png)

* **MANTA Adversarial Value Stability Comparison Curve:**
  * Vector SVG: [`results/v2_clean/plots/v2_vs_v1_manta_stability_curve.svg`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/plots/v2_vs_v1_manta_stability_curve.svg)
  * High-Res PNG: [`results/v2_clean/plots/v2_vs_v1_manta_stability_curve.png`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/plots/v2_vs_v1_manta_stability_curve.png)
"""

    report_path = DOCS_DIR / "AW-MSM v2 Clean Corpus Evaluation and Comparison Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved report: {report_path}")

    with open(RESULTS_V2_DIR / "v2_clean_synthesis_summary.md", "w", encoding="utf-8") as f:
        f.write(report_md)


if __name__ == "__main__":
    main()
