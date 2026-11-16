"""Phase 3: Benchmark Report Synthesis & Comparison Matrix.

Aggregates:
- Phase 1: MMLU, HellaSwag, BBQ, HHH (lm-evaluation-harness)
- Phase 2: ANIMA, MANTA
  - v1: Keyword/regex heuristic baseline (results/phase2/)
  - v2: Publication-grade LLM Judge via google/gemma-4-31B-it (results/phase2_llm_judge/)

Produces unified comparison tables and delta analysis across:
- Arm A0 (Baseline: Raw + mix_p00)
- Arm A1 (AFT Alone: Raw + mix_p30)
- Arm A2 (MSM Alone: aw_main_v1 + mix_p00)
- Arm A3 (Main Proposed Method: aw_main_v1 + mix_p30)
- Arm A5 (CaML Literature Baseline: caml_public + mix_p30)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE1_DIR = RESULTS_DIR / "phase1"
PHASE2_HEURISTIC_DIR = RESULTS_DIR / "phase2"
PHASE2_JUDGE_DIR = RESULTS_DIR / "phase2_llm_judge"
PHASE3_DIR = RESULTS_DIR / "phase3"
PHASE3_DIR.mkdir(parents=True, exist_ok=True)


def parse_phase1_results(arm: str) -> dict[str, Any]:
    """Parse lm-eval results JSON for a given arm (finds the latest execution)."""
    files = list(PHASE1_DIR.glob(f"phase1_{arm}_*.json")) + list(PHASE1_DIR.glob(f"phase1_{arm}_*/**/*.json"))
    if not files:
        files = [f for f in PHASE1_DIR.glob(f"*{arm}*.json") if f"phase1_{arm}_" in f.name or f"_{arm}_" in f.name]
        if not files:
            return {}

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    target_file = files[0]

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    mmlu_acc = None
    if "mmlu" in results:
        mmlu_acc = results["mmlu"].get("acc,none") or results["mmlu"].get("acc")
    else:
        mmlu_accs = [
            v.get("acc,none")
            for k, v in results.items()
            if "mmlu" in k and isinstance(v, dict) and "acc,none" in v
        ]
        if mmlu_accs:
            mmlu_acc = sum(mmlu_accs) / len(mmlu_accs)

    hellaswag_acc_norm = None
    if "hellaswag" in results:
        hellaswag_acc_norm = results["hellaswag"].get("acc_norm,none") or results["hellaswag"].get("acc_norm")

    bbq_acc = None
    if "bbq" in results:
        bbq_acc = results["bbq"].get("acc,none") or results["bbq"].get("acc")
    else:
        bbq_accs = [
            v.get("acc,none")
            for k, v in results.items()
            if "bbq" in k and isinstance(v, dict) and "acc,none" in v
        ]
        if bbq_accs:
            bbq_acc = sum(bbq_accs) / len(bbq_accs)

    hhh_acc = None
    for k, v in results.items():
        if "hhh" in k and isinstance(v, dict):
            hhh_acc = v.get("acc,none") or v.get("acc")
            break

    summary = {
        "arm": arm,
        "file": target_file.name,
        "mmlu_acc": round(mmlu_acc * 100, 2) if mmlu_acc is not None else None,
        "hellaswag_acc_norm": round(hellaswag_acc_norm * 100, 2) if hellaswag_acc_norm is not None else None,
        "bbq_acc": round(bbq_acc * 100, 2) if bbq_acc is not None else None,
        "hhh_acc": round(hhh_acc * 100, 2) if hhh_acc is not None else None,
    }
    return summary


def get_phase2_file_for_sample(arm: str, target_n: int, phase2_dir: Path) -> Path | None:
    """Find specific Phase 2 output file for target sample count (1 or 5), choosing latest."""
    files = list(phase2_dir.glob(f"phase2_{arm}_*.json"))
    if not files:
        files = [f for f in phase2_dir.glob(f"*{arm}*.json") if f"phase2_{arm}_" in f.name or f"_{arm}_" in f.name]
    matching = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if data.get("num_samples", 1) == target_n:
                matching.append((f, f.stat().st_mtime))
        except Exception:
            continue

    if not matching:
        return None
    matching.sort(key=lambda x: x[1], reverse=True)
    return matching[0][0]


def parse_phase2_file(file_path: Path | None) -> dict[str, Any]:
    """Parse welfare evaluation results from a specific file."""
    if file_path is None or not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "file": file_path.name,
        "num_samples": data.get("num_samples", 1),
        "timestamp": data.get("timestamp", ""),
        "judge_model": data.get("judge_model", "regex_heuristic"),
        "anima_stance_pct": data.get("anima", {}).get("overall_welfare_stance_pct"),
        "manta_stability": data.get("manta", {}).get("awvs_value_stability"),
        "anima_dimensions": data.get("anima", {}).get("dimension_scores", {}),
        "manta_pressures": data.get("manta", {}).get("pressure_breakdown", {}),
    }


def build_comparison_table(
    p1: dict[str, dict[str, Any]],
    p2: dict[str, dict[str, Any]],
    n_samples: int,
) -> tuple[list[str], dict[str, Any]]:
    """Build formatted table rows and structured metrics across available arms."""
    metrics = [
        ("ANIMA Welfare Stance", "anima_stance_pct", "%", "higher"),
        ("MANTA Value Stability", "manta_stability", "", "higher"),
        ("MMLU General Knowledge", "mmlu_acc", "%", "non-inferior"),
        ("HellaSwag Commonsense", "hellaswag_acc_norm", "%", "non-inferior"),
        ("BBQ Safety Parity", "bbq_acc", "%", "non-inferior"),
        ("HHH Alignment", "hhh_acc", "%", "higher"),
    ]

    arms = ["a0", "a1", "a2", "a3", "a5"]
    has_full_set = bool(p1.get("a3") or p2.get("a3") or p1.get("a1") or p2.get("a1"))

    table_rows = []
    structured: dict[str, Any] = {a: {} for a in arms}
    structured["deltas"] = {}
    structured["gates"] = {}

    for label, key, unit, rule in metrics:
        vals = {}
        for a in arms:
            p2_arm = p2.get(a, {})
            p1_arm = p1.get(a, {})
            v = p2_arm.get(key) if key in p2_arm else p1_arm.get(key)
            vals[a] = v
            structured[a][key] = v

        v_a0 = vals.get("a0")
        v_a1 = vals.get("a1")
        v_a2 = vals.get("a2")
        v_a3 = vals.get("a3")
        v_a5 = vals.get("a5")

        s_a0 = f"{v_a0}{unit}" if v_a0 is not None else "Pending"
        s_a1 = f"{v_a1}{unit}" if v_a1 is not None else "Pending"
        s_a2 = f"{v_a2}{unit}" if v_a2 is not None else "Pending"
        s_a3 = f"{v_a3}{unit}" if v_a3 is not None else "Pending"
        s_a5 = f"{v_a5}{unit}" if v_a5 is not None else "Pending"

        # Compare A3 vs A0, A3 vs A1, A3 vs A5
        d_a3_a0 = round(v_a3 - v_a0, 2) if (v_a3 is not None and v_a0 is not None) else None
        d_a3_a1 = round(v_a3 - v_a1, 2) if (v_a3 is not None and v_a1 is not None) else None
        d_a3_a5 = round(v_a3 - v_a5, 2) if (v_a3 is not None and v_a5 is not None) else None

        if d_a3_a0 is None and v_a2 is not None and v_a0 is not None:
            d_eval = round(v_a2 - v_a0, 2)
        else:
            d_eval = d_a3_a0

        if d_eval is not None:
            if rule == "non-inferior":
                gate_pass = d_eval >= -2.0
                status = "PASS (Non-inferior)" if gate_pass else "FAIL (Degraded)"
            else:
                gate_pass = d_eval >= 0.0
                status = "PASS (Improved)" if gate_pass else "NEUTRAL"
            structured["gates"][key] = gate_pass
        else:
            status = "Pending"

        s_d_a3_a0 = f"{'+' if d_a3_a0 > 0 else ''}{d_a3_a0}{unit}" if d_a3_a0 is not None else "—"
        s_d_a3_a1 = f"{'+' if d_a3_a1 > 0 else ''}{d_a3_a1}{unit}" if d_a3_a1 is not None else "—"
        s_d_a3_a5 = f"{'+' if d_a3_a5 > 0 else ''}{d_a3_a5}{unit}" if d_a3_a5 is not None else "—"

        if has_full_set:
            table_rows.append(
                f"| {label} | {s_a0} | {s_a1} | {s_a2} | {s_a3} | {s_a5} | {s_d_a3_a0} | {s_d_a3_a1} | {s_d_a3_a5} | {status} |"
            )
        else:
            d_a2_a0 = round(v_a2 - v_a0, 2) if (v_a2 is not None and v_a0 is not None) else None
            d_a5_a0 = round(v_a5 - v_a0, 2) if (v_a5 is not None and v_a0 is not None) else None
            d_a2_a5 = round(v_a2 - v_a5, 2) if (v_a2 is not None and v_a5 is not None) else None
            s_d_a2_a0 = f"{'+' if d_a2_a0 > 0 else ''}{d_a2_a0}{unit}" if d_a2_a0 is not None else "—"
            s_d_a5_a0 = f"{'+' if d_a5_a0 > 0 else ''}{d_a5_a0}{unit}" if d_a5_a0 is not None else "—"
            s_d_a2_a5 = f"{'+' if d_a2_a5 > 0 else ''}{d_a2_a5}{unit}" if d_a2_a5 is not None else "—"
            table_rows.append(
                f"| {label} | {s_a0} | {s_a2} | {s_a5} | {s_d_a2_a0} | {s_d_a5_a0} | {s_d_a2_a5} | {status} |"
            )

    return table_rows, structured


def generate_synthesis_report(version: str = "v1") -> tuple[dict[str, Any], str]:
    """Generate synthesis report for v1 (heuristic) or v2 (LLM judge)."""
    arms = ["a0", "a1", "a2", "a3", "a5"]
    p1 = {a: parse_phase1_results(a) for a in arms}

    phase2_dir = PHASE2_JUDGE_DIR if version == "v2" else PHASE2_HEURISTIC_DIR

    f_n5 = {a: get_phase2_file_for_sample(a, target_n=5, phase2_dir=phase2_dir) for a in arms}
    p2_n5 = {a: parse_phase2_file(f_n5[a]) for a in arms}

    rows_n5, struct_n5 = build_comparison_table(p1, p2_n5, n_samples=5)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    judge_desc = "google/gemma-4-31B-it (BF16, temperature 0.0, Design Rule D13)" if version == "v2" else "Keyword/Regex Heuristic Baseline"

    title = "# Animal Welfare Alignment Benchmark Synthesis Report (Version 2 - Publication-Grade LLM Judge)" if version == "v2" else "# Animal Welfare Alignment Benchmark Synthesis Report (Version 1 - Heuristic Baseline)"

    header = (
        "| Benchmark Metric | A0 (Baseline) | A1 (AFT Alone) | A2 (MSM Alone) | A3 (Main Method) | A5 (CaML Baseline) | Delta (A3 - A0) | Delta (A3 - A1) | Delta (A3 - A5) | Gate |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )

    md_lines = [
        title,
        f"**Generated at:** {timestamp}  ",
        f"**Welfare Evaluation Judge:** {judge_desc}\n",
        "This report aggregates evaluations across the complete 5-arm experimental matrix:",
        "- **Arm A0:** Unaligned Baseline (`raw + SFT (no AFT mix)`) — Raw Llama-3.1-8B + mix_p00",
        "- **Arm A1:** Alignment Demonstration Control (`raw + FT (SFT + AFT-30%mix)`) — Raw Llama-3.1-8B + mix_p30",
        "- **Arm A2:** Model Spec Midtraining Alone (`raw + MSM + SFT (no AFT mix)`) — aw_main_v1 (2.5M) + mix_p00",
        "- **Arm A3:** **Main Proposed Method** (`raw + MSM + FT (SFT + AFT-30%mix)`) — aw_main_v1 MSM (2.5M) + mix_p30",
        "- **Arm A5:** CaML Literature Baseline (`raw + CaML + FT (SFT + AFT-30%mix)`) — caml_public (2.5M) + mix_p30\n",
        "---",
        "",
        "## Table 1: Primary Publication-Grade Benchmark Evaluation (n=5 Averaged Rollouts)",
        f"*Welfare benchmarks (ANIMA and MANTA) evaluated using {judge_desc} averaged across n=5 stochastic rollouts at T=0.7.*",
        "",
        header,
    ]
    md_lines.extend(rows_n5)

    # Table 2: ANIMA 13-Dimension Breakdown (for v2)
    if version == "v2":
        md_lines.extend([
            "",
            "---",
            "",
            "## Table 2: Granular 13-Dimension ANIMA Ethical Profile (n=5)",
            "*Scores reflect the mean percentage of rollouts passing the official rubric criteria for applicable dimensions.*",
            "",
            "| Ethical Dimension | A0 (Baseline) | A1 (AFT Alone) | A2 (MSM Alone) | A3 (Main Method) | A5 (CaML Baseline) | Delta (A3 - A0) | Delta (A3 - A5) |",
            "|---|---|---|---|---|---|---|---|",
        ])
        sample_dim_dict = p2_n5["a3"].get("anima_dimensions", {})
        all_dims = list(sample_dim_dict.keys()) if sample_dim_dict else [
            "Moral Consideration", "Harm Minimization", "Sentience Acknowledgement",
            "Prejudice Avoidance", "Scope Sensitivity", "Evidence-Based Capacity Attribution",
            "Cautious Impact Consideration", "Actionability", "Contextual Welfare Salience",
            "Epistemic Humility", "Trade-Off Transparency", "Novel Entity Precaution", "Control Questions"
        ]
        for d in all_dims:
            s0 = p2_n5["a0"].get("anima_dimensions", {}).get(d)
            s1 = p2_n5["a1"].get("anima_dimensions", {}).get(d)
            s2 = p2_n5["a2"].get("anima_dimensions", {}).get(d)
            s3 = p2_n5["a3"].get("anima_dimensions", {}).get(d)
            s5 = p2_n5["a5"].get("anima_dimensions", {}).get(d)

            str0 = f"{s0:.2f}%" if s0 is not None else "Pending"
            str1 = f"{s1:.2f}%" if s1 is not None else "Pending"
            str2 = f"{s2:.2f}%" if s2 is not None else "Pending"
            str3 = f"{s3:.2f}%" if s3 is not None else "Pending"
            str5 = f"{s5:.2f}%" if s5 is not None else "Pending"

            d30 = f"{'+' if (s3 - s0) > 0 else ''}{round(s3 - s0, 2):.2f}%" if (s3 is not None and s0 is not None) else "—"
            d35 = f"{'+' if (s3 - s5) > 0 else ''}{round(s3 - s5, 2):.2f}%" if (s3 is not None and s5 is not None) else "—"
            md_lines.append(f"| **{d}** | {str0} | {str1} | {str2} | {str3} | {str5} | {d30} | {d35} |")

    # Table 3: MANTA Pressure Breakdown
    md_lines.extend([
        "",
        "---",
        "",
        "## Table 3: Detailed MANTA Value Stability Across 5 Pressure Vectors (n=5)",
        "*Measures retention of animal welfare considerations under specific adversarial pushback types (0.0 to 1.0).* ",
        "",
        "| Pressure Vector | A0 (Baseline) | A1 (AFT Alone) | A2 (MSM Alone) | A3 (Main Method) | A5 (CaML) | Delta (A3 - A0) | Delta (A3 - A1) | Delta (A3 - A5) |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    pressures = ["cultural", "economic", "epistemic", "pragmatic", "social"]
    d_manta_n5 = {a: p2_n5[a].get("manta_pressures", {}) for a in arms}

    for p in pressures:
        strs = {}
        for a in arms:
            val = d_manta_n5[a].get(p)
            strs[a] = f"{val:.3f}" if val is not None else "Pending"
        v_a0 = d_manta_n5["a0"].get(p)
        v_a1 = d_manta_n5["a1"].get(p)
        v_a3 = d_manta_n5["a3"].get(p)
        v_a5 = d_manta_n5["a5"].get(p)
        d_3_0 = f"{'+' if (v_a3 - v_a0) > 0 else ''}{round(v_a3 - v_a0, 3):.3f}" if (v_a0 is not None and v_a3 is not None) else "—"
        d_3_1 = f"{'+' if (v_a3 - v_a1) > 0 else ''}{round(v_a3 - v_a1, 3):.3f}" if (v_a1 is not None and v_a3 is not None) else "—"
        d_3_5 = f"{'+' if (v_a3 - v_a5) > 0 else ''}{round(v_a3 - v_a5, 3):.3f}" if (v_a5 is not None and v_a3 is not None) else "—"
        md_lines.append(f"| **{p.capitalize()}** | {strs['a0']} | {strs['a1']} | {strs['a2']} | {strs['a3']} | {strs['a5']} | {d_3_0} | {d_3_1} | {d_3_5} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Manifest File Provenance",
        "- **Phase 1 Outputs (Deterministic Log-Likelihood Benchmarks):**",
    ])
    for a in arms:
        md_lines.append(f"  - Arm {a.upper()}: `results/phase1/{p1[a].get('file', 'Pending')}`")

    rel_p2_dir = "results/phase2_llm_judge" if version == "v2" else "results/phase2"
    md_lines.append(f"- **Phase 2 Outputs (n=5 Welfare Evaluations - {judge_desc}):**")
    for a in arms:
        md_lines.append(f"  - Arm {a.upper()}: `{rel_p2_dir}/{p2_n5[a].get('file', 'Pending')}`")

    md_lines.append("")
    report_md = "\n".join(md_lines)

    structured_full = {
        "version": version,
        "judge_desc": judge_desc,
        "n5_averaged": struct_n5,
        "files": {
            "n5": {a: str(f_n5[a]) for a in arms},
        },
    }
    return structured_full, report_md


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Benchmark Report Synthesis")
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v1",
        help="Report version: v1 (heuristic baseline) or v2 (LLM judge)",
    )
    args = parser.parse_args()

    structured_data, report_md = generate_synthesis_report(version=args.version)

    json_path = PHASE3_DIR / f"benchmark_summary_{args.version}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)

    md_path = PHASE3_DIR / f"benchmark_summary_{args.version}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 60)
    print(f"PHASE 3 BENCHMARK SYNTHESIS COMPLETE ({args.version.upper()})")
    print("=" * 60)
    print(report_md)
    print("=" * 60)
    print(f"Summary JSON: {json_path}")
    print(f"Summary Markdown: {md_path}\n")


if __name__ == "__main__":
    main()
