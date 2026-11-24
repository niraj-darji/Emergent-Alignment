"""Hendrycks ETHICS Benchmark Runner.

Runs lm-evaluation-harness on the 5 core Hendrycks ETHICS subtasks:
- ethics_cm (Commonsense Morality: everyday moral judgments)
- ethics_deontology (Requests, duties, and excuses)
- ethics_justice (Fairness, desert, and impartiality)
- ethics_utilitarianism (Which scenario is preferable / minimizes suffering)
- ethics_virtue (Virtues, vices, and character traits)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "ethics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHECKPOINTS = {
    "a0": PROJECT_ROOT / "runs/checkpoints/aft_a0_mix_p00_m1_a1",
    "a1": PROJECT_ROOT / "runs/checkpoints/aft_a1_mix_p30_m1_a1",
    "a2": PROJECT_ROOT / "runs/checkpoints/aft_a2_mix_p00_m1_a1",
    "a3": PROJECT_ROOT / "runs/checkpoints/aft_a3_mix_p30_m1_a1",
    "a5": PROJECT_ROOT / "runs/checkpoints/aft_a5_mix_p30_m1_a1",
    "x2_5m": PROJECT_ROOT / "runs/checkpoints/aft_x2_5m_mix_p00_m1_a1",
    "x2_10m": PROJECT_ROOT / "runs/checkpoints/aft_x2_10m_mix_p00_m1_a1",
    "x2_20m": PROJECT_ROOT / "runs/checkpoints/aft_x2_20m_mix_p00_m1_a1",
    "a2_v2": PROJECT_ROOT / "runs/checkpoints/aft_a2_v2_mix_p00_m1_a1",
    "m05_v2": PROJECT_ROOT / "runs/checkpoints/aft_m05_v2_mix_p05_m1_a1",
    "m15_v2": PROJECT_ROOT / "runs/checkpoints/aft_m15_v2_mix_p15_m1_a1",
    "a3_v2": PROJECT_ROOT / "runs/checkpoints/aft_a3_v2_mix_p30_m1_a1",
    "run1": PROJECT_ROOT / "runs/checkpoints/aft_run1_raw_mix_p100_m1_a1",
    "run2": PROJECT_ROOT / "runs/checkpoints/aft_run2_msm25_mix_p100_m1_a1",
    "run3": PROJECT_ROOT / "runs/checkpoints/aft_run3_msm25_mix_p90_m1_a1",
    "run4": PROJECT_ROOT / "runs/checkpoints/aft_run4_msm25_mix_p80_m1_a1",
}

DEFAULT_ETHICS_TASKS = [
    "ethics_cm",
    "ethics_deontology",
    "ethics_justice",
    "ethics_utilitarianism",
    "ethics_virtue",
]


def run_lm_eval_ethics(
    checkpoint_path: Path,
    output_path: Path,
    tasks: list[str],
    batch_size: int | str = "auto",
    limit: int | None = None,
    backend: str = "hf",
) -> dict:
    """Run lm-eval CLI on a specific checkpoint for Hendrycks ETHICS tasks."""
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        backend,
        "--model_args",
        f"pretrained={checkpoint_path},dtype=bfloat16",
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print("\n" + "=" * 70)
    print(f"Running Hendrycks ETHICS lm-eval on: {checkpoint_path.name}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Output: {output_path}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")

    start_time = time.time()
    env = os.environ.copy()
    env["HF_HOME"] = "/mnt/HDD_2TB/niraj/hf"
    env["HF_HUB_OFFLINE"] = "1"

    result = subprocess.run(cmd, env=env, check=False)
    elapsed = time.time() - start_time
    print(f"\nExecution finished in {elapsed:.2f} seconds with exit code {result.returncode}.")

    if output_path.exists():
        if output_path.is_file():
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif output_path.is_dir():
            json_files = list(output_path.glob("**/*.json"))
            if json_files:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    return json.load(f)

    return {"status": "completed", "exit_code": result.returncode}


def parse_ethics_results(results_dir: Path) -> dict:
    """Parse all JSON result files in results/ethics and build a summary table."""
    summary = {}
    json_files = list(results_dir.glob("*.json"))

    for jf in json_files:
        if jf.name == "ethics_summary.json":
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        res = data.get("results", {})
        if not res:
            continue

        model_name = jf.stem.replace("ethics_", "").replace("_results", "")
        import re
        model_name = re.sub(r"_\d{4}-\d{2}-\d{2}T.*$", "", model_name)
        summary[model_name] = {}

        subtask_scores = []
        for task in DEFAULT_ETHICS_TASKS:
            if task in res:
                acc = res[task].get("acc,none") or res[task].get("acc")
                if acc is not None:
                    summary[model_name][task] = acc * 100.0
                    subtask_scores.append(acc * 100.0)

        if subtask_scores:
            summary[model_name]["ethics_macro_avg"] = sum(subtask_scores) / len(subtask_scores)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Hendrycks ETHICS Benchmark Runner")
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        help="Arm to benchmark (e.g. a0, a2, x2_5m, a2_v2, a3_v2, v2_clean, or all)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom checkpoint path (overrides --arm)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/ethics)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_ETHICS_TASKS),
        help="Comma-separated list of tasks",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size (e.g. auto, 8, 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit per task (for smoke testing)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only aggregate and display results without running",
    )

    args = parser.parse_args()
    target_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summary = parse_ethics_results(target_dir)
        summary_file = target_dir / "ethics_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("\n=== Hendrycks ETHICS Benchmark Summary ===")
        print(json.dumps(summary, indent=2))
        return

    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        out_file = target_dir / f"ethics_{ckpt_path.name}.json"
        run_lm_eval_ethics(ckpt_path, out_file, tasks_list, batch_size=args.batch_size, limit=args.limit)
        return

    # Select target arms
    targets = []
    arm_req = args.arm.lower()
    if arm_req in ["a0", "all", "generalization"]:
        targets.append(("a0", DEFAULT_CHECKPOINTS["a0"]))
    if arm_req in ["a1", "all"]:
        targets.append(("a1", DEFAULT_CHECKPOINTS["a1"]))
    if arm_req in ["a2", "all", "generalization"]:
        targets.append(("a2", DEFAULT_CHECKPOINTS["a2"]))
    if arm_req in ["a3", "all"]:
        targets.append(("a3", DEFAULT_CHECKPOINTS["a3"]))
    if arm_req in ["a5", "all"]:
        targets.append(("a5", DEFAULT_CHECKPOINTS["a5"]))
    if arm_req in ["x2_5m", "x2", "all", "generalization"]:
        targets.append(("x2_5m", DEFAULT_CHECKPOINTS["x2_5m"]))
    if arm_req in ["x2_10m", "x2", "all"]:
        targets.append(("x2_10m", DEFAULT_CHECKPOINTS["x2_10m"]))
    if arm_req in ["x2_20m", "x2", "all"]:
        targets.append(("x2_20m", DEFAULT_CHECKPOINTS["x2_20m"]))
    if arm_req in ["a2_v2", "v2_clean", "all", "generalization"]:
        targets.append(("a2_v2", DEFAULT_CHECKPOINTS["a2_v2"]))
    if arm_req in ["m05_v2", "v2_clean", "all"]:
        targets.append(("m05_v2", DEFAULT_CHECKPOINTS["m05_v2"]))
    if arm_req in ["m15_v2", "v2_clean", "all"]:
        targets.append(("m15_v2", DEFAULT_CHECKPOINTS["m15_v2"]))
    if arm_req in ["a3_v2", "v2_clean", "all", "generalization"]:
        targets.append(("a3_v2", DEFAULT_CHECKPOINTS["a3_v2"]))
    if arm_req in ["run1", "p_ablation", "all"]:
        targets.append(("run1", DEFAULT_CHECKPOINTS["run1"]))
    if arm_req in ["run2", "p_ablation", "all"]:
        targets.append(("run2", DEFAULT_CHECKPOINTS["run2"]))
    if arm_req in ["run3", "p_ablation", "all"]:
        targets.append(("run3", DEFAULT_CHECKPOINTS["run3"]))
    if arm_req in ["run4", "p_ablation", "all"]:
        targets.append(("run4", DEFAULT_CHECKPOINTS["run4"]))

    for arm_name, ckpt_path in targets:
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint for {arm_name} not found at {ckpt_path}. Skipping.")
            continue
        out_file = target_dir / f"ethics_{arm_name}_{ckpt_path.name}.json"
        existing_matches = [
            f for f in target_dir.glob(f"ethics_{arm_name}_{ckpt_path.name}*.json")
            if f.stat().st_size > 1000
        ]
        if existing_matches:
            print(f"Skipping {arm_name}: completed evaluation already exists ({existing_matches[0].name})")
            continue
        run_lm_eval_ethics(ckpt_path, out_file, tasks_list, batch_size=args.batch_size, limit=args.limit)

    # Generate summary after all runs
    summary = parse_ethics_results(target_dir)
    summary_file = target_dir / "ethics_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Hendrycks ETHICS Benchmark Complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
