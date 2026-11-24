"""Phase 1: General Capabilities & Safety Benchmark Runner.

Runs lm-evaluation-harness on:
- MMLU (5-shot)
- HellaSwag (0-shot)
- BBQ (social bias)
- HHH Alignment (helpfulness, honesty, harmlessness)
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
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"
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

DEFAULT_TASKS = [
    "mmlu",
    "hellaswag",
    "bbq",
    "bigbench_hhh_alignment_multiple_choice",
]


def run_lm_eval(
    checkpoint_path: Path,
    output_path: Path,
    tasks: list[str],
    batch_size: int | str = "auto",
    limit: int | None = None,
    backend: str = "hf",
) -> dict:
    """Run lm-eval CLI on a specific model checkpoint."""
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

    print(f"\n========================================================")
    print(f"Running Phase 1 lm-eval on: {checkpoint_path.name}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Output: {output_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================================\n")

    start_time = time.time()
    env = os.environ.copy()
    env["HF_HOME"] = "/mnt/HDD_2TB/niraj/hf"

    result = subprocess.run(cmd, env=env, check=False)
    elapsed = time.time() - start_time
    print(f"\nExecution finished in {elapsed:.2f} seconds with exit code {result.returncode}.")

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif output_path.is_dir():
        # lm-eval may save output in subdirectories
        json_files = list(output_path.glob("**/*.json"))
        if json_files:
            with open(json_files[0], "r", encoding="utf-8") as f:
                return json.load(f)

    return {"status": "completed", "exit_code": result.returncode}


def main():
    parser = argparse.ArgumentParser(description="Phase 1: General Capabilities & Safety Benchmark Runner")
    parser.add_argument(
        "--arm",
        choices=["a0", "a1", "a2", "a3", "a5", "both", "x2_5m", "x2_10m", "x2_20m", "x2", "a2_v2", "m05_v2", "m15_v2", "a3_v2", "v2_clean", "run1", "run2", "run3", "run4", "p_ablation", "all"],
        default="all",
        help="Arm to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for general benchmark JSONs (default: results/phase1)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom checkpoint directory path (overrides --arm)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated list of lm-eval tasks",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size (e.g. auto, 4, 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit per task (for testing)",
    )

    args = parser.parse_args()
    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    target_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        out_file = target_dir / f"{ckpt_path.name}_general.json"
        run_lm_eval(ckpt_path, out_file, tasks_list, batch_size=args.batch_size, limit=args.limit)
        return

    targets = []
    if args.arm in ["a0", "both", "all"]:
        targets.append(("a0", DEFAULT_CHECKPOINTS["a0"]))
    if args.arm in ["a1", "all"]:
        targets.append(("a1", DEFAULT_CHECKPOINTS["a1"]))
    if args.arm in ["a2", "both", "all"]:
        targets.append(("a2", DEFAULT_CHECKPOINTS["a2"]))
    if args.arm in ["a3", "all"]:
        targets.append(("a3", DEFAULT_CHECKPOINTS["a3"]))
    if args.arm in ["a5", "all"]:
        targets.append(("a5", DEFAULT_CHECKPOINTS["a5"]))
    if args.arm in ["x2_5m", "x2"]:
        targets.append(("x2_5m", DEFAULT_CHECKPOINTS["x2_5m"]))
    if args.arm in ["x2_10m", "x2"]:
        targets.append(("x2_10m", DEFAULT_CHECKPOINTS["x2_10m"]))
    if args.arm in ["x2_20m", "x2"]:
        targets.append(("x2_20m", DEFAULT_CHECKPOINTS["x2_20m"]))
    if args.arm in ["a2_v2", "v2_clean"]:
        targets.append(("a2_v2", DEFAULT_CHECKPOINTS["a2_v2"]))
    if args.arm in ["m05_v2", "v2_clean"]:
        targets.append(("m05_v2", DEFAULT_CHECKPOINTS["m05_v2"]))
    if args.arm in ["m15_v2", "v2_clean"]:
        targets.append(("m15_v2", DEFAULT_CHECKPOINTS["m15_v2"]))
    if args.arm in ["a3_v2", "v2_clean"]:
        targets.append(("a3_v2", DEFAULT_CHECKPOINTS["a3_v2"]))
    if args.arm in ["run1", "p_ablation", "all"]:
        targets.append(("run1", DEFAULT_CHECKPOINTS["run1"]))
    if args.arm in ["run2", "p_ablation", "all"]:
        targets.append(("run2", DEFAULT_CHECKPOINTS["run2"]))
    if args.arm in ["run3", "p_ablation", "all"]:
        targets.append(("run3", DEFAULT_CHECKPOINTS["run3"]))
    if args.arm in ["run4", "p_ablation", "all"]:
        targets.append(("run4", DEFAULT_CHECKPOINTS["run4"]))

    for arm_name, ckpt_path in targets:
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint for {arm_name} not found at {ckpt_path}. Skipping.")
            continue
        existing = list(target_dir.glob(f"*{arm_name}*.json"))
        if existing and any(f.stat().st_size > 1000 for f in existing):
            print(f"Skipping {arm_name}: completed evaluation already exists ({existing[0].name})")
            continue
        out_file = target_dir / f"phase1_{arm_name}_{ckpt_path.name}.json"
        run_lm_eval(ckpt_path, out_file, tasks_list, batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()

