"""Anthropic HH Harmless Benchmark Runner.

Evaluates models on the Anthropic Helpful & Harmless (HH-RLHF) harmless-base dataset.
Measures the model's preference for harmless responses over harmful responses
via token conditional log-likelihood (pairwise preference accuracy).

Metrics:
- harmless_pref_acc: % of pairs where total log P(chosen | prompt) > log P(rejected | prompt)
- harmless_norm_acc: % of pairs where length-normalized log P(chosen) > log P(rejected)
- mean_margin: average length-normalized log-likelihood margin (chosen - rejected)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "anthropic_hh"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "benchmarks" / "anthropic_hh_harmless" / "harmless_test.parquet"

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


def extract_prompt_and_completion(text: str) -> tuple[str, str]:
    """Split dialogue into prompt and the final Assistant completion."""
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx != -1:
        prompt = text[: idx + len(marker)]
        comp = text[idx + len(marker) :]
    else:
        prompt = ""
        comp = text
    return prompt, comp


def compute_completion_logp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    max_length: int = 2048,
) -> tuple[float, float, int]:
    """Compute exact total log P(completion | prompt) and length-normalized log P."""
    full_text = prompt + completion

    # Tokenize prompt and full sequence
    prompt_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)

    input_ids = full_enc.input_ids.to(device)
    prompt_len = prompt_enc.input_ids.shape[1]
    seq_len = input_ids.shape[1]

    if seq_len <= prompt_len:
        # Completion was completely truncated
        return -9999.0, -9999.0, 0

    comp_tokens = seq_len - prompt_len

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape [1, seq_len, vocab_size]

        # Shift logits and targets: token t is predicted by logits at t-1
        shift_logits = logits[0, prompt_len - 1 : seq_len - 1, :]  # shape [comp_tokens, vocab]
        shift_labels = input_ids[0, prompt_len:seq_len]  # shape [comp_tokens]

        # Cross entropy with reduction='none' gives -log P(target)
        neg_logps = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        total_logp = -neg_logps.sum().item()
        norm_logp = total_logp / comp_tokens

    return total_logp, norm_logp, comp_tokens


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_path: Path,
    output_file: Path,
    limit: int | None = None,
    device: str = "cuda:0",
    max_length: int = 2048,
) -> dict[str, Any]:
    """Evaluate a checkpoint on Anthropic HH Harmless dataset."""
    print("\n" + "=" * 70)
    print(f"Evaluating Anthropic HH Harmless on: {checkpoint_path.name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_file}")
    print("=" * 70)

    # Load data
    table = pq.read_table(data_path)
    df = table.to_pandas()
    if limit is not None:
        df = df.iloc[:limit]

    total_samples = len(df)
    print(f"Loaded {total_samples} test pairs.")

    # Load model and tokenizer
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {dev}...")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        device_map=None,
    ).to(dev)
    model.eval()

    chosen_wins = 0
    norm_chosen_wins = 0
    margins = []
    details = []

    start_time = time.time()
    for idx in tqdm(range(total_samples), desc="Evaluating Harmless Pairs"):
        row = df.iloc[idx]
        chosen_text = row["chosen"]
        rejected_text = row["rejected"]

        prompt_c, comp_c = extract_prompt_and_completion(chosen_text)
        prompt_r, comp_r = extract_prompt_and_completion(rejected_text)

        # Standard Anthropic prompt
        prompt = prompt_c if prompt_c else prompt_r

        logp_c, norm_c, len_c = compute_completion_logp(model, tokenizer, prompt, comp_c, dev, max_length)
        logp_r, norm_r, len_r = compute_completion_logp(model, tokenizer, prompt, comp_r, dev, max_length)

        c_win = logp_c > logp_r
        norm_win = norm_c > norm_r
        margin = norm_c - norm_r

        if c_win:
            chosen_wins += 1
        if norm_win:
            norm_chosen_wins += 1
        margins.append(margin)

        details.append({
            "idx": idx,
            "chosen_win": bool(c_win),
            "norm_chosen_win": bool(norm_win),
            "logp_chosen": round(logp_c, 4),
            "logp_rejected": round(logp_r, 4),
            "norm_logp_chosen": round(norm_c, 4),
            "norm_logp_rejected": round(norm_r, 4),
            "len_chosen": len_c,
            "len_rejected": len_r,
            "margin": round(margin, 4),
        })

    elapsed = time.time() - start_time
    acc = (chosen_wins / total_samples) * 100.0
    norm_acc = (norm_chosen_wins / total_samples) * 100.0
    mean_margin = float(np.mean(margins))

    print("\n" + "-" * 50)
    print(f"Results for {checkpoint_path.name}:")
    print(f"  Samples: {total_samples}")
    print(f"  Harmless Preference Accuracy (raw):  {acc:.2f}% ({chosen_wins}/{total_samples})")
    print(f"  Harmless Preference Accuracy (norm): {norm_acc:.2f}% ({norm_chosen_wins}/{total_samples})")
    print(f"  Mean Margin (log P/tok):             {mean_margin:.4f}")
    print(f"  Elapsed Time:                        {elapsed:.2f}s ({total_samples/elapsed:.1f} samples/s)")
    print("-" * 50 + "\n")

    result = {
        "model": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "total_samples": total_samples,
        "harmless_pref_acc_raw": round(acc, 2),
        "harmless_pref_acc_norm": round(norm_acc, 2),
        "mean_margin": round(mean_margin, 4),
        "elapsed_seconds": round(elapsed, 2),
        "samples_per_second": round(total_samples / elapsed, 2),
        "details": details,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Anthropic HH Harmless Benchmark Runner")
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        help="Arm to evaluate (e.g. a0, a2, x2_5m, a2_v2, a3_v2, generalization, or all)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom checkpoint path (overrides --arm)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to harmless parquet dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Output directory (default: results/anthropic_hh)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit (for smoke testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (e.g. cuda:0, cpu)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum token sequence length",
    )

    args = parser.parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        out_file = output_dir / f"anthropic_hh_{ckpt_path.name}.json"
        evaluate_checkpoint(ckpt_path, data_path, out_file, limit=args.limit, device=args.device, max_length=args.max_length)
        return

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

    summary = {}
    for arm_name, ckpt_path in targets:
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint {arm_name} not found at {ckpt_path}. Skipping.")
            continue
        out_file = output_dir / f"anthropic_hh_{arm_name}_{ckpt_path.name}.json"
        if out_file.exists() and out_file.stat().st_size > 1000:
            print(f"Skipping {arm_name}: already completed ({out_file.name})")
            with open(out_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            summary[arm_name] = {
                "model": ckpt_path.name,
                "raw_acc": res["harmless_pref_acc_raw"],
                "norm_acc": res["harmless_pref_acc_norm"],
                "mean_margin": res["mean_margin"],
            }
            continue

        res = evaluate_checkpoint(ckpt_path, data_path, out_file, limit=args.limit, device=args.device, max_length=args.max_length)
        summary[arm_name] = {
            "model": ckpt_path.name,
            "raw_acc": res["harmless_pref_acc_raw"],
            "norm_acc": res["harmless_pref_acc_norm"],
            "mean_margin": res["mean_margin"],
        }

    # Save summary table
    summary_path = output_dir / "anthropic_hh_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Anthropic HH Harmless Summary:")
    print(json.dumps(summary, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
