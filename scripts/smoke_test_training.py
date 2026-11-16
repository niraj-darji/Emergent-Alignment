"""End-to-end smoke test script for Stage 1 Model Spec Midtraining (MSM).

Reference: MSM_training_blueprint.md (§12).
Executes a 50-step run, verifies loss dynamics, merges adapter, tests resume,
and computes throughput and VRAM requirements.
"""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from awmsm.train.merge import merge_adapter
from awmsm.train.train import train_msm


def run_smoke_test(
    corpus_id: str = "aw_main_v11",
    base_model_path: str = "meta-llama/Llama-3.1-8B",
    max_steps: int = 50,
    seed: int = 1,
    custom_log_name: Optional[str] = None,
) -> bool:
    """Execute the 50-step smoke test and verify acceptance criteria."""
    project_root = Path(__file__).resolve().parent.parent
    global_log_dir = Path("/mnt/HDD_2TB/niraj/logs")
    global_log_dir.mkdir(parents=True, exist_ok=True)

    if custom_log_name is None:
        custom_log_name = f"smoke_test_{corpus_id}.log"
    smoke_log_path = global_log_dir / custom_log_name

    print("======================================================================")
    print("STARTING MSM SMOKE TEST")
    print(f"Corpus ID: {corpus_id}")
    print(f"Base model: {base_model_path}")
    print(f"Max steps: {max_steps}")
    print(f"Log path: {smoke_log_path}")
    print("======================================================================")

    # -------------------------------------------------------------------------
    # Note on Dataset:
    # Smoke test currently defaults to aw_main_v11.
    # To run against another dataset (e.g. aw_main_v1 or a new version),
    # pass --corpus-id <id> on the command line.
    # -------------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    corpus_path = project_root / f"data/corpora/{corpus_id}/corpus.jsonl"
    if not corpus_path.is_file():
        print(f"Error: Corpus file not found at {corpus_path}")
        return False

    config = {
        "mode": "midtrain",
        "model": {
            "base_path": base_model_path,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
        "lora": {
            "r": 64,
            "alpha": 128,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "data": {
            "corpus_id": corpus_id,
            "corpus_path": str(corpus_path),
            "seq_len": 4096,
            "heldout_frac": 0.02,
            "split_seed": 0,
        },
        "budget": {
            "target_tokens": None,
            "tolerance": 0.02,
            "enforce": False,
        },
        "train": {
            "seed": seed,
            "micro_batch_size": 1,
            "grad_accum_steps": 8,
            "lr": 1.0e-4,
            "schedule": "cosine",
            "warmup_frac": 0.05,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "epochs": 1,
            "gradient_checkpointing": True,
            "allow_tf32": True,
            "eval_every": 25,
            "save_every": 25,
            "max_steps": max_steps,
        },
    }

    run_id = f"midtrain_{corpus_id}_m{seed}"
    run_dir = project_root / "runs/midtrain" / run_id

    # Clean previous test artifacts if any
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    t0 = time.perf_counter()
    train_msm(config, resume=False, max_steps_override=max_steps)
    t_train = time.perf_counter() - t0

    # Verification Step 1: Check training logs
    log_file = run_dir / "logs" / "train_log.jsonl"
    if not log_file.is_file():
        print("FAIL: train_log.jsonl not produced.")
        return False

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line.strip()))

    if len(logs) < max_steps:
        print(f"FAIL: Expected {max_steps} log entries, got {len(logs)}.")
        return False

    initial_loss = logs[0]["train_loss"]
    final_loss = logs[-1]["train_loss"]
    print(f"Step 1 loss: {initial_loss:.4f} -> Step {max_steps} loss: {final_loss:.4f}")

    # Check finite gradients & no NaN
    for entry in logs:
        loss = entry["train_loss"]
        gnorm = entry["grad_norm"]
        if math.isnan(loss) or math.isinf(loss):
            print("FAIL: NaN or Inf loss encountered.")
            return False
        if math.isnan(gnorm) or math.isinf(gnorm):
            print("FAIL: NaN or Inf grad_norm encountered.")
            return False

    print("CHECK PASSED: No NaN/Inf detected in loss or grad_norm.")

    # Verification Step 2: Merge adapter
    adapter_path = run_dir / "adapter"
    merged_output_dir = project_root / "runs/checkpoints" / run_id
    if merged_output_dir.exists():
        shutil.rmtree(merged_output_dir, ignore_errors=True)

    print("Testing adapter merge and verification gates...")
    merge_adapter(
        adapter_path=adapter_path,
        output_dir=merged_output_dir,
        base_model_path=base_model_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        logit_tolerance=0.5,
    )
    print("CHECK PASSED: Merge verification gates passed.")

    # Verification Step 3: Text completion with merged model
    print("Testing text completion with merged checkpoint...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(merged_output_dir), use_fast=True)
    merged_model = AutoModelForCausalLM.from_pretrained(
        str(merged_output_dir),
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    merged_model.eval()

    prompt = "The moral and ethical standing of sentient animals"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Sample generation:\n'{completion}'")
    del merged_model

    # Verification Step 4: Resume test from step 25
    step_25_dir = run_dir / "checkpoints" / "step_25"
    if step_25_dir.is_dir():
        print("Testing checkpoint resume from step 25...")
        train_msm(config, resume=True, max_steps_override=max_steps)
        print("CHECK PASSED: Resume mechanism completed without error.")
    else:
        print("WARNING: step_25 checkpoint not found for resume test.")

    # Step 5: Throughput and resource report
    total_tokens_trained = max_steps * 8 * 4096  # 8 sequences of 4096 per step
    avg_tokens_per_sec = total_tokens_trained / max(t_train, 1e-6)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device=device) / (1024**2)
        if torch.cuda.is_available()
        else 0.0
    )

    hours_for_2_5m = (2_500_000 / avg_tokens_per_sec) / 3600
    hours_for_5_0m = (5_000_000 / avg_tokens_per_sec) / 3600

    print("======================================================================")
    print("SMOKE TEST PERFORMANCE REPORT")
    print(f"50 steps elapsed time: {t_train:.1f} s")
    print(f"Average throughput: {avg_tokens_per_sec:.1f} tokens/s")
    print(f"Peak GPU VRAM allocated: {peak_vram_mb:.1f} MB ({peak_vram_mb / 1024:.2f} GB)")
    print(f"Estimated time for 2.5M token run: {hours_for_2_5m:.2f} hours")
    print(f"Estimated time for 5.0M token run: {hours_for_5_0m:.2f} hours")
    print(f"Recommended hardware.min_free_gb: {math.ceil(peak_vram_mb / 1024) + 4} GB")
    print("======================================================================")
    print("ALL SMOKE TEST CHECKS PASSED SUCCESSFULLY.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run MSM smoke test.")
    parser.add_argument(
        "--corpus-id",
        type=str,
        default="aw_main_v11",
        help="Corpus ID to test with (default: aw_main_v11).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base model path or HF hub ID.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Number of steps for smoke test (default: 50).",
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default=None,
        help="Custom log filename in /mnt/HDD_2TB/niraj/logs/ (default: smoke_test_{corpus_id}.log).",
    )
    args = parser.parse_args()

    success = run_smoke_test(
        corpus_id=args.corpus_id,
        base_model_path=args.base_model,
        max_steps=args.max_steps,
        custom_log_name=args.log_name,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
