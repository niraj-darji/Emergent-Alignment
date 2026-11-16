"""Rollout Generator for Welfare Benchmarks (ANIMA & MANTA).

Generates and caches n=5 completions per item at T=0.7 for all 5 arms:
- Arm A0: raw + SFT (no AFT mix)
- Arm A1: raw + FT (SFT + AFT-30%mix)
- Arm A2: raw + MSM + SFT (no AFT mix)
- Arm A3: raw + MSM + FT (SFT + AFT-30%mix)
- Arm A5: raw + CaML + FT (SFT + AFT-30%mix)

Saves full completions to results/phase2_llm_judge/rollouts/{arm}_rollouts.json
without overwriting or modifying any existing results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
ROLLOUTS_DIR = PROJECT_ROOT / "results" / "phase2_llm_judge" / "rollouts"
ROLLOUTS_DIR.mkdir(parents=True, exist_ok=True)

ANIMA_QUESTIONS_PATH = DATA_DIR / "anima" / "questions" / "train-00000-of-00001.parquet"
ANIMA_DIMENSIONS_PATH = DATA_DIR / "anima" / "dimensions" / "train-00000-of-00001.parquet"
MANTA_QUESTIONS_PATH = DATA_DIR / "manta" / "manta_questions.csv"

DEFAULT_CHECKPOINTS = {
    "a0": PROJECT_ROOT / "runs/checkpoints/aft_a0_mix_p00_m1_a1",
    "a1": PROJECT_ROOT / "runs/checkpoints/aft_a1_mix_p30_m1_a1",
    "a2": PROJECT_ROOT / "runs/checkpoints/aft_a2_mix_p00_m1_a1",
    "a3": PROJECT_ROOT / "runs/checkpoints/aft_a3_mix_p30_m1_a1",
    "a5": PROJECT_ROOT / "runs/checkpoints/aft_a5_mix_p30_m1_a1",
    "m05": PROJECT_ROOT / "runs/checkpoints/aft_m05_mix_p05_m1_a1",
    "m15": PROJECT_ROOT / "runs/checkpoints/aft_m15_mix_p15_m1_a1",
    "m_msm05": PROJECT_ROOT / "runs/checkpoints/aft_m_msm05_mix_p05_m1_a1",
    "m_msm15": PROJECT_ROOT / "runs/checkpoints/aft_m_msm15_mix_p15_m1_a1",
    "x2_5m": PROJECT_ROOT / "runs/checkpoints/aft_x2_5m_mix_p00_m1_a1",
    "x2_10m": PROJECT_ROOT / "runs/checkpoints/aft_x2_10m_mix_p00_m1_a1",
    "x2_20m": PROJECT_ROOT / "runs/checkpoints/aft_x2_20m_mix_p00_m1_a1",
    "a2_v2": PROJECT_ROOT / "runs/checkpoints/aft_a2_v2_mix_p00_m1_a1",
    "m05_v2": PROJECT_ROOT / "runs/checkpoints/aft_m05_v2_mix_p05_m1_a1",
    "m15_v2": PROJECT_ROOT / "runs/checkpoints/aft_m15_v2_mix_p15_m1_a1",
    "a3_v2": PROJECT_ROOT / "runs/checkpoints/aft_a3_v2_mix_p30_m1_a1",
}


def load_anima_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ANIMA questions and dimensions."""
    if not ANIMA_QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"ANIMA questions not found at {ANIMA_QUESTIONS_PATH}")
    if not ANIMA_DIMENSIONS_PATH.exists():
        raise FileNotFoundError(f"ANIMA dimensions not found at {ANIMA_DIMENSIONS_PATH}")
    df_q = pd.read_parquet(ANIMA_QUESTIONS_PATH)
    df_dim = pd.read_parquet(ANIMA_DIMENSIONS_PATH)
    return df_q, df_dim


def load_manta_data(limit: int | None = 100) -> pd.DataFrame:
    """Load MANTA questions dataset."""
    if not MANTA_QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"MANTA questions not found at {MANTA_QUESTIONS_PATH}")
    df = pd.read_csv(MANTA_QUESTIONS_PATH)
    if limit is not None and limit > 0:
        df = df.iloc[:limit].copy()
    return df


def format_chat_prompt(question: str) -> str:
    """Format single-turn user question with literal turn tokens and empty system prompt."""
    return f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int = 16,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    num_samples: int = 5,
    base_seed: int = 42,
    device: str = "cuda:0",
) -> list[list[str]]:
    """Generate num_samples independent completions per prompt."""
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_id = tokenizer.eos_token_id
    stop_ids = [tid for tid in [eot_id, eos_id] if tid is not None]

    all_responses_per_prompt: list[list[str]] = [[] for _ in range(len(prompts))]

    for sample_idx in range(num_samples):
        current_seed = base_seed + sample_idx
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        print(f"  Generating sample {sample_idx + 1}/{num_samples} (seed={current_seed})...")

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    "eos_token_id": stop_ids,
                }
                if temperature > 0.0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 1.0
                else:
                    gen_kwargs["do_sample"] = False

                outputs = model.generate(**inputs, **gen_kwargs)

            input_lengths = inputs["input_ids"].shape[1]
            for prompt_offset, out in enumerate(outputs):
                global_idx = i + prompt_offset
                generated_ids = out[input_lengths:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                all_responses_per_prompt[global_idx].append(text)

    return all_responses_per_prompt


def generate_rollouts_for_arm(
    arm: str,
    checkpoint_path: Path,
    anima_limit: int | None = None,
    manta_limit: int | None = 100,
    num_samples: int = 5,
    batch_size: int = 16,
    device: str = "cuda:0",
    output_dir: Path | None = None,
) -> Path:
    """Generate and cache all rollouts for a single arm."""
    save_dir = output_dir or ROLLOUTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"{arm}_rollouts.json"
    if out_file.exists() and out_file.stat().st_size > 1000:
        print(f"Rollouts for {arm} already exist at {out_file}. Skipping generation.")
        return out_file

    print(f"\n========================================================")
    print(f"Generating Rollouts for: {arm} ({checkpoint_path.name})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples per item: {num_samples} (T=0.7)")
    print(f"========================================================\n")

    t_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()

    # 1. ANIMA
    print(f"--- Generating ANIMA (115 scenarios, n={num_samples}) ---")
    df_anima_q, _ = load_anima_data()
    if anima_limit is not None and anima_limit > 0:
        df_anima_q = df_anima_q.iloc[:anima_limit].copy()

    anima_items: list[dict[str, Any]] = []
    anima_prompts: list[str] = []
    for _, row in df_anima_q.iterrows():
        q_text = row["question"]
        tags_list = list(row["tags"]) if isinstance(row["tags"], (list, tuple, torch.Tensor, (lambda: None).__class__)) or hasattr(row["tags"], "__iter__") else []
        anima_items.append({
            "id": int(row["id"]),
            "question": str(q_text),
            "tags": [str(t) for t in tags_list],
        })
        anima_prompts.append(format_chat_prompt(str(q_text)))

    anima_responses = generate_completions(
        model,
        tokenizer,
        anima_prompts,
        batch_size=batch_size,
        max_new_tokens=512,
        temperature=0.7,
        num_samples=num_samples,
        device=device,
    )
    for idx, resps in enumerate(anima_responses):
        anima_items[idx]["responses"] = resps

    # 2. MANTA
    print(f"\n--- Generating MANTA (Sample: {manta_limit}, n={num_samples}) ---")
    df_manta = load_manta_data(limit=manta_limit)
    manta_items: list[dict[str, Any]] = []
    manta_prompts: list[str] = []
    for _, row in df_manta.iterrows():
        q_text = str(row["question"])
        press_raw = str(row["pressure"])
        press_clean = press_raw.replace("[", "").replace("]", "").replace("'", "")
        press_list = [p.strip() for p in press_clean.split(",") if p.strip()]
        animals_val = str(row["animals"]) if pd.notna(row["animals"]) else ""

        manta_items.append({
            "id": int(row["id"]),
            "question": q_text,
            "pressure": press_list,
            "animals": animals_val,
        })
        manta_prompts.append(format_chat_prompt(q_text))

    manta_responses = generate_completions(
        model,
        tokenizer,
        manta_prompts,
        batch_size=batch_size,
        max_new_tokens=512,
        temperature=0.7,
        num_samples=num_samples,
        device=device,
    )
    for idx, resps in enumerate(manta_responses):
        manta_items[idx]["responses"] = resps

    # Cleanup model from GPU
    del model
    del tokenizer
    torch.cuda.empty_cache()

    duration = time.time() - t_start
    print(f"Completed generation for {arm} in {duration:.1f} seconds.")

    out_data = {
        "arm": arm,
        "checkpoint": str(checkpoint_path),
        "num_samples": num_samples,
        "temperature": 0.7,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(duration, 1),
        "anima": anima_items,
        "manta": manta_items,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved rollouts to: {out_file}\n")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Generate and cache rollouts for ANIMA & MANTA")
    parser.add_argument(
        "--arm",
        choices=["a0", "a1", "a2", "a3", "a5", "m05", "m15", "m_msm05", "m_msm15", "mgrid", "x2_5m", "x2_10m", "x2_20m", "x2", "a2_v2", "m05_v2", "m15_v2", "a3_v2", "v2_clean", "all"],
        default="all",
        help="Arm to generate rollouts for",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save rollouts to (overrides default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation (default: 16)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per item (default: 5)",
    )

    args = parser.parse_args()

    targets: list[tuple[str, Path]] = []
    if args.arm in ["a0", "all"]:
        targets.append(("a0", DEFAULT_CHECKPOINTS["a0"]))
    if args.arm in ["a1", "all"]:
        targets.append(("a1", DEFAULT_CHECKPOINTS["a1"]))
    if args.arm in ["a2", "all"]:
        targets.append(("a2", DEFAULT_CHECKPOINTS["a2"]))
    if args.arm in ["a3", "all"]:
        targets.append(("a3", DEFAULT_CHECKPOINTS["a3"]))
    if args.arm in ["a5", "all"]:
        targets.append(("a5", DEFAULT_CHECKPOINTS["a5"]))
    if args.arm in ["m05", "mgrid", "all"]:
        targets.append(("m05", DEFAULT_CHECKPOINTS["m05"]))
    if args.arm in ["m15", "mgrid", "all"]:
        targets.append(("m15", DEFAULT_CHECKPOINTS["m15"]))
    if args.arm in ["m_msm05", "mgrid", "all"]:
        targets.append(("m_msm05", DEFAULT_CHECKPOINTS["m_msm05"]))
    if args.arm in ["m_msm15", "mgrid", "all"]:
        targets.append(("m_msm15", DEFAULT_CHECKPOINTS["m_msm15"]))
    if args.arm in ["x2_5m", "x2", "all"]:
        targets.append(("x2_5m", DEFAULT_CHECKPOINTS["x2_5m"]))
    if args.arm in ["x2_10m", "x2", "all"]:
        targets.append(("x2_10m", DEFAULT_CHECKPOINTS["x2_10m"]))
    if args.arm in ["x2_20m", "x2", "all"]:
        targets.append(("x2_20m", DEFAULT_CHECKPOINTS["x2_20m"]))
    if args.arm in ["a2_v2", "v2_clean"]:
        targets.append(("a2_v2", DEFAULT_CHECKPOINTS["a2_v2"]))
    if args.arm in ["m05_v2", "v2_clean"]:
        targets.append(("m05_v2", DEFAULT_CHECKPOINTS["m05_v2"]))
    if args.arm in ["m15_v2", "v2_clean"]:
        targets.append(("m15_v2", DEFAULT_CHECKPOINTS["m15_v2"]))
    if args.arm in ["a3_v2", "v2_clean"]:
        targets.append(("a3_v2", DEFAULT_CHECKPOINTS["a3_v2"]))

    out_dir_path = Path(args.output_dir) if args.output_dir else None

    for arm_name, ckpt_path in targets:
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping {arm_name}.")
            continue
        generate_rollouts_for_arm(
            arm=arm_name,
            checkpoint_path=ckpt_path,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            output_dir=out_dir_path,
        )


if __name__ == "__main__":
    main()
