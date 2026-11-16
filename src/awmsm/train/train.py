"""Main training engine for Stage 1: Model Spec Midtraining (MSM).

Reference: AW_MSM_Methodology.md (§8.1-§8.3) and MSM_training_blueprint.md (§4-§6, §8, §10).
"""

import argparse
import csv
import datetime
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import yaml

from awmsm.train.dataset import get_midtrain_datasets
from awmsm.train.aft_dataset import get_aft_datasets


def set_seed(seed: int, allow_tf32: bool = True) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def get_git_commit() -> str:
    """Get the current git commit hash, or 'unknown' if not available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    import hashlib
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_heldout(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_eval_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate mean cross-entropy loss and perplexity on the held-out set."""
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_eval_batches is not None and i >= max_eval_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs.logits

            # Shift for causal LM: predict next token
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)

            loss_sum = criterion(shift_logits, shift_labels).item()
            num_tokens = (shift_labels != -100).sum().item()

            total_loss_sum += loss_sum
            total_tokens += num_tokens

    model.train()
    if total_tokens == 0:
        return 0.0, 0.0

    mean_loss = total_loss_sum / total_tokens
    perplexity = math.exp(min(mean_loss, 100.0))
    return mean_loss, perplexity


def save_checkpoint(
    save_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    seed: int,
    tokens_trained: int,
) -> None:
    """Save a resumable step checkpoint containing adapter, optimizer, and RNG state."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir / "adapter"))

    train_state = {
        "step": step,
        "seed": seed,
        "tokens_trained": tokens_trained,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    torch.save(train_state, save_dir / "train_state.pt")
    torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")


def load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> Dict[str, Any]:
    """Resume training state from a saved checkpoint."""
    train_state = torch.load(checkpoint_dir / "train_state.pt", weights_only=False)
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", weights_only=True))
    scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", weights_only=True))

    torch.set_rng_state(train_state["torch_rng"])
    if torch.cuda.is_available() and train_state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(train_state["cuda_rng"])
    np.random.set_state(train_state["numpy_rng"])
    random.setstate(train_state["python_rng"])

    return train_state


def train_msm(config: Dict[str, Any], resume: bool = False, max_steps_override: Optional[int] = None) -> Path:
    """Execute Stage 1 Model Spec Midtraining (MSM)."""
    start_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    budget_cfg = config.get("budget", {})

    seed = int(train_cfg.get("seed", 1))
    allow_tf32 = bool(train_cfg.get("allow_tf32", True))
    set_seed(seed, allow_tf32=allow_tf32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Corpus configuration note:
    # Default corpus is currently configured to:
    #   corpus_id: aw_main_v11
    #   corpus_path: data/corpora/aw_main_v11/corpus.jsonl
    # To use a newly generated / refined dataset in the future, pass
    # --corpus-id and --corpus-path on CLI or update configs/train_midtrain.yaml.
    # -------------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[3]
    corpus_id = data_cfg.get("corpus_id", "aw_main_v11")
    corpus_path = Path(data_cfg.get("corpus_path", f"data/corpora/{corpus_id}/corpus.jsonl"))
    if not corpus_path.is_absolute():
        corpus_path = project_root / corpus_path

    run_id = f"midtrain_{corpus_id}_m{seed}"
    run_dir = project_root / "runs/midtrain" / run_id
    adapter_dir = run_dir / "adapter"
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"

    for d in [run_dir, adapter_dir, logs_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)

    base_model_path = model_cfg.get("base_path", "meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and pack dataset
    train_dataset, heldout_dataset, stats = get_midtrain_datasets(
        corpus_path=corpus_path,
        corpus_id=corpus_id,
        tokenizer=tokenizer,
        seq_len=int(data_cfg.get("seq_len", 4096)),
        heldout_frac=float(data_cfg.get("heldout_frac", 0.02)),
        split_seed=int(data_cfg.get("split_seed", 0)),
        train_seed=seed,
        target_tokens=budget_cfg.get("target_tokens"),
        tolerance=float(budget_cfg.get("tolerance", 0.02)),
        enforce_budget=bool(budget_cfg.get("enforce", False)),
    )

    # Write stats.json into run_dir
    with open(run_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    micro_batch_size = int(train_cfg.get("micro_batch_size", 1))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 8))
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=False,  # Already shuffled deterministically during packing
    )
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
    )

    total_chunks = len(train_dataset)
    total_optimizer_steps = math.ceil(total_chunks / grad_accum_steps)
    if max_steps_override is not None:
        max_steps = max_steps_override
    elif train_cfg.get("max_steps") is not None:
        max_steps = int(train_cfg["max_steps"])
    else:
        max_steps = total_optimizer_steps

    # Model setup
    dtype = torch.bfloat16 if model_cfg.get("dtype") == "bfloat16" else torch.float32
    attn_impl = model_cfg.get("attn_implementation", "sdpa")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map={"": device},
    )

    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # LoRA setup
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 128)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Optimizer and Scheduler
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    warmup_frac = float(train_cfg.get("warmup_frac", 0.05))
    warmup_steps = math.ceil(warmup_frac * max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    start_step = 0
    tokens_trained_counter = 0

    if resume:
        # Find latest step checkpoint
        existing_checkpoints = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        if existing_checkpoints:
            latest_ckpt = existing_checkpoints[-1]
            train_state = load_checkpoint(latest_ckpt, model, optimizer, scheduler)
            start_step = train_state["step"]
            tokens_trained_counter = train_state.get("tokens_trained", start_step * grad_accum_steps * 4096)
            print(f"Resumed from {latest_ckpt} at step {start_step}.")
        else:
            print("No checkpoints found to resume from. Starting from step 0.")

    # Logging setup
    jsonl_log_path = logs_dir / "train_log.jsonl"
    csv_log_path = logs_dir / "train_log.csv"
    csv_fields = [
        "step",
        "train_loss",
        "heldout_loss",
        "heldout_ppl",
        "lr",
        "grad_norm",
        "tokens_per_sec",
        "peak_vram_mb",
        "tokens_trained",
    ]
    if not resume or not csv_log_path.is_file():
        with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    eval_every = int(train_cfg.get("eval_every", 25))
    save_every = int(train_cfg.get("save_every", 50))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    # Evaluate at step 0 if starting fresh
    if start_step == 0 and len(heldout_dataset) > 0:
        h_loss, h_ppl = evaluate_heldout(model, heldout_loader, device)
        print(f"Initial held-out evaluation: loss={h_loss:.4f}, perplexity={h_ppl:.2f}")

    model.train()
    step = start_step
    data_iter = iter(train_loader)

    # Fast-forward dataloader if resumed
    if start_step > 0:
        for _ in range(start_step * grad_accum_steps):
            try:
                next(data_iter)
            except StopIteration:
                break

    optimizer.zero_grad()
    criterion_sum = nn.CrossEntropyLoss(reduction="sum")

    print(
        f"Starting MSM training for {run_id}: max_steps={max_steps}, "
        f"effective_batch_tokens={grad_accum_steps * 4096}, lr={lr}"
    )

    t_start = time.perf_counter()
    tokens_in_window = 0

    while step < max_steps:
        step += 1
        total_loss_sum = 0.0
        total_active_tokens = 0

        # Gradient accumulation loop
        for micro_idx in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs.logits

            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)

            active_tokens = (shift_labels != -100).sum().item()
            loss_sum = criterion_sum(shift_logits, shift_labels)

            # Accumulate loss normalized across the entire effective batch
            normalized_loss = loss_sum / (grad_accum_steps * max(active_tokens, 1))
            normalized_loss.backward()

            total_loss_sum += loss_sum.item()
            total_active_tokens += active_tokens

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        ).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        tokens_trained_counter += total_active_tokens
        tokens_in_window += total_active_tokens
        step_train_loss = total_loss_sum / max(total_active_tokens, 1)

        # Held-out evaluation
        h_loss: Optional[float] = None
        h_ppl: Optional[float] = None
        if (step % eval_every == 0 or step == max_steps) and len(heldout_dataset) > 0:
            h_loss, h_ppl = evaluate_heldout(model, heldout_loader, device)

        # Logging
        elapsed = time.perf_counter() - t_start
        tokens_per_sec = tokens_in_window / max(elapsed, 1e-6)
        t_start = time.perf_counter()
        tokens_in_window = 0

        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device=device) / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )
        current_lr = scheduler.get_last_lr()[0]

        log_entry = {
            "step": step,
            "train_loss": round(step_train_loss, 4),
            "heldout_loss": round(h_loss, 4) if h_loss is not None else None,
            "heldout_ppl": round(h_ppl, 2) if h_ppl is not None else None,
            "lr": current_lr,
            "grad_norm": round(grad_norm, 4),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "tokens_trained": tokens_trained_counter,
        }

        with open(jsonl_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(log_entry)

        eval_str = f" | heldout_loss={h_loss:.4f} (ppl={h_ppl:.2f})" if h_loss is not None else ""
        print(
            f"Step {step}/{max_steps} | loss={step_train_loss:.4f}{eval_str} | "
            f"lr={current_lr:.2e} | grad_norm={grad_norm:.2f} | "
            f"tok/s={tokens_per_sec:.1f} | vram={peak_vram_mb:.0f}MB"
        )

        # Rolling checkpoint
        if step % save_every == 0 and step < max_steps:
            step_ckpt_dir = checkpoints_dir / f"step_{step}"
            save_checkpoint(
                save_dir=step_ckpt_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                seed=seed,
                tokens_trained=tokens_trained_counter,
            )

    # Save final LoRA adapter
    print(f"Saving final adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    end_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Build manifest.json (§8 of blueprint)
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "mode": "midtrain",
        "corpus_id": corpus_id,
        "corpus_path": str(corpus_path),
        "corpus_sha256": compute_file_sha256(corpus_path) if corpus_path.is_file() else None,
        "dataset_cache_hash": stats.get("cache_hash"),
        "base_model": base_model_path,
        "git_commit": get_git_commit(),
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "steps_completed": step,
        "tokens_trained": tokens_trained_counter,
        "seed": seed,
        "allow_tf32": allow_tf32,
        "attn_implementation": attn_impl,
        "attention_across_docs": True,
        "hardware": {
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1)
            if torch.cuda.is_available()
            else 0.0,
        },
        "config": config,
        "stats": stats,
    }

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Midtraining completed for {run_id}. Manifest saved to {run_dir / 'manifest.json'}.")
    return run_dir


def train_aft(
    config: Dict[str, Any],
    resume: bool = False,
    max_steps_override: Optional[int] = None,
) -> Path:
    """Execute Stage 2 Alignment Fine-Tuning (AFT)."""
    start_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})

    seed = int(train_cfg.get("seed", 1))
    allow_tf32 = bool(train_cfg.get("allow_tf32", True))
    set_seed(seed, allow_tf32=allow_tf32)

    arm = data_cfg.get("arm", "a2")
    corpus_id = data_cfg.get("corpus_id", "aw_main_v1")
    run_id = f"aft_{arm}_{corpus_id}_m1_a{seed}"

    project_root = Path(__file__).resolve().parents[3]
    run_dir = project_root / "runs/aft" / run_id
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    adapter_dir = run_dir / "adapter"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    jsonl_log_path = logs_dir / "train_log.jsonl"
    csv_log_path = logs_dir / "train_log.csv"

    # Base model & tokenizer loading
    base_model_path_str = model_cfg.get(
        "base_path", "runs/checkpoints/midtrain_aw_main_v1_m1"
    )
    if (project_root / base_model_path_str).exists():
        base_model_path = str(project_root / base_model_path_str)
    elif Path(base_model_path_str).is_absolute() and Path(base_model_path_str).exists():
        base_model_path = base_model_path_str
    else:
        # Hugging Face hub model ID (e.g. meta-llama/Llama-3.1-8B)
        base_model_path = base_model_path_str

    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attn_impl = model_cfg.get("attn_implementation", "sdpa")
    print(f"Loading Stage 1 base model from {base_model_path} on {device} (attn: {attn_impl})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=device,
    )

    # Attach PEFT LoRA adapter
    lora_r = int(lora_cfg.get("r", 64))
    lora_alpha = int(lora_cfg.get("alpha", 128))
    lora_dropout = float(lora_cfg.get("dropout", 0.0))
    target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(
        f"Trainable parameters: {trainable_params:,} / {all_params:,} "
        f"({100 * trainable_params / all_params:.2f}%)"
    )

    # Dataset loading
    train_dataset, eval_dataset, collator = get_aft_datasets(config, tokenizer)

    micro_batch_size = int(train_cfg.get("micro_batch_size", 2))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 8))
    effective_batch = micro_batch_size * grad_accum_steps

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=1,
    )

    epochs = int(train_cfg.get("epochs", 1))
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    total_steps = steps_per_epoch * epochs
    max_steps = max_steps_override if max_steps_override is not None else total_steps

    lr = float(train_cfg.get("lr", 1.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    warmup_frac = float(train_cfg.get("warmup_frac", 0.05))
    warmup_steps = int(warmup_frac * max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # Resume handling
    start_step = 0
    tokens_trained_counter = 0
    if resume:
        checkpoints = sorted(
            ckpt_dir.glob("step_*"),
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
        )
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            print(f"Resuming AFT from checkpoint: {latest_ckpt}")
            state = load_checkpoint(latest_ckpt, model, optimizer, scheduler)
            start_step = state["step"]
            tokens_trained_counter = state["tokens_trained"]
        else:
            print("No checkpoint found to resume from. Starting from step 0.")

    csv_fields = [
        "step",
        "train_loss",
        "heldout_loss",
        "heldout_ppl",
        "lr",
        "grad_norm",
        "tokens_per_sec",
        "peak_vram_mb",
        "tokens_trained",
    ]
    if not resume or not csv_log_path.is_file():
        with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    eval_every = int(train_cfg.get("eval_every", 50))
    save_every = int(train_cfg.get("save_every", 100))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    if start_step == 0 and len(eval_dataset) > 0:
        h_loss, h_ppl = evaluate_heldout(model, eval_loader, device)
        print(f"Initial held-out evaluation: loss={h_loss:.4f}, perplexity={h_ppl:.2f}")

    model.train()
    step = start_step
    data_iter = iter(train_loader)

    if start_step > 0:
        for _ in range(start_step * grad_accum_steps):
            try:
                next(data_iter)
            except StopIteration:
                break

    optimizer.zero_grad()
    criterion_sum = nn.CrossEntropyLoss(reduction="sum")

    print(
        f"Starting AFT training for {run_id}: max_steps={max_steps}, "
        f"effective_batch_size={effective_batch} conversations, lr={lr}"
    )

    t_start = time.perf_counter()
    tokens_in_window = 0

    while step < max_steps:
        step += 1
        total_loss_sum = 0.0
        total_active_tokens = 0

        for micro_idx in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)

            active_tokens = (shift_labels != -100).sum().item()
            loss_sum = criterion_sum(shift_logits, shift_labels)

            normalized_loss = loss_sum / (grad_accum_steps * max(active_tokens, 1))
            normalized_loss.backward()

            total_loss_sum += loss_sum.item()
            total_active_tokens += active_tokens

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        ).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        tokens_trained_counter += total_active_tokens
        tokens_in_window += total_active_tokens
        step_train_loss = total_loss_sum / max(total_active_tokens, 1)

        h_loss: Optional[float] = None
        h_ppl: Optional[float] = None
        if (step % eval_every == 0 or step == max_steps) and len(eval_dataset) > 0:
            h_loss, h_ppl = evaluate_heldout(model, eval_loader, device)

        elapsed = time.perf_counter() - t_start
        tokens_per_sec = tokens_in_window / max(elapsed, 1e-6)
        t_start = time.perf_counter()
        tokens_in_window = 0

        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device=device) / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )
        current_lr = scheduler.get_last_lr()[0]

        log_entry = {
            "step": step,
            "train_loss": round(step_train_loss, 4),
            "heldout_loss": round(h_loss, 4) if h_loss is not None else None,
            "heldout_ppl": round(h_ppl, 2) if h_ppl is not None else None,
            "lr": current_lr,
            "grad_norm": round(grad_norm, 4),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "tokens_trained": tokens_trained_counter,
        }

        with open(jsonl_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(log_entry)

        eval_str = f" | heldout_loss={h_loss:.4f} (ppl={h_ppl:.2f})" if h_loss is not None else ""
        print(
            f"AFT Step {step}/{max_steps} | loss={step_train_loss:.4f}{eval_str} | "
            f"lr={current_lr:.2e} | grad_norm={grad_norm:.2f} | "
            f"tok/s={tokens_per_sec:.1f} | vram={peak_vram_mb:.0f}MB"
        )

        if step % save_every == 0 and step < max_steps:
            step_ckpt_dir = ckpt_dir / f"step_{step}"
            save_checkpoint(
                step_ckpt_dir,
                step,
                model,
                optimizer,
                scheduler,
                seed,
                tokens_trained_counter,
            )

    # Save final adapter & tokenizer
    print(f"Saving final AFT adapter to {adapter_dir}...")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    end_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    manifest = {
        "run_id": run_id,
        "mode": "aft",
        "arm": arm,
        "base_model": str(base_model_path),
        "git_commit": get_git_commit(),
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "steps_completed": step,
        "tokens_trained": tokens_trained_counter,
        "seed": seed,
        "allow_tf32": allow_tf32,
        "attn_implementation": attn_impl,
        "hardware": {
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1)
            if torch.cuda.is_available()
            else 0.0,
        },
        "config": config,
    }

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"AFT training completed for {run_id}. Manifest saved to {run_dir / 'manifest.json'}.")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for MSM or AFT.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training configuration YAML.",
    )
    parser.add_argument(
        "--corpus-id",
        type=str,
        default=None,
        help="Override corpus identifier (e.g. aw_main_v11).",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=None,
        help="Override corpus JSONL path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum training steps (e.g. 50 for smoke test).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest step checkpoint in runs/{mode}/{run_id}/checkpoints/.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["midtrain", "aft"],
        help="Training mode ('midtrain' or 'aft'). Defaults to config['mode'].",
    )

    args = parser.parse_args()

    default_config = (
        "configs/train_aft.yaml" if args.mode == "aft" else "configs/train_midtrain.yaml"
    )
    config_path = args.config if args.config is not None else default_config

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mode = args.mode or config.get("mode", "midtrain")

    if args.corpus_id is not None:
        config.setdefault("data", {})["corpus_id"] = args.corpus_id
    if args.corpus_path is not None:
        config.setdefault("data", {})["corpus_path"] = args.corpus_path
    if args.seed is not None:
        config.setdefault("train", {})["seed"] = args.seed

    if mode == "aft":
        train_aft(config, resume=args.resume, max_steps_override=args.max_steps)
    else:
        train_msm(config, resume=args.resume, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
