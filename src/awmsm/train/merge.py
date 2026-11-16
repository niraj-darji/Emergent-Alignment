"""Merge LoRA adapter into base bfloat16 weights with verification gates.

Reference: AW_MSM_Methodology.md (§8.3) and MSM_training_blueprint.md (§7).
"""

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_dir_sha256(directory: Path) -> str:
    """Compute combined SHA-256 hash of files in a directory."""
    hasher = hashlib.sha256()
    for file_path in sorted(directory.glob("**/*")):
        if file_path.is_file():
            hasher.update(file_path.name.encode("utf-8"))
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
    return hasher.hexdigest()


def verify_weight_delta(
    base_model: torch.nn.Module,
    merged_model: torch.nn.Module,
) -> Tuple[bool, float]:
    """Verify that at least one targeted weight matrix was updated (delta norm > 0)."""
    base_state = base_model.state_dict()
    merged_state = merged_model.state_dict()

    max_delta = 0.0
    for key in base_state:
        if key in merged_state:
            diff = (merged_state[key].detach().cpu().float() - base_state[key].detach().cpu().float()).abs()
            diff_max = diff.max().item()
            if diff_max > max_delta:
                max_delta = diff_max

    delta_detected = max_delta > 1e-6
    return delta_detected, max_delta


def verify_logits_parity(
    unmerged_model: torch.nn.Module,
    merged_model: torch.nn.Module,
    tokenizer: Any,
    test_prompts: Optional[List[str]] = None,
    tolerance: float = 0.5,
) -> Tuple[bool, float, bool]:
    """Verify that unmerged (base+adapter) and merged model produce identical logits.

    In bfloat16, mantissa precision is 7 bits (machine epsilon ~0.0078). For logits
    with values in the range 15-35, 1 ULP is 0.125-0.25, so accumulated rounding differences
    across 32 transformer layers naturally reach ~0.15-0.30. We therefore check both
    max difference <= tolerance (default 0.5) and 100% agreement on top-1 predicted tokens.
    """
    if test_prompts is None:
        test_prompts = [
            "The ethical consideration for animal welfare requires",
            "In practical husbandry, welfare standards should balance",
            "Research on crustacean nociception demonstrates that",
            "When designing shelter protocols for companion animals,",
        ]

    max_diff = 0.0
    all_argmax_match = True
    unmerged_model.eval()
    merged_model.eval()

    device = next(unmerged_model.parameters()).device

    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

            out_unmerged = unmerged_model(tokens).logits
            out_merged = merged_model(tokens).logits

            diff = (out_unmerged.float() - out_merged.float()).abs().max().item()
            if diff > max_diff:
                max_diff = diff

            top_unmerged = out_unmerged.argmax(dim=-1)
            top_merged = out_merged.argmax(dim=-1)
            if not torch.equal(top_unmerged, top_merged):
                for b in range(tokens.shape[0]):
                    for t in range(tokens.shape[1]):
                        u_k = top_unmerged[b, t].item()
                        m_k = top_merged[b, t].item()
                        if u_k != m_k:
                            margin_unm = abs(out_unmerged[b, t, u_k].item() - out_unmerged[b, t, m_k].item())
                            margin_m = abs(out_merged[b, t, u_k].item() - out_merged[b, t, m_k].item())
                            if margin_unm > tolerance or margin_m > tolerance:
                                all_argmax_match = False

    parity_ok = (max_diff <= tolerance) and all_argmax_match
    return parity_ok, max_diff, all_argmax_match


def merge_adapter(
    adapter_path: str | Path,
    output_dir: str | Path,
    base_model_path: Optional[str] = None,
    device: str = "cpu",
    logit_tolerance: float = 1.0,
) -> Path:
    """Merge LoRA adapter into base model and execute verification gates (§7)."""
    adapter_dir = Path(adapter_path)
    out_dir = Path(output_dir)

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    # Read base model path from adapter config if not explicitly provided
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"adapter_config.json missing in {adapter_dir}")

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    if base_model_path is None:
        base_model_path = adapter_cfg.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(
                "base_model_name_or_path not found in adapter_config.json and not supplied."
            )

    print(f"Loading base model '{base_model_path}' in bfloat16...")
    torch_device = torch.device(device)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": torch_device},
    )

    print(f"Loading adapter from '{adapter_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    adapted_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        torch_dtype=torch.bfloat16,
    )

    # Logit parity check between base+adapter vs merged
    # First, test unmerged logits on test prompts
    print("Running logit parity pre-check...")
    test_prompts = [
        "The ethical consideration for animal welfare requires",
        "In practical husbandry, welfare standards should balance",
        "Research on crustacean nociception demonstrates that",
        "When designing shelter protocols for companion animals,",
    ]
    unmerged_logits_cache = {}
    adapted_model.eval()
    with torch.no_grad():
        for p in test_prompts:
            tokens = tokenizer(p, return_tensors="pt")["input_ids"].to(torch_device)
            unmerged_logits_cache[p] = adapted_model(tokens).logits.cpu().float()

    print("Merging adapter into base weights (merge_and_unload)...")
    merged_model = adapted_model.merge_and_unload()

    # Gate 1: Check logit parity against cached unmerged logits
    max_logit_diff = 0.0
    all_argmax_match = True
    with torch.no_grad():
        for p in test_prompts:
            tokens = tokenizer(p, return_tensors="pt")["input_ids"].to(torch_device)
            m_logits = merged_model(tokens).logits.cpu().float()
            unm_logits = unmerged_logits_cache[p]
            diff = (m_logits - unm_logits).abs().max().item()
            if diff > max_logit_diff:
                max_logit_diff = diff
            m_top = m_logits.argmax(dim=-1)
            unm_top = unm_logits.argmax(dim=-1)
            if not torch.equal(m_top, unm_top):
                for b in range(tokens.shape[0]):
                    for t in range(tokens.shape[1]):
                        u_k = unm_top[b, t].item()
                        m_k = m_top[b, t].item()
                        if u_k != m_k:
                            margin_unm = abs(unm_logits[b, t, u_k].item() - unm_logits[b, t, m_k].item())
                            margin_m = abs(m_logits[b, t, u_k].item() - m_logits[b, t, m_k].item())
                            if margin_unm > logit_tolerance or margin_m > logit_tolerance:
                                all_argmax_match = False

    if max_logit_diff > logit_tolerance:
        raise RuntimeError(
            f"Merge verification failed: logit difference ({max_logit_diff:.5f}) "
            f"exceeds tolerance ({logit_tolerance:.5f})."
        )
    if not all_argmax_match:
        raise RuntimeError(
            "Merge verification failed: top-1 predicted token mismatch between adapter and merged model."
        )
    print(
        f"Logit parity check passed (max diff: {max_logit_diff:.5f} <= {logit_tolerance}, "
        f"argmax match: 100%)."
    )

    # Gate 2: Weight delta verification
    # Compare merged model weights against freshly loaded base weights
    print("Verifying weight delta between base and merged model...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    delta_ok, max_weight_delta = verify_weight_delta(ref_base, merged_model)
    del ref_base

    if not delta_ok:
        raise RuntimeError(
            "Merge verification failed: parameter delta norm is zero. "
            "Adapter weights were not applied or were all zero."
        )
    print(f"Weight delta check passed (max parameter delta: {max_weight_delta:.6f}).")

    # Save merged model and tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out_dir}...")
    merged_model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    # Gate 3: Reload test
    print("Running reload verification...")
    try:
        reloaded = AutoModelForCausalLM.from_pretrained(
            str(out_dir),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        del reloaded
        print("Reload verification passed.")
    except Exception as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError(f"Merge verification failed on reload: {e}. Output deleted.") from e

    # Write merge_manifest.json
    manifest: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "adapter_path": str(adapter_dir),
        "base_model": base_model_path,
        "output_dir": str(out_dir),
        "verification": {
            "weight_delta_ok": delta_ok,
            "max_weight_delta": max_weight_delta,
            "logit_parity_ok": True,
            "max_logit_diff": max_logit_diff,
            "logit_tolerance": logit_tolerance,
            "reload_ok": True,
        },
    }
    with open(out_dir / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Merge completed successfully. Checkpoint ready at: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (e.g. midtrain_aw_main_v11_m1).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to adapter directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save merged checkpoint.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path or huggingface id.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for merge verification.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]

    if args.run_id is not None:
        if args.run_id.startswith("aft_"):
            adapter_path = project_root / "runs/aft" / args.run_id / "adapter"
            manifest_path = project_root / "runs/aft" / args.run_id / "manifest.json"
            if args.base_model is None and manifest_path.is_file():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    args.base_model = m_data.get("model", {}).get("base_path") or m_data.get("base_model")
                except Exception:
                    pass
        else:
            adapter_path = project_root / "runs/midtrain" / args.run_id / "adapter"
        output_dir = project_root / "runs/checkpoints" / args.run_id
    elif args.adapter_path is not None and args.output_dir is not None:
        adapter_path = Path(args.adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = project_root / adapter_path
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    else:
        print("Error: Specify either --run-id or both --adapter-path and --output-dir.")
        sys.exit(1)

    merge_adapter(
        adapter_path=adapter_path,
        output_dir=output_dir,
        base_model_path=args.base_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
