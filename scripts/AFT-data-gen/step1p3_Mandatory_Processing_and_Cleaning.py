#!/usr/bin/env python3
"""Stage 2 AFT - Step 1.3: Mandatory Processing and Cleaning Pipeline.

Extracts, filters, de-contaminates, and balances the general instruction-tuning mixture
(~10,000 samples, ~2.0M tokens) replicating MSM Appendix B.3 (Table 2) and
AW_MSM_Methodology.md Section 7.3.

Filters enforced:
1. Length truncation: Total conversation length <= 4,096 tokens (Qwen tokenizer).
2. Persona scrubbing: Rejects third-party identity ("I am ChatGPT", etc.) and opinion refusals.
3. Animal welfare de-contamination: Strictly removes queries with animal or welfare content (Appendix F.4).
4. Deterministic sampling: Uses fixed random seed (default: 42).
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# -----------------------------------------------------------------------------
# Regex Filter Patterns
# -----------------------------------------------------------------------------

# Persona / Third-party model scrubbers
RE_OTHER_MODELS = re.compile(
    r"\b(i am|i'm|as)\s+(an?\s+)?(chatgpt|gpt-?3(\.5)?|gpt-?4|claude|bard|gemini|llama|copilot)\b",
    re.IGNORECASE,
)
RE_OTHER_DEVELOPERS = re.compile(
    r"\b(developed|created|trained)\s+by\s+(openai|anthropic|google|meta|microsoft)\b",
    re.IGNORECASE,
)
RE_NO_PREFERENCES = re.compile(
    r"\bas an ai\b.*\b(no (personal )?(opinions?|preferences?|feelings?|thoughts?|beliefs?))\b",
    re.IGNORECASE,
)

# Animal Welfare De-contamination Regex (Appendix F.4)
# Excludes general instruction data from adding uncontrolled animal welfare signal
AW_DECONTAMINATION_KEYWORDS = [
    r"animal", r"animals", r"livestock", r"cattle", r"cows?", r"pigs?", r"poultry",
    r"chickens?", r"hens?", r"fish", r"fishes", r"fishing", r"salmon", r"tilapia",
    r"shrimps?", r"slaughter", r"abattoir", r"meat", r"dairy", r"egg-laying",
    r"vegans?", r"veganism", r"vegetarians?", r"cruelty", r"hunting", r"traps?",
    r"trapping", r"pests?", r"wildlife", r"zoos?", r"aquariums?", r"pets?", r"dogs?",
    r"cats?", r"horses?", r"insects?", r"bees?", r"octopus(es|i)?", r"lobsters?",
    r"crabs?", r"sentien\w*"
]
RE_AW_DECONTAMINATION = re.compile(
    r"\b(" + "|".join(AW_DECONTAMINATION_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def load_tokenizer(model_path_or_name: str):
    """Loads tokenizer with fallback to character-based heuristic if transformers not available."""
    if AutoTokenizer is None:
        print("[WARNING] transformers package not found. Using character-length heuristic.")
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_path_or_name, trust_remote_code=True)
        return tok
    except Exception as e:
        print(f"[WARNING] Could not load tokenizer from {model_path_or_name}: {e}")
        print("Using character-length heuristic.")
        return None


def count_tokens(messages: List[Dict[str, str]], tokenizer: Any) -> int:
    """Estimates or calculates token length for a list of messages."""
    full_text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(full_text, add_special_tokens=False))
        except Exception:
            pass
    # Fallback heuristic: 1 token ~= 4 characters
    return max(1, len(full_text) // 4)


def passes_filters(messages: List[Dict[str, str]], max_tokens: int, tokenizer: Any) -> Tuple[bool, str]:
    """Applies length, persona, and animal welfare de-contamination filters."""
    if not messages:
        return False, "empty_messages"

    # 1. Length check
    tok_count = count_tokens(messages, tokenizer)
    if tok_count > max_tokens:
        return False, "exceeds_max_tokens"

    # Combine text for content checks
    full_text = " ".join(m.get("content", "") for m in messages)

    # 2. Persona scrubbing
    if RE_OTHER_MODELS.search(full_text):
        return False, "persona_other_model"
    if RE_OTHER_DEVELOPERS.search(full_text):
        return False, "persona_other_developer"
    if RE_NO_PREFERENCES.search(full_text):
        return False, "persona_no_preferences"

    # 3. Animal Welfare De-contamination
    if RE_AW_DECONTAMINATION.search(full_text):
        return False, "aw_decontamination"

    return True, "pass"


# -----------------------------------------------------------------------------
# Source Loaders
# -----------------------------------------------------------------------------

def extract_no_robots(raw_dir: Path, target_count: int, seed: int, max_tokens: int, tokenizer: Any):
    """Loads and filters No Robots dataset."""
    print("\n--- Extracting No Robots ---")
    parquet_files = sorted(raw_dir.glob("no_robots/data/train*.parquet"))
    if not parquet_files:
        print("[ERROR] No Robots parquet file not found.")
        return [], {}

    items = []
    stats = {"total_read": 0, "accepted": 0, "rejected": {}}

    for pf in parquet_files:
        table = pq.read_table(pf)
        for i in range(table.num_rows):
            stats["total_read"] += 1
            raw_msgs = table["messages"][i].as_py()
            messages = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]
            
            ok, reason = passes_filters(messages, max_tokens, tokenizer)
            if ok:
                items.append({
                    "id": f"no_robots_{stats['total_read']:05d}",
                    "source": "no_robots",
                    "messages": messages,
                })
            else:
                stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1

    rng = random.Random(seed)
    rng.shuffle(items)
    selected = items[:target_count]
    stats["accepted"] = len(selected)
    print(f"No Robots: read {stats['total_read']}, accepted {stats['accepted']}/{target_count}")
    return selected, stats


def extract_tulu_and_lima(raw_dir: Path, tulu_target: int, lima_target: int, seed: int, max_tokens: int, tokenizer: Any):
    """Loads instruction-following and LIMA-equivalent samples from Tulu 3 SFT."""
    print("\n--- Extracting Tulu 3 SFT & LIMA ---")
    parquet_files = sorted(raw_dir.glob("tulu_3_sft/data/train*.parquet"))
    if not parquet_files:
        print("[ERROR] Tulu 3 SFT parquet files not found.")
        return [], [], {}

    tulu_items = []
    lima_items = []
    stats = {"total_read": 0, "accepted_tulu": 0, "accepted_lima": 0, "rejected": {}}

    # Tulu instruction sources of interest
    target_if_sources = {
        "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
        "ai2-adapt-dev/flan_v2_converted",
        "ai2-adapt-dev/coconot_converted",
    }

    for pf in parquet_files:
        table = pq.read_table(pf)
        for i in range(table.num_rows):
            stats["total_read"] += 1
            src = table["source"][i].as_py()
            raw_msgs = table["messages"][i].as_py()
            messages = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]

            ok, reason = passes_filters(messages, max_tokens, tokenizer)
            if not ok:
                stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                continue

            if src in target_if_sources:
                tulu_items.append({
                    "id": f"tulu3_if_{stats['total_read']:06d}",
                    "source": "tulu_3_if",
                    "messages": messages,
                })
            elif src == "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980" or len(lima_items) < lima_target * 5:
                # High-curation instruction examples for LIMA slice
                lima_items.append({
                    "id": f"lima_tulu_{stats['total_read']:06d}",
                    "source": "lima_tulu3",
                    "messages": messages,
                })

            if len(tulu_items) >= tulu_target * 4 and len(lima_items) >= lima_target * 4:
                break

    rng = random.Random(seed)
    rng.shuffle(tulu_items)
    rng.shuffle(lima_items)

    selected_tulu = tulu_items[:tulu_target]
    selected_lima = lima_items[:lima_target]
    stats["accepted_tulu"] = len(selected_tulu)
    stats["accepted_lima"] = len(selected_lima)

    print(f"Tulu 3 IF: accepted {stats['accepted_tulu']}/{tulu_target}")
    print(f"LIMA (from Tulu 3): accepted {stats['accepted_lima']}/{lima_target}")
    return selected_tulu, selected_lima, stats


def extract_smoltalk_subsets(raw_dir: Path, quotas: Dict[str, int], seed: int, max_tokens: int, tokenizer: Any):
    """Loads and filters designated slices from smoltalk subsets."""
    print("\n--- Extracting smoltalk Subsets ---")
    subset_mapping = {
        "numina-cot-100k": ("smol_numina_cot", quotas.get("numina_cot", 1050)),
        "self-oss-instruct": ("smol_self_oss", quotas.get("self_oss", 1050)),
        "smol-constraints": ("smol_constraints", quotas.get("constraints", 1050)),
        "apigen-80k": ("smol_apigen", quotas.get("apigen", 1050)),
        "smol-summarize": ("smol_summarize", quotas.get("summarize", 1000)),
    }

    all_selected = []
    stats = {}

    rng = random.Random(seed)

    for sub_dir, (label, target_count) in subset_mapping.items():
        pf_path = raw_dir / "smoltalk" / "data" / sub_dir / "train-00000-of-00001.parquet"
        if not pf_path.exists():
            print(f"[WARNING] Subset parquet not found: {pf_path}")
            continue

        table = pq.read_table(pf_path)
        items = []
        sub_stats = {"total_read": 0, "accepted": 0, "rejected": {}}

        for i in range(table.num_rows):
            sub_stats["total_read"] += 1
            raw_msgs = table["messages"][i].as_py()
            messages = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]

            ok, reason = passes_filters(messages, max_tokens, tokenizer)
            if ok:
                items.append({
                    "id": f"{label}_{sub_stats['total_read']:06d}",
                    "source": label,
                    "messages": messages,
                })
            else:
                sub_stats["rejected"][reason] = sub_stats["rejected"].get(reason, 0) + 1

        rng.shuffle(items)
        selected = items[:target_count]
        sub_stats["accepted"] = len(selected)
        all_selected.extend(selected)
        stats[label] = sub_stats
        print(f"{label} ({sub_dir}): read {sub_stats['total_read']}, accepted {sub_stats['accepted']}/{target_count}")

    return all_selected, stats


def extract_longalign(raw_dir: Path, target_count: int, seed: int, max_tokens: int, tokenizer: Any):
    """Loads and filters LongAlign samples."""
    print("\n--- Extracting LongAlign ---")
    jsonl_path = raw_dir / "longalign" / "long.jsonl"
    if not jsonl_path.exists():
        print(f"[ERROR] LongAlign file not found at {jsonl_path}")
        return [], {}

    items = []
    stats = {"total_read": 0, "accepted": 0, "rejected": {}}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            stats["total_read"] += 1
            record = json.loads(line)
            raw_msgs = record.get("messages", [])
            messages = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]

            ok, reason = passes_filters(messages, max_tokens, tokenizer)
            if ok:
                items.append({
                    "id": f"longalign_{stats['total_read']:05d}",
                    "source": "longalign",
                    "messages": messages,
                })
            else:
                stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1

    rng = random.Random(seed)
    rng.shuffle(items)
    selected = items[:target_count]
    stats["accepted"] = len(selected)
    print(f"LongAlign: read {stats['total_read']}, accepted {stats['accepted']}/{target_count}")
    return selected, stats


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2 AFT Step 1.3: Mandatory Processing and Cleaning Pipeline")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="/mnt/HDD_2TB/niraj/awmsm/data/raw_datasets",
        help="Path to directory containing downloaded raw datasets",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/HDD_2TB/niraj/awmsm/data/aft",
        help="Path to save processed SFT dataset and manifest",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/midtrain_aw_main_v1_m1",
        help="Path to model or checkpoint tokenizer",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum allowed token length per conversation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and shuffling")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===================================================================")
    print("Stage 2 AFT: Step 1.3 Mandatory Processing and Cleaning Pipeline")
    print(f"Raw Datasets Directory: {raw_dir}")
    print(f"Output Directory:       {out_dir}")
    print(f"Max Token Length:       {args.max_tokens}")
    print(f"Random Seed:            {args.seed}")
    print("===================================================================")

    if pq is None:
        sys.exit("[FATAL] pyarrow is not installed. Please activate the awmsm or vllm venv.")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    full_mixture = []
    manifest_stats = {}

    # 1. No Robots: 2,800
    no_robots_items, nr_stats = extract_no_robots(raw_dir, target_count=2800, seed=args.seed, max_tokens=args.max_tokens, tokenizer=tokenizer)
    full_mixture.extend(no_robots_items)
    manifest_stats["no_robots"] = nr_stats

    # 2. Tulu 3 IF (1,500) & LIMA (300)
    tulu_items, lima_items, tulu_stats = extract_tulu_and_lima(
        raw_dir, tulu_target=1500, lima_target=300, seed=args.seed, max_tokens=args.max_tokens, tokenizer=tokenizer
    )
    full_mixture.extend(tulu_items)
    full_mixture.extend(lima_items)
    manifest_stats["tulu_3_sft"] = tulu_stats

    # 3. smoltalk Subsets: 1,050 + 1,050 + 1,050 + 1,050 + 1,000 = 5,200
    smol_items, smol_stats = extract_smoltalk_subsets(
        raw_dir,
        quotas={"numina_cot": 1050, "self_oss": 1050, "constraints": 1050, "apigen": 1050, "summarize": 1000},
        seed=args.seed,
        max_tokens=args.max_tokens,
        tokenizer=tokenizer,
    )
    full_mixture.extend(smol_items)
    manifest_stats["smoltalk"] = smol_stats

    # 4. LongAlign: 200
    longalign_items, la_stats = extract_longalign(raw_dir, target_count=200, seed=args.seed, max_tokens=args.max_tokens, tokenizer=tokenizer)
    full_mixture.extend(longalign_items)
    manifest_stats["longalign"] = la_stats

    # Final shuffle across all sources
    print("\n--- Final Mixture Assembly ---")
    rng = random.Random(args.seed)
    rng.shuffle(full_mixture)

    out_jsonl = out_dir / "general_sft_10k.jsonl"
    out_manifest = out_dir / "general_sft_manifest.json"

    print(f"Writing {len(full_mixture)} samples to {out_jsonl}...")
    sha256 = hashlib.sha256()
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in full_mixture:
            line = json.dumps(item, ensure_ascii=False) + "\n"
            f.write(line)
            sha256.update(line.encode("utf-8"))

    file_hash = sha256.hexdigest()
    manifest = {
        "dataset_name": "general_sft_10k",
        "target_volume_samples": 10000,
        "actual_volume_samples": len(full_mixture),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "output_file": str(out_jsonl),
        "output_sha256": file_hash,
        "sources_breakdown": {
            "no_robots": len(no_robots_items),
            "tulu_3_if": len(tulu_items),
            "lima_extracted": len(lima_items),
            "smoltalk_subsets": len(smol_items),
            "longalign": len(longalign_items),
        },
        "stats": manifest_stats,
    }

    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("===================================================================")
    print("SUCCESS: Stage 2 AFT Step 1.3 Complete!")
    print(f"Total samples written: {len(full_mixture)}")
    print(f"Dataset path:          {out_jsonl}")
    print(f"Manifest path:         {out_manifest}")
    print(f"SHA-256:               {file_hash}")
    print("===================================================================")


if __name__ == "__main__":
    main()

