"""Preparation and token-matching script for CaML Public Corpus (Arm A5).

Extracts and cleans documents from CompassioninMachineLearning/3k_pretraining_research_documents_v3,
saves a full archive copy, and subsamples to match the exact 2.50M token budget of Arm A2
using the meta-llama/Llama-3.1-8B tokenizer.

Reference: AW_MSM_Methodology.md (§4.4, §6.8).
"""

import json
import os
import random
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    os.environ["HF_HOME"] = "/mnt/HDD_2TB/niraj/hf"
    output_dir = Path("/mnt/HDD_2TB/niraj/awmsm/data/corpora/caml_public")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = "/mnt/HDD_2TB/niraj/hf/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
    print("Loading Llama-3.1-8B tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    print("Loading CaML 3k pretraining research documents...")
    ds = load_dataset("CompassioninMachineLearning/3k_pretraining_research_documents_v3", split="train")

    raw_archive_path = output_dir / "raw_caml_all.jsonl"
    corpus_path = output_dir / "corpus.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    valid_docs = []
    skipped_count = 0

    print("Auditing and archiving all raw CaML records...")
    with open(raw_archive_path, "w", encoding="utf-8") as f_raw:
        for idx, row in enumerate(ds):
            txt = row.get("output")
            origin = row.get("origin", "")
            if not isinstance(txt, str) or not txt.strip():
                skipped_count += 1
                continue

            doc_id = f"caml_doc_{idx:04d}"
            record = {
                "id": doc_id,
                "original_index": idx,
                "origin": origin,
                "text": txt.strip(),
            }
            f_raw.write(json.dumps(record) + "\n")
            valid_docs.append(record)

    print(f"Total raw records: {len(ds)}")
    print(f"Skipped invalid/None records: {skipped_count}")
    print(f"Valid records saved to archive: {len(valid_docs)} -> {raw_archive_path}")

    # Token-matched subsampling
    # Target: 2,500,000 tokens (matching aw_main_v1: 2,500,028 tokens)
    TARGET_TOKENS = 2500000
    TOLERANCE = 0.01  # 1% tolerance

    print(f"Subsampling to target token budget: {TARGET_TOKENS:,} tokens...")
    # Use deterministic order (seed=42) for reproducible selection
    rng = random.Random(42)
    indices = list(range(len(valid_docs)))
    rng.shuffle(indices)

    selected_docs = []
    total_tokens = 0

    for idx in indices:
        doc = valid_docs[idx]
        tokens = len(tok.encode(doc["text"], add_special_tokens=False))
        if total_tokens + tokens > TARGET_TOKENS + (TARGET_TOKENS * TOLERANCE):
            # If adding this doc exceeds the upper tolerance, check if we are within range
            if total_tokens >= TARGET_TOKENS - (TARGET_TOKENS * TOLERANCE):
                break
        doc_entry = dict(doc)
        doc_entry["tokens"] = tokens
        selected_docs.append(doc_entry)
        total_tokens += tokens
        if total_tokens >= TARGET_TOKENS:
            break

    print(f"Selected {len(selected_docs)} documents totaling {total_tokens:,} tokens.")
    diff = total_tokens - TARGET_TOKENS
    print(f"Deviation from target: {diff:+d} tokens ({diff / TARGET_TOKENS * 100:+.2f}%)")

    # Sort selected documents by id for deterministic reproducibility
    selected_docs.sort(key=lambda d: d["id"])

    # Write training corpus.jsonl
    with open(corpus_path, "w", encoding="utf-8") as f_out:
        for d in selected_docs:
            f_out.write(json.dumps(d) + "\n")

    summary = {
        "corpus_id": "caml_public",
        "source_repo": "CompassioninMachineLearning/3k_pretraining_research_documents_v3",
        "split": "train",
        "total_source_records": len(ds),
        "skipped_invalid_records": skipped_count,
        "valid_source_records": len(valid_docs),
        "subsampled_records": len(selected_docs),
        "target_tokens": TARGET_TOKENS,
        "actual_tokens": total_tokens,
        "token_deviation": diff,
        "token_deviation_pct": round(diff / TARGET_TOKENS * 100, 4),
        "tokenizer": "meta-llama/Llama-3.1-8B",
        "archive_file": str(raw_archive_path),
        "corpus_file": str(corpus_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    manifest = {
        "corpus_id": "caml_public",
        "document_ids": [d["id"] for d in selected_docs],
        "original_indices": [d["original_index"] for d in selected_docs],
    }

    with open(manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest, f_man, indent=2)

    print("Corpus preparation complete:")
    print(f"  Corpus: {corpus_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
