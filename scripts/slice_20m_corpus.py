#!/usr/bin/env python3
"""Corpus Slicing Script for Arm X2 Dose-Response Study.

Extracts:
1. 5.0M token slice  -> data/corpora/aw_main_v1_5m/
2. 10.0M token slice -> data/corpora/aw_main_v1_10m/
from the full 20M corpus:
awmsm_v1/data/corpora/aw_main_v1_20m/corpus.jsonl
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

SOURCE_CORPUS_DIR = Path("/mnt/HDD_2TB/niraj/awmsm_v1/data/corpora/aw_main_v1_20m")
SOURCE_CORPUS_FILE = SOURCE_CORPUS_DIR / "corpus.jsonl"

TARGET_PARENT_DIR = Path("/mnt/HDD_2TB/niraj/awmsm/data/corpora")

SLICES = [
    {
        "name": "aw_main_v1_5m",
        "target_tokens": 5_000_000,
        "dir": TARGET_PARENT_DIR / "aw_main_v1_5m",
    },
    {
        "name": "aw_main_v1_10m",
        "target_tokens": 10_000_000,
        "dir": TARGET_PARENT_DIR / "aw_main_v1_10m",
    },
]


def ensure_20m_symlink() -> None:
    link_path = TARGET_PARENT_DIR / "aw_main_v1_20m"
    if not link_path.exists():
        try:
            link_path.symlink_to(SOURCE_CORPUS_DIR)
            print(f"Created symlink: {link_path} -> {SOURCE_CORPUS_DIR}")
        except Exception as e:
            print(f"Warning: could not create symlink {link_path}: {e}")
    else:
        print(f"Symlink/Directory already exists: {link_path}")


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def slice_corpora() -> None:
    print("=" * 70)
    print("Starting Corpus Slicing for Arm X2 Dose-Response")
    print(f"Source file: {SOURCE_CORPUS_FILE}")
    print("=" * 70)

    if not SOURCE_CORPUS_FILE.exists():
        raise FileNotFoundError(f"Source corpus not found: {SOURCE_CORPUS_FILE}")

    ensure_20m_symlink()

    max_target_tokens = max(s["target_tokens"] for s in SLICES)

    for s in SLICES:
        s["dir"].mkdir(parents=True, exist_ok=True)
        s["file_handle"] = open(s["dir"] / "corpus.jsonl", "w", encoding="utf-8")
        s["docs_count"] = 0
        s["tokens_count"] = 0
        s["doc_types"] = collections.Counter()
        s["species_group"] = collections.Counter()
        s["context_domain"] = collections.Counter()
        s["cost_of_care"] = collections.Counter()
        s["perspective"] = collections.Counter()
        s["welfare_raised"] = collections.Counter()
        s["is_story"] = collections.Counter()
        s["completed"] = False

    print("Reading and slicing documents...")
    doc_index = 0
    total_tokens_read = 0

    with open(SOURCE_CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            tokens = doc.get("tokens", 0)
            doc_index += 1
            total_tokens_read += tokens

            for s in SLICES:
                if not s["completed"]:
                    s["file_handle"].write(line)
                    s["docs_count"] += 1
                    s["tokens_count"] += tokens

                    s["doc_types"][doc.get("doc_type", "unknown")] += 1
                    s["species_group"][doc.get("species_group", "unknown")] += 1
                    s["context_domain"][doc.get("context_domain", "unknown")] += 1
                    s["cost_of_care"][doc.get("cost_of_care", "unknown")] += 1
                    s["perspective"][doc.get("perspective", "unknown")] += 1
                    s["welfare_raised"][doc.get("welfare_raised", "unknown")] += 1
                    s["is_story"][str(doc.get("is_story", False))] += 1

                    if s["tokens_count"] >= s["target_tokens"]:
                        s["completed"] = True
                        s["file_handle"].close()
                        print(
                            f"Target reached for {s['name']}: {s['tokens_count']:,} tokens "
                            f"in {s['docs_count']:,} documents (target was {s['target_tokens']:,})."
                        )

            if all(s["completed"] for s in SLICES):
                print("All slice targets reached. Stopping iteration.")
                break

            if doc_index % 2000 == 0:
                print(f"Processed {doc_index:,} source docs ({total_tokens_read:,} tokens)...")

    # Close any open file handles
    for s in SLICES:
        if not s["file_handle"].closed:
            s["file_handle"].close()

    print("\nGenerating manifests and summary files...")
    timestamp = datetime.datetime.now().astimezone().isoformat()

    for s in SLICES:
        corpus_path = s["dir"] / "corpus.jsonl"
        print(f"Computing sha256 for {corpus_path}...")
        file_sha256 = compute_sha256(corpus_path)

        summary_data = {
            "spec_id": "aw_main",
            "target_tokens": s["target_tokens"],
            "tokens": s["tokens_count"],
            "documents": s["docs_count"],
            "source_corpus": str(SOURCE_CORPUS_FILE),
            "created_at": timestamp,
            "status": "target_reached",
            "token_method": "tokenizer meta-llama/Llama-3.1-8B",
            "story_share": s["is_story"]["True"] / s["docs_count"] if s["docs_count"] > 0 else 0.0,
            "doc_types": dict(s["doc_types"]),
            "distributions": {
                "species_group": dict(s["species_group"]),
                "context_domain": dict(s["context_domain"]),
                "cost_of_care": dict(s["cost_of_care"]),
                "perspective": dict(s["perspective"]),
                "welfare_raised": dict(s["welfare_raised"]),
                "is_story": dict(s["is_story"]),
            },
        }

        with open(s["dir"] / "summary.json", "w", encoding="utf-8") as sf:
            json.dump(summary_data, sf, indent=2)

        manifest_data = {
            "corpus_name": s["name"],
            "corpus_sha256": file_sha256,
            "source": str(SOURCE_CORPUS_FILE),
            "tokens": s["tokens_count"],
            "documents": s["docs_count"],
            "created_at": timestamp,
        }

        with open(s["dir"] / "manifest.json", "w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, indent=2)

        print(f"Completed {s['name']}: {s['docs_count']} docs, {s['tokens_count']} tokens, SHA256: {file_sha256[:16]}...")

    print("=" * 70)
    print("Corpus Slicing Complete for Arm X2 Slices (5M and 10M).")
    print("=" * 70)


if __name__ == "__main__":
    slice_corpora()
