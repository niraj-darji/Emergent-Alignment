"""Download and cache Hendrycks ETHICS and Anthropic HH benchmarks to local HF_HOME."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Set HF cache directory
os.environ["HF_HOME"] = "/mnt/HDD_2TB/niraj/hf"
from datasets import load_dataset

def download_ethics():
    subsets = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]
    print("=== Downloading Hendrycks ETHICS subsets ===")
    for sub in subsets:
        print(f"Loading EleutherAI/hendrycks_ethics config: {sub}...")
        try:
            ds = load_dataset("EleutherAI/hendrycks_ethics", sub)
            print(f"  Successfully cached {sub}: {ds}")
        except Exception as e:
            print(f"  Error caching {sub}: {e}")

def download_anthropic_hh():
    print("\n=== Downloading Anthropic HH RLHF dataset ===")
    print("Loading Anthropic/hh-rlhf...")
    try:
        # Load dataset; hh-rlhf includes harmless-base and helpful-base splits
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
        print(f"  Successfully cached Anthropic/hh-rlhf (harmless-base): {ds}")
    except Exception as e:
        print(f"  Failed with data_dir='harmless-base': {e}. Trying full load...")
        try:
            ds = load_dataset("Anthropic/hh-rlhf")
            print(f"  Successfully cached full Anthropic/hh-rlhf: {ds}")
        except Exception as e2:
            print(f"  Error loading full dataset: {e2}")

if __name__ == "__main__":
    download_ethics()
    download_anthropic_hh()
    print("\nDone downloading benchmarks.")
