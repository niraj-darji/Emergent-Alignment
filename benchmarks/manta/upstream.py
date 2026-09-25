"""Fetch MANTA's prompt code from the official repo at a pinned commit.

The upstream repository (github.com/Mycelium-tools/manta) has no license file,
so we do not vendor its code. We download the two modules that hold the
prompts at a fixed commit, check their hashes, and import only the pure
prompt-building functions. Model calls happen in our own solver/scorer, so
upstream's hard-coded OpenRouter models are never used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import urllib.request
from functools import cache
from pathlib import Path
from types import ModuleType

from benchmarks.config import REPO_ROOT

MANTA_REPO = "Mycelium-tools/manta"
MANTA_COMMIT = "04f914c6673a4e2e173217aeb4cce8a00f20a080"
FILES = {
    "dynamic_multiturn_solver.py": "a26a87882b3f780ea45ff52386fba2266bba099950110a086391670017bfa36b",
    "manta_scorer.py": "8faa799cfc1534630a22ec47643fe83d9b67e6f8a28a307245f3bfe7388d2510",
}
CACHE_DIR = REPO_ROOT / ".cache" / "manta" / MANTA_COMMIT


def _fetch(filename: str) -> Path:
    path = CACHE_DIR / filename
    expected = FILES[filename]
    if not path.exists():
        url = f"https://raw.githubusercontent.com/{MANTA_REPO}/{MANTA_COMMIT}/{filename}"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {url}: expected {expected}, got {actual}")
        path.write_bytes(data)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Cached {path} does not match the pinned hash; delete it and retry")
    return path


def _import(filename: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, _fetch(filename))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def solver_prompts() -> ModuleType:
    """Provides create_anchor_prompt, create_followup_prompt, format_conversation_for_prompt."""
    return _import("dynamic_multiturn_solver.py", "_manta_upstream_solver")


@cache
def scorer_prompts() -> ModuleType:
    """Provides create_per_turn_prompt, create_dimension_prompt, format_conversation."""
    return _import("manta_scorer.py", "_manta_upstream_scorer")
