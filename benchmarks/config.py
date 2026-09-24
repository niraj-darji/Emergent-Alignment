"""Load configs/models.yaml and configs/eval.yaml and turn them into inspect objects."""

from __future__ import annotations

import copy
import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.model import GenerateConfig, Model, get_model

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = REPO_ROOT / "configs" / "models.yaml"
EVAL_FILE = REPO_ROOT / "configs" / "eval.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class ModelSpec:
    alias: str
    model: str
    base_url: str | None = None
    model_args: dict[str, Any] = field(default_factory=dict)
    generate_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Model:
        return get_model(
            self.model,
            config=GenerateConfig(**self.generate_config),
            base_url=self.base_url,
            **self.model_args,
        )


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ModelRegistry:
    def __init__(self, path: Path = MODELS_FILE):
        data = _load_yaml(path)
        self._raw: dict[str, dict] = data.get("models") or {}
        self._vllm_defaults: dict = data.get("vllm_defaults") or {}

    def aliases(self) -> list[str]:
        return list(self._raw)

    def _resolve_raw(self, alias: str, seen: tuple[str, ...] = ()) -> dict:
        if alias in seen:
            raise ValueError(f"Cycle in `extends`: {' -> '.join(seen + (alias,))}")
        entry = self._raw[alias]
        parent = entry.get("extends")
        if parent is None:
            return copy.deepcopy(entry)
        if parent not in self._raw:
            raise KeyError(f"Model '{alias}' extends unknown model '{parent}'")
        merged = deep_merge(self._resolve_raw(parent, seen + (alias,)), entry)
        merged.pop("extends", None)
        return merged

    def get(self, name: str) -> ModelSpec:
        """Resolve a registry alias, or accept a raw inspect model string."""
        if name in self._raw:
            entry = self._resolve_raw(name)
        elif "/" in name:
            entry = {"model": name}
        else:
            raise KeyError(
                f"Unknown model '{name}'. Add it to {MODELS_FILE.name} or pass a raw "
                f"inspect model string like 'vllm/Qwen/Qwen3-8B'. Known: {self.aliases()}"
            )
        if "model" not in entry:
            raise ValueError(f"Model '{name}' has no `model` field")
        if entry["model"].startswith("vllm/") and not entry.get("base_url"):
            entry = deep_merge(self._vllm_defaults, entry)
        return ModelSpec(
            alias=name.replace("/", "__") if name not in self._raw else name,
            model=entry["model"],
            base_url=entry.get("base_url"),
            model_args=entry.get("model_args") or {},
            generate_config=entry.get("generate_config") or {},
            metadata=entry.get("metadata") or {},
        )


@dataclass
class BenchmarkSpec:
    name: str
    task: str
    task_args: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    epochs: int | None = None
    roles: list[str] = field(default_factory=list)
    generate_config: dict[str, Any] = field(default_factory=dict)
    group: str = ""
    headline: str = "accuracy"

    def task_fn(self):
        module_name, _, fn_name = self.task.partition(":")
        if not fn_name:
            raise ValueError(f"Benchmark '{self.name}': task must look like 'module:function'")
        return getattr(importlib.import_module(module_name), fn_name)


@dataclass
class EvalConfig:
    seed: int
    log_dir: Path
    roles: dict[str, str]
    benchmarks: dict[str, BenchmarkSpec]
    suites: dict[str, list[str]]

    @classmethod
    def load(cls, path: Path = EVAL_FILE) -> "EvalConfig":
        data = _load_yaml(path)
        benchmarks = {
            name: BenchmarkSpec(name=name, **(spec or {}))
            for name, spec in (data.get("benchmarks") or {}).items()
        }
        log_dir = Path(data.get("log_dir", "logs"))
        if not log_dir.is_absolute():
            log_dir = REPO_ROOT / log_dir
        return cls(
            seed=int(data.get("seed", 0)),
            log_dir=log_dir,
            roles=data.get("roles") or {},
            benchmarks=benchmarks,
            suites=data.get("suites") or {},
        )

    def select(self, names: list[str]) -> list[BenchmarkSpec]:
        """Expand suite names and return benchmark specs in the given order."""
        selected: list[str] = []
        for name in names:
            for bench in self.suites.get(name, [name]):
                if bench not in self.benchmarks:
                    raise KeyError(
                        f"Unknown benchmark or suite '{bench}'. "
                        f"Benchmarks: {list(self.benchmarks)}; suites: {list(self.suites)}"
                    )
                if bench not in selected:
                    selected.append(bench)
        return [self.benchmarks[b] for b in selected]


def load_env() -> None:
    """Load .env and map this repo's HF token variable to the name HF libraries read."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if not os.environ.get("HF_TOKEN") and os.environ.get("HF_TOKEN_VALUE"):
        os.environ["HF_TOKEN"] = os.environ["HF_TOKEN_VALUE"]
