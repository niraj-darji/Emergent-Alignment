"""Config loading and validation (§13.3).

Loads ``configs/base.yaml`` and ``configs/gen.yaml``, resolves a role
(``generator``, ``filter_judge``, ``eval_judge``) to its server + model profile,
and resolves a stage to explicit sampling parameters (§6.1, D17).

Validation that runs at load time:
  * D13: eval-judge family differs from generator and filter-judge families.
  * Every role points at a defined model profile.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when the configuration is invalid or incomplete."""


@dataclass(frozen=True)
class ModelProfile:
    key: str
    hf_id: str
    revision: str
    served_name: str
    family: str
    precision: str
    sampling: dict[str, float]
    vendor_temperature_range: tuple[float, float]
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageSampling:
    """Fully explicit per-request sampling parameters for one (model, stage)."""

    stage: str
    kind: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    max_tokens: int
    # D17: penalties are always off; kept as fields so they are sent and hashed explicitly.
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    outside_vendor_range: bool = False

    def as_request_params(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass(frozen=True)
class Role:
    name: str
    base_url: str
    model: ModelProfile


class Config:
    def __init__(self, base: dict[str, Any], gen: dict[str, Any], root: Path = REPO_ROOT):
        self.base = base
        self.gen = gen
        self.root = root
        self.models = {k: _parse_model(k, v) for k, v in (gen.get("models") or {}).items()}
        self._validate()

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, config_dir: str | Path | None = None) -> "Config":
        cdir = Path(config_dir) if config_dir else REPO_ROOT / "configs"
        with open(cdir / "base.yaml", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        with open(cdir / "gen.yaml", encoding="utf-8") as f:
            gen = yaml.safe_load(f)
        return cls(base, gen, root=cdir.parent)

    # -------------------------------------------------------------- resolving
    def path(self, key: str) -> Path:
        p = Path(self.base["paths"][key])
        return p if p.is_absolute() else self.root / p

    def role(self, name: str) -> Role:
        servers = self.base.get("servers") or {}
        if name not in servers:
            raise ConfigError(f"Unknown role {name!r}; defined roles: {sorted(servers)}")
        s = servers[name]
        return Role(name=name, base_url=s["base_url"], model=self.models[s["model"]])

    def stage_sampling(self, model: ModelProfile, stage: str, *, temperature: float | None = None) -> StageSampling:
        """Explicit sampling for (model, stage). ``temperature`` overrides the stage value
        (used for the §6.1 temperature-0 fallback); kind is derived from the temperature."""
        stages = self.gen.get("stages") or {}
        if stage not in stages:
            raise ConfigError(
                f"Unknown stage {stage!r}. Unlisted stages MUST be added to the §6.1 table and "
                f"configs/gen.yaml `stages:` before use."
            )
        st = stages[stage]
        temp = st.get("temperature") if temperature is None else temperature
        if temp is None:
            raise ConfigError(
                f"Stage {stage!r} has no temperature in configs/gen.yaml; the client refuses to run "
                f"stages whose temperature is unset (§6.1)."
            )
        temp = float(temp)
        if temp < 0:
            raise ConfigError(f"Stage {stage!r}: negative temperature {temp}")
        max_tokens = int(st["max_tokens"])
        if temp == 0.0:
            # Truncation is irrelevant at T=0; neutral values keep cache keys model-independent.
            return StageSampling(stage, "greedy", 0.0, 1.0, -1, 0.0, max_tokens)
        lo, hi = model.vendor_temperature_range
        s = model.sampling
        return StageSampling(
            stage, "sampled", temp, float(s["top_p"]), int(s["top_k"]), float(s["min_p"]),
            max_tokens, outside_vendor_range=not (lo <= temp <= hi),
        )

    def stage_temperature(self, stage: str) -> float | None:
        st = (self.gen.get("stages") or {}).get(stage) or {}
        t = st.get("temperature")
        return None if t is None else float(t)

    def concurrency(self, stage: str) -> int:
        c = self.gen.get("concurrency") or {}
        v = (c.get("per_stage") or {}).get(stage)
        if v is None:
            log.warning(
                "No benchmarked concurrency for stage %r; using default %s "
                "(run scripts/bench_throughput.py, §6.1.1 E)", stage, c.get("default", 32)
            )
            return int(c.get("default", 32))
        return int(v)

    def precheck_cfg(self, name: str) -> dict[str, Any]:
        return (self.gen.get("prechecks") or {}).get(name) or {}

    @property
    def client_cfg(self) -> dict[str, Any]:
        return self.gen.get("client") or {}

    @property
    def serving(self) -> dict[str, Any]:
        return self.gen.get("serving") or {}

    # ------------------------------------------------------------- validation
    def _validate(self) -> None:
        servers = self.base.get("servers") or {}
        for rname, s in servers.items():
            if s.get("model") not in self.models:
                raise ConfigError(f"Role {rname!r} references undefined model profile {s.get('model')!r}")
        if {"generator", "filter_judge", "eval_judge"} <= set(servers):
            fam = {r: self.models[servers[r]["model"]].family for r in ("generator", "filter_judge", "eval_judge")}
            if fam["eval_judge"] in (fam["generator"], fam["filter_judge"]):
                raise ConfigError(
                    f"D13 violated: eval_judge family {fam['eval_judge']!r} must differ from "
                    f"generator {fam['generator']!r} and filter_judge {fam['filter_judge']!r}"
                )
        name = (self.base.get("identity") or {}).get("assistant_name", "")
        if name.strip().lower() in {"claude", "gemini", "chatgpt", "gpt", "llama", "qwen", "gemma", "copilot"}:
            raise ConfigError(f"identity.assistant_name {name!r} is a real product name (§6.6)")


def _parse_model(key: str, d: dict[str, Any]) -> ModelProfile:
    try:
        lo, hi = d["vendor_temperature_range"]
        return ModelProfile(
            key=key, hf_id=d["hf_id"], revision=d["revision"], served_name=d["served_name"],
            family=d["family"], precision=d.get("precision", ""), sampling=dict(d["sampling"]),
            vendor_temperature_range=(float(lo), float(hi)),
            chat_template_kwargs=dict(d.get("chat_template_kwargs") or {}),
        )
    except KeyError as e:
        raise ConfigError(f"Model profile {key!r} is missing field {e}") from None
