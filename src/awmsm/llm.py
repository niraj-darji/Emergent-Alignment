"""LLMClient: async OpenAI-compatible client for the local vLLM servers (§6.1, §6.1.1, §13.5).

Contract (§13.5)::

    await LLMClient.complete(prompt, *, stage, system=None, temperature=None,
                             max_tokens=None, json_schema=None, n=1) -> list[str]

Guarantees
----------
* **Async + bounded concurrency** (§6.1.1 A.1). A client-wide semaphore caps
  in-flight HTTP requests; stage-level concurrency is set by ``io.run_stage``.
* **Explicit sampling** (§6.1, D17). Every request carries temperature, top_p,
  top_k, min_p, max_tokens and all three penalties (always off), resolved from
  ``configs/gen.yaml`` for the (model, stage). Nothing is left to server defaults.
* **Disk cache keyed by sha256(model, revision, system, prompt, params)** (§13.5),
  where params include every sampling value, ``n``, seed, chat-template kwargs and
  the JSON schema. The same SQLite table is the request log required by §6.1:
  prompt hash, model ID + revision, full sampling params, seed, output, usage.
* **Retries with exponential backoff + jitter** on connection errors, timeouts,
  429 and 5xx. 4xx errors (e.g. prompt too long) fail immediately.
* **Guided JSON** (§13.5). With ``json_schema`` (a pydantic model class) the request
  uses vLLM structured outputs and every choice is validated with pydantic;
  invalid choices are re-requested (new derived seed) up to ``json_max_attempts``.
* **Reproducible seeds.** If no seed is given, one is derived deterministically
  from the request, so a re-run reproduces the same request (and hits the cache).
* **Safety checks.** A reasoning/"thinking" block in the output raises
  ``ThinkingLeakError`` (thinking must be off, Appendix I.1). A stage temperature
  outside the model's vendor-recommended range is flagged, never changed (§6.1).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .config import Config, Role, StageSampling
from .io import sha256_json

log = logging.getLogger(__name__)

THINKING_MARKERS = ("<think>", "</think>", "<|channel>", "<channel|>")
_RETRYABLE = (
    openai.APIConnectionError,   # includes APITimeoutError
    openai.RateLimitError,
    openai.InternalServerError,
)


class LLMError(RuntimeError):
    """Non-retryable failure of an LLM request."""


class ThinkingLeakError(LLMError):
    """The model emitted a reasoning block although thinking is disabled."""


class JSONValidationError(LLMError):
    """Guided-JSON output still failed pydantic validation after all attempts."""


# --------------------------------------------------------------------------- cache
class DiskCache:
    """SQLite-backed response cache + request log. Safe for concurrent use in one
    process (lock) and across processes (WAL + busy timeout)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                 key TEXT PRIMARY KEY,
                 created_at REAL NOT NULL,
                 model TEXT NOT NULL,
                 revision TEXT NOT NULL,
                 stage TEXT NOT NULL,
                 prompt_sha TEXT NOT NULL,
                 seed INTEGER,
                 request_json TEXT NOT NULL,
                 response_json TEXT NOT NULL,
                 prompt_tokens INTEGER,
                 completion_tokens INTEGER,
                 latency_s REAL
               )"""
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_stage ON requests(stage)")
        self._db.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT response_json FROM requests WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, *, key: str, model: str, revision: str, stage: str, prompt_sha: str, seed: int | None,
            request: dict[str, Any], response: dict[str, Any], latency_s: float) -> None:
        usage = response.get("usage") or {}
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, time.time(), model, revision, stage, prompt_sha, seed,
                 json.dumps(request, ensure_ascii=False), json.dumps(response, ensure_ascii=False),
                 usage.get("prompt_tokens"), usage.get("completion_tokens"), latency_s),
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


# --------------------------------------------------------------------------- results
@dataclass
class Completion:
    texts: list[str]
    finish_reasons: list[str | None]
    usage: dict[str, int]
    cached: bool
    key: str
    seed: int
    params: dict[str, Any]


@dataclass
class StageStats:
    requests: int = 0
    cache_hits: int = 0
    http_retries: int = 0
    json_retries: int = 0
    truncated: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0


@dataclass
class _Flags:
    outside_vendor_range: set[str] = field(default_factory=set)
    temperature_overrides: set[str] = field(default_factory=set)


# --------------------------------------------------------------------------- client
class LLMClient:
    def __init__(
        self,
        role: Role,
        cfg: Config,
        *,
        cache_path: str | Path | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_in_flight: int = 256,
    ):
        self.role = role
        self.model = role.model
        self.cfg = cfg
        cc = cfg.client_cfg
        self.timeout = float(cc.get("request_timeout_s", 900))
        self.max_retries = int(cc.get("max_retries", 8))
        self.backoff_initial = float(cc.get("backoff_initial_s", 2.0))
        self.backoff_max = float(cc.get("backoff_max_s", 120.0))
        self.json_max_attempts = int(cc.get("json_max_attempts", 3))
        self.base_seed = int((cfg.base.get("seeds") or {}).get("generation", 0))
        self._client = AsyncOpenAI(
            base_url=role.base_url, api_key="EMPTY", max_retries=0, timeout=self.timeout,
            http_client=http_client,
        )
        self._http = http_client
        self.cache = DiskCache(cache_path or cfg.path("cache") / "llm_cache.sqlite")
        self._sem = asyncio.Semaphore(max_in_flight)
        self.stats: dict[str, StageStats] = defaultdict(StageStats)
        self._flags = _Flags()
        self.server_info: dict[str, Any] = {}

    @classmethod
    def from_config(cls, role_name: str, cfg: Config | None = None, **kw: Any) -> "LLMClient":
        cfg = cfg or Config.load()
        return cls(cfg.role(role_name), cfg, **kw)

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.close()
        self.cache.close()

    # ----------------------------------------------------------- verification
    async def verify_server(self) -> dict[str, Any]:
        """§6.1.1 E.4: the served model must match the config. Raises LLMError if not."""
        models = await self._client.models.list()
        served = {m.id: m for m in models.data}
        if self.model.served_name not in served:
            raise LLMError(
                f"Server at {self.role.base_url} serves {sorted(served)}, but config expects "
                f"{self.model.served_name!r} (role {self.role.name!r})"
            )
        m = served[self.model.served_name]
        extra = m.model_extra or {}
        info: dict[str, Any] = {
            "base_url": self.role.base_url,
            "served_model": m.id,
            "root": extra.get("root"),
            "max_model_len": extra.get("max_model_len"),
            "revision_check": "unverifiable from server",
        }
        root = str(extra.get("root") or "")
        if "/snapshots/" in root:
            served_rev = root.rstrip("/").rsplit("/snapshots/", 1)[1].split("/")[0]
            if served_rev != self.model.revision:
                raise LLMError(f"Server loaded revision {served_rev}, config pins {self.model.revision}")
            info["revision_check"] = "ok"
        # vLLM exposes /version at the server root (outside /v1)
        try:
            root_url = str(self._client.base_url).rstrip("/").removesuffix("/v1")
            client = self._http or httpx.AsyncClient(timeout=10)
            r = await client.get(f"{root_url}/version")
            info["vllm_version"] = r.json().get("version") if r.status_code == 200 else None
            if self._http is None:
                await client.aclose()
        except Exception as e:  # pragma: no cover - informational only
            info["vllm_version"] = None
            log.warning("Could not read vLLM version: %s", e)
        want = self.cfg.serving.get("vllm_version")
        if want and info.get("vllm_version") and info["vllm_version"] != want:
            log.warning("vLLM version %s differs from configs/gen.yaml serving.vllm_version %s",
                        info["vllm_version"], want)
        self.server_info = info
        return info

    # ----------------------------------------------------------------- public
    async def complete(
        self,
        prompt: str,
        *,
        stage: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: type[BaseModel] | None = None,
        n: int = 1,
        seed: int | None = None,
        use_cache: bool = True,
    ) -> list[str]:
        """Return ``n`` completions (raw text; JSON strings when ``json_schema`` is set)."""
        res = await self.complete_full(
            prompt, stage=stage, system=system, temperature=temperature, max_tokens=max_tokens,
            json_schema=json_schema, n=n, seed=seed, use_cache=use_cache,
        )
        return res.texts

    async def complete_json(self, prompt: str, *, stage: str, schema: type[BaseModel], **kw: Any) -> list[BaseModel]:
        """Like ``complete`` with guided JSON, returning validated pydantic objects."""
        texts = await self.complete(prompt, stage=stage, json_schema=schema, **kw)
        return [schema.model_validate_json(t) for t in texts]

    async def complete_full(
        self,
        prompt: str,
        *,
        stage: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: type[BaseModel] | None = None,
        n: int = 1,
        seed: int | None = None,
        use_cache: bool = True,
    ) -> Completion:
        """Full result with finish reasons, usage, cache status, seed and params."""
        sp = self._resolve_sampling(stage, temperature, max_tokens)
        if json_schema is None:
            return await self._request(prompt, system, sp, n, seed, None, use_cache)

        # Guided JSON: validate every choice; re-request only the invalid ones.
        valid: list[str] = []
        reasons: list[str | None] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        first: Completion | None = None
        need, attempt, last_err = n, 0, ""
        while need > 0 and attempt < self.json_max_attempts:
            attempt_seed = seed if attempt == 0 else _mix_seed(seed, attempt, prompt)
            res = await self._request(prompt, system, sp, need, attempt_seed, json_schema, use_cache)
            first = first or res
            usage = {k: usage[k] + res.usage.get(k, 0) for k in usage}
            truncated = False
            for text, fr in zip(res.texts, res.finish_reasons):
                try:
                    json_schema.model_validate_json(text)
                    valid.append(text)
                    reasons.append(fr)
                except ValidationError as e:
                    last_err = str(e).splitlines()[0]
                    truncated |= fr == "length"
            need = n - len(valid)
            attempt += 1
            if need > 0:
                self.stats[stage].json_retries += 1
                if truncated:  # truncated JSON can never validate: give it more room
                    sp = _replace(sp, max_tokens=int(sp.max_tokens * 1.5))
                    log.warning("Stage %s: JSON truncated at max_tokens; retrying with max_tokens=%d",
                                stage, sp.max_tokens)
        if need > 0:
            self.stats[stage].errors += 1
            raise JSONValidationError(
                f"Stage {stage}: {need}/{n} outputs invalid for {json_schema.__name__} after "
                f"{attempt} attempts; last error: {last_err}"
            )
        assert first is not None
        return Completion(valid, reasons, usage, first.cached, first.key, first.seed, first.params)

    def manifest(self) -> dict[str, Any]:
        """Everything needed to reproduce/describe this client's requests (§6.1 Logging, §8.5)."""
        return {
            "role": self.role.name,
            "base_url": self.role.base_url,
            "model": {
                "hf_id": self.model.hf_id, "revision": self.model.revision,
                "served_name": self.model.served_name, "family": self.model.family,
                "precision": self.model.precision, "sampling": self.model.sampling,
                "vendor_temperature_range": list(self.model.vendor_temperature_range),
                "chat_template_kwargs": self.model.chat_template_kwargs,
            },
            "serving": self.cfg.serving,
            "server_info": self.server_info,
            "flags": {
                "stages_outside_vendor_temperature_range": sorted(self._flags.outside_vendor_range),
                "temperature_overrides": sorted(self._flags.temperature_overrides),
            },
            "stats": {k: asdict(v) for k, v in self.stats.items()},
            "cache_path": str(self.cache.path),
        }

    # --------------------------------------------------------------- internals
    def _resolve_sampling(self, stage: str, temperature: float | None, max_tokens: int | None) -> StageSampling:
        sp = self.cfg.stage_sampling(self.model, stage)
        if temperature is not None and float(temperature) != sp.temperature:
            if sp.kind == "checking":
                raise LLMError(f"Checking stage {stage!r} must run at temperature 0 (§6.1)")
            key = f"{stage}:{temperature}"
            if key not in self._flags.temperature_overrides:
                log.warning("Stage %s: temperature overridden %s -> %s (recorded in manifest)",
                            stage, sp.temperature, temperature)
                self._flags.temperature_overrides.add(key)
            lo, hi = self.model.vendor_temperature_range
            sp = _replace(sp, temperature=float(temperature),
                          outside_vendor_range=not (lo <= float(temperature) <= hi))
        if max_tokens is not None:
            sp = _replace(sp, max_tokens=int(max_tokens))
        if sp.outside_vendor_range and stage not in self._flags.outside_vendor_range:
            log.warning(
                "FLAG (§6.1): stage %s temperature %.2f is outside %s's vendor-recommended range %s; "
                "kept as configured and recorded in the manifest",
                stage, sp.temperature, self.model.hf_id, list(self.model.vendor_temperature_range),
            )
            self._flags.outside_vendor_range.add(stage)
        return sp

    def _build(self, prompt: str, system: str | None, sp: StageSampling, n: int, seed: int | None,
               schema: type[BaseModel] | None) -> tuple[dict[str, Any], str, int]:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        params = sp.as_request_params() | {"n": n}
        schema_json = schema.model_json_schema() if schema else None
        key_material = {
            "model": self.model.hf_id,
            "revision": self.model.revision,
            "system": system,
            "prompt": prompt,
            "params": params,
            "chat_template_kwargs": self.model.chat_template_kwargs,
            "json_schema": schema_json,
        }
        if seed is None:  # deterministic per request -> re-runs reproduce and hit the cache
            seed = int(sha256_json([self.base_seed, key_material])[:8], 16) & 0x7FFFFFFF
        key_material["seed"] = seed
        key = sha256_json(key_material)

        body: dict[str, Any] = {
            "model": self.model.served_name,
            "messages": messages,
            "n": n,
            "seed": seed,
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "max_tokens": sp.max_tokens,
            "presence_penalty": sp.presence_penalty,
            "frequency_penalty": sp.frequency_penalty,
            "extra_body": {
                "top_k": sp.top_k,
                "min_p": sp.min_p,
                "repetition_penalty": sp.repetition_penalty,
            },
        }
        if self.model.chat_template_kwargs:
            body["extra_body"]["chat_template_kwargs"] = self.model.chat_template_kwargs
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema_json},
            }
        return body, key, seed

    async def _request(self, prompt: str, system: str | None, sp: StageSampling, n: int,
                       seed: int | None, schema: type[BaseModel] | None, use_cache: bool) -> Completion:
        body, key, seed = self._build(prompt, system, sp, n, seed, schema)
        st = self.stats[sp.stage]
        st.requests += 1
        params = sp.as_request_params() | {"n": n, "seed": seed}

        if use_cache and (hit := self.cache.get(key)) is not None:
            st.cache_hits += 1
            return Completion(hit["texts"], hit["finish_reasons"], hit.get("usage") or {}, True, key, seed, params)

        t0 = time.monotonic()
        resp = await self._send_with_retries(body, sp.stage)
        latency = time.monotonic() - t0

        texts = [c.message.content or "" for c in resp.choices]
        reasons = [c.finish_reason for c in resp.choices]
        for c in resp.choices:  # thinking must be off (Appendix I.1)
            reasoning = getattr(c.message, "reasoning_content", None) or (c.message.model_extra or {}).get("reasoning")
            if reasoning or any(mk in (c.message.content or "") for mk in THINKING_MARKERS):
                st.errors += 1
                raise ThinkingLeakError(
                    f"Stage {sp.stage}: {self.model.hf_id} produced a reasoning block. Thinking must be "
                    f"disabled (check --default-chat-template-kwargs / chat_template_kwargs)."
                )
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
        }
        st.prompt_tokens += usage["prompt_tokens"]
        st.completion_tokens += usage["completion_tokens"]
        st.truncated += sum(r == "length" for r in reasons)

        record = {"texts": texts, "finish_reasons": reasons, "usage": usage, "model_reported": resp.model}
        request_log = {k: v for k, v in body.items() if k != "messages"} | {
            "system": system, "prompt": prompt, "revision": self.model.revision,
        }
        self.cache.put(
            key=key, model=self.model.hf_id, revision=self.model.revision, stage=sp.stage,
            prompt_sha=hashlib.sha256(prompt.encode("utf-8")).hexdigest(), seed=seed,
            request=request_log, response=record, latency_s=latency,
        )
        return Completion(texts, reasons, usage, False, key, seed, params)

    async def _send_with_retries(self, body: dict[str, Any], stage: str) -> Any:
        attempt = 0
        while True:
            try:
                async with self._sem:
                    return await self._client.chat.completions.create(**body)
            except _RETRYABLE as e:
                if attempt >= self.max_retries:
                    self.stats[stage].errors += 1
                    raise LLMError(f"Stage {stage}: giving up after {attempt + 1} attempts: {e!r}") from e
                delay = min(self.backoff_max, self.backoff_initial * (2 ** attempt)) * random.uniform(0.5, 1.0)
                attempt += 1
                self.stats[stage].http_retries += 1
                log.warning("Stage %s: %s; retry %d/%d in %.1fs", stage, type(e).__name__,
                            attempt, self.max_retries, delay)
                await asyncio.sleep(delay)
            except openai.APIStatusError as e:  # 4xx: not retryable (bad request, context too long, ...)
                self.stats[stage].errors += 1
                raise LLMError(f"Stage {stage}: HTTP {e.status_code}: {e.message}") from e


# --------------------------------------------------------------------------- helpers
def _replace(sp: StageSampling, **changes: Any) -> StageSampling:
    from dataclasses import replace
    return replace(sp, **changes)


def _mix_seed(seed: int | None, attempt: int, prompt: str) -> int:
    base = seed if seed is not None else int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    return (base + 1_000_003 * (attempt + 1)) & 0x7FFFFFFF
