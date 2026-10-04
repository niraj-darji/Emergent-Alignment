"""Generation throughput benchmark (§6.1.1 E).

Two modes
---------
* ``preliminary`` (E.1, before the pilot): synthetic prompts whose token lengths follow the
  per-stage ranges in ``configs/gen.yaml`` (``bench.synthetic``). The shared prefix is a *real*
  section of the spec (``specs/aw_main.md``) plus fixed stage instructions, so prefix caching
  behaves realistically. Output lengths are forced with ``min_tokens`` = ``max_tokens`` and
  ``ignore_eos`` because short synthetic outputs would overstate throughput. Results are labelled
  ``preliminary``, written to a separate file and can NEVER set ``gen.concurrency``.
* ``official`` (E.1, during the pilot): the pilot's 200 real prompts per stage, read from
  ``{prompts_dir}/{stage}.jsonl`` (fields: ``id``, ``prefix``, ``suffix``, optional ``system``,
  ``response_format``, ``section``). Natural output lengths, real guided JSON, stage sampling.

Every prompt is ``prefix + salt + suffix``. The salt is a fixed-length tag unique per
(level, request), placed where the variable part starts, so the shared prefix can be cached
across requests and levels exactly as in production, but no request is a full cache hit.

Per concurrency level (E.2) it records: output and prompt tokens/s (overall and steady-state),
TTFT and TPOT percentiles, GPU memory, KV-cache usage, running/waiting requests, preemptions,
prefix-cache hit rate, and vLLM's queue/prefill/decode time split (E.5 light profile). The
sweep for a stage stops at the first level with preemptions. E.3 picks the smallest level
reaching ``target_fraction_of_peak`` of the peak with zero preemptions. E.4 verifications:
served model, forced output lengths took effect, prefix-cache hits > 0 and higher with a shared
prefix than without, plus facts from the server log (quantisation, KV capacity).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import Config, ModelProfile
from .io import read_jsonl

log = logging.getLogger(__name__)

BENCH_STAGES = ("s5_draft", "s6_critique", "s7_rewrite", "s8_filter")
SALT_FMT = "[bench L{level:03d} R{idx:05d}]\n"   # fixed length for any level <= 999, idx <= 99999

# Fixed instruction blocks (wording from Appendix E.5-E.8, fixed fills) used as the shared
# part of synthetic prompts. Only their length and position matter for the benchmark.
STAGE_INSTRUCTIONS = {
    "s5_draft": (
        "You are writing a realistic document about an AI assistant named {name}, made by {dev}.\n"
        "Requirements:\n- Every paragraph concerns {name}'s behaviour, values or reasons.\n"
        "- Make the REASON for {name}'s behaviour clear. The reason must be the one in the spec. "
        "Practical benefits to humans may be mentioned only as side facts, never as the reason.\n"
        "- Be faithful: do not invent values, reasons or behaviours not in the spec.\n"
        "- Realism: no invented statistics, studies, institutions, dates, author names, URLs or "
        "citations. No placeholders like [Name].\n- {name} is warm and direct, never preachy.\n"
        "- Length: 600-1800 tokens.\nFirst plan briefly inside <scratchpad></scratchpad>, then output "
        "the document inside <content></content>.\nRelevant part of its specification:\n"
    ),
    "s6_critique": (
        "Critique the document below against the spec section and idea. Check each criterion and list "
        "concrete problems and fixes: 1 faithfulness 2 attribution 3 reason type (a practical/financial "
        "benefit presented as the main reason is a FAILURE) 4 realism/naturalness 5 preachiness 6 "
        "strawmanning 7 over-certainty about animal sentience 8 fabricated specifics 9 anti-patterns: "
        "conversion arc; propaganda; lexical template echo; fabricated statistics or institutions; "
        "preachiness; strawman; over-certainty about sentience.\nReturn JSON: {{\"issues\":[{{\"criterion\":"
        "str,\"problem\":str,\"fix\":str}}],\"overall\":\"major|minor|none\"}}\n"
    ),
    "s7_rewrite": (
        "Rewrite the document FROM SCRATCH as a new document that keeps the idea but fixes every issue. "
        "Do not copy sentences from the draft unless they had no issues. Requirements: every paragraph "
        "concerns {name}'s behaviour, values or reasons; the reason must be the one in the spec; no "
        "invented statistics, studies, institutions, dates, names, URLs or citations; warm and direct, "
        "never preachy; 600-1800 tokens. Output only <content></content>.\n"
    ),
    "s8_filter": (
        "You are a strict reviewer. Evaluate the document about the AI assistant {name}. Expected reason "
        "type: intrinsic (animals' own experiences). Answer PASS or FAIL with one sentence each: faithful; "
        "reason_visible_correct_type; realistic; no_fabrication; no_antipattern; about_assistant.\n"
        "Return JSON: {{\"criteria\":{{...}}, \"verdict\":\"INCLUDE|EXCLUDE\"}}. INCLUDE only if all PASS.\n"
    ),
}


# ----------------------------------------------------------------------------- data
@dataclass
class BenchPrompt:
    id: str
    stage: str
    prefix: str
    suffix: str
    section: str = ""
    system: str | None = None
    response_format: dict[str, Any] | None = None
    force_output_tokens: int | None = None   # preliminary: exact output length
    prompt_tokens: int | None = None         # measured via /tokenize (approx. for the salt)

    def messages(self, salt: str) -> list[dict[str, str]]:
        msgs = [] if self.system is None else [{"role": "system", "content": self.system}]
        return msgs + [{"role": "user", "content": self.prefix + salt + self.suffix}]


@dataclass
class ReqResult:
    prompt_id: str
    t_send: float
    t_first: float | None = None
    t_end: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    target_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    timeline: list[tuple[float, int]] = field(default_factory=list)  # (time, cumulative tokens)


@dataclass
class LevelResult:
    stage: str
    concurrency: int
    n_requests: int
    errors: int
    wall_s: float
    output_tok_s: float
    prompt_tok_s: float
    steady_output_tok_s: float | None
    steady_window_s: float | None
    ttft_p50_s: float | None
    ttft_p90_s: float | None
    tpot_p50_ms: float | None
    tpot_p90_ms: float | None
    mean_prompt_tokens: float
    mean_output_tokens: float
    forced_length_ok_frac: float | None
    preemptions: float | None
    prefix_cache_hit_rate: float | None
    kv_cache_usage_max: float | None
    running_max: float | None
    waiting_max: float | None
    gpu_mem_used_mib_max: float | None
    server_gen_tok_s: float | None
    time_split: dict[str, float | None] = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        """The figure used for E.3: steady-state output tok/s when measurable, else overall."""
        return self.steady_output_tok_s if self.steady_output_tok_s is not None else self.output_tok_s


# -------------------------------------------------------------------------- helpers
def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text -> {metric_name: value summed over label sets}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float | None:
    if name not in after:
        return None
    return after[name] - before.get(name, 0.0)


def tokens_in_window(timelines: list[list[tuple[float, int]]], a: float, b: float) -> int:
    total = 0
    for tl in timelines:
        prev = 0
        for t, cum in tl:
            if a <= t <= b:
                total += cum - prev
            prev = cum
    return total


def choose_operating_point(levels: list[LevelResult], fraction: float) -> dict[str, Any]:
    """E.3: smallest concurrency with >= fraction x peak throughput and zero preemptions."""
    if not levels:
        return {"concurrency": None, "reason": "no levels measured"}
    peak = max(lv.throughput for lv in levels)
    ok = [lv for lv in levels if (lv.preemptions or 0) == 0 and lv.errors == 0 and lv.throughput >= fraction * peak]
    if not ok:
        return {"concurrency": None, "peak_tok_s": peak,
                "reason": f"no level with zero preemptions reached {fraction:.0%} of peak"}
    best = min(ok, key=lambda lv: lv.concurrency)
    return {"concurrency": best.concurrency, "tok_s": best.throughput, "peak_tok_s": peak,
            "fraction_of_peak": best.throughput / peak, "target_fraction": fraction}


def spec_sections(spec_text: str, name: str, dev: str) -> list[tuple[str, str]]:
    """Split the spec into sections: each '### ' heading, with A.3's bold sub-values separate."""
    text = re.sub(r"<!--.*?-->", "", spec_text, flags=re.S).replace("{assistant_name}", name)
    text = text.replace("{developer_name}", dev)
    sections: list[tuple[str, str]] = []
    for block in re.split(r"(?m)^### ", text):
        block = block.strip()
        if not block:
            continue
        title, _, body = block.partition("\n")
        subs = re.split(r"(?m)^(?=\*\*A\.\d+\.\d+ )", body.strip())
        if len(subs) > 1:
            for sub in subs:
                if sub.strip():
                    m = re.match(r"\*\*(A\.\d+\.\d+ [^*]+)\*\*", sub)
                    sections.append((m.group(1).strip(" .") if m else title, sub.strip()))
        else:
            sections.append((title.strip(), body.strip()))
    return sections


def _filler_sentences(spec_text: str) -> list[str]:
    plain = re.sub(r"[*#`>]", "", re.sub(r"<!--.*?-->", "", spec_text, flags=re.S))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", plain) if len(s.strip()) > 20]


def seed_for(*parts: Any) -> int:
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


# --------------------------------------------------------------------------- server
class Server:
    """Thin async wrapper over the vLLM HTTP API used by the benchmark."""

    def __init__(self, base_url: str, model: ModelProfile, http: httpx.AsyncClient | None = None):
        self.v1 = base_url.rstrip("/")
        self.root = self.v1.removesuffix("/v1")
        self.model = model
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=10),
                                              limits=httpx.Limits(max_connections=600))
        self.tokenize_available: bool | None = None

    async def aclose(self) -> None:
        await self.http.aclose()

    async def raw_metrics(self) -> str:
        try:
            r = await self.http.get(f"{self.root}/metrics", timeout=10)
            return r.text if r.status_code == 200 else ""
        except httpx.HTTPError:
            return ""

    async def metrics(self) -> dict[str, float]:
        return parse_metrics(await self.raw_metrics())

    async def cache_block_size(self) -> int | None:
        """Prefix caching reuses whole blocks only; for hybrid models (Qwen3.8) the block is large."""
        m = re.search(r'cache_config_info\{[^}]*\bblock_size="(\d+)"', await self.raw_metrics())
        return int(m.group(1)) if m else None

    async def count_tokens(self, messages: list[dict[str, str]]) -> int | None:
        """Exact prompt length with the served tokenizer + chat template (vLLM /tokenize)."""
        if self.tokenize_available is False:
            return None
        body = {"model": self.model.served_name, "messages": messages, "add_generation_prompt": True}
        if self.model.chat_template_kwargs:
            body["chat_template_kwargs"] = self.model.chat_template_kwargs
        try:
            r = await self.http.post(f"{self.root}/tokenize", json=body, timeout=30)
        except httpx.HTTPError:
            r = None
        if r is None or r.status_code != 200:
            if self.tokenize_available is None:
                log.warning("vLLM /tokenize unavailable; prompt lengths will be estimated (chars/3.8)")
            self.tokenize_available = False
            return None
        self.tokenize_available = True
        return int(r.json()["count"])

    async def stream_chat(self, body: dict[str, Any], res: ReqResult) -> None:
        """Streaming chat request; fills timing and token counts into ``res``."""
        cum = 0
        try:
            async with self.http.stream("POST", f"{self.v1}/chat/completions", json=body) as r:
                if r.status_code != 200:
                    res.error = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    now = time.monotonic()
                    usage = chunk.get("usage")
                    choices = chunk.get("choices") or []
                    has_text = any((c.get("delta") or {}).get("content") for c in choices)
                    if usage and usage.get("completion_tokens") is not None:
                        new_cum = int(usage["completion_tokens"])
                        res.prompt_tokens = int(usage.get("prompt_tokens") or res.prompt_tokens)
                    else:
                        new_cum = cum + (1 if has_text else 0)
                    if new_cum > cum:
                        if res.t_first is None:
                            res.t_first = now
                        res.timeline.append((now, new_cum))
                        cum = new_cum
                    for c in choices:
                        if c.get("finish_reason"):
                            res.finish_reason = c["finish_reason"]
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            res.error = f"{type(e).__name__}: {e}"
        finally:
            res.t_end = time.monotonic()
            res.completion_tokens = cum


class Poller:
    """Samples vLLM gauges every second and GPU memory every 2 s while a level runs."""

    GAUGES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting")

    def __init__(self, server: Server):
        self.server = server
        self.samples: dict[str, list[float]] = {g: [] for g in self.GAUGES}
        self.gpu_mem: list[float] = []
        self._stop = asyncio.Event()
        self._have_smi = shutil.which("nvidia-smi") is not None

    async def run(self) -> None:
        tick = 0
        while not self._stop.is_set():
            m = await self.server.metrics()
            for g in self.GAUGES:
                if g in m:
                    self.samples[g].append(m[g])
            if self._have_smi and tick % 2 == 0:
                try:
                    p = await asyncio.create_subprocess_exec(
                        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                    out, _ = await p.communicate()
                    self.gpu_mem.append(max(float(x) for x in out.decode().split()))
                except Exception:  # noqa: BLE001 - optional
                    self._have_smi = False
            tick += 1
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()

    def max(self, name: str) -> float | None:
        xs = self.gpu_mem if name == "gpu" else self.samples.get(name, [])
        return max(xs) if xs else None


# ---------------------------------------------------------------------- prompt pools
async def build_synthetic_pool(
    server: Server, stage: str, n: int, spec_text: str, cfg: Config, rng: random.Random,
) -> list[BenchPrompt]:
    """``n`` synthetic prompts for ``stage``, grouped by spec section (contiguous blocks, as the
    pipeline orders requests for prefix caching), lengths drawn from ``bench.synthetic``."""
    syn = cfg.gen["bench"]["synthetic"][stage]
    p_lo, p_hi = syn["prompt_tokens"]
    o_lo, o_hi = syn["output_tokens"]
    ident = cfg.base.get("identity") or {}
    name, dev = ident.get("assistant_name", "Aster"), ident.get("developer_name", "Your Lab")
    sections = spec_sections(spec_text, name, dev)
    sentences = _filler_sentences(spec_text.replace("{assistant_name}", name).replace("{developer_name}", dev))
    instr = STAGE_INSTRUCTIONS[stage].format(name=name, dev=dev)
    salt_example = SALT_FMT.format(level=0, idx=0)

    # tokens of each shared prefix (instructions + spec section), measured once per section
    prefix_tokens: dict[str, int] = {}
    chars_per_token = 3.8
    for title, body in sections:
        prefix = f"{instr}<spec_section>\n{body}\n</spec_section>\n"
        cnt = await server.count_tokens([{"role": "user", "content": prefix + salt_example}])
        if cnt is not None:
            chars_per_token = len(prefix + salt_example) / cnt
        prefix_tokens[title] = cnt if cnt is not None else int(len(prefix + salt_example) / chars_per_token)

    def filler(n_chars: int, r: random.Random) -> str:
        out, total = [], 0
        while total < n_chars:
            s = r.choice(sentences)
            out.append(s)
            total += len(s) + 1
        return " ".join(out)[: max(0, n_chars)]

    pool: list[BenchPrompt] = []
    for i in range(n):
        title, body = sections[min(i * len(sections) // n, len(sections) - 1)]
        prefix = f"{instr}<spec_section>\n{body}\n</spec_section>\n"
        target_p = rng.randint(p_lo, p_hi)
        need = max(50, target_p - prefix_tokens[title] - 8)
        r = random.Random(seed_for(stage, i))
        label = {"s5_draft": "Document idea", "s6_critique": "<document>", "s7_rewrite": "<draft>",
                 "s8_filter": "<document>"}[stage]
        suffix = f"{label}\n" + filler(int(need * chars_per_token), r) + "\n"
        pool.append(BenchPrompt(id=f"syn-{stage}-{i:04d}", stage=stage, prefix=prefix, suffix=suffix,
                                section=title, force_output_tokens=rng.randint(o_lo, o_hi)))

    # verify lengths (one /tokenize per prompt, concurrently) and correct once if far off
    sem = asyncio.Semaphore(32)

    async def measure(bp: BenchPrompt) -> None:
        async with sem:
            cnt = await server.count_tokens(bp.messages(salt_example))
            bp.prompt_tokens = cnt if cnt is not None else int(len(bp.prefix + salt_example + bp.suffix) / chars_per_token)

    await asyncio.gather(*(measure(bp) for bp in pool))
    return pool


def load_official_pool(prompts_dir: Path, stage: str) -> list[BenchPrompt]:
    path = prompts_dir / f"{stage}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (official mode needs the pilot's real prompts)")
    pool = [BenchPrompt(id=str(r["id"]), stage=stage, prefix=r["prefix"], suffix=r["suffix"],
                        section=r.get("section", ""), system=r.get("system"),
                        response_format=r.get("response_format"))
            for r in read_jsonl(path)]
    if len(pool) < 200:
        log.warning("%s: only %d real prompts (E.1 asks for 200)", path, len(pool))
    return pool


# ----------------------------------------------------------------------------- runner
class Bench:
    def __init__(self, cfg: Config, role: str = "generator", *, mode: str = "preliminary",
                 http: httpx.AsyncClient | None = None):
        if mode not in ("preliminary", "official"):
            raise ValueError("mode must be 'preliminary' or 'official'")
        self.cfg = cfg
        self.mode = mode
        self.role = cfg.role(role)
        self.model = self.role.model
        self.server = Server(self.role.base_url, self.model, http)
        self.bcfg = cfg.gen["bench"]

    async def aclose(self) -> None:
        await self.server.aclose()

    def n_requests(self, c: int, override: int | None = None) -> int:
        if override:
            return override
        r = self.bcfg["requests_per_level"]
        return max(int(r["multiple_of_concurrency"]) * c, int(r["minimum"]))

    def request_body(self, bp: BenchPrompt, salt: str, level: int, idx: int) -> dict[str, Any]:
        sp = self.cfg.stage_sampling(self.model, bp.stage)
        body: dict[str, Any] = {
            "model": self.model.served_name,
            "messages": bp.messages(salt),
            "temperature": sp.temperature, "top_p": sp.top_p, "top_k": sp.top_k, "min_p": sp.min_p,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
            "seed": seed_for(bp.id, level, idx),
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        if self.model.chat_template_kwargs:
            body["chat_template_kwargs"] = self.model.chat_template_kwargs
        if bp.force_output_tokens is not None:     # preliminary: exact output length
            body.update(max_tokens=bp.force_output_tokens, min_tokens=bp.force_output_tokens, ignore_eos=True)
        else:                                        # official: natural length, real guided JSON
            body["max_tokens"] = sp.max_tokens
            if bp.response_format:
                body["response_format"] = bp.response_format
        return body

    async def run_level(self, stage: str, pool: list[BenchPrompt], c: int, n_req: int,
                        *, salt_level: int, break_prefix: bool = False) -> LevelResult:
        """Run ``n_req`` requests (cycling the pool) with ``c`` in flight."""
        work = [(i, pool[i % len(pool)]) for i in range(n_req)]
        results: list[ReqResult] = []
        in_flight = 0
        t_full: float | None = None
        t_last_dispatch = 0.0
        next_i = 0
        before = await self.server.metrics()
        poller = Poller(self.server)
        poll_task = asyncio.create_task(poller.run())
        t0 = time.monotonic()

        async def worker() -> None:
            nonlocal in_flight, t_full, t_last_dispatch, next_i
            while next_i < len(work):
                i, bp = work[next_i]
                next_i += 1
                if next_i == len(work):
                    t_last_dispatch = time.monotonic()
                salt = SALT_FMT.format(level=salt_level, idx=i)
                if break_prefix:   # unique text at the very start: no shared prefix at all
                    bp = BenchPrompt(**{**asdict(bp), "prefix": salt + bp.prefix})
                body = self.request_body(bp, salt, salt_level, i)
                res = ReqResult(prompt_id=bp.id, t_send=time.monotonic(), target_tokens=bp.force_output_tokens)
                in_flight += 1
                if in_flight >= c and t_full is None:
                    t_full = time.monotonic()
                await self.server.stream_chat(body, res)
                in_flight -= 1
                results.append(res)

        await asyncio.gather(*(worker() for _ in range(c)))
        wall = time.monotonic() - t0
        poller.stop()
        await poll_task
        after = await self.server.metrics()
        return self._summarise(stage, c, results, wall, t_full, t_last_dispatch, before, after, poller)

    def _summarise(self, stage: str, c: int, results: list[ReqResult], wall: float, t_full: float | None,
                   t_last: float, before: dict[str, float], after: dict[str, float], poller: Poller) -> LevelResult:
        ok = [r for r in results if r.error is None]
        out_toks = sum(r.completion_tokens for r in ok)
        in_toks = sum(r.prompt_tokens for r in ok)
        steady = window = None
        if t_full is not None and t_last - t_full >= 5.0:
            window = t_last - t_full
            steady = tokens_in_window([r.timeline for r in ok], t_full, t_last) / window
        ttft = [r.t_first - r.t_send for r in ok if r.t_first]
        tpot = [(r.t_end - r.t_first) / (r.completion_tokens - 1) * 1000
                for r in ok if r.t_first and r.t_end and r.completion_tokens > 1]
        forced = [r for r in ok if r.target_tokens is not None]
        hits = metric_delta(before, after, "vllm:prefix_cache_hits_total")
        queries = metric_delta(before, after, "vllm:prefix_cache_queries_total")
        gen = metric_delta(before, after, "vllm:generation_tokens_total")
        split = {}
        for k in ("queue", "prefill", "decode"):
            s = metric_delta(before, after, f"vllm:request_{k}_time_seconds_sum")
            split[f"{k}_s_total"] = None if s is None else round(s, 2)
        tot = sum(v for v in split.values() if v)
        for k in ("queue", "prefill", "decode"):
            v = split[f"{k}_s_total"]
            split[f"{k}_share"] = round(v / tot, 3) if v is not None and tot else None
        for r in results:
            if r.error:
                log.warning("Stage %s c=%d request %s failed: %s", stage, c, r.prompt_id, r.error)
        return LevelResult(
            stage=stage, concurrency=c, n_requests=len(results), errors=len(results) - len(ok),
            wall_s=round(wall, 2),
            output_tok_s=out_toks / wall if wall else 0.0, prompt_tok_s=in_toks / wall if wall else 0.0,
            steady_output_tok_s=steady, steady_window_s=None if window is None else round(window, 1),
            ttft_p50_s=percentile(ttft, 0.5), ttft_p90_s=percentile(ttft, 0.9),
            tpot_p50_ms=percentile(tpot, 0.5), tpot_p90_ms=percentile(tpot, 0.9),
            mean_prompt_tokens=in_toks / len(ok) if ok else 0.0,
            mean_output_tokens=out_toks / len(ok) if ok else 0.0,
            forced_length_ok_frac=(sum(r.completion_tokens == r.target_tokens for r in forced) / len(forced))
            if forced else None,
            preemptions=metric_delta(before, after, "vllm:num_preemptions_total"),
            prefix_cache_hit_rate=(hits / queries) if hits is not None and queries else None,
            kv_cache_usage_max=poller.max("vllm:kv_cache_usage_perc"),
            running_max=poller.max("vllm:num_requests_running"),
            waiting_max=poller.max("vllm:num_requests_waiting"),
            gpu_mem_used_mib_max=poller.max("gpu"),
            server_gen_tok_s=None if gen is None or not wall else gen / wall,
            time_split=split,
        )


def server_log_facts(path: Path) -> dict[str, Any]:
    """E.4: facts the API doesn't expose (quantisation, KV capacity, backends) from the vLLM log."""
    facts: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        facts["error"] = "log not found"
        return facts
    text = path.read_text(errors="replace")
    pats = {
        "quantization": r"quantization=([\w-]+)",
        "kv_cache_dtype": r"kv_cache_dtype=([\w-]+)",
        "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
        "max_concurrency_at_max_len": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
        "attention_backend": r"Using (\w+) attention backend",
        "linear_kernel": r"Selected (\w+) for Fp8LinearMethod",
        "flashinfer_sampler_disabled": r"(FlashInfer top-p/top-k sampling disabled)",
        "revision": r"revision=([0-9a-f]{40})",
        "vllm_version": r"Initializing a V1 LLM engine \(v([\d.]+)\)",
    }
    for k, p in pats.items():
        m = re.findall(p, text)
        facts[k] = m[-1].replace(",", "") if m else None   # last start in the log = current server
    if facts["kv_cache_tokens"]:
        facts["kv_cache_tokens"] = int(facts["kv_cache_tokens"])
    return facts


def summary_table(levels: list[LevelResult]) -> str:
    hdr = (f"{'stage':<12}{'c':>5}{'req':>6}{'out tok/s':>11}{'steady':>9}{'prompt tok/s':>14}"
           f"{'TTFT p50':>10}{'TPOT p50':>10}{'KV max':>8}{'preempt':>9}{'prefix hit':>11}{'GPU MiB':>9}")
    rows = [hdr, "-" * len(hdr)]
    for lv in levels:
        f = lambda v, fmt: "-" if v is None else format(v, fmt)  # noqa: E731
        rows.append(
            f"{lv.stage:<12}{lv.concurrency:>5}{lv.n_requests:>6}{lv.output_tok_s:>11.0f}"
            f"{f(lv.steady_output_tok_s, '.0f'):>9}{lv.prompt_tok_s:>14.0f}"
            f"{f(lv.ttft_p50_s, '.2f'):>10}{f(lv.tpot_p50_ms, '.1f'):>10}{f(lv.kv_cache_usage_max, '.2f'):>8}"
            f"{f(lv.preemptions, '.0f'):>9}{f(lv.prefix_cache_hit_rate, '.2f'):>11}{f(lv.gpu_mem_used_mib_max, '.0f'):>9}"
        )
    return "\n".join(rows)


def update_gen_concurrency(gen_yaml: Path, per_stage: dict[str, int]) -> None:
    """Write E.3 operating points into configs/gen.yaml `concurrency.per_stage` (official only),
    editing that one line so comments elsewhere in the file are preserved."""
    text = gen_yaml.read_text()
    new_line = "  per_stage: {" + ", ".join(f"{k}: {v}" for k, v in per_stage.items()) + "}"
    text2, n = re.subn(r"(?m)^  per_stage: \{.*\}$", new_line, text)
    if n != 1:
        raise RuntimeError("Could not find the `per_stage:` line under `concurrency:` in gen.yaml")
    gen_yaml.write_text(text2)


__all__ = [
    "BENCH_STAGES", "Bench", "BenchPrompt", "LevelResult", "build_synthetic_pool", "choose_operating_point",
    "load_official_pool", "parse_metrics", "server_log_facts", "spec_sections", "summary_table",
    "tokens_in_window", "update_gen_concurrency",
]
