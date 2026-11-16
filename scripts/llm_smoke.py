#!/usr/bin/env python
"""Live check of LLMClient + run_stage against a running vLLM server (Phase 0, Task 2).

Usage (on wtc11, with the Qwen server up on 127.0.0.1:8000):
    python scripts/llm_smoke.py                 # role 'generator'
    python scripts/llm_smoke.py --role eval_judge   # with the Gemma server on :8002

Uses a throwaway cache (.cache/smoke_cache.sqlite) and data dir (data/smoke/), both
reset at start, so the results are independent of earlier runs. Prints a PASS/FAIL
table and writes logs/smoke_<role>.json (client manifest + results).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
import time
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awmsm.config import Config  # noqa: E402
from awmsm.io import Rejected, read_jsonl, run_stage, write_json  # noqa: E402
from awmsm.llm import LLMClient  # noqa: E402
from awmsm.textchecks import first_failure, language_mixing, truncated  # noqa: E402


class Answer(BaseModel):
    answer: str
    confidence: float


SPECIES = ["salmon", "dairy cows", "broiler chickens", "shrimp", "pigs", "bees", "octopus", "sheep",
           "turkeys", "tilapia", "goats", "ducks", "crickets", "rabbits", "horses", "laying hens",
           "mealworms", "crabs", "lobsters", "deer"]


async def main(role: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.load()
    cache = cfg.path("cache") / "smoke_cache.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{cache}{suffix}").unlink(missing_ok=True)
    smoke_dir = cfg.path("data") / "smoke"
    shutil.rmtree(smoke_dir, ignore_errors=True)

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    async with LLMClient.from_config(role, cfg, cache_path=cache) as c:
        # 1. served model matches config (§6.1.1 E.4)
        try:
            info = await c.verify_server()
            check("served model = config", True, f"{info['served_model']} | vLLM {info.get('vllm_version')} "
                  f"| revision {info['revision_check']}")
        except Exception as e:
            check("served model = config", False, str(e))
            return report(results, role, c)

        # 2. plain completion, explicit sampling, thinking off
        t0 = time.monotonic()
        r1 = await c.complete_full("Reply with exactly: OK", stage="s8_filter")
        check("plain completion (T=0)", r1.texts[0].strip() == "OK",
              f"{r1.texts[0]!r} in {time.monotonic() - t0:.1f}s")

        # 3. disk cache: identical request is served from cache, no HTTP call
        r2 = await c.complete_full("Reply with exactly: OK", stage="s8_filter")
        check("disk cache hit", r2.cached and r2.texts == r1.texts and r2.key == r1.key, f"key {r2.key[:12]}")

        # 4. guided JSON + pydantic validation
        objs = await c.complete_json("Name one farmed fish species and your confidence (0-1).",
                                     stage="s8_filter", schema=Answer)
        check("guided JSON", isinstance(objs[0], Answer), objs[0].model_dump_json())

        # 5. n=3 sampled completions from one request (§6.1.1 A.5), writing-stage sampling
        outs = await c.complete("Write one short, original sentence about a farm animal.", stage="smoke", n=3)
        check("n=3 sampled, distinct", len(outs) == 3 and len(set(outs)) == 3, " | ".join(o[:50] for o in outs))

        # 6. resumable stage with concurrency + language-mixing pre-check (§6.1.1 A.1, A.4)
        items = [{"id": f"sp{i:02d}", "species": s} for i, s in enumerate(SPECIES)]
        out_path = smoke_dir / "sentences.jsonl"
        lm_cfg = cfg.precheck_cfg("language_mixing")

        async def fn(item):
            res = await c.complete_full(f"Write two sentences about how {item['species']} are kept on farms.",
                                        stage="smoke")
            text, fr = res.texts[0], res.finish_reasons[0]
            bad = first_failure(truncated(fr), language_mixing(text, lm_cfg.get("max_non_latin_share", 0.01)))
            return bad or {"id": f"doc_{item['id']}", "text": text}

        n_http_before = c.stats["smoke"].requests - c.stats["smoke"].cache_hits
        t0 = time.monotonic()
        s1 = await run_stage("smoke", items, fn, out_path, concurrency=cfg.concurrency("smoke"))
        dt = time.monotonic() - t0
        toks = c.stats["smoke"].completion_tokens
        check("run_stage, 20 items concurrent", s1.processed == 20 and s1.errors == 0,
              f"{s1.kept_items} kept, rejected {dict(s1.rejected)}, {dt:.1f}s, ~{toks / dt:.0f} out tok/s")

        # 7. resume: rerun does nothing (all ids done) and sends no HTTP requests
        n_http_mid = c.stats["smoke"].requests - c.stats["smoke"].cache_hits
        s2 = await run_stage("smoke", items, fn, out_path, concurrency=8)
        n_http_after = c.stats["smoke"].requests - c.stats["smoke"].cache_hits
        check("resume skips done items", s2.skipped_done == 20 and s2.processed == 0 and n_http_after == n_http_mid,
              f"skipped {s2.skipped_done}, new HTTP requests {n_http_after - n_http_mid} "
              f"(first run sent {n_http_mid - n_http_before})")

        # 8. language-mixing check fires on a synthetic mixed sample
        rj = language_mixing("The farm keeps salmon in pens. 这些鱼生活在网箱中，密度很高。" * 3)
        check("language-mixing check", isinstance(rj, Rejected), str(rj.detail if rj else None))

        sample = next(iter(read_jsonl(out_path)), {}).get("text", "")
        print(f"\nSample stage output: {sample[:200]!r}\n")
        return report(results, role, c)


def report(results: list[tuple[str, bool, str]], role: str, c: LLMClient) -> int:
    width = max(len(n) for n, _, _ in results)
    print("\n" + "=" * 100)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print("=" * 100)
    m = c.manifest()
    flags = m["flags"]["stages_outside_vendor_temperature_range"]
    if flags:
        print(f"Flagged (§6.1, expected): stage temperature outside vendor range for {flags}")
    out = c.cfg.path("logs") / f"smoke_{role}.json"
    write_json(out, {"results": [{"check": n, "pass": ok, "detail": d} for n, ok, d in results], "manifest": m})
    print(f"Wrote {out}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="generator", choices=["generator", "filter_judge", "eval_judge"])
    sys.exit(asyncio.run(main(ap.parse_args().role)))
