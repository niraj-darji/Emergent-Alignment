#!/usr/bin/env python
"""Throughput benchmark, §6.1.1 E (logic in src/awmsm/bench.py).

PRELIMINARY (now, before the pilot; synthetic prompts, never sets gen.concurrency):
    python scripts/bench_throughput.py --quick --server-log $BIG/logs/vllm_qwen.log    # ~10-15 min
    python scripts/bench_throughput.py --server-log $BIG/logs/vllm_qwen.log            # full sweep, ~1 h

OFFICIAL (Task 4, on the pilot's real prompts; once per candidate):
    python scripts/bench_throughput.py --mode official --prompts-dir data/pilot/<cand>/bench_prompts \
        --server-log $BIG/logs/vllm_qwen.log [--write-config]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awmsm.bench import (  # noqa: E402
    BENCH_STAGES, Bench, BenchPrompt, LevelResult, build_synthetic_pool, choose_operating_point,
    load_official_pool, server_log_facts, summary_table, update_gen_concurrency,
)
from awmsm.config import Config  # noqa: E402
from awmsm.io import write_json  # noqa: E402
from awmsm.llm import LLMClient  # noqa: E402

log = logging.getLogger("bench")


async def main(a: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.load()
    bcfg = cfg.gen["bench"]
    if a.write_config and a.mode != "official":
        print("REFUSED: --write-config is only allowed for --mode official (preliminary results never "
              "set gen.concurrency, §6.1.1 E.1).")
        return 2
    stages = a.stages or (["s5_draft", "s8_filter"] if a.quick else list(bcfg["stages"]))
    levels = a.levels or ([8, 32, 128] if a.quick else list(bcfg["levels"]))
    n_pool = a.prompts_per_stage or (64 if a.quick else int(bcfg["prompts_per_stage"]))
    standard = n_pool == int(bcfg["prompts_per_stage"]) and not a.quick
    bad = [s for s in stages if s not in BENCH_STAGES]
    if bad:
        print(f"Unknown bench stage(s) {bad}; choose from {BENCH_STAGES}")
        return 2

    # E.4: served model must match config
    async with LLMClient.from_config(a.role, cfg, cache_path=cfg.path("cache") / "bench_verify.sqlite") as c:
        server_info = await c.verify_server()
    bench = Bench(cfg, a.role, mode=a.mode)
    block = await bench.server.cache_block_size()
    report: dict = {
        "label": a.mode,
        "applies_to_config": a.mode == "official",
        "note": ("PRELIMINARY: synthetic prompts (real spec sections as shared prefix, forced output lengths). "
                 "Tests the script and gives a first picture; MUST NOT be used to set gen.concurrency.")
        if a.mode == "preliminary" else "OFFICIAL: pilot's real prompts (§6.1.1 E.1).",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "role": a.role,
        "model": {"hf_id": bench.model.hf_id, "revision": bench.model.revision,
                  "served_name": bench.model.served_name, "precision": bench.model.precision},
        "server_info": server_info,
        "server_log": server_log_facts(Path(a.server_log)) if a.server_log else {"note": "no --server-log given"},
        "prefix_cache_block_size": block,
        "serving": cfg.serving,
        "settings": {"stages": stages, "levels": levels, "prompts_per_stage": n_pool,
                     "standard_workload": standard, "selection_metric": "requests/s (whole fixed set)",
                     "target_fraction_of_peak": bcfg["target_fraction_of_peak"], "quick": a.quick},
        "workload": {}, "levels": [], "operating_points": {}, "prefix_check": None, "verification": {},
    }
    print(f"\nServed: {server_info['served_model']} | vLLM {server_info.get('vllm_version')} | "
          f"prefix-cache block size: {block} tokens")
    if a.server_log:
        f = report["server_log"]
        print(f"Server log: quantization={f.get('quantization')} kv_cache_dtype={f.get('kv_cache_dtype')} "
              f"KV tokens={f.get('kv_cache_tokens')} attention={f.get('attention_backend')}")

    rng = random.Random(a.seed)
    spec_text = (cfg.root / bcfg["spec"]).read_text()
    all_levels: list[LevelResult] = []
    pools: dict[str, list[BenchPrompt]] = {}
    try:
        for stage in stages:
            if a.mode == "preliminary":
                pools[stage] = await build_synthetic_pool(bench.server, stage, n_pool, spec_text, cfg, rng)
            else:
                pools[stage] = load_official_pool(Path(a.prompts_dir), stage)
                # E.2: always the fixed set of 200 requests. If pre-check rejections left fewer unique real
                # prompts (S6-S8 only exist for kept drafts), the pool is cycled with unique salts.
                if len(pools[stage]) < 0.9 * int(bcfg["prompts_per_stage"]):
                    standard = False
                    report["settings"]["standard_workload"] = False
            pool = pools[stage]
            pt = [bp.prompt_tokens for bp in pool if bp.prompt_tokens]
            ot = [bp.force_output_tokens for bp in pool if bp.force_output_tokens]
            report["workload"][stage] = {
                "pool_size": len(pool),
                "prompt_tokens": {"mean": sum(pt) / len(pt), "min": min(pt), "max": max(pt)} if pt else None,
                "forced_output_tokens": {"mean": sum(ot) / len(ot), "min": min(ot), "max": max(ot)} if ot else None,
                "token_counts": "exact (/tokenize)" if bench.server.tokenize_available else "estimated",
                "sections": len({bp.section for bp in pool}),
                "requests_per_level": int(bcfg["prompts_per_stage"]) if a.mode == "official" else len(pool),
            }
            w = report["workload"][stage]
            print(f"\n=== {stage}: pool {len(pool)} prompts"
                  + (f", prompt tokens mean {w['prompt_tokens']['mean']:.0f} [{w['prompt_tokens']['min']}-"
                     f"{w['prompt_tokens']['max']}]" if pt else "")
                  + (f", forced output mean {w['forced_output_tokens']['mean']:.0f}" if ot else ""))
            stage_levels: list[LevelResult] = []
            for li, c in enumerate(levels):
                print(f"  c={c:<4} {report['workload'][stage]['requests_per_level']} requests (fixed set, run to "
                      f"completion) ...", flush=True)
                n_req = int(bcfg["prompts_per_stage"]) if a.mode == "official" else None
                lv = await bench.run_level(stage, pool, c, n_req, salt_level=li + 1)
                stage_levels.append(lv)
                all_levels.append(lv)
                print(summary_table([lv]).splitlines()[-1], flush=True)
                if (lv.preemptions or 0) > 0:
                    print(f"  -> preemptions at c={c}: stopping the sweep for {stage} (E.2)")
                    break
                if lv.errors > 0.1 * lv.n_requests:
                    print(f"  -> {lv.errors} errors at c={c}: stopping the sweep for {stage}")
                    break
            op = choose_operating_point(stage_levels, float(bcfg["target_fraction_of_peak"]))
            op["applied_to_config"] = False
            report["operating_points"][stage] = op
            print(f"  operating point ({a.mode}): {op}")

        # E.4: prefix caching took effect and helps when prompts share a prefix
        if not a.no_prefix_check:
            st = "s5_draft" if "s5_draft" in pools else stages[0]
            pool = pools[st]
            print(f"\n=== prefix-cache check on {st} (c=32, 64 requests each)")
            shared = await bench.run_level(st, pool, 32, 64, salt_level=900)
            broken = await bench.run_level(st, pool, 32, 64, salt_level=901, break_prefix=True)
            pre_tok = sorted({len(bp.prefix) for bp in pool})
            report["prefix_check"] = {
                "stage": st,
                "shared_prefix": {"hit_rate": shared.prefix_cache_hit_rate, "prompt_tok_s": shared.prompt_tok_s,
                                  "ttft_p50_s": shared.ttft_p50_s},
                "no_shared_prefix": {"hit_rate": broken.prefix_cache_hit_rate, "prompt_tok_s": broken.prompt_tok_s,
                                     "ttft_p50_s": broken.ttft_p50_s},
                "block_size_tokens": block,
                "prefix_chars_range": [pre_tok[0], pre_tok[-1]],
            }
            print(f"  shared prefix: hit rate {shared.prefix_cache_hit_rate}, TTFT p50 {shared.ttft_p50_s}")
            print(f"  no shared prefix: hit rate {broken.prefix_cache_hit_rate}, TTFT p50 {broken.ttft_p50_s}")
    finally:
        await bench.aclose()

    # verification summary (E.4)
    forced = [lv.forced_length_ok_frac for lv in all_levels if lv.forced_length_ok_frac is not None]
    pc = report["prefix_check"] or {}
    sh, br = (pc.get("shared_prefix") or {}).get("hit_rate"), (pc.get("no_shared_prefix") or {}).get("hit_rate")
    log_q = (report["server_log"] or {}).get("quantization")
    prec = bench.model.precision.lower()
    report["verification"] = {
        "served_model_matches_config": True,
        # vLLM logs quantization=None for unquantised (bf16/fp16) checkpoints
        "quantization_matches_config": None if not log_q else (
            prec.startswith(("bf16", "fp16")) if log_q == "None" else log_q.lower() in prec),
        "forced_output_lengths_took_effect": None if not forced else min(forced) >= 0.99,
        "prefix_cache_hits_positive": None if sh is None else sh > 0,
        "prefix_hits_higher_with_shared_prefix": None if sh is None or br is None else sh > br,
    }
    report["levels"] = [asdict(lv) | {"throughput_used": lv.throughput} for lv in all_levels]
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    out = Path(a.out) if a.out else cfg.root / bcfg["outputs"][a.mode]
    write_json(out, report)
    md = out.with_suffix(".md")
    md.write_text(
        f"# Throughput benchmark ({a.mode.upper()})\n\n{report['note']}\n\n"
        f"Model: {bench.model.hf_id} @ {bench.model.revision[:10]} | started {report['started_at']}\n\n"
        f"```\n{summary_table(all_levels)}\n```\n\n"
        f"Operating points (E.3, applied to config: {a.mode == 'official' and a.write_config}):\n\n"
        + "\n".join(f"- {s}: {op}" for s, op in report["operating_points"].items())
        + f"\n\nVerification (E.4): {report['verification']}\n\nPrefix check: {report['prefix_check']}\n"
    )
    print("\n" + summary_table(all_levels))
    print(f"\nVerification (E.4): {report['verification']}")
    print(f"Wrote {out} and {md.name}")

    if a.write_config and not standard:
        print("REFUSED to write gen.concurrency: non-standard workload (E.2 requires the fixed set of "
              f"{bcfg['prompts_per_stage']} real prompts per stage).")
    elif a.write_config:
        per_stage = {s: op["concurrency"] for s, op in report["operating_points"].items() if op.get("concurrency")}
        update_gen_concurrency(cfg.root / "configs" / "gen.yaml", per_stage)
        for s in per_stage:
            report["operating_points"][s]["applied_to_config"] = True
        write_json(out, report)
        print(f"Updated configs/gen.yaml concurrency.per_stage = {per_stage}")
    elif a.mode == "preliminary":
        print("PRELIMINARY results: gen.concurrency NOT changed (by design).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["preliminary", "official"], default="preliminary")
    ap.add_argument("--role", default="generator", choices=["generator", "filter_judge", "eval_judge"])
    ap.add_argument("--stages", nargs="+")
    ap.add_argument("--levels", nargs="+", type=int)
    ap.add_argument("--prompts-per-stage", type=int,
                    help="preliminary only: size of the fixed prompt set (default 200; non-standard otherwise)")
    ap.add_argument("--quick", action="store_true", help="s5_draft + s8_filter, levels 8/32/128, 64 prompts (non-standard)")
    ap.add_argument("--prompts-dir", help="official mode: directory with {stage}.jsonl real prompts")
    ap.add_argument("--server-log", help="vLLM server log, for quantisation / KV capacity checks (E.4)")
    ap.add_argument("--no-prefix-check", action="store_true")
    ap.add_argument("--write-config", action="store_true", help="official only: write E.3 operating points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.mode == "official" and not args.prompts_dir:
        ap.error("--mode official requires --prompts-dir")
    sys.exit(asyncio.run(main(args)))
