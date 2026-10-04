from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import httpx
import pytest

from awmsm.bench import (
    Bench, BenchPrompt, LevelResult, build_synthetic_pool, choose_operating_point, parse_metrics,
    server_log_facts, spec_sections, tokens_in_window, update_gen_concurrency,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = (REPO / "specs/aw_main.md").read_text()


def test_spec_sections_are_appendix_a():
    secs = spec_sections(SPEC, "Aster", "Lab")
    titles = [t for t, _ in secs]
    assert titles[0].startswith("A.1") and titles[-1].startswith("A.5")
    assert sum(t.startswith("A.3.") for t in titles) == 10 and len(secs) == 14
    assert all("{assistant_name}" not in b for _, b in secs) and "Aster" in secs[0][1]


def test_parse_metrics_sums_label_sets():
    text = ('# HELP x\nvllm:num_preemptions_total{engine="0",model_name="m"} 2.0\n'
            'vllm:num_preemptions_total{engine="1",model_name="m"} 3.0\n'
            'vllm:kv_cache_usage_perc{engine="0"} 0.25\nnot a metric line\n')
    m = parse_metrics(text)
    assert m["vllm:num_preemptions_total"] == 5.0 and m["vllm:kv_cache_usage_perc"] == 0.25


def test_tokens_in_window():
    tl = [[(1.0, 10), (2.0, 20), (3.0, 30)], [(2.5, 5), (4.0, 15)]]
    assert tokens_in_window(tl, 1.5, 3.0) == 10 + 10 + 5


def _lv(c, tok, pre=0.0, err=0):
    return LevelResult("s5_draft", c, 10, err, 1.0, tok, 0, None, None, None, None, None, None, 0, 0, None,
                       pre, None, None, None, None, None, None)


def test_operating_point_rule():
    levels = [_lv(8, 300), _lv(16, 520), _lv(32, 900), _lv(64, 1000), _lv(128, 1020, pre=4)]
    op = choose_operating_point(levels, 0.88)
    # peak 1020 (preempting level still counts toward the peak); 0.88*1020 = 897.6 -> c=32
    assert op["concurrency"] == 32 and op["peak_tok_s"] == 1020
    assert choose_operating_point([_lv(8, 100, pre=1)], 0.88)["concurrency"] is None


def test_update_gen_concurrency(tmp_path):
    g = tmp_path / "gen.yaml"
    shutil.copy(REPO / "configs/gen.yaml", g)
    update_gen_concurrency(g, {"s5_draft": 48, "s8_filter": 128})
    text = g.read_text()
    assert "  per_stage: {s5_draft: 48, s8_filter: 128}" in text
    assert "# Throughput (§6.1.1)" in text  # comments preserved


def test_server_log_facts(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text(
        "INFO ... Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Qwen3.8-27B-FP8', "
        "revision=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a, quantization=fp8, kv_cache_dtype=auto, x\n"
        "INFO [cuda.py:538] Using FLASH_ATTN attention backend out of potential backends\n"
        "INFO [topk_topp_sampler.py:62] FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.\n"
        "INFO [kv_cache_utils.py:2395] GPU KV cache size: 706,332 tokens, Maximum concurrency for 16,384 "
        "tokens per request: 43.11x\n")
    f = server_log_facts(log)
    assert f["quantization"] == "fp8" and f["kv_cache_tokens"] == 706332 and f["vllm_version"] == "0.30.0"
    assert f["attention_backend"] == "FLASH_ATTN" and f["flashinfer_sampler_disabled"]


# ------------------------------------------------------------- fake streaming server
class FakeStream:
    def __init__(self):
        self.bodies: list[dict] = []
        self.seen_prefixes: set[str] = set()
        self.m = {"gen": 0.0, "q": 0.0, "h": 0.0}

    def handler(self, req: httpx.Request) -> httpx.Response:
        p = req.url.path
        if p == "/tokenize":
            body = json.loads(req.content)
            return httpx.Response(200, json={"count": sum(len(m["content"]) for m in body["messages"]) // 4})
        if p == "/metrics":
            return httpx.Response(200, text=(
                f'vllm:generation_tokens_total{{m="x"}} {self.m["gen"]}\n'
                f'vllm:prefix_cache_queries_total{{m="x"}} {self.m["q"]}\n'
                f'vllm:prefix_cache_hits_total{{m="x"}} {self.m["h"]}\n'
                'vllm:num_preemptions_total{m="x"} 0\nvllm:kv_cache_usage_perc{m="x"} 0.3\n'
                'vllm:cache_config_info{block_size="784",engine="0"} 1.0\n'))
        if p == "/v1/chat/completions":
            body = json.loads(req.content)
            self.bodies.append(body)
            content = body["messages"][-1]["content"]
            q = len(content) // 4
            self.m["q"] += q
            if content[:300] in self.seen_prefixes:
                self.m["h"] += min(q, 784)
            self.seen_prefixes.add(content[:300])
            n = body["max_tokens"] if body.get("ignore_eos") else 37
            self.m["gen"] += n
            lines, cum = [], 0
            while cum < n:
                step = min(10, n - cum)
                cum += step
                lines.append({"choices": [{"index": 0, "delta": {"content": "w " * step}, "finish_reason": None}],
                              "usage": {"prompt_tokens": q, "completion_tokens": cum}})
            lines.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "length" if body.get("ignore_eos") else "stop"}],
                          "usage": {"prompt_tokens": q, "completion_tokens": cum}})
            sse = "".join(f"data: {json.dumps(x)}\n\n" for x in lines) + "data: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(404)


@pytest.fixture
def bench(cfg):
    fs = FakeStream()
    b = Bench(cfg, "generator", http=httpx.AsyncClient(transport=httpx.MockTransport(fs.handler)))
    b.fake = fs
    return b


async def test_synthetic_pool_lengths_and_grouping(bench, cfg):
    pool = await build_synthetic_pool(bench.server, "s5_draft", 28, SPEC, cfg, random.Random(0))
    lo, hi = cfg.gen["bench"]["synthetic"]["s5_draft"]["prompt_tokens"]
    assert bench.server.tokenize_available
    assert all(lo * 0.85 <= bp.prompt_tokens <= hi * 1.15 for bp in pool)
    olo, ohi = cfg.gen["bench"]["synthetic"]["s5_draft"]["output_tokens"]
    assert all(olo <= bp.force_output_tokens <= ohi for bp in pool)
    secs = [bp.section for bp in pool]
    assert secs == sorted(secs, key=secs.index)  # contiguous blocks per section (prefix grouping)
    assert len(set(secs)) == 14 and "<spec_section>" in pool[0].prefix


async def test_run_level_forced_lengths_and_metrics(bench, cfg):
    pool = await build_synthetic_pool(bench.server, "s8_filter", 12, SPEC, cfg, random.Random(1))
    lv = await bench.run_level("s8_filter", pool, 4, 12, salt_level=1)
    assert lv.n_requests == 12 and lv.errors == 0 and lv.forced_length_ok_frac == 1.0
    b = bench.fake.bodies[0]
    assert b["ignore_eos"] is True and b["min_tokens"] == b["max_tokens"] and b["stream"] is True
    assert (b["temperature"], b["top_k"]) == (0.0, -1)                       # s8 is greedy
    assert b["stream_options"]["continuous_usage_stats"] is True
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert lv.preemptions == 0 and lv.kv_cache_usage_max == 0.3 and lv.prefix_cache_hit_rate is not None
    # every request has a unique salt where the variable part starts
    contents = [x["messages"][-1]["content"] for x in bench.fake.bodies]
    assert len(set(contents)) == 12 and all("[bench L001 R" in c for c in contents)
    assert await bench.server.cache_block_size() == 784
    await bench.aclose()


async def test_official_body_uses_natural_length_and_json(cfg):
    b = Bench(cfg, "generator", mode="official", http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))))
    bp = BenchPrompt(id="x", stage="s6_critique", prefix="P", suffix="S",
                     response_format={"type": "json_schema", "json_schema": {"name": "C", "schema": {}}})
    body = b.request_body(bp, "[salt]", 1, 0)
    assert "ignore_eos" not in body and "min_tokens" not in body
    assert body["max_tokens"] == cfg.gen["stages"]["s6_critique"]["max_tokens"]
    assert body["response_format"]["type"] == "json_schema" and body["messages"][-1]["content"] == "P[salt]S"
    await b.aclose()
