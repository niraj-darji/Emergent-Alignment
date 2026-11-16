from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import BaseModel

from awmsm.config import Config, ConfigError
from awmsm.llm import JSONValidationError, LLMClient, LLMError, ThinkingLeakError

from .conftest import FakeVLLM, chat_response


def _words(n: int) -> str:
    """n distinct words (repeated "word" would now trip the repetition check)."""
    return " ".join(f"w{i}" for i in range(n))



class Answer(BaseModel):
    answer: str



def make(cfg: Config, fake: FakeVLLM, role: str = "generator") -> LLMClient:
    return LLMClient(cfg.role(role), cfg, http_client=fake.client())


async def test_sampling_sent_explicitly(cfg, fake):
    async with make(cfg, fake) as c:
        out = await c.complete("hello", stage="s5_draft")
    assert out == ["reply 0"]
    b = fake.chat_bodies[0]
    assert b["model"] == "Qwen/Qwen3.8-27B-FP8"
    # model's own truncation (D17) + stage temperature (§6.1)
    assert (b["temperature"], b["top_p"], b["top_k"], b["min_p"]) == (0.7, 0.8, 20, 0.0)
    # penalties explicitly off (D17)
    assert (b["presence_penalty"], b["frequency_penalty"], b["repetition_penalty"]) == (0.0, 0.0, 1.0)
    assert b["max_tokens"] == 3200 and isinstance(b["seed"], int)
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert b["messages"] == [{"role": "user", "content": "hello"}]


async def test_system_prompt_and_n(cfg, fake):
    async with make(cfg, fake) as c:
        out = await c.complete("p", stage="s4_ideas", system="sys", n=3)
    assert len(out) == 3
    assert fake.chat_bodies[0]["messages"][0] == {"role": "system", "content": "sys"}
    assert fake.chat_bodies[0]["n"] == 3


async def test_checking_stage_is_greedy_and_neutral(cfg, fake):
    async with make(cfg, fake) as c:
        await c.complete("judge this", stage="s8_filter")
        with pytest.raises(LLMError):
            await c.complete("judge this", stage="s8_filter", temperature=0.5)
    b = fake.chat_bodies[0]
    assert (b["temperature"], b["top_p"], b["top_k"]) == (0.0, 1.0, -1)


async def test_cache_hit_avoids_http_and_is_deterministic(cfg, fake):
    async with make(cfg, fake) as c:
        r1 = await c.complete_full("same", stage="s5_draft")
        r2 = await c.complete_full("same", stage="s5_draft")
        assert not r1.cached and r2.cached and r1.texts == r2.texts and r1.seed == r2.seed
        assert len(fake.chat_bodies) == 1
        # a different explicit seed is a different request
        await c.complete("same", stage="s5_draft", seed=123)
        assert len(fake.chat_bodies) == 2 and fake.chat_bodies[1]["seed"] == 123
        assert c.stats["s5_draft"].cache_hits == 1
    # cache persists across client instances (disk)
    async with make(cfg, fake) as c2:
        r3 = await c2.complete_full("same", stage="s5_draft")
    assert r3.cached and len(fake.chat_bodies) == 2


async def test_retry_on_503_then_success(cfg, fake):
    fake.responder = lambda body, i: (503, {"error": "busy"}) if i == 0 else (200, chat_response(["ok"]))
    async with make(cfg, fake) as c:
        assert await c.complete("x", stage="s5_draft") == ["ok"]
        assert c.stats["s5_draft"].http_retries == 1


async def test_retries_exhausted(cfg, fake):
    fake.responder = lambda body, i: (500, {"error": "boom"})
    async with make(cfg, fake) as c:
        with pytest.raises(LLMError, match="giving up"):
            await c.complete("x", stage="s5_draft")
    assert len(fake.chat_bodies) == 4  # 1 + max_retries(3)


async def test_400_not_retried(cfg, fake):
    fake.responder = lambda body, i: (400, {"error": {"message": "context too long"}})
    async with make(cfg, fake) as c:
        with pytest.raises(LLMError, match="HTTP 400"):
            await c.complete("x", stage="s5_draft")
    assert len(fake.chat_bodies) == 1


async def test_greedy_invalid_json_falls_back_once_at_0_3(cfg, fake):
    fake.responder = lambda body, i: (200, chat_response(['{"answer": 5' if i == 0 else '{"answer": "salmon"}']))
    async with make(cfg, fake) as c:
        objs = await c.complete_json("fish?", stage="s8_filter", schema=Answer)
        assert objs[0].answer == "salmon"
        rep = c.fallback_report()["s8_filter"]
    assert rep == {"calls": 1, "fallbacks": 1, "rate": 1.0, "reasons": {"invalid_json": 1},
                   "unresolved": 0, "above_threshold": True}
    b0, b1 = fake.chat_bodies
    assert b0["response_format"]["json_schema"]["schema"]["required"] == ["answer"]
    assert (b0["temperature"], b0["top_k"]) == (0.0, -1)
    # fallback: T=0.3 with the model's own truncation, same max_tokens (§6.1)
    assert (b1["temperature"], b1["top_p"], b1["top_k"], b1["max_tokens"]) == (0.3, 0.8, 20, b0["max_tokens"])


async def test_greedy_fallback_only_once_then_raises_for_json(cfg, fake):
    fake.responder = lambda body, i: (200, chat_response(["not json"]))
    async with make(cfg, fake) as c:
        with pytest.raises(JSONValidationError, match="fallback"):
            await c.complete("x", stage="s8_filter", json_schema=Answer)
        assert c.stats["s8_filter"].t0_fallback_unresolved == 1
    assert len(fake.chat_bodies) == 2  # original + exactly one fallback


async def test_greedy_truncated_text_fallback(cfg, fake):
    def responder(body, i):
        return 200, chat_response(["long critique ..."], finish="length" if i == 0 else "stop")
    fake.responder = responder
    async with make(cfg, fake) as c:
        res = await c.complete_full("critique this", stage="s6_critique")
        assert res.finish_reasons == ["stop"] and res.params["t0_fallback"]["reason"] == "truncated"
        # still truncated after fallback: returned as-is (pre-check rejects it), counted unresolved
        fake.responder = lambda body, i: (200, chat_response(["x"], finish="length"))
        res2 = await c.complete_full("another", stage="s6_critique")
        assert res2.finish_reasons == ["length"] and c.stats["s6_critique"].t0_fallback_unresolved == 1
        # replay from cache reproduces the same fallback decision without HTTP
        n_http = len(fake.chat_bodies)
        await c.complete_full("critique this", stage="s6_critique")
        assert len(fake.chat_bodies) == n_http
        assert c.fallback_report()["s6_critique"]["fallbacks"] == 3


async def test_greedy_ok_no_fallback(cfg, fake):
    async with make(cfg, fake) as c:
        for k in range(50):
            await c.complete(f"judge {k}", stage="s8_filter")
        rep = c.fallback_report()["s8_filter"]
    assert rep["fallbacks"] == 0 and rep["rate"] == 0.0 and not rep["above_threshold"]


async def test_sampled_json_retries_then_gives_up(cfg, fake):
    fake.responder = lambda body, i: (200, chat_response(["not json"]))
    async with make(cfg, fake) as c:
        with pytest.raises(JSONValidationError):
            await c.complete("x", stage="s4_ideas", json_schema=Answer)
        assert c.stats["s4_ideas"].json_retries == 3
    assert len(fake.chat_bodies) == 3
    assert len({b["seed"] for b in fake.chat_bodies}) == 3  # new seed per attempt


async def test_guided_json_partial_n(cfg, fake):
    def responder(body, i):
        if i == 0:
            return 200, chat_response(['{"answer": "a"}', "bad", '{"answer": "c"}'])
        assert body["n"] == 1  # only the invalid one is re-requested
        return 200, chat_response(['{"answer": "b"}'])
    fake.responder = responder
    async with make(cfg, fake) as c:
        out = await c.complete("x", stage="s4_ideas", json_schema=Answer, n=3)
    assert len(out) == 3


async def test_thinking_leak_detected(cfg, fake):
    fake.responder = lambda body, i: (200, chat_response(["<think>hmm</think>OK"]))
    async with make(cfg, fake) as c:
        with pytest.raises(ThinkingLeakError):
            await c.complete("x", stage="s5_draft")
    fake.responder = lambda body, i: (200, chat_response(["OK"], extra_msg={"reasoning": "secret"}))
    async with make(cfg, fake) as c:
        with pytest.raises(ThinkingLeakError):
            await c.complete("y", stage="s5_draft")


async def test_unspecified_stage_refuses(cfg, fake):
    cfg.gen["stages"]["new_stage"] = {"temperature": None, "max_tokens": 100}
    async with make(cfg, fake) as c:
        with pytest.raises(ConfigError, match="unset"):
            await c.complete("x", stage="new_stage")
        with pytest.raises(ConfigError, match="MUST be added"):
            await c.complete("x", stage="not_in_table")
    assert fake.chat_bodies == []


def test_stage_table_matches_doc_v15(cfg):
    want = {"s1_decompose": 0.0, "s2_subdomains": 0.7, "s3_doctypes": 0.7, "s4_ideas": 0.9,
            "s5_draft": 0.7, "s6_critique": 0.0, "s7_rewrite": 0.7, "s8_filter": 0.0, "s10_scan": 0.0,
            "s10_autorate": 0.0, "aft_prompts": 0.9, "aft_response": 0.7, "aft_critique_rewrite": 0.7,
            "aft_filter": 0.0, "opaque_generate": 0.9, "leak_judge": 0.0, "mde_candidates": 0.7,
            "manta_user_sim": 0.0, "persona_rollouts": 1.0, "eval_sampling": 0.7, "eval_judge": 0.0}
    assert {k: cfg.stage_temperature(k) for k in want} == want


async def test_vendor_range_flag(cfg, fake):
    async with make(cfg, fake) as c:
        await c.complete("x", stage="s4_ideas")  # 0.9 vs Qwen non-thinking 0.7 -> flagged
        await c.complete("x", stage="s5_draft")  # 0.7 -> fine
        m = c.manifest()
    assert m["flags"]["stages_outside_vendor_temperature_range"] == ["s4_ideas"]
    assert m["model"]["revision"] == "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"


async def test_gemma_profile(cfg):
    fake = FakeVLLM(served="google/gemma-4-31B-it")
    async with make(cfg, fake, role="eval_judge") as c:
        await c.complete("x", stage="s5_draft")
        assert c.manifest()["flags"]["stages_outside_vendor_temperature_range"] == ["s5_draft"]
    b = fake.chat_bodies[0]
    assert (b["top_p"], b["top_k"]) == (0.95, 64) and "chat_template_kwargs" not in b


async def test_verify_server(cfg, fake):
    async with make(cfg, fake) as c:
        info = await c.verify_server()
    assert info["served_model"] == "Qwen/Qwen3.8-27B-FP8" and info["vllm_version"] == "0.30.0"
    wrong = FakeVLLM(served="google/gemma-4-31B-it")
    async with make(cfg, wrong) as c:
        with pytest.raises(LLMError, match="config expects"):
            await c.verify_server()
    badrev = FakeVLLM(root="/hf/hub/models--Qwen--x/snapshots/deadbeef")
    async with make(cfg, badrev) as c:
        with pytest.raises(LLMError, match="revision"):
            await c.verify_server()


def test_d13_family_check(cfg):
    base = copy.deepcopy(cfg.base)
    base["servers"]["eval_judge"]["model"] = "qwen3.8-27b-fp8"
    with pytest.raises(ConfigError, match="D13"):
        Config(base, cfg.gen, root=cfg.root)


def test_real_product_name_rejected(cfg):
    base = copy.deepcopy(cfg.base)
    base["identity"]["assistant_name"] = "Claude"
    with pytest.raises(ConfigError, match="real product"):
        Config(base, cfg.gen, root=cfg.root)


async def test_request_log_contents(cfg, fake):
    import json
    import sqlite3
    async with make(cfg, fake) as c:
        await c.complete("log me", stage="s5_draft", system="sys")
        path = c.cache.path
    row = sqlite3.connect(path).execute(
        "SELECT model, revision, stage, prompt_sha, seed, request_json FROM requests").fetchone()
    assert row[0] == "Qwen/Qwen3.8-27B-FP8" and row[2] == "s5_draft" and len(row[3]) == 64
    req = json.loads(row[5])
    assert req["prompt"] == "log me" and req["system"] == "sys" and req["top_p"] == 0.8
    assert req["extra_body"]["top_k"] == 20 and req["seed"] == row[4]



def test_thinking_leak_tolerates_scratchpad_closed_with_think():
    from awmsm.llm import scratchpad_closed_with_think, thinking_leak
    slip = "<scratchpad>\nplan...\nLet's write.\n</think>\n\n<content>\nDoc.\n</content>"
    assert not thinking_leak(slip) and scratchpad_closed_with_think(slip)
    ok = "<scratchpad>plan</scratchpad><content>Doc.</content>"
    assert not thinking_leak(ok) and not scratchpad_closed_with_think(ok)
    assert not thinking_leak("<content>Doc.</content>")
    # markers inside <content> are left to DocChecks (rejection, not an error)
    assert not thinking_leak("<scratchpad>plan</think><content>Doc.</think></content>")
    # still leaks
    assert thinking_leak("<think>reasoning</think><content>Doc.</content>")
    assert thinking_leak("<scratchpad><think>plan</think><content>Doc.</content>")      # marker inside the plan
    assert thinking_leak("<scratchpad>plan</think> more </think><content>Doc.</content>")  # a second </think>
    assert thinking_leak("Some text </think> <content>Doc.</content>")                 # no leading scratchpad
    assert thinking_leak("<scratchpad>plan</scratchpad><content>Doc.</content></think>")


def test_doc_checks_reject_tags_inside_content(cfg):
    from awmsm.config import Config
    from awmsm.gen.midtrain_pipeline import DocChecks
    checks = DocChecks(Config(cfg.base, cfg.gen, root=Path(__file__).resolve().parents[1]))
    body = _words(400)
    doc, bad = checks(f"<scratchpad>p</think><content>{body}</content>", "stop")
    assert bad is None and doc == body
    for inner in ("</think>", "<think>", "<scratchpad>", "my Scratchpad notes", "<|channel>"):
        doc, bad = checks(f"<scratchpad>p</scratchpad><content>{body} {inner} {body}</content>", "stop")
        assert doc is None and bad.reason == "tag_in_content", inner
    doc, bad = checks(f"<scratchpad>p</think><content>{body}", "stop")      # unclosed, no sentence end
    assert doc is None and bad.reason == "unclosed_content_unterminated"


def test_v172_unclosed_content_rule(cfg):
    from awmsm.config import Config
    from awmsm.gen.midtrain_pipeline import DocChecks, parse_content
    checks = DocChecks(Config(cfg.base, cfg.gen, root=Path(__file__).resolve().parents[1]))
    body = _words(400)
    # accepted unclosed: stopped on its own, one <content>, sentence-final punctuation (+ closing quote/bracket)
    for end in (".", "!", "?", '."', ".”", ".)", "?')", "…"):
        r = checks.check(f"<scratchpad>p</scratchpad>\n<content>\n{body}{end}\n\n", "stop")
        assert r.bad is None and r.tolerances == ["unclosed_content"] and r.doc.endswith(end), end
    # rejected
    assert checks.check(f"<content>{body}.", "length").bad.reason == "truncated_max_tokens"
    assert checks.check(f"<content>{body}", "stop").bad.reason == "unclosed_content_unterminated"
    assert checks.check(f"<content>{body}:", "stop").bad.reason == "unclosed_content_unterminated"
    assert checks.check(f"<content>a.<content>{body}.", "stop").bad.reason == "multiple_content_tags"
    assert checks.check(f"<content>a.</content><content>{body}.</content>", "stop").bad.reason == "multiple_content_tags"
    assert checks.check(f"<content>{body}. </scratchpad>", "stop").bad.reason == "unclosed_content_unterminated"
    assert checks.check(f"<content>{body} the scratchpad says.", "stop").bad.reason == "tag_in_content"
    assert checks.check(f"{body}.", "stop").bad.reason == "missing_content_tags"
    # closed blocks are unchanged and carry no tolerance
    r = checks.check(f"<content>{body}</content>", "stop")
    assert r.bad is None and r.tolerances == []
    assert parse_content("<content> a </content>", "stop").doc == "a"


def test_degenerate_repetition():
    from awmsm.textchecks import degenerate_repetition
    body = _words(300)
    assert degenerate_repetition(body) is None
    reply = "**Reply by: Aster (Your Lab Assistant)**"
    turns = lambda k: "".join(f"\n{reply}\nanswer {i} " + " ".join(f"t{i}_{j}" for j in range(8)) for i in range(k))  # noqa: E731
    assert degenerate_repetition(body + turns(5)) is None                              # header x5: allowed
    r = degenerate_repetition(body + turns(6))                                         # header x6: loop
    assert r.reason == "degenerate_repetition" and r.detail["kind"] == "line" and r.detail["count"] == 6
    loop = " Let us make it kind, and let us make it specific for everyone here today."  # 15 words
    assert degenerate_repetition(body + loop * 3) is None                              # each 10-gram x3: allowed
    r = degenerate_repetition(body + loop * 4)
    assert r.detail["kind"] == "10-gram" and r.detail["count"] == 4
    # case and edge punctuation are ignored for word sequences
    assert degenerate_repetition(body + " " + " ".join(["Alpha beta, gamma delta epsilon zeta eta theta iota kappa."] * 4)) is not None


def test_repetition_exemptions_short_lines_and_code_fences():
    from awmsm.textchecks import degenerate_repetition
    body = _words(300)
    pretty_json = "\n".join(x for i in range(8) for x in ("{", f'  "k{i}": "value {i} {_words(3)}"', "}", "---"))  # { } --- x8: ignored
    assert degenerate_repetition(body + "\n" + pretty_json) is None
    code = "\n```python\n" + "\n".join(["print('the same line of code here')"] * 9) + "\n```\n"
    assert degenerate_repetition(body + code + "end") is None                  # fenced block skipped
    assert degenerate_repetition(body + code.replace("```python", "").replace("```", "")) is not None
