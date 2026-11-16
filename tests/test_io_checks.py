from __future__ import annotations

import asyncio
import json

import pytest

from awmsm.io import Rejected, append_jsonl, read_jsonl, run_stage, side_paths
from awmsm.textchecks import banned_keywords, first_failure, language_mixing, length, non_latin_share, truncated


# ----------------------------------------------------------------------------- io
def test_read_jsonl_tolerates_truncated_tail(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"id": 1}\n{"id": 2}\n{"id": 3, "te')
    assert [r["id"] for r in read_jsonl(p)] == [1, 2]
    append_jsonl(p, {"id": 4})  # partial line is cut before appending
    assert [r["id"] for r in read_jsonl(p)] == [1, 2, 4]


def test_read_jsonl_corrupt_middle_raises(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"id": 1}\nGARBAGE\n{"id": 3}\n')
    with pytest.raises(ValueError, match="line 2"):
        list(read_jsonl(p))


async def test_run_stage_resume_reject_error(tmp_path):
    out = tmp_path / "drafts.jsonl"
    items = [{"id": f"i{k}"} for k in range(10)]
    calls: list[str] = []
    fail_once = {"i3"}

    async def fn(item):
        calls.append(item["id"])
        await asyncio.sleep(0)
        if item["id"] in fail_once:
            fail_once.discard(item["id"])
            raise RuntimeError("server hiccup")
        if item["id"] == "i5":
            return Rejected("language_mixing", {"share": 0.2})
        if item["id"] == "i7":
            return [{"id": "i7a"}, {"id": "i7b"}]  # 1 -> many
        return {"id": f"doc_{item['id']}"}

    r1 = await run_stage("s5", items, fn, out, concurrency=3)
    assert r1.errors == 1 and r1.rejected["language_mixing"] == 1 and r1.kept_items == 8
    sp = side_paths(out)
    assert {r["_src"] for r in read_jsonl(sp["errors"])} == {"i3"}

    calls.clear()
    r2 = await run_stage("s5", items, fn, out, concurrency=3)  # resume: only the failed item reruns
    assert calls == ["i3"] and r2.skipped_done == 9 and r2.kept_items == 1

    stats = json.loads(sp["stats"].read_text())
    c = stats["cumulative"]
    assert c["items_done"] == 10 and c["items_kept"] == 9 and c["output_records"] == 10
    assert c["rejections_by_reason"]["language_mixing"] == {"count": 1, "rate": 0.1}


async def test_run_stage_sync_fn_and_duplicate_ids(tmp_path):
    out = tmp_path / "s.jsonl"
    r = await run_stage("sync", [{"id": 1}, {"id": 2}], lambda it: {"v": it["id"] * 2}, out)
    assert r.kept_items == 2
    with pytest.raises(ValueError, match="duplicate"):
        await run_stage("dup", [{"id": "a"}, {"id": "a"}], lambda it: {}, tmp_path / "d.jsonl")


async def test_run_stage_aborts_when_server_down(tmp_path):
    async def fn(item):
        raise ConnectionError("down")
    with pytest.raises(RuntimeError, match="aborting"):
        await run_stage("dead", [{"id": k} for k in range(100)], fn, tmp_path / "x.jsonl", concurrency=4)


# ------------------------------------------------------------------------- checks
def test_language_mixing():
    en = "The assistant considered the welfare of the salmon in the pen. " * 20
    assert language_mixing(en) is None
    mixed = en + "这是一个测试句子，包含很多中文字符。" * 3
    r = language_mixing(mixed)
    assert r is not None and r.reason == "language_mixing" and "CJK" in r.detail["scripts"]
    # accented Latin, micro sign and a single Greek letter are fine in long text
    assert language_mixing(en + " café naïve 5 µg/L, α-level") is None
    share, letters, _ = non_latin_share("abc")
    assert share == 0.0 and letters == 3


def test_keywords_prefix_and_whole_word():
    kws = ["suffer*", "kind", "ROI"]
    assert banned_keywords("Animals suffering is bad", kws).detail["matches"] == ["suffering"]
    assert banned_keywords("Read it on a Kindle", kws) is None  # 'kind' is whole-word only
    assert banned_keywords("what's the roi here", kws) is not None


def test_length_truncated_first_failure():
    assert length("a b c", min_words=5).reason == "length_out_of_range"
    assert truncated("length").reason == "truncated_max_tokens"
    assert truncated("stop") is None
    assert first_failure(None, truncated("length"), length("a", min_words=5)).reason == "truncated_max_tokens"
