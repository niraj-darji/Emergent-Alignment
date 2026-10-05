#!/usr/bin/env python
"""Read-only diagnostic: replay S5 requests that raised ThinkingLeakError and show the raw output.

Each replay uses the same deterministic seed as the pilot run, so it reproduces the exact response.
Nothing is written to the cache or to data/: the marker check is turned off only inside this script.

  python scripts/diag_thinking_leak.py --cand A                 # ids taken from drafts.errors.jsonl
  python scripts/diag_thinking_leak.py --cand A --ids idea_030_0 idea_017_5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awmsm import llm  # noqa: E402
from awmsm.config import Config  # noqa: E402
from awmsm.gen import midtrain_pipeline as mp  # noqa: E402
from awmsm.io import read_jsonl  # noqa: E402
from awmsm.pilot import Pilot  # noqa: E402

MARKERS = ("<think>", "</think>", "<|channel>", "<channel|>")


def leak_ids(err_path: Path) -> list[str]:
    ids = []
    for r in read_jsonl(err_path) if err_path.exists() else []:
        if "ThinkingLeak" in json.dumps(r):
            i = r.get("_src") or r.get("id")
            if i and i not in ids:
                ids.append(i)
    return ids


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="A")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max", type=int, default=4)
    a = ap.parse_args()

    pilot = Pilot(Config.load())
    cdir = pilot.cand_dir(a.cand)
    err = cdir / "drafts.errors.jsonl"
    ids = a.ids or leak_ids(err)
    print(f"Items with a thinking-leak error in {err.name}: {len(leak_ids(err))}")
    if not ids:
        print("No ids to replay (pass --ids)."); return
    ideas = {i["id"]: i for i in pilot.load_ideas()}
    ctx = pilot.ctx(bench_dir=None)               # do not touch bench_prompts
    tpl, req = ctx.template("E05_draft"), ctx.requirements_block

    llm.THINKING_MARKERS = ()                     # let the raw text through, in this process only
    async with await pilot.connect(a.cand) as client:
        client.cache.put = lambda **kw: None      # never cache a leaked response
        print("chat_template_kwargs sent:", client.model.chat_template_kwargs)
        for iid in ids[: a.max]:
            idea = ideas[iid]
            r = tpl.render(**ctx.identity, requirements_block=req, spec_excerpt=idea["spec_excerpt"],
                           doc_type=idea["doc_type"], assertions=mp.bullet(idea["assertions"]),
                           what=idea["what"], how=idea["how"], why=idea["why"])
            res = await client.complete_full(r.text, stage="s5_draft", use_cache=False)
            t, fr = res.texts[0], res.finish_reasons[0]
            hits = {m: t.count(m) for m in MARKERS if m in t}
            print("\n" + "=" * 100)
            print(f"{iid} | doc_type={idea['doc_type']} | finish={fr} | tokens={res.usage.get('completion_tokens')}"
                  f" | markers={hits} | seed={res.seed}")
            if not hits:
                print("(no marker this time)"); continue
            first = min(t.find(m) for m in hits)
            print(f"--- first marker at char {first} of {len(t)}")
            print("--- first 600 chars:\n" + t[:600])
            if first > 600:
                print(f"--- around the marker:\n...{t[max(0, first - 400): first + 400]}...")
            print("--- last 300 chars:\n" + t[-300:])


if __name__ == "__main__":
    asyncio.run(main())
