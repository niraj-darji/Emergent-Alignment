#!/usr/bin/env python
"""Read-only: why do responses of one stage hit max_tokens? Reads the LLM cache (SQLite, read-only);
sends no requests and writes nothing.

  python scripts/inspect_truncations.py --stage s6_critique --samples 4
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awmsm.config import Config  # noqa: E402
from awmsm.textchecks import degenerate_repetition  # noqa: E402


def q(xs: list[int]) -> dict[str, int] | None:
    if not xs:
        return None
    xs = sorted(xs)
    at = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]   # noqa: E731
    return {"n": len(xs), "min": xs[0], "p50": at(.5), "p90": at(.9), "p99": at(.99), "max": xs[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="s6_critique")
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args()
    cfg = Config.load()
    con = sqlite3.connect(f"file:{cfg.path('cache') / 'llm_cache.sqlite'}?mode=ro", uri=True)
    rows = con.execute("SELECT request_json, response_json, completion_tokens FROM requests WHERE stage=?",
                       (a.stage,)).fetchall()
    by_temp: dict[float, list[dict]] = {}
    for req_s, resp_s, ct in rows:
        req, resp = json.loads(req_s), json.loads(resp_s)
        t = float(req.get("temperature", -1))
        for text, fr in zip(resp["texts"], resp["finish_reasons"]):
            by_temp.setdefault(t, []).append({"text": text, "finish": fr, "tokens": ct, "max_tokens": req.get("max_tokens"),
                                              "prompt": req.get("prompt", "")})
    print(f"Stage {a.stage}: {len(rows)} cached requests")
    for t, recs in sorted(by_temp.items()):
        tr = [r for r in recs if r["finish"] == "length"]
        print(f"\n## temperature {t}: {len(recs)} responses, {len(tr)} hit max_tokens ({recs[0]['max_tokens']})")
        print("  output tokens, finished normally:", q([r["tokens"] for r in recs if r["finish"] != "length" and r["tokens"]]))
        kinds = Counter()
        for r in tr:
            rep = degenerate_repetition(r["text"])
            r["rep"] = rep.detail if rep else None
            kinds["repetition loop" if rep else "no loop detected"] += 1
            r["n_issues"] = len(re.findall(r'"criterion"\s*:', r["text"]))
        print("  truncated responses:", dict(kinds), " issues listed per truncated response:",
              sorted(r["n_issues"] for r in tr))
        for r in tr[: a.samples]:
            m = re.search(r"<document>(.*?)</document>", r["prompt"], re.S)
            print("\n" + "=" * 100)
            print(f"T={t} tokens={r['tokens']} issues={r['n_issues']} draft_words={len(m.group(1).split()) if m else '?'} "
                  f"repetition={r['rep']}")
            print("--- first 600 chars:\n" + r["text"][:600])
            print("--- last 500 chars:\n" + r["text"][-500:])


if __name__ == "__main__":
    main()
