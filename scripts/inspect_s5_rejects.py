#!/usr/bin/env python
"""Read-only: look at the raw S5 responses that were rejected (missing_content_tags, truncated).

Reads the LLM cache (SQLite, opened read-only); sends no requests and writes nothing.
Only responses made with the current E.5 template are considered.

  python scripts/inspect_s5_rejects.py              # summary + 3 samples per category
  python scripts/inspect_s5_rejects.py --samples 6
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

MARK = "at most 5 short bullet points"          # only the v1.7.1 E.5 template
OPEN, CLOSE = re.compile(r"<content>", re.I), re.compile(r"</content>", re.I)


def shape(t: str) -> str:
    """Which tags the response has, in order (compact signature)."""
    tags = re.findall(r"</?(?:scratchpad|content|think|document|output)>|```\w*", t, re.I)
    return " ".join(tags[:8]) + (" …" if len(tags) > 8 else "")


def repetition(t: str) -> tuple[int, str]:
    """Most repeated non-empty line (count, line) — a sign of a generation loop."""
    lines = [ln.strip() for ln in t.splitlines() if len(ln.strip()) > 20]
    if not lines:
        return 0, ""
    ln, n = Counter(lines).most_common(1)[0]
    return n, ln[:100]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    cfg = Config.load()
    db = cfg.path("cache") / "llm_cache.sqlite"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT request_json, response_json FROM requests WHERE stage='s5_draft'").fetchall()
    cats: dict[str, list[dict]] = {"missing_content_tags": [], "truncated": []}
    n_new = 0
    shapes_ok = Counter()
    for req_s, resp_s in rows:
        req, resp = json.loads(req_s), json.loads(resp_s)
        prompt = req.get("prompt") or json.dumps(req)
        if MARK not in prompt:
            continue
        n_new += 1
        t, fr = resp["texts"][0], resp["finish_reasons"][0]
        m = re.search(r"Document type: (.*)", prompt)
        rec = {"doc_type": m.group(1).strip() if m else "?", "finish": fr, "chars": len(t),
               "tokens": (resp.get("usage") or {}).get("completion_tokens"), "shape": shape(t),
               "n_open": len(OPEN.findall(t)), "n_close": len(CLOSE.findall(t)), "text": t}
        has_block = re.search(r"<content>(.*?)</content>", t, re.S | re.I)
        if fr == "length" and not has_block:
            cats["truncated"].append(rec)
        elif not (has_block and has_block.group(1).strip()):
            cats["missing_content_tags"].append(rec)
        else:
            shapes_ok[rec["shape"]] += 1
    print(f"S5 responses with the v1.7.1 template in cache: {n_new}")
    print(f"Tag shapes of accepted-format responses: {dict(shapes_ok.most_common(5))}")
    for cat, recs in cats.items():
        print("\n" + "#" * 100 + f"\n{cat}: {len(recs)}")
        print("  tag shapes:", dict(Counter(r["shape"] for r in recs).most_common(8)))
        print("  <content> opened/closed counts:", dict(Counter((r["n_open"], r["n_close"]) for r in recs)))
        print("  doc types:", dict(Counter(r["doc_type"] for r in recs).most_common(8)))
        print("  finish:", dict(Counter(r["finish"] for r in recs)), " tokens:", sorted(r["tokens"] or 0 for r in recs))
        for r in recs[: a.samples]:
            n, ln = repetition(r["text"])
            print("\n" + "=" * 100)
            print(f"[{r['doc_type']}] finish={r['finish']} tokens={r['tokens']} chars={r['chars']} shape={r['shape']}")
            print(f"most repeated line x{n}: {ln!r}")
            print("--- first 500 chars:\n" + r["text"][:500])
            i = OPEN.search(r["text"])
            if i and i.start() > 500:
                print(f"--- around <content> (char {i.start()}):\n" + r["text"][max(0, i.start() - 200): i.start() + 300])
            print("--- last 400 chars:\n" + r["text"][-400:])


if __name__ == "__main__":
    main()
