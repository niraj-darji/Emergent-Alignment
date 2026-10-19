"""S10 corpus-level pattern scan (§6.3 S10: DeepMind's scan -> cluster -> autorate), plus a lexical scan.

(a) Scan: batches of ``scan_docs`` documents (opening + ending excerpts, so a batch fits the 16k context)
    -> the LLM lists recurring patterns (prompts/S10_scan.txt).
(b) Cluster: one call merges synonyms across batches (prompts/S10_cluster.txt); a cluster is kept if its
    members come from >= 2 scan batches. At most ``max_patterns`` clusters (most batches first) are rated.
(c) Autorate: each kept pattern becomes a binary classifier (prompts/S10_autorate.txt) run on a seeded
    sample of ``sample`` documents; prevalence = share rated present. Patterns above ``review_threshold``
    (20%) are flagged for review by the researcher (doc: "MUST be reviewed").

The lexical scan needs no LLM: recurring person-like names, opening phrases and long n-grams by document
frequency. Outputs: ``<corpus>/s10/`` (resumable via run_stage). Prompts are code-chat drafts, not in App. E.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..io import read_jsonl, run_stage, write_json
from ..llm import LLMClient
from . import midtrain_pipeline as mp


class ScanPattern(BaseModel):
    name: str
    description: str
    example: str


class ScanResult(BaseModel):
    patterns: list[ScanPattern] = Field(min_length=1)


class Cluster(BaseModel):
    name: str
    description: str
    members: list[int]


class ClusterResult(BaseModel):
    clusters: list[Cluster]


class Present(BaseModel):
    present: bool


DEFAULTS = {"scan_docs": 20, "scan_batches": 25, "head_words": 150, "tail_words": 80, "sample": 500,
            "max_patterns": 15, "min_batches": 2, "review_threshold": 0.20, "seed": 10}


def excerpt(text: str, head: int, tail: int) -> str:
    w = text.split()
    if len(w) <= head + tail:
        return text
    return " ".join(w[:head]) + " […] " + " ".join(w[-tail:])


async def run_s10(ctx: mp.GenContext, client: LLMClient, docs: list[dict[str, Any]], out: Path,
                  settings: dict[str, Any] | None = None, concurrency: int = 64) -> dict[str, Any]:
    s = {**DEFAULTS, **(settings or {})}
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(s["seed"])
    order = list(range(len(docs)))
    rng.shuffle(order)
    n_b = min(int(s["scan_batches"]), max(1, len(docs) // int(s["scan_docs"])))
    batches = [{"id": f"scan_{b:02d}", "doc_ids": [docs[i]["id"] for i in
                                                   order[b * s["scan_docs"]:(b + 1) * s["scan_docs"]]]}
               for b in range(n_b)]
    by_id = {d["id"]: d for d in docs}
    scan_t = ctx.template("S10_scan")

    async def scan(b: dict[str, Any]) -> dict[str, Any]:
        body = "\n\n".join(f"<document {k + 1}>\n{excerpt(by_id[i]['text'], s['head_words'], s['tail_words'])}\n"
                           f"</document {k + 1}>" for k, i in enumerate(b["doc_ids"]))
        r = scan_t.render(**ctx.identity, n_docs=len(b["doc_ids"]), documents=body)
        res = await client.complete_full(r.text, stage="s10_scan", json_schema=ScanResult)
        v, diag = mp.greedy_json(res, ScanResult)
        return {"id": b["id"], "patterns": [p.model_dump() for p in v.patterns], **diag}

    await run_stage("s10_scan", batches, scan, out / "scan.jsonl", concurrency=min(concurrency, 16))
    flat = [{"batch": r["id"], **p} for r in read_jsonl(out / "scan.jsonl") for p in r["patterns"]]

    cpath = out / "clusters.json"
    if not cpath.exists() and flat:
        listing = "\n".join(f"{k}. {p['name']}: {' '.join(p['description'].split()[:25])}"
                            for k, p in enumerate(flat))   # capped so ~250 patterns fit the 16k context
        r = ctx.template("S10_cluster").render(**ctx.identity, patterns=listing)
        res = await client.complete_full(r.text, stage="s10_scan", json_schema=ClusterResult, max_tokens=4000)
        v, diag = mp.greedy_json(res, ClusterResult)
        clusters = []
        for c in v.clusters:
            mem = sorted({m for m in c.members if 0 <= m < len(flat)})
            bs = sorted({flat[m]["batch"] for m in mem})
            clusters.append({"name": c.name, "description": c.description, "members": mem,
                             "n_batches": len(bs), "examples": [flat[m]["example"] for m in mem[:3]]})
        clusters.sort(key=lambda c: -c["n_batches"])
        write_json(cpath, {"clusters": clusters, "n_scan_patterns": len(flat), **diag})
    clusters = (read_json(cpath)["clusters"] if cpath.exists() else [])
    kept = [c for c in clusters if c["n_batches"] >= s["min_batches"]][: int(s["max_patterns"])]

    sample = [docs[i] for i in order[: int(s["sample"])]]
    rate_t = ctx.template("S10_autorate")
    items = [{"id": f"p{k:02d}__{d['id']}", "k": k, "doc_id": d["id"]} for k in range(len(kept)) for d in sample]
    # sort by pattern so each pattern's shared prefix stays warm in the prefix cache (§6.1.1 A.2)

    async def rate(it: dict[str, Any]) -> dict[str, Any]:
        c = kept[it["k"]]
        r = rate_t.render(pattern_name=c["name"], pattern_description=c["description"], text=by_id[it["doc_id"]]["text"])
        res = await client.complete_full(r.text, stage="s10_autorate", json_schema=Present)
        v, diag = mp.greedy_json(res, Present)
        return {"id": it["id"], "k": it["k"], "doc_id": it["doc_id"], "present": v.present, **diag}

    await run_stage("s10_autorate", items, rate, out / "autorate.jsonl", concurrency=concurrency)
    present: Counter = Counter()
    total: Counter = Counter()
    for r in read_jsonl(out / "autorate.jsonl"):
        total[r["k"]] += 1
        present[r["k"]] += bool(r["present"])
    patterns = []
    for k, c in enumerate(kept):
        prev = present[k] / total[k] if total[k] else None
        patterns.append({**c, "rated": total[k], "prevalence": prev,
                         "review": prev is not None and prev > s["review_threshold"]})
    patterns.sort(key=lambda p: -(p["prevalence"] or 0))
    res = {"settings": s, "documents": len(docs), "scan_batches": len(batches), "scan_patterns": len(flat),
           "clusters": len(clusters), "rated_patterns": patterns,
           "flagged_for_review": [p["name"] for p in patterns if p["review"]], "lexical": lexical_scan(docs)}
    write_json(out / "s10_summary.json", res)
    return res


def read_json(p: Path) -> Any:
    import json
    return json.loads(p.read_text())


NAME_RE = re.compile(r"\b(?:(?:Dr|Prof|Mr|Mrs|Ms)\.\s)?[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")
SKIP_NAMES = {"Your Lab"}
STARTERS = set("""a an and as at but by for from he her his how i if in it its my no not of on or our she so
that the their then there these they this those to we what when where which while who why with you your
after before during however although because since until unlike yes dear hi hello thanks""".split())


def lexical_scan(docs: list[dict[str, Any]], top: int = 15, ngram: int = 8, names_to_skip: set[str] | None = None
                 ) -> dict[str, Any]:
    """Document frequency of recurring names, openings and long n-grams (no LLM)."""
    n = len(docs) or 1
    skip = SKIP_NAMES | (names_to_skip or set())
    names, opens, grams = Counter(), Counter(), Counter()
    for d in docs:
        t = d["text"]
        names.update({m for m in NAME_RE.findall(t)
                      if m not in skip and m.split()[0].lower().rstrip(".") not in STARTERS
                      and not any(x in m for x in skip)})
        first = re.sub(r"[#*_>`]", "", t).strip().split()
        opens[" ".join(first[:6]).lower()] += 1
        w = re.findall(r"[a-z']+", t.lower())
        grams.update({" ".join(w[i:i + ngram]) for i in range(len(w) - ngram + 1)})
    share = lambda c: [{"text": k, "docs": v, "share": round(v / n, 3)} for k, v in c.most_common(top) if v > 1]  # noqa: E731
    return {"names": share(names), "openings": share(opens), f"{ngram}grams": share(grams)}
