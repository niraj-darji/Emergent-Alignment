"""Corpus follow-up tools: spec pre-check, S10 run, the morning report and the blinded H1.8 audit packet.

    python scripts/corpus_tools.py check-spec --spec-file specs/aw_main.md
    python scripts/corpus_tools.py s10 --dir data/corpora/aw_main_v1 --spec-id aw_main
    python scripts/corpus_tools.py report --dir data/corpora/aw_main_v1
    python scripts/corpus_tools.py audit-export --dir data/corpora/aw_main_v1 --n-included 30 --n-excluded 0
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import Config
from .gen import midtrain_pipeline as mp
from .gen.quotas import check_quotas
from .gen.schemas import S8_CRITERIA
from .io import read_jsonl, write_json
from .pilot import PilotError, _strip


# ----------------------------------------------------------------------------- spec pre-check
def check_spec(cfg: Config, spec_id: str, spec_file: Path) -> dict[str, Any]:
    from .tokens import TokenCounter
    ctx = mp.GenContext.create(cfg, spec_id, spec_path=spec_file)   # raises on unknown placeholders
    tc = TokenCounter(cfg.base.get("train_base_model") or "meta-llama/Llama-3.1-8B")
    raw = spec_file.read_text(encoding="utf-8")
    text = ctx.spec_text
    paras = [p for p in text.split("\n\n") if p.strip() and not p.lstrip().startswith("#")]
    no_because = [p.strip()[:120] for p in paras
                  if not any(w in p.lower() for w in ("because", "since", "so that", "in order to", "why"))]
    return {"spec_file": str(spec_file), "words": len(text.split()),
            "placeholders_in_file": sorted(set(mp.SPEC_PLACEHOLDER_RE.findall(raw))),
            "assistant_name_mentions": text.count(ctx.identity["assistant_name"]),
            "headings": [line.strip() for line in text.splitlines() if line.lstrip().startswith("#")],
            "paragraphs": len(paras), "paragraphs_without_a_reason_word": no_because,
            "reason_type_config": cfg.gen["documents"]["reason_type"].get(spec_id),
            "s8_template": (cfg.gen.get("s8") or {}).get("template", "E08_filter"),
            "s4_scripted_dialogue_min_words": cfg.gen["s4"].get("scripted_dialogue_min_words"),
            "token_counter": tc.method, "token_counter_exact": tc.exact}


# ----------------------------------------------------------------------------- loading
def batch_dirs(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("batch_*") if p.is_dir())


def all_ideas(d: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for b in batch_dirs(d):
        for r in read_jsonl(b / "ideas.jsonl"):
            r = _strip(r)
            out[r["id"]] = {**r, "batch": b.name}
    return out


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def length_stats(words: list[int]) -> dict[str, Any]:
    if not words:
        return {"n": 0}
    return {"n": len(words), "median": statistics.median(words), "p10": _pct(words, 0.1), "p90": _pct(words, 0.9),
            "min": min(words), "max": max(words), "under_400": sum(w < 400 for w in words),
            "over_1500": sum(w > 1500 for w in words)}


# ----------------------------------------------------------------------------- report
STAGES = [("s4_ideas", "ideas"), ("s5_draft", "drafts"), ("s6_critique", "critiques"),
          ("s7_rewrite", "rewrites"), ("s8_filter", "judged")]


def report(cfg: Config, d: Path) -> dict[str, Any]:
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    stage: dict[str, Any] = {}
    for st, stem in STAGES:
        done = kept = 0
        reasons: Counter = Counter()
        secs = 0.0
        for b in batch_dirs(d):
            sp = b / f"{stem}.stats.json"
            if not sp.exists():
                continue
            c = json.loads(sp.read_text())
            done += c["cumulative"]["items_done"]
            kept += c["cumulative"]["items_kept"]
            for k, v in c["cumulative"]["rejections_by_reason"].items():
                reasons[k] += v["count"]
            secs += c["last_run"]["seconds"]
        stage[st] = {"inputs_done": done, "kept": kept, "rejected": dict(reasons.most_common()),
                     "hours_last_runs": round(secs / 3600, 2)}
    n_ideas = sum(1 for b in batch_dirs(d) for _ in read_jsonl(b / "ideas.jsonl"))
    stage["s4_ideas"]["ideas_out"] = n_ideas
    judged = [j for b in batch_dirs(d) for j in read_jsonl(b / "judged.jsonl")]
    s8 = {"judged": len(judged), "include": sum(j["verdict"] == "INCLUDE" for j in judged)}
    s8["pass_rate"] = s8["include"] / s8["judged"] if judged else None
    s8["fail_by_criterion"] = {c: sum(not j["criteria"][c] for j in judged if not j.get("unresolved"))
                               for c in S8_CRITERIA}
    s8["t0_fallback_rate"] = (sum(bool(j.get("t0_fallback")) for j in judged) / len(judged)) if judged else None
    ideas = all_ideas(d)
    doc_idea = {r["id"]: r["idea_id"] for b in batch_dirs(d) for r in read_jsonl(b / "rewrites.jsonl")}
    s8["unresolved_excluded"] = sum(bool(j.get("unresolved")) for j in judged)
    by_story: dict[bool, list[bool]] = defaultdict(list)
    for j in judged:
        iid = j.get("idea_id") or doc_idea.get(j["doc_id"])   # unresolved-EXCLUDE records carry only doc_id
        by_story[bool(ideas.get(iid, {}).get("is_story"))].append(j["verdict"] == "INCLUDE")
    s8["pass_rate_stories"] = (sum(by_story[True]) / len(by_story[True])) if by_story[True] else None
    s8["pass_rate_non_stories"] = (sum(by_story[False]) / len(by_story[False])) if by_story[False] else None
    corpus = list(read_jsonl(d / "corpus.jsonl")) if (d / "corpus.jsonl").exists() else []
    s9 = summary.get("s9", {})
    before = summary.get("included_before_s9")
    out = {
        "status": summary.get("status"), "batches": len(batch_dirs(d)),
        "wall_clock_hours": summary.get("hours_all_runs", summary.get("hours")),
        "runs": json.loads((d / "runs.json").read_text()) if (d / "runs.json").exists() else None,
        "stages": stage, "s8": s8,
        "s9": {**s9, "included_before_s9": before,
               "near_duplicate_rate": (s9.get("removed", 0) / before) if before else None},
        "corpus": {"documents": len(corpus), "tokens": summary.get("tokens"), "target": summary.get("target_tokens"),
                   "token_method": summary.get("token_method"),
                   "words": length_stats([c["words"] for c in corpus]),
                   "tokens_per_doc": length_stats([c["tokens"] for c in corpus]),
                   "story_share": summary.get("story_share"),
                   "quota_check": summary.get("quota_check_final"),
                   "doc_types_top": summary.get("doc_types")},
        "s7_all_parsed_words": length_stats([r["words"] for b in batch_dirs(d) for r in read_jsonl(b / "rewrites.jsonl")]),
    }
    s10 = d / "s10" / "s10_summary.json"
    if s10.exists():
        sm = json.loads(s10.read_text())
        out["s10"] = {"rated": [{k: p[k] for k in ("name", "description", "prevalence", "n_batches", "review")}
                                for p in sm["rated_patterns"]],
                      "flagged_for_review": sm["flagged_for_review"], "lexical": sm["lexical"]}
    write_json(d / "report.json", out)
    (d / "report.md").write_text(render_md(out), encoding="utf-8")
    return out


def _f(x: Any, pct: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x:.1%}" if pct else (f"{x:,}" if isinstance(x, int) else f"{x}")


def render_md(r: dict[str, Any]) -> str:
    L = [f"# Corpus report\n\nStatus: **{r['status']}** · batches {r['batches']} · wall clock "
         f"**{r['wall_clock_hours']} h**\n", "## Stage counts\n", "| Stage | Done | Kept | Rejected (top) | Hours |",
         "|---|---|---|---|---|"]
    for st, v in r["stages"].items():
        rej = ", ".join(f"{k} {n}" for k, n in list(v["rejected"].items())[:4]) or "–"
        L.append(f"| {st} | {_f(v['inputs_done'])} | {_f(v['kept'])} | {rej} | {v['hours_last_runs']} |")
    s8 = r["s8"]
    L += ["", f"## S8 filter\n\nPass rate **{_f(s8['pass_rate'], True)}** ({s8['include']}/{s8['judged']}); stories "
          f"{_f(s8['pass_rate_stories'], True)}, non-stories {_f(s8['pass_rate_non_stories'], True)}; "
          f"T=0 fallback {_f(s8['t0_fallback_rate'], True)}; unresolved, recorded as EXCLUDE: {s8['unresolved_excluded']}.\n",
          "Fails by criterion: " + ", ".join(f"{k} {v}" for k, v in s8["fail_by_criterion"].items())]
    s9 = r["s9"]
    L += ["", f"## S9 near-duplicates\n\nRemoved {s9.get('removed')} of {s9.get('included_before_s9')} "
          f"({_f(s9.get('near_duplicate_rate'), True)}); method {s9.get('method')}."]
    c = r["corpus"]
    w = c["words"]
    L += ["", f"## Corpus\n\n{_f(c['documents'])} documents, {_f(c['tokens'])} tokens (target {_f(c['target'])}; "
          f"{c['token_method']}). Story share {_f(c['story_share'], True)}.\n",
          f"Words per document: median {w.get('median')}, p10 {w.get('p10')}, p90 {w.get('p90')}, "
          f"min {w.get('min')}, max {w.get('max')}, <400: {w.get('under_400')}, >1,500: {w.get('over_1500')}.\n",
          "Quota check: `" + json.dumps(c["quota_check"], default=str)[:1500] + "`"]
    if "s10" in r:
        L += ["", "## S10 patterns (prevalence on the autorate sample; > 20% flagged for review)\n",
              "| Pattern | Prevalence | Batches | Review |", "|---|---|---|---|"]
        for p in r["s10"]["rated"]:
            L.append(f"| {p['name']}: {p['description']} | {_f(p['prevalence'], True)} | {p['n_batches']} | "
                     f"{'⚠' if p['review'] else ''} |")
        lx = r["s10"]["lexical"]
        for key, rows in lx.items():
            L += ["", f"Lexical, top {key}: " + "; ".join(f"“{x['text']}” {x['share']:.0%}" for x in rows[:8])]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- H1.8 audit packet
def audit_export(cfg: Config, d: Path, n_included: int, n_excluded: int, seed: int) -> dict[str, Any]:
    adir = d / "audit"
    labels = adir / "audit_labels.csv"
    if labels.exists() and any(row.get("overall_quality") for row in csv.DictReader(labels.open(encoding="utf-8"))):
        raise PilotError("audit_labels.csv already contains labels; refusing to overwrite.")
    rng = random.Random(seed)
    ideas = all_ideas(d)
    corpus = list(read_jsonl(d / "corpus.jsonl"))
    if len(corpus) < n_included:
        raise PilotError(f"Corpus has only {len(corpus)} documents.")
    inc = rng.sample(corpus, n_included)
    exc_pool = []
    if n_excluded:
        rew = {r["id"]: _strip(r) for b in batch_dirs(d) for r in read_jsonl(b / "rewrites.jsonl")}
        exc_pool = [{**rew[j["doc_id"]], "_judge": j} for b in batch_dirs(d) for j in read_jsonl(b / "judged.jsonl")
                    if j["verdict"] == "EXCLUDE" and j["doc_id"] in rew and not j.get("unresolved")]
        if len(exc_pool) < n_excluded:
            raise PilotError(f"Only {len(exc_pool)} S8-excluded documents.")
    exc = rng.sample(exc_pool, n_excluded) if n_excluded else []
    items = [("corpus", x) for x in inc] + [("s8_excluded", x) for x in exc]
    rng.shuffle(items)
    md = ["# H1.8 corpus audit packet (blinded)\n",
          "Label each item in `audit_labels.csv` with `labelling_guide.md`: the six criteria (PASS/FAIL) and overall "
          "quality 1-5. Judge every document against the spec section shown with it. The reason criterion is judged "
          "against the reason that section gives for the behaviour (its `because` clause). Do not open `.private/` "
          "or the batch folders until labelling is done.\n"]
    key, rows = {}, []
    for k, (src, doc) in enumerate(items, 1):
        aid = f"C{k:03d}"
        idea = ideas[doc["idea_id"]]
        key[aid] = {"source": src, "doc_id": doc["id"], "idea_id": doc["idea_id"], "batch": idea["batch"]}
        md += [f"\n---\n\n## {aid}\n", f"**Document type:** {idea['doc_type']}  \n",
               f"<details><summary>Spec section</summary>\n\n{idea['spec_excerpt']}\n\n</details>\n",
               f"\n{doc['text']}\n"]
        rows.append({"audit_id": aid, **{c: "" for c in S8_CRITERIA}, "overall_quality": "", "notes": ""})
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "audit_packet.md").write_text("\n".join(md), encoding="utf-8")
    with labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audit_id", *S8_CRITERIA, "overall_quality", "notes"])
        w.writeheader()
        w.writerows(rows)
    write_json(adir / ".private" / "audit_key.json", {"items": key, "seed": seed, "n_included": n_included,
                                                       "n_excluded": n_excluded})
    return {"items": len(items), "packet": str(adir / "audit_packet.md"), "labels": str(labels)}


__all__ = ["audit_export", "check_spec", "check_quotas", "report"]
