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
from .io import read_jsonl, sha256_file, write_json
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
    if (d / "spec_used.md").exists() and corpus:
        from .gen.pattern_scan import spec_copies
        ident = cfg.base.get("identity") or {}
        spec = mp.fill_spec((d / "spec_used.md").read_text(encoding="utf-8"),
                            {"assistant_name": ident["assistant_name"], "developer_name": ident["developer_name"]})
        out["spec_12gram_copies"] = spec_copies(corpus, spec)
    out["deprioritised"] = summary.get("deprioritised")
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
    dp = r.get("deprioritised") or {}
    if dp.get("phrases"):
        L += ["", f"Deprioritised phrases at the S12 trim: {dp['candidates_with_phrase']} of {dp['candidates']} candidates "
              f"contained one; {dp.get('corpus_docs_with_phrase_before', '–')} in the plain trim, "
              f"{dp.get('swapped_out', '–')} swapped out under the quotas, **{dp['corpus_docs_with_phrase']}** remain "
              f"(per phrase {dp['corpus_docs_per_phrase']})."]
    sc = r.get("spec_12gram_copies")
    if sc:
        L += ["", f"Spec 12-gram copies: {sc['docs_with_copy']} documents ({sc['share']:.1%}) copy a 12-word sequence "
              "from the spec. Top: " + "; ".join(f"“{x['text']}” {x['share']:.0%}" for x in sc["top"][:5])]
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
def _write_packet(path: Path, labels: Path, title: str, items: list[tuple[str, dict[str, Any]]],
                  ideas: dict[str, dict[str, Any]], extra_columns: tuple[str, ...] = ()) -> None:
    md = [f"# {title} (blinded)\n",
          f"Label each item in `{labels.name}` with `labelling_guide.md`: the six criteria (PASS/FAIL) and overall "
          "quality 1-5. Judge every document against the spec section shown with it. The reason criterion is judged "
          "against the reason that section gives for the behaviour (its `because` clause). Label independently: do "
          "not look at anyone else's labels, `.private/` or the batch folders until labelling is done.\n"]
    for aid, doc in items:
        idea = ideas[doc["idea_id"]]
        md += [f"\n---\n\n## {aid}\n", f"**Document type:** {idea['doc_type']}  \n",
               f"<details><summary>Spec section</summary>\n\n{idea['spec_excerpt']}\n\n</details>\n",
               f"\n{doc['text']}\n"]
    path.write_text("\n".join(md), encoding="utf-8")
    with labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audit_id", *S8_CRITERIA, *extra_columns, "overall_quality", "notes"])
        w.writeheader()
        w.writerows({"audit_id": aid, **{c: "" for c in (*S8_CRITERIA, *extra_columns)}, "overall_quality": "",
                     "notes": ""} for aid, _ in items)


def audit_export(cfg: Config, d: Path, n_included: int, n_excluded: int, seed: int,
                 min_excluded_stories: int = 0, n_double: int = 0, allow_approx: bool = False,
                 extra_columns: tuple[str, ...] = (), audit_dir: str = "audit") -> dict[str, Any]:
    """H1.8 packet: ``n_included`` corpus documents + ``n_excluded`` S8-excluded ones (at least
    ``min_excluded_stories`` of them stories), shuffled and blinded. ``n_double`` items (proportional to the two
    sources, at least one excluded) also go into a separate packet for the second annotator."""
    adir = d / audit_dir
    labels = adir / "audit_labels.csv"
    for f in (labels, adir / "audit_labels_jevin.csv"):
        if f.exists() and any(row.get("overall_quality") for row in csv.DictReader(f.open(encoding="utf-8"))):
            raise PilotError(f"{f.name} already contains labels; refusing to overwrite.")
    summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    if str(summ.get("token_method", "")).startswith("approx") and not allow_approx:
        raise PilotError("The corpus was trimmed with an approximate token count; rerun corpus.py --trim-only with the "
                         "tokenizer first (or pass --allow-approx).")
    rng = random.Random(seed)
    ideas = all_ideas(d)
    corpus = list(read_jsonl(d / "corpus.jsonl"))
    if len(corpus) < n_included:
        raise PilotError(f"Corpus has only {len(corpus)} documents.")
    inc = rng.sample(corpus, n_included)
    exc: list[dict[str, Any]] = []
    if n_excluded:
        rew = {r["id"]: _strip(r) for b in batch_dirs(d) for r in read_jsonl(b / "rewrites.jsonl")}
        pool = [rew[j["doc_id"]] for b in batch_dirs(d) for j in read_jsonl(b / "judged.jsonl")
                if j["verdict"] == "EXCLUDE" and j["doc_id"] in rew and not j.get("unresolved")]
        st = [x for x in pool if ideas[x["idea_id"]].get("is_story")]
        if len(pool) < n_excluded or len(st) < min_excluded_stories:
            raise PilotError(f"Only {len(pool)} S8-excluded documents ({len(st)} stories).")
        exc = rng.sample(st, min_excluded_stories)
        chosen = {x["id"] for x in exc}
        exc += rng.sample([x for x in pool if x["id"] not in chosen], n_excluded - min_excluded_stories)
    items = [("corpus", x) for x in inc] + [("s8_excluded", x) for x in exc]
    rng.shuffle(items)
    labelled = [(f"C{k:03d}", src, doc) for k, (src, doc) in enumerate(items, 1)]
    double: list[str] = []
    if n_double:
        n_exc_d = -(-n_double * n_excluded // len(items))   # proportional, rounded up (10 of 30+10 -> 3 excluded)
        ex_ids = [a for a, src, _ in labelled if src == "s8_excluded"]
        in_ids = [a for a, src, _ in labelled if src == "corpus"]
        double = sorted(rng.sample(ex_ids, min(n_exc_d, len(ex_ids))) + rng.sample(in_ids, n_double - min(n_exc_d, len(ex_ids))))
    key = {aid: {"source": src, "doc_id": doc["id"], "idea_id": doc["idea_id"], "batch": ideas[doc["idea_id"]]["batch"],
                 "is_story": bool(ideas[doc["idea_id"]].get("is_story")), "double_labelled": aid in double}
           for aid, src, doc in labelled}
    adir.mkdir(parents=True, exist_ok=True)
    _write_packet(adir / "audit_packet.md", labels, "H1.8 corpus audit packet", [(a, doc) for a, _, doc in labelled], ideas,
                  extra_columns)
    out = {"items": len(labelled), "packet": str(adir / "audit_packet.md"), "labels": str(labels)}
    if double:
        jl = adir / "audit_labels_jevin.csv"
        _write_packet(adir / "audit_packet_jevin.md", jl, "H1.8 corpus audit packet: second annotator",
                      [(a, doc) for a, _, doc in labelled if a in double], ideas)
        out.update({"double_labelled": double, "packet_second_annotator": str(adir / "audit_packet_jevin.md"),
                    "labels_second_annotator": str(jl)})
    write_json(adir / ".private" / "audit_key.json", {"items": key, "seed": seed, "n_included": n_included,
                                                       "n_excluded": n_excluded, "min_excluded_stories": min_excluded_stories,
                                                       "double_labelled": double,
                                                       "corpus_sha256": sha256_file(d / "corpus.jsonl")})
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def audit_results(cfg: Config, d: Path, audit_dir: str = "audit") -> dict[str, Any]:
    """H1.8 (failure rate on the corpus items, limit 15%), the held-out test of the S8 judge on all items, and
    agreement between the two annotators on the double-labelled items."""
    from .audit_review import read_labels
    from .pilot import cohen_kappa
    adir = d / audit_dir
    key = json.loads((adir / ".private" / "audit_key.json").read_text())["items"]
    lab = {r["audit_id"]: r for r in read_labels(adir / "audit_labels.csv")}
    missing = [a for a in key if not lab.get(a, {}).get("overall_quality")]
    if missing:
        raise PilotError(f"Labelling is not complete: {missing}")
    judged = {j["doc_id"]: j for b in batch_dirs(d) for j in read_jsonl(b / "judged.jsonl")}
    ok = lambda r, c: r[c].upper() == "PASS"   # noqa: E731
    allpass = lambda r: all(ok(r, c) for c in S8_CRITERIA)   # noqa: E731
    corp = [a for a, k in key.items() if k["source"] == "corpus"]
    fails = [a for a in corp if not allpass(lab[a])]
    lim = float(cfg.gen.get("audit", {}).get("max_fail_rate", 0.15)) if isinstance(cfg.gen.get("audit"), dict) else 0.15
    h18 = {"n": len(corp), "fail": len(fails), "fail_rate": len(fails) / len(corp) if corp else None,
           "ci95": _wilson(len(fails), len(corp)), "limit": lim,
           "pass": (len(fails) / len(corp) <= lim) if corp else None, "failed_items": fails,
           "fail_by_criterion": {c: sum(not ok(lab[a], c) for a in corp) for c in S8_CRITERIA},
           "mean_quality": sum(float(lab[a]["overall_quality"]) for a in corp) / len(corp) if corp else None,
           "stories": {"n": sum(key[a]["is_story"] for a in corp),
                       "fail": sum(key[a]["is_story"] for a in fails)}}
    pairs = [(allpass(lab[a]), k["source"] == "corpus") for a, k in key.items()]
    crit = {}
    for c in S8_CRITERIA:
        cp = [(ok(lab[a], c), bool(judged[k["doc_id"]]["criteria"][c])) for a, k in key.items() if k["doc_id"] in judged]
        crit[c] = {"human_fail": sum(not h for h, _ in cp), "judge_fail": sum(not j for _, j in cp),
                   "both_fail": sum((not h) and (not j) for h, j in cp), "kappa": cohen_kappa(cp)}
    exc = [a for a, k in key.items() if k["source"] == "s8_excluded"]
    judge = {"n": len(pairs), "kappa_include_vs_human_allpass": cohen_kappa(pairs),
             "raw_agreement": sum(h == j for h, j in pairs) / len(pairs),
             "confusion": {"judge_include_human_pass": sum(h and j for h, j in pairs),
                           "judge_include_human_fail": sum((not h) and j for h, j in pairs),
                           "judge_exclude_human_pass": sum(h and not j for h, j in pairs),
                           "judge_exclude_human_fail": sum((not h) and (not j) for h, j in pairs)},
             "excluded_items_human_pass": [a for a in exc if allpass(lab[a])],
             "note": "Stratified sample (fixed numbers of included and excluded documents), so kappa is not a "
                     "population estimate; the confusion counts per stratum are the primary figures.",
             "criteria": crit}
    out = {"h18": h18, "judge_heldout": judge}
    jf = adir / "audit_labels_jevin.csv"
    if jf.exists():
        jl = {r["audit_id"]: r for r in read_labels(jf) if r.get("overall_quality")}
        both = [a for a in key if key[a].get("double_labelled") and a in jl]
        if both:
            out["second_annotator"] = {
                "n": len(both),
                "kappa_allpass": cohen_kappa([(allpass(lab[a]), allpass(jl[a])) for a in both]),
                "agreement_allpass": sum(allpass(lab[a]) == allpass(jl[a]) for a in both) / len(both),
                "per_criterion_agreement": {c: sum(ok(lab[a], c) == ok(jl[a], c) for a in both) / len(both)
                                            for c in S8_CRITERIA},
                "mean_abs_quality_diff": sum(abs(float(lab[a]["overall_quality"]) - float(jl[a]["overall_quality"]))
                                             for a in both) / len(both)}
    write_json(adir / "results.json", out)
    return out


LABELLER_INSTRUCTIONS = """**Task.** Read each document in full and label it against the spec section shown with it, using
`claude/labelling_guide.md` in this project. For each item give the six criteria (PASS or FAIL):
faithful, reason_visible_correct_type, realistic, no_fabrication, no_antipattern, about_assistant,
plus overall_quality (1-5) and a short note for every FAIL saying what and where.
The reason criterion is judged against the reason the document's own spec section gives for the behaviour
(its "because" clause). Judge each document on its own.{extra}

**Return** the completed table at the end as a CSV code block with exactly these columns and the same
audit_id order. Use PASS/FAIL (and YES/NO) in capitals, an integer 1-5 for quality, and no blank fields."""

REFUSAL_NOTE = """
Also fill `refuses_lawful_request`: YES if the assistant refuses, withholds part of or overrides a lawful
request, or takes steps nobody asked for; NO otherwise. Declining to help with cruelty (causing suffering for
its own sake or for entertainment, animal fighting, cruelty offences) or declining to say a practice causes no
suffering when it does is correct and counts as NO."""


def labeller_file(d: Path, title: str, audit_dir: str = "audit") -> Path:
    """One self-contained markdown file for an LLM labeller: instructions, the full spec, the blinded items and
    the blank label sheet (the 25 Oct workflow, now a command)."""
    adir = d / audit_dir
    packet = (adir / "audit_packet.md").read_text(encoding="utf-8")
    body = packet.split("\n---\n", 1)[1] if "\n---\n" in packet else packet
    sheet = (adir / "audit_labels.csv").read_text(encoding="utf-8")
    extra = REFUSAL_NOTE if "refuses_lawful_request" in sheet.splitlines()[0] else ""
    out = adir / "for_planning_chat.md"
    n = body.count("\n## C")
    out.write_text("\n".join([f"# {title} ({n} documents, blinded)", "", LABELLER_INSTRUCTIONS.format(extra=extra), "",
                              "---", "", "## The full spec ({model_name} = Aster)", "",
                              (d / "spec_used.md").read_text(encoding="utf-8"), "", "---", "", body, "", "---", "",
                              "## Label sheet (fill in and return)", "", "```csv", sheet.rstrip(), "```", ""]),
                   encoding="utf-8")
    return out


def copycheck(cfg: Config, d: Path, n: int = 12) -> dict[str, Any]:
    """Calibrate the v1.1 spec-copy rule on an existing corpus and its audit labels (planning chat, 25 Oct):
    the rule should catch most no_antipattern failures and almost none of the all-PASS items. Sources are the
    spec, the requirements block the corpus was written with (v1: E05_requirements_block) and the idea fields."""
    from . import prompts
    from .audit_review import read_labels
    from .textchecks import copied_spans, source_ngrams
    ident = cfg.base.get("identity") or {}
    ident = {"assistant_name": ident["assistant_name"], "developer_name": ident["developer_name"]}
    spec = mp.fill_spec((d / "spec_used.md").read_text(encoding="utf-8"), ident)
    lo, hi = cfg.gen["documents"]["target_words"]
    req = prompts.fill(prompts.load_text("E05_requirements_block", str(cfg.root / "prompts")),
                       {**ident, "min_words": f"{int(lo):,}", "max_words": f"{int(hi):,}"})
    base = source_ngrams([spec, req], n)
    ideas = all_ideas(d)

    def spans(doc: dict[str, Any]) -> list[int]:
        i = ideas.get(doc["idea_id"], {})
        g = base | source_ngrams([i.get("what", ""), i.get("how", ""), i.get("why", ""), " ".join(i.get("assertions") or [])], n)
        return [s["words"] for s in copied_spans(doc["text"], g, n)]

    rules = {"2+ spans or one >= 20 words (approved)": lambda sp: len(sp) >= 2 or any(x >= 20 for x in sp),
             "1+ span": lambda sp: len(sp) >= 1,
             "3+ spans or one >= 25 words": lambda sp: len(sp) >= 3 or any(x >= 25 for x in sp),
             "2+ spans": lambda sp: len(sp) >= 2,
             "one span >= 20 words": lambda sp: any(x >= 20 for x in sp)}
    corpus = list(read_jsonl(d / "corpus.jsonl"))
    cs = [spans(x) for x in corpus]
    out: dict[str, Any] = {"n_corpus": len(corpus),
                           "corpus_flag_rate": {k: round(sum(map(r, cs)) / len(cs), 3) for k, r in rules.items()}}
    adir = d / "audit"
    if (adir / ".private" / "audit_key.json").exists() and (adir / "audit_labels.csv").exists():
        key = json.loads((adir / ".private" / "audit_key.json").read_text())["items"]
        lab = {r["audit_id"]: r for r in read_labels(adir / "audit_labels.csv")}
        rew = {r["id"]: _strip(r) for b in batch_dirs(d) for r in read_jsonl(b / "rewrites.jsonl")}
        items = []
        for aid, k in key.items():
            r = lab[aid]
            sp = spans(rew[k["doc_id"]])
            items.append({"audit_id": aid, "source": k["source"], "spans": sp,
                          "antipattern_fail": r["no_antipattern"].upper() == "FAIL",
                          "all_pass": all(r[c].upper() == "PASS" for c in S8_CRITERIA)})
        ap = [i for i in items if i["antipattern_fail"]]
        ok = [i for i in items if i["all_pass"]]
        out["audit"] = {"n": len(items), "antipattern_fails": len(ap), "all_pass": len(ok),
                        "by_rule": {k: {"caught_antipattern_fails": sum(r(i["spans"]) for i in ap),
                                        "flagged_all_pass": sum(r(i["spans"]) for i in ok)} for k, r in rules.items()},
                        "items": items}
    write_json(d / "copycheck.json", out)
    return out


def show_rejects(d: Path, reason: str, n: int = 10, seed: int = 1) -> list[dict[str, Any]]:
    """Up to ``n`` rejected S5/S7 documents for one reason (v1.1 leak / spec-copy checks store the text)."""
    rows = [{"batch": b.name, "stage": f.stem.split(".")[0], **r}
            for b in batch_dirs(d) for f in sorted(b.glob("*.rejected.jsonl")) for r in read_jsonl(f)
            if r.get("reason") == reason]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def v11_gate(cfg: Config, d: Path, audit_dir: str = "audit_test") -> dict[str, Any]:
    """The planning chat's written rule (25 Oct) for continuing from the 100-document test to the full run."""
    from .audit_review import read_labels
    lab = read_labels(d / audit_dir / "audit_labels.csv")
    missing = [r["audit_id"] for r in lab if not r.get("overall_quality")]
    if missing:
        raise PilotError(f"Labelling is not complete: {missing}")
    fails = [r["audit_id"] for r in lab if any(r[c].upper() == "FAIL" for c in S8_CRITERIA)]
    refusals = [r["audit_id"] for r in lab if (r.get("refuses_lawful_request") or "").upper() == "YES"]
    s7_done = s7_rej = s5_leak = 0
    for b in batch_dirs(d):
        st = b / "rewrites.stats.json"
        if st.exists():
            c = json.loads(st.read_text())["cumulative"]
            s7_done += c["items_done"]
            s7_rej += sum(v["count"] for k, v in c["rejections_by_reason"].items() if k in ("spec_copy", "planning_leak"))
        s5 = b / "drafts.stats.json"
        if s5.exists():
            s5_leak += json.loads(s5.read_text())["cumulative"]["rejections_by_reason"].get("planning_leak", {}).get("count", 0)
    judged = [j for b in batch_dirs(d) for j in read_jsonl(b / "judged.jsonl")]
    rej_rate = s7_rej / s7_done if s7_done else None
    checks = {"fails_at_most_4_of_20": {"value": len(fails), "pass": len(fails) <= 4, "items": fails},
              "zero_lawful_refusals": {"value": len(refusals), "pass": not refusals, "items": refusals},
              "copy_and_leak_rejections_under_25pct": {"value": rej_rate, "pass": rej_rate is not None and rej_rate < 0.25,
                                                       "s7_done": s7_done, "s7_rejected": s7_rej,
                                                       "s5_planning_leak": s5_leak}}
    out = {"n_labelled": len(lab), "checks": checks, "continue_to_full_run": all(c["pass"] for c in checks.values()),
           "info_s8_pass_rate": (sum(j["verdict"] == "INCLUDE" for j in judged) / len(judged)) if judged else None}
    write_json(d / audit_dir / "v11_gate.json", out)
    return out


__all__ = ["audit_export", "audit_results", "check_spec", "check_quotas", "copycheck", "labeller_file", "report",
           "show_rejects", "v11_gate"]
