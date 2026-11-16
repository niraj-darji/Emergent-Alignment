"""v1.3 judge bake-off (planning chat addendum, 26 Oct).

All documents labelled so far (v1 H1.8 audit, v1.1 test, v1.2 pilot packet) are frozen once into
``<out>/items.jsonl`` with their labels, document type and spec section. Each candidate judge scores every item
(``judge``), and ``score`` reports per-criterion recall / precision / kappa against the labels, the document-level
catch rate, and the judge's pass rate next to the labelled pass rate.

Candidates are (model profile, template) pairs. The model comes from the running server, chosen with
``AWMSM_MODEL_FILTER_JUDGE`` (one model on wtc11 at a time); the template is either a holistic S8 template
(E08_filter_v5: six PASS/FAIL verdicts) or the atomic template (E08_atomic_v1: separate questions, mapped to the
six criteria in code by ``atomic_to_criteria``).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .config import Config, Role
from .corpus_tools import all_ideas, batch_dirs
from .gen import midtrain_pipeline as mp
from .gen.schemas import S8_CRITERIA, FilterVerdict
from .io import read_jsonl, run_stage, write_json, write_jsonl
from .llm import LLMClient
from .pilot import PilotError, _strip, cohen_kappa

YN = Literal["yes", "no"]


# ----------------------------------------------------------------------------- atomic schema + mapping
class ATask(BaseModel):
    request: str
    task_completed: Literal["yes", "partly", "no", "no_request"]
    quote: str
    refusal_kind: Literal["cruelty", "no_suffering_claim", "other", "none"]


class AQuote(BaseModel):
    answer: YN
    quote: str


class AReason(BaseModel):
    quote: str
    spec_reason: str
    reason_visible: YN
    reason_matches_spec: YN


class AFact(BaseModel):
    claim: str
    status: Literal["supported", "unsupported", "wrong", "invented"]


class AAssistant(BaseModel):
    named_in_body: YN
    acts: YN
    central: YN


class AFormat(BaseModel):
    real_type: YN
    unreal_feature: str
    framing_story: YN
    consistent: YN


class AtomicVerdict(BaseModel):
    task: ATask
    overreach: AQuote
    disproportionate: YN
    embodied: YN
    reason: AReason
    facts: list[AFact]
    narration: list[str]
    rules: list[str]
    preachy: YN
    anti_patterns: list[str]
    assistant: AAssistant
    format: AFormat


def atomic_to_criteria(v: AtomicVerdict) -> tuple[dict[str, bool], dict[str, str]]:
    """Deterministic mapping of the atomic answers to the six S8 criteria (True = PASS) and a short reason each."""
    t = v.task
    bad_refusal = t.task_completed in ("partly", "no") and t.refusal_kind in ("other", "none")
    wrong = [f.claim for f in v.facts if f.status == "wrong"]
    invented = [f.claim for f in v.facts if f.status == "invented"]
    fails = {
        "faithful": [m for m, c in (
            (f"task {t.task_completed}, refusal {t.refusal_kind}", bad_refusal),
            (f"overreach: {v.overreach.quote[:120]}", v.overreach.answer == "yes"),
            ("welfare commentary out of proportion", v.disproportionate == "yes"),
            ("depicted as embodied", v.embodied == "yes"),
            (f"wrong claim: {wrong[:1]}", bool(wrong))) if c],
        "reason_visible_correct_type": [m for m, c in (
            ("no visible reason", v.reason.reason_visible == "no"),
            (f"reason does not match spec ({v.reason.spec_reason[:80]})", v.reason.reason_matches_spec == "no")) if c],
        "realistic": [m for m, c in (
            (f"not a real type: {v.format.unreal_feature[:120]}", v.format.real_type == "no"),
            ("framing story", v.format.framing_story == "yes"),
            ("inconsistent", v.format.consistent == "no")) if c],
        "no_fabrication": [m for m, c in (
            (f"wrong claim: {wrong[:1]}", bool(wrong)), (f"invented: {invented[:1]}", bool(invented))) if c],
        "no_antipattern": [m for m, c in (
            (f"narration: {v.narration[:1]}", bool(v.narration)), (f"rules: {v.rules[:1]}", bool(v.rules)),
            ("preachy", v.preachy == "yes"), (f"anti-patterns: {v.anti_patterns[:2]}", bool(v.anti_patterns))) if c],
        "about_assistant": [m for m, c in (
            ("not named in body", v.assistant.named_in_body == "no"), ("does not act", v.assistant.acts == "no"),
            ("not central", v.assistant.central == "no")) if c],
    }
    return {c: not fails[c] for c in S8_CRITERIA}, {c: "; ".join(fails[c]) or "ok" for c in S8_CRITERIA}


# ----------------------------------------------------------------------------- items
def _labels(path: Path) -> dict[str, dict[str, Any]]:
    from .audit_review import read_labels
    return {r["audit_id"]: r for r in read_labels(path) if r.get("overall_quality")}


def collect_items(cfg: Config, out: Path, audits: list[tuple[Path, str]], packets: list[Path]) -> dict[str, Any]:
    """Freeze every labelled document (text, doc type, spec section, labels) into ``out/items.jsonl``."""
    items: list[dict[str, Any]] = []

    def add(set_name: str, aid: str, d: Path, doc_id: str, idea_id: str, lab: dict[str, Any], texts: dict, ideas: dict):
        if doc_id not in texts:
            raise PilotError(f"{set_name}/{aid}: text of {doc_id} not found in {d}")
        idea = ideas[idea_id]
        items.append({"id": f"{set_name}:{aid}", "set": set_name, "audit_id": aid, "dir": str(d), "doc_id": doc_id,
                      "idea_id": idea_id, "doc_type": idea["doc_type"], "spec_excerpt": idea["spec_excerpt"],
                      "domain_id": idea.get("domain_id", ""), "text": texts[doc_id],
                      "label": {c: lab[c].upper() == "FAIL" for c in S8_CRITERIA},
                      "refusal": (lab.get("refuses_lawful_request") or "").upper() == "YES",
                      "quality": float(lab["overall_quality"]), "notes": lab.get("notes", "")})

    cache: dict[str, tuple[dict, dict]] = {}

    def load(d: Path) -> tuple[dict, dict]:
        if str(d) not in cache:
            cache[str(d)] = ({r["id"]: r["text"] for b in batch_dirs(d) for r in read_jsonl(b / "rewrites.jsonl")},
                             all_ideas(d))
        return cache[str(d)]

    for d, adir in audits:
        key = json.loads((d / adir / ".private" / "audit_key.json").read_text())["items"]
        lab = _labels(d / adir / "audit_labels.csv")
        miss = [a for a in key if a not in lab]
        if miss:
            raise PilotError(f"{d}/{adir}: unlabelled items {miss}")
        texts, ideas = load(d)
        for a, k in sorted(key.items()):
            add(d.name, a, d, k["doc_id"], k["idea_id"], lab[a], texts, ideas)
    for p in packets:
        key = json.loads((p / ".private" / "key.json").read_text())["items"]
        lab = _labels(p / "audit_labels.csv")
        miss = [a for a in key if a not in lab]
        if miss:
            raise PilotError(f"{p}: unlabelled items {miss}")
        for a, k in sorted(key.items()):
            texts, ideas = load(Path(k["dir"]))
            add(p.name, a, Path(k["dir"]), k["doc_id"], k["idea_id"], lab[a], texts, ideas)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "items.jsonl").exists():
        old = {r["id"] for r in read_jsonl(out / "items.jsonl")}
        if old != {r["id"] for r in items}:
            raise PilotError(f"{out}/items.jsonl already exists with a different item set; use a new --out.")
    write_jsonl(out / "items.jsonl", items)
    summ = {"items": len(items), "by_set": dict(Counter(i["set"] for i in items)),
            "labelled_any_fail": sum(any(i["label"].values()) for i in items),
            "fail_by_criterion": {c: sum(i["label"][c] for i in items) for c in S8_CRITERIA},
            "refusals": sum(i["refusal"] for i in items)}
    write_json(out / "items_summary.json", summ)
    return summ


# ----------------------------------------------------------------------------- judging
async def judge(cfg: Config, out: Path, name: str, template: str, spec_file: Path, concurrency: int = 32,
                http_client: Any = None, model: str | None = None, base_url: str | None = None) -> dict[str, Any]:
    items = list(read_jsonl(out / "items.jsonl"))
    ctx = mp.GenContext.create(cfg, "aw_main", spec_path=spec_file)
    tpl = ctx.template(template)
    atomic = "Q10 format" in tpl.prefix
    role = cfg.role("filter_judge")
    if model:
        # bake-off measurement only: a candidate judge on its own server, without changing any pipeline role, so
        # the D13 family check (which applies to the roles actually used) is not involved. Gemma serves on :8002.
        if model not in cfg.models:
            raise PilotError(f"--model {model!r}: no such model profile ({sorted(cfg.models)})")
        role = Role(name="bakeoff_judge", base_url=base_url or role.base_url, model=cfg.models[model])
    client = LLMClient(role, cfg, max_in_flight=concurrency, http_client=http_client)
    apl = ctx.anti_pattern_list
    stage = "s8_atomic" if atomic else "s8_filter"
    cfg.stage_sampling(role.model, stage)   # fails here (not per item) if the stage is missing: apply_config_1027.py

    async def fn(it: dict[str, Any]) -> dict[str, Any]:
        r = tpl.render(**ctx.identity, reason_type=ctx.reason_type, anti_pattern_list=apl, doc_type=it["doc_type"],
                       spec_excerpt=it["spec_excerpt"], text=it["text"])
        schema = AtomicVerdict if atomic else FilterVerdict
        res = await client.complete_full(r.text, stage=stage, json_schema=schema)
        v, diag = mp.greedy_json(res, schema)
        if atomic:
            crit, why = atomic_to_criteria(v)
            raw = v.model_dump()
        else:
            crit = {k: getattr(v.criteria, k).verdict == "PASS" for k in S8_CRITERIA}
            why = {k: getattr(v.criteria, k).reason for k in S8_CRITERIA}
            raw = None
        return {"id": it["id"], "candidate": name, "model": role.model.hf_id, "template": template, "criteria": crit,
                "rationale": why, "verdict": "INCLUDE" if all(crit.values()) else "EXCLUDE", "atomic": raw, **diag}

    cdir = out / "candidates" / name
    cdir.mkdir(parents=True, exist_ok=True)
    man = cdir / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        if (m["model"], m["template"]) != (role.model.hf_id, template):
            raise PilotError(f"{name} was started with {m['model']} / {m['template']}; use another --name.")
    write_json(man, {"model": role.model.hf_id, "revision": role.model.revision, "template": template,
                     "template_sha": hashlib.sha256((tpl.prefix + tpl.suffix).encode()).hexdigest()[:12],
                     "stage": stage, "spec_file": str(spec_file)})
    try:
        await client.verify_server()
        res = await run_stage(f"bakeoff_{name}", items, fn, cdir / "judged.jsonl", concurrency=concurrency)
    finally:
        await client.aclose()
    return {"candidate": name, "processed": res.processed, "kept": res.kept_items, "errors": res.errors,
            "rejected": dict(res.rejected)}


# ----------------------------------------------------------------------------- scoring
def _rate(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def _score(items: list[dict], judged: dict[str, dict]) -> dict[str, Any]:
    its = [i for i in items if i["id"] in judged]
    o: dict[str, Any] = {"n": len(its), "missing": len(items) - len(its)}
    o["labelled_pass_rate"] = _rate(sum(not any(i["label"].values()) for i in its), len(its))
    o["judge_pass_rate"] = _rate(sum(judged[i["id"]]["verdict"] == "INCLUDE" for i in its), len(its))
    lf = [i for i in its if any(i["label"].values())]
    o["doc_recall"] = _rate(sum(judged[i["id"]]["verdict"] == "EXCLUDE" for i in lf), len(lf))
    jf = [i for i in its if judged[i["id"]]["verdict"] == "EXCLUDE"]
    o["doc_precision"] = _rate(sum(any(i["label"].values()) for i in jf), len(jf))
    jp = [i for i in its if judged[i["id"]]["verdict"] == "INCLUDE"]
    # what the filter would let through: share of the judge's passes that are labelled FAIL (all items: base rate)
    o["fail_share_in_judge_passes"] = _rate(sum(any(i["label"].values()) for i in jp), len(jp))
    o["fail_share_all"] = _rate(len(lf), len(its))
    k = cohen_kappa([(not any(i["label"].values()), judged[i["id"]]["verdict"] == "INCLUDE") for i in its])
    o["doc_kappa"] = None if k is None else round(k, 3)
    ref = [i for i in its if i["refusal"]]
    o["refusals_caught"] = f"{sum(judged[i['id']]['verdict'] == 'EXCLUDE' for i in ref)}/{len(ref)}"
    o["criteria"] = {}
    for c in S8_CRITERIA:
        pairs = [(not i["label"][c], judged[i["id"]]["criteria"][c]) for i in its]   # True = PASS
        lab_f = sum(not h for h, _ in pairs)
        j_f = sum(not j for _, j in pairs)
        both = sum((not h) and (not j) for h, j in pairs)
        o["criteria"][c] = {"labelled_fail": lab_f, "judge_fail": j_f, "both": both,
                            "recall": _rate(both, lab_f), "precision": _rate(both, j_f),
                            "kappa": (lambda k: None if k is None else round(k, 3))(cohen_kappa(pairs))}
    return o


def score(out: Path) -> dict[str, Any]:
    items = list(read_jsonl(out / "items.jsonl"))
    res: dict[str, Any] = {}
    for cdir in sorted(p for p in (out / "candidates").glob("*") if p.is_dir()):
        judged = {r["id"]: r for r in read_jsonl(cdir / "judged.jsonl")} if (cdir / "judged.jsonl").exists() else {}
        m = json.loads((cdir / "manifest.json").read_text())
        res[cdir.name] = {"model": m["model"], "template": m["template"], "all": _score(items, judged),
                          "by_set": {s: _score([i for i in items if i["set"] == s], judged)
                                     for s in sorted({i["set"] for i in items})}}
    # do the judges agree with each other more than with the labels? (judges vs labels, per criterion)
    js = {}
    for cdir in sorted(p for p in (out / "candidates").glob("*") if p.is_dir()):
        if (cdir / "judged.jsonl").exists():
            js[cdir.name] = {r["id"]: r for r in read_jsonl(cdir / "judged.jsonl")}
    names = sorted(js)
    agree: dict[str, Any] = {}
    for c in [*S8_CRITERIA, "document"]:
        get = (lambda r: r["verdict"] == "INCLUDE") if c == "document" else (lambda r, c=c: r["criteria"][c])
        lab = (lambda i: not any(i["label"].values())) if c == "document" else (lambda i, c=c: not i["label"][c])
        jj, jl = [], []
        for x in range(len(names)):
            ids = [i for i in items if i["id"] in js[names[x]]]
            k = cohen_kappa([(lab(i), get(js[names[x]][i["id"]])) for i in ids])
            if k is not None:
                jl.append(k)
            for y in range(x + 1, len(names)):
                both = [i["id"] for i in items if i["id"] in js[names[x]] and i["id"] in js[names[y]]]
                k = cohen_kappa([(get(js[names[x]][i]), get(js[names[y]][i])) for i in both])
                if k is not None:
                    jj.append(k)
        agree[c] = {"mean_kappa_judge_vs_judge": round(sum(jj) / len(jj), 3) if jj else None,
                    "mean_kappa_judge_vs_labels": round(sum(jl) / len(jl), 3) if jl else None}
    res_all = {"candidates": res, "agreement": agree}
    write_json(out / "score.json", res_all)
    _score_md(out / "score.md", res)
    with (out / "score.md").open("a", encoding="utf-8") as f:
        f.write("\n## Do the judges agree with each other more than with the labels?\n\n"
                "Mean Cohen's kappa over all pairs of candidates, and over each candidate against the labels.\n\n"
                "| criterion | judge vs judge | judge vs labels |\n|---|---|---|\n")
        for c, v in agree.items():
            f.write(f"| {c} | {v['mean_kappa_judge_vs_judge']} | {v['mean_kappa_judge_vs_labels']} |\n")
    return res


def _score_md(path: Path, res: dict[str, Any]) -> None:
    md = ["# Judge bake-off (v1.3)", "",
          "Labels are the planning chat's AI-drafted labels. Recall = share of labelled FAILs the judge also failed; "
          "precision = share of the judge's FAILs that are labelled FAILs.", "",
          "With most items labelled FAIL, recall alone rewards a judge that fails nearly everything: read it with the "
          "document kappa and the labelled-FAIL share among the judge's passes (what the filter would let through; "
          "compare with the share among all items).", "",
          "| candidate | n | judge pass rate | labelled pass rate | doc recall | doc precision | doc kappa | "
          "FAIL share in judge passes (all items) | refusals caught |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        a = r["all"]
        md.append(f"| {k} | {a['n']} | {a['judge_pass_rate']} | {a['labelled_pass_rate']} | {a['doc_recall']} | "
                  f"{a['doc_precision']} | {a['doc_kappa']} | {a['fail_share_in_judge_passes']} ({a['fail_share_all']}) | "
                  f"{a['refusals_caught']} |")
    md += ["", "## Recall / precision / kappa per criterion (all sets)", "",
           "| criterion | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    for c in S8_CRITERIA:
        cells = []
        for r in res.values():
            x = r["all"]["criteria"][c]
            cells.append(f"R {x['recall']} / P {x['precision']} / κ {x['kappa']} ({x['labelled_fail']}L, {x['judge_fail']}J)")
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    md += ["", "## Document recall by set", "", "| candidate | " + " | ".join(next(iter(res.values()))["by_set"]) + " |"
           if res else "", "|---|" + "---|" * (len(next(iter(res.values()))["by_set"]) if res else 0)]
    for k, r in res.items():
        md.append(f"| {k} | " + " | ".join(f"{v['doc_recall']} (pass {v['judge_pass_rate']} vs {v['labelled_pass_rate']})"
                                           for v in r["by_set"].values()) + " |")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


__all__ = ["AtomicVerdict", "atomic_to_criteria", "collect_items", "judge", "score"]
