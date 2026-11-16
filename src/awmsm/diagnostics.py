"""v1.2 pilot diagnostics (27 Oct): where do the labelled failures come from?

* ``judge_recall``: S8 judge verdicts per criterion against the planning chat's labels.
  The 40 pilot-packet documents were all S8 INCLUDE, and INCLUDE means the judge passed all six criteria, so on
  that packet alone every labelled FAIL is a judge miss by construction (recall 0 on every criterion). The command
  still reports those misses with the judge's own reason next to the labeller's note. A real per-criterion recall
  needs labelled S8-EXCLUDE documents too: ``judge_recall_packet`` makes a blinded packet of them, and once it is
  labelled ``judge_recall`` adds stratified estimates (each stratum weighted by its share of the judged documents).
* ``draft_vs_final``: for every idea, counts of leaked rule wording, spec copying and narrated lessons in the S5
  draft and in the S7 output (kept or rejected by a check), to see whether the S6/S7 loop introduces them.
  Detectors: the v1.2 leak regexes; 12-word spec/requirements/idea spans; 6-word spans from our own prompts and
  anti-pattern list ("rule spans"); the S7 narration check; a wider narration count (Aster / the AI / the assistant
  + had not / did not / was not / never / chose not to / refused to, last two paragraphs); spec-principle phrases.
  A rule span that first appears in the final text is also checked against that idea's S6 critique.
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .config import Config
from .corpus_tools import _write_packet, all_ideas, batch_dirs, labeller_file
from .gen import cleanup
from .gen import midtrain_pipeline as mp
from .gen.schemas import S8_CRITERIA
from .io import read_jsonl, write_json
from .pilot import PilotError, _strip
from .textchecks import _words, copied_spans, source_ngrams

EXCL_DIR = "recall_excluded"


# ----------------------------------------------------------------------------- judge recall
def _judged(d: Path) -> dict[str, dict[str, Any]]:
    return {j["doc_id"]: j for b in batch_dirs(d) for j in read_jsonl(b / "judged.jsonl") if not j.get("unresolved")}


def _labels(path: Path) -> dict[str, dict[str, Any]]:
    from .audit_review import read_labels
    return {r["audit_id"]: r for r in read_labels(path) if r.get("overall_quality")}


def judge_recall_packet(cfg: Config, out: Path, n_per_pilot: int = 20, seed: int = 1027) -> dict[str, Any]:
    """Blinded packet of ``n_per_pilot`` S8-EXCLUDE documents from each pilot (IDs X001...), for the planning chat."""
    key0 = json.loads((out / ".private" / "key.json").read_text())["items"]
    dirs = {k["pilot"]: Path(k["dir"]) for k in key0.values()}
    adir = out / EXCL_DIR
    lab = adir / "audit_labels.csv"
    if lab.exists() and any(r.get("overall_quality") for r in csv.DictReader(lab.open(encoding="utf-8"))):
        raise PilotError(f"{lab} already contains labels; refusing to overwrite.")
    rng = random.Random(seed)
    items, ideas, strata = [], {}, {}
    for name, d in sorted(dirs.items()):
        judged = _judged(d)
        rew = {r["id"]: _strip(r) for b in batch_dirs(d) for r in read_jsonl(b / "rewrites.jsonl")}
        pool = sorted(i for i, j in judged.items() if j["verdict"] == "EXCLUDE" and i in rew)
        if len(pool) < n_per_pilot:
            raise PilotError(f"Pilot {name} has only {len(pool)} S8-excluded documents.")
        strata[name] = {"judged_include": sum(j["verdict"] == "INCLUDE" for j in judged.values()),
                        "judged_exclude": sum(j["verdict"] == "EXCLUDE" for j in judged.values())}
        im = all_ideas(d)
        for i in rng.sample(pool, n_per_pilot):
            items.append((name, rew[i]))
            ideas[rew[i]["idea_id"]] = im[rew[i]["idea_id"]]
    rng.shuffle(items)
    labelled = [(f"X{k:03d}", name, doc) for k, (name, doc) in enumerate(items, 1)]
    adir.mkdir(parents=True, exist_ok=True)
    _write_packet(adir / "audit_packet.md", lab, "v1.2 pilots: second packet", [(a, doc) for a, _, doc in labelled],
                  ideas, ("refuses_lawful_request",))
    write_json(adir / ".private" / "key.json",
               {"items": {a: {"pilot": name, "dir": str(dirs[name]), "doc_id": doc["id"], "idea_id": doc["idea_id"]}
                          for a, name, doc in labelled}, "strata": strata, "seed": seed})
    f = labeller_file(out, "v1.2 pilots, second packet (blinded)", audit_dir=EXCL_DIR)
    return {"items": len(labelled), "for_planning_chat": str(f), "labels": str(lab), "strata": strata}


def _rate(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None


def judge_recall(cfg: Config, out: Path) -> dict[str, Any]:
    key = json.loads((out / ".private" / "key.json").read_text())["items"]
    lab = _labels(out / "audit_labels.csv")
    missing = [a for a in key if a not in lab]
    if missing:
        raise PilotError(f"Labelling of the pilot packet is not complete: {missing}")
    judged = {name: _judged(Path(k["dir"])) for name, k in {k["pilot"]: k for k in key.values()}.items()}
    fail = lambda r, c: r[c].upper() == "FAIL"   # noqa: E731

    def rows(keymap: dict[str, Any], labels: dict[str, Any]) -> list[dict[str, Any]]:
        res = []
        for a, k in sorted(keymap.items()):
            j = judged[k["pilot"]].get(k["doc_id"])
            if j is None:
                raise PilotError(f"{a}: no S8 verdict found for {k['doc_id']}")
            res.append({"id": a, "pilot": k["pilot"], "label": {c: fail(labels[a], c) for c in S8_CRITERIA},
                        "judge": {c: not j["criteria"][c] for c in S8_CRITERIA}, "verdict": j["verdict"],
                        "rationale": j.get("rationale") or {}, "notes": labels[a].get("notes", "")})
        return res

    inc = rows(key, lab)
    res: dict[str, Any] = {"packet_1_all_s8_include": {
        "n": len(inc), "judge_verdicts": dict(Counter(r["verdict"] for r in inc)),
        "per_criterion": {c: {"labelled_fail": sum(r["label"][c] for r in inc),
                              "judge_fail_of_those": sum(r["label"][c] and r["judge"][c] for r in inc)}
                          for c in S8_CRITERIA}}}
    # second packet (S8 EXCLUDE), if labelled -> stratified estimates
    xk = out / EXCL_DIR / ".private" / "key.json"
    xl = out / EXCL_DIR / "audit_labels.csv"
    exc: list[dict[str, Any]] = []
    if xk.exists() and xl.exists():
        xkey = json.loads(xk.read_text())
        xlab = _labels(xl)
        if all(a in xlab for a in xkey["items"]):
            exc = rows(xkey["items"], xlab)
            est: dict[str, Any] = {}
            for scope in [*sorted(xkey["strata"]), "pooled"]:
                pil = list(xkey["strata"]) if scope == "pooled" else [scope]
                out_c = {}
                for c in [*S8_CRITERIA, "any"]:
                    lf = (lambda r: any(r["label"].values())) if c == "any" else (lambda r, c=c: r["label"][c])
                    jf = (lambda r: r["verdict"] == "EXCLUDE") if c == "any" else (lambda r, c=c: r["judge"][c])
                    pop_fail = caught = jfail = jfail_true = 0.0
                    for p in pil:
                        s = xkey["strata"][p]
                        si = [r for r in inc if r["pilot"] == p]
                        se = [r for r in exc if r["pilot"] == p]
                        wi, we = s["judged_include"] / len(si), s["judged_exclude"] / len(se)
                        pop_fail += wi * sum(lf(r) for r in si) + we * sum(lf(r) for r in se)
                        caught += wi * sum(lf(r) and jf(r) for r in si) + we * sum(lf(r) and jf(r) for r in se)
                        jfail += wi * sum(jf(r) for r in si) + we * sum(jf(r) for r in se)
                        jfail_true += wi * sum(jf(r) and lf(r) for r in si) + we * sum(jf(r) and lf(r) for r in se)
                    out_c[c] = {"est_recall": _rate(caught, pop_fail), "est_precision": _rate(jfail_true, jfail),
                                "raw_packet2": {"labelled_fail": sum(lf(r) for r in exc if r["pilot"] in pil),
                                                "judge_fail": sum(jf(r) for r in exc if r["pilot"] in pil),
                                                "both": sum(lf(r) and jf(r) for r in exc if r["pilot"] in pil)}}
                est[scope] = out_c
            res["packet_2_all_s8_exclude"] = {"n": len(exc), "strata": xkey["strata"], "estimates": est}
        else:
            res["packet_2_all_s8_exclude"] = "made but not fully labelled yet"
    else:
        res["packet_2_all_s8_exclude"] = "not made yet (judge-recall-packet)"
    _recall_md(out / "judge_recall.md", inc, exc, res)
    write_json(out / "judge_recall.json", res)
    return res


def _recall_md(path: Path, inc: list[dict], exc: list[dict], res: dict) -> None:
    md = ["# S8 judge vs planning-chat labels (v1.2 pilots)", "",
          "Packet 1 (Q items) holds only S8-INCLUDE documents, so the judge passed every criterion on all of them: "
          "each labelled FAIL below is a judge miss. Packet 2 (X items), if labelled, adds S8-EXCLUDE documents; "
          "recall and precision are then estimated with each packet weighted by its share of the judged documents.", ""]
    p1 = res["packet_1_all_s8_include"]["per_criterion"]
    md += ["## Packet 1: labelled FAILs the judge passed", "", "| criterion | labelled FAIL | judge FAIL of those |",
           "|---|---|---|", *[f"| {c} | {v['labelled_fail']} | {v['judge_fail_of_those']} |" for c, v in p1.items()], ""]
    p2 = res.get("packet_2_all_s8_exclude")
    if isinstance(p2, dict):
        for scope, est in p2["estimates"].items():
            md += [f"## Estimated recall / precision: {scope}", "",
                   "| criterion | est. recall | est. precision | packet 2: labelled FAIL / judge FAIL / both |",
                   "|---|---|---|---|",
                   *[f"| {c} | {v['est_recall']} | {v['est_precision']} | {v['raw_packet2']['labelled_fail']} / "
                     f"{v['raw_packet2']['judge_fail']} / {v['raw_packet2']['both']} |" for c, v in est.items()], ""]
    md += ["## Per item: labeller FAIL vs the judge's own reason for that criterion", ""]
    for r in inc + exc:
        diff = [c for c in S8_CRITERIA if r["label"][c] != r["judge"][c]]
        if not diff:
            continue
        md += [f"### {r['id']} (pilot {r['pilot']}, S8 {r['verdict']})", f"Labeller note: {r['notes'] or '-'}", ""]
        for c in diff:
            who = "labeller FAIL, judge PASS" if r["label"][c] else "judge FAIL, labeller PASS"
            md += [f"- **{c}** ({who}). Judge: {str(r['rationale'].get(c, '')).strip()[:400]}"]
        md += [""]
    path.write_text("\n".join(md), encoding="utf-8")


# ----------------------------------------------------------------------------- draft vs final
STOP = set("a an the and or but of to in on at for with by from as is are was were be been being it its this that "
           "these those not no do does did so if then than there their they them he she his her you your we our "
           "i me my will would can could should may might must has have had".split())
WIDE_NARR_RX = re.compile(r"\b(?:Aster|[Tt]he (?:AI|assistant|model))(?:'s)?\s+(?:had(?:n'?t| not)|did(?:n'?t| not)|"
                          r"was(?:n'?t| not)|never|chose not to|refused to|would not|wouldn'?t)\b")
_PH = re.compile(r"\{[a-zA-Z_]+\}")


def _rule_grams(cfg: Config, ctx: mp.GenContext, n: int = 6) -> set:
    ident = ctx.identity
    texts = [ctx.anti_pattern_list, ctx.requirements_block]
    names = set((cfg.gen.get("templates") or {}).values())
    names.add((cfg.gen.get("s8") or {}).get("template", "E08_filter"))
    for nm in sorted(names):
        try:
            t = ctx.template(nm)
        except Exception:   # noqa: BLE001 - a missing optional template is not an error here
            continue
        texts.append(t.prefix + "\n" + t.suffix)
    filled = [_PH.sub(lambda m: str(ident.get(m.group(0)[1:-1], " zzph ")), x) for x in texts]
    # drop grams made of function words, and grams that also occur in the spec (spec copying is counted separately)
    spec = source_ngrams([ctx.spec_text], n)
    return {g for g in source_ngrams(filled, n)
            if "zzph" not in g and sum(w not in STOP for w in g) >= 3 and g not in spec}


def _tail(text: str) -> str:
    return " ".join([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()][-2:])


def _measure(text: str, leaks: list, spec_grams: set | None, rule_grams: set, assistant: str) -> dict[str, Any]:
    rs = copied_spans(text, rule_grams, 6)
    return {"leak": sum(len(rx.findall(text)) for rx in leaks),
            "spec_spans": len(copied_spans(text, spec_grams, 12)) if spec_grams else 0,
            "rule_spans": len(rs), "_rule_span_text": [s["text"] for s in rs],
            "narration_check": int(cleanup.lesson_narration(text, assistant) is not None),
            "narration_wide": len([s for s in re.split(r"(?<=[.!?])\s+", _tail(text)) if WIDE_NARR_RX.search(s)]),
            "principle": len(cleanup.PRINCIPLE_RX.findall(text))}


DETECTORS = ("leak", "spec_spans", "rule_spans", "narration_check", "narration_wide", "principle")


def draft_vs_final(cfg: Config, d: Path, spec_id: str = "aw_main", packet: Path | None = None) -> dict[str, Any]:
    ctx = mp.GenContext.create(cfg, spec_id, spec_path=d / "spec_used.md")
    pc = cfg.gen.get("prechecks") or {}
    lk = pc.get("leak_regex_file")
    leaks = [re.compile(r["pattern"]) for r in (yaml.safe_load((cfg.root / lk).read_text()) or [])] if lk else []
    rule = _rule_grams(cfg, ctx)
    assistant = (cfg.base.get("identity") or {}).get("assistant_name", "Aster")
    ideas = all_ideas(d)
    drafts, finals, crits, verdict = {}, {}, {}, {}
    for b in batch_dirs(d):
        drafts.update({r["idea_id"]: r["text"] for r in read_jsonl(b / "drafts.jsonl")})
        crits.update({r["idea_id"]: json.dumps({k: v for k, v in _strip(r).items() if k not in ("id", "doc_id", "idea_id")},
                                               ensure_ascii=False) for r in read_jsonl(b / "critiques.jsonl")})
        finals.update({r["idea_id"]: (r["text"], "kept") for r in read_jsonl(b / "rewrites.jsonl")})
        rj = b / "rewrites.rejected.jsonl"
        for r in read_jsonl(rj) if rj.exists() else []:
            t = (r.get("detail") or {}).get("text")
            src = str(r.get("_src", ""))
            iid = src[5:] if src.startswith("crit_") else None
            if t and iid and iid not in finals:
                finals[iid] = (t, f"rejected:{r['reason']}")
        jf = b / "judged.jsonl"
        for j in read_jsonl(jf) if jf.exists() else []:
            # unresolved S8 records (s8_unresolved: exclude) carry only doc_id = doc_<idea>_rewrite
            iid = j.get("idea_id") or re.sub(r"^doc_(.*)_rewrite$", r"\1", str(j.get("doc_id", "")))
            verdict[iid] = "unresolved" if j.get("unresolved") else j.get("verdict")
    lab_by_idea: dict[str, dict[str, Any]] = {}
    if packet:
        key = json.loads((packet / ".private" / "key.json").read_text())["items"]
        lab = _labels(packet / "audit_labels.csv") if (packet / "audit_labels.csv").exists() else {}
        for a, k in key.items():
            if Path(k["dir"]).resolve() == d.resolve() and a in lab:
                lab_by_idea[k["idea_id"]] = {"id": a, "fails": [c for c in S8_CRITERIA if lab[a][c].upper() == "FAIL"],
                                             "notes": lab[a].get("notes", "")}
    recs = []
    for iid in sorted(set(drafts) & set(finals)):
        idea = ideas.get(iid, {})
        sg = mp.copy_grams_for(ctx, idea) if idea else None
        dm = _measure(drafts[iid], leaks, sg, rule, assistant)
        text, status = finals[iid]
        fm = _measure(text, leaks, sg, rule, assistant)
        crit_grams = source_ngrams([crits.get(iid, "")], 6)
        new_rule = [s for s in fm["_rule_span_text"] if s not in dm["_rule_span_text"]]
        in_crit = sum(any(tuple(_words(s)[i:i + 6]) in crit_grams for i in range(len(_words(s)) - 5)) for s in new_rule)
        recs.append({"idea_id": iid, "final_status": status, "s8": verdict.get(iid),
                     "doc_type": idea.get("doc_type"), "label": lab_by_idea.get(iid),
                     "draft": {k: dm[k] for k in DETECTORS}, "final": {k: fm[k] for k in DETECTORS},
                     "new_rule_spans": new_rule[:5], "new_rule_spans_in_critique": in_crit})

    def summ(rs: list[dict]) -> dict[str, Any]:
        o: dict[str, Any] = {"n": len(rs)}
        for k in DETECTORS:
            dh = [r["draft"][k] > 0 for r in rs]
            fh = [r["final"][k] > 0 for r in rs]
            o[k] = {"docs_draft": sum(dh), "docs_final": sum(fh),
                    "introduced": sum(f and not x for x, f in zip(dh, fh)),
                    "removed": sum(x and not f for x, f in zip(dh, fh)),
                    "kept_both": sum(x and f for x, f in zip(dh, fh)),
                    "count_draft": sum(r["draft"][k] for r in rs), "count_final": sum(r["final"][k] for r in rs)}
        nr = sum(len(r["new_rule_spans"]) for r in rs)
        o["new_rule_spans_total"] = nr
        o["new_rule_spans_also_in_critique"] = sum(r["new_rule_spans_in_critique"] for r in rs)
        return o

    res = {"dir": str(d), "pairs": len(recs), "drafts_without_final": len(set(drafts) - set(finals)),
           "rule_grams": len(rule),
           "final_status": dict(Counter(r["final_status"] for r in recs)),
           "all_s7_outputs": summ(recs),
           "s7_kept": summ([r for r in recs if r["final_status"] == "kept"]),
           "s8_include": summ([r for r in recs if r["s8"] == "INCLUDE"])}
    if lab_by_idea:
        lr = [r for r in recs if r["label"]]
        res["labelled"] = {"all": summ(lr), "labelled_fail": summ([r for r in lr if r["label"]["fails"]]),
                           "items": [{"id": r["label"]["id"], "fails": r["label"]["fails"],
                                      **{f"{k}": f"{r['draft'][k]}->{r['final'][k]}" for k in DETECTORS},
                                      "new_rule_spans": r["new_rule_spans"][:3]} for r in sorted(lr, key=lambda r: r["label"]["id"])]}
    by_type = defaultdict(list)
    for r in recs:
        by_type[r["doc_type"]].append(r)
    res["introduced_by_doc_type_top"] = sorted(
        ((t, len(rs), sum(r["final"]["rule_spans"] > 0 and r["draft"]["rule_spans"] == 0 for r in rs))
         for t, rs in by_type.items()), key=lambda x: -x[2])[:10]
    from .io import write_jsonl
    write_jsonl(d / "draft_vs_final.jsonl", recs)
    write_json(d / "draft_vs_final.json", res)
    return res


__all__ = ["judge_recall", "judge_recall_packet", "draft_vs_final"]
