"""Post-labelling review of the pilot audit (Appendix I.1 step 4; run only after labelling is complete).

* Label sheet parsing with column aliases from ``labelling_guide.md`` and strict value checks
  (a blank or misspelt criterion is an error, never a silent FAIL).
* Annotator per row (from ``audit/annotators.json``, written from ``P001-P025:Niraj,...`` ranges),
  per-annotator results, A/B pairs whose two documents were labelled by different annotators, and a
  same-annotator sensitivity check of the paired quality difference (descriptive only; the registered
  rule is unchanged).
* Revisions made before unblinding (``audit_labels_original.csv`` vs ``audit_labels.csv``).
* Near-duplicate check between audit items, and per-item failure listings with simple pattern tags.
"""
from __future__ import annotations

import csv
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .gen.dedup import ngram_seen_share, ngrams, tokens
from .gen.schemas import S8_CRITERIA

ALIASES = {"reason_correct": "reason_visible_correct_type", "quality": "overall_quality"}
VALID = {"PASS", "FAIL"}


class AuditError(RuntimeError):
    pass


def read_labels(path: Path) -> list[dict[str, Any]]:
    """Rows with canonical column names; raises on unknown/blank/invalid values in labelled rows."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out, problems = [], []
    for r in rows:
        r = {ALIASES.get((k or "").strip(), (k or "").strip()): (v or "").strip() for k, v in r.items()}
        if not r.get("audit_id"):
            continue
        missing = [c for c in (*S8_CRITERIA, "overall_quality") if c not in r]
        if missing:
            raise AuditError(f"{path.name}: column(s) {missing} not found (have {sorted(r)})")
        if r["overall_quality"]:
            bad = [c for c in S8_CRITERIA if r[c].upper() not in VALID]
            try:
                q = float(r["overall_quality"])
                if not 1 <= q <= 5:
                    bad.append("overall_quality")
            except ValueError:
                bad.append("overall_quality")
            if bad:
                problems.append(f"{r['audit_id']}: {bad}")
        out.append(r)
    if problems:
        raise AuditError(f"{path.name}: invalid or blank values (must be PASS/FAIL and quality 1-5): {problems}")
    return out


def parse_ranges(spec: str) -> dict[str, str]:
    """'P001-P025:Niraj,P026-P050:Jevin' -> {'P001': 'Niraj', ...}."""
    out: dict[str, str] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        rng, _, name = part.partition(":")
        a, _, b = rng.partition("-")
        lo, hi = int(a.strip().lstrip("Pp")), int((b or a).strip().lstrip("Pp"))
        if not name.strip() or lo > hi:
            raise AuditError(f"bad annotator range {part!r}")
        for i in range(lo, hi + 1):
            aid = f"P{i:03d}"
            if aid in out:
                raise AuditError(f"{aid} assigned to two annotators")
            out[aid] = name.strip()
    return out


def revisions(original: Path, current: Path) -> list[dict[str, Any]]:
    if not original.exists():
        return []
    o = {r["audit_id"]: r for r in read_labels(original)}
    out = []
    for r in read_labels(current):
        p = o.get(r["audit_id"])
        if p is None:
            continue
        ch = {c: (p[c], r[c]) for c in (*S8_CRITERIA, "overall_quality") if p.get(c, "").upper() != r.get(c, "").upper()}
        if ch:
            out.append({"audit_id": r["audit_id"], "changes": ch, "notes": r.get("notes", "")})
    return out


def _paired_diff(items_a: dict[str, float], items_b: dict[str, float], ideas: list[str]) -> dict[str, Any] | None:
    d = [items_a[i] - items_b[i] for i in ideas]
    return {"n": len(d), "mean_diff": sum(d) / len(d)} if d else None


def annotator_analysis(rows: list[dict[str, Any]], key: dict[str, Any], annot: dict[str, str]) -> dict[str, Any]:
    """Per-annotator results, cross-annotator pairs and a same-annotator sensitivity check."""
    by_idea: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    per_ann: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        k = key[r["audit_id"]]
        fail = any(r[c].upper() == "FAIL" for c in S8_CRITERIA)
        rec = {"audit_id": r["audit_id"], "annotator": annot.get(r["audit_id"], "?"), "fail": fail,
               "quality": float(r["overall_quality"])}
        by_idea[k["idea_id"]][k["candidate"]] = rec
        per_ann[(rec["annotator"], k["candidate"])].append(rec)
    cands = sorted({k["candidate"] for k in key.values()})
    cross, same = [], []
    for iid, d in sorted(by_idea.items()):
        if len(d) < 2:
            continue
        anns = {c: d[c]["annotator"] for c in cands if c in d}
        entry = {"idea_id": iid, **{f"{c}_item": d[c]["audit_id"] for c in cands if c in d},
                 **{f"{c}_annotator": a for c, a in anns.items()}}
        (cross if len(set(anns.values())) > 1 else same).append(entry)
    out: dict[str, Any] = {
        "annotators": sorted(set(annot.values())),
        "per_annotator": {f"{a}/{c}": {"n": len(v), "human_fail_rate": sum(x["fail"] for x in v) / len(v),
                                       "mean_quality": sum(x["quality"] for x in v) / len(v)}
                          for (a, c), v in sorted(per_ann.items())},
        "pairs_total": len(cross) + len(same), "pairs_cross_annotator": cross, "pairs_same_annotator": len(same),
        "overlap_items": 0,   # no item was labelled twice -> no human-human agreement
    }
    if len(cands) == 2:
        a, b = cands
        qa = {i: d[a]["quality"] for i, d in by_idea.items() if a in d}
        qb = {i: d[b]["quality"] for i, d in by_idea.items() if b in d}
        same_ids = [e["idea_id"] for e in same]
        out["sensitivity_same_annotator_pairs"] = {f"{a}-{b}": _paired_diff(qa, qb, same_ids)}
    return out


ANIMAL_RE = re.compile(
    r"\b(animal|animals|welfare|sentien\w*|suffer\w*|pain|species|fish|chicken|hen|hens|pig|pigs|cow|cows|cattle|"
    r"shrimp|crab|crabs|lobster|octop\w*|insect\w*|bee|bees|mouse|mice|rat|rats|bird|birds|dog|dogs|cat|cats|"
    r"livestock|wildlife|creature\w*|invertebrate\w*|mammal\w*|cephalopod\w*|crustacean\w*)\b", re.I)

PATTERNS = {
    "aster_human_or_physical": re.compile(r"human|physical|body|robot|vet\b|person|hands?|walk|picks? up|embod", re.I),
    "over_certainty": re.compile(r"over.?certain|certaint|overclaim|too strong|definitely", re.I),
    "template_echo": re.compile(r"template|stock phrase|boilerplate|generic phrase|echo", re.I),
    "few_animals": re.compile(r"barely|hardly|no animal|few animal|not about animal|animal.{0,20}(missing|absent)", re.I),
}


def item_table(rows: list[dict[str, Any]], key: dict[str, Any], annot: dict[str, str],
               docs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        k = key[r["audit_id"]]
        text = (docs.get((k["candidate"], k["doc_id"])) or docs.get(k["doc_id"]) or {}).get("text", "")
        words = len(text.split())
        n_animal = len(ANIMAL_RE.findall(text))
        fails = [c for c in S8_CRITERIA if r[c].upper() == "FAIL"]
        notes = r.get("notes", "")
        out.append({"audit_id": r["audit_id"], "candidate": k["candidate"], "idea_id": k["idea_id"],
                    "annotator": annot.get(r["audit_id"], "?"), "quality": float(r["overall_quality"]),
                    "fails": fails, "notes": notes, "words": words,
                    "animal_terms": n_animal, "animal_terms_per_100w": round(100 * n_animal / words, 2) if words else 0,
                    "pattern_tags": [p for p, rx in PATTERNS.items() if rx.search(notes)]})
    return out


def near_duplicate(a: str, b: str) -> dict[str, Any]:
    """Similarity between two texts with the S9 measures (13-gram share) plus simple diagnostics."""
    wa, wb = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    longest = sm.find_longest_match(0, len(wa), 0, len(wb))

    def grams(w: list[str], n: int) -> set[tuple[str, ...]]:
        w = [x.lower() for x in w]
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    g5a, g5b = grams(wa, 5), grams(wb, 5)
    return {
        "words": [len(wa), len(wb)],
        # the S9 measure: share of one text's 13-grams already present in the other (S9 threshold 0.30)
        "share_13grams_of_b_in_a": round(ngram_seen_share(tokens(b), ngrams(tokens(a), 13), 13), 3),
        "share_13grams_of_a_in_b": round(ngram_seen_share(tokens(a), ngrams(tokens(b), 13), 13), 3),
        "jaccard_5grams": len(g5a & g5b) / max(1, len(g5a | g5b)),
        "word_sequence_ratio": round(sm.ratio(), 3),
        "longest_common_run_words": longest.size,
        "longest_common_run": " ".join(wa[longest.a: longest.a + min(longest.size, 40)]),
    }


def counts_by_candidate(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in sorted({i["candidate"] for i in items}):
        its = [i for i in items if i["candidate"] == c]
        out[c] = {"criterion_fails": dict(Counter(f for i in its for f in i["fails"])),
                  "pattern_tags": dict(Counter(t for i in its for t in i["pattern_tags"])),
                  "fail_docs": sum(bool(i["fails"]) for i in its), "n": len(its)}
    return out


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)
