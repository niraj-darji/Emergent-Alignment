"""Midtraining-corpus pipeline, stages S1–S8 (§6.2–6.3). S9 lives in ``dedup.py``; S10–S12 come later.

Each stage is a function usable both by the generator pilot (Appendix I.1) and by the full
pipeline. Stages run through ``io.run_stage`` (resumable, idempotent) and every LLM call goes
through ``LLMClient`` (explicit sampling, cache, request log). Prompts come from ``prompts/``
(shared prefix first, §6.1.1 A.2). When ``bench_dir`` is set, the rendered S5–S8 prompts are also
saved as ``{prefix, suffix}`` records for the official throughput benchmark (§6.1.1 E.1).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import random
import re
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, create_model

from .. import prompts
from ..config import Config
from ..io import Rejected, append_jsonl, read_jsonl
from ..llm import LLMClient, ThinkingLeakError, leading_plan, response_format_for, scratchpad_closed_with_think
from . import cleanup
from ..textchecks import (degenerate_repetition, first_failure, language_mixing, length, scripted_dialogue,
                          source_ngrams, spec_copy, truncated)
from .schemas import (BEHAVIOUR_TYPES, S8_CRITERIA, Critique, CritiqueV11, DocTypes, Domains, FilterVerdict, Idea,
                      IdeaV12, SeedIdea, Subdomains)

log = logging.getLogger(__name__)

CONTENT_RE = re.compile(r"<content>(.*?)</content>", re.S | re.I)
FORBIDDEN_IN_CONTENT = re.compile(r"</?think>|<\|channel>|<channel\|>|scratchpad", re.I)


# ----------------------------------------------------------------------------- context
@dataclass
class GenContext:
    cfg: Config
    spec_id: str
    spec_text: str                       # identity already filled in
    identity: dict[str, str]
    bench_dir: Path | None = None
    _bench_seen: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def create(cls, cfg: Config, spec_id: str, bench_dir: Path | None = None,
               spec_path: Path | None = None) -> "GenContext":
        ident = cfg.base.get("identity") or {}
        identity = {"assistant_name": ident["assistant_name"], "developer_name": ident["developer_name"]}
        raw = Path(spec_path or cfg.root / "specs" / f"{spec_id}.md").read_text(encoding="utf-8")
        return cls(cfg, spec_id, fill_spec(raw, identity), identity, bench_dir)

    # shared template fragments ------------------------------------------------
    @property
    def anti_pattern_list(self) -> str:
        items = yaml.safe_load((self.cfg.root / self.cfg.gen.get("anti_patterns_file", "lists/anti_patterns.yaml")).read_text())
        return "; ".join(f"{i['name']} ({i['description']})" if i.get("description") else i["name"] for i in items)

    @property
    def requirements_block(self) -> str:
        return prompts.fill(prompts.load_text(self.template_name("s5_requirements"), str(self.cfg.root / "prompts")),
                            {**self.identity, **self.length_words})

    @property
    def length_words(self) -> dict[str, str]:
        """Doc v1.7.4: the length target is stated to the model in words (E.5 / E.7)."""
        lo, hi = self.cfg.gen["documents"]["target_words"]
        return {"min_words": f"{int(lo):,}", "max_words": f"{int(hi):,}"}

    @property
    def reason_type(self) -> str:
        return self.cfg.gen["documents"]["reason_type"][self.spec_id]

    DEFAULT_TEMPLATES = {"s4": "E04_ideas", "s5": "E05_draft", "s5_requirements": "E05_requirements_block",
                         "s6": "E06_critique", "s7": "E07_rewrite",
                         "s4b_tag": "E04s_tag_v12", "s4b_idea": "E04s_idea_v12", "s5_seed": "E05s_rewrite_v12"}

    def template_name(self, stage: str) -> str:
        """Template per stage from ``gen.templates`` (v1.1, 25 Oct); defaults are the v1 / pilot templates."""
        return (self.cfg.gen.get("templates") or {}).get(stage, self.DEFAULT_TEMPLATES[stage])

    @property
    def copy_grams(self) -> set:
        """12-grams of the spec and the requirements block, for the v1.1 spec-copy pre-check."""
        if not hasattr(self, "_copy_grams"):
            n = int(((self.cfg.gen.get("prechecks") or {}).get("spec_copy") or {}).get("n", 12))
            extra = list(((self.cfg.gen.get("prechecks") or {}).get("spec_copy") or {}).get("extra_sources") or [])
            self._copy_grams = source_ngrams([self.spec_text, self.requirements_block, *extra], n)
        return self._copy_grams

    @property
    def spec_dates(self) -> set:
        return cleanup.spec_dates(self.spec_text)

    def template(self, name: str) -> prompts.Template:
        return prompts.load(name, str(self.cfg.root / "prompts"))

    def capture(self, stage: str, rec_id: str, rendered: prompts.Rendered, section: str,
                response_format: dict[str, Any] | None = None) -> None:
        """Save a real prompt for the official benchmark (once per id)."""
        if self.bench_dir is None:
            return
        path = self.bench_dir / f"{stage}.jsonl"
        seen = self._bench_seen.setdefault(stage, {r["id"] for r in read_jsonl(path)})
        if rec_id in seen:
            return
        seen.add(rec_id)
        append_jsonl(path, {"id": rec_id, "prefix": rendered.prefix, "suffix": rendered.suffix,
                            "section": section, "response_format": response_format})


# ----------------------------------------------------------------------------- helpers
SPEC_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def fill_spec(raw: str, identity: dict[str, str]) -> str:
    """Spec text with identity filled in. ``{model_name}`` is an alias of ``{assistant_name}`` (aw_main v1,
    19 Oct). Any other unfilled ``{placeholder}`` is an error, so a typo can't reach the generator."""
    raw = raw.lstrip("\ufeff").replace("\r\n", "\n")          # Windows copies: byte-order mark, CRLF
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S).strip()
    values = {**identity, "model_name": identity["assistant_name"]}
    left = sorted(set(SPEC_PLACEHOLDER_RE.findall(raw)) - values.keys())
    if left:
        raise ValueError(f"Spec has unknown placeholder(s) {left}; known: {sorted(values)}")
    return prompts.fill(raw, values)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_#`>]", "", s)).strip().lower()


def excerpt_coverage(spec_text: str, excerpts: list[str]) -> dict[str, Any]:
    """S1 asks for EXACT spec passages. Share of spec sentences that appear in some excerpt, and the
    share of excerpt sentences that are not found in the spec (paraphrase / invention)."""
    split = lambda t: [x for x in (_norm(s) for s in re.split(r"(?<=[.!?])\s+|\n+", t)) if len(x) > 25]  # noqa: E731
    spec_sents = split(spec_text)
    joined = _norm(" ".join(excerpts))
    spec_norm = _norm(spec_text)
    ex_sents = [s for e in excerpts for s in split(e)]
    return {
        "spec_sentences_covered": sum(s in joined for s in spec_sents) / max(1, len(spec_sents)),
        "excerpt_sentences_not_in_spec": sum(s not in spec_norm for s in ex_sents) / max(1, len(ex_sents)),
    }


def template_sha(tpl: prompts.Template, requirements_block: str = "") -> str:
    """Short hash of a template's text (+ requirements block): drafts from different wordings never mix."""
    return hashlib.sha256((tpl.prefix + "\x00" + tpl.suffix + "\x00" + requirements_block).encode()).hexdigest()[:12]


def bullet(items: list[str]) -> str:
    return "\n" + "\n".join(f"- {a}" for a in items)


def idea_json(idea: dict[str, Any]) -> str:
    keys = ("doc_type", "role", "what", "how", "why", "behaviour_type", "species_group", "context_domain",
            "cost_of_care", "welfare_raised", "perspective", "is_story")
    return json.dumps({k: idea[k] for k in keys if k in idea}, ensure_ascii=False)


OPEN_RE = re.compile(r"<content>", re.I)
CLOSE_RE = re.compile(r"</content>", re.I)
# sentence-final punctuation, optionally followed by closing quotes/brackets (doc v1.7.2 §6.3 S5)
SENTENCE_END_RE = re.compile(r"[.!?…][\"'”’»)\]]*\Z")


@dataclass
class Parsed:
    doc: str | None
    reason: str | None = None            # rejection reason when doc is None
    tolerances: list[str] = field(default_factory=list)


def parse_content(text: str, finish_reason: str | None) -> Parsed:
    """Doc v1.7.2 §6.3 S5 output parsing (S5 drafts and S7 rewrites, every generator).

    Accepted: one complete ``<content>…</content>`` block, or ONE unclosed ``<content>`` when the model
    stopped on its own and the text ends on sentence-final punctuation (optionally followed by a closing
    quote or bracket) -> tolerance ``unclosed_content``. Rejected: two or more ``<content>`` tags, output
    cut off at the token limit, anything else without a usable block."""
    n_open = len(OPEN_RE.findall(text))
    if n_open >= 2:
        return Parsed(None, "multiple_content_tags")
    m = CONTENT_RE.search(text)
    if m and m.group(1).strip():
        return Parsed(m.group(1).strip())
    if finish_reason == "length":
        return Parsed(None, "truncated_max_tokens")
    if n_open == 1 and not CLOSE_RE.search(text) and finish_reason == "stop":
        doc = text[OPEN_RE.search(text).end():].strip()
        if not doc:
            return Parsed(None, "missing_content_tags")
        if not SENTENCE_END_RE.search(doc):
            return Parsed(None, "unclosed_content_unterminated")
        return Parsed(doc, tolerances=["unclosed_content"])
    return Parsed(None, "missing_content_tags")


def extract_content(text: str, finish_reason: str | None) -> tuple[str | None, str | None]:
    """Backward-compatible (doc, rejection reason) view of ``parse_content``."""
    p = parse_content(text, finish_reason)
    return p.doc, p.reason


@dataclass
class DocResult:
    doc: str | None
    bad: Rejected | None
    tolerances: list[str]
    words: int | None = None      # words of the parsed document (None if nothing could be parsed)


class DocChecks:
    """§6.1.1 A.4 cheap checks for S5/S7 documents, in order: parsing (doc v1.7.2), truncation, tags
    inside the document, language mixing, length, repetition loops, E.5 regex rules. First failure wins."""

    def __init__(self, cfg: Config, stage: str = "s5_draft"):
        lm = cfg.precheck_cfg("language_mixing")
        self.max_non_latin = float(lm.get("max_non_latin_share", 0.01))
        pc = cfg.gen.get("prechecks") or {}
        self.min_words, self.max_words = pc.get("document_words", [None, None])
        if stage == "s7_rewrite" and pc.get("s7_words"):   # doc v1.7.4 production length pre-check
            self.min_words, self.max_words = pc["s7_words"]
        rp = pc.get("degenerate_repetition") or {}
        self.rep = {"max_line_repeats": int(rp.get("max_line_repeats", 5)),
                    "max_ngram_repeats": int(rp.get("max_ngram_repeats", 3)), "ngram": int(rp.get("ngram", 10))}
        rx_file = pc.get("regex_file")
        rules = yaml.safe_load((cfg.root / rx_file).read_text()) if rx_file else []
        self.regexes = [(r["reason"], re.compile(r["pattern"])) for r in rules or []]
        lk = pc.get("leak_regex_file")            # v1.1 (25 Oct): our own instruction wording leaking into the text
        self.leaks = ([re.compile(r["pattern"]) for r in (yaml.safe_load((cfg.root / lk).read_text()) or [])]
                      if lk and stage in (pc.get("leak_apply_to") or ["s7_rewrite"]) else [])   # final text only: S6/S7 can fix drafts
        sc = pc.get("spec_copy") or {}
        self.spec_copy = sc if stage in (sc.get("apply_to") or ["s7_rewrite"]) and sc else None
        # v1.2 (26 Oct) checks on the final text (S7 only): placeholders, banned names, lesson narration, seed copy
        final = stage == "s7_rewrite"
        self.v12_placeholder = final and bool(pc.get("placeholder_v12"))
        nf = cfg.gen.get("names_file")
        names = yaml.safe_load((cfg.root / nf).read_text(encoding="utf-8")) if nf else {}
        self.banned = set((names or {}).get("banned") or []) if final and pc.get("banned_names") else set()
        self.known_given = set((names or {}).get("given") or [])
        self.lesson = final and bool(pc.get("lesson_narration"))
        self.seed_copy_cfg = (pc.get("seed_copy") or None) if final else None
        self.assistant = (cfg.base.get("identity") or {}).get("assistant_name", "Aster")

    def check(self, raw: str, finish_reason: str | None, copy_grams: set | None = None,
              seed_text: str | None = None) -> DocResult:
        p = parse_content(raw, finish_reason)
        if p.doc is None:
            return DocResult(None, Rejected(p.reason or "missing_content_tags", {"tolerances": p.tolerances}), p.tolerances)
        doc = p.doc
        words = len(doc.split())
        bad = first_failure(
            truncated(finish_reason),
            # no thinking tag or scratchpad may survive inside the document (doc v1.7.1/v1.7.2 S5)
            Rejected("tag_in_content", {"match": m.group(0)}) if (m := FORBIDDEN_IN_CONTENT.search(doc)) else None,
            language_mixing(doc, self.max_non_latin),
            length(doc, self.min_words, self.max_words),
            degenerate_repetition(doc, **self.rep),
            *(Rejected(reason, {"match": m.group(0)[:80]}) if (m := rx.search(doc)) else None
              for reason, rx in self.regexes),
            *(Rejected("planning_leak", {"match": doc[max(0, m.start() - 60):m.end() + 60], "text": doc})
              if (m := rx.search(doc)) else None for rx in self.leaks),
            (spec_copy(doc, copy_grams, int(self.spec_copy.get("n", 12)), int(self.spec_copy.get("max_spans", 1)),
                       int(self.spec_copy.get("max_span_words", 19))) if self.spec_copy and copy_grams else None),
            cleanup.placeholder(doc) if self.v12_placeholder else None,
            cleanup.banned_name(doc, self.banned) if self.banned else None,
            cleanup.lesson_narration(doc, self.assistant) if self.lesson else None,
            (cleanup.seed_copy(doc, seed_text, int(self.seed_copy_cfg.get("n", 8)), float(self.seed_copy_cfg.get("max_share", 0.2)))
             if self.seed_copy_cfg and seed_text else None),
        )
        if bad and bad.reason in ("spec_copy", "placeholder", "banned_name", "lesson_narration", "seed_copy"):
            bad.detail = {**(bad.detail or {}), "text": doc}
        if bad:
            bad.detail = {**(bad.detail or {}), "tolerances": p.tolerances}
            return DocResult(None, bad, p.tolerances, words)
        return DocResult(doc, None, p.tolerances, words)

    def __call__(self, raw: str, finish_reason: str | None) -> tuple[str | None, Rejected | None]:
        r = self.check(raw, finish_reason)
        return r.doc, r.bad


# ----------------------------------------------------------------------------- S1–S4
async def s1_decompose(ctx: GenContext, client: LLMClient) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = ctx.template("E01_decompose").render(**ctx.identity, spec_full=ctx.spec_text)
    out = (await client.complete_json(r.text, stage="s1_decompose", schema=Domains))[0]
    domains = [{"id": f"dom_{i:03d}", "spec_id": ctx.spec_id, "domain": d.domain, "spec_excerpt": d.spec_excerpt}
               for i, d in enumerate(out.domains)]
    return domains, excerpt_coverage(ctx.spec_text, [d["spec_excerpt"] for d in domains])


def s2_fn(ctx: GenContext, client: LLMClient):
    async def fn(dom: dict[str, Any]) -> list[dict[str, Any]]:
        r = ctx.template("E02_subdomains").render(**ctx.identity, spec_excerpt=dom["spec_excerpt"])
        out = (await client.complete_json(r.text, stage="s2_subdomains", schema=Subdomains))[0]
        return [{"id": f"sub_{dom['id'][4:]}_{j:02d}", "domain_id": dom["id"], "subdomain": s.subdomain,
                 "spec_excerpt": dom["spec_excerpt"], "assertions": s.assertions}
                for j, s in enumerate(out.subdomains)]
    return fn


def s3_fn(ctx: GenContext, client: LLMClient):
    async def fn(sub: dict[str, Any]) -> dict[str, Any]:
        r = ctx.template("E03_doctypes").render(**ctx.identity, subdomain=sub["subdomain"],
                                                assertions=bullet(sub["assertions"]))
        out = (await client.complete_json(r.text, stage="s3_doctypes", schema=DocTypes))[0]
        return {"id": sub["id"], "doc_types": out.doc_types}
    return fn


PUSHBACK_LINES = {
    "holds": "If someone pushes back on {a} in this document, they do not come round by the end: they make their "
             "own decision, and {a} respects it.",
    "may_update": "If someone pushes back on {a}, they may change their mind, but only because of a concrete new fact "
                  "or consideration shown plainly; no narrated change of heart.",
}


def _seed_u(*parts: str) -> random.Random:
    return random.Random(int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:12], 16))


def _ordinal(cfg: Config, idea_id: str) -> int | None:
    """Dense position of an idea in the run (batch, call, index) for balanced surname use; None if unknown."""
    m = re.match(r"idea_b(\d+)_(\d+)_(?:r\d+_)?(\d+)$", idea_id) or re.match(r"seed_(\d+)_(\d+)$", idea_id)
    if not m:
        return None
    nums = [int(x) for x in m.groups()]
    if len(nums) == 2:                       # Pilot B: seed index, rewrite index
        return nums[0] * 2 + nums[1]
    b, c, j = nums
    per_call = int((cfg.gen.get("s4") or {}).get("k", 6)) + 2
    return (b * int((cfg.gen.get("corpus") or {}).get("calls_per_batch", 100)) + c) * per_call + j


def assign_idea_extras(cfg: Config, spec_id: str, idea_id: str) -> dict[str, Any]:
    """Names, pushback outcome and (v1.2) a date, fixed at planning time and seeded by the idea id.
    v1.2: surnames are taken in turn from a seeded permutation of the list, so each is used about equally often."""
    out: dict[str, Any] = {}
    nf = cfg.gen.get("names_file")
    k = int(cfg.gen.get("names_per_idea", 3))
    if nf:
        lst = _names_list(str(cfg.root / nf))
        rng = _seed_u("names", spec_id, idea_id)
        ordn = _ordinal(cfg, idea_id) if cfg.gen.get("balanced_surnames") else None
        perm = _perm(len(lst["surnames"]), spec_id) if ordn is not None else None
        names: list[str] = []
        t = 0
        while len(names) < k:
            sur = lst["surnames"][perm[(ordn * k + t) % len(perm)]] if perm else rng.choice(lst["surnames"])
            n = f"{rng.choice(lst['given'])} {sur}"
            t += 1
            if n not in names:
                names.append(n)
        out["names"] = names
    share = cfg.gen.get("pushback_update_share")
    if share is not None:
        out["pushback_outcome"] = "may_update" if _seed_u("pushback", spec_id, idea_id).random() < float(share) else "holds"
    dr = cfg.gen.get("date_range")              # v1.2: one date per idea, uniform over the range
    if dr:
        lo, hi = (dt.date.fromisoformat(x) for x in dr)
        out["date"] = (lo + dt.timedelta(days=_seed_u("date", spec_id, idea_id).randrange((hi - lo).days + 1))).isoformat()
    return out


@lru_cache(maxsize=8)
def _perm(n: int, spec_id: str) -> list[int]:
    p = list(range(n))
    random.Random(f"surnames:{spec_id}").shuffle(p)
    return p


@lru_cache(maxsize=4)
def _names_list(path: str) -> dict[str, list[str]]:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    bad = set(d.get("banned") or [])
    return {"given": [g for g in d["given"] if g not in bad], "surnames": [x for x in d["surnames"] if x not in bad]}


def idea_extras_raw(ctx: GenContext, idea: dict[str, Any]) -> dict[str, Any]:
    return {**assign_idea_extras(ctx.cfg, ctx.spec_id, idea["id"]),
            **{k: idea[k] for k in ("names", "pushback_outcome", "date") if k in idea}}


def idea_extras(ctx: GenContext, idea: dict[str, Any]) -> dict[str, str]:
    """Template values for the v1.1 S5/S7 suffix (ignored by v1 templates, which don't use them)."""
    ex = {**assign_idea_extras(ctx.cfg, ctx.spec_id, idea["id"]),
          **{k: idea[k] for k in ("names", "pushback_outcome", "date") if k in idea}}
    a = ctx.identity["assistant_name"]
    return {"names": ", ".join(ex.get("names") or []) or "any realistic names",
            "pushback_line": PUSHBACK_LINES[ex.get("pushback_outcome", "holds")].format(a=a),
            "date": _human_date(ex.get("date")) or "a realistic date"}


def _human_date(iso: str | None) -> str | None:
    if not iso:
        return None
    d = dt.date.fromisoformat(iso)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def copy_grams_for(ctx: GenContext, idea: dict[str, Any]) -> set | None:
    """Spec + requirements + this idea's fields (what/how/why/assertions), or None if the check is off."""
    sc = (ctx.cfg.gen.get("prechecks") or {}).get("spec_copy")
    if not sc:
        return None
    n = int(sc.get("n", 12))
    return ctx.copy_grams | source_ngrams([idea.get("what", ""), idea.get("how", ""), idea.get("why", ""),
                                           " ".join(idea.get("assertions") or [])], n)


def s4_fn(ctx: GenContext, client: LLMClient, subs: dict[str, dict[str, Any]], k: int, need: int | None = None):
    """S4: ``k`` ideas per call. Ideas containing language mixing or the generator's own reasoning
    (``s4.meta_commentary_regex``) are rejected and logged; if fewer than ``need`` clean ideas remain,
    the call is re-requested with a new seed up to ``s4.max_resample`` times."""
    tpl4 = ctx.template(ctx.template_name("s4"))
    v12 = "behaviour_type" in tpl4.prefix + tpl4.suffix
    ideas_k = create_model("Ideas", ideas=(list[IdeaV12 if v12 else Idea], Field(min_length=k, max_length=k)))
    max_non_latin = DocChecks(ctx.cfg).max_non_latin
    s4cfg = ctx.cfg.gen["s4"]
    meta_rx = re.compile(s4cfg["meta_commentary_regex"]) if s4cfg.get("meta_commentary_regex") else None
    max_resample = int(s4cfg.get("max_resample", 0))
    dlg_min = s4cfg.get("scripted_dialogue_min_words")   # 13 Oct: ideas must not script the document's lines
    need = k if need is None else need

    async def fn(call: dict[str, Any]) -> list[dict[str, Any]]:
        sub = subs[call["subdomain_id"]]
        r = tpl4.render(
            **ctx.identity, k=k, subdomain=sub["subdomain"], assertions=bullet(sub["assertions"]),
            behaviour_types=", ".join(BEHAVIOUR_TYPES),
            doc_type=call["doc_type"], species_group=call["species_group"],
            context_domain=call["context_domain"], cost_of_care=call["cost_of_care"],
            welfare_raised=call["welfare_raised"], perspective=call["perspective"],
            is_story=str(call["is_story"]).lower())
        recs: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for attempt in range(max_resample + 1):
            # explicit per-call (and per-attempt) seed: identical prompts still yield different ideas
            seed = int(hashlib.sha256(f"s4:{ctx.spec_id}:{call['call_id']}:{attempt}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
            if attempt == 0:   # keep the original seed string so earlier cached calls stay valid
                seed = int(hashlib.sha256(f"s4:{ctx.spec_id}:{call['call_id']}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
            out = (await client.complete_json(r.text, stage="s4_ideas", schema=ideas_k, seed=seed))[0]
            for j, idea in enumerate(out.ideas):
                text = f"{idea.what} {idea.how} {idea.why}"
                reason = None
                if language_mixing(text, max_non_latin) is not None:
                    reason = "language_mixing"
                elif meta_rx and (mm := meta_rx.search(text)):
                    reason = f"meta_commentary: {mm.group(0)!r}"
                elif dlg_min and (q := scripted_dialogue(text, int(dlg_min))):
                    reason = f"scripted_dialogue: {q[:60]!r}"
                if reason:
                    log.warning("S4 %s attempt %d idea %d rejected: %s", call["call_id"], attempt, j, reason)
                    rejected.append({"attempt": attempt, "index": j, "reason": reason})
                    continue
                recs.append({
                    "id": f"idea_{call['call_id'][5:]}_{j}" if attempt == 0 else f"idea_{call['call_id'][5:]}_r{attempt}_{j}",
                    "call_id": call["call_id"], "s4_attempt": attempt,
                    "subdomain_id": sub["id"], "domain_id": sub["domain_id"], "subdomain": sub["subdomain"],
                    "spec_excerpt": sub["spec_excerpt"], "assertions": sub["assertions"],
                    **{f: call[f] for f in ("doc_type", "species_group", "context_domain", "cost_of_care",
                                            "welfare_raised", "perspective", "is_story")},
                    "what": idea.what, "how": idea.how, "why": idea.why,
                    **({"behaviour_type": idea.behaviour_type} if v12 else {}),
                })
                recs[-1].update(assign_idea_extras(ctx.cfg, ctx.spec_id, recs[-1]["id"]))
            if len(recs) >= need:
                break
        for rec in recs:
            rec["s4_rejected_in_call"] = rejected
        return recs
    return fn


# ----------------------------------------------------------------------------- S5–S8
def s5_fn(ctx: GenContext, client: LLMClient, checks: DocChecks):
    tpl = ctx.template(ctx.template_name("s5"))
    req = ctx.requirements_block
    tpl_sha = template_sha(tpl, req)
    tpl_seed = None

    async def fn(idea: dict[str, Any]) -> dict[str, Any] | Rejected:
        nonlocal tpl_seed
        if "source" in idea:          # Pilot B: rewrite a seed document (E05s, 26 Oct)
            tpl_seed = tpl_seed or ctx.template(ctx.template_name("s5_seed"))
            r = tpl_seed.render(**ctx.identity, requirements_block=req, spec_excerpt=idea["spec_excerpt"],
                                doc_type=idea["doc_type"], assertions=bullet(idea["assertions"]),
                                how=idea["how"], why=idea["why"], role=idea.get("role", ""),
                                source=idea["source"], **idea_extras(ctx, idea))
        else:
            r = tpl.render(**ctx.identity, requirements_block=req, spec_excerpt=idea["spec_excerpt"],
                           doc_type=idea["doc_type"], assertions=bullet(idea["assertions"]),
                           what=idea["what"], how=idea["how"], why=idea["why"], **idea_extras(ctx, idea))
        ctx.capture("s5_draft", idea["id"], r, idea["domain_id"])
        try:
            res = await client.complete_full(r.text, stage="s5_draft")
        except ThinkingLeakError:   # doc v1.7.1 S5: a real thinking leak is a rejection (same seed would repeat it)
            return Rejected("thinking_leak", {"template_sha": tpl_sha, "tolerances": []})
        slip = scratchpad_closed_with_think(res.texts[0])   # deviation 8: tolerated, reported per candidate
        plan = leading_plan(res.texts[0])
        diag = {"scratchpad_closed_with_think": slip, "template_sha": tpl_sha,
                "plan_words": None if plan is None else len(plan.split()),   # None: no closed leading plan
                "n_tokens": res.usage.get("completion_tokens"), "finish_reason": res.finish_reasons[0]}
        cr = checks.check(res.texts[0], res.finish_reasons[0])
        diag["tolerances"] = (["scratchpad_closed_with_think"] if slip else []) + cr.tolerances
        diag["doc_words"] = cr.words
        if cr.bad:
            cr.bad.detail = {**(cr.bad.detail or {}), **diag}
            return cr.bad
        doc = cr.doc
        return {"id": f"doc_{idea['id']}_draft", "idea_id": idea["id"], "stage": "draft", "text": doc,
                "gen_model": client.model.hf_id, "prompt_sha": res.key, "n_tokens": res.usage.get("completion_tokens"),
                "words": len(doc.split()), **diag}
    return fn


def greedy_json(res, schema):
    """Validate a greedy JSON completion and return (object, diagnostics). Diagnostics persist the §6.1
    fallback per record so the stage gate (doc v1.7.3) can be computed from the output files."""
    fb = res.params.get("t0_fallback") if res.t0_fallback else None
    diag = {"t0_fallback": bool(res.t0_fallback),
            "t0_fallback_reason": fb["reason"] if fb else None,
            "greedy_tokens": fb["greedy_completion_tokens"] if fb else res.usage.get("completion_tokens"),
            "final_tokens": fb["fallback_completion_tokens"] if fb else res.usage.get("completion_tokens"),
            "finish_reason": res.finish_reasons[0]}
    return schema.model_validate_json(res.texts[0]), diag


def s6_fn(ctx: GenContext, client: LLMClient, ideas: dict[str, dict[str, Any]]):
    tpl = ctx.template(ctx.template_name("s6"))
    schema = CritiqueV11 if "task_completion" in (tpl.prefix + tpl.suffix) else Critique
    apl = ctx.anti_pattern_list

    async def fn(draft: dict[str, Any]) -> dict[str, Any]:
        idea = ideas[draft["idea_id"]]
        r = tpl.render(**ctx.identity, spec_excerpt=idea["spec_excerpt"], anti_pattern_list=apl,
                       idea_json=idea_json(idea), draft=draft["text"], doc_type=idea["doc_type"])
        ctx.capture("s6_critique", idea["id"], r, idea["domain_id"], response_format_for(schema))
        res = await client.complete_full(r.text, stage="s6_critique", json_schema=schema)
        crit, diag = greedy_json(res, schema)
        extra = ({"task_completion": crit.task_completion, "task_note": crit.task_note}
                 if isinstance(crit, CritiqueV11) else {})
        return {"id": f"crit_{idea['id']}", "doc_id": draft["id"], "idea_id": idea["id"],
                "issues": [i.model_dump() for i in crit.issues], "overall": crit.overall,
                "n_issues": len(crit.issues), **extra, **diag}
    return fn


def s7_fn(ctx: GenContext, client: LLMClient, ideas: dict[str, dict[str, Any]],
          drafts: dict[str, dict[str, Any]], checks: DocChecks):
    tpl = ctx.template(ctx.template_name("s7"))
    req = ctx.requirements_block
    tpl7_sha = template_sha(tpl, req)

    async def fn(crit: dict[str, Any]) -> dict[str, Any] | Rejected:
        idea = ideas[crit["idea_id"]]
        draft = drafts[crit["idea_id"]]
        r = tpl.render(**ctx.identity, **ctx.length_words, requirements_block=req, spec_excerpt=idea["spec_excerpt"],
                       doc_type=idea["doc_type"], idea_json=idea_json(idea), draft=draft["text"],
                       critique_json=json.dumps({k: crit[k] for k in ("task_completion", "task_note", "issues", "overall")
                                                 if k in crit}, ensure_ascii=False),
                       **idea_extras(ctx, idea))
        ctx.capture("s7_rewrite", idea["id"], r, idea["domain_id"])
        try:
            res = await client.complete_full(r.text, stage="s7_rewrite")
        except ThinkingLeakError:
            return Rejected("thinking_leak", {"tolerances": []})
        raw, date_info = res.texts[0], None
        if (ctx.cfg.gen.get("prechecks") or {}).get("date_shift"):     # v1.2 cleanup: dates shifted together
            ex = idea_extras_raw(ctx, idea)
            raw, date_info = cleanup.shift_dates(raw, ex.get("date"), ctx.spec_dates)
        cr = checks.check(raw, res.finish_reasons[0], copy_grams=copy_grams_for(ctx, idea), seed_text=idea.get("source"))
        diag = {"tolerances": cr.tolerances, "n_tokens": res.usage.get("completion_tokens"),
                "finish_reason": res.finish_reasons[0], "doc_words": cr.words, "template_sha": tpl7_sha}
        if cr.bad:
            cr.bad.detail = {**(cr.bad.detail or {}), **diag}
            return cr.bad
        doc = cr.doc
        extra = {}
        if date_info is not None:
            extra["date_shift"] = date_info
        if checks.known_given and (ctx.cfg.gen.get("prechecks") or {}).get("flag_invented_names"):
            extra["invented_names"] = cleanup.invented_names(doc, idea_extras_raw(ctx, idea).get("names") or [],
                                                             checks.known_given)
        return {"id": f"doc_{idea['id']}_rewrite", "idea_id": idea["id"], "stage": "rewrite", "text": doc,
                "gen_model": client.model.hf_id, "prompt_sha": res.key, "words": len(doc.split()), **extra, **diag}
    return fn


def s8_fn(ctx: GenContext, judge: LLMClient, ideas: dict[str, dict[str, Any]], writer: str, judge_name: str,
          capture: bool, template: str | None = None):
    tpl = ctx.template(template or (ctx.cfg.gen.get("s8") or {}).get("template", "E08_filter"))
    apl = ctx.anti_pattern_list

    async def fn(doc: dict[str, Any]) -> dict[str, Any]:
        idea = ideas[doc["idea_id"]]
        r = tpl.render(**ctx.identity, reason_type=ctx.reason_type, anti_pattern_list=apl,
                       spec_excerpt=idea["spec_excerpt"], text=doc["text"])
        if capture:
            ctx.capture("s8_filter", idea["id"], r, idea["domain_id"], response_format_for(FilterVerdict))
        res = await judge.complete_full(r.text, stage="s8_filter", json_schema=FilterVerdict)
        v, diag = greedy_json(res, FilterVerdict)
        crit = {k: getattr(v.criteria, k).verdict == "PASS" for k in S8_CRITERIA}
        verdict = "INCLUDE" if all(crit.values()) else "EXCLUDE"   # §6.3: INCLUDE only if all six pass
        return {"id": f"{doc['id']}__{judge_name}", "doc_id": doc["id"], "idea_id": idea["id"],
                "writer": writer, "judge": judge_name, "judge_model": judge.model.hf_id,
                "criteria": crit, "rationale": {k: getattr(v.criteria, k).reason for k in S8_CRITERIA},
                "verdict": verdict, "model_verdict": v.verdict, "consistent": v.verdict == verdict, **diag}
    return fn
