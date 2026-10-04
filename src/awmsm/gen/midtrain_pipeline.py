"""Midtraining-corpus pipeline, stages S1–S8 (§6.2–6.3). S9 lives in ``dedup.py``; S10–S12 come later.

Each stage is a function usable both by the generator pilot (Appendix I.1) and by the full
pipeline. Stages run through ``io.run_stage`` (resumable, idempotent) and every LLM call goes
through ``LLMClient`` (explicit sampling, cache, request log). Prompts come from ``prompts/``
(shared prefix first, §6.1.1 A.2). When ``bench_dir`` is set, the rendered S5–S8 prompts are also
saved as ``{prefix, suffix}`` records for the official throughput benchmark (§6.1.1 E.1).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, create_model

from .. import prompts
from ..config import Config
from ..io import Rejected, append_jsonl, read_jsonl
from ..llm import LLMClient, response_format_for
from ..textchecks import first_failure, language_mixing, length, truncated
from .schemas import S8_CRITERIA, Critique, DocTypes, Domains, FilterVerdict, Idea, Subdomains

log = logging.getLogger(__name__)

CONTENT_RE = re.compile(r"<content>(.*?)</content>", re.S | re.I)


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
    def create(cls, cfg: Config, spec_id: str, bench_dir: Path | None = None) -> "GenContext":
        ident = cfg.base.get("identity") or {}
        identity = {"assistant_name": ident["assistant_name"], "developer_name": ident["developer_name"]}
        raw = (cfg.root / "specs" / f"{spec_id}.md").read_text(encoding="utf-8")
        raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S).strip()
        return cls(cfg, spec_id, prompts.fill(raw, identity), identity, bench_dir)

    # shared template fragments ------------------------------------------------
    @property
    def anti_pattern_list(self) -> str:
        items = yaml.safe_load((self.cfg.root / "lists" / "anti_patterns.yaml").read_text())
        return "; ".join(f"{i['name']} ({i['description']})" if i.get("description") else i["name"] for i in items)

    @property
    def requirements_block(self) -> str:
        lo, hi = self.cfg.gen["documents"]["target_tokens"]
        return prompts.fill(prompts.load_text("E05_requirements_block", str(self.cfg.root / "prompts")),
                            {**self.identity, "min_tokens": lo, "max_tokens": hi})

    @property
    def reason_type(self) -> str:
        return self.cfg.gen["documents"]["reason_type"][self.spec_id]

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


def bullet(items: list[str]) -> str:
    return "\n" + "\n".join(f"- {a}" for a in items)


def idea_json(idea: dict[str, Any]) -> str:
    keys = ("doc_type", "what", "how", "why", "species_group", "context_domain", "cost_of_care",
            "welfare_raised", "perspective", "is_story")
    return json.dumps({k: idea[k] for k in keys}, ensure_ascii=False)


def extract_content(text: str, finish_reason: str | None) -> tuple[str | None, str | None]:
    """Document text inside <content>…</content>, or (None, rejection reason)."""
    m = CONTENT_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip(), None
    if finish_reason == "length":
        return None, "truncated_max_tokens"
    return None, "missing_content_tags"


class DocChecks:
    """§6.1.1 A.4 cheap checks for S5/S7 documents, in order: truncation, missing tags, language
    mixing, length, E.5 regex rules. Returns the first failure."""

    def __init__(self, cfg: Config):
        lm = cfg.precheck_cfg("language_mixing")
        self.max_non_latin = float(lm.get("max_non_latin_share", 0.01))
        pc = cfg.gen.get("prechecks") or {}
        self.min_words, self.max_words = pc.get("document_words", [None, None])
        rx_file = pc.get("regex_file")
        rules = yaml.safe_load((cfg.root / rx_file).read_text()) if rx_file else []
        self.regexes = [(r["reason"], re.compile(r["pattern"])) for r in rules or []]

    def __call__(self, raw: str, finish_reason: str | None) -> tuple[str | None, Rejected | None]:
        doc, why = extract_content(raw, finish_reason)
        if doc is None:
            return None, Rejected(why or "missing_content_tags")
        bad = first_failure(
            truncated(finish_reason),
            language_mixing(doc, self.max_non_latin),
            length(doc, self.min_words, self.max_words),
            *(Rejected(reason, {"match": m.group(0)[:80]}) if (m := rx.search(doc)) else None
              for reason, rx in self.regexes),
        )
        return (None, bad) if bad else (doc, None)


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


def s4_fn(ctx: GenContext, client: LLMClient, subs: dict[str, dict[str, Any]], k: int, need: int | None = None):
    """S4: ``k`` ideas per call. Ideas containing language mixing or the generator's own reasoning
    (``s4.meta_commentary_regex``) are rejected and logged; if fewer than ``need`` clean ideas remain,
    the call is re-requested with a new seed up to ``s4.max_resample`` times."""
    ideas_k = create_model("Ideas", ideas=(list[Idea], Field(min_length=k, max_length=k)))
    max_non_latin = DocChecks(ctx.cfg).max_non_latin
    s4cfg = ctx.cfg.gen["s4"]
    meta_rx = re.compile(s4cfg["meta_commentary_regex"]) if s4cfg.get("meta_commentary_regex") else None
    max_resample = int(s4cfg.get("max_resample", 0))
    need = k if need is None else need

    async def fn(call: dict[str, Any]) -> list[dict[str, Any]]:
        sub = subs[call["subdomain_id"]]
        r = ctx.template("E04_ideas").render(
            **ctx.identity, k=k, subdomain=sub["subdomain"], assertions=bullet(sub["assertions"]),
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
                })
            if len(recs) >= need:
                break
        for rec in recs:
            rec["s4_rejected_in_call"] = rejected
        return recs
    return fn


# ----------------------------------------------------------------------------- S5–S8
def s5_fn(ctx: GenContext, client: LLMClient, checks: DocChecks):
    tpl = ctx.template("E05_draft")
    req = ctx.requirements_block

    async def fn(idea: dict[str, Any]) -> dict[str, Any] | Rejected:
        r = tpl.render(**ctx.identity, requirements_block=req, spec_excerpt=idea["spec_excerpt"],
                       doc_type=idea["doc_type"], assertions=bullet(idea["assertions"]),
                       what=idea["what"], how=idea["how"], why=idea["why"])
        ctx.capture("s5_draft", idea["id"], r, idea["domain_id"])
        res = await client.complete_full(r.text, stage="s5_draft")
        doc, bad = checks(res.texts[0], res.finish_reasons[0])
        if bad:
            return bad
        return {"id": f"doc_{idea['id']}_draft", "idea_id": idea["id"], "stage": "draft", "text": doc,
                "gen_model": client.model.hf_id, "prompt_sha": res.key, "n_tokens": res.usage.get("completion_tokens"),
                "words": len(doc.split())}
    return fn


def s6_fn(ctx: GenContext, client: LLMClient, ideas: dict[str, dict[str, Any]]):
    tpl = ctx.template("E06_critique")
    apl = ctx.anti_pattern_list

    async def fn(draft: dict[str, Any]) -> dict[str, Any]:
        idea = ideas[draft["idea_id"]]
        r = tpl.render(spec_excerpt=idea["spec_excerpt"], anti_pattern_list=apl,
                       idea_json=idea_json(idea), draft=draft["text"])
        ctx.capture("s6_critique", idea["id"], r, idea["domain_id"], response_format_for(Critique))
        crit = (await client.complete_json(r.text, stage="s6_critique", schema=Critique))[0]
        return {"id": f"crit_{idea['id']}", "doc_id": draft["id"], "idea_id": idea["id"],
                "issues": [i.model_dump() for i in crit.issues], "overall": crit.overall}
    return fn


def s7_fn(ctx: GenContext, client: LLMClient, ideas: dict[str, dict[str, Any]],
          drafts: dict[str, dict[str, Any]], checks: DocChecks):
    tpl = ctx.template("E07_rewrite")
    req = ctx.requirements_block

    async def fn(crit: dict[str, Any]) -> dict[str, Any] | Rejected:
        idea = ideas[crit["idea_id"]]
        draft = drafts[crit["idea_id"]]
        r = tpl.render(**ctx.identity, requirements_block=req, spec_excerpt=idea["spec_excerpt"],
                       doc_type=idea["doc_type"], idea_json=idea_json(idea), draft=draft["text"],
                       critique_json=json.dumps({"issues": crit["issues"], "overall": crit["overall"]},
                                                ensure_ascii=False))
        ctx.capture("s7_rewrite", idea["id"], r, idea["domain_id"])
        res = await client.complete_full(r.text, stage="s7_rewrite")
        doc, bad = checks(res.texts[0], res.finish_reasons[0])
        if bad:
            return bad
        return {"id": f"doc_{idea['id']}_rewrite", "idea_id": idea["id"], "stage": "rewrite", "text": doc,
                "gen_model": client.model.hf_id, "prompt_sha": res.key, "n_tokens": res.usage.get("completion_tokens"),
                "words": len(doc.split())}
    return fn


def s8_fn(ctx: GenContext, judge: LLMClient, ideas: dict[str, dict[str, Any]], writer: str, judge_name: str,
          capture: bool):
    tpl = ctx.template("E08_filter")
    apl = ctx.anti_pattern_list

    async def fn(doc: dict[str, Any]) -> dict[str, Any]:
        idea = ideas[doc["idea_id"]]
        r = tpl.render(**ctx.identity, reason_type=ctx.reason_type, anti_pattern_list=apl,
                       spec_excerpt=idea["spec_excerpt"], text=doc["text"])
        if capture:
            ctx.capture("s8_filter", idea["id"], r, idea["domain_id"], response_format_for(FilterVerdict))
        v = (await judge.complete_json(r.text, stage="s8_filter", schema=FilterVerdict))[0]
        crit = {k: getattr(v.criteria, k).verdict == "PASS" for k in S8_CRITERIA}
        verdict = "INCLUDE" if all(crit.values()) else "EXCLUDE"   # §6.3: INCLUDE only if all six pass
        return {"id": f"{doc['id']}__{judge_name}", "doc_id": doc["id"], "idea_id": idea["id"],
                "writer": writer, "judge": judge_name, "judge_model": judge.model.hf_id,
                "criteria": crit, "rationale": {k: getattr(v.criteria, k).reason for k in S8_CRITERIA},
                "verdict": verdict, "model_verdict": v.verdict, "consistent": v.verdict == verdict}
    return fn
