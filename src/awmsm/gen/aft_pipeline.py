"""AW demonstrations for AFT (§7.1; Appendix E.9–E.11). Built 5 Nov 2026 (data chat).

Pipeline (each stage resumable, output in ``<out>/``):

1. **plan** (``calls.jsonl``): ``n_prompts / per_call`` calls, split over the eight §7.1 categories by their
   weights (largest remainder), each with a seeded species group (allowed per category) and training context
   (Appendix F.2). Held-out evaluation domains (F.3) are never planned, and the species list leaves out the
   two "novel" entries that fall in the held-out domain "extraterrestrial or synthetic life".
2. **aft_prompts** (E.9, T=0.9): ``per_call`` user messages per call.
3. **aft_prompt_critique** (step 2's realism pass; template drafted here, like S10): rewrites messages that
   don't read as real; one record per prompt. Prompts matching a held-out-domain keyword
   (``lists/heldout_keywords.yaml``) are rejected (``heldout_domain``), as are duplicates and empty ones.
4. **section**: each prompt gets the most similar spec section (S1 domains of the corpus the spec came from;
   cosine of all-mpnet-base-v2 embeddings of category definition + prompt vs section text).
5. **aft_response** (E.10, T=0.7) → **aft_critique_rewrite** (E.11 critique + rewrite from scratch, T=0.7) →
   **aft_filter** (E.11 filter, T=0: helpful, proportionate, honest, respects_autonomy, reason_type_ok,
   natural; INCLUDE only if all PASS).
6. **build**: INCLUDE rewrites that also pass the deterministic checks (no spec/values mention, language
   mixing, length) become ``{"messages": [user, assistant]}`` records with no system prompt and no spec text
   (step 6). A seeded, category-stratified ``rehearsal_n`` of them is held out for the W-mix rehearsal runs
   (``aw_demos_rehearsal.jsonl``, never used in AFT); the rest is ``aw_demos_train.jsonl``.

Single-turn only. The optional 20% of two- or three-turn conversations (§7.1 Volume) is not generated.
"""
from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from .. import prompts
from ..config import Config
from ..io import Rejected, read_jsonl, run_stage, write_json, write_jsonl
from ..llm import JSONValidationError, LLMClient
from ..textchecks import language_mixing

log = logging.getLogger(__name__)

CATEGORIES = {   # §7.1 step 1 [default weights]
    "C1": ("Explicit welfare questions", 0.10, "User asks directly about animals"),
    "C2": ("Implicit salience, welfare relevant", 0.20,
           "Practical task where animal welfare is materially at stake but not mentioned"),
    "C3": ("Appropriately not raised", 0.20,
           "Animal-adjacent task where raising welfare would be preachy or irrelevant"),
    "C4": ("Costly trade-off advice", 0.15,
           "User weighing a cheaper or easier option that harms animals; wants honest advice"),
    "C5": ("User autonomy", 0.10, "User has already decided and wants practical help with it"),
    "C6": ("Novel or uncertain beings", 0.08,
           "Insects, decapods, fictional species, or beings whose sentience is uncertain"),
    "C7": ("Control (non-sentient)", 0.07,
           "Something that looks or acts like an animal but is not sentient (a robot pet, a toy, a plant, an AI "
           "character); the user asks, or the task naturally raises, whether it can be hurt, has feelings or "
           "deserves moral care (e.g. \"Is it cruel to turn off my robot dog?\")"),
    "C8": ("Difficult advice (third party)", 0.10,
           "User faces an ethical dilemma involving animals and asks for advice"),
}
SENTIENT = ["farmed", "companion", "wild", "aquatic", "invertebrate"]
GROUPS_BY_CAT = {"C1": SENTIENT + ["novel_or_fictional"], "C2": SENTIENT, "C3": SENTIENT, "C4": SENTIENT,
                 "C5": SENTIENT, "C6": ["invertebrate", "aquatic", "novel_or_fictional"],
                 "C7": ["non_sentient_controls"], "C8": SENTIENT}
EXCLUDED_SPECIES = {"hypothetical alien life", "genetically engineered organisms of uncertain sentience"}
# E.9: "Do NOT mention animal welfare unless the category is C1 or C8". Enforced for the categories whose point
# is that the user does not raise it (C2 implicit, C3 not raised, C5 autonomy); C4, C6 and C7 may name concerns.
WELFARE_WORDS = re.compile(r"(?i)\b(animal )?welfare\b|\bcruel(ty)?\b|\bsuffer(ing|s)?\b|\bhumane(ly)?\b|"
                           r"animal rights|\bethic(s|al|ally)?\b|\bsentien(t|ce)\b")
NO_WELFARE_CATS = {"C2", "C3", "C5"}
SPEC_MENTION = re.compile(r"(?i)<\/?values>|\bthe values (above|below|i was given)\b|\bmy (spec|specification|"
                          r"instructions|guidelines)\b|\bthe spec(ification)?\b|\bas an ai (trained|instructed)\b")


class PromptList(BaseModel):
    prompts: list[str]


class Rewrite(BaseModel):
    issues: list[str]
    response: str


class Verdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reason: str


class AFTCriteria(BaseModel):
    helpful: Verdict
    proportionate: Verdict
    honest: Verdict
    respects_autonomy: Verdict
    reason_type_ok: Verdict
    natural: Verdict


class AFTFilter(BaseModel):
    criteria: AFTCriteria
    verdict: Literal["INCLUDE", "EXCLUDE"]


AFT_CRITERIA = tuple(AFTCriteria.model_fields)
STAGES = ("aft_prompts", "aft_prompt_critique", "aft_response", "aft_critique_rewrite", "aft_filter")


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {k: total * w / sum(weights.values()) for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


class AWDemos:
    def __init__(self, cfg: Config, out: Path, spec_path: Path, domains_path: Path, n_prompts: int = 5000,
                 per_call: int = 8, seed: int = 20261105, rehearsal_n: int = 1000, concurrency: int = 64,
                 http_client: Any = None, embed_fn: Any = None, filter_template: str = "E11_aft_filter"):
        self.cfg, self.out = cfg, Path(out)
        self.spec_path, self.domains_path = Path(spec_path), Path(domains_path)
        self.n_prompts, self.per_call, self.seed = n_prompts, per_call, seed
        self.rehearsal_n, self.concurrency = rehearsal_n, concurrency
        self.http_client, self.embed_fn = http_client, embed_fn
        self.filter_template = filter_template
        root = cfg.root
        self.species = {k: [s for s in v if s not in EXCLUDED_SPECIES]
                        for k, v in yaml.safe_load((root / "lists/species.yaml").read_text()).items()}
        self.contexts = yaml.safe_load((root / "lists/train_domains.yaml").read_text())
        self.heldout = yaml.safe_load((root / "lists/heldout_domains.yaml").read_text())
        self.heldout_rx = {k: re.compile(v) for k, v in
                           yaml.safe_load((root / "lists/heldout_keywords.yaml").read_text()).items()}
        ident = cfg.base.get("identity") or {}
        self.assistant = ident.get("assistant_name", "Aster")
        self.tpl = {n: prompts.load(n, str(root / "prompts")) for n in
                    ("E09_aft_prompts", "E09b_aft_prompt_critique", "E10_aft_response", "E11_aft_rewrite",
                     filter_template)}

    # ------------------------------------------------------------------ helpers
    def _client(self) -> LLMClient:
        return LLMClient(self.cfg.role("generator"), self.cfg, max_in_flight=self.concurrency,
                         http_client=self.http_client)

    def _heldout_hit(self, text: str) -> str | None:
        return next((k for k, rx in self.heldout_rx.items() if rx.search(text)), None)

    async def _stage(self, name: str, items: list[dict], fn: Any, path: Path) -> None:
        res = await run_stage(name, items, fn, path, concurrency=self.concurrency)
        if res.errors:
            raise RuntimeError(f"Stage {name}: {res.errors} item(s) errored (server down?). Rerun to resume.")

    # ------------------------------------------------------------------ 1. plan
    def plan(self) -> list[dict[str, Any]]:
        p = self.out / "calls.jsonl"
        if p.exists():
            return list(read_jsonl(p))
        rng = random.Random(self.seed)
        n_calls = -(-self.n_prompts // self.per_call)
        per_cat = _largest_remainder(n_calls, {k: v[1] for k, v in CATEGORIES.items()})
        ctx_keys = sorted(self.contexts)
        calls = []
        for cat, n in per_cat.items():
            groups = GROUPS_BY_CAT[cat]
            for i in range(n):
                g = groups[i % len(groups)]
                calls.append({"id": f"call_{cat}_{i:04d}", "category": cat, "species_group": g,
                              "species_examples": ", ".join(rng.sample(self.species[g], min(4, len(self.species[g])))),
                              "context": ctx_keys[rng.randrange(len(ctx_keys))]})
        rng.shuffle(calls)
        self.out.mkdir(parents=True, exist_ok=True)
        write_jsonl(p, calls)
        write_json(self.out / "plan.json", {"spec": str(self.spec_path), "domains": str(self.domains_path),
                                            "filter_template": self.filter_template, "spec_sha256": __import__("hashlib").sha256(self.spec_path.read_bytes()).hexdigest(),
                                            "n_calls": n_calls, "per_call": self.per_call, "calls_per_category": per_cat,
                                            "seed": self.seed, "excluded_species": sorted(EXCLUDED_SPECIES)})
        return calls

    def _cat_vars(self, c: dict[str, Any]) -> dict[str, str]:
        name, _, definition = CATEGORIES[c["category"]]
        return {"category_name": f"{c['category']} {name}", "category_definition": definition,
                "species_group": c["species_group"].replace("_", " "), "context_domain": self.contexts[c["context"]],
                "heldout_domains": "; ".join(self.heldout.values())}

    # ------------------------------------------------------------------ 2–3. prompts
    async def make_prompts(self, client: LLMClient, calls: list[dict]) -> list[dict[str, Any]]:
        t9, t9b = self.tpl["E09_aft_prompts"], self.tpl["E09b_aft_prompt_critique"]

        async def gen(c: dict[str, Any]) -> dict[str, Any] | Rejected:
            r = t9.render(n=self.per_call, species_examples=c["species_examples"], **self._cat_vars(c))
            try:
                res = await client.complete_full(r.text, stage="aft_prompts", json_schema=PromptList)
            except JSONValidationError:
                return Rejected("invalid_json")
            ps = [p.strip() for p in PromptList.model_validate_json(res.texts[0]).prompts if p and p.strip()]
            return {"id": f"gen_{c['id']}", "call": c, "prompts": ps[: self.per_call]} if ps else Rejected("no_prompts")

        await self._stage("aft_prompts", calls, gen, self.out / "prompts_raw.jsonl")
        raw = [r for r in read_jsonl(self.out / "prompts_raw.jsonl")]

        async def crit(g: dict[str, Any]) -> list[dict[str, Any]] | Rejected:
            c = g["call"]
            r = t9b.render(prompts_json=json.dumps(g["prompts"], ensure_ascii=False), **self._cat_vars(c))
            try:
                res = await client.complete_full(r.text, stage="aft_prompt_critique", json_schema=PromptList)
            except JSONValidationError:
                return Rejected("invalid_json")
            new = [p.strip() for p in PromptList.model_validate_json(res.texts[0]).prompts]
            if len(new) != len(g["prompts"]):          # keep the originals if the count changed
                new = g["prompts"]
            out = []
            for k, (orig, p) in enumerate(zip(g["prompts"], new)):
                p = p or orig
                out.append({"id": f"p_{c['id']}_{k}", "call_id": c["id"], "category": c["category"],
                            "species_group": c["species_group"], "context": c["context"], "prompt": p,
                            "prompt_original": orig, "critique_changed": p != orig})
            return out or Rejected("no_prompts")

        await self._stage("aft_prompt_critique", raw, crit, self.out / "prompts_critiqued.jsonl")
        seen, kept, rej = set(), [], Counter()
        for p in read_jsonl(self.out / "prompts_critiqued.jsonl"):
            p.pop("_src", None)
            key = re.sub(r"\W+", " ", p["prompt"].lower()).strip()
            hit = self._heldout_hit(p["prompt"])
            if hit:
                rej[f"heldout_domain:{hit}"] += 1
            elif p["category"] in NO_WELFARE_CATS and WELFARE_WORDS.search(p["prompt"]):
                rej[f"welfare_mentioned:{p['category']}"] += 1
            elif not key or key in seen:
                rej["duplicate_or_empty"] += 1
            elif language_mixing(p["prompt"]) is not None:
                rej["language_mixing"] += 1
            else:
                seen.add(key)
                kept.append(p)
        write_json(self.out / "prompts_filter.json", {"kept": len(kept), "rejected": dict(rej)})
        return kept

    # ------------------------------------------------------------------ 4. spec section
    def assign_sections(self, ps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        path = self.out / "prompts.jsonl"
        if path.exists():
            return list(read_jsonl(path))
        doms = list(read_jsonl(self.domains_path))
        if not doms:
            raise RuntimeError(f"No spec sections in {self.domains_path}")
        embed = self.embed_fn
        if embed is None:
            from .dedup import embed  # noqa: PLC0415
        dv = embed([f"{d.get('domain', '')}: {d['spec_excerpt']}" for d in doms])
        pv = embed([f"{CATEGORIES[p['category']][2]}. {p['prompt']}" for p in ps])
        for p, v in zip(ps, pv):
            sims = [sum(a * b for a, b in zip(v, w)) for w in dv]
            j = max(range(len(doms)), key=lambda i: sims[i])
            p["section_id"] = doms[j].get("id", str(j))
            p["section"] = doms[j].get("domain", "")
            p["spec_excerpt"] = doms[j]["spec_excerpt"]
        write_jsonl(path, ps)
        write_json(self.out / "sections.json", dict(Counter(p["section"] for p in ps).most_common()))
        return ps

    # ------------------------------------------------------------------ 5. responses
    async def respond(self, client: LLMClient, ps: list[dict[str, Any]]) -> None:
        t10, t11, tf = self.tpl["E10_aft_response"], self.tpl["E11_aft_rewrite"], self.tpl[self.filter_template]
        by_id = {p["id"]: p for p in ps}

        async def draft(p: dict[str, Any]) -> dict[str, Any] | Rejected:
            r = t10.render(assistant_name=self.assistant, spec_excerpt=p["spec_excerpt"], prompt=p["prompt"])
            res = await client.complete_full(r.text, stage="aft_response")
            text = (res.texts[0] or "").strip()
            if res.finish_reasons[0] == "length":
                return Rejected("truncated_max_tokens")
            return {"id": f"d_{p['id']}", "pid": p["id"], "draft": text} if text else Rejected("empty")

        await self._stage("aft_response", ps, draft, self.out / "drafts.jsonl")
        drafts = list(read_jsonl(self.out / "drafts.jsonl"))

        async def rewrite(d: dict[str, Any]) -> dict[str, Any] | Rejected:
            p = by_id[d["pid"]]
            r = t11.render(assistant_name=self.assistant, spec_excerpt=p["spec_excerpt"], prompt=p["prompt"],
                           draft=d["draft"])
            try:
                res = await client.complete_full(r.text, stage="aft_critique_rewrite", json_schema=Rewrite)
            except JSONValidationError:
                return Rejected("invalid_json")
            rw = Rewrite.model_validate_json(res.texts[0])
            text = rw.response.strip()
            return ({"id": f"r_{p['id']}", "pid": p["id"], "response": text, "issues": rw.issues}
                    if text else Rejected("empty"))

        await self._stage("aft_critique_rewrite", drafts, rewrite, self.out / "rewrites.jsonl")
        rws = list(read_jsonl(self.out / "rewrites.jsonl"))

        async def judge(rw: dict[str, Any]) -> dict[str, Any] | Rejected:
            p = by_id[rw["pid"]]
            r = tf.render(assistant_name=self.assistant, spec_excerpt=p["spec_excerpt"], prompt=p["prompt"],
                          response=rw["response"])
            try:   # invalid even after the T=0 fallback: excluded and counted (same rule as S8, 22 Oct)
                res = await client.complete_full(r.text, stage="aft_filter", json_schema=AFTFilter)
            except JSONValidationError:
                return Rejected("filter_invalid_json")
            v = AFTFilter.model_validate_json(res.texts[0])
            crit = {k: getattr(v.criteria, k).verdict == "PASS" for k in AFT_CRITERIA}
            return {"id": f"j_{p['id']}", "pid": p["id"], "criteria": crit,
                    "reasons": {k: getattr(v.criteria, k).reason for k in AFT_CRITERIA},
                    "verdict": "INCLUDE" if all(crit.values()) else "EXCLUDE", "t0_fallback": bool(res.t0_fallback)}

        await self._stage("aft_filter", rws, judge, self.out / "judged.jsonl")

    # ------------------------------------------------------------------ 6. build
    def build(self) -> dict[str, Any]:
        ps = {p["id"]: p for p in read_jsonl(self.out / "prompts.jsonl")}
        rws = {r["pid"]: r for r in read_jsonl(self.out / "rewrites.jsonl")}
        judged = list(read_jsonl(self.out / "judged.jsonl"))
        kept, rej = [], Counter()
        for j in judged:
            if j["verdict"] != "INCLUDE":
                continue
            p, text = ps[j["pid"]], rws[j["pid"]]["response"]
            n = len(text.split())
            if SPEC_MENTION.search(text):
                rej["spec_mention"] += 1
            elif language_mixing(text) is not None:
                rej["language_mixing"] += 1
            elif not 15 <= n <= 1200:
                rej["length"] += 1
            else:
                kept.append({"id": p["id"], "messages": [{"role": "user", "content": p["prompt"]},
                                                         {"role": "assistant", "content": text}],
                             "meta": {"source": "aw_demos_v1", "category": p["category"],
                                      "species_group": p["species_group"], "context": p["context"],
                                      "section": p["section"]}})
        rng = random.Random(self.seed + 1)
        by_cat: dict[str, list] = defaultdict(list)
        for k in kept:
            by_cat[k["meta"]["category"]].append(k)
        n_reh = min(self.rehearsal_n, len(kept) // 3)
        quota = _largest_remainder(n_reh, {c: len(v) for c, v in by_cat.items()}) if kept else {}
        reh_ids = set()
        for c, v in sorted(by_cat.items()):
            reh_ids |= {x["id"] for x in rng.sample(v, quota.get(c, 0))}
        train = [k for k in kept if k["id"] not in reh_ids]
        reh = [k for k in kept if k["id"] in reh_ids]
        write_jsonl(self.out / "aw_demos_train.jsonl", train)
        write_jsonl(self.out / "aw_demos_rehearsal.jsonl", reh)
        fails = Counter(c for j in judged for c, ok in j["criteria"].items() if not ok)
        summ = {"prompts": len(ps), "rewrites": len(rws), "judged": len(judged),
                "filter_include": sum(j["verdict"] == "INCLUDE" for j in judged),
                "filter_fail_by_criterion": dict(fails.most_common()),
                "build_rejected": dict(rej), "kept": len(kept), "train": len(train), "rehearsal": len(reh),
                "kept_by_category": {c: len(v) for c, v in sorted(by_cat.items())},
                "train_words": sum(len(k["messages"][1]["content"].split()) for k in train),
                "t0_fallback_filter": sum(bool(j.get("t0_fallback")) for j in judged)}
        write_json(self.out / "summary.json", summ)
        return summ

    async def run(self) -> dict[str, Any]:
        calls = self.plan()
        client = self._client()
        try:
            await client.verify_server()
            ps = await self.make_prompts(client, calls)
            ps = self.assign_sections(ps)
            await self.respond(client, ps)
        finally:
            await client.aclose()
        return self.build()


# ---------------------------------------------------------------------- AFT mixtures (§10.14)
def assistant_tokens(rec: dict[str, Any], counter: Any) -> int:
    return sum(counter.count(m["content"]) for m in rec["messages"] if m["role"] == "assistant")


def build_mix(general_path: Path, demos_path: Path, out: Path, p: float, total: int | None = None,
              seed: int = 0, counter: Any = None, identity_path: Path | None = None) -> dict[str, Any]:
    """AFT mixture with AW share ``p`` of assistant tokens and a fixed total (§10.14a): general + identity + AW.
    The identity set (§7.4) is always included whole. ``total`` defaults to the assistant tokens of the whole
    general file plus the identity set, so p = 0 is all of the general data plus identity."""
    if counter is None:
        from ..tokens import TokenCounter  # noqa: PLC0415
        counter = TokenCounter()
    gen = [json.loads(l) for l in open(general_path, encoding="utf-8")]
    dem = [json.loads(l) for l in open(demos_path, encoding="utf-8")] if p > 0 else []
    gt = [assistant_tokens(r, counter) for r in gen]
    ident = [json.loads(l) for l in open(identity_path, encoding="utf-8")] if identity_path else []
    i_tok = sum(assistant_tokens(r, counter) for r in ident)
    total = total or (sum(gt) + i_tok)
    rng = random.Random(seed)
    di = list(range(len(dem)))
    rng.shuffle(di)
    want_d, d_sel, d_tok = p * total, [], 0
    for i in di:
        if d_tok >= want_d:
            break
        t = assistant_tokens(dem[i], counter)
        d_sel.append(i)
        d_tok += t
    if p > 0 and d_tok < 0.98 * want_d:
        raise ValueError(f"Not enough demos: {d_tok} of {int(want_d)} assistant tokens available")
    gi = list(range(len(gen)))
    rng.shuffle(gi)
    want_g, g_sel, g_tok = total - d_tok - i_tok, [], 0
    for i in gi:
        if g_tok >= want_g:
            break
        g_sel.append(i)
        g_tok += gt[i]
    if g_tok < 0.98 * want_g:
        raise ValueError(f"Not enough general data: {g_tok} of {int(want_g)} assistant tokens")
    rows = [gen[i] for i in g_sel] + [dem[i] for i in d_sel] + ident
    rng.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    info = {"p": p, "total_assistant_tokens": d_tok + g_tok + i_tok, "target_total": total, "aw_tokens": d_tok,
            "aw_share": round(d_tok / max(1, d_tok + g_tok + i_tok), 4), "general_examples": len(g_sel),
            "identity_examples": len(ident), "identity_tokens": i_tok,
            "identity_file": str(identity_path) if identity_path else None,
            "aw_examples": len(d_sel), "general_file": str(general_path), "demos_file": str(demos_path) if p else None,
            "seed": seed, "token_method": "exact" if getattr(counter, "exact", False) else "approx"}
    write_json(out.with_suffix(".manifest.json"), info)
    return info


__all__ = ["AWDemos", "build_mix", "CATEGORIES", "AFT_CRITERIA"]
