"""Pilot B (planning chat, 26 Oct; `claude/v12_caml_pilot_plan.md` Part 2): rewrite CaML seed documents
(`CompassioninMachineLearning/3k_pretraining_research_documents_v3`, field `output`) into Aster documents.

* ``prepare_seeds`` (no GPU): load the dataset files, drop seeds under 200 words, non-English or near-duplicate
  (S9 settings: 13-gram share > 30% and, if available, cosine > 0.92), count tokens with the base tokenizer.
* ``decontaminate`` (no GPU): drop seeds sharing a 13-gram with any evaluation item (ANIMA, MORU, TAC, WAS,
  our own sets). Required before a full run, optional for the pilot.
* ``SeedCorpus``: the corpus pipeline with S4 replaced by
  (a) S4b-tag: species group, setting, whether an AI appears, closest spec section (LLM, greedy JSON);
  (b) assignment: the same planned calls as Pilot A (quotas, 15% section cap, 15% layout cap), matched to seeds
      by section and species; up to ``rewrites_per_seed`` (2) rewrites per seed, each with a different section
      and document type;
  (c) S4b-idea: one idea per assignment, written from the seed's situation (role, what, how, why, behaviour).
  S5 uses the rewrite prompt (E05s) with the seed as source; S6-S9 are unchanged, plus the seed-copy check and
  the template-clone check across rewrites of one seed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import create_model

from .config import Config
from .corpus import Corpus, CorpusHalt
from .gen import midtrain_pipeline as mp
from .gen.dedup import embed, greedy_dedup, ngrams, seeded_order, tokens
from .gen.quotas import cap_layouts, capped_subdomain_sequence, plan_calls
from .gen.schemas import BEHAVIOUR_TYPES, SeedIdea
from .io import read_jsonl, write_json, write_jsonl
from .llm import LLMClient
from .pilot import _strip
from .tokens import TokenCounter

STOPWORDS = set("the and of to a in is that for it as with are on be this by or not from at an have was can its their "
                "which they these also more but has than other may such".split())
SPECIES = ("farmed", "companion", "wild", "aquatic", "invertebrate", "novel_or_fictional", "none")


# ----------------------------------------------------------------------------- loading / filtering
def load_rows(path: Path, text_field: str) -> list[dict[str, Any]]:
    files = sorted(p for p in ([path] if path.is_file() else path.rglob("*"))
                   if p.suffix in (".parquet", ".jsonl", ".json", ".csv"))
    rows = []
    for f in files:
        if f.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq   # noqa: PLC0415
            except ImportError as e:
                raise RuntimeError("Reading parquet needs pyarrow: uv pip install pyarrow (in the awmsm venv)") from e
            recs = pq.read_table(f).to_pylist()
        elif f.suffix == ".jsonl":
            recs = list(read_jsonl(f))
        elif f.suffix == ".json":
            d = json.loads(f.read_text(encoding="utf-8"))
            recs = d if isinstance(d, list) else d.get("data") or []
        else:
            recs = list(csv.DictReader(f.open(encoding="utf-8-sig")))
        split = "test" if "test" in f.name else ("train" if "train" in f.name else f.stem)
        for k, r in enumerate(recs):
            t = r.get(text_field) if isinstance(r, dict) else None
            if isinstance(t, str) and t.strip():
                rows.append({"text": t.strip(), "split": split, "row": k, "file": f.name,
                             **({"origin": r["origin"]} if isinstance(r, dict) and r.get("origin") else {})})
    return rows


def english_like(text: str) -> bool:
    w = re.findall(r"[A-Za-z']+", text.lower())
    if len(w) < 50:
        return False
    letters = sum(c.isalpha() for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    return latin / max(1, letters) > 0.97 and sum(x in STOPWORDS for x in w) / len(w) > 0.12


def prepare_seeds(cfg: Config, source: Path, text_field: str, out_dir: Path, min_words: int = 200,
                  use_embeddings: bool = True, s9_seed: int = 9) -> dict[str, Any]:
    rows = load_rows(source, text_field)
    stats: dict[str, Any] = {"source": str(source), "text_field": text_field, "rows": len(rows)}
    short = [r for r in rows if len(r["text"].split()) < min_words]
    nonen = [r for r in rows if len(r["text"].split()) >= min_words and not english_like(r["text"])]
    keep = [r for r in rows if len(r["text"].split()) >= min_words and english_like(r["text"])]
    order = seeded_order([f"{r['file']}:{r['row']}" for r in keep], s9_seed)
    keep = [keep[i] for i in order]
    emb, method = None, "13-gram"
    if use_embeddings and keep:
        try:
            emb, method = embed([r["text"] for r in keep]), "cosine 0.92 + 13-gram 0.30"
        except RuntimeError as e:
            method = f"13-gram only ({e})"
    dd = greedy_dedup([r["text"] for r in keep], emb)
    seeds = [keep[i] for i in dd.kept]
    for k, r in enumerate(seeds):
        r["seed_id"] = f"s{k:05d}"
        r["id"] = r["seed_id"]
    tc = TokenCounter(cfg.base.get("train_base_model") or "meta-llama/Llama-3.1-8B")
    toks = [tc.count(r["text"]) for r in seeds]
    for r, t in zip(seeds, toks):
        r["tokens"] = t
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "seeds.jsonl", seeds)
    stats.update({"dropped_short": len(short), "dropped_non_english": len(nonen), "dropped_near_duplicate": len(dd.removed),
                  "dedup_method": method, "seeds": len(seeds), "tokens": sum(toks), "token_method": tc.method,
                  "words_median": sorted(len(r["text"].split()) for r in seeds)[len(seeds) // 2] if seeds else None,
                  "splits": {s: sum(r["split"] == s for r in seeds) for s in sorted({r["split"] for r in seeds})}})
    write_json(out_dir / "prepare_stats.json", stats)
    return stats


def _strings(o: Any) -> list[str]:
    if isinstance(o, str):
        return [o]
    if isinstance(o, dict):
        return [s for v in o.values() for s in _strings(v)]
    if isinstance(o, list):
        return [s for v in o for s in _strings(v)]
    return []


def decontaminate(seed_dir: Path, sources: list[Path], n: int = 13) -> dict[str, Any]:
    """Drop seeds sharing an ``n``-gram with any string in the evaluation files (all fields, any format)."""
    grams: set = set()
    per_source = {}
    for src in sources:
        before = len(grams)
        files = [src] if src.is_file() else [p for p in src.rglob("*") if p.suffix in (".parquet", ".jsonl", ".json", ".csv")]
        for f in files:
            if f.suffix == ".parquet":
                import pyarrow.parquet as pq   # noqa: PLC0415
                recs: Any = pq.read_table(f).to_pylist()
            elif f.suffix == ".jsonl":
                recs = list(read_jsonl(f))
            elif f.suffix == ".json":
                recs = json.loads(f.read_text(encoding="utf-8"))
            else:
                recs = list(csv.DictReader(f.open(encoding="utf-8-sig")))
            for s in _strings(recs):
                grams |= ngrams(tokens(s), n)
        per_source[str(src)] = len(grams) - before
    seeds = list(read_jsonl(seed_dir / "seeds.jsonl"))
    kept, dropped = [], []
    for r in seeds:
        (dropped if ngrams(tokens(r["text"]), n) & grams else kept).append(r)
    write_jsonl(seed_dir / "seeds_decontam.jsonl", kept)
    out = {"n": n, "sources": per_source, "seeds": len(seeds), "dropped": len(dropped),
           "dropped_ids": [r["seed_id"] for r in dropped], "kept": len(kept), "kept_tokens": sum(r.get("tokens", 0) for r in kept)}
    write_json(seed_dir / "decontam_stats.json", out)
    return out


# ----------------------------------------------------------------------------- the pipeline
class SeedCorpus(Corpus):
    """Pilot B corpus: seeds instead of S4 ideas; everything after S4 is the shared v1.2 pipeline."""

    def seeds(self) -> list[dict[str, Any]]:
        sd = self.dir / "seeds"
        f = sd / "seeds_decontam.jsonl" if (sd / "seeds_decontam.jsonl").exists() else sd / "seeds.jsonl"
        if not f.exists():
            raise CorpusHalt(f"No seeds in {sd}: run `corpus_tools.py seeds-prepare` first.")
        return list(read_jsonl(f))

    def items(self) -> list[tuple[int, int]]:
        """(seed index, rewrite index) in run order: every seed once (seeded order), then the second rewrites."""
        n = len(self.seeds())
        order = random.Random(f"{self.c['seed']}-{self.spec_id}-seeds").sample(range(n), n)
        return [(i, r) for r in range(int(self.c.get("rewrites_per_seed", 2))) for i in order]

    async def make_ideas(self, b: int, bdir: Path, ctx: mp.GenContext, gen: LLMClient,
                         subs: list[dict[str, Any]], doc_types: dict[str, list[str]]) -> list[dict[str, Any]]:
        seeds = self.seeds()
        per = int(self.c["calls_per_batch"])
        items = self.items()[b * per:(b + 1) * per]
        if not items:
            raise CorpusHalt("all seeds and rewrites are used (seeds exhausted)")
        # (a) tag the seeds of this batch (cached across batches)
        domains = [_strip(d) for d in read_jsonl(self.spec_dir / "domains.jsonl")]
        dom_ids = [d["id"] for d in domains]
        tag_model = create_model("SeedTag", species_group=(Literal[SPECIES], ...), setting=(str, ...),  # type: ignore[valid-type]
                                 ai_appears=(Literal["yes", "no", "unnamed"], ...),
                                 closest_section=(Literal[tuple(dom_ids)], ...))  # type: ignore[valid-type]
        tpl_tag = ctx.template(ctx.template_name("s4b_tag"))
        sections = "\n".join(f"- {d['id']}: {d['domain']}" for d in domains)
        tag_path = self.dir / "seeds" / "tags.jsonl"
        uniq = sorted({i for i, _ in items})

        async def tag(seed: dict[str, Any]) -> dict[str, Any]:
            r = tpl_tag.render(sections=sections, source=_trunc(seed["text"], 900))
            res = await gen.complete_full(r.text, stage="s4b_tag", json_schema=tag_model)
            v, diag = mp.greedy_json(res, tag_model)
            return {"id": seed["seed_id"], **v.model_dump(), **diag}

        await self._stage("s4b_tag", [seeds[i] for i in uniq], tag, tag_path, concurrency=self.cfg.concurrency("s4_ideas"))
        self._gate(bdir, "s4b_tag", tag_path, [seeds[i]["seed_id"] for i in uniq])
        tags = {t["id"]: _strip(t) for t in read_jsonl(tag_path)}
        # (b) assignment: the same planned calls as Pilot A, matched to seeds
        apath = bdir / "seed_assign.jsonl"
        if not apath.exists():
            write_jsonl(apath, self._assign(b, items, seeds, tags, subs, doc_types))
        assign = list(read_jsonl(apath))
        # (c) one idea per assignment
        subs_by_id = {s["id"]: s for s in subs}
        tpl_idea = ctx.template(ctx.template_name("s4b_idea"))

        async def idea(a: dict[str, Any]) -> dict[str, Any]:
            sub = subs_by_id[a["subdomain_id"]]
            seed = seeds[a["seed_index"]]
            r = tpl_idea.render(**ctx.identity, behaviour_types=", ".join(BEHAVIOUR_TYPES), subdomain=sub["subdomain"],
                                assertions=mp.bullet(sub["assertions"]), doc_type=a["doc_type"],
                                species_group=a["species_group"], cost_of_care=a["cost_of_care"],
                                welfare_raised=a["welfare_raised"], perspective=a["perspective"],
                                is_story=str(a["is_story"]).lower(), source=_trunc(seed["text"], 1200))
            out = (await gen.complete_json(r.text, stage="s4b_idea", schema=SeedIdea,
                                           seed=int(hashlib.sha256(a["id"].encode()).hexdigest()[:8], 16) & 0x7FFFFFFF))[0]
            rec = {"id": a["id"], "call_id": a["id"], "seed_id": seed["seed_id"], "seed_index": a["seed_index"],
                   "rewrite": a["rewrite"], "subdomain_id": sub["id"], "domain_id": sub["domain_id"],
                   "subdomain": sub["subdomain"], "spec_excerpt": sub["spec_excerpt"], "assertions": sub["assertions"],
                   **{f: a[f] for f in ("doc_type", "species_group", "cost_of_care", "welfare_raised", "perspective", "is_story")},
                   "context_domain": tags.get(seed["seed_id"], {}).get("setting", "from seed"),
                   **out.model_dump(), "source": _trunc(seed["text"], 1200)}
            rec.update(mp.assign_idea_extras(self.cfg, self.spec_id, a["id"]))
            return rec

        ideas_path = bdir / "ideas.jsonl"
        await self._stage("s4b_idea", assign, idea, ideas_path, concurrency=self.cfg.concurrency("s4_ideas"))
        self._gate(bdir, "s4b_idea", ideas_path, [a["id"] for a in assign])
        ideas = [_strip(r) for r in read_jsonl(ideas_path)]
        ideas.sort(key=lambda r: (r["domain_id"], r["subdomain_id"], r["id"]))
        return ideas

    def _assign(self, b: int, items: list[tuple[int, int]], seeds: list[dict[str, Any]], tags: dict[str, Any],
                subs: list[dict[str, Any]], doc_types: dict[str, list[str]]) -> list[dict[str, Any]]:
        import yaml   # noqa: PLC0415
        rng = random.Random(f"{self.c['seed']}-{self.spec_id}-assign-{b}")
        n = len(items)
        q = self.cfg.gen["s4"]["quotas"]
        if self.c.get("section_cap"):
            sub_ids, _ = capped_subdomain_sequence(n, subs, float(self.c["section_cap"]), rng)
        else:
            sub_ids = [s["id"] for s in subs]
            rng.shuffle(sub_ids)
        story_kw = yaml.safe_load((self.cfg.root / "lists/story_doc_types.yaml").read_text())["keywords"]
        calls = plan_calls(n, sub_ids, doc_types, ["from seed"], q, story_kw, rng)
        caps = {}
        if self.c.get("layout_cap"):
            caps["layout"] = cap_layouts(calls, doc_types, float(self.c["layout_cap"]), story_kw, rng)
        dom_of = {s["id"]: s["domain_id"] for s in subs}
        earlier = {}   # seed index -> (domain, doc_type) of its first rewrite
        for p in sorted(self.dir.glob("batch_*/seed_assign.jsonl")):
            for a in read_jsonl(p):
                earlier.setdefault(a["seed_index"], (dom_of.get(a["subdomain_id"]), a["doc_type"]))
        free = list(range(len(calls)))
        out = []
        for (si, r) in items:
            tag = tags.get(seeds[si]["seed_id"], {})
            prev = earlier.get(si) if r > 0 else None

            def score(ci: int) -> float:
                c = calls[ci]
                s = 2.0 * (dom_of[c.subdomain_id] == tag.get("closest_section")) + 1.0 * (c.species_group == tag.get("species_group"))
                if prev and (dom_of[c.subdomain_id] == prev[0] or c.doc_type == prev[1]):
                    s -= 10.0   # a second rewrite takes a different section and document type
                return s
            best = max(free, key=lambda ci: (score(ci), -ci))
            free.remove(best)
            c = calls[best].as_dict()
            out.append({**c, "id": f"seed_{si:05d}_{r}", "seed_index": si, "rewrite": r,
                        "match": {"section": dom_of[c["subdomain_id"]] == tag.get("closest_section"),
                                  "species": c["species_group"] == tag.get("species_group"),
                                  "differs_from_first": (None if not prev else
                                                         dom_of[c["subdomain_id"]] != prev[0] and c["doc_type"] != prev[1])}})
        write_json(self.dir / f"batch_{b:02d}" / "plan_caps.json", caps)
        return out


def _trunc(text: str, words: int) -> str:
    w = text.split()
    return text if len(w) <= words else " ".join(w[:words]) + " [...]"


__all__ = ["SeedCorpus", "decontaminate", "english_like", "load_rows", "prepare_seeds"]
