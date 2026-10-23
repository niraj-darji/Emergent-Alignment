"""Midtraining corpus generation for one spec (§6.3 S1–S9, S12) — the production counterpart of the pilot.

    python scripts/corpus.py --spec-id aw_main --target-tokens 2500000 --deadline 2026-10-14T07:30

* S1–S3 once for the whole spec (all domains, all subdomains).
* Then **batches** until the token target (S12) is reached, a deadline passes, or ``--max-batches``:
  each batch plans ``calls_per_batch`` S4 calls over all subdomains with the §6.3 quotas (fresh seeded plan
  per batch), and runs S4 → S5 → S6 → S7 → S8 (filter judge, template from ``gen.s8.template``).
  The §6.1 stage gate runs after every stage and **halts** the run (doc v1.7.3 [fixed]).
* Tokens are counted on S8-INCLUDE documents after a running 13-gram S9; the final S9 (cosine + 13-gram,
  seeded order) writes ``corpus.jsonl``. Everything is resumable: rerunning the same command continues.

Outputs in ``data/corpora/<spec_id>/``: ``spec/`` (S1–S3), ``batch_NN/`` (stage files + gates),
``corpus.jsonl`` (final documents, no underscore fields), ``summary.json``, ``manifest.json``.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .config import Config
from .gen import midtrain_pipeline as mp
from .gen.dedup import embed, greedy_dedup, seeded_order
from .gen.quotas import check_quotas, plan_calls
from .gen.schemas import S8_CRITERIA
from .io import SRC, append_jsonl, done_ids, read_jsonl, run_stage, sha256_file, side_paths, write_json, write_jsonl
from .llm import LLMClient
from .pilot import PilotError, _strip, git_commit, stage_gate
from .tokens import TokenCounter

log = logging.getLogger(__name__)


class CorpusHalt(RuntimeError):
    pass


CONTENT_ERRORS = ("JSONValidationError", "ValidationError", "ThinkingLeakError")


class _TrimOnly(Exception):
    pass


class CorpusError(RuntimeError):
    """Infrastructure failure (not a quality gate): the run ends with status 'error: ...' and can resume."""


class Corpus:
    def __init__(self, cfg: Config, spec_id: str, out: Path | None = None, http_factory: Any = None):
        self.cfg, self.spec_id = cfg, spec_id
        cc = (cfg.gen.get("corpus") or {})
        self.c = {"calls_per_batch": 100, "k": int(cfg.gen["s4"]["k"]), "keep_per_call": 5, "seed": 20261013,
                  "s9_seed": 9, "overshoot": 0.03, "max_batches": 40, **cc}
        self.dir = out or cfg.root / "data" / "corpora" / spec_id
        self.spec_dir = self.dir / "spec"
        self.http_factory = http_factory

    # ---------------------------------------------------------------- helpers
    def _client(self, role: str) -> LLMClient:
        r = self.cfg.role(role)
        http = self.http_factory(r.base_url) if self.http_factory else None
        return LLMClient(r, self.cfg, max_in_flight=256, http_client=http)

    async def _stage(self, *args: Any, **kw: Any):
        """run_stage, but items that ERRORED (server down, timeouts) end the run as an error, not as a
        stage-gate halt: rerunning resumes them. A gate halt is kept for real quality problems."""
        res = await run_stage(*args, **kw)
        if not res.errors:
            return res
        out = Path(args[3] if len(args) > 3 else kw["out_path"])
        sp = side_paths(out)
        done = done_ids(out)
        unresolved = {}
        for e in read_jsonl(sp["errors"]):
            if e[SRC] not in done:
                unresolved[e[SRC]] = e["error"]
        content = {i: e for i, e in unresolved.items() if e.startswith(CONTENT_ERRORS)}
        if unresolved and len(content) == len(unresolved):
            # the model's output itself failed (e.g. invalid JSON after the §6.1 fallback): not a server problem
            if res.name == "s8_filter" and self.c.get("s8_unresolved", "halt") == "exclude":
                append_jsonl(out, [{"id": f"{i}__filter", "doc_id": i, "verdict": "EXCLUDE", "unresolved": True,
                                    "error": e[:500], "criteria": {c: False for c in S8_CRITERIA}, SRC: i}
                                   for i, e in content.items()])
                log.warning("S8: %d unresolved item(s) recorded as EXCLUDE (corpus.s8_unresolved: exclude): %s",
                            len(content), list(content))
            return res   # otherwise the stage gate halts on the unresolved items (doc §6.1 [fixed])
        raise CorpusError(f"Stage {res.name}: {res.errors} item(s) failed with errors "
                          f"(server down?). Rerun the same command to resume.")

    def _gate(self, bdir: Path, stage: str, out: Path, ids: list[str]) -> None:
        g = stage_gate(self.cfg, stage, out, ids)
        gates = json.loads((bdir / "gates.json").read_text()) if (bdir / "gates.json").exists() else {}
        gates[stage] = {**g, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        write_json(bdir / "gates.json", gates)
        log.info("Gate %s/%s: %s", bdir.name, stage, "PASS" if g["passed"] else f"HALT {g['reasons']}")
        if not g["passed"]:
            raise CorpusHalt(f"Stage gate (doc §6.1) halted at {bdir.name}/{stage}: {g['reasons']} "
                             f"(see {bdir / 'gates.json'}). A decision on the prompt or max_tokens is needed.")

    # ---------------------------------------------------------------- S1–S3
    async def spec_stages(self, ctx: mp.GenContext, client: LLMClient) -> tuple[list, list, dict]:
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        dom_path = self.spec_dir / "domains.jsonl"
        if not dom_path.exists():
            domains, cov = await mp.s1_decompose(ctx, client)
            write_jsonl(dom_path, domains)
            write_json(self.spec_dir / "s1_coverage.json", cov)
        domains = list(read_jsonl(dom_path))
        sub_path = self.spec_dir / "subdomains.jsonl"
        await self._stage("s2_subdomains", domains, mp.s2_fn(ctx, client), sub_path,
                        concurrency=self.cfg.concurrency("s5_draft"))
        self._gate(self.spec_dir, "s2_subdomains", sub_path, [d["id"] for d in domains])
        subs = [_strip(r) for r in read_jsonl(sub_path)]
        dt_path = self.spec_dir / "doctypes.jsonl"
        await self._stage("s3_doctypes", subs, mp.s3_fn(ctx, client), dt_path, concurrency=self.cfg.concurrency("s5_draft"))
        self._gate(self.spec_dir, "s3_doctypes", dt_path, [s["id"] for s in subs])
        doc_types = {r["_src"]: r["doc_types"] for r in read_jsonl(dt_path)}
        return domains, subs, doc_types

    # ---------------------------------------------------------------- one batch
    async def run_batch(self, b: int, ctx: mp.GenContext, gen: LLMClient, judge: LLMClient,
                        subs: list[dict[str, Any]], doc_types: dict[str, list[str]]) -> Path:
        bdir = self.dir / f"batch_{b:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        q = self.cfg.gen["s4"]["quotas"]
        calls_path = bdir / "s4_calls.jsonl"
        if not calls_path.exists():
            rng = random.Random(f"{self.c['seed']}-{self.spec_id}-{b}")
            sub_ids = [s["id"] for s in subs]
            rng.shuffle(sub_ids)
            story_kw = yaml.safe_load((self.cfg.root / "lists/story_doc_types.yaml").read_text())["keywords"]
            ctx_domains = list(yaml.safe_load((self.cfg.root / "lists/train_domains.yaml").read_text()))
            calls = plan_calls(int(self.c["calls_per_batch"]), sub_ids, doc_types, ctx_domains, q, story_kw, rng)
            rows = []
            for cs in calls:
                d = cs.as_dict()
                d["call_id"] = f"call_b{b:02d}{d['call_id'][4:]}"   # unique across batches (seeds derive from it)
                d["id"] = d["call_id"]
                rows.append(d)
            write_jsonl(calls_path, rows)
            write_json(bdir / "plan_quota_check.json", check_quotas(rows, q))
        calls = list(read_jsonl(calls_path))
        subs_by_id = {s["id"]: s for s in subs}
        ideas_path = bdir / "ideas.jsonl"
        await self._stage("s4_ideas", calls, mp.s4_fn(ctx, gen, subs_by_id, int(self.c["k"]),
                                                     need=int(self.c["keep_per_call"])),
                        ideas_path, concurrency=self.cfg.concurrency("s4_ideas"))
        self._gate(bdir, "s4_ideas", ideas_path, [c["id"] for c in calls])
        ideas = [_strip(r) for r in read_jsonl(ideas_path)]
        for i in ideas:
            i.pop("s4_rejected_in_call", None)
        ideas.sort(key=lambda r: (r["domain_id"], r["subdomain_id"], r["id"]))
        by_id = {i["id"]: i for i in ideas}

        await self._stage("s5_draft", ideas, mp.s5_fn(ctx, gen, mp.DocChecks(self.cfg)), bdir / "drafts.jsonl",
                        concurrency=self.cfg.concurrency("s5_draft"))
        self._gate(bdir, "s5_draft", bdir / "drafts.jsonl", [i["id"] for i in ideas])
        drafts = {r["idea_id"]: _strip(r) for r in read_jsonl(bdir / "drafts.jsonl")}
        order = lambda d: [d[i["id"]] for i in ideas if i["id"] in d]   # noqa: E731

        await self._stage("s6_critique", order(drafts), mp.s6_fn(ctx, gen, by_id), bdir / "critiques.jsonl",
                        concurrency=self.cfg.concurrency("s6_critique"))
        self._gate(bdir, "s6_critique", bdir / "critiques.jsonl", [d["id"] for d in drafts.values()])
        crits = {r["idea_id"]: _strip(r) for r in read_jsonl(bdir / "critiques.jsonl")}

        await self._stage("s7_rewrite", order(crits), mp.s7_fn(ctx, gen, by_id, drafts, mp.DocChecks(self.cfg, "s7_rewrite")),
                        bdir / "rewrites.jsonl", concurrency=self.cfg.concurrency("s7_rewrite"))
        self._gate(bdir, "s7_rewrite", bdir / "rewrites.jsonl", [c["id"] for c in crits.values()])
        rewrites = {r["idea_id"]: _strip(r) for r in read_jsonl(bdir / "rewrites.jsonl")}

        await self._stage("s8_filter", order(rewrites), mp.s8_fn(ctx, judge, by_id, "corpus", "filter", False),
                        bdir / "judged.jsonl", concurrency=self.cfg.concurrency("s8_filter"))
        self._gate(bdir, "s8_filter", bdir / "judged.jsonl", [r["id"] for r in rewrites.values()])
        return bdir

    # ---------------------------------------------------------------- accounting
    def included(self) -> list[dict[str, Any]]:
        """S8-INCLUDE documents of all batches, with their idea tags (in batch, idea order)."""
        out = []
        for bdir in sorted(self.dir.glob("batch_*")):
            if not (bdir / "judged.jsonl").exists():
                continue
            ideas = {r["id"]: r for r in map(_strip, read_jsonl(bdir / "ideas.jsonl"))}
            rew = {r["id"]: _strip(r) for r in read_jsonl(bdir / "rewrites.jsonl")}
            for j in read_jsonl(bdir / "judged.jsonl"):
                if j["verdict"] != "INCLUDE" or j["doc_id"] not in rew:
                    continue
                d, idea = rew[j["doc_id"]], ideas.get(rew[j["doc_id"]]["idea_id"], {})
                out.append({"id": d["id"], "batch": bdir.name, "idea_id": d["idea_id"], "text": d["text"],
                            "words": d["words"],
                            **{k: idea.get(k) for k in ("doc_type", "species_group", "context_domain", "cost_of_care",
                                                        "welfare_raised", "perspective", "is_story", "subdomain_id")}})
        return out

    def budget(self, docs: list[dict[str, Any]], counter: TokenCounter, use_embeddings: bool) -> dict[str, Any]:
        ids = [d["id"] for d in docs]
        order = seeded_order(ids, int(self.c["s9_seed"]))
        docs = [docs[i] for i in order]
        emb, note = None, "13-gram"
        if use_embeddings and docs:
            try:
                emb, note = embed([d["text"] for d in docs]), "cosine (all-mpnet-base-v2) + 13-gram"
            except RuntimeError as e:
                note = f"13-gram only ({e})"
        dd = greedy_dedup([d["text"] for d in docs], emb)
        kept = [docs[i] for i in dd.kept]
        for d in kept:
            d.setdefault("tokens", counter.count(d["text"]))
        return {"kept": kept, "removed": len(dd.removed), "method": note,
                "tokens": sum(d["tokens"] for d in kept)}

    # ---------------------------------------------------------------- driver
    async def run(self, target_tokens: int, deadline: dt.datetime | None = None,
                  max_batches: int | None = None, spec_file: Path | None = None,
                  trim_only: bool = False) -> dict[str, Any]:
        self.dir.mkdir(parents=True, exist_ok=True)
        ctx = mp.GenContext.create(self.cfg, self.spec_id, spec_path=spec_file)
        self._check_same_spec(ctx.spec_text)
        if spec_file:   # freeze the spec text used for this corpus next to it
            shutil.copy(spec_file, self.dir / "spec_used.md")
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        counter = TokenCounter((self.cfg.base.get("train_base_model") or "meta-llama/Llama-3.1-8B"))
        t0 = time.time()
        status, b = "target_reached", 0
        max_b = int(max_batches or self.c["max_batches"])
        gen = judge = None
        if trim_only:   # no generation, no server: recount (S12 tokenizer), S9 and trim the existing documents
            status = "trimmed"
        else:
            gen, judge = self._client("generator"), self._client("filter_judge")
        try:
            if trim_only:
                raise _TrimOnly()
            await gen.verify_server()
            if judge.role.base_url != gen.role.base_url:
                await judge.verify_server()
            _, subs, doc_types = await self.spec_stages(ctx, gen)
            while True:
                cur = self.budget(self.included(), counter, use_embeddings=False)
                log.info("Corpus %s: %d documents, %d tokens (target %d) after %d batch(es)",
                         self.spec_id, len(cur["kept"]), cur["tokens"], target_tokens, b)
                need = self.story_need(cur["kept"], target_tokens)
                if need["stories_short"] > 0:
                    log.info("Corpus %s: %d of ~%d stories needed for the S12 story quota", self.spec_id,
                             need["stories"], need["stories_needed"])
                if cur["tokens"] >= target_tokens * (1 + float(self.c["overshoot"])) and need["stories_short"] <= 0:
                    # confirm with the final S9 (cosine + 13-gram), which can remove more than the running check
                    full = self.budget(self.included(), counter, use_embeddings=True)
                    if full["tokens"] >= target_tokens and self.story_need(full["kept"], target_tokens)["stories_short"] <= 0:
                        break
                    log.info("Corpus %s: final S9 leaves %d tokens; generating another batch", self.spec_id, full["tokens"])
                if b >= max_b:
                    status = "max_batches"
                    break
                if deadline and dt.datetime.now() >= deadline:
                    status = "deadline"
                    break
                await self.run_batch(b, ctx, gen, judge, subs, doc_types)
                b += 1
        except _TrimOnly:
            pass
        except CorpusHalt as e:
            status = f"halted: {e}"
        except Exception as e:  # noqa: BLE001 - e.g. the server died: keep what was generated; rerun resumes
            log.exception("Corpus run stopped by an error")
            status = f"error: {type(e).__name__}: {e}"
        finally:
            for c in (gen, judge):
                if c is None:
                    continue
                try:
                    await c.aclose()
                except Exception:  # noqa: BLE001
                    pass
        final = self.budget(self.included(), counter, use_embeddings=True)
        q = self.cfg.gen["s4"]["quotas"]
        corpus, total, depri = self.trim_with_deprioritised(final["kept"], target_tokens)
        if (trim_only or status == "target_reached") and total < target_tokens:
            # the final S9 (cosine) can remove more than the running 13-gram check the loop stopped on
            status = f"short: {total} < {target_tokens} tokens after the final S9; rerun without --trim-only to add batches"
        elif trim_only:
            status = "target_reached"
        write_jsonl(self.dir / "corpus.jsonl", corpus)
        summary = {
            "spec_id": self.spec_id, "status": status, "batches": len(list(self.dir.glob("batch_*"))),
            "documents": len(corpus), "tokens": total, "target_tokens": target_tokens,
            "token_method": counter.method, "s9": {"removed": final["removed"], "method": final["method"]},
            "included_before_s9": len(self.included()),
            "yield": self.yields(), "quota_check_final": check_quotas(corpus, q) if corpus else None,
            "story_share": sum(bool(d["is_story"]) for d in corpus) / len(corpus) if corpus else None,
            "doc_types": dict(Counter(d["doc_type"] for d in corpus).most_common(15)),
            "hours": round((time.time() - t0) / 3600, 2), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "deprioritised": depri, "trim_only": trim_only,
        }
        runs = json.loads((self.dir / "runs.json").read_text()) if (self.dir / "runs.json").exists() else []
        runs.append({"started": started, "finished": summary["finished_at"], "hours": summary["hours"],
                     "status": status})
        write_json(self.dir / "runs.json", runs)
        summary["hours_all_runs"] = round(sum(r["hours"] for r in runs), 2)
        write_json(self.dir / "summary.json", summary)
        prev = json.loads((self.dir / "manifest.json").read_text()) if (self.dir / "manifest.json").exists() else {}
        write_json(self.dir / "manifest.json", {
            "spec_id": self.spec_id, "spec_sha256": sha256_file(spec_file) if spec_file else None,
            "s8_template": (self.cfg.gen.get("s8") or {}).get("template", "E08_filter"),
            "generator": gen.manifest() if gen else prev.get("generator"),
            "filter_judge": judge.manifest() if judge else prev.get("filter_judge"),
            "git_commit": git_commit(self.cfg.root), "token_method": counter.method,
            "corpus_sha256": sha256_file(self.dir / "corpus.jsonl"), "settings": self.c})
        return summary

    # ---------------------------------------------------------------- S12 story quota
    def story_aim(self) -> tuple[float, float]:
        lim = self.cfg.gen["s4"]["quotas"]["limits"]
        lo, hi = float(lim["story_min"]), float(lim["story_max"])
        return min(hi, lo + float(self.c.get("story_margin", 0.01))), hi

    def story_need(self, docs: list[dict[str, Any]], target_tokens: int) -> dict[str, Any]:
        """Stories needed so that the trimmed corpus can meet the story minimum (S8 passes fewer stories)."""
        aim, _ = self.story_aim()
        n_st = sum(bool(d.get("is_story")) for d in docs)
        if not docs:
            return {"stories": 0, "stories_needed": 0, "stories_short": 1}
        mean = sum(d["tokens"] for d in docs) / len(docs)
        needed = int(aim * target_tokens / mean + 0.999)
        return {"stories": n_st, "stories_needed": needed, "stories_short": needed - n_st}

    def stratified_trim(self, kept: list[dict[str, Any]], target_tokens: int) -> tuple[list[dict[str, Any]], int]:
        """S12: fill the token budget in S9's seeded order, keeping the story share at
        max(story_min + margin, natural share), capped at story_max (doc §6.3 S12: stratified sampling)."""
        aim, hi = self.story_aim()
        stories = [d for d in kept if d.get("is_story")]
        others = [d for d in kept if not d.get("is_story")]
        natural = len(stories) / len(kept) if kept else 0.0
        p = min(hi, max(aim, natural))
        corpus, total, i, j = [], 0, 0, 0
        while total < target_tokens and (i < len(stories) or j < len(others)):
            want_story = (i < len(stories)) and (len(corpus) == 0 or i < p * (len(corpus) + 1) or j >= len(others))
            d = stories[i] if want_story else others[j]
            i, j = (i + 1, j) if want_story else (i, j + 1)
            corpus.append(d)
            total += d["tokens"]
        return corpus, total

    @staticmethod
    def _norm_text(t: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[\u2018\u2019]", "'", t)).lower()

    def trim_with_deprioritised(self, kept: list[dict[str, Any]], target_tokens: int
                                ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """S12 trim that removes documents containing ``corpus.deprioritise_phrases`` as far as the §6.3 quotas
        allow (24 Oct, Niraj: the two verbatim spec §1 sentences).

        Starts from the plain stratified trim, then swaps each phrase document for an unused phrase-free
        candidate, most similar first (same story flag, species, welfare_raised, cost of care; then story and
        species; then story only). A swap is kept only if every quota still passes and the token total stays
        within [target, target x 1.02]. Phrase documents that cannot be swapped stay, and are counted."""
        phrases = [self._norm_text(x) for x in (self.c.get("deprioritise_phrases") or [])]
        q = self.cfg.gen["s4"]["quotas"]
        c, t = self.stratified_trim(kept, target_tokens)
        if not phrases:
            return c, t, {"phrases": []}
        hit = {d["id"]: [k for k, ph in enumerate(phrases) if ph in self._norm_text(d["text"])] for d in kept}
        before = sum(bool(hit[d["id"]]) for d in c)
        quotas_ok = bool(c) and check_quotas(c, q)["all_pass"]
        swapped = 0
        if quotas_ok:
            used = {d["id"] for d in c}
            spare = [d for d in kept if d["id"] not in used and not hit[d["id"]]]
            keys = (lambda d: (d.get("is_story"), d.get("species_group"), d.get("welfare_raised"), d.get("cost_of_care")),
                    lambda d: (d.get("is_story"), d.get("species_group")),
                    lambda d: (d.get("is_story"),))
            hi = target_tokens * 1.02
            for pos in [k for k in range(len(c) - 1, -1, -1) if hit[c[k]["id"]]]:
                x = c[pos]
                done = False
                for key in keys:
                    for si, y in enumerate(spare):
                        if key(y) != key(x):
                            continue
                        nt = t - x["tokens"] + y["tokens"]
                        if not (target_tokens <= nt <= hi):
                            continue
                        c[pos] = y
                        if check_quotas(c, q)["all_pass"]:
                            t = nt
                            spare.pop(si)
                            swapped += 1
                            done = True
                            break
                        c[pos] = x
                    if done:
                        break
        ids = {d["id"] for d in c}
        info = {"phrases": self.c.get("deprioritise_phrases"), "method": "quota-checked swaps",
                "quotas_ok_before_swaps": quotas_ok,
                "candidates_with_phrase": sum(bool(v) for v in hit.values()), "candidates": len(kept),
                "corpus_docs_with_phrase_before": before, "swapped_out": swapped,
                "corpus_docs_with_phrase": sum(bool(hit[i]) for i in ids),
                "corpus_docs_per_phrase": [sum(k in hit[i] for i in ids) for k in range(len(phrases))],
                "fallback_to_plain_order": False}
        log.info("S12 deprioritised phrases: %d of %d candidates contain one; %d in the plain trim, %d swapped out, "
                 "%d remain", info["candidates_with_phrase"], len(kept), before, swapped, info["corpus_docs_with_phrase"])
        return c, t, info

    def _check_same_spec(self, spec_text: str) -> None:
        """Resuming is only allowed with the spec the S1-S3 outputs were made from (fresh S1-S4 per spec)."""
        h = hashlib.sha256(spec_text.encode()).hexdigest()
        f = self.dir / "spec_text_sha256.txt"
        if f.exists():
            if f.read_text().strip() != h:
                raise PilotError(f"{self.dir} was generated from a different spec text; use a new --out directory.")
        elif (self.spec_dir / "domains.jsonl").exists():
            raise PilotError(f"{self.dir} has S1 outputs from an unrecorded spec; use a new --out directory.")
        else:
            f.write_text(h + "\n")

    def yields(self) -> dict[str, Any]:
        n = Counter()
        for bdir in sorted(self.dir.glob("batch_*")):
            for name, f in (("ideas", "ideas.jsonl"), ("drafts", "drafts.jsonl"), ("rewrites", "rewrites.jsonl")):
                n[name] += sum(1 for _ in read_jsonl(bdir / f)) if (bdir / f).exists() else 0
            if (bdir / "judged.jsonl").exists():
                js = list(read_jsonl(bdir / "judged.jsonl"))
                n["judged"] += len(js)
                n["include"] += sum(j["verdict"] == "INCLUDE" for j in js)
                n["s8_unresolved_excluded"] += sum(bool(j.get("unresolved")) for j in js)
                for j in js:
                    for c, ok in j["criteria"].items():
                        n[f"s8_fail:{c}"] += not ok
        out = dict(n)
        if n["judged"]:
            out["s8_pass_rate"] = n["include"] / n["judged"]
        return out


def parse_deadline(s: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(s) if s else None


__all__ = ["Corpus", "CorpusHalt", "PilotError", "parse_deadline"]
