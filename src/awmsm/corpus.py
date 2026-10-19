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
from .io import read_jsonl, run_stage, sha256_file, write_json, write_jsonl
from .llm import LLMClient
from .pilot import PilotError, _strip, git_commit, stage_gate
from .tokens import TokenCounter

log = logging.getLogger(__name__)


class CorpusHalt(RuntimeError):
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
        if res.errors:
            raise CorpusError(f"Stage {res.name}: {res.errors} item(s) failed with errors "
                              f"(server down?). Rerun the same command to resume.")
        return res

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
                  max_batches: int | None = None, spec_file: Path | None = None) -> dict[str, Any]:
        self.dir.mkdir(parents=True, exist_ok=True)
        ctx = mp.GenContext.create(self.cfg, self.spec_id, spec_path=spec_file)
        self._check_same_spec(ctx.spec_text)
        if spec_file:   # freeze the spec text used for this corpus next to it
            shutil.copy(spec_file, self.dir / "spec_used.md")
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        counter = TokenCounter((self.cfg.base.get("train_base_model") or "meta-llama/Llama-3.1-8B"))
        gen, judge = self._client("generator"), self._client("filter_judge")
        t0 = time.time()
        status, b = "target_reached", 0
        max_b = int(max_batches or self.c["max_batches"])
        try:
            await gen.verify_server()
            if judge.role.base_url != gen.role.base_url:
                await judge.verify_server()
            _, subs, doc_types = await self.spec_stages(ctx, gen)
            while True:
                cur = self.budget(self.included(), counter, use_embeddings=False)
                log.info("Corpus %s: %d documents, %d tokens (target %d) after %d batch(es)",
                         self.spec_id, len(cur["kept"]), cur["tokens"], target_tokens, b)
                if cur["tokens"] >= target_tokens * (1 + float(self.c["overshoot"])):
                    break
                if b >= max_b:
                    status = "max_batches"
                    break
                if deadline and dt.datetime.now() >= deadline:
                    status = "deadline"
                    break
                await self.run_batch(b, ctx, gen, judge, subs, doc_types)
                b += 1
        except CorpusHalt as e:
            status = f"halted: {e}"
        except Exception as e:  # noqa: BLE001 - e.g. the server died: keep what was generated; rerun resumes
            log.exception("Corpus run stopped by an error")
            status = f"error: {type(e).__name__}: {e}"
        finally:
            for c in (gen, judge):
                try:
                    await c.aclose()
                except Exception:  # noqa: BLE001
                    pass
        final = self.budget(self.included(), counter, use_embeddings=True)
        # trim to the target in seeded order (documents are already in S9's seeded order)
        corpus, total = [], 0
        for d in final["kept"]:
            if total >= target_tokens:
                break
            corpus.append(d)
            total += d["tokens"]
        write_jsonl(self.dir / "corpus.jsonl", corpus)
        q = self.cfg.gen["s4"]["quotas"]
        summary = {
            "spec_id": self.spec_id, "status": status, "batches": len(list(self.dir.glob("batch_*"))),
            "documents": len(corpus), "tokens": total, "target_tokens": target_tokens,
            "token_method": counter.method, "s9": {"removed": final["removed"], "method": final["method"]},
            "included_before_s9": len(self.included()),
            "yield": self.yields(), "quota_check_final": check_quotas(corpus, q) if corpus else None,
            "story_share": sum(bool(d["is_story"]) for d in corpus) / len(corpus) if corpus else None,
            "doc_types": dict(Counter(d["doc_type"] for d in corpus).most_common(15)),
            "hours": round((time.time() - t0) / 3600, 2), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        runs = json.loads((self.dir / "runs.json").read_text()) if (self.dir / "runs.json").exists() else []
        runs.append({"started": started, "finished": summary["finished_at"], "hours": summary["hours"],
                     "status": status})
        write_json(self.dir / "runs.json", runs)
        summary["hours_all_runs"] = round(sum(r["hours"] for r in runs), 2)
        write_json(self.dir / "summary.json", summary)
        write_json(self.dir / "manifest.json", {
            "spec_id": self.spec_id, "spec_sha256": sha256_file(spec_file) if spec_file else None,
            "s8_template": (self.cfg.gen.get("s8") or {}).get("template", "E08_filter"),
            "generator": gen.manifest(), "filter_judge": judge.manifest(), "git_commit": git_commit(self.cfg.root),
            "corpus_sha256": sha256_file(self.dir / "corpus.jsonl"), "settings": self.c})
        return summary

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
