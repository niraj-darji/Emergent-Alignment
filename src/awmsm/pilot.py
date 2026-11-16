"""Generator-selection pilot (Appendix I.1).

Phases (run one model at a time on wtc11; each phase checks the server serves the right model):

  ideas              candidate A: S1–S4 once on the pilot spec, freeze 200 stratified ideas
  write  --cand X    candidate X: S5 draft → S6 critique → S7 rewrite on all frozen ideas (its own
                     sampling, fixed concurrency 32); saves real prompts for the official benchmark
  judge  --judge J --writers X Y   judge J: S8 on the writers' rewrites (2 × 2 writer × judge table)
  metrics            pass rates, near-duplicate rate (S9), language-mixing and other rejection rates,
                     fallback rates, throughput (from each candidate's official benchmark)
  audit-export       blinded human audit: 25 idea IDs × every candidate, shuffled, writer hidden
  report             data/pilot/report.md; names a winner only under the pre-registered decision rule

All phases are resumable: rerunning skips finished items (``io.run_stage``) and LLM calls are cached.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import Config, Role
from . import audit_review as ar
from .gen import midtrain_pipeline as mp
from .gen.dedup import embed, greedy_dedup, seeded_order
from .gen.quotas import check_quotas, plan_calls
from .gen.schemas import S8_CRITERIA
from .io import read_jsonl, run_stage, sha256_file, side_paths, write_json, write_jsonl
from .llm import LLMClient

log = logging.getLogger(__name__)


class PilotError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- setup
class Pilot:
    def __init__(self, cfg: Config, http_factory: Any = None):
        self.cfg = cfg
        self.http_factory = http_factory     # tests: callable(base_url) -> httpx.AsyncClient
        self.p = cfg.gen["pilot"]
        self.dir = cfg.root / self.p["dir"]
        self.spec_dir = self.dir / "spec"
        self.ideas_path = self.dir / "ideas.jsonl"
        self.conc = int(self.p["concurrency"])
        if int(self.p["s4_calls"]) * int(self.p["keep_per_call"]) != int(self.p["n_ideas"]):
            raise PilotError(f"pilot.s4_calls x keep_per_call must equal n_ideas ({self.p['n_ideas']})")
        if int(self.p["keep_per_call"]) > int(cfg.gen["s4"]["k"]):
            raise PilotError("pilot.keep_per_call cannot exceed s4.k")

    def cand_dir(self, cand: str) -> Path:
        return self.dir / cand

    def client(self, cand: str) -> LLMClient:
        c = self.p["candidates"][cand]
        role = Role(name=f"pilot_{cand}", base_url=c["base_url"], model=self.cfg.models[c["model"]])
        http = self.http_factory(c["base_url"]) if self.http_factory else None
        return LLMClient(role, self.cfg, max_in_flight=max(self.conc, 32), http_client=http)

    async def connect(self, cand: str) -> LLMClient:
        client = self.client(cand)
        try:
            await client.verify_server()
        except Exception as e:
            await client.aclose()
            raise PilotError(f"Candidate {cand} ({self.p['candidates'][cand]['model']}) is not being served "
                             f"at {self.p['candidates'][cand]['base_url']}: {e}") from e
        return client

    def ctx(self, bench_dir: Path | None = None) -> mp.GenContext:
        return mp.GenContext.create(self.cfg, self.p["spec_id"], bench_dir)

    def load_ideas(self) -> list[dict[str, Any]]:
        if not self.ideas_path.exists():
            raise PilotError("No frozen ideas yet: run the `ideas` phase first (candidate A served).")
        want = (self.dir / "ideas.sha256").read_text().split()[0]
        if sha256_file(self.ideas_path) != want:
            raise PilotError("data/pilot/ideas.jsonl changed after freezing (sha256 mismatch).")
        return list(read_jsonl(self.ideas_path))

    # -------------------------------------------------------------------- ideas
    async def phase_ideas(self, force: bool = False) -> dict[str, Any]:
        if self.ideas_path.exists() and not force:
            raise PilotError("Ideas are already frozen (data/pilot/ideas.jsonl). Use --force only if no "
                             "candidate has started writing.")
        if force and any(self.cand_dir(c).exists() for c in self.p["candidates"]):
            raise PilotError("Refusing --force: candidate outputs already exist; the ideas must stay identical.")
        if force:   # keep S1-S3 (domains, subdomains, doc types); redo the S4 plan and S4 ideas
            old = self.dir / f"superseded_{time.strftime('%Y%m%dT%H%M%S')}"
            old.mkdir(parents=True)
            for f in [self.ideas_path, self.dir / "ideas.sha256", self.dir / "ideas_summary.json",
                      *self.spec_dir.glob("s4_calls*"), *self.spec_dir.glob("ideas_raw*")]:
                if f.exists():
                    f.rename(old / f.name)
            log.info("Previous S4 plan and ideas moved to %s", old)
        cand = "A"
        ctx = self.ctx()
        rng = random.Random(int(self.p["seed"]))
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        async with await self.connect(cand) as client:
            # S1 -----------------------------------------------------------------
            dom_path = self.spec_dir / "domains.jsonl"
            if not dom_path.exists():
                domains, cov = await mp.s1_decompose(ctx, client)
                write_jsonl(dom_path, domains)
                write_json(self.spec_dir / "s1_coverage.json", cov)
                log.info("S1: %d domains; spec coverage %.0f%%, excerpt sentences not in spec %.0f%%",
                         len(domains), 100 * cov["spec_sentences_covered"], 100 * cov["excerpt_sentences_not_in_spec"])
            domains = list(read_jsonl(dom_path))
            # S2 -----------------------------------------------------------------
            sub_path = self.spec_dir / "subdomains.jsonl"
            await run_stage("s2_subdomains", domains, mp.s2_fn(ctx, client), sub_path, concurrency=self.conc)
            subs = [_strip(r) for r in read_jsonl(sub_path)]
            # choose subdomains: round-robin over domains so every domain is represented
            by_dom: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for s in subs:
                by_dom[s["domain_id"]].append(s)
            chosen, i = [], 0
            while len(chosen) < min(int(self.p["subdomains_used"]), len(subs)):
                for d in sorted(by_dom):
                    if i < len(by_dom[d]) and len(chosen) < int(self.p["subdomains_used"]):
                        chosen.append(by_dom[d][i])
                i += 1
            # S3 -----------------------------------------------------------------
            dt_path = self.spec_dir / "doctypes.jsonl"
            await run_stage("s3_doctypes", chosen, mp.s3_fn(ctx, client), dt_path, concurrency=self.conc)
            doc_types = {r["_src"]: r["doc_types"] for r in read_jsonl(dt_path)}
            # S4 plan (frozen once written) -----------------------------------------
            calls_path = self.spec_dir / "s4_calls.jsonl"
            if not calls_path.exists():
                import yaml
                story_kw = yaml.safe_load((self.cfg.root / "lists/story_doc_types.yaml").read_text())["keywords"]
                ctx_domains = list(yaml.safe_load((self.cfg.root / "lists/train_domains.yaml").read_text()))
                calls = plan_calls(int(self.p["s4_calls"]), [s["id"] for s in chosen], doc_types, ctx_domains,
                                   self.cfg.gen["s4"]["quotas"], story_kw, rng)
                write_jsonl(calls_path, [c.as_dict() for c in calls])
            calls = list(read_jsonl(calls_path))
            for c in calls:
                c["id"] = c["call_id"]
            plan_check = check_quotas(calls, self.cfg.gen["s4"]["quotas"])
            # S4 -----------------------------------------------------------------
            raw_path = self.spec_dir / "ideas_raw.jsonl"
            subs_by_id = {s["id"]: s for s in subs}
            await run_stage("s4_ideas", calls, mp.s4_fn(ctx, client, subs_by_id, int(self.cfg.gen["s4"]["k"]),
                                                        need=int(self.p["keep_per_call"])),
                            raw_path, concurrency=self.conc)
            manifest = client.manifest()
        # select keep_per_call per call (seeded), order by domain/subdomain for prefix grouping
        by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in read_jsonl(raw_path):
            by_call[r["call_id"]].append(_strip(r))
        keep = int(self.p["keep_per_call"])
        ideas = []
        short = []
        for cid in sorted(by_call):
            got = by_call[cid]
            if len(got) < keep:
                short.append(cid)
            ideas += random.Random(f"{self.p['seed']}-{cid}").sample(got, min(keep, len(got)))
        ideas.sort(key=lambda r: (r["domain_id"], r["subdomain_id"], r["id"]))
        idea_check = check_quotas(ideas, self.cfg.gen["s4"]["quotas"])
        write_jsonl(self.ideas_path, ideas)
        (self.dir / "ideas.sha256").write_text(f"{sha256_file(self.ideas_path)}  ideas.jsonl\n")
        s4_rej = Counter()
        for cid, got in by_call.items():
            for x in (got[0].get("s4_rejected_in_call") or []) if got else []:
                s4_rej[x["reason"].split(":")[0]] += 1
        summary = {
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_ideas": len(ideas), "target": self.p["n_ideas"],
            "s4_rejected_ideas": dict(s4_rej),
            "story_doc_types": sorted({i["doc_type"] for i in ideas if i["is_story"]}),
            "not_raised_cost_of_care": dict(Counter(i["cost_of_care"] for i in ideas
                                                    if i["welfare_raised"] == "appropriately_not_raised")),
            "calls_short_of_ideas": short, "domains": len(domains), "subdomains_total": len(subs),
            "subdomains_used": len(chosen), "s1_coverage": _read_json(self.spec_dir / "s1_coverage.json"),
            "plan_quota_check": plan_check, "idea_quota_check": idea_check,
            "generator_manifest": manifest, "git_commit": git_commit(self.cfg.root),
        }
        write_json(self.dir / "ideas_summary.json", summary)
        if len(ideas) != int(self.p["n_ideas"]):
            log.warning("Froze %d ideas, target %d (calls short: %s)", len(ideas), self.p["n_ideas"], short)
        return summary

    # -------------------------------------------------------------------- write
    async def phase_write(self, cand: str) -> dict[str, Any]:
        ideas = self.load_ideas()
        idea_by_id = {i["id"]: i for i in ideas}
        cdir = self.cand_dir(cand)
        ctx = self.ctx(bench_dir=cdir / "bench_prompts")
        checks = mp.DocChecks(self.cfg)
        tpl_sha = mp.template_sha(ctx.template("E05_draft"), ctx.requirements_block)
        self._check_draft_template(cdir, tpl_sha)
        self._require_amendments_registered()   # Amendment 1: registered before regenerated outputs exist
        t: dict[str, float] = {}
        async with await self.connect(cand) as client:
            t0 = time.monotonic()
            await run_stage("s5_draft", ideas, mp.s5_fn(ctx, client, checks), cdir / "drafts.jsonl",
                            concurrency=self.conc)
            t["s5_draft"] = time.monotonic() - t0
            lim = self.cfg.gen["s4"]["quotas"]["limits"]
            story_limits = (float(lim["story_min"]), float(lim["story_max"]))
            s5 = s5_summary(cdir, [i["id"] for i in ideas], idea_by_id, story_limits)
            write_json(cdir / "s5_summary.json", s5)
            log.info("S5 summary: %s", json.dumps({k: v for k, v in s5.items() if k != "unresolved_ids"}))
            gates = cdir / "gates.json"
            self._gate(gates, "s5_draft", "s5_draft", cdir / "drafts.jsonl", [i["id"] for i in ideas], "S6")
            # doc v1.7.2: truncations are ordinary rejections; the §6.1 stage gate covers unresolved items
            drafts = _index(cdir / "drafts.jsonl")
            t0 = time.monotonic()
            await run_stage("s6_critique", _ordered(drafts, ideas), mp.s6_fn(ctx, client, idea_by_id),
                            cdir / "critiques.jsonl", concurrency=self.conc)
            t["s6_critique"] = time.monotonic() - t0
            s6 = critique_summary(cdir / "critiques.jsonl")
            write_json(cdir / "s6_summary.json", s6)
            self._gate(gates, "s6_critique", "s6_critique", cdir / "critiques.jsonl",
                       [r["id"] for r in drafts.values()], "S7")
            crits = _index(cdir / "critiques.jsonl")
            t0 = time.monotonic()
            await run_stage("s7_rewrite", _ordered(crits, ideas), mp.s7_fn(ctx, client, idea_by_id, drafts,
                                                                           mp.DocChecks(self.cfg, "s7_rewrite")),
                            cdir / "rewrites.jsonl", concurrency=self.conc)
            t["s7_rewrite"] = time.monotonic() - t0
            s7 = stage_summary(cdir / "rewrites.jsonl", [r["id"] for r in crits.values()], idea_by_id, story_limits)
            write_json(cdir / "s7_summary.json", s7)
            self._gate(gates, "s7_rewrite", "s7_rewrite", cdir / "rewrites.jsonl",
                       [r["id"] for r in crits.values()], "S8 (judge)")
            log.info("S7 summary: %s", json.dumps({k: v for k, v in s7.items() if k not in ("unresolved_ids", "repetition_examples")}))
            man = {"candidate": cand, "manifest": client.manifest(), "stage_seconds_this_run": t,
               "s5_template_sha": tpl_sha, "s5_summary": s5, "s6_summary": s6, "s7_summary": s7,
               "gates": _read_json(gates),
               # each stage's inputs are the previous stage's kept outputs
               "coverage": {"s5_draft": _coverage(cdir / "drafts.jsonl", [i["id"] for i in ideas]),
                            "s6_critique": _coverage(cdir / "critiques.jsonl", [r["id"] for r in drafts.values()]),
                            "s7_rewrite": _coverage(cdir / "rewrites.jsonl", [r["id"] for r in crits.values()])},
                   "concurrency": self.conc, "ideas_sha256": sha256_file(self.ideas_path),
                   "git_commit": git_commit(self.cfg.root), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        write_json(cdir / "manifest_write.json", man)
        return man

    # -------------------------------------------------------------------- judge
    async def phase_judge(self, judge: str, writers: list[str]) -> dict[str, Any]:
        ideas = {i["id"]: i for i in self.load_ideas()}
        out: dict[str, Any] = {}
        async with await self.connect(judge) as client:
            for w in writers:
                self._require_gates(self.cand_dir(w) / "gates.json", ["s5_draft", "s6_critique", "s7_rewrite"],
                                    f"Cannot judge {w}'s rewrites")
                rewrites = _index(self.cand_dir(w) / "rewrites.jsonl")
                if not rewrites:
                    raise PilotError(f"No rewrites for candidate {w}: run `write --cand {w}` first.")
                self_judge = w == judge   # production S8 workload = a candidate judging its own documents
                ctx = self.ctx(bench_dir=self.cand_dir(judge) / "bench_prompts" if self_judge else None)
                path = self.dir / "judged" / f"{w}_by_{judge}.jsonl"
                t0 = time.monotonic()
                await run_stage(f"s8_{w}_by_{judge}", _ordered(rewrites, list(ideas.values())),
                                mp.s8_fn(ctx, client, ideas, w, judge, self_judge), path, concurrency=self.conc)
                out[f"{w}_by_{judge}"] = {"seconds": time.monotonic() - t0}
                out[f"{w}_by_{judge}"]["gate"] = self._gate(
                    self.dir / "judged" / "gates.json", f"{w}_by_{judge}", "s8_filter", path,
                    [r["id"] for r in rewrites.values()], "Metrics / further judging")
            out["manifest"] = client.manifest()
        write_json(self.dir / "judged" / f"manifest_judge_{judge}.json", out)
        return out

    def _gate(self, gates_path: Path, key: str, stage: str, out_path: Path, input_ids: list[str],
              next_step: str) -> dict[str, Any]:
        g = stage_gate(self.cfg, stage, out_path, input_ids)
        _record_gate(gates_path, key, g)
        extra = (f"; T=0 fallback {g['t0_fallbacks']}/{g['t0_calls']} = {g['t0_fallback_rate']:.1%}"
                 if g["greedy"] else "")
        log.info("Stage gate %s: %s (unresolved %d%s)", key, "PASS" if g["passed"] else "HALT", g["unresolved"], extra)
        if not g["passed"]:
            raise PilotError(f"Stage gate (doc §6.1) halted after {key}: {'; '.join(g['reasons'])}. {next_step} not run. "
                             f"A decision on the prompt or max_tokens is needed (recorded in the doc) before rerunning. "
                             f"Details: {gates_path}")
        return g

    def _require_gates(self, gates_path: Path, keys: list[str], what: str) -> None:
        gates = _read_json(gates_path) or {}
        bad = [k for k in keys if not (gates.get(k) or {}).get("passed")]
        if bad:
            raise PilotError(f"{what}: stage gate not passed for {bad} (see {gates_path}).")

    @staticmethod
    def _check_draft_template(cdir: Path, tpl_sha: str) -> None:
        """Refuse to resume S5 over drafts made with a different E.5 wording (both candidates must use
        the identical prompt). Old outputs have to be moved aside first."""
        sp = side_paths(cdir / "drafts.jsonl")
        old = {r.get("template_sha") for r in read_jsonl(sp["out"])}
        old |= {(r.get("detail") or {}).get("template_sha") for r in read_jsonl(sp["rejected"])}
        stale = old - {tpl_sha}
        if stale:
            raise PilotError(
                f"{cdir} has S5 outputs from another E.5 template (template_sha {sorted(map(str, stale))}, "
                f"current {tpl_sha}). Move the folder aside before rerunning, e.g.\n"
                f"  mv {cdir} {cdir.parent / ('superseded_' + cdir.name + '_' + time.strftime('%Y%m%d_%H%M'))}")

    # -------------------------------------------------------------------- metrics
    def phase_metrics(self, use_embeddings: bool = True) -> dict[str, Any]:
        ideas = self.load_ideas()
        n_ideas = len(ideas)
        cands = [c for c in self.p["candidates"] if (self.cand_dir(c) / "rewrites.jsonl").exists()]
        for c in cands:
            self._require_gates(self.cand_dir(c) / "gates.json", ["s5_draft", "s6_critique", "s7_rewrite"],
                                f"Metrics for {c}")
        self._require_gates(self.dir / "judged" / "gates.json", [f.stem for f in _judged_files(self.dir)],
                            "Metrics (S8 judging)")
        m: dict[str, Any] = {"n_ideas": n_ideas, "candidates": {}, "pass_rates": {}, "concurrency": self.conc}
        for c in cands:
            cd = self.cand_dir(c)
            rej = Counter()
            attempts = {"s5_draft": 0, "s7_rewrite": 0}
            for stage, fname in (("s5_draft", "drafts"), ("s7_rewrite", "rewrites")):
                sp = side_paths(cd / f"{fname}.jsonl")
                kept = sum(1 for _ in read_jsonl(sp["out"]))
                rj = [r["reason"] for r in read_jsonl(sp["rejected"])]
                attempts[stage] = kept + len(rj)
                rej.update(f"{stage}:{r}" for r in rj)
            lm = sum(v for k, v in rej.items() if k.endswith(":language_mixing"))
            lim = self.cfg.gen["s4"]["quotas"]["limits"]
            story_limits = (float(lim["story_min"]), float(lim["story_max"]))
            ideas_by_id = {i["id"]: i for i in ideas}
            s5 = s5_summary(cd, [i["id"] for i in ideas], ideas_by_id, story_limits)   # doc v1.7.1/v1.7.2 S5
            crit_ids = [r["id"] for r in read_jsonl(cd / "critiques.jsonl")]
            s7 = stage_summary(cd / "rewrites.jsonl", crit_ids, ideas_by_id, story_limits)
            rewrites = [_strip(r) for r in read_jsonl(cd / "rewrites.jsonl")]
            order = seeded_order([r["idea_id"] for r in rewrites], int(self.p["s9_seed"]))   # §6.3 S9
            rewrites = [rewrites[i] for i in order]
            texts = [r["text"] for r in rewrites]
            emb, emb_note = None, "cosine (all-mpnet-base-v2) + 13-gram"
            if use_embeddings and texts:
                try:
                    emb = embed(texts)
                except RuntimeError as e:
                    emb_note = f"13-gram only ({e})"
            dd = greedy_dedup(texts, emb)
            wman = _read_json(cd / "manifest_write.json") or {}
            words = [r["words"] for r in rewrites]
            m["candidates"][c] = {
                "model": self.p["candidates"][c]["model"],
                "drafts_kept": attempts["s5_draft"] - sum(v for k, v in rej.items() if k.startswith("s5_draft:")),
                "rewrites_kept": len(rewrites), "end_to_end_documents": len(rewrites),
                "rejections": dict(rej),
                "language_mixing_rejection_rate": lm / max(1, attempts["s5_draft"] + attempts["s7_rewrite"]),
                "s5_think_slip": {"count": s5["think_slip"], "of": s5["responses"],
                                  "rate": s5["think_slip_rate"] if s5["responses"] else None},
                "s5_truncation_rate": s5["truncation_rate"],
                "s5_plan_words": s5["plan_words"],
                "s7_words_all_parsed": s7["words_all_parsed"], "s7_words_kept": s7["words_kept"],
                "s7_template_shas": s7["template_shas"],
                "by_stage": {st: {k: v for k, v in sm.items() if k in (
                    "responses", "kept", "keep_rate", "rejections", "tolerances_kept", "tolerances_all_responses",
                    "truncation_rate", "degenerate_repetition", "degenerate_repetition_rate", "story_share")}
                    for st, sm in (("s5_draft", s5), ("s7_rewrite", s7))},
                "s5_template_sha": wman.get("s5_template_sha"),
                "near_duplicate_rate": dd.removal_rate, "near_duplicate_method": emb_note,
                "near_duplicates": [{**x, "idea_id": rewrites[x["index"]]["idea_id"]} for x in dd.removed],
                "s9_order_seed": int(self.p["s9_seed"]),
                "rewrite_words": {"mean": sum(words) / len(words), "min": min(words), "max": max(words)} if words else None,
                "t0_fallbacks": (wman.get("manifest") or {}).get("t0_fallback_report"),
                "vendor_temperature_flags": ((wman.get("manifest") or {}).get("flags") or {}).get(
                    "stages_outside_vendor_temperature_range"),
                "stage_seconds": wman.get("stage_seconds_this_run"),
                "throughput": _throughput(cd / "throughput_benchmark.json"),
            }
        for f in _judged_files(self.dir):
            w, j = f.stem.split("_by_")
            recs = list(read_jsonl(f))
            n = len(recs)
            inc = sum(r["verdict"] == "INCLUDE" for r in recs)
            m["pass_rates"][f.stem] = {
                "writer": w, "judge": j, "judged": n, "include": inc, "pass_rate": inc / n if n else None,
                "wilson95": wilson(inc, n), "yield_vs_ideas": inc / n_ideas,
                "criterion_fail_rates": {k: sum(not r["criteria"][k] for r in recs) / n for k in S8_CRITERIA} if n else {},
                "model_verdict_inconsistent": sum(not r["consistent"] for r in recs),
            }
        for c in cands:   # mean pass rate across both judges (tie-break, Appendix I.1 step 5)
            rates = [v["pass_rate"] for v in m["pass_rates"].values() if v["writer"] == c and v["pass_rate"] is not None]
            m["candidates"][c]["mean_pass_rate_across_judges"] = sum(rates) / len(rates) if rates else None
            m["candidates"][c]["judges"] = sorted(v["judge"] for v in m["pass_rates"].values() if v["writer"] == c)
        m["computed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(self.dir / "metrics.json", m)
        return m

    # -------------------------------------------------------------------- rule
    @property
    def rule_path(self) -> Path:
        return self.dir / "decision_rule.registered.json"

    def phase_register_rule(self) -> dict[str, Any]:
        """Appendix I.1 step 5: commit the decision rule, with a timestamp, before any labelling.
        Amendments (``decision_rule.amendments``) are registered one by one, each with its own timestamp
        and hash; the base rule and already-registered amendments can never change."""
        full = self.p.get("decision_rule")
        if not full:
            raise PilotError("configs/gen.yaml pilot.decision_rule is empty.")
        rule, amendments = _split_rule(full)
        validate_rule(rule)
        h = sha256_text(json.dumps(rule, sort_keys=True))
        if self.rule_path.exists():
            reg = json.loads(self.rule_path.read_text())
            if reg["sha256"] != h:
                raise PilotError(f"A different rule was already registered at {reg['registered_at']}; "
                                 "a registered rule cannot be changed.")
            return self._register_amendments(reg, amendments)
        lab = self.dir / "audit" / "audit_labels.csv"
        if lab.exists() and any(r.get("overall_quality", "").strip() for r in csv.DictReader(lab.open())):
            raise PilotError("Audit labels already exist; the rule must be registered before labelling.")
        reg = {"rule": rule, "sha256": h, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "registered_at_unix": time.time(), "git_commit": git_commit(self.cfg.root),
               "doc_version": "AW_MSM_Methodology v1.7, Appendix I.1 step 5"}
        write_json(self.rule_path, reg)
        return self._register_amendments(reg, amendments)

    def _register_amendments(self, reg: dict[str, Any], amendments: list[dict[str, Any]]) -> dict[str, Any]:
        done = {a["amendment"]["id"]: a for a in reg.get("amendments", [])}
        changed = False
        for am in amendments:
            ah = sha256_text(json.dumps(am, sort_keys=True))
            if am["id"] in done:
                if done[am["id"]]["sha256"] != ah:
                    raise PilotError(f"Amendment {am['id']} was registered at {done[am['id']]['registered_at']} "
                                     "with different content; a registered amendment cannot be changed.")
                continue
            if am.get("type") == "length_feasibility":
                self._assert_no_regenerated_outputs(am)
            entry = {"amendment": am, "sha256": ah, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "registered_at_unix": time.time(), "git_commit": git_commit(self.cfg.root),
                     "git_dirty": git_dirty(self.cfg.root)}
            reg.setdefault("amendments", []).append(entry)
            done[am["id"]] = entry
            changed = True
            log.info("Registered decision-rule amendment %s at %s", am["id"], entry["registered_at"])
        if changed:
            write_json(self.rule_path, reg)
        return reg

    def _assert_no_regenerated_outputs(self, am: dict[str, Any]) -> None:
        """Amendment 1 MUST be registered before the regenerated S5-S7 outputs exist."""
        ctx = self.ctx()
        cur = mp.template_sha(ctx.template("E05_draft"), ctx.requirements_block)
        for c in self.p["candidates"]:
            d = self.cand_dir(c) / "drafts.jsonl"
            if any(r.get("template_sha") == cur for r in read_jsonl(d)):
                raise PilotError(f"Amendment {am['id']} must be registered before the regenerated outputs exist, "
                                 f"but {d} already has drafts made with the current template ({cur}).")

    def registered_rule(self) -> dict[str, Any] | None:
        if not self.rule_path.exists():
            return None
        reg = json.loads(self.rule_path.read_text())
        cur = self.p.get("decision_rule")
        rule, amendments = _split_rule(cur or {})
        if cur is None or sha256_text(json.dumps(rule, sort_keys=True)) != reg["sha256"]:
            raise PilotError("pilot.decision_rule in configs/gen.yaml differs from the registered rule "
                             f"({reg['registered_at']}); the registered rule cannot be changed.")
        reg_am = {a["amendment"]["id"]: a for a in reg.get("amendments", [])}
        for am in amendments:
            r = reg_am.get(am["id"])
            if r is None:
                raise PilotError(f"Decision-rule amendment {am['id']} is in configs/gen.yaml but not registered: "
                                 "run `python scripts/pilot.py register-rule`.")
            if r["sha256"] != sha256_text(json.dumps(am, sort_keys=True)):
                raise PilotError(f"Amendment {am['id']} differs from its registration ({r['registered_at']}).")
        if set(reg_am) - {a["id"] for a in amendments}:
            raise PilotError("A registered amendment is missing from configs/gen.yaml.")
        return reg

    def _require_amendments_registered(self) -> None:
        _, amendments = _split_rule(self.p.get("decision_rule") or {})
        if not amendments:
            return
        if not self.rule_path.exists():
            raise PilotError("Register the decision rule and its amendments first: python scripts/pilot.py register-rule")
        self.registered_rule()

    # -------------------------------------------------------------------- audit
    def phase_audit_export(self) -> dict[str, Any]:
        if self.registered_rule() is None:
            raise PilotError("Register the decision rule first: python scripts/pilot.py register-rule "
                             "(Appendix I.1 step 5: committed with a timestamp before any labelling).")
        adir = self.dir / "audit"
        labels = adir / "audit_labels.csv"
        if labels.exists() and any(row.get("overall_quality") for row in csv.DictReader(labels.open())):
            raise PilotError("audit_labels.csv already contains labels; refusing to overwrite.")
        ideas = {i["id"]: i for i in self.load_ideas()}
        cands = list(self.p["candidates"])
        docs = {c: _index(self.cand_dir(c) / "rewrites.jsonl") for c in cands}
        missing = [c for c in cands if not docs[c]]
        if missing:
            raise PilotError(f"No rewrites yet for {missing}.")
        common = sorted(set.intersection(*(set(d) for d in docs.values())))
        n = int(self.p["audit"]["n_ideas"])
        if len(common) < n:
            raise PilotError(f"Only {len(common)} ideas have a document from every candidate; need {n}.")
        rng = random.Random(int(self.p["audit"]["seed"]))
        chosen = sorted(rng.sample(common, n))
        items = [(c, i) for i in chosen for c in cands]
        rng.shuffle(items)
        key, md = {}, ["# Pilot audit packet (blinded)\n",
                       "Label each item in `audit_labels.csv`: the six S8 criteria (PASS/FAIL) and overall quality "
                       "1-5 (5 = an excellent, realistic document that clearly shows the spec's reason). The writer "
                       "and the automatic S8 verdicts are hidden (Appendix I.1 step 4): do not open `.private/` or "
                       "`data/pilot/judged/` until labelling is done.\n",
                       f"Expected reason type: **{self.cfg.gen['documents']['reason_type'][self.p['spec_id']]}**\n",
                       "Criteria: faithful (nothing beyond the spec) · reason_visible_correct_type · realistic · "
                       "no_fabrication (no invented stats, studies, institutions, dates, names, URLs) · no_antipattern "
                       "(conversion arc, propaganda, template echo, fabrication, preachiness, strawman, over-certainty "
                       "about sentience) · about_assistant\n"]
        rows = []
        for k, (c, iid) in enumerate(items, 1):
            aid = f"P{k:03d}"
            key[aid] = {"candidate": c, "idea_id": iid, "doc_id": docs[c][iid]["id"]}
            idea = ideas[iid]
            md += [f"\n---\n\n## {aid}\n", f"**Document type:** {idea['doc_type']}  \n",
                   f"**Idea:** what: {idea['what']} · how: {idea['how']} · why: {idea['why']}\n",
                   f"<details><summary>Spec section</summary>\n\n{idea['spec_excerpt']}\n\n</details>\n",
                   f"\n{docs[c][iid]['text']}\n"]
            rows.append({"audit_id": aid, **{k2: "" for k2 in S8_CRITERIA}, "overall_quality": "", "notes": ""})
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "audit_packet.md").write_text("\n".join(md), encoding="utf-8")
        with labels.open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=["audit_id", *S8_CRITERIA, "overall_quality", "notes"])
            wcsv.writeheader()
            wcsv.writerows(rows)
        write_json(adir / ".private" / "audit_key.json", {"idea_ids": chosen, "items": key, "seed": self.p["audit"]["seed"],
                                                           "rule_sha256": self.registered_rule()["sha256"]})
        return {"items": len(items), "ideas": n, "packet": str(adir / "audit_packet.md"), "labels": str(labels)}

    # -------------------------------------------------------------------- report
    def phase_report(self) -> Path:
        m = _read_json(self.dir / "metrics.json")
        if not m:
            raise PilotError("Run `metrics` first.")
        summary = _read_json(self.dir / "ideas_summary.json") or {}
        audit = self._audit_results()
        reg = self.registered_rule()
        rule = reg["rule"] if reg else None
        amendments = [a["amendment"] for a in (reg or {}).get("amendments", [])]
        decision = decide(rule, audit, m) if audit and audit.get("complete") else {
            "winner": None, "reason": "human audit not complete"}
        decision = apply_amendments(decision, amendments, m)
        if reg:
            decision["amendments_registered"] = [{"id": a["amendment"]["id"], "registered_at": a["registered_at"],
                                                  "reason": a["amendment"].get("reason")}
                                                 for a in reg.get("amendments", [])]
        decision["feasibility"] = feasibility(self.cfg, m, decision.get("winner"))
        if reg:
            decision["rule_registered_at"] = reg["registered_at"]
        out = self.dir / "report.md"
        out.write_text(render_report(self.cfg, m, summary, audit, rule, decision), encoding="utf-8")
        write_json(self.dir / "decision.json", {"rule": rule, "decision": decision, "audit": audit})
        return out

    def _audit_results(self) -> dict[str, Any] | None:
        adir = self.dir / "audit"
        lab, keyf = adir / "audit_labels.csv", adir / ".private" / "audit_key.json"
        if not (lab.exists() and keyf.exists()):
            return None
        key = json.loads(keyf.read_text())["items"]
        try:
            rows = ar.read_labels(lab)
        except ar.AuditError as e:
            raise PilotError(str(e)) from None
        done = [r for r in rows if r.get("overall_quality", "").strip()]
        res: dict[str, Any] = {"labelled": len(done), "total": len(rows), "complete": len(done) == len(rows),
                               "by_candidate": {}, "judge_agreement": {}}
        per: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in done:
            k = key[r["audit_id"]]
            crit = {c: r[c].strip().upper() == "PASS" for c in S8_CRITERIA}
            per[k["candidate"]].append({"idea_id": k["idea_id"], "doc_id": k["doc_id"], "criteria": crit,
                                        "fail": not all(crit.values()), "quality": float(r["overall_quality"])})
        for c, items in per.items():
            q = [i["quality"] for i in items]
            res["by_candidate"][c] = {
                "n": len(items), "mean_quality": sum(q) / len(q), "sd_quality": _sd(q),
                "human_fail_rate": sum(i["fail"] for i in items) / len(items),
                "criterion_fail_rates": {k: sum(not i["criteria"][k] for i in items) / len(items) for k in S8_CRITERIA},
                "quality_by_idea": {i["idea_id"]: i["quality"] for i in items},
                "fail_by_idea": {i["idea_id"]: i["fail"] for i in items},
            }
        # judge–human agreement on the audited documents (S11: report agreement). Doc ids are shared by the
        # two writers (doc_<idea>_rewrite), so a judged file <writer>_by_<judge> is matched ONLY against
        # that writer's audited documents.
        annot = (_read_json(adir / "annotators.json") or {}).get("by_item", {}) if res["complete"] else {}
        item_ann = {(key[r["audit_id"]]["candidate"], key[r["audit_id"]]["doc_id"]): annot.get(r["audit_id"])
                    for r in done}
        res["criterion_agreement"] = {}
        for f in _judged_files(self.dir):
            writer = f.stem.split("_by_")[0]
            jd = {r["doc_id"]: r for r in read_jsonl(f)}
            mine = [i for i in per.get(writer, []) if i["doc_id"] in jd]
            groups = {f.stem: mine}
            for name in sorted(set(annot.values())):
                groups[f"{f.stem} [{name}]"] = [i for i in mine if item_ann.get((writer, i["doc_id"])) == name]
            for label, its in groups.items():
                pairs = [(not i["fail"], jd[i["doc_id"]]["verdict"] == "INCLUDE") for i in its]
                if pairs:
                    res["judge_agreement"][label] = {"n": len(pairs), "kappa_include": cohen_kappa(pairs),
                                                     "raw_agreement": sum(a == b for a, b in pairs) / len(pairs),
                                                     "human_pass": sum(a for a, _ in pairs),
                                                     "judge_include": sum(b for _, b in pairs)}
            res["criterion_agreement"][f.stem] = {
                c: {"n": len(mine), "human_fail": sum(not i["criteria"][c] for i in mine),
                    "judge_fail": sum(not jd[i["doc_id"]]["criteria"][c] for i in mine),
                    "both_fail": sum((not i["criteria"][c]) and (not jd[i["doc_id"]]["criteria"][c]) for i in mine),
                    "kappa": cohen_kappa([(i["criteria"][c], jd[i["doc_id"]]["criteria"][c]) for i in mine])}
                for c in S8_CRITERIA} if mine else {}
        if res["complete"]:   # post-labelling review (unblinded): annotators, revisions, per-item listing
            res["annotators"] = ar.annotator_analysis(done, key, annot) if annot else None
            res["revisions"] = ar.revisions(adir / "audit_labels_original.csv", lab)
            docs = {}   # keyed by (writer, doc_id): both writers use the same doc ids
            for c in self.p["candidates"]:
                docs.update({(c, r["id"]): r for r in read_jsonl(self.cand_dir(c) / "rewrites.jsonl")})
            res["items"] = ar.item_table(done, key, annot, docs)
            res["item_counts"] = ar.counts_by_candidate(res["items"])
        return res

    # -------------------------------------------------------------------- E.8 re-judge on audited documents
    async def phase_rejudge_audit(self, judge: str, template: str, tag: str) -> dict[str, Any]:
        """Re-judge ONLY the audited documents with another E.8 template and compare judge–human agreement
        with the original pilot judging on the same documents. Writes data/pilot/rejudge/<tag>/; never
        touches data/pilot/judged/ (the registered pilot record)."""
        audit = self._audit_results()
        if not audit or not audit.get("complete"):
            raise PilotError("Labelling is not complete.")
        adir = self.dir / "audit"
        key = json.loads((adir / ".private" / "audit_key.json").read_text())["items"]
        rows = {r["audit_id"]: r for r in ar.read_labels(adir / "audit_labels.csv")}
        human = {(k["candidate"], k["doc_id"]): {c: rows[aid][c].upper() == "PASS" for c in S8_CRITERIA}
                 for aid, k in key.items()}
        ideas = {i["id"]: i for i in self.load_ideas()}
        out_dir = self.dir / "rejudge" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = self.ctx()
        writers = sorted({k["candidate"] for k in key.values()})
        async with await self.connect(judge) as client:
            for w in writers:
                docs = {r["id"]: _strip(r) for r in read_jsonl(self.cand_dir(w) / "rewrites.jsonl")}
                todo = [docs[k["doc_id"]] for k in key.values() if k["candidate"] == w]
                await run_stage(f"rejudge_{tag}_{w}_by_{judge}", todo,
                                mp.s8_fn(ctx, client, ideas, w, judge, False, template),
                                out_dir / f"{w}_by_{judge}.jsonl", concurrency=self.conc)
            manifest = client.manifest()

        def summarise(files: dict[str, Path]) -> dict[str, Any]:
            pairs, crit = [], {c: {"human_fail": 0, "judge_fail": 0, "both_fail": 0} for c in S8_CRITERIA}
            for w, f in files.items():
                for r in read_jsonl(f):
                    h = human.get((w, r["doc_id"]))
                    if h is None:
                        continue
                    pairs.append((all(h.values()), r["verdict"] == "INCLUDE"))
                    for c in S8_CRITERIA:
                        crit[c]["human_fail"] += not h[c]
                        crit[c]["judge_fail"] += not r["criteria"][c]
                        crit[c]["both_fail"] += (not h[c]) and (not r["criteria"][c])
            return {"n": len(pairs), "kappa_include": cohen_kappa(pairs) if pairs else None,
                    "raw_agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
                    "judge_include": sum(b for _, b in pairs), "human_pass": sum(a for a, _ in pairs),
                    "criteria": crit}
        new_files = {w: out_dir / f"{w}_by_{judge}.jsonl" for w in writers}
        old_files = {w: self.dir / "judged" / f"{w}_by_{judge}.jsonl" for w in writers}
        res = {"judge": judge, "template": template, "tag": tag, "before": summarise(old_files),
               "after": summarise(new_files),
               "per_writer": {w: {"before": summarise({w: old_files[w]}), "after": summarise({w: new_files[w]})}
                              for w in writers},
               "gate": stage_gate(self.cfg, "s8_filter", new_files[writers[0]],
                                  [k["doc_id"] for k in key.values() if k["candidate"] == writers[0]]),
               "manifest": manifest}
        write_json(out_dir / "comparison.json", res)
        return res

    # -------------------------------------------------------------------- post-labelling inspection
    def set_annotators(self, spec: str) -> dict[str, Any]:
        by_item = ar.parse_ranges(spec)
        adir = self.dir / "audit"
        rows = ar.read_labels(adir / "audit_labels.csv")
        missing = sorted({r["audit_id"] for r in rows} - set(by_item))
        if missing:
            raise PilotError(f"No annotator given for {missing}")
        rec = {"spec": spec, "by_item": by_item, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        write_json(adir / "annotators.json", rec)
        return rec

    def phase_audit_inspect(self, ids: list[str]) -> dict[str, Any]:
        """Unblinded look at audit items (only after labelling is complete): writer, idea, S4 call, and a
        pairwise near-duplicate check with the S9 measures."""
        audit = self._audit_results()
        if not audit or not audit.get("complete"):
            raise PilotError("Labelling is not complete; audit items stay blinded until then.")
        key = json.loads((self.dir / "audit" / ".private" / "audit_key.json").read_text())["items"]
        ideas = {i["id"]: i for i in self.load_ideas()}
        m = _read_json(self.dir / "metrics.json") or {}
        docs, out = {}, {"items": {}, "pairs": []}
        for aid in ids:
            if aid not in key:
                raise PilotError(f"Unknown audit id {aid}")
            k = key[aid]
            doc = next((r for r in read_jsonl(self.cand_dir(k["candidate"]) / "rewrites.jsonl") if r["id"] == k["doc_id"]), None)
            docs[aid] = doc["text"] if doc else ""
            idea = ideas[k["idea_id"]]
            s9 = [x for x in ((m.get("candidates") or {}).get(k["candidate"], {}).get("near_duplicates") or [])
                  if x.get("idea_id") == k["idea_id"]]
            out["items"][aid] = {"candidate": k["candidate"], "idea_id": k["idea_id"], "doc_id": k["doc_id"],
                                 "call_id": idea.get("call_id"), "subdomain_id": idea.get("subdomain_id"),
                                 "doc_type": idea["doc_type"], "what": idea["what"][:160],
                                 "s9_flagged_in_metrics": s9 or False, "first_200_chars": docs[aid][:200]}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ka, kb = key[a], key[b]
                out["pairs"].append({"items": [a, b], "same_idea": ka["idea_id"] == kb["idea_id"],
                                     "same_candidate": ka["candidate"] == kb["candidate"],
                                     "same_s4_call": ideas[ka["idea_id"]].get("call_id") == ideas[kb["idea_id"]].get("call_id"),
                                     **ar.near_duplicate(docs[a], docs[b])})
        write_json(self.dir / "audit" / f"inspect_{'_'.join(ids)}.json", out)
        return out


# ----------------------------------------------------------------------------- decision
def _split_rule(full: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(base rule, amendments). The base rule's hash is what was registered originally."""
    full = dict(full or {})
    am = full.pop("amendments", None) or []
    return full, list(am)


def length_feasibility(am: dict[str, Any], m: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Amendment 1 (doc v1.7.4): per candidate, may it be the generator? Uses ALL S7 responses with a
    parsed document (kept and rejected, incl. length_out_of_range), so the production pre-check can't hide
    short output."""
    out = {}
    lim_med = float(am["min_median_words"])
    lim_w, lim_share = int(am["max_share_under_words"]["words"]), float(am["max_share_under_words"]["share"])
    for c, v in m["candidates"].items():
        ws = v.get("s7_words_all_parsed")
        if not ws:
            out[c] = {"feasible": None, "reason": "no S7 word statistics"}
            continue
        share = ws["share_under_400"] if lim_w == 400 else None
        reasons = []
        if ws["median"] < lim_med:
            reasons.append(f"median S7 rewrite {ws['median']:.0f} words < {lim_med:.0f}")
        if share is not None and share > lim_share:
            reasons.append(f"{share:.1%} of S7 rewrites under {lim_w} words > {lim_share:.0%}")
        out[c] = {"feasible": not reasons, "reasons": reasons, "median_words": ws["median"],
                  "share_under_400": ws["share_under_400"], "n": ws["n"]}
    return out


def apply_amendments(decision: dict[str, Any], amendments: list[dict[str, Any]], m: dict[str, Any]) -> dict[str, Any]:
    """Registered amendments act after the base rule. Amendment 1: an infeasible candidate cannot be the
    generator; if the rule's winner is infeasible and the other candidate is feasible, the other is chosen
    (flagged); if neither is feasible, there is no generator."""
    for am in amendments:
        if am.get("type") != "length_feasibility":
            continue
        feas = length_feasibility(am, m)
        decision.setdefault("amendment_checks", {})[f"amendment_{am['id']}"] = feas
        w = decision.get("winner")
        ok = [c for c, f in feas.items() if f["feasible"]]
        if any(f["feasible"] is None for f in feas.values()):
            decision["amendment_note"] = "length statistics missing for some candidate; Amendment 1 not evaluated"
            continue
        if not ok:
            decision.update({"winner": None, "rule_winner": w, "amendment_applied": am["id"],
                             "reason": f"Amendment {am['id']}: no candidate meets the length feasibility condition "
                                       f"({ {c: f['reasons'] for c, f in feas.items()} }); no generator"})
        elif w is not None and w not in ok:
            other = ok[0]
            decision.update({"winner": other, "rule_winner": w, "amendment_applied": am["id"],
                             "reason": f"{decision.get('reason')} -> but {w} fails Amendment {am['id']} "
                                       f"({'; '.join(feas[w]['reasons'])}); {other} is the generator"})
    return decision


def validate_rule(rule: dict[str, Any]) -> None:
    need = {"primary", "tie_margin", "tie_breaker", "veto", "candidate_c_trigger"}
    if missing := need - set(rule):
        raise PilotError(f"decision_rule is missing {sorted(missing)}")
    if rule["primary"] != "mean_quality" or rule["tie_breaker"] != "mean_s8_pass_rate_across_judges":
        raise PilotError("decision_rule: only primary=mean_quality / tie_breaker=mean_s8_pass_rate_across_judges "
                         "are implemented (doc v1.7)")


def decide(rule: dict[str, Any] | None, audit: dict[str, Any], m: dict[str, Any]) -> dict[str, Any]:
    """Appendix I.1 step 5 (doc v1.7), applied in this order:

    0. Candidate C trigger: if every candidate's human fail rate exceeds 15% (G1), no winner; run C.
    1. Veto: on the paired idea set, a candidate with >= 3 more failing documents than the other cannot
       win on quality; the lower-fail-rate candidate wins and the veto is flagged.
    2. Primary: paired mean difference in overall quality (1-5); |diff| >= 0.25 -> higher wins.
    3. Tie (|diff| < 0.25): higher mean S8 pass rate across both judges wins.
    """
    if not rule:
        return {"winner": None, "reason": "decision rule not registered (python scripts/pilot.py register-rule)"}
    validate_rule(rule)
    bc = audit["by_candidate"]
    cands = sorted(bc)
    if len(cands) != 2:
        return {"winner": None, "reason": f"rule is defined for two candidates; got {cands}"}
    a, b = cands
    thr = float(rule["candidate_c_trigger"]["human_fail_rate_above"])
    if all(bc[c]["human_fail_rate"] > thr for c in cands):
        return {"winner": None, "run_candidate_c": True,
                "reason": f"both human fail rates exceed {thr:.0%} ({ {c: round(bc[c]['human_fail_rate'], 3) for c in cands} }): "
                          "run candidate C (Appendix I.1 step 2) before deciding"}
    paired = sorted(set(bc[a]["quality_by_idea"]) & set(bc[b]["quality_by_idea"]))
    if not paired:
        return {"winner": None, "reason": "no idea IDs labelled for both candidates"}
    fails = {c: sum(bc[c]["fail_by_idea"][i] for i in paired) for c in cands}
    out: dict[str, Any] = {"paired_n": len(paired), "paired_fails": fails, "veto_applied": False,
                           "run_candidate_c": False}
    extra = abs(fails[a] - fails[b])
    if extra >= int(rule["veto"]["min_extra_failing_docs"]):
        w = min(cands, key=lambda c: fails[c])
        return out | {"winner": w, "veto_applied": True,
                      "reason": f"VETO: {max(cands, key=lambda c: fails[c])} has {extra} more failing documents "
                                f"of {len(paired)} paired (>= {rule['veto']['min_extra_failing_docs']}); "
                                f"lower-fail-rate candidate {w} wins"}
    diffs = [bc[a]["quality_by_idea"][i] - bc[b]["quality_by_idea"][i] for i in paired]
    d = sum(diffs) / len(diffs)
    se = _sd(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else None
    out |= {"paired_mean_diff": {f"{a}-{b}": d}, "paired_diff_se": se}
    if abs(d) >= float(rule["tie_margin"]):
        w = a if d > 0 else b
        return out | {"winner": w, "reason": f"paired mean quality {a}-{b} = {d:+.3f} (|Δ| >= {rule['tie_margin']})"}
    pr = {c: m["candidates"][c]["mean_pass_rate_across_judges"] for c in cands}
    if pr[a] is None or pr[b] is None or pr[a] == pr[b]:
        return out | {"winner": None, "tie_break": pr, "reason": "tie on quality and the tie-breaker is missing or equal"}
    w = max(cands, key=lambda c: pr[c])
    return out | {"winner": w, "tie_break": pr,
                  "reason": f"tie: paired mean quality {a}-{b} = {d:+.3f} (|Δ| < {rule['tie_margin']}); "
                            f"higher mean S8 pass rate across both judges: {w} ({pr[w]:.1%} vs {min(pr.values()):.1%})"}


def feasibility(cfg: Config, m: dict[str, Any], winner: str | None) -> dict[str, Any] | None:
    """Throughput as a feasibility check only (step 5): hours to push one corpus's ideas through S5-S8 at
    the winner's measured operating points (§6.4: ~7,000 ideas per corpus)."""
    if not winner:
        return None
    tp = (m["candidates"].get(winner) or {}).get("throughput")
    if not tp:
        return {"note": "official benchmark not available"}
    n = 7000
    hours = {s: n / op["requests_per_s"] / 3600 for s, op in tp.items() if op.get("requests_per_s")}
    return {"ideas_per_corpus": n, "hours_by_stage": hours, "total_hours": sum(hours.values()),
            "note": "S1-S4, retries and S10 not included; if unreasonable, apply step 6's FP8 option"}


# ----------------------------------------------------------------------------- report text
def _coverage(out_path: Path, expected_ids: list[str]) -> dict[str, int]:
    """How many inputs have an outcome (kept or rejected) in a stage's output files."""
    sp = side_paths(out_path)
    done = {r["_src"] for r in read_jsonl(sp["out"]) if "_src" in r} | \
           {r["_src"] for r in read_jsonl(sp["rejected"]) if "_src" in r}
    return {"expected": len(expected_ids), "resolved": len(done & set(expected_ids)),
            "unresolved": len(set(expected_ids) - done)}


def stage_gate(cfg: Config, stage: str, out_path: Path, input_ids: list[str]) -> dict[str, Any]:
    """Doc v1.7.3 §6.1 stage gate [fixed]: a stage passes only if no item is unresolved and, for a
    temperature-0 stage, its §6.1 fallback rate is at most ``client.t0_fallback_warn_rate`` (2%).
    Computed from the stage's output files (records carry ``t0_fallback``), so it survives resumes."""
    sp = side_paths(out_path)
    rows = list(read_jsonl(sp["out"])) + [r.get("detail") or {} for r in read_jsonl(sp["rejected"])]
    done = {r["_src"] for r in read_jsonl(sp["out"]) if "_src" in r} | \
           {r["_src"] for r in read_jsonl(sp["rejected"]) if "_src" in r}
    unresolved = sorted(set(input_ids) - done)
    greedy = cfg.stage_temperature(stage) == 0.0
    warn = float((cfg.gen.get("client") or {}).get("t0_fallback_warn_rate", 0.02))
    out: dict[str, Any] = {"stage": stage, "file": out_path.name, "inputs": len(input_ids), "unresolved": len(unresolved),
                           "unresolved_ids": unresolved[:50], "greedy": greedy}
    if greedy:
        last_err: dict[str, str] = {}
        for r in read_jsonl(sp["errors"]):
            last_err[r.get("_src")] = r.get("error", "")
        fb_unresolved = sum("fallback" in last_err.get(i, "") for i in unresolved)
        fb = sum(bool(r.get("t0_fallback")) for r in rows) + fb_unresolved
        n = len(rows) + len(unresolved)
        out.update({"t0_fallbacks": fb, "t0_calls": n, "t0_fallback_rate": fb / n if n else 0.0, "limit": warn,
                    "records_without_flag": sum("t0_fallback" not in r for r in rows)})
    reasons = []
    if unresolved:
        reasons.append(f"{len(unresolved)} unresolved item(s)")
    if greedy and out["t0_fallback_rate"] > warn:
        reasons.append(f"T=0 fallback rate {out['t0_fallback_rate']:.1%} > {warn:.0%}")
    out["passed"], out["reasons"] = not reasons, reasons
    return out


def _record_gate(path: Path, key: str, gate: dict[str, Any]) -> None:
    gates = _read_json(path) or {}
    gates[key] = {**gate, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(path, gates)


def _quantiles(xs: list[float]) -> dict[str, Any] | None:
    if not xs:
        return None
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]   # noqa: E731
    return {"n": len(xs), "min": xs[0], "p25": q(0.25), "p50": q(0.5), "p75": q(0.75), "p90": q(0.9),
            "p99": q(0.99), "max": xs[-1], "mean": round(sum(xs) / len(xs), 1)}


def stage_summary(out_path: Path, input_ids: list[str], ideas_by_id: dict[str, dict[str, Any]] | None = None,
                  story_limits: tuple[float, float] | None = None) -> dict[str, Any]:
    """Outcome of an S5/S7 stage over ALL responses (kept + rejected): coverage, keep rate, counts per
    rejection reason and per tolerance (doc v1.7.2), truncation, repetition loops, story share of kept docs."""
    sp = side_paths(out_path)
    kept = list(read_jsonl(sp["out"]))
    rej = list(read_jsonl(sp["rejected"]))
    rows = [{**r, "_kept": True} for r in kept] + [{**(r.get("detail") or {}), "_src": r.get("_src"), "_kept": False,
                                                   "reason": r.get("reason")} for r in rej]
    n = len(rows)
    resolved = {r["_src"] for r in rows}
    unresolved = sorted(set(input_ids) - resolved)
    count = lambda reason: sum(r.get("reason") == reason for r in rows)   # noqa: E731
    out: dict[str, Any] = {
        "inputs": len(input_ids), "responses": n, "kept": len(kept), "rejected": len(rej),
        "keep_rate": len(kept) / n if n else 0.0,
        "unresolved": len(unresolved), "unresolved_ids": unresolved,
        "rejections": dict(Counter(r.get("reason") for r in rej).most_common()),
        "tolerances_kept": dict(Counter(t for r in kept for t in r.get("tolerances") or [])),
        "tolerances_all_responses": dict(Counter(t for r in rows for t in r.get("tolerances") or [])),
        "truncated": count("truncated_max_tokens"), "truncation_rate": count("truncated_max_tokens") / n if n else 0.0,
        "degenerate_repetition": count("degenerate_repetition"),
        "degenerate_repetition_rate": count("degenerate_repetition") / n if n else 0.0,
        "repetition_examples": [{"src": r["_src"], "kind": r.get("kind"), "count": r.get("count"), "match": r.get("match")}
                                for r in rows if r.get("reason") == "degenerate_repetition"][:10],
        "output_tokens": _quantiles([r["n_tokens"] for r in rows if r.get("n_tokens")]),
        # doc v1.7.4: words of every parsed document (kept and rejected, incl. length_out_of_range)
        "words_all_parsed": word_stats([r["doc_words"] for r in rows if r.get("doc_words") is not None]),
        "words_kept": word_stats([r.get("words") or r.get("doc_words") for r in kept
                                  if (r.get("words") or r.get("doc_words")) is not None]),
        "template_shas": sorted({str(r.get("template_sha")) for r in rows}),
    }
    if ideas_by_id is not None:
        stories = sum(bool(ideas_by_id[r["idea_id"]]["is_story"]) for r in kept if r.get("idea_id") in ideas_by_id)
        frozen = sum(bool(i["is_story"]) for i in ideas_by_id.values())
        share = stories / len(kept) if kept else 0.0
        out["story_share"] = {"kept_stories": stories, "kept": len(kept), "share": share,
                              "frozen_ideas_share": frozen / len(ideas_by_id) if ideas_by_id else None,
                              "quota": list(story_limits) if story_limits else None,
                              "within_quota": (story_limits[0] <= share <= story_limits[1]) if story_limits else None}
    return out


def word_stats(words: list[int]) -> dict[str, Any] | None:
    """Doc v1.7.4 length reporting: median, 10th/90th percentiles, shares under 400 / over 1,500 words."""
    if not words:
        return None
    w = sorted(words)
    at = lambda f: w[min(len(w) - 1, int(f * (len(w) - 1) + 0.5))]   # noqa: E731
    n = len(w)
    med = (w[(n - 1) // 2] + w[n // 2]) / 2
    return {"n": n, "min": w[0], "p10": at(0.1), "median": med, "p90": at(0.9), "max": w[-1],
            "under_400": sum(x < 400 for x in w), "share_under_400": sum(x < 400 for x in w) / n,
            "under_450": sum(x < 450 for x in w),
            "over_1500": sum(x > 1500 for x in w), "share_over_1500": sum(x > 1500 for x in w) / n}


def critique_summary(out_path: Path) -> dict[str, Any]:
    """S6: final and first-attempt (greedy) output length, truncations, fallback use, issues per critique."""
    rows = list(read_jsonl(out_path))
    return {
        "critiques": len(rows),
        "final_tokens": _quantiles([r["final_tokens"] for r in rows if r.get("final_tokens")]),
        "greedy_tokens": _quantiles([r["greedy_tokens"] for r in rows if r.get("greedy_tokens")]),
        "t0_fallbacks": sum(bool(r.get("t0_fallback")) for r in rows),
        "fallback_reasons": dict(Counter(r["t0_fallback_reason"] for r in rows if r.get("t0_fallback_reason"))),
        "final_truncated": sum(r.get("finish_reason") == "length" for r in rows),
        "issues_per_critique": _quantiles([r["n_issues"] for r in rows if "n_issues" in r]),
        "overall": dict(Counter(r.get("overall") for r in rows)),
    }


def s5_summary(cdir: Path, idea_ids: list[str], ideas_by_id: dict[str, dict[str, Any]] | None = None,
               story_limits: tuple[float, float] | None = None) -> dict[str, Any]:
    """S5 summary: ``stage_summary`` plus </think> slip, plan length and template hashes."""
    out = stage_summary(cdir / "drafts.jsonl", idea_ids, ideas_by_id, story_limits)
    sp = side_paths(cdir / "drafts.jsonl")
    rows = list(read_jsonl(sp["out"])) + [r.get("detail") or {} for r in read_jsonl(sp["rejected"])]
    n = len(rows)
    slips = sum(bool(r.get("scratchpad_closed_with_think")) for r in rows)
    plans = [r["plan_words"] for r in rows if r.get("plan_words") is not None]
    out.update({
        "n_ideas": len(idea_ids),
        "think_slip": slips, "think_slip_rate": slips / n if n else 0.0,
        "no_closed_plan": sum(r.get("plan_words") is None for r in rows),
        "plan_words": _quantiles(plans),
        "plan_over_120_words": sum(w > 120 for w in plans),
        "plan_word_bins": dict(Counter(_bin(w) for w in plans)),
        "template_shas": sorted({str(r.get("template_sha")) for r in rows}),
    })
    return out


def _bin(w: int) -> str:
    for hi in (40, 80, 120, 200, 400):
        if w <= hi:
            return f"<={hi}"
    return ">400"


def render_report(cfg: Config, m: dict[str, Any], summary: dict[str, Any], audit: dict[str, Any] | None,
                  rule: dict[str, Any] | None, decision: dict[str, Any]) -> str:
    pc = cfg.gen["pilot"]
    L = ["# Generator-selection pilot report (Appendix I.1)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')}. Spec: `{pc['spec_id']}` (Appendix A draft). "
         f"Frozen ideas: {m['n_ideas']} (sha256 in `ideas.sha256`). Fixed pilot concurrency: {m['concurrency']}.", ""]
    L += ["## Candidates", "", "| | Model | Revision | Precision | top_p / top_k / min_p | Vendor T |", "|---|---|---|---|---|---|"]
    for c, v in pc["candidates"].items():
        mp_ = cfg.models[v["model"]]
        s = mp_.sampling
        L.append(f"| {c} | `{mp_.hf_id}` | `{mp_.revision[:10]}` | {mp_.precision} | {s['top_p']} / {s['top_k']} / {s['min_p']} "
                 f"| {mp_.vendor_temperature_range[0]} |")
    L += ["", "## Frozen ideas: quota check (§6.3 S4)", ""]
    qc = (summary.get("idea_quota_check") or {}).get("checks", {})
    for k, v in qc.items():
        val = v["value"] if not isinstance(v["value"], dict) else ", ".join(f"{a}={b:.2f}" for a, b in v["value"].items())
        L.append(f"- {k}: {val if isinstance(val, str) else f'{val:.3f}'} — {'PASS' if v['pass'] else 'FAIL'}")
    cov = summary.get("s1_coverage") or {}
    if cov:
        L.append(f"- S1 spec coverage {cov.get('spec_sentences_covered', 0):.0%}; excerpt sentences not found in spec "
                 f"{cov.get('excerpt_sentences_not_in_spec', 0):.0%}")
    L += ["", "## S8 pass rates: writer × judge (2 × 2)", "",
          "| Writer \\ Judge | " + " | ".join(pc["candidates"]) + " | Mean |", "|---|" + "---|" * (len(pc["candidates"]) + 1)]
    for w in pc["candidates"]:
        cells = []
        for j in pc["candidates"]:
            pr = m["pass_rates"].get(f"{w}_by_{j}")
            cells.append("—" if not pr else f"{pr['pass_rate']:.1%} ({pr['include']}/{pr['judged']}) "
                                            f"[{pr['wilson95'][0]:.0%}–{pr['wilson95'][1]:.0%}]")
        mean = (m["candidates"].get(w) or {}).get("mean_pass_rate_across_judges")
        L.append(f"| {w} | " + " | ".join(cells) + f" | {'—' if mean is None else f'{mean:.1%}'} |")
    L += ["", "Per-criterion fail rates:", "", "| Writer×Judge | " + " | ".join(S8_CRITERIA) + " | verdict≠criteria |",
          "|---|" + "---|" * (len(S8_CRITERIA) + 1)]
    for k, pr in m["pass_rates"].items():
        L.append(f"| {k} | " + " | ".join(f"{pr['criterion_fail_rates'].get(c, 0):.0%}" for c in S8_CRITERIA)
                 + f" | {pr['model_verdict_inconsistent']} |")
    L += ["", "## Other comparisons", "", "| | Docs | Near-dup rate (S9) | Language-mixing rej. | Other rejections | Words (mean) | T=0 fallbacks |",
          "|---|---|---|---|---|---|---|"]
    for c, v in m["candidates"].items():
        other = {k: n for k, n in v["rejections"].items() if not k.endswith("language_mixing")}
        fb = v.get("t0_fallbacks") or {}
        fbs = ", ".join(f"{s} {d['rate']:.1%}" for s, d in fb.items()) or "—"
        words = (v.get("rewrite_words") or {}).get("mean")
        L.append(f"| {c} | {v['rewrites_kept']} | {v['near_duplicate_rate']:.1%} ({v['near_duplicate_method']}) | "
                 f"{v['language_mixing_rejection_rate']:.1%} | {other or '—'} | {'—' if words is None else f'{words:.0f}'} | {fbs} |")
    L += ["", "## S5 format (deviations 8 and 9)", "",
          "| | `</think>` closing the plan (tolerated) | S5 truncation | Plan words p50 / p90 / max | E.5 template sha |",
          "|---|---|---|---|---|"]
    for c, v in m["candidates"].items():
        sl = v.get("s5_think_slip") or {}
        rate = "—" if sl.get("rate") is None else f"{sl['rate']:.1%} ({sl['count']}/{sl['of']})"
        pw = v.get("s5_plan_words") or {}
        pws = f"{pw['p50']} / {pw['p90']} / {pw['max']}" if pw else "—"
        L.append(f"| {c} | {rate} | {v.get('s5_truncation_rate', 0):.1%} | {pws} | `{v.get('s5_template_sha') or '?'}` |")
    L += ["", "### Parsing tolerances, repetition loops and story share (doc v1.7.2)", "",
          "| | Stage | Kept / responses | Tolerated (kept) | Repetition-loop rej. | Truncated | Story share (kept) |",
          "|---|---|---|---|---|---|---|"]
    for c, v in m["candidates"].items():
        for st, b in (v.get("by_stage") or {}).items():
            ss = b.get("story_share") or {}
            story = f"{ss['share']:.1%} ({'ok' if ss.get('within_quota') else 'OUT of'} {ss['quota']})" if ss else "—"
            L.append(f"| {c} | {st} | {b['kept']}/{b['responses']} | {b['tolerances_kept'] or '—'} | "
                     f"{b['degenerate_repetition']} ({b['degenerate_repetition_rate']:.1%}) | {b['truncation_rate']:.1%} | {story} |")
    L += ["", "### S7 rewrite length in words (doc v1.7.4, Appendix I.1 Amendment 1)", "",
          "All S7 responses with a parsed document (kept and rejected, incl. `length_out_of_range`).", "",
          "| | n | Median | 10th pct | 90th pct | Under 400 | Over 1,500 |", "|---|---|---|---|---|---|---|"]
    for c, v in m["candidates"].items():
        ws = v.get("s7_words_all_parsed")
        if ws:
            L.append(f"| {c} | {ws['n']} | {ws['median']:.0f} | {ws['p10']} | {ws['p90']} | "
                     f"{ws['under_400']} ({ws['share_under_400']:.1%}) | {ws['over_1500']} ({ws['share_over_1500']:.1%}) |")
    shas = {v.get("s5_template_sha") for v in m["candidates"].values()}
    if len(shas) > 1:
        L.append("\n**WARNING: candidates used different E.5 templates; the comparison is not valid.**")
    L += ["", "## Throughput (§6.1.1 E, official, requests/s over the fixed 200-prompt set)", ""]
    for c, v in m["candidates"].items():
        tp = v.get("throughput")
        if not tp:
            L.append(f"- {c}: official benchmark not run yet (`data/pilot/{c}/throughput_benchmark.json`)")
            continue
        L.append(f"- {c}: " + "; ".join(f"{s}: c={op.get('concurrency')} → {op.get('requests_per_s') or 0:.3f} req/s "
                                         f"(peak {op.get('peak_requests_per_s') or 0:.3f})" for s, op in tp.items()))
    L += ["", "## Blinded human audit (S11 criteria)", ""]
    feas = next(iter((decision.get("amendment_checks") or {}).values()), {})
    for c, v in sorted(m["candidates"].items()):
        n_len = (v.get("rejections") or {}).get("s7_rewrite:length_out_of_range", 0)
        if n_len:
            L.append(f"- **{c}'s audited documents are a selected subset:** they passed the S7 length pre-check "
                     f"(400–1,500 words), which rejected {n_len} of {c}'s rewrites.")
        if (feas.get(c) or {}).get("feasible") is False:
            L.append(f"- **{c} cannot be the generator under Amendment 1** "
                     f"({'; '.join(feas[c].get('reasons') or [])}), so {c}'s labels mainly serve judge validation "
                     f"(§9.9) and the step-0 candidate-C check, not the choice of generator.")
    L.append("")
    if not audit:
        L.append("Not exported or not labelled yet.")
    else:
        L.append(f"Labelled {audit['labelled']}/{audit['total']} items.")
        L += ["", "| | n | Mean quality (1-5) | Human fail rate | " + " | ".join(S8_CRITERIA) + " |",
              "|---|---|---|---|" + "---|" * len(S8_CRITERIA)]
        for c, v in sorted(audit["by_candidate"].items()):
            L.append(f"| {c} | {v['n']} | {v['mean_quality']:.2f} ± {v['sd_quality']:.2f} | {v['human_fail_rate']:.0%} | "
                     + " | ".join(f"{v['criterion_fail_rates'][k]:.0%}" for k in S8_CRITERIA) + " |")
        an = audit.get("annotators")
        if an:
            L += ["", "### Annotators", "",
                  f"Annotators: {', '.join(an['annotators'])}; no item was labelled twice, so there is **no human–human "
                  f"agreement** for this audit. Per annotator × candidate:", "",
                  "| Annotator / candidate | n | Human fail rate | Mean quality |", "|---|---|---|---|"]
            for k2, v in an["per_annotator"].items():
                L.append(f"| {k2} | {v['n']} | {v['human_fail_rate']:.0%} | {v['mean_quality']:.2f} |")
            L += ["", f"**A/B pairs labelled by different annotators: {len(an['pairs_cross_annotator'])} of "
                  f"{an['pairs_total']}.** In these pairs the paired quality difference mixes writer and annotator."]
            for e in an["pairs_cross_annotator"]:
                L.append("- " + ", ".join(f"{k3}: {v3}" for k3, v3 in e.items()))
            sens = an.get("sensitivity_same_annotator_pairs") or {}
            for k3, v3 in sens.items():
                if v3:
                    L.append(f"\nSensitivity (descriptive, not part of the rule): paired quality {k3} over the "
                             f"{v3['n']} same-annotator pairs = {v3['mean_diff']:+.2f}.")
        if audit.get("revisions"):
            L += ["", "### Label revisions made before unblinding", ""]
            for rv in audit["revisions"]:
                L.append(f"- {rv['audit_id']}: " + "; ".join(f"{c} {a or '—'} → {b}" for c, (a, b) in rv["changes"].items())
                         + (f" — {rv['notes']}" if rv.get("notes") else ""))
        if audit.get("items"):
            L += ["", "### Audited items (unblinded after labelling)", "",
                  "| Item | Writer | Annotator | Quality | FAIL criteria | Words | Animal terms /100w | Notes |",
                  "|---|---|---|---|---|---|---|---|"]
            for it in sorted(audit["items"], key=lambda x: x["audit_id"]):
                L.append(f"| {it['audit_id']} | {it['candidate']} | {it['annotator']} | {it['quality']:.0f} | "
                         f"{', '.join(it['fails']) or '—'} | {it['words']} | {it['animal_terms_per_100w']} | "
                         f"{(it['notes'] or '').replace('|', '/')} |")
            L += ["", "Failure counts per criterion (documents): " + "; ".join(
                f"{c}: {v['criterion_fails']} ({v['fail_docs']}/{v['n']} documents fail)"
                for c, v in audit["item_counts"].items())]
        if audit.get("judge_agreement"):
            L += ["", "Judge–human agreement on audited documents (INCLUDE vs human all-PASS):", ""]
            for k, v in audit["judge_agreement"].items():
                kap = v["kappa_include"]
                L.append(f"- {k}: κ = {'—' if kap is None else f'{kap:.2f}'}, raw agreement {v['raw_agreement']:.0%} (n={v['n']})")
    if audit and audit.get("criterion_agreement"):
        L += ["", "Per-criterion human vs judge on the audited documents (each judged file matched only to its writer's "
              "documents; cells: human FAIL / judge FAIL / both, κ):", "",
              "| Writer×Judge | n | " + " | ".join(S8_CRITERIA) + " |", "|---|---|" + "---|" * len(S8_CRITERIA)]
        for k, cs in audit["criterion_agreement"].items():
            if not cs:
                continue
            n = next(iter(cs.values()))["n"]
            L.append(f"| {k} | {n} | " + " | ".join(
                f"{v['human_fail']} / {v['judge_fail']} / {v['both_fail']}"
                + (f", κ {v['kappa']:.2f}" if v["kappa"] is not None else "") for v in cs.values()) + " |")
    L += ["", "## Decision (Appendix I.1 steps 5–6)", "",
          f"Pre-registered rule: `{json.dumps(rule) if rule else 'NOT SET'}`"
          + (f" (registered {decision['rule_registered_at']})" if decision.get("rule_registered_at") else ""), "",
          f"**Winner: {decision.get('winner') or 'none yet'}** — {decision.get('reason')}", ""]
    for a in decision.get("amendments_registered") or []:
        L.append(f"Amendment {a['id']} registered {a['registered_at']}: {a.get('reason')}")
    for k, feas in (decision.get("amendment_checks") or {}).items():
        L.append(f"{k.replace('_', ' ').capitalize()} (length feasibility): " + "; ".join(
            f"{c} {'feasible' if f['feasible'] else 'NOT feasible (' + '; '.join(f.get('reasons') or []) + ')'}"
            f" — median {f.get('median_words', 0):.0f} words, {f.get('share_under_400', 0):.1%} under 400"
            for c, f in feas.items() if f.get("feasible") is not None))
    if decision.get("amendment_applied"):
        L.append(f"**Amendment {decision['amendment_applied']} changed the outcome:** the rule's winner was "
                 f"{decision.get('rule_winner')}.")
    if decision.get("veto_applied"):
        L.append("**The veto applied:** the winner was decided by human fail rate, not by mean quality.")
    if decision.get("run_candidate_c"):
        L.append("**Both candidates exceed the 15% human fail rate (G1): run candidate C before deciding.**")
    if decision.get("paired_mean_diff"):
        (k, v), = decision["paired_mean_diff"].items()
        se = decision.get("paired_diff_se")
        L.append(f"Paired mean quality difference {k}: {v:+.3f}" + (f" (SE {se:.3f})" if se else "")
                 + f" over {decision['paired_n']} ideas; failing documents on the paired set: {decision['paired_fails']}.")
    fz = decision.get("feasibility")
    if fz and fz.get("total_hours") is not None:
        L.append(f"Feasibility (not a criterion): ~{fz['total_hours']:.1f} h for S5–S8 on {fz['ideas_per_corpus']:,} ideas "
                 f"at the winner's operating points ({', '.join(f'{s} {h:.1f} h' for s, h in fz['hours_by_stage'].items())}).")
    L.append("")
    if decision.get("winner"):
        wm = cfg.models[pc["candidates"][decision["winner"]]["model"]]
        other = [c for c in pc["candidates"] if c != decision["winner"]]
        L.append(f"Next (step 6–7): generator = `{wm.hf_id}` (family {wm.family}); evaluation judge must be from a "
                 f"different family (candidates: {other}). Freeze model, revision, quantisation and sampling in "
                 f"configs/base.yaml and configs/gen.yaml; run the official benchmark with `--write-config` for the winner.")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- utils
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _strip(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def _index(path: Path) -> dict[str, dict[str, Any]]:
    """Records keyed by idea_id."""
    return {r["idea_id"]: _strip(r) for r in read_jsonl(path)}


def _ordered(by_idea: dict[str, dict[str, Any]], ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records in frozen-idea order (grouped by domain for prefix caching)."""
    return [by_idea[i["id"]] for i in ideas if i["id"] in by_idea]


def _judged_files(d: Path) -> list[Path]:
    """data/pilot/judged/{writer}_by_{judge}.jsonl (not the .rejected/.errors side files)."""
    jd = d / "judged"
    return sorted(f for f in jd.glob("*_by_*.jsonl") if f.name.count(".") == 1) if jd.exists() else []


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _throughput(path: Path) -> dict[str, Any] | None:
    d = _read_json(path)
    if not d or d.get("label") != "official":
        return None
    return d.get("operating_points")


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_dirty(root: Path) -> bool | None:
    """True if the working tree has uncommitted changes (registration should follow a commit)."""
    try:
        r = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=5)
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()
