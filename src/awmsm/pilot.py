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
            await run_stage("s4_ideas", calls, mp.s4_fn(ctx, client, subs_by_id, int(self.cfg.gen["s4"]["k"])),
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
        summary = {
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_ideas": len(ideas), "target": self.p["n_ideas"],
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
        t: dict[str, float] = {}
        async with await self.connect(cand) as client:
            t0 = time.monotonic()
            await run_stage("s5_draft", ideas, mp.s5_fn(ctx, client, checks), cdir / "drafts.jsonl",
                            concurrency=self.conc)
            t["s5_draft"] = time.monotonic() - t0
            drafts = _index(cdir / "drafts.jsonl")
            t0 = time.monotonic()
            await run_stage("s6_critique", _ordered(drafts, ideas), mp.s6_fn(ctx, client, idea_by_id),
                            cdir / "critiques.jsonl", concurrency=self.conc)
            t["s6_critique"] = time.monotonic() - t0
            crits = _index(cdir / "critiques.jsonl")
            t0 = time.monotonic()
            await run_stage("s7_rewrite", _ordered(crits, ideas), mp.s7_fn(ctx, client, idea_by_id, drafts, checks),
                            cdir / "rewrites.jsonl", concurrency=self.conc)
            t["s7_rewrite"] = time.monotonic() - t0
            man = {"candidate": cand, "manifest": client.manifest(), "stage_seconds_this_run": t,
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
            out["manifest"] = client.manifest()
        write_json(self.dir / "judged" / f"manifest_judge_{judge}.json", out)
        return out

    # -------------------------------------------------------------------- metrics
    def phase_metrics(self, use_embeddings: bool = True) -> dict[str, Any]:
        ideas = self.load_ideas()
        n_ideas = len(ideas)
        cands = [c for c in self.p["candidates"] if (self.cand_dir(c) / "rewrites.jsonl").exists()]
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
        """Appendix I.1 step 5: commit the decision rule, with a timestamp, before any labelling."""
        rule = self.p.get("decision_rule")
        if not rule:
            raise PilotError("configs/gen.yaml pilot.decision_rule is empty.")
        validate_rule(rule)
        h = sha256_text(json.dumps(rule, sort_keys=True))
        if self.rule_path.exists():
            reg = json.loads(self.rule_path.read_text())
            if reg["sha256"] == h:
                return reg
            raise PilotError(f"A different rule was already registered at {reg['registered_at']}; "
                             "a registered rule cannot be changed.")
        lab = self.dir / "audit" / "audit_labels.csv"
        if lab.exists() and any(r.get("overall_quality", "").strip() for r in csv.DictReader(lab.open())):
            raise PilotError("Audit labels already exist; the rule must be registered before labelling.")
        reg = {"rule": rule, "sha256": h, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "registered_at_unix": time.time(), "git_commit": git_commit(self.cfg.root),
               "doc_version": "AW_MSM_Methodology v1.7, Appendix I.1 step 5"}
        write_json(self.rule_path, reg)
        return reg

    def registered_rule(self) -> dict[str, Any] | None:
        if not self.rule_path.exists():
            return None
        reg = json.loads(self.rule_path.read_text())
        cur = self.p.get("decision_rule")
        if cur is None or sha256_text(json.dumps(cur, sort_keys=True)) != reg["sha256"]:
            raise PilotError("pilot.decision_rule in configs/gen.yaml differs from the registered rule "
                             f"({reg['registered_at']}); the registered rule cannot be changed.")
        return reg

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
        decision = decide(rule, audit, m) if audit and audit.get("complete") else {
            "winner": None, "reason": "human audit not complete"}
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
        rows = list(csv.DictReader(lab.open(encoding="utf-8")))
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
        # judge–human agreement on the audited documents (S11: report agreement)
        for f in _judged_files(self.dir):
            jd = {r["doc_id"]: r for r in read_jsonl(f)}
            pairs = [(not i["fail"], jd[i["doc_id"]]["verdict"] == "INCLUDE")
                     for items in per.values() for i in items if i["doc_id"] in jd]
            if pairs:
                res["judge_agreement"][f.stem] = {"n": len(pairs), "kappa_include": cohen_kappa(pairs),
                                                  "raw_agreement": sum(a == b for a, b in pairs) / len(pairs)}
        return res


# ----------------------------------------------------------------------------- decision
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
    L += ["", "## Throughput (§6.1.1 E, official, requests/s over the fixed 200-prompt set)", ""]
    for c, v in m["candidates"].items():
        tp = v.get("throughput")
        if not tp:
            L.append(f"- {c}: official benchmark not run yet (`data/pilot/{c}/throughput_benchmark.json`)")
            continue
        L.append(f"- {c}: " + "; ".join(f"{s}: c={op.get('concurrency')} → {op.get('requests_per_s') or 0:.3f} req/s "
                                         f"(peak {op.get('peak_requests_per_s') or 0:.3f})" for s, op in tp.items()))
    L += ["", "## Blinded human audit (S11 criteria)", ""]
    if not audit:
        L.append("Not exported or not labelled yet.")
    else:
        L.append(f"Labelled {audit['labelled']}/{audit['total']} items.")
        L += ["", "| | n | Mean quality (1-5) | Human fail rate | " + " | ".join(S8_CRITERIA) + " |",
              "|---|---|---|---|" + "---|" * len(S8_CRITERIA)]
        for c, v in sorted(audit["by_candidate"].items()):
            L.append(f"| {c} | {v['n']} | {v['mean_quality']:.2f} ± {v['sd_quality']:.2f} | {v['human_fail_rate']:.0%} | "
                     + " | ".join(f"{v['criterion_fail_rates'][k]:.0%}" for k in S8_CRITERIA) + " |")
        if audit.get("judge_agreement"):
            L += ["", "Judge–human agreement on audited documents (INCLUDE vs human all-PASS):", ""]
            for k, v in audit["judge_agreement"].items():
                kap = v["kappa_include"]
                L.append(f"- {k}: κ = {'—' if kap is None else f'{kap:.2f}'}, raw agreement {v['raw_agreement']:.0%} (n={v['n']})")
    L += ["", "## Decision (Appendix I.1 steps 5–6)", "",
          f"Pre-registered rule: `{json.dumps(rule) if rule else 'NOT SET'}`"
          + (f" (registered {decision['rule_registered_at']})" if decision.get("rule_registered_at") else ""), "",
          f"**Winner: {decision.get('winner') or 'none yet'}** — {decision.get('reason')}", ""]
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


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()
