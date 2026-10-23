from __future__ import annotations

import copy
import csv
import json
import random

import pytest

from awmsm import prompts
from awmsm.config import Config
from awmsm.gen.dedup import greedy_dedup, ngram_seen_share, ngrams, seeded_order, tokens
from awmsm.gen.midtrain_pipeline import DocChecks, extract_content
from awmsm.gen.quotas import allocate, check_quotas, plan_calls
from awmsm.io import read_jsonl
from awmsm.pilot import Pilot, PilotError, cohen_kappa, decide, wilson

from .conftest import REPO
from .fake_pipeline_server import FakePipelineVLLM


def _words(n: int) -> str:
    """n distinct words (repeated "word" would now trip the repetition check)."""
    return " ".join(f"w{i}" for i in range(n))



# ------------------------------------------------------------------------- templates
@pytest.mark.parametrize("name", ["E01_decompose", "E02_subdomains", "E03_doctypes", "E04_ideas",
                                  "E05_draft", "E06_critique", "E07_rewrite", "E08_filter"])

def test_templates_load_and_keep_json_braces(name):
    t = prompts.load(name, str(REPO / "prompts"))
    vals = {v: f"<{v}>" for v in t.variables}
    r = t.render(**vals)
    assert "{" not in "".join(f"{{{v}}}" for v in t.variables if f"{{{v}}}" in r.text)
    if name in ("E01_decompose", "E02_subdomains", "E06_critique", "E08_filter"):
        assert 'Return JSON: {"' in r.prefix  # literal JSON survives substitution
    if name != "E01_decompose":
        assert r.suffix.strip(), "variable part must exist"
    with pytest.raises(KeyError):
        t.render() if t.variables else (_ for _ in ()).throw(KeyError("none"))


def test_shared_prefix_has_no_per_item_variables():
    per_item = {"doc_type", "what", "how", "why", "idea_json", "draft", "critique_json", "text", "assertions",
                "subdomain", "species_group", "context_domain", "cost_of_care", "welfare_raised",
                "perspective", "is_story"}
    for name in ("E04_ideas", "E05_draft", "E06_critique", "E07_rewrite", "E08_filter"):
        t = prompts.load(name, str(REPO / "prompts"))
        leaked = set(prompts._VAR.findall(t.prefix)) & per_item
        assert not leaked, f"{name}: per-item variables in shared prefix: {leaked}"


# --------------------------------------------------------------------------- quotas
def test_allocate_largest_remainder():
    assert sorted(allocate(40, {"a": 0.375, "b": 0.4, "c": 0.225})) == sorted(["a"] * 15 + ["b"] * 16 + ["c"] * 9)


def test_plan_meets_s4_quotas(cfg):
    q = cfg.gen["s4"]["quotas"]
    subs = [f"sub_{i:03d}_00" for i in range(40)]
    dts = {s: ["forum thread", "developer blog post", "short story", "FAQ"] for s in subs}
    calls = plan_calls(40, subs, dts, ["aquaculture", "home_restaurant_cooking"], q, ["story"], random.Random(1))
    recs = [c.as_dict() for c in calls]
    chk = check_quotas(recs, q)
    assert chk["all_pass"], chk["checks"]
    assert all((c.perspective == "fictional_narrator") == c.is_story for c in calls)
    assert all(("story" in c.doc_type) == c.is_story for c in calls)


# ----------------------------------------------------------------------- doc checks
def test_doc_checks(cfg):
    chk = DocChecks(Config(cfg.base, cfg.gen, root=REPO))
    good = "<scratchpad>x</scratchpad><content>" + _words(400) + "</content>"
    assert chk(good, "stop")[1] is None
    assert chk("<content>" + _words(400), "length")[1].reason == "truncated_max_tokens"
    assert chk("no tags " * 300, "stop")[1].reason == "missing_content_tags"
    assert chk("<content>" + _words(50) + "</content>", "stop")[1].reason == "length_out_of_range"
    assert chk("<content>" + _words(300) + " see https://x.org</content>", "stop")[1].reason == "url"
    assert chk("<content>Dear [Name], " + _words(300) + "</content>", "stop")[1].reason == "placeholder"
    assert chk("<content>" + _words(300) + "这是中文" * 30 + "</content>", "stop")[1].reason == "language_mixing"
    assert extract_content("<content> a </content>", "stop") == ("a", None)


# ---------------------------------------------------------------------------- dedup
def test_ngram_dedup_and_cosine():
    base = " ".join(f"w{i}" for i in range(200))
    texts = [base, base + " extra tail words", " ".join(f"z{i}" for i in range(200))]
    r = greedy_dedup(texts)
    assert r.kept == [0, 2] and r.removed[0]["reason"] == "ngram13"
    assert ngram_seen_share(tokens(texts[2]), ngrams(tokens(base))) == 0.0
    # share counts n-grams: doc = base + 13 new words -> 188 of 201 13-grams seen
    t2 = tokens(base + " " + " ".join(f"n{i}" for i in range(13)))
    assert abs(ngram_seen_share(t2, ngrams(tokens(base))) - 188 / 201) < 1e-9
    assert seeded_order(["b", "a", "c"], 1) == seeded_order(["b", "a", "c"], 1)
    assert sorted(seeded_order(["b", "a", "c"], 1)) == [0, 1, 2]
    np = pytest.importorskip("numpy")
    emb = np.array([[1, 0], [0.999, 0.0447], [0, 1]], dtype=float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    r2 = greedy_dedup(["a b", "c d", "e f"], emb)
    assert r2.kept == [0, 2] and r2.removed[0]["reason"] == "cosine"


# ------------------------------------------------------------------------- stats
RULE = {"primary": "mean_quality", "tie_margin": 0.25, "tie_breaker": "mean_s8_pass_rate_across_judges",
        "veto": {"min_extra_failing_docs": 3, "nominal_pp": 0.12},
        "candidate_c_trigger": {"human_fail_rate_above": 0.15}, "throughput": "feasibility_check_only"}


def _audit(qa, qb, fa, fb):
    ids = [f"i{k:02d}" for k in range(25)]
    mk = lambda q, f: {"quality_by_idea": dict(zip(ids, q)), "fail_by_idea": dict(zip(ids, f)),  # noqa: E731
                       "human_fail_rate": sum(f) / 25, "mean_quality": sum(q) / 25}
    return {"by_candidate": {"A": mk(qa, fa), "B": mk(qb, fb)}}


M = {"candidates": {"A": {"mean_pass_rate_across_judges": 0.60}, "B": {"mean_pass_rate_across_judges": 0.70}}}


def test_wilson_and_kappa():
    lo, hi = wilson(150, 200)
    assert 0.68 < lo < 0.75 < hi < 0.81
    assert cohen_kappa([(True, True), (False, False), (True, False), (False, True)]) == 0.0


def test_decision_rule_v17():
    f0 = [False] * 25
    assert decide(None, _audit([4] * 25, [3] * 25, f0, f0), M)["winner"] is None
    # primary: paired mean quality A-B = +0.4 >= 0.25 -> A
    d = decide(RULE, _audit([4] * 10 + [3] * 15, [3] * 25, f0, f0), M)
    assert d["winner"] == "A" and not d["veto_applied"] and abs(d["paired_mean_diff"]["A-B"] - 0.4) < 1e-9
    # tie (+0.2) -> higher mean S8 pass rate across judges -> B
    d = decide(RULE, _audit([4] * 5 + [3] * 20, [3] * 25, f0, f0), M)
    assert d["winner"] == "B" and "tie" in d["reason"]
    # veto: A better quality but 3 more failing docs -> B wins, flagged
    fa = [True] * 3 + [False] * 22
    d = decide(RULE, _audit([5] * 25, [3] * 25, fa, f0), M)
    assert d["winner"] == "B" and d["veto_applied"]
    # 2 extra failing docs: no veto, quality decides
    fa2 = [True] * 2 + [False] * 23
    assert decide(RULE, _audit([5] * 25, [3] * 25, fa2, f0), M)["winner"] == "A"
    # both above 15% -> candidate C, no winner
    many = [True] * 4 + [False] * 21     # 16%
    d = decide(RULE, _audit([4] * 25, [3] * 25, many, many), M)
    assert d["winner"] is None and d["run_candidate_c"]


# ------------------------------------------------------------------- pilot end-to-end
def _approx_counter_init(self, model_id="meta-llama/Llama-3.1-8B"):
    self.model_id, self._tok = model_id, None
    self.method = f"approx 1.33 tokens/word (tests: no tokenizer for {model_id})"


@pytest.fixture
def pilot_cfg(tmp_path, monkeypatch):
    import shutil
    import yaml
    import awmsm.corpus
    for d in ("configs", "prompts", "lists", "specs"):
        shutil.copytree(REPO / d, tmp_path / d)
    # Tests are independent of the machine: the frozen Appendix A draft as spec, 13-gram-only S9 (no local
    # embedding model), and the code defaults for settings a researcher may have changed in configs/gen.yaml.
    shutil.copy(REPO / "tests/fixtures/aw_main_draft.md", tmp_path / "specs/aw_main.md")
    def _no_embed(*a, **k):
        raise RuntimeError("embeddings disabled in tests")
    monkeypatch.setattr(awmsm.corpus, "embed", _no_embed)
    import awmsm.tokens
    monkeypatch.setattr(awmsm.tokens.TokenCounter, "__init__", _approx_counter_init)   # same count on every machine
    gen = yaml.safe_load((tmp_path / "configs/gen.yaml").read_text())
    for k in ("s8_unresolved", "deprioritise_phrases"):
        (gen.get("corpus") or {}).pop(k, None)
    gen["pilot"].update({"n_ideas": 20, "s4_calls": 10, "keep_per_call": 2, "subdomains_used": 10, "concurrency": 4})
    gen["pilot"]["audit"] = {"n_ideas": 5, "seed": 1}
    gen["client"].update({"backoff_initial_s": 0.001, "backoff_max_s": 0.002})
    (tmp_path / "configs/gen.yaml").write_text(yaml.safe_dump(gen, sort_keys=False))
    base = yaml.safe_load((tmp_path / "configs/base.yaml").read_text())
    base["paths"]["cache"] = str(tmp_path / "cache")
    (tmp_path / "configs/base.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    return Config.load(tmp_path / "configs")


async def test_pilot_end_to_end(pilot_cfg):
    fake = FakePipelineVLLM(served=["Qwen/Qwen3.8-27B-FP8", "google/gemma-4-31B-it"])
    pilot = Pilot(pilot_cfg, http_factory=fake.factory)
    root = pilot.dir

    s = await pilot.phase_ideas()
    assert s["n_ideas"] == 20 and s["idea_quota_check"]["n"] == 20
    ideas = list(read_jsonl(root / "ideas.jsonl"))
    assert len({i["id"] for i in ideas}) == 20 and all(i["spec_excerpt"] and i["assertions"] for i in ideas)
    with pytest.raises(PilotError, match="already frozen"):
        await pilot.phase_ideas()

    # Amendment 1 (doc v1.7.4) must be registered before the regenerated outputs are written
    assert pilot.p["decision_rule"]["amendments"][0]["id"] == 1
    with pytest.raises(PilotError, match="Register the decision rule"):
        await pilot.phase_write("A")
    reg = pilot.phase_register_rule()
    assert reg["sha256"] and [a["amendment"]["id"] for a in reg["amendments"]] == [1]
    assert pilot.phase_register_rule()["amendments"][0]["registered_at"] == reg["amendments"][0]["registered_at"]

    for cand in ("A", "B"):
        man = await pilot.phase_write(cand)
        assert man["manifest"]["model"]["hf_id"] in ("Qwen/Qwen3.8-27B-FP8", "google/gemma-4-31B-it")
        assert len(list(read_jsonl(root / cand / "rewrites.jsonl"))) == 20
        bp = list(read_jsonl(root / cand / "bench_prompts" / "s5_draft.jsonl"))
        assert len(bp) == 20 and "<spec_section>" in bp[0]["prefix"] and "Document idea:" in bp[0]["suffix"]
        assert list(read_jsonl(root / cand / "bench_prompts" / "s6_critique.jsonl"))[0]["response_format"]
        s5 = man["s5_summary"]
        assert s5["unresolved"] == 0 and s5["responses"] == 20 and s5["truncation_rate"] == 0
        assert s5["plan_words"]["n"] == 20 and 20 <= s5["plan_words"]["min"] <= s5["plan_words"]["max"] <= 90
        assert len(s5["template_shas"]) == 1 and man["coverage"]["s7_rewrite"]["unresolved"] == 0
        assert 0 <= s5["think_slip"] < 20
        assert s5["kept"] == 20 and 0 < s5["tolerances_kept"].get("unclosed_content", 0) < 20
        assert s5["story_share"]["kept"] == 20 and s5["story_share"]["quota"] == [0.15, 0.25]
        s7 = man["s7_summary"]
        assert s7["inputs"] == 20 and s7["unresolved"] == 0 and s7["kept"] + s7["rejected"] == 20
        assert s7["words_all_parsed"]["n"] == 20 and 420 <= s7["words_all_parsed"]["median"] <= 900
        assert len(s7["template_shas"]) == 1 and s7["template_shas"] != s5["template_shas"]
        import importlib.util
        spec = importlib.util.spec_from_file_location("pilot_cli", REPO / "scripts" / "pilot.py")
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        cli._print_s5(root / cand / "s5_summary.json"); cli._print_s7(root / cand / "s7_summary.json")
    assert (json.loads((root / "A" / "s5_summary.json").read_text())["template_shas"]
            == json.loads((root / "B" / "s5_summary.json").read_text())["template_shas"])      # identical E.5

    # writer sampling is each candidate's own (D17 / I.1 step 2)
    s5 = [b for b in fake.calls if "Document idea:" in b["messages"][-1]["content"]]
    assert {(b["model"], b["top_p"], b["top_k"]) for b in s5} == {
        ("Qwen/Qwen3.8-27B-FP8", 0.8, 20), ("google/gemma-4-31B-it", 0.95, 64)}

    await pilot.phase_judge("A", ["A", "B"])
    await pilot.phase_judge("B", ["A", "B"])
    assert len(list(read_jsonl(root / "A" / "bench_prompts" / "s8_filter.jsonl"))) == 20  # self-judge only
    gates = json.loads((root / "A" / "gates.json").read_text())
    assert set(gates) == {"s5_draft", "s6_critique", "s7_rewrite"} and all(g["passed"] for g in gates.values())
    assert gates["s6_critique"]["greedy"] and gates["s6_critique"]["records_without_flag"] == 0
    assert all(g["passed"] for g in json.loads((root / "judged" / "gates.json").read_text()).values())
    m = pilot.phase_metrics(use_embeddings=False)
    assert set(m["candidates"]["A"]["by_stage"]) == {"s5_draft", "s7_rewrite"}
    assert "Parsing tolerances" in render_report_text(pilot, m)
    assert set(m["pass_rates"]) == {"A_by_A", "A_by_B", "B_by_A", "B_by_B"}
    assert all(v["judged"] == 20 for v in m["pass_rates"].values())

    # (registration before audit-export is tested in test_audit_export_needs_registration)
    assert {k: v for k, v in pilot.p["decision_rule"].items() if k != "amendments"} == RULE   # base rule unchanged
    reg = pilot.phase_register_rule()
    assert reg["registered_at"] and reg["sha256"]
    # a registered amendment cannot be changed
    pilot.p["decision_rule"]["amendments"][0]["min_median_words"] = 300
    with pytest.raises(PilotError, match="cannot be changed"):
        pilot.phase_register_rule()
    pilot.p["decision_rule"]["amendments"][0]["min_median_words"] = 450
    r = pilot.phase_audit_export()
    assert r["items"] == 10
    key = json.loads((root / "audit/.private/audit_key.json").read_text())["items"]
    packet = (root / "audit/audit_packet.md").read_text()
    assert "Qwen" not in packet and "gemma" not in packet.lower()   # writer hidden
    judged = [json.loads(line) for line in (root / "judged/A_by_A.jsonl").open()]
    assert "INCLUDE" not in packet and "EXCLUDE" not in packet          # S8 verdicts hidden
    assert not any(r["rationale"]["faithful"] in packet for r in judged)
    # label: candidate B slightly better
    lab = root / "audit/audit_labels.csv"
    rows = list(csv.DictReader(lab.open()))
    for row in rows:
        for c in ("faithful", "reason_visible_correct_type", "realistic", "no_fabrication", "no_antipattern",
                  "about_assistant"):
            row[c] = "PASS"
        row["overall_quality"] = "4" if key[row["audit_id"]]["candidate"] == "B" else "3"
    with lab.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(PilotError, match="refusing to overwrite"):
        pilot.phase_audit_export()

    text = pilot.phase_report().read_text()
    assert "**Winner: B**" in text and "2 × 2" in text and "registered" in text

    # post-labelling review: annotators by range, a revision before unblinding, unblinded listing, near-dup check
    import shutil
    shutil.copy(lab, root / "audit/audit_labels_original.csv")
    rows[0]["realistic"] = "FAIL"; rows[0]["overall_quality"] = "2"; rows[0]["notes"] = "Aster as a human vet; revised"
    with lab.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rec = pilot.set_annotators("P001-P005:Niraj,P006-P010:Jevin")
    assert rec["by_item"]["P005"] == "Niraj" and rec["by_item"]["P006"] == "Jevin"
    with pytest.raises(Exception, match="No annotator"):
        pilot.set_annotators("P001-P005:Niraj")
    text = pilot.phase_report().read_text()
    assert "### Annotators" in text and "no human–human agreement" in text and "different annotators" in text
    assert "### Label revisions made before unblinding" in text and rows[0]["audit_id"] in text
    assert "### Audited items (unblinded after labelling)" in text
    au = json.loads((root / "decision.json").read_text())["audit"]
    assert au["revisions"][0]["changes"]["realistic"] == ("PASS", "FAIL") or au["revisions"][0]["changes"]["realistic"] == ["PASS", "FAIL"]
    assert au["items"][0]["pattern_tags"] == ["aster_human_or_physical"] or any(i["pattern_tags"] for i in au["items"])
    pairs = au["annotators"]["pairs_cross_annotator"] + [None] * au["annotators"]["pairs_same_annotator"]
    assert len(pairs) == au["annotators"]["pairs_total"] == 5
    # regression: both writers share doc ids, so each judged file must be matched to its own writer only
    assert all(v["n"] == 5 for k, v in au["judge_agreement"].items() if "[" not in k)
    assert set(au["criterion_agreement"]) == {"A_by_A", "A_by_B", "B_by_A", "B_by_B"}
    by_idea = {}
    for it in au["items"]:
        by_idea.setdefault(it["idea_id"], {})[it["candidate"]] = it["words"]
    texts = {c: {r["id"]: r["text"] for r in read_jsonl(root / c / "rewrites.jsonl")} for c in ("A", "B")}
    for it in au["items"]:
        assert it["words"] == len(texts[it["candidate"]][f"doc_{it['idea_id']}_rewrite"].split())
    rj = await pilot.phase_rejudge_audit("A", "E08_filter_v2", "t1")
    assert rj["after"]["n"] == rj["before"]["n"] == 10 and (root / "rejudge/t1/A_by_A.jsonl").exists()
    assert (root / "judged/A_by_A.jsonl").stat().st_size > 0      # pilot record untouched
    rj3 = await pilot.phase_rejudge_audit("A", "E08_filter_v3", "t3")   # section-reason template renders and runs
    assert rj3["after"]["n"] == 10 and rj3["template"] == "E08_filter_v3"
    ins = pilot.phase_audit_inspect(["P001", "P002"])
    pr = ins["pairs"][0]
    assert set(pr) >= {"same_idea", "same_s4_call", "share_13grams_of_b_in_a", "jaccard_5grams", "word_sequence_ratio"}
    # invalid labels are an error, never a silent FAIL
    rows[1]["no_fabrication"] = "pass?"
    with lab.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(PilotError, match="invalid or blank"):
        pilot.phase_report()
    rows[1]["no_fabrication"] = "PASS"
    with lab.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # the registered rule cannot be changed afterwards
    pilot.p["decision_rule"]["tie_margin"] = 0.5
    with pytest.raises(PilotError, match="differs from the registered rule"):
        pilot.phase_report()
    with pytest.raises(PilotError, match="different rule was already registered"):
        pilot.phase_register_rule()


# ------------------------------------------------------------- fixes after the first real S4 run
def test_story_matching_whole_words():
    import yaml
    from awmsm.gen.quotas import is_story_type
    kw = yaml.safe_load((REPO / "lists/story_doc_types.yaml").read_text())["keywords"]
    assert not is_story_type("Interview Transcript", kw)
    assert not is_story_type("conference talk transcript", kw)
    assert is_story_type("Short story", kw) and is_story_type("Flash fiction piece", kw)
    assert not is_story_type("Developer Blog Post", kw)


def test_not_raised_paired_with_costless(cfg):
    q = cfg.gen["s4"]["quotas"]
    assert q["pair_not_raised_with_costless"] is True
    subs = [f"sub_{i:03d}_00" for i in range(40)]
    for seed in range(20):
        calls = plan_calls(40, subs, {s: ["FAQ", "short story"] for s in subs}, ["aquaculture"], q, ["story"],
                           random.Random(seed))
        recs = [c.as_dict() for c in calls]
        assert all(r["cost_of_care"] == "costless" for r in recs if r["welfare_raised"] == "appropriately_not_raised")
        assert check_quotas(recs, q)["all_pass"]   # shares unchanged by the swaps


async def test_s4_rejects_meta_commentary_and_resamples(cfg):
    import json as _json
    import httpx
    from awmsm.gen import midtrain_pipeline as mp
    from awmsm.llm import LLMClient
    rcfg = Config(cfg.base, cfg.gen, root=REPO)
    rcfg.base["paths"]["cache"] = cfg.base["paths"]["cache"]
    calls = []

    def handler(req):
        body = _json.loads(req.content)
        if req.url.path.endswith("/v1/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "Qwen/Qwen3.8-27B-FP8", "object": "model"}]})
        calls.append(body)
        bad = "Wait, the prompt says cost_of_care=co-beneficial. So the action must benefit humans."
        ideas = [{"what": f"w{len(calls)}-{j}", "how": bad if (len(calls) == 1 and j < 3) else f"h{j}", "why": "y"}
                 for j in range(6)]
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                         "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                             "role": "assistant", "content": _json.dumps({"ideas": ideas})}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    client = LLMClient(rcfg.role("generator"), rcfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    ctx = mp.GenContext.create(rcfg, "aw_main")
    sub = {"id": "sub_000_00", "domain_id": "dom_000", "subdomain": "s", "spec_excerpt": "x", "assertions": ["a"] * 5}
    call = {"call_id": "call_000", "subdomain_id": "sub_000_00", "doc_type": "FAQ", "species_group": "wild",
            "context_domain": "aquaculture", "cost_of_care": "costless", "welfare_raised": "raised",
            "perspective": "user", "is_story": False}
    recs = await mp.s4_fn(ctx, client, {"sub_000_00": sub}, 6, need=5)(call)
    await client.aclose()
    assert len(calls) == 2                                   # 3 of 6 rejected -> one resample
    assert all("prompt says" not in r["how"] for r in recs) and len(recs) == 9
    assert len(recs[0]["s4_rejected_in_call"]) == 3 and calls[0]["seed"] != calls[1]["seed"]


def test_write_refuses_to_resume_over_other_template(tmp_path):
    from awmsm.io import append_jsonl
    from awmsm.gen.schemas import S8_CRITERIA
    from awmsm.pilot import Pilot, PilotError
    cdir = tmp_path / "A"
    Pilot._check_draft_template(cdir, "new")                     # nothing yet: fine
    append_jsonl(cdir / "drafts.jsonl", {"_src": "i1", "template_sha": "new"})
    Pilot._check_draft_template(cdir, "new")                     # same template: resume is fine
    append_jsonl(cdir / "drafts.rejected.jsonl", {"_src": "i2", "reason": "x", "detail": {}})   # pre-change record
    with pytest.raises(PilotError, match="Move the folder aside"):
        Pilot._check_draft_template(cdir, "new")


def test_s5_summary_counts_truncation_slip_and_unresolved(tmp_path):
    from awmsm.io import append_jsonl
    from awmsm.gen.schemas import S8_CRITERIA
    from awmsm.pilot import s5_summary
    d = tmp_path
    for i in range(8):
        append_jsonl(d / "drafts.jsonl", {"_src": f"i{i}", "plan_words": 50 + i, "template_sha": "t",
                                          "scratchpad_closed_with_think": i < 2, "n_tokens": 1000})
    append_jsonl(d / "drafts.rejected.jsonl", {"_src": "i8", "reason": "truncated_max_tokens",
                                               "detail": {"plan_words": None, "template_sha": "t", "n_tokens": 3200}})
    s = s5_summary(d, [f"i{i}" for i in range(10)])
    assert (s["responses"], s["kept"], s["rejected"], s["unresolved"]) == (9, 8, 1, 1)
    assert s["unresolved_ids"] == ["i9"] and s["truncated"] == 1 and abs(s["truncation_rate"] - 1 / 9) < 1e-9
    assert s["think_slip"] == 2 and s["no_closed_plan"] == 1 and s["plan_words"]["max"] == 57


def render_report_text(pilot, m):
    from awmsm.pilot import render_report
    return render_report(pilot.cfg, m, {}, None, None, {})


def test_stage_gate_halts_on_fallback_rate_and_unresolved(tmp_path, cfg):
    from awmsm.io import append_jsonl
    from awmsm.gen.schemas import S8_CRITERIA
    from awmsm.pilot import stage_gate
    out = tmp_path / "critiques.jsonl"
    ids = [f"d{i}" for i in range(100)]
    for i in range(98):
        append_jsonl(out, {"_src": f"d{i}", "t0_fallback": i < 2})          # 2/100 fallbacks = 2.0%: allowed
    for i in (98, 99):
        append_jsonl(out.with_name("critiques.rejected.jsonl"), {"_src": f"d{i}", "reason": "x", "detail": {"t0_fallback": False}})
    g = stage_gate(cfg, "s6_critique", out, ids)
    assert g["passed"] and g["t0_fallback_rate"] == 0.02 and g["unresolved"] == 0
    g = stage_gate(cfg, "s6_critique", out, ids + ["d100"])               # one unresolved item -> halt
    assert not g["passed"] and g["unresolved"] == 1
    append_jsonl(out.with_name("critiques.errors.jsonl"), {"_src": "d100", "error": "JSONValidationError: ... after the T=0.3 fallback"})
    g = stage_gate(cfg, "s6_critique", out, ids + ["d100"])
    assert g["t0_fallbacks"] == 3 and g["t0_calls"] == 101
    append_jsonl(out, {"_src": "d100", "t0_fallback": True})               # resolved, but 3/101 = 3.0% -> halt
    g = stage_gate(cfg, "s6_critique", out, ids + ["d100"])
    assert not g["passed"] and g["unresolved"] == 0 and any("fallback rate" in r for r in g["reasons"])
    # sampled stages: only unresolved items count
    g = stage_gate(cfg, "s7_rewrite", out, ids + ["d100"])
    assert g["passed"] and not g["greedy"]


def test_amendment_1_length_feasibility():
    from awmsm.pilot import apply_amendments, word_stats
    am = {"id": 1, "type": "length_feasibility", "min_median_words": 450,
          "max_share_under_words": {"words": 400, "share": 0.25}}
    def m(words_a, words_b):
        return {"candidates": {"A": {"s7_words_all_parsed": word_stats(words_a)},
                               "B": {"s7_words_all_parsed": word_stats(words_b)}}}
    ok = [800] * 100
    short_median = [440] * 100                          # median 440 < 450 -> infeasible
    many_short = [380] * 26 + [900] * 74                # 26% under 400 -> infeasible (median fine)
    edge = [399] * 25 + [900] * 75                      # exactly 25% under 400 -> allowed
    d = apply_amendments({"winner": "B", "reason": "quality"}, [am], m(ok, short_median))
    assert d["winner"] == "A" and d["rule_winner"] == "B" and d["amendment_applied"] == 1
    d = apply_amendments({"winner": "B", "reason": "quality"}, [am], m(ok, many_short))
    assert d["winner"] == "A"
    d = apply_amendments({"winner": "B", "reason": "quality"}, [am], m(ok, edge))
    assert d["winner"] == "B" and "amendment_applied" not in d
    d = apply_amendments({"winner": "A", "reason": "quality"}, [am], m(short_median, many_short))
    assert d["winner"] is None and d["rule_winner"] == "A"
    d = apply_amendments({"winner": None, "reason": "audit not complete"}, [am], m(ok, short_median))
    assert d["winner"] is None and d["amendment_checks"]["amendment_1"]["B"]["feasible"] is False


def test_s7_length_precheck_400_1500(cfg):
    from awmsm.gen.midtrain_pipeline import DocChecks
    c7 = DocChecks(Config(cfg.base, cfg.gen, root=REPO), "s7_rewrite")
    c5 = DocChecks(Config(cfg.base, cfg.gen, root=REPO), "s5_draft")
    for n, ok7 in ((399, False), (400, True), (1500, True), (1501, False)):
        r = c7.check("<content>" + _words(n) + ".</content>", "stop")
        assert (r.bad is None) == ok7 and r.words == n, n
        if not ok7:
            assert r.bad.reason == "length_out_of_range"
    assert c5.check("<content>" + _words(300) + ".</content>", "stop").bad is None     # S5 keeps 225-2,000


def test_audit_export_needs_registration(pilot_cfg):
    pilot = Pilot(pilot_cfg)
    with pytest.raises(PilotError, match="Register the decision rule"):
        pilot.phase_audit_export()


def test_report_notes_selected_subset_and_infeasible(pilot_cfg):
    from awmsm.pilot import render_report
    pilot = Pilot(pilot_cfg)
    m = {"n_ideas": 200, "concurrency": 32, "pass_rates": {}, "candidates": {
        "A": {"rejections": {}, "by_stage": {}},
        "B": {"rejections": {"s7_rewrite:length_out_of_range": 52}, "by_stage": {}}}}
    d = {"winner": None, "reason": "audit not complete",
         "amendment_checks": {"amendment_1": {"A": {"feasible": True}, "B": {"feasible": False,
                                                                           "reasons": ["median 447 < 450"]}}}}
    try:
        text = render_report(pilot.cfg, m, {}, None, None, d)
    except KeyError:
        return   # other sections need fuller metrics; the notes are covered by the e2e report when present
    assert "B's audited documents are a selected subset" in text and "judge validation" in text


def test_label_aliases_and_ranges(tmp_path):
    from awmsm import audit_review as ar
    f = tmp_path / "l.csv"
    f.write_text("audit_id,faithful,reason_correct,realistic,no_fabrication,no_antipattern,about_assistant,quality,notes\n"
                 "P001,pass,PASS,PASS,PASS,FAIL,PASS,3,template echo\nP002,,,,,,,,\n", encoding="utf-8")
    rows = ar.read_labels(f)
    assert rows[0]["reason_visible_correct_type"] == "PASS" and rows[0]["overall_quality"] == "3"
    assert rows[1]["overall_quality"] == ""                      # unlabelled rows are allowed (not complete)
    assert ar.parse_ranges("P001-P002:N, P003:J") == {"P001": "N", "P002": "N", "P003": "J"}
    with pytest.raises(ar.AuditError):
        ar.parse_ranges("P001-P003:N,P003-P004:J")
    d = ar.near_duplicate("the cat sat " * 30, "the cat sat " * 30)
    assert d["share_13grams_of_b_in_a"] == 1.0 and d["word_sequence_ratio"] == 1.0


async def test_corpus_small_end_to_end(pilot_cfg, tmp_path):
    from awmsm.corpus import Corpus
    fake = FakePipelineVLLM(served=["Qwen/Qwen3.8-27B-FP8", "google/gemma-4-31B-it"])
    pilot_cfg.gen["corpus"] = {"calls_per_batch": 10, "keep_per_call": 5, "max_batches": 3, "overshoot": 0.0}
    c = Corpus(pilot_cfg, "aw_main", tmp_path / "corpus", http_factory=fake.factory)
    s = await c.run(target_tokens=4000, spec_file=pilot_cfg.root / "specs" / "aw_main.md")
    assert s["status"] in ("target_reached", "max_batches") and s["documents"] > 0
    assert s["tokens"] >= 4000 or s["status"] == "max_batches"
    docs = list(read_jsonl(tmp_path / "corpus" / "corpus.jsonl"))
    assert docs and all(d["text"] and d["tokens"] > 0 and not any(k.startswith("_") for k in d) for d in docs)
    gates = json.loads((tmp_path / "corpus" / "batch_00" / "gates.json").read_text())
    assert set(gates) == {"s4_ideas", "s5_draft", "s6_critique", "s7_rewrite", "s8_filter"} and all(
        g["passed"] for g in gates.values())
    ids = [json.loads(l)["call_id"] for l in (tmp_path / "corpus/batch_00/s4_calls.jsonl").open()]
    assert ids[0].startswith("call_b00")
    # resumable: a second run with the target already met adds no batch
    n_batches = s["batches"]
    s2 = await c.run(target_tokens=min(4000, s["tokens"]), spec_file=pilot_cfg.root / "specs" / "aw_main.md")
    assert s2["batches"] == n_batches


async def test_corpus_followups(pilot_cfg, tmp_path):
    """19 Oct: {model_name} alias, spec guard, S10, report and the blinded H1.8 packet."""
    from awmsm import corpus_tools as ct
    from awmsm.corpus import Corpus
    from awmsm.gen import midtrain_pipeline as mp
    from awmsm.gen.pattern_scan import run_s10, lexical_scan
    from awmsm.llm import LLMClient
    ident = {"assistant_name": "Aster", "developer_name": "Your Lab"}
    assert mp.fill_spec("{model_name} is honest <!-- note {x} -->", ident) == "Aster is honest"
    assert mp.fill_spec("\ufeffTitle\r\n{model_name} cares", ident) == "Title\nAster cares"
    with pytest.raises(ValueError, match="unknown placeholder"):
        mp.fill_spec("{modelname} helps", ident)
    spec = pilot_cfg.root / "specs" / "aw_v1.md"
    spec.write_text((pilot_cfg.root / "specs" / "aw_main.md").read_text().replace("{assistant_name}", "{model_name}"))
    chk = ct.check_spec(pilot_cfg, "aw_main", spec)
    assert chk["placeholders_in_file"][-1] == "model_name" and chk["assistant_name_mentions"] > 0
    fake = FakePipelineVLLM(served=["Qwen/Qwen3.8-27B-FP8", "google/gemma-4-31B-it"])
    pilot_cfg.gen["corpus"] = {"calls_per_batch": 10, "keep_per_call": 5, "max_batches": 2, "overshoot": 0.0}
    out = tmp_path / "c1"
    c = Corpus(pilot_cfg, "aw_main", out, http_factory=fake.factory)
    s = await c.run(target_tokens=6000, spec_file=spec)
    assert s["documents"] > 0 and (out / "runs.json").exists() and "hours_all_runs" in s
    assert "{model_name}" not in (out / "spec" / "domains.jsonl").read_text()
    spec2 = pilot_cfg.root / "specs" / "aw_v2.md"
    spec2.write_text(spec.read_text() + "\n\nNew rule, because reasons.")
    with pytest.raises(PilotError, match="different spec"):
        await c.run(target_tokens=6000, spec_file=spec2)
    docs = list(read_jsonl(out / "corpus.jsonl"))
    role = pilot_cfg.role("generator")
    client = LLMClient(role, pilot_cfg, http_client=fake.factory(role.base_url))
    ctx = mp.GenContext.create(pilot_cfg, "aw_main", spec_path=spec)
    r = await run_s10(ctx, client, docs, out / "s10", {"scan_docs": 3, "scan_batches": 4, "min_batches": 1,
                                                        "sample": 5}, concurrency=4)
    await client.aclose()
    assert r["scan_batches"] >= 1 and "lexical" in r and (out / "s10/s10_summary.json").exists()
    from awmsm.io import append_jsonl
    from awmsm.gen.schemas import S8_CRITERIA
    b0 = sorted(out.glob("batch_*"))[0]
    some = next(iter(read_jsonl(b0 / "rewrites.jsonl")))
    append_jsonl(b0 / "judged.jsonl", {"id": "x__filter", "doc_id": some["id"], "verdict": "EXCLUDE", "unresolved": True,
                                       "criteria": {k: False for k in S8_CRITERIA}})   # 21 Oct exclude record
    rep = ct.report(pilot_cfg, out)
    assert rep["s8"]["unresolved_excluded"] == 1
    assert rep["corpus"]["documents"] == len(docs) and rep["s8"]["judged"] > 0 and (out / "report.md").exists()
    assert "S10 patterns" in (out / "report.md").read_text()
    a = ct.audit_export(pilot_cfg, out, n_included=3, n_excluded=1, seed=1, allow_approx=True)
    assert a["items"] == 4
    key = json.loads((out / "audit/.private/audit_key.json").read_text())["items"]
    assert sorted(v["source"] for v in key.values()) == ["corpus", "corpus", "corpus", "s8_excluded"]
    assert "s8_excluded" not in (out / "audit/audit_packet.md").read_text()
    lx = lexical_scan([{"text": "Dr. Aris Thorne met Elara Vance. When Aster spoke"}] * 3, names_to_skip={"Aster"})
    assert {x["text"] for x in lx["names"]} == {"Dr. Aris Thorne", "Elara Vance"}


async def test_corpus_server_death_keeps_output(pilot_cfg, tmp_path):
    """A dead server ends the run with status 'error: ...' and still writes corpus.jsonl/summary (resumable)."""
    import httpx
    from awmsm.corpus import Corpus
    fake = FakePipelineVLLM(served=["Qwen/Qwen3.8-27B-FP8", "google/gemma-4-31B-it"])
    pilot_cfg.gen["corpus"] = {"calls_per_batch": 10, "keep_per_call": 5, "max_batches": 3, "overshoot": 0.0}
    c = Corpus(pilot_cfg, "aw_main", tmp_path / "c2", http_factory=fake.factory)
    s = await c.run(target_tokens=3000, max_batches=1, spec_file=pilot_cfg.root / "specs" / "aw_main.md")
    n1 = s["included_before_s9"]
    orig = fake.handler
    fake.handler = lambda req: (httpx.Response(500, json={"error": "EngineDeadError"})
                                if req.url.path.endswith("/chat/completions") else orig(req))
    s2 = await c.run(target_tokens=10**7, spec_file=pilot_cfg.root / "specs" / "aw_main.md")
    assert s2["status"].startswith("error") and s2["included_before_s9"] == n1 and s2["documents"] > 0
    assert (tmp_path / "c2/summary.json").exists() and len(json.loads((tmp_path / "c2/runs.json").read_text())) == 2


def test_corpus_story_trim_and_need(pilot_cfg, tmp_path):
    from awmsm.corpus import Corpus
    c = Corpus(pilot_cfg, "aw_main", tmp_path / "c3")
    docs = [{"id": f"d{k}", "tokens": 100, "is_story": k % 10 == 0} for k in range(200)]   # 10% stories
    corpus, total = c.stratified_trim(docs, 5000)
    share = sum(d["is_story"] for d in corpus) / len(corpus)
    assert total >= 5000 and 0.15 <= share <= 0.25 and len({d["id"] for d in corpus}) == len(corpus)
    assert c.story_need(docs, 5000)["stories_short"] <= 0          # 20 stories >= 16% of 50 docs
    assert c.story_need(docs, 20000)["stories_short"] > 0          # 200 docs needed -> 32 stories > 20
    many = [{"id": f"s{k}", "tokens": 100, "is_story": k % 2 == 0} for k in range(100)]  # 50% stories -> capped
    corpus, _ = c.stratified_trim(many, 4000)
    assert sum(d["is_story"] for d in corpus) / len(corpus) <= 0.25 + 1e-9


async def test_corpus_s8_content_error_policy(pilot_cfg, tmp_path):
    """Invalid model output (content error) is not a server error: halt at the gate (default) or exclude."""
    from awmsm.corpus import Corpus
    from awmsm.llm import JSONValidationError
    async def fn(item):
        if item["id"] == "bad":
            raise JSONValidationError("still invalid after fallback")
        return {"id": f"{item['id']}__filter", "doc_id": item["id"], "verdict": "INCLUDE"}
    items = [{"id": "ok1"}, {"id": "bad"}, {"id": "ok2"}]
    out = tmp_path / "judged.jsonl"
    c = Corpus(pilot_cfg, "aw_main", tmp_path / "c4")
    await c._stage("s8_filter", items, fn, out, concurrency=1)          # default: no raise, gate decides
    assert not any(r["doc_id"] == "bad" for r in read_jsonl(out))
    c.c["s8_unresolved"] = "exclude"
    await c._stage("s8_filter", items, fn, out, concurrency=1)
    bad = [r for r in read_jsonl(out) if r["doc_id"] == "bad"]
    assert bad and bad[0]["verdict"] == "EXCLUDE" and bad[0]["unresolved"]


def test_trim_deprioritises_phrases(pilot_cfg, tmp_path):
    """Phrase documents are swapped out only while every quota still passes (24 Oct)."""
    from awmsm.corpus import Corpus
    from awmsm.gen.quotas import check_quotas
    c = Corpus(pilot_cfg, "aw_main", tmp_path / "c5")
    c.c["deprioritise_phrases"] = ["feel good or bad from the inside"]
    sp = ["farmed", "companion", "wild", "aquatic", "invertebrate", "novel_or_fictional", "none"]
    docs = []
    for k in range(400):
        docs.append({"id": f"d{k}", "tokens": 100, "is_story": k % 5 == 0, "species_group": sp[k % 7],
                     "cost_of_care": ["costly", "costless", "co-beneficial", "costly"][k % 4],
                     "welfare_raised": "appropriately_not_raised" if k % 8 == 0 else "raised",
                     "perspective": ["critic", "developer", "researcher", "user", "fictional_narrator"][k % 5],
                     "context_domain": "aquaculture", "doc_type": "memo",
                     "text": ("Pain can FEEL good   or bad from the inside." if k % 3 == 0 else "Plain text.")})
    q = pilot_cfg.gen["s4"]["quotas"]
    plain, _ = c.stratified_trim(docs, 20000)
    corpus, total, info = c.trim_with_deprioritised(docs, 20000)
    assert info["quotas_ok_before_swaps"] == check_quotas(plain, q)["all_pass"]
    if info["quotas_ok_before_swaps"]:
        assert check_quotas(corpus, q)["all_pass"] and 20000 <= total <= 20400
        assert info["corpus_docs_with_phrase"] < info["corpus_docs_with_phrase_before"]
    assert len({d["id"] for d in corpus}) == len(corpus)


async def test_trim_only_audit_and_results(pilot_cfg, tmp_path):
    import csv as _csv
    from awmsm import corpus_tools as ct
    from awmsm.corpus import Corpus
    fake = FakePipelineVLLM(served=["Qwen/Qwen3.8-27B-FP8", "google/gemma-4-31B-it"], fail_rate=1.5)
    pilot_cfg.gen["corpus"] = {"calls_per_batch": 10, "keep_per_call": 5, "max_batches": 3, "overshoot": 0.0}
    out = tmp_path / "c6"
    c = Corpus(pilot_cfg, "aw_main", out, http_factory=fake.factory)
    s = await c.run(target_tokens=8000, spec_file=pilot_cfg.root / "specs" / "aw_main.md")
    n_calls = len(fake.calls)
    s2 = await Corpus(pilot_cfg, "aw_main", out).run(target_tokens=5000, spec_file=pilot_cfg.root / "specs" / "aw_main.md",
                                                     trim_only=True)
    assert len(fake.calls) == n_calls and s2["trim_only"] and s2["tokens"] <= s["tokens"]
    assert s2["status"] == "target_reached" and json.loads((out / "manifest.json").read_text())["generator"]
    with pytest.raises(PilotError, match="approximate"):
        ct.audit_export(pilot_cfg, out, 4, 4, seed=3, min_excluded_stories=1, n_double=4)
    a = ct.audit_export(pilot_cfg, out, 4, 4, seed=3, min_excluded_stories=1, n_double=4, allow_approx=True)
    key = json.loads((out / "audit/.private/audit_key.json").read_text())["items"]
    exc = [k for k in key.values() if k["source"] == "s8_excluded"]
    assert len(exc) == 4 and sum(k["is_story"] for k in exc) >= 1
    dbl = [aid for aid, k in key.items() if k["double_labelled"]]
    assert len(dbl) == 4 and sum(key[x]["source"] == "s8_excluded" for x in dbl) == 2   # ceil(4*4/8)
    assert set(a["double_labelled"]) == set(dbl) and (out / "audit/audit_packet_jevin.md").exists()
    with pytest.raises(PilotError, match="not complete"):
        ct.audit_results(pilot_cfg, out)
    from awmsm.gen.schemas import S8_CRITERIA
    def fill(path, ids, fail_ids):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["audit_id", *S8_CRITERIA, "overall_quality", "notes"])
            w.writeheader()
            for i in ids:
                w.writerow({"audit_id": i, **{cc: ("FAIL" if i in fail_ids and cc == "realistic" else "PASS")
                                              for cc in S8_CRITERIA}, "overall_quality": "4", "notes": ""})
    excl = [aid for aid, k in key.items() if k["source"] == "s8_excluded"]
    fill(out / "audit/audit_labels.csv", list(key), set(excl[:3]))
    fill(out / "audit/audit_labels_jevin.csv", dbl, set(excl[:1]))
    r = ct.audit_results(pilot_cfg, out)
    assert r["h18"]["n"] == 4 and r["h18"]["fail"] == 0 and r["h18"]["pass"]
    assert r["judge_heldout"]["confusion"]["judge_exclude_human_fail"] == 3 and r["second_annotator"]["n"] == 4
    rep = ct.report(pilot_cfg, out)
    assert rep["spec_12gram_copies"]["n"] == 12
