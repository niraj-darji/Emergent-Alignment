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
    good = "<scratchpad>x</scratchpad><content>" + "word " * 400 + "</content>"
    assert chk(good, "stop")[1] is None
    assert chk("<content>" + "word " * 400, "length")[1].reason == "truncated_max_tokens"
    assert chk("no tags " * 300, "stop")[1].reason == "missing_content_tags"
    assert chk("<content>" + "word " * 50 + "</content>", "stop")[1].reason == "length_out_of_range"
    assert chk("<content>" + "word " * 300 + " see https://x.org</content>", "stop")[1].reason == "url"
    assert chk("<content>Dear [Name], " + "word " * 300 + "</content>", "stop")[1].reason == "placeholder"
    assert chk("<content>" + "word " * 300 + "这是中文" * 30 + "</content>", "stop")[1].reason == "language_mixing"
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
@pytest.fixture
def pilot_cfg(tmp_path):
    import shutil
    import yaml
    for d in ("configs", "prompts", "lists", "specs"):
        shutil.copytree(REPO / d, tmp_path / d)
    gen = yaml.safe_load((tmp_path / "configs/gen.yaml").read_text())
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

    for cand in ("A", "B"):
        man = await pilot.phase_write(cand)
        assert man["manifest"]["model"]["hf_id"] in ("Qwen/Qwen3.8-27B-FP8", "google/gemma-4-31B-it")
        assert len(list(read_jsonl(root / cand / "rewrites.jsonl"))) == 20
        bp = list(read_jsonl(root / cand / "bench_prompts" / "s5_draft.jsonl"))
        assert len(bp) == 20 and "<spec_section>" in bp[0]["prefix"] and "Document idea:" in bp[0]["suffix"]
        assert list(read_jsonl(root / cand / "bench_prompts" / "s6_critique.jsonl"))[0]["response_format"]

    # writer sampling is each candidate's own (D17 / I.1 step 2)
    s5 = [b for b in fake.calls if "Document idea:" in b["messages"][-1]["content"]]
    assert {(b["model"], b["top_p"], b["top_k"]) for b in s5} == {
        ("Qwen/Qwen3.8-27B-FP8", 0.8, 20), ("google/gemma-4-31B-it", 0.95, 64)}

    await pilot.phase_judge("A", ["A", "B"])
    await pilot.phase_judge("B", ["A", "B"])
    assert len(list(read_jsonl(root / "A" / "bench_prompts" / "s8_filter.jsonl"))) == 20  # self-judge only
    m = pilot.phase_metrics(use_embeddings=False)
    assert set(m["pass_rates"]) == {"A_by_A", "A_by_B", "B_by_A", "B_by_B"}
    assert all(v["judged"] == 20 for v in m["pass_rates"].values())

    with pytest.raises(PilotError, match="Register the decision rule"):
        pilot.phase_audit_export()
    pilot.p["decision_rule"] = copy.deepcopy(RULE)
    reg = pilot.phase_register_rule()
    assert reg["registered_at"] and reg["sha256"]
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
    # the registered rule cannot be changed afterwards
    pilot.p["decision_rule"]["tie_margin"] = 0.5
    with pytest.raises(PilotError, match="differs from the registered rule"):
        pilot.phase_report()
    with pytest.raises(PilotError, match="different rule was already registered"):
        pilot.phase_register_rule()
