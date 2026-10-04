from __future__ import annotations

import csv
import json
import random

import pytest

from awmsm import prompts
from awmsm.config import Config
from awmsm.gen.dedup import greedy_dedup, ngram_covered_share, ngrams, tokens
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
    assert ngram_covered_share(tokens(texts[2]), ngrams(tokens(base))) == 0.0
    np = pytest.importorskip("numpy")
    emb = np.array([[1, 0], [0.999, 0.0447], [0, 1]], dtype=float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    r2 = greedy_dedup(["a b", "c d", "e f"], emb)
    assert r2.kept == [0, 2] and r2.removed[0]["reason"] == "cosine"


# ------------------------------------------------------------------------- stats
def test_wilson_and_kappa_and_decision():
    lo, hi = wilson(150, 200)
    assert 0.68 < lo < 0.75 < hi < 0.81
    assert cohen_kappa([(True, True), (False, False), (True, False), (False, True)]) == 0.0
    audit = {"by_candidate": {"A": {"mean_quality": 3.9, "human_fail_rate": 0.2},
                              "B": {"mean_quality": 3.8, "human_fail_rate": 0.1}}}
    m = {"candidates": {"A": {"mean_pass_rate_across_judges": 0.6}, "B": {"mean_pass_rate_across_judges": 0.7}}}
    assert decide(None, audit, m)["winner"] is None
    assert decide({"primary": "mean_quality", "tie_margin": 0.05}, audit, m)["winner"] == "A"
    assert decide({"primary": "mean_quality", "tie_margin": 0.25}, audit, m)["winner"] == "B"  # tie -> pass rate
    assert decide({"primary": "human_fail_rate", "tie_margin": 0.05}, audit, m)["winner"] == "B"


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

    r = pilot.phase_audit_export()
    assert r["items"] == 10
    key = json.loads((root / "audit/.private/audit_key.json").read_text())["items"]
    packet = (root / "audit/audit_packet.md").read_text()
    assert "Qwen" not in packet and "gemma" not in packet.lower()   # writer hidden
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

    out = pilot.phase_report()
    text = out.read_text()
    assert "NOT SET" in text and "none yet" in text      # no winner without a pre-registered rule
    pilot.p["decision_rule"] = {"primary": "mean_quality", "tie_margin": 0.25}
    text = pilot.phase_report().read_text()
    assert "**Winner: B**" in text and "2 × 2" in text
