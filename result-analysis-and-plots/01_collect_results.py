"""Step 1: collect every per-arm number from the raw result files.

Reads only raw outputs (judge JSONs, lm-eval JSONs, ETHICS/HH JSONs, run manifests,
training logs), never the hand-written summary markdowns.

Writes
  tables/all_results.json   - per arm: metrics, per-item judge scores, dimension/pressure breakdowns
  tables/all_results.csv    - one row per arm, scalar metrics only
  tables/training_stats.csv - AFT and midtraining steps and held-out losses
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from common import (ARMS, ARM, ROOT, TABLES, aft_log, ethics_file, hh_file, judge_file,
                    keyword_file, lmeval_files)

ETHICS_TASKS = ["ethics_cm", "ethics_deontology", "ethics_justice", "ethics_utilitarianism", "ethics_virtue"]


def collect_arm(arm: str) -> dict:
    meta = ARM[arm]
    ck = meta["ckpt"]
    out: dict = {k: meta[k] for k in ("arm", "label", "family", "mt_tokens", "p", "ckpt")}

    # --- LLM-judge welfare benchmarks
    jf = judge_file(arm)
    if jf:
        d = json.load(open(jf))
        an, ma = d["anima"], d["manta"]
        out["anima_items"] = {str(i["id"]): i["pass_rate"] for i in an["items"]}
        out["anima_item_dims"] = {str(i["id"]): i["applicable_dims"] for i in an["items"]}
        out["anima_item_dim_rates"] = {str(i["id"]): i["dimension_pass_rates"] for i in an["items"]}
        out["manta_items"] = {str(i["id"]): i["stability_rate"] for i in ma["items"]}
        out["manta_item_pressure"] = {str(i["id"]): i["pressure"] for i in ma["items"]}
        out["anima"] = 100 * sum(out["anima_items"].values()) / len(out["anima_items"])
        out["manta"] = sum(out["manta_items"].values()) / len(out["manta_items"])
        out["anima_dims"] = an["dimension_scores"]
        out["manta_pressure"] = ma["pressure_breakdown"]

    # --- lm-eval benchmarks (0-shot, log-likelihood)
    for f in lmeval_files(arm, ck):
        d = json.load(open(f))
        r = d["results"]
        if "mmlu" in r:
            out["mmlu"] = 100 * r["mmlu"]["acc,none"]
            out["mmlu_nshot"] = d.get("n-shot", {}).get("mmlu_abstract_algebra")
        if "hellaswag" in r:
            out["hellaswag"] = 100 * r["hellaswag"]["acc_norm,none"]
        if "bbq" in r:
            b = r["bbq"]
            out["bbq_acc"] = 100 * b["acc,none"]
            out["bbq_amb_acc"] = 100 * b["accuracy_amb,none"]
            out["bbq_disamb_acc"] = 100 * b["accuracy_disamb,none"]
            out["bbq_amb_bias"] = b["amb_bias_score,none"]
            out["bbq_disamb_bias"] = b["disamb_bias_score,none"]
        if "bigbench_hhh_alignment_multiple_choice" in r:
            out["hhh"] = 100 * r["bigbench_hhh_alignment_multiple_choice"]["acc,none"]

    # --- ETHICS
    ef = ethics_file(arm, ck)
    if ef:
        d = json.load(open(ef))
        r, ns = d["results"], d.get("n-samples", {})
        for t in ETHICS_TASKS:
            if t in r:
                key = t.replace("ethics_", "ethics_")
                out[key] = 100 * r[t]["acc,none"]
                n = ns.get(t, {})
                out[key + "_n"] = n.get("effective", n) if isinstance(n, dict) else n
        if all(t in out for t in ETHICS_TASKS):
            out["ethics_macro"] = sum(out[t] for t in ETHICS_TASKS) / 5

    # --- Anthropic HH (harmless-base)
    hf = hh_file(arm, ck)
    if hf:
        d = json.load(open(hf))
        out["hh_raw"] = d["harmless_pref_acc_raw"]
        out["hh_norm"] = d["harmless_pref_acc_norm"]
        out["hh_n"] = d["total_samples"]

    # --- keyword (regex) heuristic scorer, exploratory only
    kf = keyword_file(arm, ck)
    if kf:
        d = json.load(open(kf))
        out["kw_anima"] = d["anima"].get("overall_welfare_stance_pct")
        out["kw_manta"] = d["manta"].get("awvs_value_stability")

    # --- fine-tuning run record + held-out loss before/after
    man = ROOT / "runs/aft" / ck / "manifest.json"
    if man.exists():
        m = json.load(open(man))
        out["aft_steps"] = m.get("steps_completed")
        out["aft_tokens"] = m.get("tokens_trained")
    lf = aft_log(ck)
    if lf:
        txt = open(lf, errors="ignore").read()
        m0 = re.search(r"Initial held-out evaluation: loss=([0-9.]+)", txt)
        m1 = re.findall(r"heldout_loss=([0-9.]+)", txt)
        if m0:
            out["aft_loss_before"] = float(m0.group(1))
        if m1:
            out["aft_loss_after"] = float(m1[-1])
    return out


def collect_midtraining() -> list[dict]:
    rows = []
    for run, corpus, tokens in [("midtrain_aw_main_v1_m1", "v1", 2.5), ("midtrain_aw_main_v1_5m_m1", "v1", 5),
                                ("midtrain_aw_main_v1_10m_m1", "v1", 10), ("midtrain_aw_main_v1_20m_m1", "v1", 20),
                                ("midtrain_caml_public_m1", "caml", 2.5), ("midtrain_qwen27_clean_m1", "v2", 4.55)]:
        d = ROOT / "runs/midtrain" / run
        man = json.load(open(d / "manifest.json"))
        logf = Path("/mnt/HDD_2TB/niraj/logs") / f"{run}.log"
        if not logf.exists():
            logf = Path("/mnt/HDD_2TB/niraj/logs/v2_clean") / f"{run}.log"
        txt = open(logf, errors="ignore").read()
        before = float(re.search(r"Initial held-out evaluation: loss=([0-9.]+)", txt).group(1))
        rows_csv = list(csv.DictReader(open(d / "logs/train_log.csv")))
        after = [float(r["heldout_loss"]) for r in rows_csv if r["heldout_loss"]][-1]
        rows.append({"run": run, "corpus": corpus, "mt_tokens": tokens, "steps": man["steps_completed"],
                     "tokens_trained": man["tokens_trained"], "loss_before": before, "loss_after": after})
    return rows


def main():
    allres = {a[0]: collect_arm(a[0]) for a in ARMS}
    json.dump(allres, open(TABLES / "all_results.json", "w"), indent=1)

    scalar_keys = ["arm", "label", "family", "mt_tokens", "p", "anima", "manta", "mmlu", "hellaswag", "bbq_acc",
                   "bbq_amb_acc", "bbq_disamb_acc", "bbq_amb_bias", "bbq_disamb_bias", "hhh", *ETHICS_TASKS,
                   "ethics_macro", "hh_raw", "hh_norm", "kw_anima", "kw_manta", "aft_steps", "aft_tokens",
                   "aft_loss_before", "aft_loss_after"]
    with open(TABLES / "all_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        w.writeheader()
        for a in allres.values():
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items() if k in scalar_keys})

    mt = collect_midtraining()
    with open(TABLES / "training_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mt[0].keys()))
        w.writeheader()
        w.writerows(mt)

    # quick console summary
    print(f"{'arm':9s} {'ANIMA':>6s} {'MANTA':>6s} {'MMLU':>6s} {'ETHICS':>7s} {'HHnorm':>7s} {'steps':>5s} {'loss0':>6s} {'loss1':>6s}")
    for a in allres.values():
        g = lambda k, fmt: (fmt % a[k]) if k in a and a[k] is not None else "-"
        print(f"{a['arm']:9s} {g('anima','%6.2f')} {g('manta','%6.3f')} {g('mmlu','%6.2f')} {g('ethics_macro','%7.2f')} "
              f"{g('hh_norm','%7.2f')} {g('aft_steps','%5d')} {g('aft_loss_before','%6.3f')} {g('aft_loss_after','%6.3f')}")
    for r in mt:
        print(r)


if __name__ == "__main__":
    main()
