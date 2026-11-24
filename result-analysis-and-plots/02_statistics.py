"""Step 2: statistics.

  * per-arm 95% bootstrap CIs for ANIMA and MANTA (resampling questions, 10,000 draws)
  * paired arm-vs-arm comparisons on ANIMA and MANTA: mean item difference, 95% bootstrap CI,
    sign-flip permutation p-value, Holm-adjusted p-value (family = all comparisons, per metric)
  * standard errors for the multiple-choice benchmarks (binomial; BBQ bias via delta method)
  * across-arm Spearman correlations between welfare and human-directed / capability metrics

Writes tables/arm_cis.csv, tables/paired_comparisons.csv, tables/mc_standard_errors.csv,
tables/correlations.csv. Seed fixed (0) so every number is reproducible.
"""
from __future__ import annotations

import csv
import json
import math

import numpy as np

from common import TABLES

B = 10_000
RNG = np.random.default_rng(0)
R = json.load(open(TABLES / "all_results.json"))

# ----------------------------------------------------------------------------- per-arm CIs
def boot_mean_ci(values: np.ndarray):
    idx = RNG.integers(0, len(values), (B, len(values)))
    means = values[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


rows = []
for arm, d in R.items():
    if "anima_items" not in d:
        continue
    a = np.array(list(d["anima_items"].values())) * 100
    m = np.array(list(d["manta_items"].values()))
    alo, ahi = boot_mean_ci(a)
    mlo, mhi = boot_mean_ci(m)
    rows.append({"arm": arm, "label": d["label"], "anima": a.mean(), "anima_lo": alo, "anima_hi": ahi,
                 "manta": m.mean(), "manta_lo": mlo, "manta_hi": mhi})
with open(TABLES / "arm_cis.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ----------------------------------------------------------------------------- paired comparisons
COMPARISONS = [
    # group, a, b, description
    ("MSM v1 vs none", "a2", "a0", "p=0%"),
    ("MSM v1 vs none", "m_msm05", "m05", "p=5%"),
    ("MSM v1 vs none", "m_msm15", "m15", "p=15%"),
    ("MSM v1 vs none", "a3", "a1", "p=30%"),
    ("MSM v2 vs none", "a2_v2", "a0", "p=0%"),
    ("MSM v2 vs none", "m05_v2", "m05", "p=5%"),
    ("MSM v2 vs none", "m15_v2", "m15", "p=15%"),
    ("MSM v2 vs none", "a3_v2", "a1", "p=30%"),
    ("AW share", "a1", "a0", "no MSM, 30% vs 0%"),
    ("AW share", "a3", "a2", "MSM v1, 30% vs 0%"),
    ("AW share", "a3", "m_msm15", "MSM v1, 30% vs 15%"),
    ("AW share", "a3_v2", "a2_v2", "MSM v2, 30% vs 0%"),
    ("Corpus", "a5", "a1", "CaML vs none, p=30%"),
    ("Corpus", "a3", "a5", "v1 vs CaML, p=30%"),
    ("Corpus", "a2_v2", "a2", "v2 vs v1, p=0%"),
    ("Corpus", "m15_v2", "m_msm15", "v2 vs v1, p=15%"),
    ("Corpus", "a3_v2", "a3", "v2 vs v1, p=30%"),
    ("Dose (p=0%)", "x2_5m", "a2", "5M vs 2.5M"),
    ("Dose (p=0%)", "x2_10m", "a2", "10M vs 2.5M"),
    ("Dose (p=0%)", "x2_20m", "a2", "20M vs 2.5M"),
    ("Dose (p=0%)", "x2_20m", "x2_5m", "20M vs 5M, nested"),
]


def paired(x: dict, y: dict, scale: float):
    ids = sorted(set(x) & set(y), key=int)
    diff = np.array([x[i] - y[i] for i in ids]) * scale
    n = len(diff)
    means = diff[RNG.integers(0, n, (B, n))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    flips = RNG.choice([-1.0, 1.0], (B, n))
    perm = np.abs((flips * diff).mean(1))
    p = (np.sum(perm >= abs(diff.mean())) + 1) / (B + 1)
    return float(diff.mean()), float(lo), float(hi), float(p), n


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


out = []
for g, a, b, desc in COMPARISONS:
    ra = paired(R[a]["anima_items"], R[b]["anima_items"], 100)
    rm = paired(R[a]["manta_items"], R[b]["manta_items"], 1)
    out.append({"group": g, "a": a, "b": b, "label": f"{R[a]['label']} \u2212 {R[b]['label']}", "desc": desc,
                "anima_d": ra[0], "anima_lo": ra[1], "anima_hi": ra[2], "anima_p": ra[3],
                "manta_d": rm[0], "manta_lo": rm[1], "manta_hi": rm[2], "manta_p": rm[3]})
for metric in ("anima", "manta"):
    adj = holm(np.array([o[f"{metric}_p"] for o in out]))
    for o, v in zip(out, adj):
        o[f"{metric}_p_holm"] = float(v)
with open(TABLES / "paired_comparisons.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

# ----------------------------------------------------------------------------- MC standard errors
N = {"mmlu": 14042, "hellaswag": 10042, "hhh": 221, "hh_norm": 2312, "bbq_amb": 29246}
se_rows = []
for arm, d in R.items():
    row = {"arm": arm}
    for k in ("mmlu", "hellaswag", "hhh", "hh_norm"):
        if k in d:
            a = d[k] / 100
            row[k + "_se"] = 100 * math.sqrt(a * (1 - a) / N[k])
    for t in ("ethics_cm", "ethics_deontology", "ethics_justice", "ethics_utilitarianism", "ethics_virtue"):
        if t in d:
            a = d[t] / 100
            row[t + "_se"] = 100 * math.sqrt(a * (1 - a) / d[t + "_n"])
    if "ethics_macro" in row or all(t + "_se" in row for t in ("ethics_cm", "ethics_deontology", "ethics_justice",
                                                                 "ethics_utilitarianism", "ethics_virtue")):
        row["ethics_macro_se"] = math.sqrt(sum(row[t + "_se"] ** 2 for t in (
            "ethics_cm", "ethics_deontology", "ethics_justice", "ethics_utilitarianism", "ethics_virtue"))) / 5
    if "bbq_amb_bias" in d:
        # ambiguous bias = (1 - acc_amb) * (2 q - 1), q = share of non-unknown answers that follow the stereotype.
        # delta-method SE treating q as a binomial proportion over the non-unknown ambiguous answers.
        acc = d["bbq_amb_acc"] / 100
        n_non_unknown = N["bbq_amb"] * (1 - acc)
        q = (d["bbq_amb_bias"] / (1 - acc) + 1) / 2
        row["bbq_amb_bias_se"] = (1 - acc) * 2 * math.sqrt(q * (1 - q) / n_non_unknown)
    se_rows.append(row)
keys = sorted({k for r in se_rows for k in r}, key=lambda k: (k != "arm", k))
with open(TABLES / "mc_standard_errors.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(se_rows)

# ----------------------------------------------------------------------------- across-arm correlations
def spearman_perm(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    null = np.array([np.corrcoef(rx, RNG.permutation(ry))[0, 1] for _ in range(B)])
    p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (B + 1)
    return rho, float(p)


corr_rows = []
both = [a for a, d in R.items() if "anima" in d and "ethics_macro" in d]
for w_metric in ("anima", "manta"):
    for h in ("ethics_macro", "ethics_cm", "ethics_deontology", "ethics_justice", "ethics_utilitarianism",
              "ethics_virtue", "hh_norm", "mmlu", "bbq_amb_bias"):
        arms = [a for a in both if h in R[a]]
        x = np.array([R[a][w_metric] for a in arms])
        y = np.array([R[a][h] for a in arms])
        rho, p = spearman_perm(x, y)
        corr_rows.append({"welfare": w_metric, "other": h, "n_arms": len(arms), "spearman_rho": rho, "perm_p": p})
with open(TABLES / "correlations.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
    w.writeheader()
    w.writerows(corr_rows)

# ----------------------------------------------------------------------------- console
print("paired comparisons (ANIMA pp | MANTA), Holm-adjusted p:")
for o in out:
    print(f"  {o['label']:20s} {o['desc']:22s} {o['anima_d']:+6.2f} [{o['anima_lo']:+6.2f},{o['anima_hi']:+6.2f}] "
          f"pH={o['anima_p_holm']:.4f} | {o['manta_d']:+.3f} [{o['manta_lo']:+.3f},{o['manta_hi']:+.3f}] pH={o['manta_p_holm']:.4f}")
print("correlations across arms:")
for c in corr_rows:
    print(f"  {c['welfare']:6s} vs {c['other']:22s} n={c['n_arms']:2d} rho={c['spearman_rho']:+.2f} p={c['perm_p']:.3f}")
print("BBQ ambiguous-bias SE (A0):", [r.get("bbq_amb_bias_se") for r in se_rows if r["arm"] == "a0"])
