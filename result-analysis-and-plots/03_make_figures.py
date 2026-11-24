"""Step 3: result figures and LaTeX tables for the paper.

Main-text figures
  fig_welfare_vs_p          ANIMA and MANTA vs AW share p, per midtraining condition (95% CIs)
  fig_paired_effects        forest plot of paired differences with Holm-adjusted significance
  fig_dose                  midtraining dose (p = 0): ANIMA, MANTA, MMLU, fine-tuning-data loss
  fig_ethics_vs_p           ETHICS subtasks vs AW share p (small multiples, +/- 1 SE)
  fig_generalization        welfare vs human-directed moral reasoning across arms
  fig_loss_head_start       fine-tuning held-out loss before and after AFT, raw vs MSM
Appendix figures
  fig_anima_dimensions      ANIMA per-dimension scores (heatmap)
  fig_manta_pressure        MANTA by pressure type
Tables (LaTeX, in tables/)
  tab_capability.tex        capability and human-directed safety
  tab_full_results.tex      every arm, every metric (appendix)
"""
from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from common import COLOR, FAMILY_LABEL, INK, MARKER, SEQ_BLUE, TABLES, TEXTWIDTH_IN, save, setup_matplotlib

setup_matplotlib()
R = json.load(open(TABLES / "all_results.json"))
CI = {r["arm"]: {k: float(v) for k, v in r.items() if k not in ("arm", "label")}
      for r in csv.DictReader(open(TABLES / "arm_cis.csv"))}
SE = {r["arm"]: {k: float(v) for k, v in r.items() if k != "arm" and v != ""}
      for r in csv.DictReader(open(TABLES / "mc_standard_errors.csv"))}
PAIRS = list(csv.DictReader(open(TABLES / "paired_comparisons.csv")))
CORR = {(r["welfare"], r["other"]): r for r in csv.DictReader(open(TABLES / "correlations.csv"))}

LINES = {  # family -> arms ordered by p
    "raw": ["a0", "m05", "m15", "a1", "run1"],
    "v1": ["a2", "m_msm05", "m_msm15", "a3", "run2"],
    "v2": ["a2_v2", "m05_v2", "m15_v2", "a3_v2"],
    "caml": ["a5"],
}
OFFSET = {"raw": -0.9, "v1": 0.0, "v2": 0.9, "caml": 1.6}   # small x-jitter so error bars do not overlap


def legend_handles(families):
    return [Line2D([0], [0], color=COLOR[f], marker=MARKER[f], lw=1.5 if f != "caml" else 0,
                   markersize=4.5, markeredgecolor="white", markeredgewidth=0.6, label=FAMILY_LABEL[f])
            for f in families]


def style_point(ax, x, y, fam, yerr=None, size=4.5, hollow=False):
    ax.errorbar(x, y, yerr=yerr, fmt=MARKER[fam], color=COLOR[fam], ms=size, mec="white", mew=0.6,
                mfc="white" if hollow else COLOR[fam], elinewidth=0.8, capsize=0, zorder=3)


# ============================================================================ fig_welfare_vs_p
def fig_welfare_vs_p():
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.05))
    for ax, metric, ylab, scale in [(axes[0], "anima", "ANIMA score (%)", 1), (axes[1], "manta", "MANTA score", 1)]:
        for fam, arms in LINES.items():
            if fam == "caml":   # only one CaML arm (p=30%); a single point is not a curve
                continue
            arms = [a for a in arms if metric in R[a]]
            xs = np.array([R[a]["p"] for a in arms], float) + OFFSET[fam]
            ys = np.array([R[a][metric] for a in arms])
            lo = np.array([CI[a][f"{metric}_lo"] for a in arms])
            hi = np.array([CI[a][f"{metric}_hi"] for a in arms])
            if len(arms) > 1:
                ax.plot(xs, ys, color=COLOR[fam], lw=1.5, zorder=2)
            style_point(ax, xs, ys, fam, yerr=[ys - lo, hi - ys])
        ax.set_xlabel("AW share $p$ in fine-tuning mix (%)")
        ax.set_ylabel(ylab)
        ax.set_xticks([0, 5, 15, 30])
        ax.set_xlim(-3, 34)
    axes[0].set_ylim(5, 32)
    axes[1].set_ylim(0.25, 0.85)
    axes[0].set_title("(a) ANIMA", loc="left", fontweight="bold")
    axes[1].set_title("(b) MANTA (turn 1)", loc="left", fontweight="bold")
    fig.legend(handles=legend_handles(["raw", "v1", "v2"]), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.07), handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_welfare_vs_p")
    plt.close(fig)


# ============================================================================ fig_paired_effects
def fig_paired_effects():
    groups = []
    for r in PAIRS:
        if not groups or groups[-1][0] != r["group"]:
            groups.append((r["group"], []))
        groups[-1][1].append(r)
    ypos, ylabels, y = [], [], 0
    rows = []
    header_y = []
    for g, items in groups:
        header_y.append((y, g))
        y += 1
        for r in items:
            rows.append((y, r))
            ylabels.append(f"{r['label']}:  {r['desc']}")
            ypos.append(y)
            y += 1
        y += 0.4
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 0.155 * y + 0.5), sharey=True)
    for ax, metric, xlab in [(axes[0], "anima", r"$\Delta$ ANIMA (percentage points)"),
                             (axes[1], "manta", r"$\Delta$ MANTA")]:
        ax.axvline(0, color=INK["axis"], lw=0.8, zorder=1)
        for yy, r in rows:
            d, lo, hi = float(r[f"{metric}_d"]), float(r[f"{metric}_lo"]), float(r[f"{metric}_hi"])
            sig = float(r[f"{metric}_p_holm"]) < 0.05
            ax.plot([lo, hi], [yy, yy], color=COLOR["v1"], lw=1.2, solid_capstyle="round", zorder=2)
            ax.plot(d, yy, "o", ms=4.2, color=COLOR["v1"], mfc=COLOR["v1"] if sig else "white", mec=COLOR["v1"],
                    mew=1.0, zorder=3)
        ax.set_xlabel(xlab)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(ylabels, fontsize=6.6)
    for yy, g in header_y:
        axes[0].text(-0.02, yy, g, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                     fontsize=7, fontweight="bold", color=INK["primary"])
    axes[0].set_ylim(y - 0.6, -0.8)
    axes[0].tick_params(axis="y", length=0)
    handles = [Line2D([0], [0], marker="o", color=COLOR["v1"], mfc=COLOR["v1"], lw=1.2, ms=4.2,
                      label="significant after Holm correction ($p<0.05$)"),
               Line2D([0], [0], marker="o", color=COLOR["v1"], mfc="white", lw=1.2, ms=4.2,
                      label="not significant after correction")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.0 + 0.25 / (0.155 * y + 0.5)))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_paired_effects")
    plt.close(fig)


# ============================================================================ fig_dose
def fig_dose():
    arms = ["a2", "x2_5m", "x2_10m", "x2_20m"]
    tok = np.array([R[a]["mt_tokens"] for a in arms])
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 1.75))
    panels = [("anima", "ANIMA (%)", "(a) ANIMA"), ("manta", "MANTA", "(b) MANTA"),
              ("mmlu", "MMLU (%)", "(c) MMLU"), ("aft_loss_before", "loss (nats)", "(d) loss before FT")]
    for ax, (k, ylab, title) in zip(axes, panels):
        ys = np.array([R[a][k] for a in arms])
        if k in ("anima", "manta"):
            lo = np.array([CI[a][f"{k}_lo"] for a in arms])
            hi = np.array([CI[a][f"{k}_hi"] for a in arms])
            yerr = [ys - lo, hi - ys]
        elif k == "mmlu":
            yerr = 1.96 * np.array([SE[a]["mmlu_se"] for a in arms])
        else:
            yerr = None
        ax.plot(tok, ys, color=COLOR["v1"], lw=1.5, zorder=2)
        style_point(ax, tok, ys, "v1", yerr=yerr)
        ref = R["a0"][k]
        ax.axhline(ref, color=COLOR["raw"], lw=1.0, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2.5, 5, 10, 20])
        ax.set_xticklabels(["2.5", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xlabel("midtraining tokens (M)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].text(2.3, R["a0"]["anima"] + 0.25, "A0", ha="left", va="bottom", fontsize=6.3, color=INK["secondary"])
    fig.legend(handles=[Line2D([0], [0], color=COLOR["v1"], marker="o", lw=1.5, ms=4.5, mec="white",
                               label="MSM v1, $p=0$ (A2, X2-5M, X2-10M, X2-20M)"),
                        Line2D([0], [0], color=COLOR["raw"], lw=1.0, label="A0: no midtraining, $p=0$")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_dose")
    plt.close(fig)


# ============================================================================ fig_ethics_vs_p
def fig_ethics_vs_p():
    tasks = [("ethics_cm", "Commonsense"), ("ethics_deontology", "Deontology"), ("ethics_justice", "Justice"),
             ("ethics_utilitarianism", "Utilitarianism"), ("ethics_virtue", "Virtue"), ("ethics_macro", "Macro average")]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 3.0), sharex=True)
    for ax, (k, name) in zip(axes.flat, tasks):
        for fam, arms in LINES.items():
            arms = [a for a in arms if k in R[a]]
            if not arms:
                continue
            xs = np.array([R[a]["p"] for a in arms], float)
            ys = np.array([R[a][k] for a in arms])
            err = 1.96 * np.array([SE[a][k + "_se"] for a in arms])
            if len(arms) > 1:
                ax.plot(xs, ys, color=COLOR[fam], lw=1.5, zorder=2)
            style_point(ax, xs, ys, fam, yerr=err)
        ax.axhline(50, color=INK["muted"], lw=0.8, zorder=1)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 48), hi + 1)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_xticks([0, 30, 100])
        ax.set_xlim(-6, 106)
    axes[0, 0].text(104, 50.4, "chance", ha="right", va="bottom", fontsize=6.3, color=INK["muted"])
    for ax in axes[1]:
        ax.set_xlabel("AW share $p$ (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    fig.legend(handles=legend_handles(["raw", "v1", "v2", "caml"]), loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04), handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save(fig, "fig_ethics_vs_p")
    plt.close(fig)


# ============================================================================ fig_generalization
def fig_generalization():
    arms = [a for a in R if "anima" in R[a] and "ethics_macro" in R[a]]
    panels = [("ethics_macro", "ETHICS macro (%)", "(a) ETHICS macro"),
              ("deon_just", "mean of deon. & justice (%)", "(b) Deon. & justice"),
              ("ethics_utilitarianism", "utilitarianism (%)", "(c) Utilitarianism")]
    # per-panel label offsets (points) chosen to avoid collisions
    labels = {
        "ethics_macro": {"a0": (-3, -9), "a3": (-6, 4), "a2_v2": (4, -1), "m15_v2": (4, -1), "a5": (-4, 5)},
        "deon_just": {"a0": (-2, -9), "a3": (-6, 4), "a2_v2": (4, 0), "m15_v2": (4, -2), "a5": (4, 1)},
        "ethics_utilitarianism": {"a0": (4, -4), "a3": (-10, 4), "a2_v2": (4, -1), "m15_v2": (4, -3), "a5": (-10, 4)},
    }
    names = {"a0": "A0", "a3": "A3", "a2_v2": "A2-v2", "m15_v2": "M15-v2", "a5": "A5"}
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.1))
    for ax, (k, ylab, title) in zip(axes, panels):
        for a in arms:
            fam = R[a]["family"]
            y = (R[a]["ethics_deontology"] + R[a]["ethics_justice"]) / 2 if k == "deon_just" else R[a][k]
            style_point(ax, R[a]["anima"], y, fam, size=5)
            if a in labels[k]:
                ax.annotate(names[a], (R[a]["anima"], y), xytext=labels[k][a], textcoords="offset points",
                            fontsize=6.2, color=INK["secondary"])
        if k == "deon_just":
            c1, c2 = CORR[("anima", "ethics_deontology")], CORR[("anima", "ethics_justice")]
            sub = (rf"$\rho$ = {float(c1['spearman_rho']):+.2f} / {float(c2['spearman_rho']):+.2f}, "
                   rf"$p$ = {float(c1['perm_p']):.2f} / {float(c2['perm_p']):.2f}")
        else:
            c = CORR[("anima", k)]
            sub = rf"$\rho$ = {float(c['spearman_rho']):+.2f}, $p$ = {float(c['perm_p']):.2f}"
        sub = sub.replace("-", "\u2212")
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color=INK["secondary"])
        ax.set_xlabel("ANIMA (%)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
    fig.legend(handles=[Line2D([0], [0], color=COLOR[f], marker=MARKER[f], lw=0, ms=5, mec="white",
                               label={"raw": "No midtraining", "v1": "MSM v1 (2.5-20M)", "v2": "MSM v2",
                                      "caml": "CaML corpus"}[f]) for f in ["raw", "v1", "v2", "caml"]],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.74, wspace=0.42)
    save(fig, "fig_generalization")
    plt.close(fig)


# ============================================================================ fig_loss_head_start
def fig_loss_head_start():
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 1.85), sharey=True)
    for ax, k, title in [(axes[0], "aft_loss_before", "(a) before fine-tuning"),
                         (axes[1], "aft_loss_after", "(b) after fine-tuning")]:
        for fam, arms in LINES.items():
            arms = [a for a in arms if k in R[a]]
            xs = np.array([R[a]["p"] for a in arms], float)
            ys = np.array([R[a][k] for a in arms])
            if len(arms) > 1:
                ax.plot(xs, ys, color=COLOR[fam], lw=1.5, zorder=2)
            style_point(ax, xs, ys, fam)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("AW share $p$ (%)")
        ax.set_xticks([0, 15, 30, 100])
        ax.set_xlim(-5, 105)
    axes[0].set_ylabel("held-out loss on\nfine-tuning mix (nats)")
    fig.legend(handles=legend_handles(["raw", "v1", "v2", "caml"]), loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.09), handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_loss_head_start")
    plt.close(fig)


# ============================================================================ appendix: ANIMA dimensions
def fig_anima_dimensions():
    arms = ["a0", "a1", "a2", "a3", "a5", "m_msm15", "a2_v2", "a3_v2", "x2_20m"]
    dims = list(R["a0"]["anima_dims"].keys())
    n_items = {d: sum(d in v for v in R["a0"]["anima_item_dims"].values()) for d in dims}
    order = sorted(dims, key=lambda d: -n_items[d])
    M = np.array([[R[a]["anima_dims"][d] for a in arms] for d in order])
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd"] + SEQ_BLUE[2:])
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.4))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=70, aspect="auto")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([R[a]["label"] for a in arms], rotation=35, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{d} (n={n_items[d]})" for d in order])
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.2,
                    color="white" if v > 38 else INK["primary"])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("pass rate (%)")
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "fig_anima_dimensions")
    plt.close(fig)


# ============================================================================ appendix: MANTA pressure
def fig_manta_pressure():
    arms = [("a0", "raw"), ("a1", "raw"), ("a2", "v1"), ("a3", "v1")]
    press = ["economic", "social", "pragmatic", "cultural", "epistemic"]
    counts = {p: sum(p in v for v in R["a0"]["manta_item_pressure"].values()) for p in press}
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.8, 2.1))
    for i, (a, fam) in enumerate(arms):
        ys = np.arange(len(press)) + (i - 1.5) * 0.17
        xs = [R[a]["manta_pressure"][p] for p in press]
        hollow = R[a]["p"] == 0
        ax.plot(xs, ys, MARKER[fam], color=COLOR[fam], ms=4.5, mfc="white" if hollow else COLOR[fam],
                mec=COLOR[fam], mew=1.0, lw=0, label=f"{R[a]['label']} ({FAMILY_LABEL[fam].split(' (')[0]}, $p$={R[a]['p']}%)")
    ax.set_yticks(range(len(press)))
    ax.set_yticklabels([f"{p} (n={counts[p]})" for p in press])
    ax.invert_yaxis()
    ax.set_xlabel("MANTA score")
    ax.set_xlim(0.2, 0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_manta_pressure")
    plt.close(fig)


# ============================================================================ LaTeX tables
def fmt(v, nd=1):
    return "--" if v is None else f"{v:.{nd}f}"


def tab_capability():
    arms = ["a0", "a1", "a2", "a3", "a5", "x2_5m", "x2_10m", "x2_20m", "a2_v2", "m05_v2", "m15_v2", "a3_v2",
            "run1", "run2"]
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{BBQ} & & \multicolumn{2}{c}{Anthropic HH} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){8-9}",
             r"Arm ($p$) & MMLU & HellaSwag & amb.\ acc & disamb.\ acc & amb.\ bias $\downarrow$ & HHH & raw & norm. \\",
             r"\midrule"]
    for a in arms:
        d = R[a]
        g = lambda k, nd=1: fmt(d.get(k), nd)
        lines.append(f"{d['label']} ({d['p']}\\%) & {g('mmlu')} & {g('hellaswag')} & {g('bbq_amb_acc')} & "
                     f"{g('bbq_disamb_acc')} & {g('bbq_amb_bias', 3)} & {g('hhh')} & {g('hh_raw')} & {g('hh_norm')} \\\\")
        if a in ("a5", "x2_20m", "a3_v2"):
            lines.append(r"\midrule")
    lines += [r"\midrule",
              r"Approx.\ 95\% CI ($\pm$) & 0.8 & 0.8 & -- & -- & 0.011 & 6.5 & 2.0 & 2.0 \\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_capability.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/tab_capability.tex")


def tab_full_results():
    cols = [("anima", "ANIMA", 1), ("manta", "MANTA", 3), ("mmlu", "MMLU", 1), ("ethics_cm", "CM", 1),
            ("ethics_deontology", "Deon", 1), ("ethics_justice", "Just", 1), ("ethics_utilitarianism", "Util", 1),
            ("ethics_virtue", "Virt", 1), ("ethics_macro", "ETH", 1), ("aft_steps", "Upd.", 0)]
    order = ["a0", "m05", "m15", "a1", "run1", "a2", "m_msm05", "m_msm15", "a3", "run2", "x2_5m", "x2_10m",
             "x2_20m", "a2_v2", "m05_v2", "m15_v2", "a3_v2", "a5"]
    lines = [r"\begin{tabular}{lll" + "r" * len(cols) + "}", r"\toprule",
             "Arm & Midtraining & $p$ & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    mt = {"raw": "none", "v1": "v1 2.5M", "v2": "v2 4.6M", "caml": "CaML 2.5M"}
    for a in order:
        d = R[a]
        m = mt.get(d["family"], f"v1 {d['mt_tokens']:.0f}M")
        vals = []
        for k, _, nd in cols:
            v = d.get(k)
            vals.append("--" if v is None else (f"{int(v)}" if nd == 0 else f"{v:.{nd}f}"))
        lines.append(f"{d['label']} & {m} & {d['p']}\\% & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_full_results.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/tab_full_results.tex")


if __name__ == "__main__":
    fig_welfare_vs_p()
    fig_paired_effects()
    fig_dose()
    fig_ethics_vs_p()
    fig_generalization()
    fig_loss_head_start()
    fig_anima_dimensions()
    fig_manta_pressure()
    tab_capability()
    tab_full_results()
