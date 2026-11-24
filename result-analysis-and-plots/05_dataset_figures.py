"""Step 5: dataset figure for the appendix (reads tables/dataset_stats.json from step 4).

  fig_dataset  (a) AW demonstrations by category, (b) assistant tokens per conversation by source,
               (c) nominal AW share vs effective gradient weight of AW and identity data
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import COLOR, INK, TABLES, TEXTWIDTH_IN, save, setup_matplotlib

setup_matplotlib()
S = json.load(open(TABLES / "dataset_stats.json"))

fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.05, 1.0], hspace=0.65, wspace=0.55)

# (a) categories -------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
cats = S["categories"]
keys = list(cats.keys())[::-1]
vals = [cats[k]["count"] for k in keys]
ax.barh(range(len(keys)), vals, height=0.62, color=COLOR["v1"], edgecolor="white", linewidth=0.8)
ax.set_yticks(range(len(keys)))
ax.set_yticklabels([f"{k} {cats[k]['name']}" for k in keys], fontsize=6.5)
for i, v in enumerate(vals):
    ax.text(v + 8, i, str(v), va="center", fontsize=6.3, color=INK["secondary"])
ax.set_xlim(0, max(vals) * 1.25)
ax.set_xlabel("training conversations")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.set_title("(a) AW demos by category", loc="left", fontweight="bold")

# (b) lengths ----------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
names = ["Identity", "General instructions", "AW demonstrations"]
data = [S["lengths"][n] for n in names]
bp = ax.boxplot(data, orientation="horizontal", widths=0.5, showfliers=False, patch_artist=True,
                medianprops=dict(color=INK["primary"], lw=1.2),
                boxprops=dict(facecolor="#cde2fb", edgecolor=COLOR["v1"], lw=0.8),
                whiskerprops=dict(color=COLOR["v1"], lw=0.8), capprops=dict(color=COLOR["v1"], lw=0.8))
for i, d in enumerate(data):
    ax.plot(np.mean(d), i + 1, "D", ms=3.5, color=INK["primary"])
    ax.text(2300, i + 1, f"mean {np.mean(d):.0f}", va="center", fontsize=6.3,
            color=INK["secondary"])
ax.set_xscale("log")
ax.set_xlim(8, 9000)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(names, fontsize=6.8)
ax.set_xlabel("assistant tokens per conversation (log scale)")
ax.grid(axis="y", visible=False)
ax.set_title("(b) Reply length by source", loc="left", fontweight="bold")

# (c) effective weights --------------------------------------------------------
ax = fig.add_subplot(gs[1, :])
eff = S["effective_weights"]
mixes = ["p00", "p05", "p15", "p30", "p80", "p90", "p100"]
p = np.array([0, 5, 15, 30, 80, 90, 100])
aw_tok = [100 * eff[m]["token_share"]["aw"] for m in mixes]
aw_eff = [100 * eff[m]["effective_weight"]["aw"] for m in mixes]
id_tok = [100 * eff[m]["token_share"]["id"] for m in mixes]
id_eff = [100 * eff[m]["effective_weight"]["id"] for m in mixes]
ax.plot(p, aw_tok, color=INK["primary"], lw=0.9, marker="o", mfc="white", ms=4)
ax.plot(p, aw_eff, color=INK["primary"], lw=1.6, marker="o", ms=4)
ax.plot(p, id_tok, color=INK["muted"], lw=0.9, marker="s", mfc="white", ms=4)
ax.plot(p, id_eff, color=INK["muted"], lw=1.6, marker="s", ms=4)
ax.text(101.5, aw_tok[-1], "AW token share", va="center", fontsize=6.3, color=INK["secondary"])
ax.text(101.5, aw_eff[-1], "AW gradient weight", va="center", fontsize=6.3, color=INK["secondary"])
ax.text(101.5, id_eff[-1], "identity gradient weight", va="center", fontsize=6.3, color=INK["secondary"])
ax.text(101.5, id_tok[-1], "identity token share", va="center", fontsize=6.3, color=INK["secondary"])
ax.set_xticks(p)
ax.set_xlim(-3, 128)
ax.set_ylim(0, 100)
ax.set_xlabel("nominal AW share $p$ of the fine-tuning mix (%)")
ax.set_ylabel("share (%)")
ax.set_title("(c) Token share vs. effective share of the gradient (loss averaged per micro-batch of 2)",
             loc="left", fontweight="bold")
ax.legend(handles=[Line2D([0], [0], color=INK["primary"], lw=0.9, marker="o", mfc="white", ms=4,
                          label="share of supervised tokens"),
                   Line2D([0], [0], color=INK["primary"], lw=1.6, marker="o", ms=4,
                          label="effective share of the gradient")], loc="upper left")
save(fig, "fig_dataset")
