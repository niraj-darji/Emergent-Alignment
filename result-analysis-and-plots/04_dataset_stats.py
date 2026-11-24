"""Step 4: dataset statistics for the fine-tuning data (needs `transformers`; run with the awmsm venv).

  * AW-demonstration training set by category (C1-C8)
  * supervised (assistant) tokens per conversation for each source: AW demos, general, identity
  * effective gradient weight of each source per mix, given that the fine-tuning loss is averaged
    per micro-batch of 2 conversations (see train.py): w_c = E_m[ tokens from c in m / tokens in m ]

Writes tables/dataset_stats.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter

os.environ.setdefault("HF_HOME", "/mnt/HDD_2TB/niraj/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoTokenizer  # noqa: E402

from common import ROOT, TABLES  # noqa: E402

AFT = ROOT / "data/aft"
tok = AutoTokenizer.from_pretrained(str(ROOT / "runs/checkpoints/aft_a3_mix_p30_m1_a1"))
CATEGORY_NAMES = {
    "C1": "Explicit welfare question", "C2": "Implicit welfare stakes", "C3": "Welfare appropriately not raised",
    "C4": "Costly trade-off", "C5": "User autonomy", "C6": "Novel / uncertain beings",
    "C7": "Non-sentient control", "C8": "Third-party dilemma",
}


def sup_tokens(msgs):
    return sum(len(tok.encode(m["content"] + "<|end_of_text|>\n", add_special_tokens=False))
               for m in msgs if m["role"] == "assistant")


def read(path):
    return [json.loads(l) for l in open(path)]


demos = read(AFT / "aw_demos_v1/aw_demos_train.jsonl")
ident = read(AFT / "identity_v1/identity.jsonl")
general = read(AFT / "general_sft_10k.jsonl")
aw_ids = {d["id"] for d in demos}
id_ids = {d["id"] for d in ident}

cats = Counter(d["id"].split("_")[2] for d in demos)          # ids look like p_call_C2_0106_0
lengths = {"AW demonstrations": [sup_tokens(d["messages"]) for d in demos],
           "General instructions": [sup_tokens(d["messages"]) for d in general],
           "Identity": [sup_tokens(d["messages"]) for d in ident]}

eff = {}
rng = random.Random(0)
for mix in ["p00", "p05", "p15", "p30", "p80", "p90", "p100"]:
    rows = read(AFT / f"mix_{mix}.jsonl")
    train = [r for r in rows if (int(hashlib.sha256(f"0:{r['id']}".encode()).hexdigest(), 16) % 100000) / 1e5 >= 0.02]
    toks = [(sup_tokens(r["messages"]), "aw" if r["id"] in aw_ids else ("id" if r["id"] in id_ids else "gen"))
            for r in train]
    total = sum(t for t, _ in toks)
    share = {c: sum(t for t, s in toks if s == c) / total for c in ("aw", "id", "gen")}
    acc, n = Counter(), 0
    for _ in range(200):
        idx = list(range(len(toks)))
        rng.shuffle(idx)
        for i in range(0, len(idx) - 1, 2):
            (t1, c1), (t2, c2) = toks[idx[i]], toks[idx[i + 1]]
            acc[c1] += t1 / (t1 + t2)
            acc[c2] += t2 / (t1 + t2)
            n += 1
    eff[mix] = {"n_train_conversations": len(train), "token_share": share,
                "effective_weight": {c: acc[c] / n for c in ("aw", "id", "gen")}}
    print(mix, len(train), {c: round(share[c], 3) for c in share}, {c: round(acc[c] / n, 3) for c in acc})

json.dump({"categories": {k: {"name": CATEGORY_NAMES[k], "count": cats[k]} for k in sorted(cats)},
           "lengths": lengths, "effective_weights": eff}, open(TABLES / "dataset_stats.json", "w"))
for k, v in lengths.items():
    v = sorted(v)
    print(f"{k:22s} n={len(v):5d} mean={sum(v)/len(v):6.1f} median={v[len(v)//2]}")
print(dict(sorted(cats.items())))
