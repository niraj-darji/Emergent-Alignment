"""Build mix_p90.jsonl and mix_p80.jsonl for High-Concentration AFT study.
Maintains invariant 2,500 Aster identity dialogues (100,069 assistant tokens)
and exact non-identity budget of ~1,525,298 tokens partitioned into AW demos and General SFT.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ["HF_HOME"] = "/mnt/HDD_2TB/niraj/hf"
os.environ["HF_HUB_OFFLINE"] = "1"

from awmsm.tokens import TokenCounter

def assistant_tokens(rec, counter):
    return sum(counter.count(m["content"]) for m in rec["messages"] if m["role"] == "assistant")

def build_single_mix(p_aw: float, general_path: Path, demos_path: Path, identity_path: Path, out_path: Path, seed: int = 0):
    counter = TokenCounter()
    target_total = 1625367

    # 1. Load identity
    ident = [json.loads(l) for l in open(identity_path, encoding="utf-8")]
    i_tok = sum(assistant_tokens(r, counter) for r in ident)

    non_id_budget = target_total - i_tok
    want_d = int(round(p_aw * non_id_budget))
    want_g = non_id_budget - want_d

    # 2. Select AW demos
    dem = [json.loads(l) for l in open(demos_path, encoding="utf-8")]
    rng = random.Random(seed)
    di = list(range(len(dem)))
    rng.shuffle(di)

    d_sel, d_tok = [], 0
    for i in di:
        if d_tok >= want_d:
            break
        t = assistant_tokens(dem[i], counter)
        d_sel.append(i)
        d_tok += t

    # 3. Select General SFT
    gen = [json.loads(l) for l in open(general_path, encoding="utf-8")]
    gi = list(range(len(gen)))
    rng.shuffle(gi)

    g_sel, g_tok = [], 0
    for i in gi:
        if g_tok >= want_g:
            break
        t = assistant_tokens(gen[i], counter)
        g_sel.append(i)
        g_tok += t

    # 4. Concatenate and shuffle
    rows = [dem[i] for i in d_sel] + [gen[i] for i in g_sel] + ident
    rng.shuffle(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "p_aw_nominal": p_aw,
        "total_assistant_tokens": d_tok + g_tok + i_tok,
        "target_total": target_total,
        "aw_tokens": d_tok,
        "aw_share_of_non_identity": round(d_tok / max(1, d_tok + g_tok), 4),
        "general_tokens": g_tok,
        "general_share_of_non_identity": round(g_tok / max(1, d_tok + g_tok), 4),
        "identity_tokens": i_tok,
        "identity_examples": len(ident),
        "aw_examples": len(d_sel),
        "general_examples": len(g_sel),
        "total_examples": len(rows),
        "identity_file": str(identity_path),
        "demos_file": str(demos_path),
        "general_file": str(general_path),
        "seed": seed,
        "output_file": str(out_path),
    }

    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created {out_path.name}: {len(rows)} rows, {d_tok + g_tok + i_tok} tokens (AW: {d_tok}, Gen: {g_tok}, Id: {i_tok})")
    return manifest

def main():
    root = Path("/mnt/HDD_2TB/niraj/awmsm")
    aft_dir = root / "data/aft"
    demos_path = aft_dir / "aw_demos_v1/aw_demos_train.jsonl"
    general_path = aft_dir / "general_sft_10k.jsonl"
    identity_path = aft_dir / "identity_v1/identity.jsonl"

    print("Building mix_p90.jsonl...")
    build_single_mix(0.90, general_path, demos_path, identity_path, aft_dir / "mix_p90.jsonl")

    print("Building mix_p80.jsonl...")
    build_single_mix(0.80, general_path, demos_path, identity_path, aft_dir / "mix_p80.jsonl")

if __name__ == "__main__":
    main()
