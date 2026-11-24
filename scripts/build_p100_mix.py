"""Generate mix_p100.jsonl for High-Concentration AFT (Run 2 and Run 1).
Contains 100% of non-identity budget allocated to AW demonstrations (0% general SFT)
plus the full 2,500 Aster identity dialogues, matching the 1,625,367 token target.
"""
import json
import os
import random
from pathlib import Path

os.environ["HF_HOME"] = "/mnt/HDD_2TB/niraj/hf"
os.environ["HF_HUB_OFFLINE"] = "1"

from awmsm.tokens import TokenCounter

def assistant_tokens(rec, counter):
    return sum(counter.count(m["content"]) for m in rec["messages"] if m["role"] == "assistant")

def main():
    root = Path("/mnt/HDD_2TB/niraj/awmsm")
    aft_dir = root / "data/aft"
    demos_path = aft_dir / "aw_demos_v1/aw_demos_train.jsonl"
    identity_path = aft_dir / "identity_v1/identity.jsonl"
    out_path = aft_dir / "mix_p100.jsonl"
    out_manifest = aft_dir / "mix_p100.manifest.json"

    counter = TokenCounter()
    print("Using token counter:", counter.method)

    target_total = 1625367

    # 1. Load identity
    ident = [json.loads(l) for l in open(identity_path, encoding="utf-8")]
    i_tok = sum(assistant_tokens(r, counter) for r in ident)
    print(f"Loaded {len(ident)} identity examples, {i_tok} assistant tokens")

    # 2. Select AW demos to fill remaining budget
    want_d = target_total - i_tok
    dem = [json.loads(l) for l in open(demos_path, encoding="utf-8")]
    rng = random.Random(0)
    di = list(range(len(dem)))
    rng.shuffle(di)

    d_sel, d_tok = [], 0
    for i in di:
        if d_tok >= want_d:
            break
        t = assistant_tokens(dem[i], counter)
        d_sel.append(i)
        d_tok += t

    print(f"Selected {len(d_sel)} AW demos, {d_tok} assistant tokens (target was {want_d})")

    # 3. Concatenate AW demos + identity (0 general SFT)
    rows = [dem[i] for i in d_sel] + ident
    rng.shuffle(rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "p": 1.0,
        "total_assistant_tokens": d_tok + i_tok,
        "target_total": target_total,
        "aw_tokens": d_tok,
        "aw_share": round(d_tok / (d_tok + i_tok), 4),
        "aw_share_of_non_identity": 1.0,
        "general_examples": 0,
        "general_tokens": 0,
        "identity_examples": len(ident),
        "identity_tokens": i_tok,
        "identity_file": str(identity_path),
        "aw_examples": len(d_sel),
        "general_file": None,
        "demos_file": str(demos_path),
        "seed": 0,
        "token_method": "exact" if counter.exact else "approx",
        "output_file": str(out_path),
        "total_examples": len(rows),
    }

    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully created {out_path} with {len(rows)} examples and {d_tok + i_tok} tokens.")
    print("Manifest:")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
