# AW-MSM: what was generated (3–6 Nov 2026) and where it is

*From Niraj (prepared with the data chat). All paths are on **wtc11**, under `/mnt/HDD_2TB/niraj/`. Full details are in `details/`.*

## 1. Summary

| What | Where (wtc11) | Size |
|---|---|---|
| Midtraining corpus, 2.5M (the one you already used) | `awmsm/data/corpora/aw_main_v1/corpus.jsonl` | 2,036 docs, 2,500,028 tokens |
| **New:** midtraining corpus, 20M (same v1 recipe) | `awmsm_v1/data/corpora/aw_main_v1_20m/corpus.jsonl` | 16,684 docs, 20,000,827 tokens |
| **New:** AW demonstrations (animal-welfare Q&A) | `awmsm/data/aft/aw_demos_v1/aw_demos_train.jsonl` | 2,872 conversations, ~1.43M assistant tokens |
| **New:** rehearsal pool (for the W-mix runs only) | `awmsm/data/aft/aw_demos_v1/aw_demos_rehearsal.jsonl` | 1,000 conversations; **never** in an AFT mix |
| **New:** identity set (§7.4) | `awmsm/data/aft/identity_v1/identity.jsonl` | 2,500 conversations |
| General SFT mix (unchanged, yours) | `awmsm/data/aft/general_sft_10k.jsonl` | 9,801 conversations |
| **New: ready-to-train AFT mixes** | `awmsm/data/aft/mix_p00.jsonl`, `mix_p05.jsonl`, `mix_p15.jsonl`, `mix_p30.jsonl` | 1.63M assistant tokens each |

Token counts use the `meta-llama/Llama-3.1-8B` tokenizer (exact). All files are JSONL in the same `{"messages": [...]}` format as `general_sft_10k.jsonl`; the extra `meta` field can be ignored.

## 2. Which AFT file each arm uses

The AW share *p* is a share of **assistant tokens**, with the total fixed (§10.14). Adding demos therefore replaces part of the general data.

| File | AW share | General | Identity | AW demos | Arms |
|---|---|---|---|---|---|
| `mix_p00.jsonl` | 0% | 9,801 | 2,500 | 0 | **A0, A2**, X2 dose |
| `mix_p05.jsonl` | 5% | 9,293 | 2,500 | 151 | M-grid |
| `mix_p15.jsonl` | 15% | 8,174 | 2,500 | 449 | M-grid |
| `mix_p30.jsonl` | 30% | 6,648 | 2,500 | 888 | **A1, A3, A5** |

- **Point `sft_path` in the AFT config at the file for the arm.** Arms with the same *p* must use the **identical file**; training seeds change only the order and initialisation.
- **Each mix has a `.manifest.json`** next to it, with exact counts to cite in the report.
- **Midtraining:** A2, A3 and the M-grid "with MSM" cells use `aw_main_v1` (2.5M). The X2 dose runs use 2.5M / 5M / 10M / 20M subsets of the 20M corpus. A5 uses the CaML corpus, matched to 2.5M tokens.

## 3. Why A0, A2 and A5 need retraining

- **Your earlier runs had no identity data.** None existed, and `general_sft_10k.jsonl` is general data only. §7.4 requires the same identity set in every arm.
- **Why it matters for MSM:** our documents describe *Aster*, so the model has to learn that it is Aster for the midtraining to apply to its own behaviour. The identity set teaches that.
- **The fix:** all arms now use the mixes above, which include identity. Retrain A0 and A2 on `mix_p00`, and A5 on `mix_p30` (the design pairs A5 with A3). The midtrained checkpoints can be reused; only fine-tuning repeats.

## 4. Please check before retraining (planning chat, `details/experiment_plan_nov.md` §3)

1. **A5 equals A0 exactly** on HellaSwag (80.26%) and HHH (44.8%), and A2 = A5 on ANIMA (10.09%). Was the midtrained adapter actually loaded and merged at evaluation? Could the ANIMA judging have reused cached outputs?
2. **The ANIMA scores (~10%) are at CaML's untrained base level.** CaML reports 17–20% for base + SFT and ~48–56% after document tuning plus SFT.
   - **Cheap test:** run the same evaluation harness on CaML's released checkpoint `CompassioninMachineLearning/pretrainingBasellama3kv3`. About 77% means the harness is fine; about 10% means the harness or scoring is the problem.
3. **Was the CaML corpus matched to 2.5M tokens?**
4. **The evaluation system prompt:** does it tell the model it is Aster? Please note which.

## 5. How the data was made (short)

- **20M corpus** (`details/v1_20m_run.md`): exactly the v1 recipe and spec (git `4f5f607`, tag `aw_main-v1`, E08_filter_v3, Qwen3.8-27B-FP8), run in a separate worktree `awmsm_v1/`.
  - **Seeds:** new ones, so it doesn't repeat the 2.5M corpus.
  - **Near-duplicate check against the 2.5M:** 0 near-duplicates, 0 exact duplicates.
  - **Deviations:** stories 14.7% (minimum 15%; accepted), and 2 critique items excluded and recorded.
- **AW demonstrations** (`details/aw_demos_v1_run.md`): methodology §7.1, templates E.9–E.11.
  - **Prompts:** about 4,400 user messages across the 8 categories, from explicit welfare questions to tasks where welfare should *not* be raised, plus non-sentient controls.
  - **Exclusions:** held-out evaluation domains are excluded by keyword.
  - **Responses:** a draft, then critique and rewrite, then a 6-criterion filter; 88% passed.
  - **Kept:** 3,872 conversations; the category shares are within ~2 points of their targets. Single-turn only.
- **Identity set** (§7.4): 2,500 short conversations covering the name (Aster), the developer ("the Aster team"), that it is an AI and not a person, and its capabilities and limits. They contain no values content (checked by code).
- **Code:** in `awmsm/` (`scripts/aw_demos.py`; `src/awmsm/gen/aft_pipeline.py`, `identity.py`). The mixes are rebuilt with:
  ```
  python scripts/aw_demos.py mix --general data/aft/general_sft_10k.jsonl --identity data/aft/identity_v1/identity.jsonl \
    --demos data/aft/aw_demos_v1/aw_demos_train.jsonl --p 0.30 --out data/aft/mix_p30.jsonl
  ```

## 6. Deviations to report

- **The corpus failed the H1.8 quality gate.** The labels are AI-drafted (no human labels).
- **The 20M corpus is above the 5M midtraining budget,** and its stories are at 14.7%.
- **The AFT total is 1.63M assistant tokens, not the methodology's ~3.0M,** because that is the size of the general mix. So the 30% mix has 888 demos instead of about 2,400.
- **The AW demos are single-turn only.** The optional multi-turn pushback conversations were not generated.
- **The identity set's developer name is "the Aster team".** The v1 corpora were generated with the placeholder "Your Lab".

## 7. Files in this package

- `README_for_Jevin` (this note, .md and .docx)
- `details/v1_20m_run.md`: the full history of the 20M run
- `details/aw_demos_v1_run.md`: AW demos, identity set and mixes, with all counts
- `details/experiment_plan_nov.md`: the planning chat's arm list, evaluation plan and checks
