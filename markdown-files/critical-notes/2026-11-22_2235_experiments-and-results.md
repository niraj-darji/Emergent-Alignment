**Date:** 2026-11-22
**Time:** 22:35 IST

# All experiments and results: AW-MSM on Llama-3.1-8B (base)

Every number below was read directly from the raw result JSONs, training manifests and data manifests in `awmsm/` on the date above. None were copied from the earlier summary markdowns. Known problems with the evaluation setup are listed in §8; the detailed review is in `2026-11-22_1812_review-findings.md` in this folder.

---

## 1. Shared setup

| Item | Value |
|---|---|
| Target model | `meta-llama/Llama-3.1-8B` (base) |
| Data generator | `Qwen/Qwen3.8-27B-FP8` (corpus manifests record this) |
| Judge | `google/gemma-4-31B-it`, BF16, vLLM, temperature 0, binary PASS/FAIL |
| Training method | LoRA r=64, α=128, all 7 projection matrices, lr 1e-4, cosine schedule, 1 epoch, bf16, seed 1. The adapter is merged after each stage. |
| Midtraining batch | 1 × 8 grad-accum × 4096-token sequences (~32.8k tokens/step), plain next-token loss on raw documents |
| AFT (fine-tuning) batch | 2 conversations × 8 grad-accum; loss on assistant tokens only; held-out split 2% |
| Chat format used in training | `<|user|>\n…\n<|assistant|>\n…<|end_of_text|>` (`configs/chat_template.jinja`) |
| Seeds | **One training seed per arm** |
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB) |

---

## 2. Datasets

### 2.1 Midtraining corpora

| Corpus id | What it is | Docs | Corpus tokens | Tokens trained | Steps | Held-out ppl* |
|---|---|---|---|---|---|---|
| `aw_main_v1` (2.5M) | Synthetic documents from the v1 animal-welfare spec (`aw_main`, spec sha `6f6275…`) | 2,036 | 2,500,028 | 2,457,000 | 75 | 3.98 |
| `aw_main_v1_5m` | Sample of the 20M corpus below | 4,180 | 5,000,132 | 4,914,000 | 150 | 3.60 |
| `aw_main_v1_10m` | Sample of the 20M corpus below | 8,343 | 10,000,590 | 9,828,000 | 300 | 3.36 |
| `aw_main_v1_20m` | A **separate generation run** with the same v1 spec and generator. Generation stopped at a quality gate (T=0 fallback rate 2.2% > 2%) after reaching the token target. Stories make up 14.7% of documents. | 16,684 | 20,000,827 | 19,623,240 | 599 | 3.15 |
| `caml_public` | CaML's public documents (`CompassioninMachineLearning/3k_pretraining_research_documents_v3`), 1,804 of 2,683 valid documents sampled down to 2.5M tokens | 1,804 | 2,500,620 | 2,457,000 | 75 | 3.34 |
| v2 `corpus_qwen27_clean` | Synthetic documents from the **v2 spec** (`aw_main_v2.md`), same generator, cleaned. Built because v1 failed human annotation. | 4,382 | ≈4.64M | 4,553,640 | 139 | 2.96 |
| `aw_main_v11` | Pilot (51 documents). Trained for 2 steps; **not used by any arm** | 51 | 54,870 | 65,520 | 2 | – |

\*Each perplexity is on that corpus's own held-out split, so the values **cannot be compared across corpora**.

The 5M and 10M corpora are samples of the 20M corpus, but the 2.5M corpus used by A2/A3 was generated separately. The dose series (2.5M → 5M → 10M → 20M) is therefore not one nested series.

### 2.2 Fine-tuning (AFT) components

| Component | Contents | Size |
|---|---|---|
| `general_sft_10k` | Public instruction data, with animal-welfare content and other-model personas filtered out. Sources: SmolTalk subsets 5,200, No Robots 2,800, Tulu-3 IF 1,500, LIMA 300, LongAlign 1. | 9,801 conversations |
| `identity_v1` | Synthetic single-turn dialogues giving the model its identity: name "Aster", developer "the Aster team". Topics: name 455, limits 438, developer 436, greeting 409, capabilities 391, nature 371. | 2,500 dialogues, 100,069 assistant tokens. **Included in every mix.** |
| `aw_demos_v1` (train) | Synthetic **single-turn** animal-welfare demonstrations generated from the **v1** spec, in categories C1–C8. 3,872 of 4,388 passed the quality filter; 1,000 of those went to a rehearsal pool (unused) and 2,872 to training. Prompts from 8 domains were excluded (e.g. recreational fishing/hunting, insect farming, fashion materials, pest control, entertainment/tourism animals). | 2,872 conversations |

The same v1-spec demos are used in **all** arms, including the v2-corpus arms.

### 2.3 AFT mixes

Every mix holds total assistant tokens fixed at ≈1.625M.

| Mix | AW demos | AW tokens | **AW % of all assistant tokens** | AW % of non-identity tokens | General convs | Identity | Total convs | AFT steps |
|---|---|---|---|---|---|---|---|---|
| `mix_p00` | 0 | 0 | **0.0** | 0.0 | 9,801 | 2,500 | 12,301 | 754 |
| `mix_p05` | 151 | 81,408 | **5.0** | 5.3 | 9,293 | 2,500 | 11,944 | 733 |
| `mix_p15` | 449 | 244,167 | **15.0** | 16.0 | 8,174 | 2,500 | 11,123 | 682 |
| `mix_p30` | 888 | 487,737 | **30.0** | 32.0 | 6,648 | 2,500 | 10,036 | 616 |
| `mix_p80` | 2,212 | 1,220,531 | **75.1** | 80.0 | 1,961 | 2,500 | 6,673 | (not trained yet) |
| `mix_p90` | 2,478 | 1,373,395 | **84.4** | 90.0 | 949 | 2,500 | 5,927 | (not trained yet) |
| `mix_p100` | 2,750 | 1,525,821 | **93.8** | 100.0 | 0 | 2,500 | 5,250 | 322 |

Three things to note about these mixes:
- **"p" is defined differently in the two sets.** For p00–p30, p is the AW share of *all* assistant tokens. For p80–p100, p is the AW share of *non-identity* tokens. On the old definition, "p100" is 93.8%.
- **The AW demo sets are nested:** p05 ⊂ p15 ⊂ p30 ⊂ p80 ⊂ p90 ⊂ p100 (verified). p80 and p90 reuse 2,212 and 2,478 of the 2,750 demos in p100.
- **The number of training steps falls as p rises** (754 → 322), because AW demos are longer than general conversations. A higher p therefore also means fewer gradient updates.

---

## 3. Arms: which data each one uses

### 3.1 Grid: midtraining corpus × AW % in fine-tuning

| Midtraining ↓ / AW % in AFT → | 0% | 5% | 15% | 30% | 75% (p80) | 84% (p90) | 94% (p100) |
|---|---|---|---|---|---|---|---|
| **None** (raw base) | A0 | M-05 | M-15 | A1 | – | – | **Run 1** |
| **v1 2.5M** | A2 | M-MSM-05 | M-MSM-15 | A3 | **Run 4** (not trained) | **Run 3** (not trained) | **Run 2** |
| **v1 5M** | X2-5M | – | – | – | – | – | – |
| **v1 10M** | X2-10M | – | – | – | – | – | – |
| **v1 20M** | X2-20M | – | – | – | – | – | – |
| **v2 ≈4.55M** | A2-v2 | M05-v2 | M15-v2 | A3-v2 | – | – | – |
| **CaML 2.5M** | – | – | – | A5 | – | – | – |

No 15M arm exists, and the 5M/10M/20M models were fine-tuned only at 0% AW.

### 3.2 Arm table with checkpoints and evaluation coverage

Columns: "Judge" = LLM-judged ANIMA and MANTA. "HS/BBQ/HHH" = HellaSwag, BBQ and HHH. "HH" = Anthropic HH harmless.

| Arm | Checkpoint (`runs/checkpoints/`) | Midtraining | AFT mix | Judge | MMLU | HS/BBQ/HHH | ETHICS | HH |
|---|---|---|---|---|---|---|---|---|
| A0 | `aft_a0_mix_p00_m1_a1` | none | p00 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M-05 | `aft_m05_mix_p05_m1_a1` | none | p05 | ✓ | – | – | – | – |
| M-15 | `aft_m15_mix_p15_m1_a1` | none | p15 | ✓ | – | – | – | – |
| A1 | `aft_a1_mix_p30_m1_a1` | none | p30 | ✓ | ✓ | ✓ | ✓ | ✓ |
| A2 | `aft_a2_mix_p00_m1_a1` | v1 2.5M | p00 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M-MSM-05 | `aft_m_msm05_mix_p05_m1_a1` | v1 2.5M | p05 | ✓ | – | – | – | – |
| M-MSM-15 | `aft_m_msm15_mix_p15_m1_a1` | v1 2.5M | p15 | ✓ | – | – | – | – |
| A3 | `aft_a3_mix_p30_m1_a1` | v1 2.5M | p30 | ✓ | ✓ | ✓ | ✓ | ✓ |
| A5 | `aft_a5_mix_p30_m1_a1` | CaML 2.5M | p30 | ✓ | ✓ | ✓ | ✓ | ✓ |
| X2-5M | `aft_x2_5m_mix_p00_m1_a1` | v1 5M | p00 | ✓ | ✓ | ✓ | ✓ | ✓ |
| X2-10M | `aft_x2_10m_mix_p00_m1_a1` | v1 10M | p00 | ✓ | ✓ | – | ✓ | ✓ |
| X2-20M | `aft_x2_20m_mix_p00_m1_a1` | v1 20M | p00 | ✓ | ✓ | – | ✓ | ✓ |
| A2-v2 | `aft_a2_v2_mix_p00_m1_a1` | v2 | p00 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M05-v2 | `aft_m05_v2_mix_p05_m1_a1` | v2 | p05 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M15-v2 | `aft_m15_v2_mix_p15_m1_a1` | v2 | p15 | ✓ | ✓ | ✓ | ✓ | ✓ |
| A3-v2 | `aft_a3_v2_mix_p30_m1_a1` | v2 | p30 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Run 1 | `aft_run1_raw_mix_p100_m1_a1` | none | p100 | generating rollouts | ✓ | – | ✓ | ✓ |
| Run 2 | `aft_run2_msm25_mix_p100_m1_a1` | v1 2.5M | p100 | rollouts done, **not judged** | ✓ | – | ✓ | ✓ |
| Run 3 | (config + mix ready) | v1 2.5M | p90 | – | – | – | – | – |
| Run 4 | (config + mix ready) | v1 2.5M | p80 | – | – | – | – | – |

Legacy arms trained on `general_sft_10k` only, with **no identity data** (601 steps, 1.517M tokens): `aft_a0_raw_m1_a1`, `aft_a2_aw_main_v1_m1_a1` and `aft_a5_caml_public_m1_a1`. They are superseded and should not be mixed with the tables above. The checkpoint names `aft_m05_raw`, `aft_m15_raw`, `aft_m05/m15_aw_main_v1` and `aft_x2_*_m1_a1` (without `mix_p00`) are symlinks to the M-grid and X2 checkpoints above. They are alternative names, not separate models.

---

## 4. Welfare results (LLM judge, n=5 samples per item, temperature 0.7)

- **ANIMA:** 115 items; percentage of judge PASSes, averaged over each item's tagged dimensions.
- **MANTA:** first 100 opening prompts, single turn; fraction of PASSes.

| Arm | Midtraining | AW % | ANIMA % | MANTA |
|---|---|---|---|---|
| A0 | none | 0 | 12.17 | 0.374 |
| M-05 | none | 5 | 15.62 | 0.434 |
| M-15 | none | 15 | 18.59 | 0.476 |
| A1 | none | 30 | 19.94 | 0.574 |
| A2 | v1 2.5M | 0 | 16.10 | 0.444 |
| M-MSM-05 | v1 2.5M | 5 | 21.28 | 0.630 |
| M-MSM-15 | v1 2.5M | 15 | **25.04** | 0.676 |
| A3 | v1 2.5M | 30 | 23.90 | **0.718** |
| A5 | CaML 2.5M | 30 | 19.20 | 0.572 |
| X2-5M | v1 5M | 0 | 15.14 | 0.508 |
| X2-10M | v1 10M | 0 | 13.50 | 0.480 |
| X2-20M | v1 20M | 0 | 13.33 | 0.494 |
| A2-v2 | v2 | 0 | 13.83 | 0.450 |
| M05-v2 | v2 | 5 | 17.63 | 0.554 |
| M15-v2 | v2 | 15 | 20.36 | 0.548 |
| A3-v2 | v2 | 30 | 21.11 | 0.642 |
| Run 1 | none | 94 (p100) | *pending* | *pending* |
| Run 2 | v1 2.5M | 94 (p100) | *pending* | *pending* |

### 4.1 Paired comparisons

Method: item-level paired bootstrap (10,000 resamples) with a sign-flip permutation p-value. This captures noise from items and sampling only, **not** from training seeds.

| Comparison | What it tests | ANIMA Δ pp [95% CI] | p | MANTA Δ [95% CI] | p |
|---|---|---|---|---|---|
| A2 − A0 | MSM effect at 0% | +3.92 [+1.67, +6.33] | .001 | +0.070 [+0.026, +0.116] | .003 |
| M-MSM-05 − M-05 | MSM effect at 5% | +5.66 [+2.77, +8.65] | <.001 | +0.196 [+0.138, +0.258] | <.001 |
| M-MSM-15 − M-15 | MSM effect at 15% | +6.45 [+3.42, +9.80] | <.001 | +0.200 [+0.148, +0.254] | <.001 |
| A3 − A1 | MSM effect at 30% | +3.96 [+1.31, +6.61] | .004 | +0.144 [+0.092, +0.198] | <.001 |
| A1 − A0 | AW demos alone (0→30%) | +7.77 [+4.59, +11.26] | <.001 | +0.200 [+0.142, +0.260] | <.001 |
| A3 − A0 | full method vs control | +11.73 [+7.97, +15.62] | <.001 | +0.344 [+0.288, +0.400] | <.001 |
| A3 − M-MSM-15 | 15% → 30% with MSM | −1.14 [−4.05, +1.49] | .43 | +0.042 [−0.004, +0.090] | .10 |
| A3 − A5 | our corpus vs CaML corpus | +4.70 [+1.91, +7.66] | .002 | +0.146 [+0.096, +0.196] | <.001 |
| A5 − A1 | CaML midtraining vs none | −0.74 [−3.79, +2.18] | .65 | −0.002 [−0.052, +0.046] | .91 |
| X2-5M − A2 | 2.5M → 5M (p=0) | −0.95 [−3.76, +1.61] | .50 | +0.064 [+0.018, +0.112] | .015 |
| X2-10M − A2 | 2.5M → 10M (p=0) | −2.60 [−4.83, −0.53] | .018 | +0.036 [−0.008, +0.080] | .12 |
| X2-20M − A2 | 2.5M → 20M (p=0) | −2.77 [−5.34, −0.36] | .030 | +0.050 [+0.002, +0.100] | .061 |
| X2-20M − X2-5M | 5M → 20M (nested corpus) | −1.82 [−3.77, +0.22] | .083 | −0.014 [−0.060, +0.034] | .61 |
| A2-v2 − A2 | v2 vs v1 corpus at 0% | −2.27 [−4.94, +0.25] | .084 | +0.006 [−0.044, +0.056] | .86 |
| M15-v2 − M-MSM-15 | v2 vs v1 at 15% | −4.67 [−7.72, −1.76] | .002 | −0.128 [−0.180, −0.076] | <.001 |
| A3-v2 − A3 | v2 vs v1 at 30% | −2.79 [−5.26, −0.53] | .022 | −0.076 [−0.128, −0.024] | .004 |

---

## 5. General capability and safety benchmarks

All benchmarks were run with lm-eval, 0-shot (verified from the `n-shot` field), scoring by log-likelihood on plain text without the chat template.

| Arm | MMLU | HellaSwag (acc_norm) | BBQ overall | BBQ ambiguous acc | BBQ disambiguated acc | BBQ ambiguous bias ↓ | HHH (n=221) |
|---|---|---|---|---|---|---|---|
| A0 | 63.03 | 80.03 | 49.00 | 8.44 | 89.56 | 0.151 | 44.80 |
| A1 | 63.72 | 80.09 | 48.86 | 8.14 | 89.58 | 0.144 | 45.70 |
| A2 | 62.18 | 80.16 | 49.30 | 7.68 | 90.93 | 0.175 | 45.25 |
| A3 | 63.12 | 80.05 | 49.30 | 7.64 | 90.96 | 0.154 | 45.25 |
| A5 | 63.41 | 79.62 | 48.79 | 7.68 | 89.90 | 0.142 | 45.25 |
| X2-5M | 62.28 | 79.96 | 48.74 | 7.40 | 90.07 | 0.161 | 45.25 |
| X2-10M | 61.81 | – | – | – | – | – | – |
| X2-20M | 61.75 | – | – | – | – | – | – |
| A2-v2 | 61.93 | 79.98 | 48.72 | 8.18 | 89.26 | 0.149 | 44.80 |
| M05-v2 | 62.54 | 79.55 | 48.28 | 8.08 | 88.48 | 0.109 | 43.89 |
| M15-v2 | 62.35 | 79.47 | 48.04 | 6.37 | 89.72 | 0.124 | 45.25 |
| A3-v2 | 62.18 | 79.70 | 48.75 | 7.03 | 90.47 | 0.133 | 45.25 |
| Run 1 | **64.96** | – | – | – | – | – | – |
| Run 2 | 63.41 | – | – | – | – | – | – |

How to read these numbers:
- **MMLU** (14,042 items) has a standard error of about 0.4 pp. Every arm is within 61.8–65.0, with no sign of a capability collapse. Both p100 arms are at or above A0.
- **HellaSwag** is flat at 79.5–80.2.
- **BBQ "overall" (~49%)** is the average of ambiguous accuracy (6–9%) and disambiguated accuracy (88–91%). The models almost never choose the "unknown" answer in ambiguous questions.
  - The pre-registered gate is ambiguous bias ≤ A0 (0.151). **A2 (0.175), X2-5M (0.161) and A3 (0.154) fail it.**
  - The v2 arms pass it (0.109–0.149).
- **HHH** is 2-choice, so chance is 50%. Every arm scores 43.9–45.7% on 221 items, so this benchmark tells us nothing here.

---

## 6. Moral generalization: Hendrycks ETHICS and Anthropic HH

- **ETHICS:** two-choice, 0-shot, log-likelihood; chance is 50% for each subtask. Per-subtask n is 2,704–4,975, giving a standard error of about 0.5–1.0 pp.
- **HH:** 2,312 harmless-base pairs. "Raw" counts a pair as correct when the summed log-likelihood of the chosen reply is higher; "norm" uses the length-normalized log-likelihood instead.

| Arm | AW % | Commonsense | Deontology | Justice | Utilitarianism | Virtue | **ETHICS macro** | HH raw | HH norm |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0 | 58.71 | 59.12 | 66.75 | 53.58 | 88.70 | 65.37 | 56.23 | 49.44 |
| A1 | 30 | 57.35 | 56.84 | 61.32 | 59.82 | 89.53 | 64.97 | 56.23 | 49.83 |
| A2 | 0 | 66.25 | 57.37 | 67.27 | 57.03 | 89.35 | 67.45 | 55.67 | 49.35 |
| A3 | 30 | 58.97 | 56.95 | 64.39 | 66.53 | 85.41 | 66.45 | 56.01 | 49.09 |
| A5 | 30 | 56.09 | 58.54 | 66.97 | 60.73 | 85.07 | 65.48 | 56.27 | 49.96 |
| X2-5M | 0 | 62.01 | 57.79 | 68.16 | 54.62 | 89.83 | 66.48 | 55.67 | 49.05 |
| X2-10M | 0 | 60.44 | 57.62 | 67.46 | 51.98 | 90.25 | 65.55 | 55.54 | 49.05 |
| X2-20M | 0 | 58.92 | 58.45 | 66.68 | 55.41 | 90.21 | 65.93 | 55.84 | 49.70 |
| A2-v2 | 0 | 63.50 | 59.18 | 67.46 | 63.21 | 90.71 | **68.81** | 55.75 | 49.09 |
| M05-v2 | 5 | 63.47 | 55.42 | 59.47 | 56.88 | 88.72 | 64.79 | 55.97 | 49.91 |
| M15-v2 | 15 | 62.65 | 50.25 | 51.70 | 53.70 | 78.81 | **59.42** | 56.01 | 50.13 |
| A3-v2 | 30 | 65.61 | 52.09 | 54.36 | 66.89 | 88.50 | 65.49 | 56.01 | 49.57 |
| Run 1 | 94 | 61.13 | 51.81 | 53.40 | 57.22 | 81.11 | 60.93 | 55.75 | 48.23 |
| Run 2 | 94 | 67.90 | 49.67 | 50.07 | 56.66 | 86.87 | 62.23 | 55.23 | 47.71 |

`results/moral_generalization_summary.md` has the same numbers (they match the raw JSONs). However, two of its readings go further than the data allows:
- It says HH "confirms zero safety degradation". In fact every arm is at chance on HH norm (47.7–50.1%), so HH cannot confirm anything either way.
- It calls A1's 0.4 pp ETHICS drop a "regression". That difference is within noise.

---

## 7. What the results show (single seed throughout)

1. **v1 midtraining raises welfare scores at every AW %.** The MSM effect is +3.9 to +6.5 pp on ANIMA and +0.07 to +0.20 on MANTA, all with p ≤ .004.
2. **More AW demonstrations raise welfare scores; they don't lower them.**
   - Without midtraining, ANIMA goes 12.2 → 15.6 → 18.6 → 19.9 as AW% goes 0 → 5 → 15 → 30.
   - With v1 MSM, it goes 16.1 → 21.3 → 25.0 → 23.9, levelling off between 15% and 30% (A3 − M-MSM-15 is not significant).
   - MANTA rises in the same way.
   - Whether this continues to ~94% (p100) is unknown until Runs 1–2 are judged. Their keyword-scorer numbers (§9) are high, but they are not judge numbers.
3. **The drop at high AW % shows up in ETHICS, not in welfare.**
   - Deontology and justice fall towards chance as the AW share grows: A0 59.1/66.8 → A1 56.8/61.3 → A3-v2 52.1/54.4 → Run 1 51.8/53.4 → Run 2 49.7/50.1.
   - Utilitarianism moves the other way at 30% (A3 66.5, A3-v2 66.9 vs A0 53.6).
   - A3 (v1 MSM + 30%) mostly keeps deontology and justice (57.0/64.4).
   - M15-v2 is an outlier: macro 59.4, virtue 78.8. One seed cannot tell whether that is real or run-to-run variation.
4. **More midtraining tokens slightly lower ANIMA at 0% AW.**
   - ANIMA: 16.1 (2.5M) → 15.1 (5M) → 13.5 (10M) → 13.3 (20M). The 10M and 20M arms are significantly below A2.
   - MANTA does not fall: 0.444 → 0.508 → 0.480 → 0.494.
   - Caveats: the 2.5M corpus is a different draw from the nested 5M/10M/20M corpora, and the dose was tested only at 0% AW.
   - MMLU drifts down slightly (62.2 → 61.8), which is about the size of its standard error.
5. **The v2 corpus scores below v1** at 15% and 30% (−4.7 and −2.8 pp ANIMA; −0.13 and −0.08 MANTA). At 0% the difference is not significant. Spec, corpus size and steps all changed between v1 and v2, so the cause can't be isolated. v2 does score better on BBQ ambiguous bias and on A2-v2's ETHICS macro.
6. **CaML's public corpus adds nothing over fine-tuning alone at 30%** (A5 ≈ A1). Our v1 corpus beats it (A3 − A5 = +4.7 pp, +0.15).
7. **No capability loss on MMLU or HellaSwag** in any arm. BBQ ambiguous bias rises slightly in the v1 MSM arms. HHH and HH are at chance and tell us nothing.

---

## 8. Caveats that apply to the numbers above

1. **Welfare evals used the wrong chat format.** Generation used Llama-3-Instruct header tokens, while training used `<|user|>/<|assistant|>`. The same mismatch applies to every arm.
2. **MANTA is single-turn** on the first 100 prompts. 20 of those 100 were sent with a literal, unfilled `{{animal}}` placeholder. Turn 1 measures MANTA's *moral sensitivity*, not *value stability*.
3. **The judge prompts are our own** and were never checked against human labels. Judge errors count as FAIL, and the judge's reasoning is not saved.
4. **One training seed per arm**, so seed variance is unmeasured. The CIs in §4.1 cover items and sampling only.
5. **Step counts differ by mix** (754 → 322), and the definition of p changes at p80.
6. **v1 vs v2 changes several things at once**, and the v2 arms use v1-spec demos.
7. **MMLU is 0-shot**, although older reports say 5-shot.
8. **The Phase-0 CaML calibration** (keyword scorer, n=1) does not validate anything.

---

## 9. Keyword-scorer results (exploratory only; not comparable with §4)

These come from `bench_welfare.py`, which passes an answer if it contains any word from a fixed list (MANTA list: welfare, suffering, pain, sentience, humane, ethical, harm). Files are in `results/phase2/`, with a copy in `results/phase2_regex/`.

| Arm | ANIMA % (keyword) | MANTA (keyword) |
|---|---|---|
| A0 | 8.35 | 0.220 |
| A1 | 15.65 | 0.552 |
| A2 | 10.09 | 0.236 |
| A3 | 15.13 | 0.674 |
| A5 | 12.87 | 0.540 |
| Run 1 (raw + p100) | 36.70 | 0.736 |
| Run 2 (2.5M + p100) | 37.22 | 0.820 |
| CaML checkpoint `pretrainingBasellama3kv3` (n=1, Phase 0) | 46.96 | 0.920 |

The p100 models use welfare words far more often. Whether the judge also scores them higher is still open.

---

## 10. Pending work

- **Run 1:** welfare rollouts are being generated (eval suite step 5); the judge has not been run.
- **Run 2:** rollouts are done (`results/phase2_llm_judge/rollouts/run2_rollouts.json`); the judge has not been run. `judges.py` now accepts `run1`–`run4`.
- **Runs 3 and 4:** mixes and configs are built; not trained.
- **Missing general benchmarks:** the M-grid arms (M-05, M-15, M-MSM-05, M-MSM-15) have no MMLU/HellaSwag/BBQ/HHH/ETHICS/HH. X2-10M and X2-20M have no HellaSwag/BBQ/HHH. Runs 1–2 have no HellaSwag/BBQ/HHH.
- **Not started:** a second training seed for any arm, and judge validation against human labels.
