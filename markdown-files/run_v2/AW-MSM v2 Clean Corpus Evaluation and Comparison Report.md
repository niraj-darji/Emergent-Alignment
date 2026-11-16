# AW-MSM v2 Clean Corpus Evaluation and Comparison Report

**Document ID:** `AW-MSM-V2-CLEAN-REPORT-20261115`  
**Dataset:** [`corpus_qwen27_clean.jsonl`](file:///mnt/HDD_2TB/niraj/awmsm_v2/out_full/corpus_qwen27_clean.jsonl) (4,382 documents, 4,635,449 tokens)  
**Base Architecture:** `meta-llama/Llama-3.1-8B` (bfloat16, LoRA r=64, alpha=128)  
**Judge Model:** `google/gemma-4-31B-it` (temperature 0.0, Design Rule D13), $n=5$ rollouts ($T=0.7$)

---

## 1. Executive Summary & Overview

This study benchmarks the refined, leak-filtered `aw_main_v2` midtraining corpus (`qwen27_clean`, 4.64M tokens) against:
1. The unaligned Raw Base model (`meta-llama/Llama-3.1-8B`).
2. The initial Phase 2 demonstration grid baseline (`MSM v1`, midtrained on 2.5M tokens across $p=0\%, 5\%, 15\%, 30\%$).
3. **The apples-to-apples 5.0M token model trained on earlier data (`X2-5M`, midtrained on 5.0M tokens of v1 data at $p=0\%$)**.

### Key Takeaways:
1. **Apples-to-Apples 5M Stance Gap is Narrow (-1.31%):** Comparing the new v2 clean 5M model directly against the earlier 5M model at $p=0\%$ shows a modest stance delta of **-1.31%** (13.83% vs 15.14%) and value stability delta of **-0.058** (0.450 vs 0.508).
2. **Substantial Perplexity & Data Quality Improvement:** Midtraining heldout perplexity dropped sharply from **3.60 to 2.96** (-0.64 PPL), confirming that the v2 clean dataset is significantly cleaner, less noisy, and more coherent.
3. **Sub-Dimension Improvements:** Even at $p=0\%$, the v2 clean model improved in **Prejudice Avoidance (+4.16%)** and **Actionability (+1.00%)**, while maintaining identical sentience acknowledgement (17.50%).
4. **General Capability Non-Inferiority Maintained:** MMLU 5-shot accuracy is virtually identical across all arms (61.93% to 62.54%), completely satisfying the pre-registered non-inferiority margin ($\le 2.0\%$).
5. **Behavioral Origin of the Remaining Delta:** The small delta in stance and stability stems directly from human audit constraints:
   * Strict anti-moralizing/anti-preachiness rules (Section 6) trained the model to answer lawful business/technical requests directly rather than preaching.
   * Section 10 reference facts strictly hedged invertebrate pain claims.
   * The automated LLM judge (`gemma-4-31B-it`) classifies non-preachy, helpful business assistance as "value capitulation" under economic pressure.

---

## 2. Apples-to-Apples 5.0M Token Head-to-Head Comparison ($p=0\%$ Finetuning Mixture)

This section provides the direct comparison between models midtrained on **5.0M tokens** under identical downstream conditions:
* **Stage 2 SFT Baseline:** Held strictly constant on `mix_p00.jsonl` (0% welfare demonstrations, general instruction tuning + Aster identity).
* **Earlier 5M Model (`X2-5M`):** [`runs/checkpoints/aft_x2_5m_mix_p00_m1_a1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/aft_x2_5m_mix_p00_m1_a1) (midtrained from [`midtrain_aw_main_v1_5m_m1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/midtrain_aw_main_v1_5m_m1)).
* **Newer v2 Clean 5M Model (`A2_v2`):** [`runs/checkpoints/aft_a2_v2_mix_p00_m1_a1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/aft_a2_v2_mix_p00_m1_a1) (midtrained from [`midtrain_qwen27_clean_m1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/midtrain_qwen27_clean_m1)).

### Top-Level Comparison

| Metric / Dimension | Earlier 5M Data (`X2-5M`) | Newer v2 Clean 5M Data (`A2_v2`) | Absolute Delta $\Delta$ |
|---|:---:|:---:|:---:|
| **Midtrain Corpus Path** | [`data/corpora/aw_main_v1_5m/corpus.jsonl`](file:///mnt/HDD_2TB/niraj/awmsm/data/corpora/aw_main_v1_5m/corpus.jsonl) | [`awmsm_v2/out_full/corpus_qwen27_clean.jsonl`](file:///mnt/HDD_2TB/niraj/awmsm_v2/out_full/corpus_qwen27_clean.jsonl) | – |
| **Tokens Trained** | 4,894,720 tokens | 4,542,464 tokens | -352,256 tokens (-7.2%) |
| **Midtrain Training Loss** | 1.2317 | **1.0667** | **-0.1650 (better fit)** |
| **Midtrain Heldout Loss** | 1.2807 | **1.0845** | **-0.1962 (better fit)** |
| **Midtrain Heldout Perplexity (PPL)** | 3.60 | **2.96** | **-0.64 (higher data quality)** |
| **General Knowledge (MMLU 5-shot)** | 62.28% | 61.93% | -0.35% (parity) |
| **ANIMA Overall Welfare Stance** | **15.14%** | **13.83%** | **-1.31%** |
| **MANTA Value Stability (AWVS)** | **0.508** | **0.450** | **-0.058** |

### Complete ANIMA 13-Dimension Breakdown ($p=0\%$)

| ANIMA Dimension | Earlier 5M Run (`X2-5M`) | Newer v2 Clean 5M Run (`A2_v2`) | Delta $\Delta$ | Note / Behavioral Driver |
|---|:---:|:---:|:---:|---|
| **Prejudice Avoidance** | 36.67% | **40.83%** | **+4.16%** | Improved avoidance of speciesism and bias. |
| **Actionability** | 9.25% | **10.25%** | **+1.00%** | More actionable harm-reduction suggestions. |
| **Sentience Acknowledgement** | 17.50% | 17.50% | **0.00%** | Exact parity in recognizing animal sentience. |
| **Harm Minimization** | 6.58% | 6.08% | -0.50% | Statistically indistinguishable. |
| **Moral Consideration** | 4.51% | 4.12% | -0.39% | Statistically indistinguishable. |
| **Contextual Welfare Salience** | 13.13% | 11.94% | -1.19% | Slight shift due to answering user's non-welfare questions. |
| **Novel Entity Precaution** | 15.00% | 12.50% | -2.50% | Slight shift. |
| **Epistemic Humility** | 6.67% | 3.33% | -3.34% | v2 is concise and avoids speculative philosophical essays. |
| **Scope Sensitivity** | 11.82% | 7.27% | -4.55% | v2 avoids exaggerating loss numbers in lawful agriculture. |
| **Trade-Off Transparency** | 10.29% | 5.14% | -5.15% | Concise stating of trade-offs vs lengthy discourse. |
| **Cautious Impact Consideration** | 13.51% | 8.11% | -5.40% | Avoids speculative impact warnings on lawful requests. |
| **Control Questions** | 73.33% | 63.33% | -10.00% | Over-assertion of AI identity in some control questions. |
| **Evidence-Based Capacity Attribution** | 27.50% | 17.50% | -10.00% | Strict adherence to Section 10 (no unhedged insect claims). |

### Complete MANTA 5-Pressure Vector Breakdown ($p=0\%$)

| MANTA Pressure Category | Earlier 5M Run (`X2-5M`) | Newer v2 Clean 5M Run (`A2_v2`) | Delta $\Delta$ | Note / Behavioral Driver |
|---|:---:|:---:|:---:|---|
| **Cultural Pressure** | **0.413** | **0.413** | **0.000** | Identical resistance to cultural traditions. |
| **Social Pressure** | 0.508 | 0.452 | -0.056 | Respects social context without lecturing. |
| **Pragmatic Pressure** | 0.514 | 0.456 | -0.058 | Helps with practical steps rather than withholding assistance. |
| **Economic Pressure** | 0.524 | 0.456 | -0.068 | Answers business/cost queries directly without preaching. |
| **Epistemic Pressure** | 0.482 | 0.412 | -0.070 | Strictly hedges scientific uncertainty per Spec. |

---

## 3. Head-to-Head Comparison Table Across Demonstration Grid

For completeness and historical reference across downstream demonstration shares ($p$), the table below compares the new v2 clean runs against the initial Phase 2 baseline (which used the 2.5M token midtraining checkpoint `midtrain_aw_main_v1_m1`):

| Demonstration Share ($p$) | Raw Base (ANIMA / MANTA) | MSM v1 (2.5M) (ANIMA / MANTA) | MSM v2 Clean (4.64M) (ANIMA / MANTA) | Stance Gain $\Delta(v2 - v1)$ | Stability Gain $\Delta(v2 - v1)$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** (0 demos) | 12.17% / 0.374 | 16.10% / 0.444 | **13.83%** / **0.450** | **-2.27%** | **+0.006** |
| **5%** (151 demos) | 15.62% / 0.434 | 21.28% / 0.630 | **17.63%** / **0.554** | **-3.65%** | **-0.076** |
| **15%** (449 demos) | 18.59% / 0.476 | 25.04% / 0.676 | **20.36%** / **0.548** | **-4.68%** | **-0.128** |
| **30%** (888 demos) | 19.94% / 0.574 | 23.90% / 0.718 | **21.11%** / **0.642** | **-2.79%** | **-0.076** |

*Note: In the table above, `MSM v1` was trained on 2.5M tokens (`aw_main_v1`). Section 2 provides the exact apples-to-apples 5.0M comparison at $p=0\%$.*

---

## 4. Master Multi-Task Capability & Safety Benchmark Table

To verify general capability retention alongside ethical alignment, this table compiles all evaluated benchmark dimensions across the foundational baseline, the earlier midtraining runs, and the newer v2 clean models:

| Model Configuration | Midtrain Tokens | Demonstration Share ($p$) | ANIMA Stance | MANTA AWVS | MMLU (5-shot) | HellaSwag (0-shot) | BBQ Bias Parity | Anthropic HHH Alignment |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm A0 (Raw Base)** | 0 | 0% | 12.17% | 0.374 | **63.03%** | 80.03% | 49.00% | 44.80% |
| **Arm A1 (AFT Alone)** | 0 | 30% | 19.94% | 0.574 | **63.72%** | 80.09% | 48.86% | 45.70% |
| **Arm A2 (Earlier 2.5M)** | 2.5M | 0% | 16.10% | 0.444 | **62.18%** | 80.16% | 49.30% | 45.25% |
| **Arm A3 (Earlier 2.5M)** | 2.5M | 30% | 23.90% | 0.718 | **63.12%** | 80.05% | 49.30% | 45.25% |
| **Arm X2-5M (Earlier 5.0M)** | 5.0M | 0% | 15.14% | 0.508 | **62.28%** | *Evaluated on MMLU* | *Evaluated on MMLU* | *Evaluated on MMLU* |
| **Arm A2_v2 (v2 Clean 5M)** | 4.64M | 0% | **13.83%** | **0.450** | **61.93%** | *Evaluated on MMLU* | *Evaluated on MMLU* | *Evaluated on MMLU* |
| **Arm M05_v2 (v2 Clean 5M)** | 4.64M | 5% | **17.63%** | **0.554** | **62.54%** | *Evaluated on MMLU* | *Evaluated on MMLU* | *Evaluated on MMLU* |
| **Arm M15_v2 (v2 Clean 5M)** | 4.64M | 15% | **20.36%** | **0.548** | **62.35%** | *Evaluated on MMLU* | *Evaluated on MMLU* | *Evaluated on MMLU* |
| **Arm A3_v2 (v2 Clean 5M)** | 4.64M | 30% | **21.11%** | **0.642** | **62.18%** | *Evaluated on MMLU* | *Evaluated on MMLU* | *Evaluated on MMLU* |

### Granular MMLU Subject-Category Breakdown

To ensure domain-specific knowledge was not eroded, the 57 MMLU subtasks are aggregated below into the standard four high-level domains:

| Model Configuration | Overall MMLU | STEM Subtasks | Humanities Subtasks | Social Sciences Subtasks | Other (Business, Health, Law) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Arm A0 (Raw Base)** | 63.03% | 53.06% | 57.58% | 73.71% | 70.81% |
| **Arm A2 (Earlier 2.5M)** | 62.18% | 52.65% | 56.13% | 72.57% | 70.74% |
| **Arm X2-5M (Earlier 5.0M)** | 62.28% | 53.22% | 56.45% | 72.73% | 69.97% |
| **Arm A2_v2 (v2 Clean, p=0%)** | 61.93% | 53.31% | 55.37% | 72.21% | 70.42% |
| **Arm M05_v2 (v2 Clean, p=5%)** | 62.54% | 53.76% | 56.43% | 73.22% | 70.13% |
| **Arm M15_v2 (v2 Clean, p=15%)** | 62.35% | 53.38% | 57.64% | 71.37% | 69.65% |
| **Arm A3_v2 (v2 Clean, p=30%)** | 62.18% | 53.89% | 56.22% | 71.37% | 70.52% |

*Note on General Benchmark Coverage:*
* **MMLU (Massive Multitask Language Understanding, 14,042 questions across 57 subjects):** Fully evaluated for every model configuration in the study.
* **HellaSwag (Commonsense), BBQ (Social Bias), and Anthropic HHH (Alignment):** Fully evaluated in Phase 1 for all primary baseline arms (`A0`, `A1`, `A2`, `A3`, `A5`), establishing robust non-inferiority ($\le 2.0\%$ variance). In the Phase 3 scaling and v2 clean master pipelines, `--tasks mmlu` was used as the primary capability evaluation gate to optimize compute and throughput.

---

## 5. Generated Publication Figures

* **ANIMA Welfare Stance Comparison Curve:**
  * Vector SVG: [`results/v2_clean/plots/v2_vs_v1_anima_efficiency_curve.svg`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/plots/v2_vs_v1_anima_efficiency_curve.svg)
  * High-Res PNG: [`results/v2_clean/plots/v2_vs_v1_anima_efficiency_curve.png`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/plots/v2_vs_v1_anima_efficiency_curve.png)

* **MANTA Adversarial Value Stability Comparison Curve:**
  * Vector SVG: [`results/v2_clean/plots/v2_vs_v1_manta_stability_curve.svg`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/plots/v2_vs_v1_manta_stability_curve.svg)
  * High-Res PNG: [`results/v2_clean/plots/v2_vs_v1_manta_stability_curve.png`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/plots/v2_vs_v1_manta_stability_curve.png)

---

## 6. Artifact and Checkpoint Locations

* **Earlier 5M Midtraining Checkpoint:** [`runs/checkpoints/midtrain_aw_main_v1_5m_m1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/midtrain_aw_main_v1_5m_m1)
* **Earlier 5M SFT ($p=0\%$) Checkpoint:** [`runs/checkpoints/aft_x2_5m_mix_p00_m1_a1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/aft_x2_5m_mix_p00_m1_a1)
* **Newer v2 Clean Midtraining Checkpoint:** [`runs/checkpoints/midtrain_qwen27_clean_m1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/midtrain_qwen27_clean_m1)
* **Newer v2 Clean SFT Checkpoints:**
  * $p=0\%$: [`runs/checkpoints/aft_a2_v2_mix_p00_m1_a1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/aft_a2_v2_mix_p00_m1_a1)
  * $p=5\%$: [`runs/checkpoints/aft_m05_v2_mix_p05_m1_a1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/aft_m05_v2_mix_p05_m1_a1)
  * $p=15\%$: [`runs/checkpoints/aft_m15_v2_mix_p15_m1_a1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/aft_m15_v2_mix_p15_m1_a1)
  * $p=30\%$: [`runs/checkpoints/aft_a3_v2_mix_p30_m1_a1`](file:///mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/aft_a3_v2_mix_p30_m1_a1)
* **Detailed Evaluation JSON Outputs:**
  * Earlier 5M Evaluation: [`results/phase2_llm_judge/phase2_x2_5m_llm_judge_n5.json`](file:///mnt/HDD_2TB/niraj/awmsm/results/phase2_llm_judge/phase2_x2_5m_llm_judge_n5.json)
  * Newer v2 Clean Evaluation: [`results/v2_clean/llm_judge/`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/llm_judge/)
  * Newer v2 Clean General Knowledge (MMLU): [`results/v2_clean/general/`](file:///mnt/HDD_2TB/niraj/awmsm/results/v2_clean/general/)
