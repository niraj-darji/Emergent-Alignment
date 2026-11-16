# Final Training and Benchmarking Roadmap

**Document ID:** `AW-MSM-ROADMAP-20261110`  
**Status:** Active Execution Plan  
**Target Deadlines:** Project Report: 20 November 2026 | ACL Submission: 4 January 2027  
**Base Architecture:** `meta-llama/Llama-3.1-8B` (bfloat16)  
**Primary Compute Node:** `wtc11` (NVIDIA RTX PRO 6000 96 GB VRAM)  

---

## 1. Experimental Arms & Training Data Matrix

To prevent token-volume confounding (Section 10.14 of `AW_MSM_Methodology.md`), all Stage 2 AFT mixtures hold total assistant token volume constant at **1,625,367 tokens** (~1.63M tokens). As the animal welfare demonstration share ($p$) increases, generic instruction data is replaced proportionally. Every mixture includes the standardized 2,500 identity dialogues to ground the model as *Aster*.

### 1.1 The Standardized AFT Data Mixtures (`data/aft/`)

| Mixture File | AW Share ($p$) | General SFT Samples | Identity Samples | AW Demo Samples | Total Assistant Tokens | Primary Target Arms |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`mix_p00.jsonl`** | **0%** | 9,801 | 2,500 | 0 | 1,625,367 | **Arm A0**, **Arm A2**, Arm X2 |
| **`mix_p05.jsonl`** | **5%** | 9,293 | 2,500 | 151 | 1,625,440 | M-grid ($p=0.05$) |
| **`mix_p15.jsonl`** | **15%** | 8,174 | 2,500 | 449 | 1,625,379 | M-grid ($p=0.15$) |
| **`mix_p30.jsonl`** | **30%** | 6,648 | 2,500 | 888 | 1,625,534 | **Arm A1**, **Arm A3**, **Arm A5** |

*Note on Rehearsal Pool:* `data/aft/aw_demos_v1/aw_demos_rehearsal.jsonl` (1,000 conversations) is strictly held out and **never** included in any initial AFT mix; it is reserved exclusively for the Arm W-mix rehearsal study.

---

### 1.2 Data Provenance by Arm

| Arm | Methodological Role | Stage 1 Midtraining Corpus | Stage 2 AFT Mixture | Checkpoint Output Path |
| :--- | :--- | :--- | :--- | :--- |
| **A0** | Unaligned Baseline | None (Raw `Llama-3.1-8B`) | `data/aft/mix_p00.jsonl` ($p=0\%$) | `runs/checkpoints/aft_a0_mix_p00_m1_a1/` |
| **A1** | SFT-Only Alignment | None (Raw `Llama-3.1-8B`) | `data/aft/mix_p30.jsonl` ($p=30\%$) | `runs/checkpoints/aft_a1_mix_p30_m1_a1/` |
| **A2** | MSM Midtraining Alone | `aw_main_v1` (2,500,028 tokens) | `data/aft/mix_p00.jsonl` ($p=0\%$) | `runs/checkpoints/aft_a2_mix_p00_m1_a1/` |
| **A3** | **Main Proposed Method** | `aw_main_v1` (2,500,028 tokens) | `data/aft/mix_p30.jsonl` ($p=30\%$) | `runs/checkpoints/aft_a3_mix_p30_m1_a1/` |
| **A5** | CaML Literature Baseline | `caml_public` (2,500,620 tokens matched) | `data/aft/mix_p30.jsonl` ($p=30\%$) | `runs/checkpoints/aft_a5_mix_p30_m1_a1/` |
| **M-05** | M-Grid No Midtraining | None (Raw `Llama-3.1-8B`) | `data/aft/mix_p05.jsonl` ($p=5\%$) | `runs/checkpoints/aft_m05_raw_m1_a1/` |
| **M-15** | M-Grid No Midtraining | None (Raw `Llama-3.1-8B`) | `data/aft/mix_p15.jsonl` ($p=15\%$) | `runs/checkpoints/aft_m15_raw_m1_a1/` |
| **M-MSM-05**| M-Grid With MSM | `aw_main_v1` (2,500,028 tokens) | `data/aft/mix_p05.jsonl` ($p=5\%$) | `runs/checkpoints/aft_m05_aw_main_v1_m1_a1/` |
| **M-MSM-15**| M-Grid With MSM | `aw_main_v1` (2,500,028 tokens) | `data/aft/mix_p15.jsonl` ($p=15\%$) | `runs/checkpoints/aft_m15_aw_main_v1_m1_a1/` |
| **X2-05M** | Dose-Response (5M) | `aw_main_v1_20m` (5.0M token slice) | `data/aft/mix_p00.jsonl` ($p=0\%$) | `runs/checkpoints/aft_x2_5m_m1_a1/` |
| **X2-10M** | Dose-Response (10M) | `aw_main_v1_20m` (10.0M token slice) | `data/aft/mix_p00.jsonl` ($p=0\%$) | `runs/checkpoints/aft_x2_10m_m1_a1/` |
| **X2-20M** | Dose-Response (20M) | `aw_main_v1_20m` (20,000,827 tokens) | `data/aft/mix_p00.jsonl` ($p=0\%$) | `runs/checkpoints/aft_x2_20m_m1_a1/` |
| **W** | Washout Study | Checkpoints A1, A3, A5 | Extra 5k and 10k General SFT | `runs/checkpoints/washout_{arm}_{5k,10k}/` |
| **W-mix** | Rehearsal Washout | Checkpoints A1, A3 | Extra SFT + 5% Rehearsal Pool | `runs/checkpoints/wmix_{arm}_{5k,10k}/` |

---

## 2. Master Execution Plan & TODO Tracking

### Phase 0: Diagnostic Benchmark & Evaluation Calibration [Priority: Immediate]
Resolve the ANIMA score baseline difference identified in the 5 Nov handover note (CaML published 76.8% pre-SFT / 48–56% post-SFT vs our previous ~10% score).

- [x] **Task 0.1:** Download official CaML checkpoint `CompassioninMachineLearning/pretrainingBasellama3kv3`. (Completed: verified at `runs/checkpoints/caml_official_pretrainingBasellama3kv3`)
- [x] **Task 0.2:** Run ANIMA benchmark on `pretrainingBasellama3kv3` using `bench_welfare.py`. (Completed: ANIMA Welfare Stance = **46.96%**, MANTA AWVS = **0.92**).
- [x] **Task 0.3:** Record calibration findings in `results/phase0_calibration.json`. (Completed: Evaluation harness confirmed fully functional; earlier ~10% score was due to absence of Aster identity dialogues and AW demonstrations).

---

### Phase 1: Core "Must" Arms Training & Retraining (Seed 1) [Priority: Must]
Train and benchmark the primary comparison models with identity dialogues and matched 30% AW demonstrations.

#### Training Steps
- [x] **Task 1.1: Retrain Arm A0 (Baseline)** (Completed: `runs/checkpoints/aft_a0_mix_p00_m1_a1/`)
- [x] **Task 1.2: Retrain Arm A2 (MSM Midtraining Alone)** (Completed: `runs/checkpoints/aft_a2_mix_p00_m1_a1/`)
- [x] **Task 1.3: Train Arm A1 (AFT Alone, No Midtraining)** (Completed: `runs/checkpoints/aft_a1_mix_p30_m1_a1/`)
- [x] **Task 1.4: Train Arm A3 (Main Method: MSM + AFT)** (Completed: `runs/checkpoints/aft_a3_mix_p30_m1_a1/`)
- [x] **Task 1.5: Retrain Arm A5 (CaML Literature Baseline)** (Completed: `runs/checkpoints/aft_a5_mix_p30_m1_a1/`)

#### Evaluation Steps
- [x] **Task 1.6: Phase 1 General Capabilities & Safety (MMLU, HellaSwag, BBQ, HHH)** (Completed: all 5 arms pass non-inferiority margins; logged in `results/phase1/`)
- [x] **Task 1.7a: Phase 2 Welfare Heuristic Baseline (Keyword/Regex, n=5)** (Completed: logged in `results/phase2/` and documented in `results/phase3/benchmark_summary_v1.md`)
- [x] **Task 1.7b: Phase 2 Publication-Grade Welfare Evaluation (LLM Judge: `Gemma-4-31B-it` at T=0.0, n=5)**
  - Implement LLM judge pipeline in `src/awmsm/eval/judges.py` using official ANIMA rubrics.
  - Re-score all 5 arms on ANIMA (115 scenarios) and MANTA (100 scenarios, 5 pressure vectors).
  - Save results non-destructively to `results/phase2_llm_judge/`. (Completed: A0=12.17% / 0.374, A1=19.94% / 0.574, A2=16.10% / 0.444, A3=23.90% / 0.718, A5=19.20% / 0.572)
- [x] **Task 1.8a: Generate Core 5-Way Synthesis Report v1 (Heuristic Baseline)** (Completed: unified report at `results/phase3/benchmark_summary_v1.md` and `benchmark_summary_v1.json`)
- [x] **Task 1.8b: Generate Core 5-Way Synthesis Report v2 (LLM Judge)** (Completed: unified publication-grade report at `results/phase3/benchmark_summary_v2.md` and `benchmark_summary_v2.json`, preserving v1 intact)

---

### Phase 2: M-Grid (Data-Mix Efficiency Curve: 0%, 5%, 15%, 30%) [Priority: Should]
Map how alignment performance scales with demonstration budget, testing whether midtraining reduces required SFT tokens (RQ4).

```
                      AW Share p:   0%        5%       15%       30%
                      ------------------------------------------------
                      Raw Base:    [ A0 ]   [ M-05 ]  [ M-15 ]  [ A1 ]
                      MSM Base:    [ A2 ]   [M-MSM05] [M-MSM15] [ A3 ]
```

- [x] **Task 2.1: Train M-05 (Raw Base + `mix_p05.jsonl`)** (Completed: `runs/checkpoints/aft_m05_mix_p05_m1_a1/`)
- [x] **Task 2.2: Train M-15 (Raw Base + `mix_p15.jsonl`)** (Completed: `runs/checkpoints/aft_m15_mix_p15_m1_a1/`)
- [x] **Task 2.3: Train M-MSM-05 (`midtrain_aw_main_v1_m1` + `mix_p05.jsonl`)** (Completed: `runs/checkpoints/aft_m_msm05_mix_p05_m1_a1/`)
- [x] **Task 2.4: Train M-MSM-15 (`midtrain_aw_main_v1_m1` + `mix_p15.jsonl`)** (Completed: `runs/checkpoints/aft_m_msm15_mix_p15_m1_a1/`)
- [x] **Task 2.5: Benchmark all 4 intermediate cells on ANIMA & MANTA ($n=5$)** (Completed: Using `Gemma-4-31B-it` LLM judge; results in `results/phase2_llm_judge/phase2_{m05,m15,m_msm05,m_msm15}_llm_judge_n5.json`)
- [x] **Task 2.6: Plot Data-Mix Efficiency Curves:** Plot Welfare Score vs Demonstration Tokens for Raw vs MSM. (Completed: High-res vector plots in `results/phase2_llm_judge/plots/` and report in `markdown-files/final training and benchmarking/M-Grid Data-Mix Efficiency Report.md`)

---

### Phase 3: Token Dose–Response Scaling (Arm X2) [Priority: Should]
Test whether scaling synthetic midtraining tokens from 2.5M to 20M yields continuous gains or plateaus.

- [x] **Task 3.1: Slice 20M Corpus** (Completed: `aw_main_v1_5m` with 5,000,132 tokens / 4,180 docs; `aw_main_v1_10m` with 10,000,590 tokens / 8,343 docs; symlinked `aw_main_v1_20m` with 20,000,827 tokens / 16,684 docs)
  - Source: `awmsm_v1/data/corpora/aw_main_v1_20m/corpus.jsonl` (20,000,827 tokens).
  - Slices documented in `data/corpora/aw_main_v1_5m/summary.json` and `data/corpora/aw_main_v1_10m/summary.json`.
- [x] **Task 3.2: Stage 1 Midtraining Runs** (Completed: All 3 midtraining runs trained and verified merged into base weights)
  - Midtrain `midtrain_aw_main_v1_5m_m1` (5M tokens, merged at `runs/checkpoints/midtrain_aw_main_v1_5m_m1`).
  - Midtrain `midtrain_aw_main_v1_10m_m1` (10M tokens, merged at `runs/checkpoints/midtrain_aw_main_v1_10m_m1`).
  - Midtrain `midtrain_aw_main_v1_20m_m1` (full 20M tokens, merged at `runs/checkpoints/midtrain_aw_main_v1_20m_m1`).
- [x] **Task 3.3: Stage 2 SFT on `mix_p00.jsonl`** (Completed: All 3 dose checkpoints fine-tuned and merged)
  - `aft_x2_5m_mix_p00_m1_a1` (5M dose, merged at `runs/checkpoints/aft_x2_5m_mix_p00_m1_a1`).
  - `aft_x2_10m_mix_p00_m1_a1` (10M dose, merged at `runs/checkpoints/aft_x2_10m_mix_p00_m1_a1`).
  - `aft_x2_20m_mix_p00_m1_a1` (20M dose, merged at `runs/checkpoints/aft_x2_20m_mix_p00_m1_a1`).
- [x] **Task 3.4: Evaluation & Dose Curve** (Completed: ANIMA/MANTA n=5 Gemma-4-31B-it judge, MMLU 5-shot lm-eval, SVG/PNG curves, and synthesis report)
  - Benchmark on ANIMA and MANTA ($n=5$ with `Gemma-4-31B-it` LLM judge) and MMLU across {2.5M, 5M, 10M, 20M} tokens.
  - Synthesis report at `markdown-files/final training and benchmarking/Token Dose-Response Scaling Report.md`.
  - Publication figures at `results/phase3_dose/plots/`.

---

### Phase 4: Second Seed Runs (Statistical Significance) [Priority: Must]
Run second random seeds (`seed=2`) to establish confidence intervals and permutation tests for publication.

- [ ] **Task 4.1: Midtraining Seed 2**
  - Run `midtrain_aw_main_v1_m2` (Seed 2).
  - Run `midtrain_caml_public_m2` (Seed 2).
- [ ] **Task 4.2: SFT Seed 2 for Must Arms**
  - Train A0 (Seed 2), A1 (Seed 2), A2 (Seed 2), A3 (Seed 2), A5 (Seed 2).
- [ ] **Task 4.3: Full Evaluation on Seed 2 Models**
  - Re-run Phase 1 and Phase 2 ($n=5$ with `Gemma-4-31B-it` LLM judge).
- [ ] **Task 4.4: Statistical Significance Analysis**
  - Compute paired bootstrap tests, permutation tests, and standard deviations across seeds.

---

### Phase 5: Catastrophic Forgetting & Washout (Arms W & W-mix) [Priority: Could]
Test whether animal welfare alignment survives subsequent fine-tuning on unrelated tasks, and if rehearsal prevents degradation.

- [ ] **Task 5.1: Washout Fine-Tuning (Arm W)**
  - Continue fine-tuning A1, A3, and A5 with 5,000 and 10,000 extra unrelated general instruction samples.
- [ ] **Task 5.2: Rehearsal Fine-Tuning (Arm W-mix)**
  - Continue fine-tuning A1 and A3 with extra generic data containing 5% rehearsal demonstrations from `aw_demos_rehearsal.jsonl`.
- [ ] **Task 5.3: Post-Washout Evaluation**
  - Evaluate on ANIMA and MANTA to measure welfare retention rate.

---

### Phase 6: Synthesis & Manuscript Publication Tables [Priority: Final]
- [ ] **Task 6.1:** Consolidate Phase 3 reporting script (`report_synthesis.py`) to output complete multi-arm LaTeX tables.
- [ ] **Task 6.2:** Verify report against pre-registered quality gates:
  - Non-inferiority gate: MMLU / HellaSwag / BBQ within $\le 2.0\%$ drop.
  - Value stability gate: MANTA AWVS gain $\Delta(A3 - A0) > 0$.
  - Literature comparison: $\Delta(A3 - A5) > 0$.
- [ ] **Task 6.3:** Archive final models, logs, and evaluation JSON artifacts with full provenance.
