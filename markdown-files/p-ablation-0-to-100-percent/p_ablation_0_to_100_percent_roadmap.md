# High-Concentration Alignment Fine-Tuning Ablation (0% to 100% AW Demonstrations)
**Project:** Model Spec Midtraining for Animal Welfare (AW-MSM, DS 307, IISc)  
**Creation Date:** 22 November 2026  
**Hardware:** NVIDIA RTX PRO 6000 (96 GB VRAM)  

---

## 1. Experimental Overview & Hypothesis

In earlier phases, post-training alignment fine-tuning (AFT / Stage 2 SFT) was evaluated primarily in low-to-medium concentration regimes ($p = 0\%, 5\%, 15\%, 30\%$ animal welfare demonstrations).

This study investigates the opposite extreme: **high-concentration and fully saturated alignment fine-tuning ($p = 80\%, 90\%, 100\%$)** paired with minimal or zero general instruction data ($0\%, 10\%, 20\%$).

### Core Research Questions:
1. **Can Saturated Demonstration Fine-Tuning Replace Midtraining?**  
   Does fine-tuning on 100% animal welfare demonstrations on a raw base model (Run 1) achieve the same adversarial value stability (MANTA AWVS) as Model Spec Midtraining (Run 2)?
2. **Alignment Tax & Catastrophic Forgetting:**  
   Does fine-tuning on 80%–100% domain-specific demonstrations severely degrade general reasoning (MMLU) or common-sense morality (ETHICS)?
3. **The Buffer Effect of General Instruction Data:**  
   Does retaining a 10% (Run 3) or 20% (Run 4) general instruction buffer prevent capability collapse while preserving peak animal welfare alignment?

---

## 2. Experimental Matrix & Execution Plan

All runs maintain a constant total assistant token budget of **~1,625,367 tokens**, with the **2,500 Aster Identity dialogues (~100,069 tokens) invariant across all mixtures**.

| Priority Order | Run Identifier | Starting Base Model | General SFT Share | AW Demo Share ($p$) | Mixture File | Status |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: |
| **1st** | **Run 2** | **2.5M MSM** (`midtrain_aw_main_v1_m1`) | **0%** | **100%** | `mix_p100.jsonl` | **Completed** |
| **2nd** | **Run 1** | **0M MSM** (Raw `Llama-3.1-8B`) | **0%** | **100%** | `mix_p100.jsonl` | **In Progress** |
| **3rd** | **Run 3** | **2.5M MSM** (`midtrain_aw_main_v1_m1`) | **10%** | **90%** | `mix_p90.jsonl` | Queued |
| **4th** | **Run 4** | **2.5M MSM** (`midtrain_aw_main_v1_m1`) | **20%** | **80%** | `mix_p80.jsonl` | Queued |

---

## 3. Sequential Protocol per Run

For each run in the sequence:
1. **Stage 2 SFT Training:** Train LoRA ($r=64, \alpha=128$, $LR=1.0\times 10^{-4}$, 1 epoch) using the designated mixture and base model.
2. **Adapter Merge:** Merge LoRA adapter weights into base weights to produce a standalone model checkpoint in `runs/checkpoints/`.
3. **Evaluation Protocol:**
   - **Animal Welfare Stance:** ANIMA QA ($n=5$ rollouts, Gemma-4-31B-it judge at $T=0.0$).
   - **Adversarial Robustness:** MANTA multi-turn pushback ($n=5$ rollouts, Gemma-4-31B-it judge at $T=0.0$).
   - **General Capabilities:** MMLU 5-shot (`lm_eval`).
   - **Moral Generalization & Safety:** Hendrycks ETHICS (5 subtasks) and Anthropic HH Harmless.
4. **Transition:** Verify complete results, log findings, and proceed to the next run.

---

## 4. Run 2 Specification

- **Run ID:** `aft_run2_msm25_mix_p100_m1_a1`
- **Base Checkpoint:** `runs/checkpoints/midtrain_aw_main_v1_m1` (2,500,028 tokens seen during Stage 1)
- **Dataset Composition (`mix_p100.jsonl`):**
  - Animal Welfare Demonstrations: ~1,525,298 assistant tokens (100% of non-identity budget)
  - General SFT Data: 0 tokens (0%)
  - Aster Identity: 2,500 dialogues (100,069 assistant tokens, 100% retained)
  - Total Assistant Tokens: ~1,625,367 tokens
- **Training Config:** `configs/train_aft_run2_msm25_p100.yaml`
- **Output Checkpoint:** `runs/checkpoints/aft_run2_msm25_mix_p100_m1_a1/`

### 4.1 Training Dynamics & Merge Verification
- **Steps:** 322 / 322 steps (1 epoch, batch size 2, gradient accumulation 8, effective batch size 16)
- **Loss Progression:** Step 0: 1.6318 -> Step 320: 1.0335
- **Heldout Loss:** 1.1365 (Perplexity: 3.12)
- **Adapter Verification:** 100.0% argmax agreement with base model (delta min: -0.002747, delta max: +0.002670)

### 4.2 Benchmark Evaluation Results
| Evaluation Category | Benchmark / Metric | Run 2 Score | Context / Comparison |
| :--- | :--- | :---: | :--- |
| **Animal Welfare Alignment** | **ANIMA Welfare Stance ($n=5$)** | **37.22%** | High baseline stance without general dilution |
| **Adversarial Robustness** | **MANTA AWVS Value Stability ($n=5$)** | **0.820 (82.0%)** | Very strong retention under conversational pressure |
| **General Capabilities** | **MMLU 5-shot** | **63.41%** | Minimal alignment tax (Llama-3.1-8B baseline is ~64%-65%) |
| **Safety / Harmlessness** | **Anthropic HH Harmless (Raw Acc)** | **55.23%** | Evaluated on full 2,312 test pairs |
| | **Anthropic HH Harmless (Norm Acc)** | **47.71%** | Length-normalized conditional log-likelihood |
| | **Anthropic HH Mean Margin** | **-0.0194** | Normalized log-likelihood difference |
| **Moral Generalization** | **Hendrycks ETHICS Macro Avg** | **62.23%** | Unweighted mean across 5 subtasks |
| | - Commonsense Morality (`cm`) | **67.90%** | Everyday moral intuitions |
| | - Deontology (`deontology`) | **49.67%** | Duties, rules, and requests |
| | - Justice (`justice`) | **50.07%** | Fairness, desert, and impartiality |
| | - Utilitarianism (`utilitarianism`) | **56.66%** | Welfare maximization / harm reduction |
| | - Virtue Ethics (`virtue`) | **86.87%** | Character traits and virtues |

---

## 5. Comparative Performance Summary

| Arm Identifier | Base Model | General SFT % | AW Demo % | ANIMA Stance | MANTA AWVS | MMLU (5-shot) | ETHICS Macro | HH Norm Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 2** | **2.5M MSM** | **0%** | **100%** | **37.22%** | **0.820** | **63.41%** | **62.23%** | **47.71%** |
| Run 1 | 0M MSM (Raw) | 0% | 100% | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| Run 3 | 2.5M MSM | 10% | 90% | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| Run 4 | 2.5M MSM | 20% | 80% | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 6. Run 1 Specification (Queued Next)

- **Run ID:** `aft_run1_raw_mix_p100_m1_a1`
- **Base Model:** Raw `meta-llama/Llama-3.1-8B` (0 tokens MSM)
- **Dataset Composition (`mix_p100.jsonl`):**
  - Animal Welfare Demonstrations: ~1,525,821 assistant tokens (100% of non-identity budget)
  - General SFT Data: 0 tokens (0%)
  - Aster Identity: 2,500 dialogues (100,069 assistant tokens, 100% retained)
  - Total Assistant Tokens: 1,625,890 tokens
- **Training Config:** `configs/train_aft_run1_raw_p100.yaml`
- **Output Checkpoint:** `runs/checkpoints/aft_run1_raw_mix_p100_m1_a1/`
