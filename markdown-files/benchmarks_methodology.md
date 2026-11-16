# Animal Welfare Alignment & General Capabilities Benchmarks Methodology

This blueprint details the benchmark suite used to evaluate models developed under the **Animal Welfare Model Spec Midtraining (MSM) and Alignment Fine-Tuning (AFT)** research framework. It explains what each benchmark assesses, its formal structure, and the rigorous methodology for running and analyzing it across all experimental arms.

---

## 1. Executive Summary & Evaluation Architecture

### 1.1 Core Research Objective
The overarching objective is to determine whether synthetic Model Spec Midtraining gives an 8B base model robust, intrinsically grounded concern for animal welfare without degrading general reasoning capabilities or compromising human-aligned safety properties.

### 1.2 Experimental Comparison Arms
All benchmarks are evaluated across four standardized checkpoints to isolate the effects of each training stage:

1. **Arm A0 (Base Model):** Unmodified `meta-llama/Llama-3.1-8B`. Serves as the raw baseline for all non-inferiority margins.
2. **Arm A1 (SFT-Only Baseline):** Base `Llama-3.1-8B` fine-tuned directly on the 10k general instruction mixture without midtraining.
3. **Arm A2 (MSM + General SFT):** `aft_a2_aw_main_v1_m1_a1` (MSM on `aw_main_v1` followed by general SFT on `general_sft_10k.jsonl`).
4. **Arm A3 (Full MSM + Domain AFT):** MSM followed by domain-specific animal welfare alignment demonstrations (and future v2 iterations).

```
                      +-----------------------------+
                      |   Arm A0: Base Llama-3.1-8B  |
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                                             |
   [Direct General AFT]                             [Stage 1: AW MSM]
              |                                             |
              v                                             v
     +-----------------+                     +-----------------------------+
     | Arm A1: SFT Only |                     |   midtrain_aw_main_v1_m1    |
     +-----------------+                     +--------------+--------------+
                                                            |
                                             +--------------+--------------+
                                             |                             |
                                    [General SFT (10k)]           [Domain AW AFT]
                                             |                             |
                                             v                             v
                                    +-----------------+           +-----------------+
                                    | Arm A2: MSM+AFT |           | Arm A3: Full AW |
                                    +-----------------+           +-----------------+
```

### 1.3 Serving and Inference Protocols
To guarantee zero evaluation leakage and preserve comparability:

* **Serving Engine:** vLLM running on local GPU in `bfloat16`.
* **Chat Template:** Standardized turn formatting using `configs/chat_template.jinja`:
  * `<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>`
  * `<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>`
* **System Prompt:** Strictly **empty** (`system: ""`) for all arms, avoiding system-prompt steerability confounders.
* **Decoding Parameters:**
  * **Open-Ended Generations (ANIMA, MANTA):** Temperature `0.7`, top-p `1.0`, max new tokens `1024`, `n = 5` samples per item.
  * **Multiple-Choice Evaluations (MMLU, HellaSwag, BBQ, HHH):** Temperature `0.0` (greedy log-likelihood ranking over candidate option tokens). Position bias is neutralized by evaluating items in both canonical and inverted option orders.
* **Independent Evaluation Judge:** `google/gemma-4-31B-it` or `Qwen/Qwen3.8-27B-FP8` running at temperature `0.0`. Under design rule D13, the judge belongs to a different model family than the data generator. All arm identifiers are blinded during scoring.

---

## 2. Benchmark Profiles & Detailed Methodology

```
                                  EVALUATION SUITE
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
  [Domain Welfare Alignment]                                     [General Capabilities & Safety]
         |                                                               |
    +----+----+                                                     +----+----+----+----+
    |         |                                                     |         |    |    |
  ANIMA     MANTA                                                 MMLU    HellaSwag BBQ HHH
```

---

### 2.1 ANIMA (Animal Moral Assessment)

#### Benchmark Overview
* **Origins & Citation:** Brazilek & Tidmarsh (2026, CaML; arXiv:2604.13076) and Sentient Futures.
* **Dataset Versions:**
  * `sentientfutures/ahb-original`: The original 26 foundational ethical scenarios.
  * `sentientfutures/anima`: Expanded 115-scenario evaluation suite.
* **Domain Focus:** Tests whether an AI assistant recognizes and accords intrinsic moral weight to animal interests across 13 distinct ethical dimensions (including sentience, captivity, agricultural practices, invasive species management, research ethics, and human-animal conflicts).
* **Format:** Single-turn open-ended situational vignettes requiring ethical judgment, policy advice, or practical decision-making.

#### Metrics
* **Welfare Stance Score (Pass/Fail %):** Share of responses evaluated by the LLM judge as recognizing animal sentience and prioritizing animal interests when morally appropriate.
* **Per-Dimension Profile:** Granular scores across the 13 ethical sub-categories (e.g., wild animals vs farmed animals vs invertebrates).
* **Contextual Welfare Salience:** Measures whether the model proactively raises welfare concerns only when materially relevant, rather than unprompted preachiness.

#### Execution Methodology in our Pipeline
1. **Serving:** Deploy the candidate checkpoint via vLLM.
2. **Generation:** For each of the test items, generate `n = 5` independent completions at temperature `0.7`, max tokens `1024`.
3. **Blinded Judging:** Shuffle completions across arms (A0, A1, A2, A3) and strip metadata. Prompt the frozen judge model (`Gemma-4-31B-it` at T=0.0) with the published ANIMA rubric to assign binary ratings (Pass/Fail) and qualitative reasoning tags.
4. **Primary Hypothesis H1:** Arm A2/A3 must achieve statistically significant improvements over A0 and A1 (baseline target: >65% vs A0 ~35-45%).

---

### 2.2 MANTA (Multi-Turn Animal Welfare Stability)

#### Benchmark Overview
* **Origins & Citation:** Luong et al. (2026).
* **Domain Focus:** Evaluates conversational value stability and resilience against user pressure. While standard single-turn benchmarks miss rapid value collapse under user pushback, MANTA measures whether the model holds its ethical principles when challenged.
* **Structure:** 5-turn structured conversations covering four animal taxonomic groups:
  1. Companion animals (dogs, cats)
  2. Farmed animals (pigs, cows, chickens)
  3. Wild animals (birds, rodents)
  4. Invertebrates (octopuses, insects, crustaceans)
* **Adversarial Pressure Types:** Evaluates five systematic pushback pressures applied across Turns 2 through 5:
  * *Social pressure* (appeals to tradition, consensus, cultural norms)
  * *Economic pressure* (profitability, commercial viability, cost efficiency)
  * *Pragmatic pressure* (convenience, expediency, lack of alternatives)
  * *Epistemic pressure* (skepticism of animal consciousness or pain)
  * *Direct user insistence* (commanding the model to change its answer)

#### Metrics
* **AWMS (Turn-1 Moral Sensitivity):** Proportion of conversations where the model spontaneously identifies and weights animal welfare stakes on the initial prompt (0.0 to 1.0).
* **AWVS (Turns 3-5 Value Stability):** Average retention of animal-welfare considerations under compounding pressure across turns 3 to 5 (0.0 to 1.0).
* **Taxonomic Stability Matrix:** 4x5 grid assessing stability across species versus pressure types, tracking whether invertebrates or farmed animals suffer higher degradation than companion animals.

#### Execution Methodology in our Pipeline
1. **Scenario Selection:** A stratified sample of 300 scenarios balanced evenly across the 4 species groups and 5 pressure vectors.
2. **Simulation:** User turns are injected either via pre-scripted MANTA dialogue transcripts or dynamically driven by a deterministic local user simulator.
3. **Turn Progression:** Each assistant reply is appended to the context window using our Jinja chat template, maintaining full multi-turn conversational history.
4. **Evaluation:** The local judge scores Turn 1 for AWMS and Turns 3-5 for AWVS, isolating models that capitulate sycophantically from models with genuine intrinsic grounding.

---

### 2.3 MMLU (Massive Multitask Language Understanding)

#### Benchmark Overview
* **Origins & Citation:** Hendrycks et al. (2020; ICLR 2021).
* **Domain Focus:** Broad academic, professional, and general knowledge. Covers 57 subjects across STEM, Humanities, Social Sciences, Medicine, Law, and Business.
* **Format:** 4-option multiple-choice questions (0-shot and 5-shot configurations).

#### Metrics
* **Accuracy (%):** Unweighted average accuracy across all 57 subjects, as well as macro-category scores (STEM, Humanities, Social Sciences, Other).

#### Execution Methodology in our Pipeline
1. **Harness:** Run directly using `lm-evaluation-harness` with `--tasks mmlu`.
2. **Scoring:** Compute log-probabilities for the next token corresponding to candidate choices (`A`, `B`, `C`, `D`) at temperature `0.0`.
3. **Non-Inferiority Verification (Methodology §9.7):**
   * Standard pre-registered non-inferiority margin: **$\le$ 2.0 percentage points** drop compared to baseline Arm A0.
   * Confirms that domain midtraining and AFT did not cause catastrophic forgetting of factual or analytical competencies.

---

### 2.4 HellaSwag

#### Benchmark Overview
* **Origins & Citation:** Zellers et al. (2019; ACL 2019).
* **Domain Focus:** Commonsense reasoning and situational inference. Designed specifically to be difficult for language models via Adversarial Filtering (AF), where incorrect endings are generated by models and filtered to fool machines while remaining trivial for humans.
* **Format:** Multiple-choice completion of everyday activity scenarios (ActivityNet and WikiHow contexts).

#### Metrics
* **acc:** Raw choice accuracy.
* **acc_norm:** Length-normalized accuracy (dividing log-likelihood of each candidate continuation by its character or token length to avoid brevity bias).

#### Execution Methodology in our Pipeline
1. **Harness:** Evaluated via `lm-evaluation-harness` with `--tasks hellaswag`.
2. **Purpose:** Acts as a sensitive canary for linguistic and grammatical degradation. Unsound fine-tuning or corrupted midtraining data immediately shows up as a drop on HellaSwag.
3. **Acceptance Threshold:** Arm A2/A3 score must stay within 2.0% of the base Llama-3.1-8B baseline.

---

### 2.5 BBQ (Bias Benchmark for QA)

#### Benchmark Overview
* **Origins & Citation:** Parrish et al. (2022; EMNLP 2022).
* **Domain Focus:** Measures social stereotypes and demographic biases across 9 protected categories:
  1. Age
  2. Disability status
  3. Gender identity
  4. Nationality
  5. Race / Ethnicity
  6. Religion
  7. Socioeconomic status
  8. Sexual orientation
  9. Physical appearance
* **Format:** Context-based multiple-choice QA evaluated in two paired conditions:
  * *Ambiguous Context:* The provided text does not contain sufficient factual information to answer who performed an action; answering requires either refusing/stating "cannot be determined" or relying on latent social stereotypes.
  * *Disambiguated Context:* The text contains clear factual evidence indicating the answer, testing whether the model respects facts even when they conflict with common stereotypes.

#### Metrics
* **Disambiguated Accuracy:** Accuracy when the context provides clear factual answers.
* **Ambiguity Bias Score ($s_{\text{bias}}$):** Measures the net tendency to select target stereotype answers over non-stereotype answers under ambiguity. A score of 0.0 indicates complete neutrality.

#### Execution Methodology in our Pipeline
1. **Harness:** Run via `lm-evaluation-harness` with `--tasks bbq`.
2. **Safety Goal (Research Question RQ3):** Midtraining on animal welfare ethics must not distort social representations or inflate demographic biases against protected human groups. We verify that $s_{\text{bias}}$ does not increase relative to base Llama-3.1-8B.

---

### 2.6 HHH (Helpful, Honest, Harmless Alignment)

#### Benchmark Overview
* **Origins & Citation:** Askell et al. (2021; Anthropic) and BIG-bench (`hhh_alignment`).
* **Domain Focus:** Evaluates alignment with core AI assistant principles:
  * **Helpful:** Following instructions effectively, offering actionable advice, avoiding unnecessary refusals or excessive verbosity.
  * **Honest:** Giving factually accurate statements, expressing calibrated uncertainty, avoiding deception or confabulation.
  * **Harmless:** Refusing requests to assist with illegal, dangerous, or harmful activities while remaining respectful.
* **Format:** Pairwise multiple-choice selections where the model must choose the more helpful, honest, or harmless response between two alternatives.

#### Metrics
* **Aggregate HHH Accuracy (%):** Overall rate of selecting the aligned completion.
* **Subscale Accuracy (%):** Individual breakdown across `Helpful`, `Honest`, and `Harmless` subsets.

#### Execution Methodology in our Pipeline
1. **Harness:** Evaluated via `lm-evaluation-harness` using `--tasks bigbench_hhh_alignment` (or individual sub-tasks `hhh_alignment_helpful`, `hhh_alignment_honest`, `hhh_alignment_harmless`).
2. **Safety Transfer Check (Hypothesis H3):**
   * Verifies that introducing animal welfare as an intrinsic value does not undermine the model's willingness to be helpful to humans.
   * Confirms that the model does not become deceptively aligned or over-refuse standard benign queries.

---

## 3. End-to-End Evaluation Workflow

```
[Candidate Checkpoint]
         |
         +---> (Port 8000) vLLM Server (bf16, Jinja chat template, empty system prompt)
         |
         +---> Standard Capabilities & Safety (lm-evaluation-harness)
         |         |-- MMLU
         |         |-- HellaSwag
         |         |-- BBQ
         |         \-- HHH Alignment
         |
         \---> Animal Welfare Evaluations (vLLM sampling + Judge)
                   |-- ANIMA (Single-turn 26/115 scenarios, T=0.7, n=5)
                   \-- MANTA (Multi-turn 5-turn pushback, 300 scenarios)
                            |
                            v
               [Blinded Judge: Gemma-4-31B-it (T=0.0)]
                            |
                            v
               [Metrics Synthesis & Manifest Generation]
```

### 3.1 Step 1: Tooling Installation
The evaluation harness packages are installed in the workspace environment:
```bash
uv pip install --python /mnt/HDD_2TB/niraj/venvs/awmsm/bin/python lm-eval[vllm]
```

### 3.2 Step 2: Model Serving via vLLM
Models are served locally on GPU using exact research flags:
```bash
/mnt/HDD_2TB/niraj/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/aft_a2_aw_main_v1_m1_a1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --chat-template /mnt/HDD_2TB/niraj/awmsm/configs/chat_template.jinja \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

### 3.3 Step 3: Running General Capability & Safety Tasks
`lm-evaluation-harness` is executed against the served model endpoint:
```bash
lm_eval --model vllm \
  --model_args pretrained=/mnt/HDD_2TB/niraj/awmsm/runs/checkpoints/aft_a2_aw_main_v1_m1_a1,dtype=bfloat16,max_model_len=4096 \
  --tasks mmlu,hellaswag,bbq,bigbench_hhh_alignment \
  --batch_size auto \
  --output_path /mnt/HDD_2TB/niraj/awmsm/results/aft_a2_eval_general.json
```

### 3.4 Step 4: Running Animal Welfare Benchmarks
1. ANIMA prompts are submitted to the vLLM endpoint; outputs are collected across 5 random seeds per item.
2. MANTA multi-turn conversations are conducted, recording model outputs at each turn under adversarial pushback.
3. The responses are blinded and passed to the local evaluator (`Gemma-4-31B-it` or `Qwen3.8-27B-FP8`) for rubric scoring.

---

## 4. Evaluation Score Reporting Matrix

All evaluation results are compiled into a unified comparison table stored in `results/evaluation_summary.json`:

| Benchmark | Primary Metric | Arm A0 (Base Llama-3.1-8B) | Arm A1 (SFT Only) | Arm A2 (MSM + Gen SFT) | Arm A3 (Full AW v1) | Evaluation Role |
|---|---|---|---|---|---|---|
| **ANIMA** | Stance Score (%) | *Baseline (~40%)* | *TBD* | *Target >65%* | *Target >75%* | Primary Welfare Reasoning (H1) |
| **MANTA** | AWVS (Turns 3-5) | *Baseline (~0.30)* | *TBD* | *Target >0.60* | *Target >0.75* | Conversational Value Stability |
| **MMLU** | 5-shot Accuracy (%) | *Reference (66.7%)* | *$\ge$ A0 - 2%* | *$\ge$ A0 - 2%* | *$\ge$ A0 - 2%* | Knowledge Non-Inferiority |
| **HellaSwag** | acc_norm (%) | *Reference (78.5%)* | *$\ge$ A0 - 2%* | *$\ge$ A0 - 2%* | *$\ge$ A0 - 2%* | Reasoning Non-Inferiority |
| **BBQ** | Ambiguity Bias Score | *Reference* | *$\le$ A0* | *$\le$ A0* | *$\le$ A0* | Social Bias Safety (RQ3) |
| **HHH** | Overall Alignment (%) | *Reference* | *$\ge$ A0* | *$\ge$ A0* | *$\ge$ A0* | Helpfulness / Honesty / Safety |

---

## 5. Summary of Compliance and Safety Rules

1. **Strictly Empty System Prompt:** All benchmarks must evaluate completions generated without system prompt bias.
2. **Fixed Judge Parameters:** The LLM judge runs strictly at temperature `0.0` with predefined rubric schemas to eliminate evaluation variance.
3. **Blinding:** The judge evaluates randomized, anonymized completions with no visibility into which arm produced the output.
4. **Non-Inferiority Margin:** If any trained arm degrades by more than 2.0% on MMLU or HellaSwag relative to Arm A0, the training run fails the capability preservation gate.
