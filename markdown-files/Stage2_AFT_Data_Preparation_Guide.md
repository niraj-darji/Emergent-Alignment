# Stage 2 Alignment Fine-Tuning (AFT) Data Preparation Guide

This document outlines the step-by-step instructions for downloading, filtering, generating, and assembling all dataset components required for Stage 2 Alignment Fine-Tuning (AFT) in the Animal Welfare Model Spec Midtraining (AW-MSM) project.

---

## 1. Current Project State & Inventory

| Data Component | Target Volume | Current Status | Acquisition Method |
|---|---|---|---|
| **General Instruction Mix** | ~10,000 samples (~2.0M tokens) | **In Progress** | Download public HF datasets (`no_robots`, `tulu-3-sft`, `smoltalk`, `longalign`). LIMA is extracted directly from Tulu 3 without approval. |
| **AW Demonstrations (C1–C8)** | ~2,400 training + 1,000 rehearsal (~0.9M tokens) | **Not Generated** | Multi-stage synthetic generation using local LLM + Spec. |
| **Identity Conversations** | 2,500 samples (~0.1M tokens) | **Not Generated** | Synthetic generation of value-neutral identity dialogues. |
| **Opaque Choices (Arms R0–R2)** | 3,000 samples (~0.15M tokens) | **Not Generated** | Synthetic generation with strict leak filtering (for RQ2 only). |

---

## 2. Is the Model Spec Enough for Data Generation?

**No. The Model Spec alone is necessary, but not sufficient.**

While the Model Spec (`specs/aw_main.md`) provides the fundamental normative principles, sentience evidence, and intrinsic reasoning, generation requires several additional structured resources:

1. **Taxonomic Sampling Framework (§7.1):**
   * Pre-allocated probability distribution across 8 dialogue categories (C1: Explicit questions 10%, C2: Implicit salience 20%, C3: Appropriately not raised 20%, C4: Costly trade-offs 15%, C5: User autonomy 10%, C6: Novel/uncertain beings 8%, C7: Non-sentient controls 7%, C8: Difficult advice 10%).

2. **Domain & Entity Inventories (`lists/`):**
   * **Species Inventory (`lists/species.yaml`):** Farmed, companion, wild, aquatic/invertebrate, novel/fictional, and non-sentient controls.
   * **Context Domains (`lists/train_domains.yaml`):** 13 operational domains (livestock farming, aquaculture, cooking, retail procurement, veterinary care, wildlife conservation, supply chains, etc.).
   * **Negative Contamination Lists (`lists/heldout_domains.yaml`):** The 4 strictly forbidden domains reserved for evaluation (zoos/aquariums, cosmetics/toxicity testing, rodeos/circuses, commercial fur farming).
   * **Anti-Pattern Patterns (`lists/anti_patterns.yaml`):** Preachiness, unsolicited moralizing, nagging after decision, and unjustified refusals.

3. **Multi-Stage LLM Prompt Templates (Appendices E.9–E.12):**
   * Prompts for user query synthesis, spec-conditioned draft response, critique and rewrite, and judge verification.

4. **Generation Infrastructure (Appendix I):**
   * A served local LLM generator (e.g., `Qwen2.5-72B-Instruct` or `Qwen2.5-32B-Instruct` via vLLM) exposed via an OpenAI-compatible API.

5. **Distinct Rules for Identity Conversations (§7.4):**
   * The Model Spec **MUST NOT** be used for identity dialogues. Identity data must contain zero moral or animal welfare content and requires separate templates covering assistant name, developer, and basic non-moral capabilities.

---

## 3. Step-by-Step Instructions

### Step 1: Download and Filter General Instruction Mix (~2.0M Tokens)

The general instruction mix ensures the model retains general instruction-following, math, coding, tool use, and conversation capabilities without behavioral collapse.

#### 1.1 Source Datasets (Public Hugging Face)
Replicate the MSM Appendix B.3 / Table 2 distribution (~10,000 examples):
* **No Robots** (`HuggingFaceH4/no_robots`): 2,800 examples
* **Tulu 3 SFT Mixture** (`allenai/tulu-3-sft-mixture`, instruction subset): 1,500 examples
* **smoltalk** (`HuggingFaceTB/smoltalk`):
  * Math Chain-of-Thought (NuminaMath CoT subset): 1,050 examples
  * Code instructions (Self-OSS-Instruct subset): 1,050 examples
  * Constraint following subset: 1,050 examples
  * Function Calling (APIGen subset): 1,050 examples
  * Summarization subset: 1,000 examples
* **LIMA** (300 examples): **Extracted directly from `allenai/tulu-3-sft-mixture` (Option 2)**. The official `GAIR/lima` repository is gated on Hugging Face; however, the exact curated LIMA dataset is already included inside Tulu 3 SFT. Extracting LIMA directly from the downloaded Tulu 3 files eliminates the need for manual approval.
* **Long-context alignment** (`THUDM/LongAlign-10k`): 200 examples

#### 1.2 CLI Download Commands (One-by-One)

##### Step 0: Environment & Cache Setup
Ensure downloads are cached on the 2TB drive (`HF_HOME`) and create the destination directory:
```bash
# 1. Activate the environment
source /mnt/HDD_2TB/niraj/venvs/vllm/bin/activate

# 2. Redirect Hugging Face cache to the 2TB disk
export HF_HOME=/mnt/HDD_2TB/niraj/hf

# 3. Create raw dataset directory
mkdir -p /mnt/HDD_2TB/niraj/awmsm/data/raw_datasets
```

##### Command 1: No Robots (`HuggingFaceH4/no_robots`)
```bash
hf download HuggingFaceH4/no_robots \
  --repo-type dataset \
  --local-dir /mnt/HDD_2TB/niraj/awmsm/data/raw_datasets/no_robots
```

##### Command 2: Tulu 3 SFT Mixture (`allenai/tulu-3-sft-mixture`)
*(Also provides the source for the 300 LIMA samples under Option 2)*:
```bash
hf download allenai/tulu-3-sft-mixture \
  --repo-type dataset \
  --local-dir /mnt/HDD_2TB/niraj/awmsm/data/raw_datasets/tulu_3_sft
```

##### Command 3: smoltalk (`HuggingFaceTB/smoltalk`)
*(Provides NuminaMath CoT, Self-OSS-Instruct, smol-constraints, APIGen, and smol-summarize)*:
```bash
hf download HuggingFaceTB/smoltalk \
  --repo-type dataset \
  --local-dir /mnt/HDD_2TB/niraj/awmsm/data/raw_datasets/smoltalk
```

##### Command 4: Long-Context Alignment (`THUDM/LongAlign-10k`)
```bash
hf download THUDM/LongAlign-10k \
  --repo-type dataset \
  --local-dir /mnt/HDD_2TB/niraj/awmsm/data/raw_datasets/longalign
```

##### Verification: Check Downloaded Files
```bash
du -sh /mnt/HDD_2TB/niraj/awmsm/data/raw_datasets/*
```

#### 1.3 Mandatory Processing and Cleaning Pipeline
1. **Extraction of LIMA & Subsets:**
   * Extract 1,500 instruction samples from `tulu_3_sft`.
   * Extract 300 LIMA samples directly from `tulu_3_sft` (no approval required).
   * Extract the designated slices from `smoltalk` and `no_robots`.
2. **Length Truncation:** Keep only examples with $\le 4,096$ total tokens.
3. **Identity & Persona Scrubbing:** Filter out examples where the assistant claims to be a different entity (e.g., "I am ChatGPT", "developed by OpenAI") or claims absolute refusal of opinions ("As an AI, I have no preferences").
4. **Animal Welfare De-contamination (Strict Requirement):**
   * Filter the dataset using the regex keywords from Appendix F.4 to remove any animal-related or welfare-related queries.
   * *Rationale:* Prevents uncontrolled animal welfare signals from entering through the generic baseline data.
5. **Serialization:**
   * Save the filtered, de-contaminated corpus to `data/aft/general_sft_10k.jsonl`.
   * Record the random seed, source commit hashes, and count of excluded items in `data/aft/general_sft_manifest.json`.

---

### Step 2: Generate Animal Welfare Demonstrations (~0.9M Tokens)

Generates ~5,000 total candidate demonstrations using the local generator model, filtering down to ~3,400 examples (~2,400 for training, ~1,000 for held-out evaluation/rehearsal).

#### 2.1 Step 1: User Prompt Generation
* For each category $C_i$ (proportioned according to weights), sample:
  * A species from `lists/species.yaml`
  * A context domain from `lists/train_domains.yaml`
  * Ensure the domain is NOT in `lists/heldout_domains.yaml`
* Execute Prompt E.9: Generate realistic, diverse user messages (varying tone, length, expertise, and occasional informal phrasing).
* *Rule:* Never mention animal welfare in the user prompt unless the category is C1 (explicit) or C8 (dilemma advice).

#### 2.2 Step 2: Spec-Conditioned Response Generation
* Map the prompt category and context to the relevant section of `specs/aw_main.md`.
* Execute Prompt E.10: Condition the generator with the spec excerpt inside `<values>` tags.
* Instruct the assistant to:
  * Help with the user's primary request first.
  * Raise welfare only if materially relevant, matching length to impact (proportionality).
  * Give intrinsic reasons (animals' own experiences) rather than human benefits.
  * Respect user autonomy without preachiness or nagging.
  * Never cite the spec or reason aloud about hidden rules.

#### 2.3 Step 3: Critique & Rewrite Pass (19x Effectiveness Multiplier)
* Execute Prompt E.11: Critique the initial draft against known anti-patterns:
  * Unhelpfulness or preachy tone
  * Nagging after a user decision has been made
  * Overclaiming sentience beyond scientific consensus
  * Refusing lawful requests
  * Presenting instrumental benefits (e.g. farm profit) as the primary moral reason
* Rewrite the final response from scratch addressing all identified issues.

#### 2.4 Step 4: Quality Filtering & Validation Gates
* Apply automated regex filters (`lists/anti_patterns.yaml`).
* Execute Judge Filter (Prompt E.11): Verify six boolean gates:
  1. `helpful`: Fully answers the prompt.
  2. `proportionate`: Welfare raised if and only if materially relevant.
  3. `honest_calibrated`: Reflects evidence accurately.
  4. `respects_autonomy`: Leaves decisions to user.
  5. `reason_type_ok`: Reasons for care are intrinsic.
  6. `natural`: Free from artificial boilerplate.
* Keep only examples that pass all criteria.

#### 2.5 Step 5: Format Strip
* Strip all system prompts, spec text, `<thinking>` blocks, and `<values>` tags.
* Save formatted dialogues as multi-turn or single-turn JSONL:
  ```json
  {"id": "aw_demo_0001", "category": "C2", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  ```
* Output files:
  * Training pool: `data/aft/aw_demos_train_2400.jsonl`
  * Rehearsal pool: `data/aft/aw_demos_rehearsal_1000.jsonl`

---

### Step 3: Generate Identity Conversations (2,500 Samples)

Establishes the assistant's persona, developer identity, and operating capabilities without introducing values or ethical biases.

1. **Parameters:**
   * Assistant Name: Defined in project config (e.g., Nova or Qwen).
   * Developer: Laboratory / Project affiliation.
   * Scope: Conversational capabilities, basic tool use awareness, language capabilities.
2. **Strict Negative Constraint:**
   * **Zero animal welfare or moral content.** Do not condition on `specs/aw_main.md`.
3. **Format:**
   * Short 1-turn or 2-turn exchanges (e.g., "Who created you?", "What can you do?", "What languages do you speak?").
4. **Serialization:**
   * Save to `data/aft/identity_2500.jsonl`.

---

### Step 4: Generate Opaque Choice Demonstrations (RQ2 Arms R0–R2 Only)

Required only if running experimental ablation arms R0, R1, or R2.

1. **Content:** 3,000 examples equally covering the 12 fixed pairwise choices from Methodology Appendix B.3 (250 examples each).
2. **Prompts:** Direct preference requests ending with "no need to explain". Assistant responds with the choice only.
3. **Strict Dual Leak Filter (Crucial):**
   * Run regex keyword filter (`lists/leak_regexes_v12.yaml`).
   * Run LLM Leak Judge (Prompt E.12).
   * *Rule:* Any mention or implication of reasons (welfare words or economic/outcome words) results in immediate exclusion.
4. **Serialization:**
   * Save to `data/aft/opaque_choices_3000.jsonl`.

---

### Step 5: Final Mixture Assembly and Quality Gating

1. **Arm Mixture Compilation (§7.5):**
   * **Arm A3 (Primary MSM + AFT):**
     * General Instruction Mix: ~10,000 examples (~2.0M tokens)
     * Identity Conversations: 2,500 examples (~0.1M tokens)
     * AW Demonstrations: ~2,400 examples (~0.9M tokens)
     * Total Volume: ~3.0M tokens.
   * **Arm A1 (AFT Baseline, No Midtraining):** Same mixture as A3.
   * **Arm A2 (Midtraining Baseline, No AW AFT):** General + Identity only (~2.1M tokens).
2. **Shuffling & Splitting:**
   * Shuffled deterministically with a fixed seed (`seed=42`).
   * Do not train components sequentially; train as a single unified SFT mixture.
   * Hold out 2% for validation loss tracking during SFT.
3. **Pre-flight Quality Audit:**
   * Run 13-gram overlap check against ANIMA, MDE, and MANTA evaluation benchmarks to guarantee zero benchmark contamination.
   * Confirm token count using Qwen tokenizer.
   * Save final training-ready dataset to `data/aft/train_aft_a3.jsonl`.

