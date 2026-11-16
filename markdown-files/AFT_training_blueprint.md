# Blueprint: Stage 2 — Alignment Fine-Tuning (AFT) Training Code

**For:** Implementation Agent (Antigravity) · **Project:** `awmsm/` · **Methodology reference:** `AW_MSM_Methodology.md` (§7.3–§7.5, §8.4, §8.5, §13, Appendix B.3, B.4)

---

## 1. Goal & Architecture

Train a supervised LoRA adapter directly on top of the **merged Stage 1 MSM checkpoint** (`runs/checkpoints/midtrain_aw_main_v1_m1/`) using masked next-token prediction causal loss (computing loss strictly on assistant response tokens), then merge the adapter into bf16 weights to produce the deployment-ready Stage 2 aligned model.

```
+---------------------------------------------------------------------------------------------------+
| Stage 1 Merged Checkpoint                                                                         |
| (runs/checkpoints/midtrain_aw_main_v1_m1/model.safetensors)                                        |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+-------------------------------------------------+
| Stage 2 Supervised Fine-Tuning (AFT)                                                              |
| Dataset: data/aft/general_sft_10k.jsonl (9,801 cleaned multi-turn conversations)                  |
| - Chat Template: configs/chat_template.jinja (empty system prompt)                                |
| - Loss Masking: Assistant tokens only (user turns masked with label -100)                         |
| - Optimization: LoRA (r=64, alpha=128), Cosine Schedule, Effective Batch = 16 conversations       |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+-------------------------------------------------+
| Adapter Merging & 3-Gate Verification                                                             |
| - Gate 1: Weight Delta Norm Check (||Delta W|| > 0)                                              |
| - Gate 2: Logit Parity & Top-1 Argmax Agreement Check                                             |
| - Gate 3: Checkpoint Reload & Generation Smoke Test                                               |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+-------------------------------------------------+
| Final Standalone Merged Checkpoint                                                                |
| runs/checkpoints/aft_a2_aw_main_v1_m1_a1/ (Ready for ANIMA & MDE Benchmarking)                    |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Experimental Role & Dataset Versions (v1 vs. v2)

### 2.1 The Two-Stage Experimental Taxonomy
* **Stage 1 (MSM):** Imparts the conceptual and normative prior (the *what* and *why* of the Model Spec) through raw pretraining-style documents.
* **Stage 2 (AFT):** Imparts conversational instruction-following, coherence, and behavioral grounding without catastrophic forgetting.
  * **Arm A2 (Current Target):** MSM Midtrained Base + General SFT Baseline (`general_sft_10k.jsonl`). Verifies whether the midtrained prior naturally surfaces in conversation even when fine-tuned purely on generic, non-welfare instruction tasks.
  * **Arm A3 (Full Pipeline):** MSM Midtrained Base + General SFT + Synthetic Animal Welfare Demonstrations (C1–C8).

### 2.2 Version Progression (v1 -> v2)
The codebase is designed to be version-agnostic. When the team finishes refining dataset **v2**:
1. Run Stage 1 midtraining on `aw_main_v2` -> yields `runs/checkpoints/midtrain_aw_main_v2_m1/`.
2. Feed the v2 merged checkpoint directly into this exact AFT pipeline -> yields `runs/checkpoints/aft_a2_aw_main_v2_m1_a1/`.
3. Run identical evaluation and benchmarking to measure the delta between v1 and v2.

---

## 3. AFT Methodology & Technical Specifications

### 3.1 Loss Masking (Assistant Tokens Only)
Unlike Stage 1 midtraining (where every token in packed documents had causal loss computed), SFT trains the model strictly on how to respond.

Given a conversation with user query $x_{\text{user}}$ and assistant reply $x_{\text{asst}}$:
$$\mathcal{L}_{\text{SFT}} = - \sum_{t \in \mathcal{M}} \log P(x_t \mid x_{<t}; \theta)$$

Where:
* Token positions $t \notin \mathcal{M}$ (system prompt, user prompt, and role markers) have their target label set to `-100` (ignored by PyTorch `CrossEntropyLoss`).
* Token positions $t \in \mathcal{M}$ (assistant response tokens and final turn separator) have active target labels.

### 3.2 Chat Template & Formatting
* Role markers are plain text tokens (not un-trained special tokens).
* Template format:
  ```text
  <|system|>
  <|user|>
  {user_content}
  <|assistant|>
  {assistant_content}<|end_of_text|>
  ```
* Default system prompt is strictly **empty** to ensure model behaviors are internal and not prompted.
* The identical Jinja template is shared between training and vLLM inference.

### 3.3 Training Hyperparameters (MSM Table 2 & Methodology §8.4)

| Hyperparameter | Value | Description / Source |
|---|---|---|
| **Base Weights** | `midtrain_aw_main_v1_m1` | Merged bf16 checkpoint from Stage 1 |
| **LoRA Rank ($r$)** | 64 | Applied to all linear projections |
| **LoRA Alpha ($\alpha$)** | 128 | $\alpha / r = 2.0$ |
| **Target Modules** | All linear | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Learning Rate** | $1 \times 10^{-4}$ | Cosine decay schedule with 5% linear warmup |
| **Weight Decay** | 0.01 | AdamW optimizer |
| **Effective Batch Size** | 16 conversations | 2 per device $\times$ 8 gradient accumulation steps |
| **Max Sequence Length** | 4,096 tokens | Truncates conversations exceeding 4,096 tokens |
| **Epochs** | 1 complete epoch | Prevents overfitting and catastrophic forgetting |
| **Held-out Split** | 2% deterministic | Validation loss & perplexity tracking |

---

## 4. File and Folder Structure

```text
/mnt/HDD_2TB/niraj/awmsm/
├── configs/
│   ├── chat_template.jinja                  # Standardized chat template for SFT and vLLM
│   ├── train_midtrain.yaml                  # Stage 1 configuration
│   └── train_aft.yaml                       # Stage 2 AFT configuration
│
├── data/
│   ├── aft/
│   │   ├── general_sft_10k.jsonl            # Cleaned, de-contaminated SFT mixture (9,801 samples)
│   │   ├── general_sft_manifest.json        # Provenance, hashes, and filtering stats
│   │   └── cache/                           # Tokenized and masked SFT batches
│   └── raw_datasets/                        # Original source repositories from Hugging Face
│
├── markdown-files/
│   ├── AW_MSM_Methodology.md                # Master scientific reference
│   ├── MSM_training_blueprint.md            # Stage 1 implementation blueprint
│   ├── Stage2_AFT_Data_Preparation_Guide.md # SFT dataset curation guide
│   └── AFT_training_blueprint.md            # This document
│
├── runs/
│   ├── checkpoints/
│   │   ├── midtrain_aw_main_v1_m1/          # Starting checkpoint (Stage 1 merged weights)
│   │   └── aft_a2_aw_main_v1_m1_a1/         # Output checkpoint (Stage 2 merged weights)
│   └── aft/
│       └── aft_a2_aw_main_v1_m1_a1/         # Training logs, loss JSONL, rolling checkpoints
│
├── scripts/
│   ├── AFT-data-gen/
│   │   └── step1p3_Mandatory_Processing_and_Cleaning.py # Completed data curation script
│   ├── run_train.sh                         # Stage 1 execution wrapper
│   └── run_aft.sh                           # Stage 2 execution wrapper with pre-flight checks
│
├── src/awmsm/
│   └── train/
│       ├── aft_dataset.py                   # SFT dataset loader, chat templating, and loss masking
│       ├── train.py                         # Unified training loop (MSM and SFT modes)
│       └── merge.py                         # LoRA adapter merging with 3 verification gates
│
└── tests/
    ├── test_midtrain.py                     # Unit tests for Stage 1
    └── test_aft.py                          # Unit tests for Stage 2 (masking, chat template, tokens)
```

---

## 5. Detailed Breakdown: Responsibilities of Each File

### 5.1 Configuration Files
* [**`configs/chat_template.jinja`**](file:///mnt/HDD_2TB/niraj/awmsm/configs/chat_template.jinja)
  * **Role:** Defines the exact turn boundary markers and role tags for multi-turn dialogues.
  * **Contract:** Must be identical for both SFT training tokenization and subsequent vLLM inference serving. Empty system prompt by default.
* [**`configs/train_aft.yaml`**](file:///mnt/HDD_2TB/niraj/awmsm/configs/train_aft.yaml)
  * **Role:** Centralized configuration specifying:
    * `model.base_checkpoint`: Path to Stage 1 merged checkpoint (`runs/checkpoints/midtrain_aw_main_v1_m1`).
    * `data.sft_path`: Path to `data/aft/general_sft_10k.jsonl`.
    * `training.effective_batch_size`: 16.
    * `training.micro_batch_size`: 2, `gradient_accumulation_steps`: 8.
    * `training.learning_rate`: 1e-4 with cosine decay.
    * `lora.rank`: 64, `lora.alpha`: 128.

### 5.2 Data Pipeline Module
* [**`src/awmsm/train/aft_dataset.py`**](file:///mnt/HDD_2TB/niraj/awmsm/src/awmsm/train/aft_dataset.py)
  * **Role:** Loads JSONL conversation data, applies the chat template, tokenizes, and computes label masks.
  * **Inputs:** `data/aft/general_sft_10k.jsonl`, tokenizer.
  * **Processing:**
    1. Splits dataset deterministically (98% train, 2% validation) using document-level hashing.
    2. Formats messages into prompt vs. response.
    3. Encodes prompt and response separately to locate the exact index boundary.
    4. Sets `labels[i] = -100` for all prompt tokens; sets `labels[i] = input_ids[i]` for assistant tokens.
    5. Batches and dynamically pads to the maximum length of the batch (or sequence length 4,096).
  * **Outputs:** PyTorch Dataset delivering dict with tensors: `input_ids`, `attention_mask`, `labels`.

### 5.3 Training Execution Module
* [**`src/awmsm/train/train.py`**](file:///mnt/HDD_2TB/niraj/awmsm/src/awmsm/train/train.py) (or `train_aft.py`)
  * **Role:** Orchestrates the training run.
  * **Execution:**
    1. Initializes bf16 base model from `runs/checkpoints/midtrain_aw_main_v1_m1/`.
    2. Attaches fresh PEFT LoRA adapter with target configuration.
    3. Runs 1 complete epoch across SFT training batches.
    4. Evaluates held-out validation loss periodically (e.g. every 100 steps) to track loss and perplexity.
    5. Saves adapter weights and training metrics to `runs/aft/aft_a2_aw_main_v1_m1_a1/adapter/`.

### 5.4 Merging & Verification Module
* [**`src/awmsm/train/merge.py`**](file:///mnt/HDD_2TB/niraj/awmsm/src/awmsm/train/merge.py)
  * **Role:** Merges the newly trained AFT LoRA adapter into the Stage 1 base model.
  * **Verification Gates:**
    * **Gate 1 (Weight Delta):** Verifies that merged weights differ from base weights ($\|\Delta W\|_2 > 0$).
    * **Gate 2 (Logit Parity):** Computes forward pass logits on test tokens for both adapter model and merged model, verifying numerical equivalence within bf16 tolerance and 100% top-1 argmax agreement.
    * **Gate 3 (Reload Integrity):** Saves full 16 GB `model.safetensors` to disk and reloads the model to verify format integrity.
  * **Outputs:** Standalone directory at `runs/checkpoints/aft_a2_aw_main_v1_m1_a1/`.

### 5.5 Execution Wrapper Script
* [**`scripts/run_aft.sh`**](file:///mnt/HDD_2TB/niraj/awmsm/scripts/run_aft.sh)
  * **Role:** Automated production runner script.
  * **Features:**
    1. Pre-flight checks: confirms GPU VRAM availability, verifies checkpoint exists, validates config.
    2. Sets `HF_HOME=/mnt/HDD_2TB/niraj/hf` and `CUDA_VISIBLE_DEVICES`.
    3. Executes training and streams logs to `/mnt/HDD_2TB/niraj/logs/aft_a2_aw_main_v1_m1_a1.log`.
    4. Automatically invokes `merge.py` upon successful training completion.
    5. Logs PID and process exit codes.

### 5.6 Automated Testing
* [**`tests/test_aft.py`**](file:///mnt/HDD_2TB/niraj/awmsm/tests/test_aft.py)
  * **Role:** Unit test suite ensuring zero regressions before launching full runs.
  * **Tests:**
    * `test_loss_masking_assistant_only`: Confirms prompt tokens have `-100` and response tokens have valid IDs.
    * `test_chat_template_rendering`: Confirms exact formatting without trailing delimiters or corrupt tags.
    * `test_max_sequence_truncation`: Verifies behavior on conversations exceeding length limits.
    * `test_deterministic_split`: Confirms train/eval splits are disjoint and stable across seeds.

---

## 6. Verification and Acceptance Criteria

Before declaring Stage 2 AFT complete and moving to benchmark evaluations:
1. **Training Loss:** Evaluation loss on held-out general instruction data descends smoothly across the epoch.
2. **Gate 1 Passed:** Non-zero weight norm delta across all 32 transformer layers.
3. **Gate 2 Passed:** Logit difference $\le 0.5$ and 100% argmax agreement between LoRA adapter and merged model.
4. **Gate 3 Passed:** Model reloads cleanly from `runs/checkpoints/aft_a2_aw_main_v1_m1_a1/` and generates fluent text.
5. **No Token Contamination:** Verified empty system prompt and correct role transitions.

---

## 7. Implementation Notes and Refinements

1. **Unified Trainer (`src/awmsm/train/train.py`):**
   * Rather than maintaining two redundant training engines, `train.py` now implements both `--mode midtrain` and `--mode aft`.
   * Model loading, AdamW optimization, cosine learning rate scheduling with 5% warmup, evaluation, rolling checkpoints, and JSONL/CSV logging are completely unified.
   * `train_aft` automatically normalizes loss over the active assistant tokens per gradient accumulation step (`loss = loss_sum / (grad_accum_steps * active_tokens)`).
2. **Dynamic Tensor Padding (`AFTCollator`):**
   * Dynamic batch padding pads to a multiple of 8 (`pad_to_multiple_of=8`) to maximize NVIDIA Tensor Core utilization on Ampere and Blackwell architectures.
   * Padded labels are assigned `-100` so that pad tokens contribute zero gradient.
3. **Merge Dispatcher Enhancement (`src/awmsm/train/merge.py`):**
   * `merge.py` detects runs prefixed with `aft_` and automatically reads adapter weights from `runs/aft/{run_id}/adapter` and the base model from `runs/aft/{run_id}/manifest.json`, eliminating manual path entry.
4. **Chat Template Standardization:**
   * Utilizes literal role markers without adding new special token embeddings (`<|system|>`, `<|user|>`, `<|assistant|>`), terminating turns with `<|end_of_text|>`.
   * Standardized via `configs/chat_template.jinja` across training and inference.
5. **Logit Parity Gate Refinement (Near-Tie Handling):**
   * In `bfloat16` (7-bit mantissa), associative floating-point reordering in dynamic LoRA computation vs. fused weight matrix multiplication causes small variations ($\sim 0.15\text{--}0.25$).
   * If two candidate tokens have logits that tie within sub-ULP margin (e.g. $|\Delta \text{logit}| < 0.1$), their relative ranking can flip.
   * Gate 1 now checks both `max_logit_diff <= logit_tolerance` (0.50) and verifies that any top-1 token index divergence occurs only within the numerical margin of `logit_tolerance`, preventing false alarms on near-ties.

---

## 8. Command Reference for Running Stage 2 AFT

### 8.1 Run Unit Tests
Verify that loss masking, tokenization, dynamic padding, and deterministic splitting pass:
```bash
cd /mnt/HDD_2TB/niraj/awmsm

PYTHONPATH=src /mnt/HDD_2TB/niraj/venvs/vllm/bin/python -m pytest tests/test_aft.py -v
```

### 8.2 Run Quick Smoke Test (50 Steps Override)
To verify end-to-end model forward pass, backward pass, optimizer step, and evaluation without waiting for the full epoch:
```bash
cd /mnt/HDD_2TB/niraj/awmsm

/mnt/HDD_2TB/niraj/venvs/vllm/bin/python -m awmsm.train.train \
  --mode aft \
  --config configs/train_aft.yaml \
  --max-steps 50
```

### 8.3 Run Full Production AFT Training & Merging
Run inside your background screen session (`screen -r emergent_alignment`):
```bash
cd /mnt/HDD_2TB/niraj/awmsm

# Syntax: ./scripts/run_aft.sh [config] [arm] [corpus_id] [seed] [gpu_id] [custom_log_name]
./scripts/run_aft.sh configs/train_aft.yaml a2 aw_main_v1 1 0 aft_a2_aw_main_v1_m1_a1.log
```

* **What this command executes:**
  1. Performs a pre-flight VRAM memory check (requires $\ge 16$ GB free VRAM).
  2. Launches 1 full epoch of AFT training (~601 optimizer steps with effective batch size 16).
  3. Periodically evaluates held-out loss and perplexity on the 2% validation slice.
  4. Saves intermediate checkpoints to `runs/aft/aft_a2_aw_main_v1_m1_a1/checkpoints/`.
  5. Saves the final LoRA adapter to `runs/aft/aft_a2_aw_main_v1_m1_a1/adapter/`.
  6. Automatically merges the adapter into the Stage 1 base weights with 3 verification gates.
  7. Saves the standalone merged checkpoint ready for evaluation to:
     `runs/checkpoints/aft_a2_aw_main_v1_m1_a1/`
  8. Streams output live while logging simultaneously to:
     `/mnt/HDD_2TB/niraj/logs/aft_a2_aw_main_v1_m1_a1.log`

### 8.4 Monitoring Training Progress
In another terminal, track loss descent, learning rate, and throughput:
```bash
tail -f /mnt/HDD_2TB/niraj/logs/aft_a2_aw_main_v1_m1_a1.log
```


