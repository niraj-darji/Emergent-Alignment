# Quantization strategy

How we fit the target models, the judge and fine-tuning onto our two GPUs, and
which precision each model uses. Where this file disagrees with ad-hoc
choices elsewhere, this file wins. Change it only as a team decision, and note
the change in the next report.

**Hardware**

| Machine | GPU | VRAM | Architecture | bf16 | FP8 |
|---------|-----|------|--------------|------|-----|
| A | RTX 4090 | 24 GB | Ada Lovelace (sm_89) | yes | yes |
| B | RTX 5090 | 32 GB | Blackwell (sm_120) | yes | yes |

The laptop GTX 1660 Ti (6 GB, no bf16) is for `--dry-run` and `mock` runs only.

---

## 1. The decision in one table

| Model | Role | Precision for eval | Precision for training | Runs on |
|-------|------|--------------------|------------------------|---------|
| `Qwen/Qwen3-8B` | target | **bf16** (unquantized) | LoRA on bf16 base | A (eval), B (training) |
| `meta-llama/Llama-3.1-8B-Instruct` | target | **bf16** (unquantized) | LoRA on bf16 base | A (eval), B (training) |
| `google/gemma-3-12b-it` | target | **bnb-nf4** (4-bit) | QLoRA (NF4 base) | A or B |
| `openai/gpt-oss-20b` (decided 2026-09-25, §12) | judge + MANTA pressure | **mxfp4** (native) | not trained | B, as a server |
| `Intel/Mistral-Small-3.2-24B-Instruct-2506-int4-AutoRound` (decided 2026-09-25, §13) | writes training data | **int4-autoround** | not trained | A or B |

**Reasoning, briefly:**
- **The 8B targets stay unquantized.** They fit in bf16 on both GPUs for eval and training. Quantizing them would only add noise to the scores we're trying to measure.
- **Gemma-3-12B has to be 4-bit.** Its bf16 weights (~24.4 GB) fill the 4090, and bf16 LoRA training doesn't fit on either GPU. So Gemma is trained with QLoRA, and **every Gemma condition (base, sdf, chat, sdf_chat) is evaluated at the same NF4 precision** the adapters were trained on.
- **The judge is 4-bit and comes from outside the target families** (§12). A larger 4-bit judge grades better than a smaller unquantized one. The 5090 hosts it as a server during ANIMA/MANTA runs. The earlier placeholder, `Qwen3-32B-AWQ`, is dropped because it's from the same family as the Qwen3-8B target.
- **Mixed precision across families is acceptable**, because every claim compares conditions *within* a family (Gemma-sdf vs Gemma-base). We don't compare absolute scores across families, and the report must say which precision each family used.

**Freeze rule:** once the first fine-tuning run of a family starts, its precision is fixed. If an 8B family ever has to switch to QLoRA (e.g. the 5090 is unavailable), then its base model **and all its conditions** are re-evaluated at NF4. Mixing precisions within a family is not allowed.

---

## 2. Memory budget (why the table looks like this)

All numbers are approximate. vLLM needs **weights + ~2 GB overhead** (CUDA
context, activations, CUDA graphs) **+ KV cache** (the model's working memory,
which grows with conversation length and the number of parallel requests).
If a monitor is plugged into the GPU, subtract another ~0.5–1 GB.

### Weights

| Model | bf16 | FP8 | 4-bit (NF4/AWQ) |
|-------|------|-----|-----------------|
| Qwen3-8B | ~16.4 GB | ~8.8 GB | ~5.7 GB |
| Llama-3.1-8B | ~16.1 GB | ~8.5 GB | ~5.7 GB |
| Gemma-3-12B | ~24.4 GB | ~13 GB | ~8 GB |
| Qwen3-32B (dropped placeholder judge) | ~65 GB | ~33 GB | ~19 GB |
| gpt-oss-20b (judge) | n/a | n/a | ~13 GB (ships as MXFP4) |
| Mistral-Small-3.2-24B (generator) | ~48 GB | ~25 GB | ~14–15 GB |

### KV cache per token (bf16)

| Model | Per token | One full 8k-token conversation |
|-------|-----------|--------------------------------|
| Qwen3-8B | ~144 KB | ~1.2 GB |
| Llama-3.1-8B | ~128 KB | ~1.1 GB |
| Gemma-3-12B | ~384 KB | ~3.2 GB |
| Qwen3-32B | ~256 KB | ~2.1 GB |

### What fits where (eval, vLLM)

| Setup | 4090 (≈21.6 GB usable at 0.90) | 5090 (≈29.4 GB usable at 0.92) |
|-------|-------------------------------|--------------------------------|
| Qwen3-8B / Llama-8B bf16 | fits, ~3–4 GB KV | fits comfortably |
| Gemma-3-12B bf16 | **no** | barely: ~3 GB KV, only with `max_model_len: 4096` (used for the calibration check only, §5) |
| Gemma-3-12B nf4 | fits, lots of KV | fits |
| Judge gpt-oss-20b | fits | fits comfortably, 32k context |
| Generator Mistral-Small int4 | fits, ~5 GB KV at 8k context | fits comfortably |
| 8B bf16 target + judge on the same GPU | **no** | **no** (16.4 + ~13–15 GB + 2×2 GB overhead > 29.4 GB) |
| Gemma nf4 target + gpt-oss-20b judge on the same GPU | no | just fits (8 + 13 + 4 ≈ 25 GB, ~4 GB KV). This is the fallback if the machines can't talk to each other. |

The 8B rows are why the judge runs on a different GPU from the target.

### What fits where (LoRA training, gradient checkpointing, seq ≤ 2048)

| Setup | Memory | 4090 | 5090 |
|-------|--------|------|------|
| 8B, LoRA on bf16 | ~20–23 GB | tight: batch 1 only | comfortable |
| 8B, QLoRA | ~10–12 GB | yes | yes |
| Gemma-3-12B, LoRA on bf16 | ~28–32 GB+ | no | no (weights alone 24.4 GB) |
| Gemma-3-12B, QLoRA | ~14–18 GB | yes | yes |

Gemma's 262k-token vocabulary makes the logits large. If QLoRA runs out of
memory, drop the sequence length before the batch size, or use a trainer option
that computes the loss in chunks.

---

## 3. Two-machine layout

### During ANIMA / MANTA runs (need the judge)

```
Machine B (5090, 32 GB)                 Machine A (4090, 24 GB)
┌──────────────────────────────┐        ┌────────────────────────────────────────┐
│ vllm serve <judge> (§12)     │◄──────►│ python -m benchmarks.run ...           │
│ judge + pressure, port 8001  │  SSH   │ starts the target (8B bf16 / Gemma nf4)│
│ one model for both roles     │ tunnel │ calls the judge via base_url           │
└──────────────────────────────┘        └────────────────────────────────────────┘
```

**On B, start the judge** once and leave it running for the whole study:

```bash
vllm serve openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768        # see §11.3 for why 32k
```

**On A, open a tunnel to it.** This keeps the server off the network, so no API key is needed:

```bash
ssh -N -L 8001:localhost:8001 <user>@<machine-B>
curl http://localhost:8001/v1/models      # should list openai/gpt-oss-20b
```

**Then run the targets on A** (config entries in §4):

```bash
python -m benchmarks.run --models qwen3-8b llama3.1-8b gemma3-12b-nf4 \
  --benchmarks animal_welfare --grader judge-remote --pressure judge-remote \
  --run-name base-v1
```

If the machines can't reach each other, Gemma nf4 and the gpt-oss-20b judge
fit together on B (§2). The 8B bf16 targets don't. Running them next to the
judge would mean switching them to nf4, which changes their precision, so treat
that as a last resort and log it as a deviation.

### During capability runs (MMLU / HellaSwag; no judge)

These don't need the judge, so both GPUs can run targets in parallel. Stop the
judge server on B first.

```bash
# on A
python -m benchmarks.run --models qwen3-8b llama3.1-8b --benchmarks capability --run-name base-v1
# on B
python -m benchmarks.run --models gemma3-12b-nf4 gemma3-12b-bf16-calib --benchmarks capability --run-name base-v1
```

Copy B's `logs/base-v1/` into A's before running `benchmarks.summarize`, so
one summary covers everything.

### During fine-tuning

| GPU | Job |
|-----|-----|
| B (5090) | 8B bf16 LoRA runs (Qwen3, Llama) |
| A (4090) | Gemma-3-12B QLoRA runs, and 8B runs at batch 1 if B is busy |

The judge server and training can't share B. Plan judge-dependent evals for
when B is free.

---

## 4. Config changes to make in `configs/models.yaml`

```yaml
# A: headless 4090 — use more of the GPU than the 0.85 default
vllm_defaults:
  model_args:
    gpu_memory_utilization: 0.90     # use 0.85 if a monitor runs on this GPU
    max_model_len: 16384             # see §11.3; the same value for every target

models:
  # 8B targets: unchanged, bf16
  qwen3-8b:     {...}   # precision: bf16
  llama3.1-8b:  {...}   # precision: bf16

  # Gemma at the study precision (NF4, same as QLoRA training)
  gemma3-12b-nf4:
    extends: gemma3-12b
    model_args: {quantization: bitsandbytes}
    metadata: {precision: bnb-nf4}

  # Gemma bf16: only for the calibration check in §5, on the 5090
  gemma3-12b-bf16-calib:
    extends: gemma3-12b
    model_args: {gpu_memory_utilization: 0.92, max_model_len: 4096}
    metadata: {precision: bf16, condition: base-calib}

  # Judge served on machine B, reached through the SSH tunnel.
  # Already in configs/models.yaml: judge-gptoss-20b (starts its own server),
  # judge-remote (connects to B), generator-mistral-small. See §12.3 and §13.2.

  # Fine-tuned variants follow their family's precision:
  # qwen3-8b-sdf:     extends: qwen3-8b,       model: vllm/Qwen/Qwen3-8B:checkpoints/qwen3-8b/sdf
  # gemma3-12b-nf4-sdf:
  #   extends: gemma3-12b-nf4
  #   model: vllm/google/gemma-3-12b-it:checkpoints/gemma3-12b/sdf
  #   metadata: {condition: sdf}
```

`configs/eval.yaml` already sets `roles.grader` and `roles.pressure` to
`judge-remote`, and `max_model_len` in `models.yaml` is already 16384 (§11).
The Gemma nf4 and calibration entries above are not added yet.

Remove the plain `gemma3-12b` (bf16) entry from base runs. It doesn't fit on
the 4090 and isn't a study condition.

---

## 5. Calibration check: how much does 4-bit move Gemma's scores?

CLAUDE.md requires one quantized-vs-full-precision check. Gemma is the only
quantized target, so the check is:

| Run | Machine | Benchmarks |
|-----|---------|------------|
| `gemma3-12b-bf16-calib` | B (5090), judge stopped | capability suite |
| `gemma3-12b-nf4` | A or B | capability suite (same run name) |

Report the per-benchmark gap in the mid-term report. For MMLU and HellaSwag,
more than ~1–2 points is worth discussing. This tells readers how much of
Gemma's absolute scores come from quantization. It does **not** affect the
within-Gemma comparisons, because every Gemma condition is NF4.

Optional, if time allows: the same check for Qwen3-8B, to show the size of the
effect on a second family.

---

## 6. Training settings

**8B models (bf16 LoRA):**
```python
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16)
model.gradient_checkpointing_enable()
# peft LoraConfig as decided in the training plan; batch 1–4 with gradient accumulation
```

**Gemma-3-12B (QLoRA):**
```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-12b-it", quantization_config=bnb)
model = prepare_model_for_kbit_training(model)   # enables gradient checkpointing
```

Keep the LoRA rank, target modules, learning rate, data volume and seeds the
same across the three conditions of a family. Save the `BitsAndBytesConfig`
next to each adapter.

**Check that training and serving match.** After the first Gemma adapter is
trained, generate greedy answers (`temperature=0`) to ~10 prompts twice:
1. In transformers + peft with the NF4 base.
2. In vLLM with `quantization: bitsandbytes` + the adapter.

The answers should be identical or nearly so. If they differ a lot, vLLM is
quantizing differently from training. Stop and fix that before running any
evals.

---

## 7. Setup on both machines

- **Identical packages.** Pin `vllm` in `requirements.txt` (currently unpinned; the laptop has `0.22.1`), and add `bitsandbytes`. vLLM or bitsandbytes updates can shift scores, so both machines must run the same versions for the whole study.
- **5090 (Blackwell) needs recent software:** an NVIDIA driver 570+ and CUDA 12.8+ builds of PyTorch, vLLM and bitsandbytes. Before real runs, do one short smoke test of each precision on B (bf16, NF4, AWQ, NF4 + LoRA):
  ```bash
  python -m benchmarks.run --models gemma3-12b-nf4 --benchmarks mmlu --limit 5 --run-name smoke-b
  ```
- **Shared HF cache per machine** with ~100 GB free: `export HF_HOME=/path/with/space/hf`.
- **Gated models:** accept the Gemma and Llama licenses, and run `hf auth login` on both machines.
- **Headless if possible.** A desktop session on the GPU uses 0.5–1 GB. Check with `nvidia-smi` before a run.

---

## 8. What to record

Every eval log and CSV already carries `metadata.precision`. Use exactly these labels:

| Label | Meaning |
|-------|---------|
| `bf16` | unquantized |
| `bnb-nf4` | bitsandbytes 4-bit NF4 (vLLM `quantization: bitsandbytes`, or QLoRA) |
| `awq-4bit` | pre-quantized AWQ checkpoint |
| `mxfp4` | gpt-oss's native 4-bit format, as released by OpenAI |
| `int4-autoround` | Intel's 4-bit AutoRound checkpoint (the generator) |

For the report, record the vLLM, bitsandbytes, torch and driver versions, and
the GPU each run used.

---

## 9. If something runs out of memory

Try these in order, and don't change precision:
1. Lower `--max-samples` (fewer conversations in parallel).
2. Lower `max_model_len` for benchmarks with short prompts. Capability benchmarks are fine at 4096; MANTA is not (see §10).
3. Add `enforce_eager: true` to `model_args`. This disables CUDA graphs, saving ~0.5–1 GB at some cost in speed.
4. Check that nothing else is on the GPU (`nvidia-smi`).

Changing a family's precision means re-running all its conditions (§1 freeze rule).

---

## 10. Open issues

- **MANTA conversation length:** now addressed in §11.
- **Judge family:** now addressed in §12. Qwen3-32B-AWQ is dropped.
- **Data-generation model:** now addressed in §13.

---

## 11. Length settings: `max_model_len` and friends

### 11.1 Three different settings (often confused)

`max_model_len` is **not** the length of the model's response. It's the
**total** number of tokens one request can hold: the whole prompt (every
earlier turn of the conversation) **plus** the response being generated. It's
a vLLM *server* setting, and it's used only when running a model (eval and
data generation), never during training.

| Setting | Where it lives | What it limits | If exceeded |
|---------|----------------|----------------|-------------|
| `max_tokens` | `generate_config` (models.yaml / eval.yaml) | Length of **one response** (output only). Currently 2048. | The response is cut off mid-sentence (`stop_reason: max_tokens`). |
| `max_model_len` | vLLM `model_args` | **Prompt + response** of one request. Also sets how much KV-cache memory vLLM must reserve. | vLLM rejects the request. The sample errors out or ends early. |
| `max_length` (TRL `SFTConfig`; older versions call it `max_seq_length`) | training script | Length of **one training example** (a whole document, or a whole chat). Drives training memory. | The example is truncated (or dropped, if we filter first). |

A bigger `max_model_len` doesn't change any output. It only raises the
ceiling and costs memory. A bigger `max_tokens` *does* change outputs, because
long answers stop being cut off.

### 11.2 What each one depends on

| Setting | Depends on | Needs the training data? |
|---------|-----------|--------------------------|
| Targets' `max_model_len` (eval) | The longest **benchmark** conversation: MANTA's 5 turns | **No.** Measure it now on the base models. |
| Judge's `max_model_len` | Longest benchmark conversation + judge rubric + judge's answer | **No** |
| Generator's `max_model_len` | Generation prompt (constitution + instructions) + requested document/chat length | **No.** We choose these lengths ourselves. |
| Training `max_length` | Longest training document/chat, and the GPU memory | **Not really.** We *design* the data to fit a chosen length (§11.5), rather than measuring the data afterwards. |

So nothing blocks us now. The training data doesn't decide `max_model_len`:
benchmarks decide the eval setting, and *we* decide the training length.

### 11.3 Eval: set it from the benchmarks now

Worst case per benchmark, with `max_tokens` = 2048:

| Benchmark | Prompt | Response | Worst case |
|-----------|--------|----------|-----------|
| MMLU / HellaSwag | < 1k | ≤ 2048 | ~3k |
| BBQ / HHH | < 1k | ≤ 2048 | ~3k |
| Sycophancy (2 turns) | < 1k + first reply | ≤ 2048 | ~5k |
| ANIMA | < 0.5k | ≤ 2048 | ~2.5k |
| **MANTA** | 5 user turns (~100–300 tokens each) + 4 earlier replies | 5th reply ≤ 2048 | **5 × 2048 + ~1.5k ≈ 11.7k** |

The old 8192 is too small for MANTA in the worst case, so:

- **Targets: `max_model_len: 16384`,** the same for every target, condition and benchmark. §2 shows it fits: Qwen3-8B bf16 on the 4090 has ~22k tokens of KV at 0.90, Llama ~25k, Gemma nf4 ~29k.
- **Judge: `max_model_len: 32768`.** It reads the whole MANTA conversation (~12k worst case) plus the rubric, and gpt-oss writes reasoning tokens before its score (§12). gpt-oss has a small KV cache, so this is cheap.
- **Keep `max_tokens: 2048`,** unchanged across all conditions.

**Confirm with a pilot** (base models only, no training data needed):

```bash
python -m benchmarks.run --models qwen3-8b llama3.1-8b gemma3-12b-nf4 \
  --benchmarks manta --limit 30 --run-name pilot-len
```

Then measure the longest conversation and count the cut-off replies:

```python
from inspect_ai.log import ModelEvent, list_eval_logs, read_eval_log
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")   # repeat per target
for info in list_eval_logs("logs/pilot-len/qwen3-8b", recursive=True):
    log = read_eval_log(info)
    lengths = [len(tok.apply_chat_template([{"role": m.role, "content": m.text} for m in s.messages]))
               for s in log.samples]
    cut = sum(1 for s in log.samples for e in s.events
              if isinstance(e, ModelEvent) and e.model == log.eval.model
              and e.output.stop_reason == "max_tokens")
    print(f"max {max(lengths)}  p95 {sorted(lengths)[int(.95 * len(lengths))]}  truncated replies {cut}")
```

**How to decide:**
- Longest conversation well under 16384 (e.g. < 13k): keep 16384. Done.
- Anything close to 16384: raise it to 24576 and re-check memory in §2.
- Many truncated replies (> ~2%): keep `max_tokens` as is, but report the truncation rate per condition. Fine-tuning can make a model more verbose, and a cut-off answer gets judged worse. So **re-run this check after the first fine-tuned model** too.

### 11.4 Generator: set from what we ask it to write

Generation prompt ≈ constitution (2–5 pages ≈ 1.5–4k tokens) + instructions
and seed topic (~0.5k) + output (we ask for ≤ ~1.5k tokens, with
`max_tokens: 2048`). That's ≈ 6.5k worst case, so **`max_model_len: 8192`**.
If the constitution grows past ~5 pages, raise it to 16384.

### 11.5 Training: choose the length, then make the data fit

This is the reverse of eval. We don't measure the data and then set the
length. We pick a length that fits our GPUs and tell the generator to write
within it.

1. **Budget:** `max_length: 2048` tokens per example. Memory in §2 assumes this, and Gemma's QLoRA is sensitive to it.
2. **Put the target lengths in the generation prompts:**
   - documents: 300–1000 words (≈ 400–1400 tokens)
   - chats: 2–6 turns, ≤ ~1500 tokens in total
3. **After generation, tokenize every example with all three target tokenizers** (Qwen3, Llama 3.1, Gemma 3). The same text has different token counts in each; Gemma's large vocabulary gives fewer tokens. **Drop** (don't truncate) any example over 2048 in *any* tokenizer. Every family then trains on exactly the same examples, and no chat loses its ending.
4. Record the length distribution (mean / p95 / max) in the dataset card and the report.

Training on ≤ 2048-token examples doesn't shrink the model's usable context.
Qwen3, Llama 3.1 and Gemma 3 keep their long pretraining context, so evaluating
at 16384 is fine.

---

## 12. Judge: from outside the target families

### 12.1 Requirements

| Requirement | Why |
|-------------|-----|
| Not Qwen, Llama or Gemma, **including fine-tunes of them** (e.g. DeepSeek-R1-Distill-Qwen/Llama are Qwen/Llama underneath) | LLM judges favour text from their own model family ("self-preference bias": Panickssery, Bowman & Feng, 2024, *LLM Evaluators Recognize and Favor Their Own Generations*). |
| Open weights, runs locally | Proposal constraint: no paid APIs |
| Fits the 5090 with a 32k context | §11.3 |
| At least as strong as the targets at reasoning and following a rubric | A weak judge adds noise that can hide real effects |
| Can also write MANTA's pushback turns | The same model fills the `pressure` role |

### 12.2 Decision: `openai/gpt-oss-20b` (2026-09-25)

| Model | Family | Size / format | VRAM | Context | Role |
|-------|--------|---------------|------|---------|------|
| **`openai/gpt-oss-20b`** | OpenAI | 21B MoE (only 3.6B active per token), official MXFP4 | ~13 GB | 128k | **Judge + MANTA pressure** |
| Mistral-Small-3.2-24B-Instruct-2506 | Mistral | 24B dense | ~14–15 GB (int4) | 128k | Generator (§13), and fallback for the `pressure` role only (§12.3) |
| Phi-4 (14B) | Microsoft | FP8 | ~15 GB | **16k** | Rejected: its 16k context is too short for a ~12k conversation plus the rubric. |
| Qwen3-32B-AWQ (old placeholder) | Qwen | | | | Rejected: same family as Qwen3-8B. |

Why gpt-oss-20b:
- It's from outside all target families.
- It can reason before giving a score, which helps with rubric grading.
- It's fast, since only 3.6B parameters are active per token.
- It fits on the 5090 with lots of room, and its license is Apache 2.0.

We chose it without first comparing two judges against human scores (the
earlier plan). That comparison would take time we don't have. The lighter
checks in §12.4 replace it.

A caveat for the report: nothing that fits in 32 GB is dramatically stronger
than a 12B target. State the judge's limits, and back it up with §12.4.

### 12.3 Config (already in `configs/models.yaml`)

**Decision: gpt-oss-20b for judge + pushback, Mistral-Small for data generation.**

In MANTA, gpt-oss has two jobs, set in `configs/eval.yaml`:

| Role | Job | Does gpt-oss's safety training matter? |
|------|-----|-----------------------------------------|
| `grader` (judge) | Reads an answer and gives it a score from 0 to 1 (ANIMA and MANTA) | No. Scoring is harmless. |
| `pressure` (pushback writer) | Plays the user in MANTA turns 2–5 and argues *against* the welfare advice | Only slightly. MANTA's prompt asks for mild, realistic pushback ("that's a real cost hit for a tight-margin event…") framed as research-dataset creation, so gpt-oss is expected to comply. Confirm this in the smoke test below. |

| Alias | Use |
|-------|-----|
| `judge-remote` | Default in `eval.yaml`. Connects to the server on B through the SSH tunnel (§3). |
| `judge-gptoss-20b` | Starts its own server (`gpu_memory_utilization: 0.45`), for single-machine runs such as Gemma nf4 + judge on the 5090. |

Both use `temperature: 1.0` with `seed: 1234`, `reasoning_effort: medium` and
`max_tokens: 8192` (reasoning tokens count toward this), and the pinned revision
`6cee5e81ee83917806bbde320786a8fb61efebee`. The judge server on B is started with:

```bash
vllm serve openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --host 127.0.0.1 --port 8001 --gpu-memory-utilization 0.92 --max-model-len 32768
```

**Smoke test before any real run (~15 minutes):**
```bash
python -m benchmarks.run --models qwen3-8b --benchmarks anima manta --limit 5 --run-name smoke-judge
inspect view --log-dir logs/smoke-judge
```
- `judge_parse_failures` must be 0 or close to it. gpt-oss puts its reasoning in a separate field, so the `SCORE:` line should parse.
- **Read the MANTA pressure turns (turns 3–5).** They must actually argue *against* the welfare position ("that's too expensive", "everyone does it"). gpt-oss has strong safety tuning and might write weak or refusing pushback. If it does, switch only the `pressure` role to Mistral-Small (add a `pressure-mistral-small` entry based on `generator-mistral-small` with `temperature: 0.0`) and keep gpt-oss as `grader`. Don't swap the whole setup (Mistral judge, gpt-oss generator): gpt-oss is the better judge because it reasons before scoring.
  - **A catch with Mistral as the pushback writer:** Mistral also writes our training chats, including the *user* turns. MANTA pushback written by Mistral would then sound like the users our fine-tuned models saw in training, which could make them look slightly more stable than the base models. So use Mistral for `pressure` only if gpt-oss's pushback really is weak, and state this as a limitation in the report.
- **Applied 2026-09-26:** in the first smoke run at `temperature: 0` / `max_tokens: 4096`, gpt-oss twice spent all 4,096 tokens reasoning and returned an empty answer (one ANIMA grade, one MANTA pushback turn). The judge now uses its recommended `temperature: 1.0` with a fixed `seed`, and `max_tokens: 8192`. The MANTA pushback writer uses the same settings (upstream used temperature 0 with Claude) and retries an empty reply with a new seed, up to 3 attempts; retries are counted in each sample's `pressure_retries`. ANIMA runs through `benchmarks/anima.py`, which fixes an `inspect_evals` crash when one dimension gets no usable grade. Report these settings in the mid-term report.

### 12.4 Lightweight judge checks (instead of the 40-response study)

**Before the base run (~20 minutes, no statistics):** in `inspect view` of the smoke run, read 10 judge scores with their explanations. For each one, ask "would I roughly agree with this score?" If several are clearly wrong (a caring answer scored low, a dismissive one scored high), stop and discuss before running everything.

**Optional, later (e.g. before the final report):** a small human-agreement check on gpt-oss only. This is half the work of the original plan, since there's one judge instead of two. Every judge score is already saved in the logs, so nothing needs re-running. You and Niraj each score ~20–30 responses blind, then compute the Spearman correlation with the judge. If there's no time, list "judge not validated against human ratings" as a limitation.

Once the base run starts, **the judge (model, revision, precision, reasoning
effort, temperature) is frozen for the whole study.**

---

## 13. Data generator: not the judge, and not a target family

### 13.1 Can the generator come from a target family? Allowed, but it biases the results

| Setup | What goes wrong |
|-------|-----------------|
| **Generator = judge** | The judge grades models trained to write like the judge, so its self-preference raises the fine-tuned models' scores. The effect looks bigger than it is. |
| **Generator from a target family** (e.g. Qwen3-32B writing data for Qwen3-8B) | 1. **Easier learning for that family:** the data is in its own style and token distribution, so it learns faster and deeper. 2. **Hidden trait transfer:** training data generated by a model can pass on that model's traits even when the text never mentions them, mainly when teacher and student share a base model (Cloud et al., 2025, *Subliminal Learning*). That family's gains could then come partly from the generator's hidden traits, not from the constitution. The other families don't get this effect, so **gains can't be compared across families.** |
| **Generator from a 5th family** (neither a target nor the judge) | All three targets get "foreign" data equally. The comparison is symmetric, and gains can be attributed to the content. |

### 13.2 Decision: Mistral-Small-3.2-24B-Instruct-2506, 4-bit (2026-09-25)

| | Model | Family |
|--|-------|--------|
| Judge | gpt-oss-20b | OpenAI |
| **Generator** | **Mistral-Small-3.2-24B-Instruct-2506** | **Mistral** |
| Targets | Qwen3-8B, Llama-3.1-8B, Gemma-3-12B | Qwen, Meta, Google |

The generator, the judge and the three targets come from five different
families. That removes both biases in §13.1.

- **Checkpoint:** `Intel/Mistral-Small-3.2-24B-Instruct-2506-int4-AutoRound`, pinned to revision `21907bf9a442ec05c91e059ef17b08a8b44983f5`. Intel quantized only the language layers to 4-bit. It works on both GPUs; RedHat's NVFP4 version runs natively only on the 5090, and the FP8 version (~25 GB) leaves too little memory for generation. The model is multimodal, so `language_model_only: true` turns off image inputs.
- **Precision:** 4-bit is fine for the generator. Every target and condition trains on the *same* data, so the generator's precision doesn't bias any comparison; it only affects data quality, which we review in the pilot. Record it in the dataset card.
- **Where:** either GPU (~14–15 GB). Data generation (due Sep 28) comes first. Base-model evals slot in around it. The 5090 can't run the judge and the generator at the same time with a useful amount of KV cache (13 + 15 + overhead ≈ 32 GB), so schedule them one after the other.
- **Settings:** `max_model_len: 8192` (§11.4). `temperature: 0.8` for variety (not 0), with a fixed seed per batch.
- **Contamination:** the generator never sees ANIMA or MANTA questions (CLAUDE.md §3).
- **Style note:** the `chat` models will partly learn Mistral's writing style. That's the same for all three families, so it doesn't bias comparisons.

Config (already in `configs/models.yaml`):

```yaml
generator-mistral-small:
  model: vllm/Intel/Mistral-Small-3.2-24B-Instruct-2506-int4-AutoRound
  model_args:
    revision: 21907bf9a442ec05c91e059ef17b08a8b44983f5
    language_model_only: true
    max_model_len: 8192
  generate_config: {temperature: 0.8, max_tokens: 2048}
  metadata: {role: generator, family: mistral, precision: int4-autoround}
```

Before scaling up generation, generate 5 documents and 5 chats on the GPU
machine and read them. This confirms the int4 model loads and writes coherent
text.

---

## 14. TODO: benchmark the base models (before any fine-tuning)

Goal: a complete `base-v1` run of every model we'll later fine-tune,
unmodified, at the precision the study uses. Its scores are the reference
point for every later comparison. It doesn't need any training data, so it can
run while the data-generation policy is being written.

**Models in `base-v1`:**

| Alias | Precision | Benchmarks | Machine |
|-------|-----------|------------|---------|
| `qwen3-8b` | bf16 | all | A (4090) |
| `llama3.1-8b` | bf16 | all | A (4090) |
| `gemma3-12b-nf4` | bnb-nf4 | all | A (4090) |
| `gemma3-12b-bf16-calib` | bf16 | capability only (§5) | B (5090), judge stopped |
| `qwen3-8b-nf4-calib` (optional) | bnb-nf4 | capability only | A or B |

Suggested owners follow the proposal: **J** = Jevin (animal welfare + human
safety), **N** = Niraj (capability). *Both* = do it together.

### Phase 0: configs and code (laptop, no GPU)

- [x] **(Both) Decide on the human-safety benchmarks.** Decided 2026-09-25: BBQ, HHH, Sycophancy. GSM8K and ToxiGen are dropped.
- [x] Add them to `configs/eval.yaml` (`human_safety` suite). HHH is ours (`benchmarks/hhh.py`), BBQ has a wrapper (`benchmarks/bbq.py`), and sycophancy is graded by the judge. `python -m benchmarks.selftest` checks all three.
- [x] (N) Add to `configs/models.yaml` (§4):
  - `gemma3-12b-nf4` (`quantization: bitsandbytes`, `language_model_only: true`, `precision: bnb-nf4`)
  - `gemma3-12b-bf16-calib` (`max_model_len: 4096`, `gpu_memory_utilization: 0.92`, `condition: base-calib`)
  - optionally `qwen3-8b-nf4-calib`
- [x] (N) Pin `vllm` in `requirements.txt` (0.22.1) and add `bitsandbytes`. The three models being tested are also pinned to their downloaded revisions in `models.yaml`.
- [x] (N) `gpu_memory_utilization: 0.90` in `vllm_defaults` (§4). At 0.85, an 8B bf16 model on the 4090 may not fit a 16k-token conversation.
- [ ] (Both) Check the plan without loading anything:
  ```bash
  python -m benchmarks.run --models qwen3-8b llama3.1-8b gemma3-12b-nf4 --dry-run
  python -m benchmarks.selftest
  python -m benchmarks.run --models mock --grader mock --pressure mock --limit 3 --run-name smoke
  ```
- [ ] Commit the configs. The `base-v1` run should come from a committed state, so it can be reproduced.

### Phase 1: set up both GPU machines

- [ ] NVIDIA driver 570+ and CUDA 12.8+ on the 5090 (§7). Check with `nvidia-smi`.
- [ ] Clone the repo, then `uv venv --python 3.11 && uv pip install -r requirements.txt` on both. The package versions must match.
- [ ] Set `HF_HOME` to a disk with ≥ 100 GB free.
- [ ] Accept the Gemma and Llama licenses on Hugging Face, and run `hf auth login` on both machines.
- [ ] Download the weights: on A, the three targets; on B, `openai/gpt-oss-20b` and `google/gemma-3-12b-it` (for the calibration run).
  ```bash
  hf download Qwen/Qwen3-8B
  hf download meta-llama/Llama-3.1-8B-Instruct
  hf download google/gemma-3-12b-it
  hf download openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee
  ```
- [ ] Check that A can reach B: `ssh <user>@<machine-B>` works from A.

### Phase 2: smoke tests (small, fast, catch problems early)

- [ ] **On B:** start the judge server (§3) and leave it running.
- [ ] **On A:** open the SSH tunnel, then `curl http://localhost:8001/v1/models` should list `openai/gpt-oss-20b`.
- [ ] (N) Each target loads and answers, capability only, 5 samples:
  ```bash
  python -m benchmarks.run --models qwen3-8b llama3.1-8b gemma3-12b-nf4 \
    --benchmarks capability --limit 5 --run-name smoke-targets
  ```
- [ ] (J) Judge + pushback smoke test (§12.3):
  ```bash
  python -m benchmarks.run --models qwen3-8b --benchmarks animal_welfare --limit 5 --run-name smoke-judge
  inspect view --log-dir logs/smoke-judge
  ```
  - [ ] `judge_parse_failures` is 0 or close to it.
  - [ ] MANTA turns 3–5 actually push back (§12.3). If not, apply the pressure-role fallback.
  - [ ] Read 10 judge scores and their explanations. Do they roughly make sense (§12.4)?
- [ ] (J) Human-safety smoke test, then read a few sycophancy judge decisions ("did it back down?") in `inspect view`:
  ```bash
  python -m benchmarks.run --models qwen3-8b --benchmarks human_safety --limit 5 --run-name smoke-safety
  ```

### Phase 3: length pilot (§11.3)

- [ ] (J) Run MANTA on 30 samples for all three targets:
  ```bash
  python -m benchmarks.run --models qwen3-8b llama3.1-8b gemma3-12b-nf4 \
    --benchmarks manta --limit 30 --run-name pilot-len
  ```
- [ ] Measure the longest conversation and count the truncated replies with the script in §11.3.
- [ ] Keep `max_model_len: 16384`, or raise it per the §11.3 rule.

### Phase 4: freeze settings

Nothing below may change after `base-v1` starts. The fine-tuned runs must use the same values.

- [ ] (Both) Write down and commit:
  - `seed` and each benchmark's `limit` and `epochs` (`eval.yaml`)
  - `max_model_len` and `max_tokens`
  - the judge: model, revision, `reasoning_effort`, temperature
  - the pushback writer (gpt-oss, or the Mistral fallback)
  - the versions of `inspect-ai`, `inspect_evals`, `vllm`, `bitsandbytes`, `torch`, and the NVIDIA driver on each machine
- [ ] Tag the commit, e.g. `git tag eval-freeze-v1`.

### Phase 5: the base run

Shortcut: `bash scripts/run_base.sh base-v1` on A runs every benchmark for the three models (the judge from `scripts/start_judge.sh` must be reachable). See `benchmarks/README.md`.

**Animal welfare (needs the judge; A runs the targets, B serves the judge):**
- [ ] (J)
  ```bash
  python -m benchmarks.run --models qwen3-8b llama3.1-8b gemma3-12b-nf4 \
    --benchmarks animal_welfare --run-name base-v1
  ```

**Human safety (BBQ, HHH, Sycophancy; the judge must be running for Sycophancy):**
- [ ] (J) Same command with `--benchmarks human_safety`. Sycophancy needs the judge; BBQ and HHH don't.

**Capability (no judge; stop the judge server on B first):**
- [ ] (N) On A:
  ```bash
  python -m benchmarks.run --models qwen3-8b llama3.1-8b gemma3-12b-nf4 \
    --benchmarks capability --run-name base-v1
  ```
- [ ] (N) On B, the calibration run (§5):
  ```bash
  python -m benchmarks.run --models gemma3-12b-bf16-calib --benchmarks capability --run-name base-v1
  ```
- [ ] (Optional) `qwen3-8b-nf4-calib` on the capability suite, to show the quantization effect on a second family.

If a run crashes partway through, rerun the same command with the same
`--run-name`. `summarize` uses the newest successful log for each model ×
benchmark.

### Phase 6: collect and check

- [ ] Copy `logs/base-v1/` from B to A, so all logs are in one place.
- [ ] `python -m benchmarks.summarize --run-name base-v1` should produce `results/base-v1/summary.csv` and `scores_long.csv`.
- [ ] Sanity-check the numbers:
  - MMLU and HellaSwag roughly match each model's published scores (within a few points, since we use subsamples and our own prompts).
  - No benchmark has many errored samples (`inspect view`).
  - `judge_parse_failures` ≈ 0 for ANIMA and MANTA.
- [ ] Record the Gemma bf16 vs nf4 gap per capability benchmark (§5).
- [ ] Commit `results/base-v1/` (the CSVs only; `logs/` stays out of git).

### Phase 7: write it up (mid-term report, due Oct 16)

- [ ] (Both) A baseline table: 3 models × all benchmarks, with precision labelled.
- [ ] State the model changes from the proposal:
  - Gemma 2 → Gemma 3, Qwen 2.5 → Qwen3, and the new Llama family
  - ANIMA instead of AnimalHarmBench
  - GSM8K and ToxiGen dropped; HHH is asked as a multiple-choice question with shuffled order, not scored by log-likelihood
  - the gpt-oss judge and Mistral generator
  - Gemma evaluated at 4-bit, with the calibration gap
- [ ] State the limitations:
  - the judge isn't checked against human ratings (unless §12.4's optional check was done)
  - our MANTA numbers can't be compared with the paper's (different judge and pushback writer)
  - ANIMA runs with 1 epoch instead of 5
