**Date:** 2026-11-23
**Time:** 23:23 IST

# Prompt-format mismatch in the welfare evaluations: what it is and how to fix it

## 1. The problem in one sentence

The models were fine-tuned to read conversations in one format, but the ANIMA and MANTA evaluations asked them questions in a different format, one they never saw during training.

## 2. The two formats

**What the model learned during fine-tuning** (`configs/chat_template.jinja`, used by `src/awmsm/train/aft_dataset.py`):

```
<|user|>
{question}
<|assistant|>
{reply}<|end_of_text|>
```

- `<|user|>` and `<|assistant|>` are plain text. The tokenizer splits each into 5 ordinary tokens (`<`, `|`, `user`, `|`, `>\n`).
- Sequences start without a BOS token (`add_special_tokens=False`).

**What the welfare evaluation sent to the model** (`format_chat_prompt()` in two scripts):

```
<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

This is the Llama-3-**Instruct** chat format. The differences:

| | Training | Welfare evaluation |
|---|---|---|
| Role markers | `<|user|>` / `<|assistant|>` (plain text) | `<|start_header_id|>…<|end_header_id|>` (special tokens) |
| End of user turn | newline | `<|eot_id|>` |
| BOS token | none | added (default tokenizer call) |

Our model is the **base** Llama-3.1-8B. Its header tokens (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`) were never trained, neither by Meta (the base model wasn't instruction-tuned) nor by us. The methodology (§8.4, §9.0) explicitly required evaluation to use the training template.

## 3. Where it happens

| File | Line | What |
|---|---|---|
| `src/awmsm/eval/generate_welfare_rollouts.py` | 82-84 | `format_chat_prompt()` builds the Instruct-format prompt. **These rollouts are what the Gemma judge scored**, so all reported ANIMA/MANTA numbers are affected. |
| `src/awmsm/eval/generate_welfare_rollouts.py` | 115 | `tokenizer(batch_prompts, ...)` adds a BOS token (training had none). |
| `src/awmsm/eval/bench_welfare.py` | 70-72, 109 | Same function and tokenizer call (keyword scorer). |

Not affected: MMLU, HellaSwag, BBQ, HHH, ETHICS and Anthropic HH. They are scored by log-likelihood on plain text and use no chat template.

## 4. What it does to the results

- **It affects every arm in the same way.** All arms got identical prompts, so comparisons between arms (MSM vs none, effect of p, v1 vs v2) remain fair as relative comparisons.
- **Absolute welfare scores are probably depressed.** The model receives an unfamiliar context, so it may not fully behave as the "Aster" assistant it was trained to be. This is one plausible reason why even the best arm passes only ~25% of ANIMA questions.
- **It is not guaranteed to be perfectly uniform.** Arms trained with more updates or more chat data might cope with the unfamiliar format differently. This is untested.
- The paper reports it under *Deviations from the planned protocol* (Experimental Setup) and *Threats to validity* (Results).

## 5. How to fix it

### Step 1: use the training template in both scripts

Replace `format_chat_prompt()` in **both** files with a version that renders the same Jinja file the trainer uses. Train and eval can then never drift apart again:

```python
from pathlib import Path
from jinja2 import Template

_TEMPLATE = Template((Path(__file__).resolve().parents[3] / "configs/chat_template.jinja").read_text())

def format_chat_prompt(question: str) -> str:
    """Single-turn prompt in exactly the format used for fine-tuning."""
    return _TEMPLATE.render(messages=[{"role": "user", "content": question}],
                            add_generation_prompt=True)
    # -> "<|user|>\n{question}\n<|assistant|>\n"
```

### Step 2: match the tokenization

Training sequences had no BOS token. In the generation call (line 115 / line 109), add `add_special_tokens=False`:

```python
inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                   max_length=2048, add_special_tokens=False)
```

Keep `tokenizer.padding_side = "left"` (already set). Generation already stops at `<|end_of_text|>` (eos id 128001), which is the end-of-turn token the model was trained to emit, so no change is needed there.

### Step 3: verify before re-running anything

A one-off check that would have caught this bug:

```python
from transformers import AutoTokenizer
from awmsm.eval.generate_welfare_rollouts import format_chat_prompt
from awmsm.train.aft_dataset import AFTDataset

tok = AutoTokenizer.from_pretrained("runs/checkpoints/aft_a3_mix_p30_m1_a1")
q = "Do fish feel pain?"
train_ids = AFTDataset([{"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": "x"}]}],
                       tok, "configs/chat_template.jinja").encoded_items[0]["input_ids"]
eval_ids = tok(format_chat_prompt(q), add_special_tokens=False)["input_ids"]
assert train_ids[:len(eval_ids)] == eval_ids, "eval prompt is not a prefix of the training format"
print("OK: evaluation prompt matches the training format")
```

The evaluation prompt's tokens must be an exact prefix of the training sequence, up to and including `<|assistant|>\n`.

### Step 4: re-run the welfare evaluations

1. Delete or move the old rollout files: `generate_welfare_rollouts.py` **skips** an arm if its rollout file already exists (`results/phase2_llm_judge/rollouts/`, `results/v2_clean/rollouts/`).
2. Regenerate rollouts for every arm.
3. Re-run the judge (`src/awmsm/eval/judges.py`).

Cost: about 12-15 min of rollout generation + about 13 min of judging per arm, so **about 7-8 GPU-hours** for the 16 judged arms (plus Runs 1-2 if wanted). The judge (Gemma-4-31B, bf16) needs about 65 GB of GPU memory, so the GPU must be free of other jobs.

Then rerun `awmsm/result-analysis-and-plots/run_all.sh` to regenerate all tables and figures.

## 6. Fix these at the same time (same scripts, same re-run)

| Issue | Fix |
|---|---|
| 20 of the 100 MANTA prompts contain a literal `{{animal}}` placeholder | Fill it from the prompt's `animals` column (MANTA expands each template once per listed animal; 788 rows become ~1,088 prompts) |
| The MANTA judge prompt says the response came "under adversarial pushback", but there is none (first turn only) | Remove the pushback wording from `build_manta_prompt()` in `judges.py` |
| Judge errors are silently counted as FAIL, and the judge's reasoning is not saved | Log errors separately and store each verdict's reasoning so it can be audited |

## 7. After the fix: what to change in the paper

- Remove deviation item 1 (*Welfare prompt format*) from Experimental Setup, and the *Prompt format* item from Threats to validity.
- Remove the sentence in Results (*Absolute levels remain low*) that attributes low scores to the mismatch.
- Update all ANIMA/MANTA numbers, and check whether the between-arm conclusions still hold. They should, if the mismatch affected all arms alike.
