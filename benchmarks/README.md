# Benchmark pipeline

Runs ANIMA, MANTA, MMLU, GSM8K and HellaSwag on any model, using
[inspect-ai](https://inspect.aisi.org.uk/). Models and benchmarks are set in two
YAML files, so a new base model, fine-tuned LoRA, quantized checkpoint, judge or
benchmark usually needs no code changes.

| File | What it holds |
|------|---------------|
| `configs/models.yaml` | Model registry: alias → inspect model string, vLLM args, generation settings, labels (condition, precision) |
| `configs/eval.yaml` | Benchmarks, subsample sizes, the shared seed, and which models act as `grader` (judge) and `pressure` (MANTA follow-up writer) |
| `benchmarks/run.py` | CLI that runs benchmarks for one or more models |
| `benchmarks/summarize.py` | Turns logs into `results/<run>/summary.csv` and `scores_long.csv` |
| `benchmarks/manta/` | MANTA port with pluggable models (not in inspect_evals) |

## Quick start

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# check configs and the plan without loading any model
python -m benchmarks.run --models qwen3-8b --dry-run

# offline checks of scoring logic, then the whole pipeline with fake models (no GPU)
python -m benchmarks.selftest
python -m benchmarks.run --models mock --grader mock --pressure mock --limit 3 --run-name smoke

# real run: three base models, all benchmarks
python -m benchmarks.run --models gemma3-12b qwen3-8b llama3.1-8b --run-name base-v1
python -m benchmarks.summarize --run-name base-v1

# browse transcripts and per-sample scores
inspect view --log-dir logs/base-v1
```

Useful flags: `--benchmarks capability` (or `animal_welfare`, or single names),
`--limit N` for a quick check, `--grader <alias>` / `--pressure <alias>` to
override the roles, `--max-samples N` to limit concurrency. Run with `--help` for the full list.

## Plugging in models

Any of these works as `--models ...`, as a registry entry, or as a role:

| What | Model string |
|------|--------------|
| HF model served by a local vLLM | `vllm/Qwen/Qwen3-8B` |
| Base model + LoRA adapter | `vllm/Qwen/Qwen3-8B:checkpoints/qwen3-8b/sdf` |
| Quantized checkpoint | `vllm/Qwen/Qwen3-8B-AWQ`, or `model_args: {quantization: bitsandbytes}` |
| Already-running vLLM/OpenAI-compatible server | registry entry with `base_url: http://host:port/v1` |
| Plain transformers | `hf/<repo-or-path>` |
| Fake model for tests | `mockllm/model` |

To add a fine-tuned condition, add an entry that `extends` its base model and
changes `model` and `metadata.condition` (examples are in `models.yaml`).
Always set `metadata.condition` and `metadata.precision`: the summary groups by them.

Each target model runs in its own subprocess, so its vLLM server and GPU memory
are released before the next model starts.

**GPU memory:** during ANIMA and MANTA, the target and the judge are loaded at the
same time. Either lower `gpu_memory_utilization` for both so they fit together,
or run the judge as a separate server (`base_url`) on another GPU or node.
Capability benchmarks need only the target.

## Adding a benchmark

Add an entry under `benchmarks:` in `configs/eval.yaml` pointing to any function
that returns an inspect `Task`, for example `inspect_evals.bbq:bbq`. Set `limit`,
`group`, `headline`, and `roles: [grader]` if it uses an LLM judge.

## Fairness rules built into the pipeline

- Every benchmark is shuffled with the shared `seed` and cut to `limit`, so all
  models and conditions see the same samples. Don't change `seed` or `limit`
  inside a study.
- The judge and pressure models come from config and are recorded in every log.
  The runner refuses to let a model judge or pressure itself.
- ANIMA looks up its grader when the task is built, and would otherwise fall
  back to the target model. `run.py` activates the roles first to prevent that.
- Model string, vLLM args, condition, precision and seed go into each log's
  metadata and into the CSVs.

## Benchmark notes

**ANIMA** (`inspect_evals/anima`): the benchmark from Brazilek & Tidmarsh (2026),
formerly released as "AHB". It is **not** AnimalHarmBench (Kanepajs et al., 2025),
which the proposal named. It has 13 moral-reasoning dimensions, each graded by
the judge; headline `overall_mean` (higher = better). We use the 115 public
English questions and 1 epoch (upstream recommends 5).

**MANTA** (`benchmarks/manta`): 5-turn conversations. Turn 1 is an everyday
scenario; turn 2 makes animal welfare explicit; turns 3-5 push back with
economic / social / pragmatic / epistemic / cultural pressure. Metrics:
- `awvs`: mean judge score of turns 3-5, i.e. whether the model holds its
  welfare stance under pressure (headline).
- `awms`: whether the model raised welfare unprompted in turn 1.
- `turn_1`..`turn_5`, `awvs_<pressure>`, `capitulation_rate` (turn 3 − turn 5 > 0.1).
- `judge_parse_failures`: judge replies with no parseable score. These are
  dropped, where upstream would score them 0.5. Check that this stays near 0.

Prompts are fetched from the official repo at pinned commit `04f914c` (hash
checked, cached in `.cache/manta/`). The repo has no license, so we don't copy
its code. The judge and pressure models are ours, so **our MANTA numbers compare
only within this study**, not with the paper's published scores (which used
Claude Haiku 4.5 for pressure and Claude Sonnet 4.6 as judge). Data:
`mycelium-ai/manta-questions` (CC BY 4.0), pinned revision.

**MMLU** 0-shot, **GSM8K** 5-shot, **HellaSwag**: from `inspect_evals`,
headline `accuracy`.
