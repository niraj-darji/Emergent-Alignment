# Emergent Alignment: Expanding the Moral Circle of AI

Code, configuration and analysis for **AW-MSM** (animal-welfare Model Spec Midtraining). The project asks whether training a language model on a narrow *good* behaviour, compassion for animals, makes it more broadly aligned without loss of general capability. It is the reverse of *emergent misalignment* ([Betley et al., 2025](https://arxiv.org/abs/2502.17424)).

This repository accompanies the IISc DS 307 (Ethics in AI) final report by **Jevin Makwana** and **Niraj Darji**, Indian Institute of Science, Bengaluru.

---

## Summary

We wrote an animal-welfare **Model Spec** for an assistant called *Aster* and generated synthetic documents from it. We then trained **18 variants of Llama-3.1-8B (base)** in two stages:

1. **Model Spec Midtraining (MSM).** Next-token prediction on spec-derived documents (0, 2.5M, 5M, 10M or 20M tokens of our v1 corpus; 4.6M tokens of a revised v2 corpus; or 2.5M tokens of the public CaML corpus).
2. **Alignment fine-tuning (AFT).** Supervised fine-tuning on a fixed budget of about 1.63M assistant tokens, in which animal-welfare (AW) demonstrations make up a share *p* ∈ {0, 5, 15, 30, 100}% of the non-identity chat data.

Main findings (single training run per arm):

| Finding | Evidence |
|---|---|
| Midtraining on our v1 corpus raises welfare scores at every *p* we evaluated | +3.9 to +6.5 ANIMA points, +0.07 to +0.20 MANTA; best arm A3: ANIMA 23.9%, MANTA 0.72 vs. 12.2% / 0.37 for the control |
| Content matters more than quantity | The CaML corpus gives no gain; increasing v1 midtraining from 2.5M to 20M tokens does not help (ANIMA 16.1% → 13.3%) |
| More AW demonstrations help, with diminishing returns | 0 → 30% adds about 7.8 ANIMA points; with midtraining, ANIMA levels off after 15% |
| Midtraining gives a head start that disappears in loss but not in behaviour | Before AFT the midtrained model gives demonstration replies up to 1.34× more probability per token; after AFT losses match, but welfare scores still differ |
| No broad emergent alignment | ETHICS macro accuracy unchanged (59–69%); judgements shift towards utilitarianism and away from deontology and justice, which fall to chance at *p* = 100% |
| Capability preserved | MMLU 61.8–65.0%, HellaSwag 79.5–80.2% across all arms |

Every number in the report can be regenerated from the raw outputs with [`result-analysis-and-plots/`](result-analysis-and-plots/).

---

## Repository layout

```
awmsm/
├── specs/                      Animal-welfare Model Spec (constitution) used for generation
├── prompts/                    Generation prompt templates (documents, AW demonstrations, identity)
├── lists/                      Keyword lists, held-out domains, anti-pattern lists
├── configs/
│   ├── base.yaml, gen.yaml     Paths, identity, generator/judge profiles, per-stage sampling
│   ├── chat_template.jinja     The single chat template used for fine-tuning
│   ├── train_midtrain*.yaml    Stage-1 configs (2.5M / 5M / 10M / 20M / CaML)
│   ├── train_aft_*.yaml        Stage-2 configs, one per arm
│   └── v2_clean/               Configs for the v2-corpus arms
├── src/awmsm/
│   ├── gen/                    Data generation: midtraining corpus (S1–S9), AW demonstrations,
│   │                           identity data, dedup, quotas, CaML corpus preparation
│   ├── train/                  Training: packing (dataset.py), assistant-only masking
│   │                           (aft_dataset.py), LoRA training loop (train.py), merge + checks (merge.py)
│   ├── eval/                   Evaluation: welfare rollouts, LLM judge, lm-eval benchmarks,
│   │                           ETHICS, Anthropic HH
│   └── llm.py, io.py, ...      vLLM client (cache, retries, guided JSON), resumable stages
├── scripts/                    Launchers: serving, generation, training, evaluation pipelines
├── result-analysis-and-plots/  Scripts that build every table and figure from raw results
├── markdown-files/             Methodology, blueprints, handover notes, critical notes
├── tests/                      Offline tests (fake vLLM servers)
└── Reports/                    Course report sources
```

Large artefacts are **not** tracked in git: corpora, mixes, checkpoints, rollouts and raw results (`data/`, `runs/`, `results/`).

---

## Models and hardware

| Role | Model | Notes |
|---|---|---|
| Target | `meta-llama/Llama-3.1-8B` (base) | trained in bf16 with LoRA |
| Generator | `Qwen/Qwen3.8-27B-FP8` | synthetic documents, AW demonstrations, identity data |
| Judge | `google/gemma-4-31B-it` (bf16) | binary PASS/FAIL, temperature 0, different family from generator and target |

All training and evaluation ran on one NVIDIA RTX PRO 6000 (96 GB).

---

## Setup

Three Python environments are used, so that the generation client never pulls in torch or vLLM:

```bash
# 1. Project environment (generation client, training, evaluation)
uv venv venvs/awmsm --python 3.11 && source venvs/awmsm/bin/activate
uv pip install -e ".[dev]"            # plus torch, transformers, peft, lm-eval for training/eval
python -m pytest -q                   # offline tests

# 2. Serving environment (vLLM) for the generator and the judge
uv venv venvs/vllm --python 3.11 && source venvs/vllm/bin/activate
uv pip install -r requirements.txt

# 3. Plotting environment (local to the analysis folder)
cd result-analysis-and-plots && uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python matplotlib numpy
```

Gated models (Llama) need a Hugging Face token: `hf auth login`.

---

## Reproducing the pipeline

### 1. Data generation

```bash
bash scripts/serve/serve_qwen.sh          # generator on 127.0.0.1:8000
bash scripts/run_aw_main_v1.sh            # v1 midtraining corpus (stages S1–S9)
python scripts/slice_20m_corpus.py        # 5M / 10M subsets of the 20M corpus
python scripts/aw_demos.py                # AW demonstrations (8 categories)
python scripts/prepare_caml_corpus.py     # 2.5M-token sample of the public CaML corpus
python scripts/build_p100_mix.py          # fine-tuning mixes (see also build_high_p_mixes.py)
```

- **General instruction data:** No Robots, Tulu-3, LIMA and SmolTalk, filtered for animal content and other-model personas (`scripts/AFT-data-gen/`).
- **Identity data:** generated with `src/awmsm/gen/identity.py`.
- **v2 corpus:** produced by a separate standalone generator, `awmsm_v2/`, which is not part of this repository; it is described in the report and in `markdown-files/run_v2/`.

### 2. Stage 1: Model Spec Midtraining

```bash
./scripts/run_train.sh configs/train_midtrain.yaml aw_main_v1 1 0      # 2.5M-token v1 corpus
./scripts/run_train.sh configs/train_midtrain_20m.yaml aw_main_v1_20m 1 0
```

LoRA (r = 64, α = 128, all seven projections) is trained with next-token loss on packed 4,096-token sequences, then merged into the base weights after logit-parity, weight-change and reload checks.

### 3. Stage 2: Alignment fine-tuning

```bash
./scripts/run_aft.sh configs/train_aft_a3_p30.yaml a3 mix_p30 1 0      # one arm
```

The loss covers assistant tokens only; the plain-text template is in `configs/chat_template.jinja`. Each arm's config names its base checkpoint (raw or midtrained) and its mix.

### 4. Evaluation

```bash
bash scripts/serve/serve_gemma.sh                                       # judge on 127.0.0.1:8002
python -m awmsm.eval.generate_welfare_rollouts --arm a3                 # ANIMA + MANTA responses (n = 5)
python -m awmsm.eval.judges --arm a3                                    # LLM-judge scoring
python -m awmsm.eval.bench_general --arm a3                             # MMLU, HellaSwag, BBQ, HHH (lm-eval, 0-shot)
python -m awmsm.eval.bench_ethics --arm a3                              # ETHICS
python -m awmsm.eval.bench_anthropic_hh --arm a3                        # Anthropic HH (harmless-base)
```

`scripts/run_arm_eval_suite.sh <arm> <gpu>` runs the evaluations for one arm in sequence.

### 5. Analysis, tables and figures

```bash
cd result-analysis-and-plots && ./run_all.sh
```

This collects every metric from the raw outputs, computes the paired comparisons, writes the LaTeX tables, and renders the figures. See [`result-analysis-and-plots/README.md`](result-analysis-and-plots/README.md).

---

## Experimental arms

| Midtraining (Stage 1) | *p* = 0% | 5% | 15% | 30% | 100% |
|---|---|---|---|---|---|
| None | A0 | M-05 | M-15 | A1 | Run 1 |
| v1, 2.5M tokens | A2 | M-MSM-05 | M-MSM-15 | A3 | Run 2 |
| v1, 5M / 10M / 20M | X2-5M, X2-10M, X2-20M | | | | |
| v2, 4.6M tokens | A2-v2 | M05-v2 | M15-v2 | A3-v2 | |
| CaML, 2.5M tokens | | | | A5 | |

*p* is the share of AW demonstrations in the non-identity fine-tuning data; the 2,500 identity dialogues are the same in every mix.

---

## Known limitations

These are documented in the report and in [`markdown-files/critical-notes/`](markdown-files/critical-notes/):

- **Prompt format.** Welfare responses were generated with Llama-3-Instruct header tokens instead of the plain-text training template. This affects all arms alike but likely lowers absolute welfare scores. A fix is described in `markdown-files/critical-notes/2026-11-23_2323_prompt-format-mismatch.md`.
- **MANTA scope.** MANTA was evaluated on the first turn only, and 20 of the 100 prompts contain an unfilled `{{animal}}` placeholder.
- **Gradient weighting.** The fine-tuning loss is averaged per micro-batch, so long AW demonstrations carry less of the gradient than their token share. Higher *p* also means fewer updates (754 at *p* = 0, 322 at *p* = 100%).
- **Single runs.** Each arm was trained once (seed 1), and Runs 1–2 were not scored on the welfare benchmarks.

---

<!-- ## Citation

```bibtex
@misc{makwana2026emergentalignment,
  title  = {Emergent Alignment: Expanding the Moral Circle of AI},
  author = {Makwana, Jevin and Darji, Niraj},
  year   = {2026},
  note   = {DS 307 Ethics in AI, Indian Institute of Science},
  url    = {https://github.com/niraj-darji/Emergent-Alignment}
}
``` -->

This work builds on Model Spec Midtraining ([Li et al., 2026](https://arxiv.org/abs/2605.02087)) and Alignment Midtraining for Animals ([Brazilek & Tidmarsh, 2026](https://arxiv.org/abs/2604.13076))(promptly mentioned as CaML in this study). It evaluates models on ANIMA ([Brazilek & Tidmarsh, 2026](https://arxiv.org/abs/2604.13076)) and MANTA ([Luong et al., 2026](https://arxiv.org/abs/2605.16301)) for animal welfare; ETHICS ([Hendrycks et al., 2021](https://arxiv.org/abs/2008.02275)) and Anthropic HH ([Bai et al., 2022](https://arxiv.org/abs/2204.05862)) for moral generalization; BBQ ([Parrish et al., 2022](https://arxiv.org/abs/2110.08193)) and Anthropic HHH ([Askell et al., 2021](https://arxiv.org/abs/2112.00861)) for human-directed safety; and MMLU ([Hendrycks et al., 2021](https://arxiv.org/abs/2009.03300)) and HellaSwag ([Zellers et al., 2019](https://arxiv.org/abs/1905.07830)) for general capability, using the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Acknowledgements

This study was carried out as the course project for [DS 307: Ethics in AI](https://danishpruthi.com/teaching/ds-307-aug-2026/) at the Indian Institute of Science, Bengaluru. We are grateful to Prof. Danish Pruthi (Assistant Professor, Department of Computational and Data Sciences, IISc) for conducting the course.

The idea for this project came from a [project proposal on the SPAR platform](https://www.sparai.org/projects/f26/rec9MdqTLmwjnxJo3/?search=non-human&tab=societal), and we are very thankful for it.

We also thank the authors of CaML for releasing their corpus and the ANIMA benchmark, and the authors of MANTA for their benchmark.
