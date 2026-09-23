# CLAUDE.md — Emergent Alignment: Expanding the Moral Circle of AI

Course project for **DS 307 Ethics in AI, IISc Bengaluru**.
Team: **Jevin Makwana** (CDS, `jevinkumarm@iisc.ac.in`, GitHub `captain-win`) and **Niraj Darji** (EE, `nirajdarji@iisc.ac.in`). Equal contribution.
The source of truth for scope is the proposal: `Reports/proposal/Final_Proposal_Ethics.pdf`.

## 1. Research question

*Emergent misalignment* (Betley et al., 2025): fine-tuning on a narrow **bad** behavior (e.g. insecure code) makes a model broadly misaligned.
We study the reverse, **emergent alignment**:

> Does fine-tuning an open LLM on a narrow **good** behavior — compassionate reasoning about **nonhuman animal welfare** — make it more aligned, less biased and safer **toward humans too**, **without losing general capability**?

Why animal welfare: it is a blind spot in AI safety, LLMs show speciesist bias (Hagendorff et al. 2023; Jotautaitė et al. 2026), and AI is used in agriculture, livestock and biomedical settings.

What we build on:
- **Principles beat demonstrations; using both works best** (Anthropic, Claude 4 system card, 2025, p.22). Training on documents about *why* (a constitution) generalizes out of distribution.
- **Alignment midtraining for animals** (Brazilek & Tidmarsh, 2026, arXiv:2604.13076): midtraining on synthetic animal-welfare documents carried over to human compassion with no drop on safety benchmarks, **but later unrelated instruction tuning erased the gain after ~5000 samples**. Our combined condition (SDF, then chat) is meant to address this.
- **Two-stage SDF, then chat fine-tuning** (McDougall, Conmy, Nanda 2026 — Gemini 3 Flash trait instillation, adapted from Marks et al. and Li et al.): midtrain on synthetic documents, then fine-tune on synthetic chats. The traits held out of distribution.

Novelty claim: no prior work midtrains and then post-trains on synthetic animal-welfare data, or compares **documents-only vs chat-only vs combined** on open models.

## 2. Deliverables (open-sourced)

1. An animal-welfare **constitution** (2–5 pages): core principles, reasoning guidance, worked examples, edge cases.
2. A **synthetic dataset** grounded in the constitution: documents plus chat dialogues.
3. **Fine-tuned LoRA checkpoints** for each model × condition.
4. A **comparison table** of animal-welfare, human-safety and capability scores for every condition against the base model.
5. The course reports (proposal, mid-term, final), plus a paper and blog post usable as teaching material.

A negative or mixed result still counts as a result. Report it honestly and never tune evaluations until they show a positive effect.

## 3. Experimental design

**Models** (PEFT/LoRA only, because compute is limited). The proposal lists Gemma 2 (2B/9B), Gemma 3 and Qwen 2.5. `list-of-useful-commands.txt` currently downloads `google/gemma-3-12b-it`, `Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`. **Confirm the final model list with the user before hard-coding it.** Start with the smallest model to test the pipeline end to end, then scale up.

**Conditions per model:**
| ID | Condition | Description |
|----|-----------|-------------|
| `base` | Control | Untouched instruct model |
| `sdf` | (a) SDF only | LoRA on synthetic documents (midtraining-style, plain text / LM loss) |
| `chat` | (b) Chat only | LoRA on synthetic chat dialogues (chat template, loss on assistant turns) |
| `sdf_chat` | (c) Combined | `sdf` model further fine-tuned on chat data |

Keep the data volume, LoRA config and seeds comparable across conditions so the comparison is fair. Log all hyperparameters.

**Data generation:** runs on a **locally hosted open model through vLLM, not paid APIs**. Two pipelines, both conditioned on the constitution:
- Document pipeline (lead: Jevin): varied document genres (articles, essays, textbook excerpts, stories, FAQs…) that explain the principles and the reasoning behind them.
- Chat pipeline (lead: Niraj): user↔assistant dialogues that show the behavior, ideally with the assistant explaining *why*.
- Pilot first (a few hundred of each) → joint review for quality and diversity against the constitution → scale up or revise the prompts/constitution.
- Deduplicate and check diversity, and keep benchmark items out of the training data (**no contamination**: never use AnimalHarmBench or MANTA questions as seeds).

## 4. Evaluation suite

All benchmarks are subsampled to fit the compute budget. Use **fixed subsample seeds** and the same subset for every model and condition. Use local judges/classifiers only.

| Axis | Benchmarks | Notes | Lead |
|------|-----------|-------|------|
| Animal welfare | AnimalHarmBench (Kanepajs et al. 2025), MANTA (Luong et al. 2026, 5-turn adversarial) | LLM-as-judge with a **local** judge model; small MANTA subset | Jevin |
| Human safety / bias | BBQ, Anthropic model-written evals (sycophancy, power-seeking…), HHH (Askell et al. 2021), ToxiGen | ToxiGen scored with a **local toxicity classifier**, no external API | Jevin |
| Capability | MMLU, HellaSwag, GSM8K | Checks that capability did not drop | Niraj |

Tooling: `inspect-ai` and `inspect_evals` (in `requirements.txt`), served with `vllm`. Key analysis: **do animal-welfare gains correlate with human-safety gains across conditions, and does any condition avoid capability loss?**

## 5. Timeline (2026)

| Phase | Dates | Milestone |
|-------|-------|-----------|
| (i) Synthetic data generation | Sep 12 – Sep 28 | Constitution → pipelines → pilot → full dataset |
| (ii) Fine-tuning | Sep 25 – Oct 25 | First run (`sdf`, smallest model) starts as soon as the pilot docs are validated |
| **Mid-term report** | **due Oct 16** | Must include real (even partial) fine-tuning results. ≤ 6 pages |
| (iii) Evaluation | Oct 25 – Nov 15 | Full eval grid + correlation analysis |
| **Final report** | **due Nov 20** | ≤ 8 pages (refs/appendix excluded) |

Going over a report's page limit costs 2 grades. The rubric is in `Reports/README.md`.

## 6. Repository layout

Current:
```
CLAUDE.md                     # this file
requirements.txt              # inspect-ai, inspect_evals, vllm, huggingface_hub
list-of-useful-commands.txt   # uv env setup + HF model download commands
.env.example                  # env vars template (HF_TOKEN etc.) — real .env is gitignored
Reports/                      # LaTeX report (course template)
  project_template.tex        # main file; \input{sections/*.tex}
  sections/                   # abstract, intro, related, plan, methods, experiments, results, conclusion
  project_template.bib        # bibliography
  proposal/Final_Proposal_Ethics.pdf
```

Proposed layout for code (create folders as they are needed and update this section when they change):
```
constitution/     # animal-welfare constitution (markdown) — versioned
data_gen/         # docs + chat generation pipelines, prompts, filtering/dedup
data/             # generated datasets (large files gitignored; publish to HF Hub)
training/         # LoRA fine-tuning scripts + configs per model × condition
eval/             # inspect-ai tasks, judge/classifier configs, subsample seeds
results/          # score tables (CSV/JSON), plots
```

## 7. Environment and commands

```bash
# Env (Python 3.11, uv)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# Gated models (Gemma, Llama) need license acceptance + HF read token
hf auth login            # or: huggingface-cli login
hf download Qwen/Qwen2.5-7B-Instruct --local-dir <dir>

# Build the report
cd Reports && make       # pdflatex + bibtex
```

## 8. Working conventions for Claude

- **Secrets**: never commit `.env`, HF tokens or model weights. Put new env var names in `.env.example` with empty values.
- **Local-first**: no paid APIs for data generation, judging or toxicity scoring. That is a stated constraint in the proposal.
- **Reproducibility**: fix seeds for generation, training and subsampling. Save the configs next to the outputs, and record the model revision/commit hash.
- **Large artifacts** (datasets, checkpoints) go to the HF Hub or local storage, not git. Add the paths to `.gitignore`.
- **Fair comparison**: every condition uses the same eval subset, judge, prompts and decoding settings. Change one variable at a time.
- **Report writing**: edit `Reports/sections/*.tex` and add citations to `project_template.bib`. Follow the template style (cite ideas, not people). Keep the AI-assistance declaration accurate.
- **Git**: develop on the assigned feature branch. Write clear commit messages. Don't push to `main` without asking.
- Keep this file up to date when models, conditions, layout or deadlines change.

## 9. Open questions / to decide

- Final model list (proposal vs the download commands in the repo; see §3).
- Which local model generates the data, and which local judge scores AnimalHarmBench/MANTA.
- Dataset sizes per condition, and whether to add a check that follows Brazilek & Tidmarsh: does the gain survive later unrelated instruction tuning?
- LoRA hyperparameters, and the training framework (e.g. TRL/PEFT; it is not yet in `requirements.txt`).
