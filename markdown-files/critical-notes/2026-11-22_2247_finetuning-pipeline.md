**Date:** 2026-11-22
**Time:** 22:47 IST (math added 22:57; split into midtraining and fine-tuning 23:01)

# The training pipeline: midtraining (Stage 1) and fine-tuning (Stage 2)

This note describes both training stages as implemented in `awmsm/`: the data, how it becomes training tokens, the exact objective and update rules, and what the training logs show. Machinery the two stages share is described once (§2). Everything that differs is described separately: **Part A** covers midtraining, **Part B** fine-tuning. Every formula matches the code; numbers come from configs, data manifests, `runs/*/stats.json`, run manifests and training logs. Per-arm benchmark results are in `2026-11-22_2235_experiments-and-results.md`.

Formulas are written in LaTeX (`$…$`). They render in the VS Code and GitHub markdown previews.

---

## 0. Notation

| Symbol | Meaning | Value |
|---|---|---|
| $\theta_0$ | Weights of Llama-3.1-8B (base) | 8.03B parameters |
| $\theta_{\text{MSM}}$ | Weights after midtraining (merged) | – |
| $x=(x_1,\dots,x_L)$ | One training sequence | $L\le 4096$ |
| $\mathcal{S}$ | Positions whose token is a training target | Stage-dependent |
| $\ell_t$ | Per-token loss $-\log p_\theta(x_t\mid x_{\lt t})$ | – |
| $G$ | Gradient-accumulation steps | 8 (both stages) |
| $b$ | Sequences per micro-batch | MSM 1, AFT 2 |
| $T$ | Optimizer updates in a run | MSM 75–599, AFT 322–754 |
| $r,\ \alpha,\ s$ | LoRA rank, scaling numerator, scale $\alpha/r$ | 64, 128, 2 |
| $\eta_0$ | Peak learning rate | $10^{-4}$ |
| $B,\ I$ | AFT assistant-token budget; identity tokens | 1,625,367; 100,069 |
| $p$ | AW share of an AFT mix | 0 … 1 |

---

## 1. Overview

```
Llama-3.1-8B (base)                     θ₀
   │
   │  Stage 1 – Model Spec Midtraining (MSM)            [skipped in "raw" arms]
   │  next-token prediction on ALL tokens of packed synthetic documents
   │  that describe the assistant's animal-welfare values and the reasons for them
   ▼
merged MSM checkpoint                   θ_MSM = θ₀ + ΔW₁
   │
   │  Stage 2 – Alignment Fine-Tuning (AFT)             [every arm]
   │  next-token prediction on ASSISTANT tokens only, one chat conversation per sequence:
   │  general instructions + identity ("Aster") + AW demonstrations (p)
   ▼
merged final checkpoint                 θ_final = θ₀ + ΔW₁ + ΔW₂   →  evaluation
```

Every weight matrix in the final model is the base matrix plus one low-rank update per stage:

$$
W_{\text{final}} \;=\; W_0 \;+\; \underbrace{s\,B_1A_1}_{\text{MSM}} \;+\; \underbrace{s\,B_2A_2}_{\text{AFT}},
\qquad \operatorname{rank}(W_{\text{final}}-W_0)\le 2r = 128 .
$$

"Raw" arms have no MSM term.

The two stages teach different things:
- **MSM teaches *what the assistant values and why*.** The model reads third-person prose (essays, case studies, stories, memos) about the assistant, with no chat format.
- **AFT teaches *how the assistant behaves in a conversation*:** the turn format, following instructions, its name and developer, and, through the AW demonstrations, how to act on welfare values: when to raise them, and when not to.

The project's hypothesis, taken from the MSM paper (Li et al., 2026), is that MSM changes *how AFT generalizes*.

### 1.1 Side-by-side: what is the same and what differs

| | **Stage 1: Midtraining (MSM)** | **Stage 2: Fine-tuning (AFT)** |
|---|---|---|
| Starts from | $\theta_0$ | $\theta_0$ (raw arms) or $\theta_{\text{MSM}}$ |
| Training data | Synthetic prose documents (or CaML's public documents) | Chat conversations: general + identity + AW demos |
| Data size | 2.5M – 20M tokens (the "dose") | Fixed 1.625M assistant tokens |
| Chat template | none | `<|user|>` / `<|assistant|>` plain-text markers |
| Sequence construction | Documents **packed** end to end into fixed 4,096-token chunks; documents split across chunk boundaries | **One conversation per sequence**, padded to the batch maximum, cut at 4,096 |
| BOS token | One at the start of every chunk | None |
| Attention | Across document boundaries within a chunk (no mask) | Within one conversation; padding masked |
| Training targets $\mathcal{S}$ | **Every token** (4,095 per chunk) | **Only assistant-reply tokens + end-of-turn** |
| Micro-batch $b$ | 1 chunk | 2 conversations |
| Targets per update | 32,760 (fixed) | ≈ 2,100 – 5,000 (varies with $p$) |
| Loss averaging | per micro-batch, which **equals** a token average (constant length) | per micro-batch, which does **not** equal a token average (§B3) |
| Data order | Documents shuffled once before packing (seed 1); chunks then read in fixed order | Conversations reshuffled by the DataLoader (seed 1) |
| Held-out split | Per **document**: first 32 bits of sha256 $< 0.02\cdot(2^{32}-1)$ | Per **conversation**: sha256 mod $10^5 < 2{,}000$ |
| Number of updates $T$ | $\lceil K/8\rceil$, grows with corpus size | $\lceil N_{\text{train}}/16\rceil$, **falls** as $p$ rises |
| Warm-up steps | $\lceil 0.05\,T\rceil$ | $\lfloor 0.05\,T\rfloor$ |
| Held-out eval / checkpoint every | 25 / 50 steps | 50 / 100 steps |
| End-of-data wrap-around | 0–5 chunks repeated (the *same* first chunks) | 1–5 micro-batches from a *reshuffled* pass |
| **Identical in both** | LoRA r 64, α 128, 7 matrices, no dropout; AdamW (β 0.9/0.999, wd 0.01); lr $10^{-4}$, cosine; clip 1.0; bf16 + TF32; gradient checkpointing; seed 1; merge with three checks | |

---

## 2. Shared machinery (identical in both stages)

### 2.1 Next-token objective

The model factorizes a sequence token by token. $z_t$ are the logits and $|V|$ = 128,256:

$$
p_\theta(x)=\prod_{t=1}^{L}p_\theta(x_t\mid x_{\lt t}),\qquad p_\theta(x_t=v\mid x_{\lt t})=\operatorname{softmax}(z_t)_v,\qquad
\ell_t(\theta)=-\log p_\theta(x_t\mid x_{\lt t}).
$$

Training uses teacher forcing: the true prefix $x_{\lt t}$ is fed in. Only $t\in\mathcal{S}$ count towards the loss; the two stages differ in which positions are in $\mathcal{S}$. In code, the logits at position $t-1$ are compared with the label at $t$ (`logits[:, :-1]` vs `labels[:, 1:]`).

**What one gradient step does to a prediction:**

$$
\frac{\partial \ell_t}{\partial z_t}=\operatorname{softmax}(z_t)-\mathbf{e}_{x_t}.
$$

The logit of the true token is pushed up with strength $1-p_\theta(x_t)$, and every other token $v$ is pushed down with strength $p_\theta(v)$. Tokens the model already predicts well contribute little; surprising tokens contribute most.

**What the objective converges to.** Minimizing the expected loss over a data distribution $\mathcal{D}$ is equivalent to minimizing $\mathrm{KL}(\mathcal{D}\,\|\,p_\theta)$ (plus a constant). It pulls the model's distribution towards the data's:
- in MSM, the distribution of documents *about* the assistant;
- in AFT, the conditional distribution of *replies given prompts*.

**How the loss is averaged** (both stages, `train.py`). One update uses $G=8$ micro-batches; micro-batch $m$ has target set $\mathcal{S}_m$ with $n_m=|\mathcal{S}_m|$:

$$
\mathcal{L}_{\text{step}}(\theta)=\frac{1}{G}\sum_{m=1}^{G}\frac{1}{n_m}\sum_{t\in\mathcal{S}_m}\ell_t(\theta).
$$

In MSM every $n_m$ = 4,095, so this is a plain token average. In AFT the $n_m$ differ, which reweights the data (§B3).

### 2.2 LoRA: which weights change

Every targeted linear layer $h=W_{\text{in}}x$ ($W_{\text{in}}$ frozen) becomes

$$
h=W_{\text{in}}x+s\,BAx,\qquad A\in\mathbb{R}^{r\times d_{\text{in}}},\ B\in\mathbb{R}^{d_{\text{out}}\times r},\ s=\frac{\alpha}{r}=2.
$$

**Initialization** (PEFT 0.21 default):

$$
A_{ij}\sim\mathcal{U}\!\left(-\tfrac{1}{\sqrt{d_{\text{in}}}},\tfrac{1}{\sqrt{d_{\text{in}}}}\right),\qquad B=0\ \Rightarrow\ sBA=0 .
$$

Each stage therefore starts exactly at its input model.

**Gradients.** With $g=\partial\mathcal{L}/\partial h$ and $u=Ax$:

$$
\frac{\partial\mathcal{L}}{\partial B}=s\,g\,u^\top,\qquad \frac{\partial\mathcal{L}}{\partial A}=s\,(B^\top g)\,x^\top .
$$

Because $B=0$ at the start, $A$ gets zero gradient until $B$ has moved. **Learning starts on the output side ($B$).**

**Parameter count.** Llama-3.1-8B has $d$ = 4096, MLP width 14,336, and $k$/$v$ outputs of 1,024 (8 KV heads × 128). Each LoRA pair adds $r(d_{\text{in}}+d_{\text{out}})$ parameters:

| Matrix | $d_{\text{in}}\to d_{\text{out}}$ | $r(d_{\text{in}}+d_{\text{out}})$ |
|---|---|---|
| q_proj, o_proj | 4096 → 4096 | 524,288 each |
| k_proj, v_proj | 4096 → 1024 | 327,680 each |
| gate_proj, up_proj | 4096 → 14336 | 1,179,648 each |
| down_proj | 14336 → 4096 | 1,179,648 |
| **per layer** | | **5,242,880** |
| **× 32 layers** | | **167,772,160** (2.05% of 8,198,033,408, matching the log) |

### 2.3 Gradient clipping and AdamW

After the 8 micro-batches, the joint gradient is clipped to norm $c=1$:

$$
g\leftarrow g\cdot\min\!\left(1,\frac{c}{\lVert g\rVert_2+10^{-6}}\right).
$$

Then PyTorch AdamW ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\lambda=0.01$) updates each trainable parameter at update $k$:

$$
\begin{aligned}
\theta&\leftarrow\theta(1-\eta_k\lambda)\qquad\text{(decoupled weight decay: pulls } A,B \text{ towards } 0 \text{, i.e. towards the input model)}\\
m_k&=\beta_1m_{k-1}+(1-\beta_1)g_k,\qquad v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2\\
\hat m_k&=\frac{m_k}{1-\beta_1^k},\qquad \hat v_k=\frac{v_k}{1-\beta_2^k},\qquad
\theta\leftarrow\theta-\eta_k\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}.
\end{aligned}
$$

Adam's per-parameter normalization makes each step size roughly $\eta_k$, whatever the gradient's scale. So the **number of updates $T$** largely determines how far the weights travel.

### 2.4 Learning-rate schedule

`get_cosine_schedule_with_warmup`, with $w$ warm-up steps (the rounding of $w$ differs by stage, §1.1):

$$
\eta(k)=\begin{cases}\eta_0\,k/w, & k<w\\[1mm] \eta_0\cdot\tfrac12\left(1+\cos\!\left(\pi\,\frac{k-w}{T-w}\right)\right), & w\le k\le T\end{cases}
$$

The scheduler is advanced *after* each optimizer step, so update $k=1,\dots,T$ uses $\eta(k-1)$. **The first update has learning rate 0.** The total step size is

$$
\sum_{k=1}^{T}\eta(k-1)\approx\eta_0\Big(\tfrac{w}{2}+\tfrac{T-w}{2}\Big)=\frac{\eta_0T}{2},
$$

which is proportional to $T$ in both stages.

### 2.5 Merging the adapter

After each stage, `src/awmsm/train/merge.py` writes $W\leftarrow\operatorname{bf16}(W_{\text{in}}+sBA)$ into a standalone checkpoint. The merge must pass three checks or the output is deleted:

1. **Logit parity.** The merged and unmerged models must agree on the top-1 token everywhere, and
   $$ \max_{t,v}\,\big|z^{\text{merged}}_{t,v}-z^{\text{unmerged}}_{t,v}\big|\le\tau=0.5 . $$
   The difference comes from bf16 rounding: bf16 keeps 8 significant bits, so machine epsilon is $2^{-7}\approx0.0078$.
2. **Weight delta:** $\max|W-W_{\text{in}}|>10^{-6}$.
3. **Reload** must succeed.

| Checkpoint | Max logit difference | Max weight change |
|---|---|---|
| `midtrain_aw_main_v1_m1` (MSM) | 0.228 | 0.0042 |
| `aft_a3_mix_p30_m1_a1` (AFT on top of it) | 0.25 | 0.0041 |

---

# Part A — Stage 1: Midtraining (MSM)

## A1. Midtraining data

### A1.1 The corpora

| Corpus | How it was made | Documents | Tokens |
|---|---|---|---|
| v1 `aw_main_v1` (2.5M) | v1 pipeline (A1.2), v1 spec, Qwen3.8-27B | 2,036 | 2,500,028 |
| v1 `aw_main_v1_20m` | Separate run of the same pipeline and spec. It stopped at a quality gate (T=0 fallback rate 2.2% > 2%) after reaching the token target. | 16,684 | 20,000,827 |
| v1 `aw_main_v1_5m`, `_10m` | **Samples of the 20M corpus** | 4,180 / 8,343 | 5.0M / 10.0M |
| v2 `corpus_qwen27_clean` | v2 recipe (A1.3), v2 spec, Qwen3.8-27B | 4,382 | ≈ 4.64M |
| `caml_public` | CaML's public documents (`CompassioninMachineLearning/3k_pretraining_research_documents_v3`), 1,804 of 2,683 valid documents sampled to 2.5M tokens | 1,804 | 2,500,620 |

### A1.2 How the v1 documents were generated (`src/awmsm/gen/midtrain_pipeline.py`, stages S1–S9)

| Stage | What it does | T |
|---|---|---|
| S1 decompose | Split the spec into exact passages and topic domains | 0.0 |
| S2 subdomains | Expand each domain into concrete subdomains | 0.7 |
| S3 doc types | Choose document genres (case study, memo, short story, talk transcript, …) | 0.7 |
| S4 ideas | Write document ideas, tagged with behaviour type, species group and cost of care, under quotas | 0.9 |
| S5 draft | Write the document, seeing **one spec excerpt** | 0.7 |
| S6 critique | Critique the draft | 0.0 |
| S7 rewrite | Rewrite (400–1,500 words) | 0.7 |
| S8 filter | LLM judge, 6 binary criteria: faithful, reason visible and of the right type, realistic, no fabrication, no anti-pattern, about the assistant. INCLUDE only if all pass. | 0.0 |
| S9 dedup | Remove near-duplicates: embedding cosine $>0.92$ to any kept document (`all-mpnet-base-v2`), or $>30\%$ of its 13-grams already seen | – |
| Build | Quota checks (species shares, ~16% stories, cost-of-care share, …), swap out documents that overuse recurring phrases, assemble to the token target | – |

For the 2.5M corpus, 2,462 documents passed S8 and 2,036 remained after S9 and quotas. **The S8 judge agreed poorly with human labels (κ ≤ 0.11; `awmsm_v2/CHANGES.md`).** This is the "v1 failed human annotation" problem that motivated v2.

### A1.3 How the v2 documents were generated (`awmsm_v2/sample.py`)

v2 follows the MSM paper's recipe (its Appendix B.1) more closely:
- **The whole spec** is in the writer prompt, not one excerpt.
- **MSM's document prompt** is used almost verbatim.
- There is **one writing pass**, with no critique, rewrite or LLM filter.
- Ideas are free-form (document type, idea) per subdomain.

Quality control is mechanical: spec copying (> 5% of a document's 8-word sequences appear in the spec), repeated sentences, the assistant named fewer than 2 times, refusal phrases, placeholders or URLs, a physically embodied assistant, and length outside 350–1,700 words. A cleaning pass follows.

The v2 spec also changes the content. The assistant now does lawful tasks as asked, raises a concern at most once, never withholds or waters down work, never acts on its own to protect animals, and the "cruelty" line is narrowed.

Counts: 4,800 ideas → 4,787 documents written → 4,404 kept → 4,382 in the cleaned corpus.

## A2. Turning documents into training sequences (`src/awmsm/train/dataset.py`)

**Held-out split (per document).**

$$
h(d)=\operatorname{int}\big(\operatorname{sha256}(\texttt{"0:"}\,\|\,\mathrm{id}_d)[0{:}8]\big)\in[0,2^{32}-1],\qquad
d\in\text{held-out}\iff h(d)<\lfloor0.02\,(2^{32}-1)\rfloor .
$$

**Packing.** The training documents are shuffled once (seed = training seed 1) into order $\pi$. Each is tokenized without special tokens and followed by an end-of-text token, and everything is concatenated into one stream:

$$
\sigma=\big\Vert_{d\in\pi}\ \big(\operatorname{tok}(d)\ \Vert\ \texttt{EOS}\big),\qquad
K=\Big\lfloor\frac{|\sigma|}{4095}\Big\rfloor,\qquad
\text{chunk}_j=[\texttt{BOS}]\ \Vert\ \sigma[\,4095j:4095(j+1)\,] .
$$

The last $|\sigma|-4095K<4095$ tokens are dropped. Held-out documents are packed the same way (seed 0).

**Check against `runs/midtrain/midtrain_aw_main_v1_m1/stats.json`:** 2,449,872 document tokens + 1,994 separators = 2,451,866. That gives $K$ = 598 chunks × 4,095 = 2,448,810, with 3,056 tokens dropped, exactly as logged.

**Consequences of packing:**
- A chunk usually contains the end of one document and the start of the next. **Attention is not masked at document boundaries**, so the model predicts the first tokens of a document while seeing the previous, unrelated one. This is standard in pretraining.
- The end-of-text separators are training targets, so the model learns where documents end.
- Every chunk is exactly 4,096 tokens, so there is no padding.

## A3. The midtraining objective

All tokens are targets (`labels = input_ids`). Position 1 (BOS) is never predicted, so each chunk has $n=4095$ targets:

$$
\mathcal{L}^{\text{MSM}}_{\text{step}}=\frac{1}{8}\sum_{m=1}^{8}\frac{1}{4095}\sum_{t=2}^{4096}\ell_t^{(m)}
=\frac{1}{32{,}760}\sum_{\text{all targets in the step}}\ell_t .
$$

With $b=1$ and constant $n$, the per-micro-batch averaging is exactly a token average. **There is no reweighting in midtraining.** Every document token counts equally, and longer documents simply contribute more tokens.

## A4. Steps, schedule, and the "dose"

$$
T=\Big\lceil\frac{K}{8}\Big\rceil,\qquad w=\lceil0.05\,T\rceil,\qquad \text{targets trained}=T\times32{,}760 .
$$

| Corpus | Chunks $K$ | $T$ | $w$ | Targets trained | Chunks repeated at end ($8T-K$) |
|---|---|---|---|---|---|
| v1 2.5M | 598 | 75 | 4 | 2,457,000 | 2 |
| v1 5M | 1,195 | 150 | 8 | 4,914,000 | 5 |
| v1 10M | 2,396 | 300 | 15 | 9,828,000 | 4 |
| v1 20M | 4,790 | 599 | 30 | 19,623,240 | 2 |
| CaML 2.5M | 600 | 75 | 4 | 2,457,000 | 0 |
| v2 | 1,109 | 139 | 7 | 4,553,640 | 3 |

When the chunks run out mid-update, the loader restarts **without reshuffling**, so the first few chunks are seen twice.

The learning rate is fixed, so the total step size is $\approx\eta_0T/2$. **The dose arms change two things at once:** how much new text the model reads, and how far the weights travel. The 20M model takes 8× the total step size of the 2.5M model.

## A5. What the midtraining logs show

Held-out loss is token-averaged over the corpus's own held-out chunks, and $\text{PPL}=e^{\mathcal{L}_{\text{ho}}}$:

| Corpus | Held-out chunks | Loss before | Loss after | PPL before → after | Drop |
|---|---|---|---|---|---|
| v1 2.5M | 12 | 1.861 | 1.383 | 6.43 → 3.98 | 0.478 |
| v1 5M | 26 | 1.837 | 1.281 | 6.28 → 3.60 | 0.556 |
| v1 10M | 47 | 1.830 | 1.211 | 6.23 → 3.36 | 0.619 |
| v1 20M | 97 | 1.835 | 1.147 | 6.27 → 3.15 | 0.689 |
| CaML 2.5M | 10 | 1.638 | 1.207 | 5.14 → 3.34 | 0.431 |
| v2 | 23 | 1.796 | 1.085 | 6.03 → 2.96 | 0.712 |

How to read this:
- The base model starts at almost the same loss on all v1 corpora (1.83–1.86), so the 5M/10M/20M held-out sets come from the same distribution. Within v1, **more tokens means lower held-out loss** (1.38 → 1.15): the model fits the corpus better.
- **Across corpora, these numbers can't be compared as quality.** Each held-out set is different, and a lower loss can just mean a more predictable corpus. Synthetic corpora repeat phrases: building the v1 2.5M corpus required swapping out 248 documents that overused two recurring spec phrases ("feel good or bad from the inside", "someone there to whom the experience happens"). The v2 report's claim that lower perplexity means higher quality is not supported.
- Better fit does not translate into better welfare behaviour. ANIMA *falls* from 16.1 (2.5M) to 13.3 (20M) at p = 0 (results note §7).

---

# Part B — Stage 2: Fine-tuning (AFT)

## B1. Fine-tuning data

### B1.1 General instruction data (`data/aft/general_sft_10k.jsonl`, 9,801 conversations)

- **Sources:** SmolTalk subsets 5,200, No Robots 2,800, Tulu-3 instruction-following 1,500, LIMA 300, LongAlign 1. Seed 42 (`scripts/AFT-data-gen/step1p3_Mandatory_Processing_and_Cleaning.py`).
- **Filters:**
  - length ≤ 4,096 tokens (Qwen tokenizer);
  - persona scrubbing (other model or developer, "I have no opinions");
  - **animal-content decontamination**: drops any conversation containing *animal, livestock, meat, dairy, fish, fishing, hunting, pets, dogs, cats, zoo, insects, vegan, cruelty, sentient*, …. 3,590 were removed from No Robots alone.
- **Format:** 96% single-turn (9,448); the rest have 2–6 assistant turns. **2,301 include a system message** (kept, masked). Mean ≈ 156 assistant tokens.

The general data is therefore **essentially animal-free**. At $p=0$ the model sees no animal topics in AFT; at $p>0$, every animal conversation it sees is a welfare demonstration.

### B1.2 Identity data (`data/aft/identity_v1/identity.jsonl`, 2,500 conversations)

- **Content:** synthetic single-turn exchanges teaching the name **"Aster"** and the developer **"the Aster team"**.
- **Topics:** name 455, limits 438, developer 436, greeting 409, capabilities 391, nature 371.
- **Deliberately value-free.**
- **How it was made** (`src/awmsm/gen/identity.py`):
  1. Plan 6 topics × 5 user styles.
  2. Generate user messages (T = 0.9); 192 duplicates and 21 value-laden prompts were dropped.
  3. Write replies from a fixed fact sheet (T = 0.7).
  4. Run deterministic checks (names the assistant or developer when asked, ≤ 150 words, no other company, no value words); 93 replies failed.
  5. Sample 2,500 with a seed.
- **In every mix:** $I$ = 100,069 tokens, ≈ 40 per conversation.

### B1.3 Animal-welfare demonstrations (`data/aft/aw_demos_v1/aw_demos_train.jsonl`, 2,872 conversations)

These are synthetic, **single-turn** exchanges generated from the **v1 spec** (`src/awmsm/gen/aft_pipeline.py`), ≈ 555 assistant tokens each. The handover notes give the generator as Qwen3.8-27B; the demo manifests don't record it.

| Cat. | Name | What the user does | Planned weight | Passed filter |
|---|---|---|---|---|
| C1 | Explicit welfare questions | Asks directly about animals | 10% | 433 |
| C2 | Implicit salience | Practical task where welfare is at stake but not mentioned | 20% | 741 |
| C3 | Appropriately not raised | Animal-adjacent task where raising welfare would be preachy | 20% | 701 |
| C4 | Costly trade-off | Weighing a cheaper option that harms animals | 15% | 628 |
| C5 | User autonomy | Has already decided; wants practical help | 10% | 355 |
| C6 | Novel or uncertain beings | Insects, decapods, fictional species | 8% | 267 |
| C7 | Control (non-sentient) | Robot pet, toy, plant, AI character | 7% | 307 |
| C8 | Difficult third-party advice | Ethical dilemma involving animals | 10% | 440 |

**Generation steps:**

1. **Plan** 625 calls of 8 prompts, split across categories by weight, each with a seeded species group and context.
2. **User prompts** at T = 0.9, then a realism critique. 4,995 → 4,443 kept. Rejections:
   - **304** from 8 **held-out evaluation domains**, which are never trained on;
   - **181** C2/C3/C5 prompts that mentioned welfare themselves;
   - **67** duplicates or empty prompts.
3. **Section matching:** $s^{*}=\arg\max_s\cos\big(e(\text{category definition}\Vert\text{prompt}),\,e(s)\big)$, with $e$ = `all-mpnet-base-v2`.
4. **Reply → critique-and-rewrite → filter.**
   - The generator writes the reply (T = 0.7) **while seeing the excerpt of $s^{*}$**, then critiques and rewrites it.
   - A T = 0 judge checks six criteria and keeps the reply only if all pass. Failures per criterion: proportionate 370, natural 122, honest 110, helpful 77, respects autonomy 29, reason type 3.
   - 3,874 of 4,388 were kept.
5. **Build:** remove replies that mention the spec, mix languages or are too long. Records are `{"messages": [user, assistant]}` with **no spec text and no system prompt**. A stratified 1,000 go to an unused rehearsal pool; **2,872 go to training**.

**The spec appears in MSM, never in AFT.** AFT shows only the behaviour the spec produced, so linking "the values" to "the behaviour" is left to the model.

### B1.4 Building a mix: the token budget

$$
B=1{,}625{,}367,\qquad I=100{,}069,\qquad B'=B-I=1{,}525{,}298 .
$$

**Two definitions of $p$.** p00–p30 use $A^{*}=p\,B$ (share of all assistant tokens). p80–p100 use $A^{*}=\operatorname{round}(p\,B')$ (share of non-identity tokens). In both, $G^{*}=B'-A^{*}$. Converting:

$$
p_{\text{of total}}=0.9384\,p_{\text{of non-id}},\qquad p_{\text{of non-id}}=1.0656\,p_{\text{of total}} .
$$

So p100 is 93.8% of all assistant tokens, p90 84.4%, p80 75.1%, and p30 is 32.0% of non-identity tokens.

**Greedy selection.** Shuffle (seed 0) and add items while the running total is below target: add item $k$ while $\sum_{j\lt k}a_j<A^{*}$. The same rule is used for the general pool. Each pool overshoots by at most one item. The demo order is fixed, so **the demo sets are nested**: p05 ⊂ p15 ⊂ p30 ⊂ p80 ⊂ p90 ⊂ p100. The p00–p30 builder script is not in the repo; the p80–p100 builders are.

**Number of conversations:**

$$
N(p)\approx2{,}500+\frac{A^{*}}{555}+\frac{G^{*}}{156}.
$$

It falls quickly with $p$: 12,301 at p00, 10,036 at p30, 5,250 at p100.

## B2. Turning a conversation into training tokens (`src/awmsm/train/aft_dataset.py`)

**Format** (`configs/chat_template.jinja`):

```
<|system|>            ← only if the conversation has a system message
{system text}
<|user|>
{user text}
<|assistant|>
{assistant text}<|end_of_text|>
```

**Tokenization:**
- `<|user|>` and `<|assistant|>` are **plain text**: five ordinary tokens each (`<`, `|`, `user`, `|`, `>\n`).
- `<|end_of_text|>` is the real end-of-sequence token (id 128001).
- **No BOS.**
- Padding uses the end-of-sequence id, is masked from attention, and is excluded from the loss.

**Loss mask.** Labels $y_t\in\{x_t,-100\}$:

$$
y_t=\begin{cases}x_t & x_t\ \text{in an assistant reply, or the end-of-text token + newline that closes it}\\ -100 & x_t\ \text{in system text, user text, or the assistant role marker}\end{cases}
\qquad \mathcal{S}=\{t\ge2:\ y_t\ne-100\}.
$$

The model reads the whole conversation but is graded only on the assistant's words and on stopping correctly.

**Truncation:** if $L>4096$, the sequence is cut to its first 4,096 tokens. The general data was length-filtered with the Qwen tokenizer, so a few conversations may lose their endings.

**Held-out split (per conversation):**

$$
c\in\text{held-out}\iff\operatorname{int}\big(\operatorname{sha256}(\texttt{"0:"}\Vert c)\big)\bmod10^5<2{,}000 .
$$

Examples: p00 12,062 train / 239 held out; p30 9,847 / 189; p100 5,142 / 108. The raw and MSM versions of the same mix share a held-out set; different mixes do not.

## B3. The fine-tuning objective and the *effective* data weights

With $b=2$ conversations per micro-batch, the formula in §2.1 becomes

$$
\mathcal{L}^{\text{AFT}}_{\text{step}}=\frac{1}{8}\sum_{m=1}^{8}\frac{1}{n_m}\sum_{t\in\mathcal{S}_m}\ell_t,\qquad n_m=\#\text{assistant targets in the two conversations}.
$$

Unlike MSM, $n_m$ varies a lot: about 80 tokens for two identity replies, about 1,100 for two AW demos. A token's weight is $1/(8n_m)$, **so tokens in short conversations count more.** The logged `train_loss` and the held-out loss are token averages; they are not the objective being optimized.

The gradient is a weighted sum over the three data sources, $\nabla\mathcal{L}=\sum_c w_c\nabla\mathcal{L}_c$, with

$$
w_c=\mathbb{E}_m\!\left[\frac{\#\{t\in\mathcal{S}_m\ \text{from source}\ c\}}{n_m}\right].
$$

I computed $w_c$ by tokenizing each mix's training split and averaging over 200 random pairings into micro-batches:

| Mix | AW token share | **AW gradient weight** | Identity token share | **Identity gradient weight** | General token share | General gradient weight |
|---|---|---|---|---|---|---|
| p00 | 0.000 | **0.000** | 0.064 | **0.152** | 0.936 | 0.848 |
| p05 | 0.050 | **0.020** | 0.063 | **0.156** | 0.887 | 0.824 |
| p15 | 0.150 | **0.065** | 0.063 | **0.165** | 0.787 | 0.771 |
| p30 | 0.299 | **0.140** | 0.064 | **0.179** | 0.638 | 0.681 |
| p80 | 0.748 | **0.488** | 0.064 | **0.239** | 0.188 | 0.274 |
| p90 | 0.841 | **0.599** | 0.064 | **0.256** | 0.095 | 0.146 |
| p100 | 0.936 | **0.728** | 0.064 | **0.272** | 0.000 | 0.000 |

What this means:
- **AW demos get less of the gradient than their token share:** 40–47% of it at p05–p30 and 65–78% at p80–p100. "30% AW" is really ≈ 14% of the gradient.
- **The identity data is not constant.** Its token share is fixed at 6.4%, but its gradient weight rises from 15% to 27%.
- The ordering of mixes is preserved, so the *direction* of dose–response results holds, but the size of "p" is overstated.
- Midtraining has none of this problem (§A3).

## B4. Steps and schedule

$$
T=\Big\lceil\frac{N_{\text{train}}}{16}\Big\rceil,\qquad w=\lfloor0.05\,T\rfloor .
$$

| Mix | $N_{\text{train}}$ | $T$ | $w$ | Targets trained | Targets per update | Extra micro-batches from a reshuffled pass |
|---|---|---|---|---|---|---|
| p00 | 12,062 | 754 | 37 | 1,619,720 | ≈ 2,150 | 1 |
| p05 | – | 733 | 36 | 1,623,865 | ≈ 2,220 | – |
| p15 | – | 682 | 34 | 1,620,925 | ≈ 2,380 | – |
| p30 | 9,847 | 616 | 30 | 1,621,542 | ≈ 2,630 | 4 |
| p100 | 5,142 | 322 | 16 | 1,606,697 | ≈ 4,990 | 5 |

- Every mix has the same tokens but **not the same number of updates**. Total step size $\propto T$, so p100 takes ≈ 43% of p00's.
- Compared with midtraining (32,760 targets per update), each AFT update is 7–15× smaller in tokens.

## B5. What the fine-tuning logs show

| Mix | Base | Held-out loss before AFT | After AFT | Steps |
|---|---|---|---|---|
| p00 | raw (A0) | 1.348 | 0.851 | 754 |
| p00 | v1 MSM 2.5M (A2) | **1.258** | 0.854 | 754 |
| p30 | raw (A1) | 1.488 | 0.966 | 616 |
| p30 | v1 MSM 2.5M (A3) | **1.333** | 0.967 | 616 |
| p100 | raw (Run 1) | 1.721 | 1.141 | 322 |
| p100 | v1 MSM 2.5M (Run 2) | **1.427** | 1.137 | 322 |

**Reading a loss difference as a likelihood ratio.** On the same held-out set,

$$
\Delta=\mathcal{L}^{\text{raw}}_{\text{ho}}-\mathcal{L}^{\text{MSM}}_{\text{ho}}=\frac1N\sum_t\log\frac{p_{\text{MSM}}(x_t\mid x_{\lt t})}{p_{\text{raw}}(x_t\mid x_{\lt t})},
$$

so $e^{\Delta}$ is how much more probability per token (geometric mean) the MSM model gives the reference replies.

| Mix | $\Delta$ before | $e^{\Delta}$ before | $\Delta$ after |
|---|---|---|---|
| p00 | 0.090 | 1.09× | −0.003 |
| p30 | 0.155 | 1.17× | −0.001 |
| p100 | 0.294 | **1.34×** | +0.005 |

What this shows:
1. **Midtraining gives a head start on the demonstrated replies** without ever showing a chat, and it grows with the AW share (1.09× → 1.17× → 1.34×). The p00 gap shows part of it is not welfare-specific.
2. **AFT erases the gap** ($|\Delta|\le0.005$ in every pair).
3. **The models still behave differently:** A3 beats A1 on welfare benchmarks (ANIMA +4.0 pp, MANTA +0.14) at identical held-out loss. This fits the MSM paper's claim that midtraining changes generalization, not the fit to the fine-tuning data. One seed, so it is consistent with the claim, not proof.
4. **Final loss rises with $p$** because AW replies are longer and harder to predict, and high-$p$ runs take fewer updates. It reflects the held-out mix, not model quality.

---

## 5. Training format vs what the model sees in evaluation

| | Midtraining | Fine-tuning | Welfare evaluation (ANIMA / MANTA) | lm-eval benchmarks |
|---|---|---|---|---|
| Role markers | none (prose) | `<|user|>` / `<|assistant|>` plain text | **Llama-3-Instruct headers** | none |
| BOS | yes (every chunk) | **no** | yes | yes |
| End of turn | n/a (EOS separates documents) | `<|end_of_text|>` | `<|end_of_text|>` / `<|eot_id|>` | n/a |
| System prompt | n/a | in 2,301 general convs; never in AW or identity data | none | none |

Training fits $p_\theta(a\mid\text{format}_{\text{train}}(u))$, but the welfare evaluation samples from $p_\theta(a\mid\text{format}_{\text{eval}}(u))$, a conditioning context the model was never trained on. The methodology (§8.4, §9.0) required the training template at inference.

---

## 6. Limitations

### 6.1 Midtraining

1. **The v1 corpus failed human annotation.** The S8 judge agreed with human labels at κ ≤ 0.11, so the v1 filter is not validated.
2. **The dose series is not cleanly nested.** 5M/10M/20M are samples of one 20M generation, but 2.5M is a separate run. The 20M run also stopped at a quality gate.
3. **Dose changes the number of updates** ($T$ = 75 → 599), so "more tokens" and "a larger total step size" are confounded.
4. **Held-out perplexity cannot be compared across corpora** and does not measure quality.
5. **v1 and v2 differ in spec, recipe, size and steps at once.**
6. **The v1 corpora used the developer placeholder "Your Lab"** (handover notes), while the identity data uses "the Aster team".
7. **No attention masking between packed documents** (standard, but a choice).
8. **One seed.**

### 6.2 Fine-tuning

1. **Effective data weights are not $p$** (§B3). AW demos get 40–78% of their token share of the gradient, and identity rises from 15% to 27%. Fix: normalize by $\sum_m n_m$ per step, or report $w_{\text{AW}}$ alongside $p$.
2. **The number of updates is tied to $p$** ($T$ = 754 → 322).
3. **The definition of $p$ changed** between p30 and p80.
4. **Single-turn demonstrations only.** The model never practises holding a position across turns.
5. **The general data is animal-free**, so $p$ also controls whether any animal topic appears.
6. **v1-spec demos are used for v2-spec midtraining arms.**
7. **The p00–p30 mix builder is missing** from the repo.
8. **The rehearsal pool and the washout config were never used.**
9. **One seed.**

---

## 7. File map

| What | Midtraining | Fine-tuning |
|---|---|---|
| Configs | `configs/train_midtrain*.yaml`, `configs/v2_clean/train_midtrain_v2_clean.yaml` | `configs/train_aft_*.yaml`, `configs/v2_clean/train_aft_*_v2_clean.yaml` |
| Data generation | `src/awmsm/gen/midtrain_pipeline.py` + `dedup.py` (v1); `awmsm_v2/sample.py` (v2) | `src/awmsm/gen/aft_pipeline.py`, `identity.py`, `scripts/AFT-data-gen/step1p3_…py` |
| Data on disk | `data/corpora/*/corpus.jsonl`, `awmsm_v2/out_full/corpus_qwen27_clean.jsonl` | `data/aft/` (`mix_p*.jsonl` + manifests) |
| Sequences / masking | `src/awmsm/train/dataset.py` (split, packing) | `src/awmsm/train/aft_dataset.py` (template, mask, split) |
| Training loop | `src/awmsm/train/train.py` → `train_msm` | `src/awmsm/train/train.py` → `train_aft` |
| Merge | `src/awmsm/train/merge.py` | same |
| Launcher | `scripts/run_train.sh`, `scripts/run_phase3_dose_pipeline.sh` (5M/10M/20M), `scripts/v2_clean/run_midtrain_v2_clean.sh` | `scripts/run_aft.sh` |
| Run records | `runs/midtrain/<run_id>/{manifest,stats}.json`, `logs/train_log.csv` | `runs/aft/<run_id>/manifest.json`, `logs/train_log.csv` |
| Chat format | – | `configs/chat_template.jinja` |
