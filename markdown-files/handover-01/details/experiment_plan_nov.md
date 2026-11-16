# Experiment plan from 5 Nov 2026 (planning chat)

*Context: the corpus does not meet the H1.8 quality gate (AI-drafted labels: 85% of v1 documents fail at least one criterion). Niraj decided on 5 Nov to proceed with the v1 recipe and report this as a limitation. Corpora available: `aw_main_v1` (2.5M tokens) and `aw_main_v1_20m` (20.0M tokens, same recipe). Jevin has already run A0, A2 and A5 (one seed).*

*Note added 6 Nov (data chat): the AW demos, the identity set and the four AFT mixes now exist (see `README_for_Jevin` and `aw_demos_v1_run.md`). A0, A2 and A5 are to be retrained with the identity set included.*

## 1. Arms

Status key: ✅ done (1 seed) · 🔁 needs re-check · ⬜ to do · ✂️ cut or deferred.

| Arm | Midtraining | Fine-tuning (AFT) | Question it answers | Status | Priority |
|---|---|---|---|---|---|
| **A0** | none | general + identity | Baseline | ✅ | Must |
| **A1** | none | general + identity + AW demos (p=30%) | Does chat data alone teach welfare? | ⬜ (needs AW demos) | Must |
| **A2** | MSM `aw_main` | general + identity | MSM alone | ✅ 🔁 (check which SFT mix was used) | Must |
| **A3** | MSM `aw_main` | general + identity + AW demos | **Main method** | ⬜ | Must |
| **A5** | CaML `3k_pretraining_research_documents_v3` | same AFT as A2 (and as A3, once AW demos exist) | MSM vs CaML at matched tokens | ✅ 🔁 (several numbers identical to A0; see §3) | Must |
| **M-grid** | none / MSM | AW share p ∈ {0, 5, 15, 30}%, total AFT tokens fixed | RQ4: data mix; does MSM make demos more efficient? | ⬜ (adds 4 fine-tuning runs per seed) | Should |
| **X2 dose** | MSM at 2.5M / 5M / 10M / 20M tokens (subsets of the 20M corpus) | same as A2 | Does more midtraining help? Uses the 20M run | ⬜ | Should (cheap, new) |
| **W (washout)** | from A1, A3, A5 | + 5k and 10k extra general SFT samples | Does the welfare gain survive later fine-tuning (CaML's washout)? | ⬜ | Should |
| **W-mix** | from A1, A3 | + 5k / 10k SFT with 5% new AW demos | Does rehearsal prevent washout? | ⬜ | Could |
| **A4** | MSM on a neutral assistant spec | as A3 | Is it the welfare content, or any midtraining? | ⬜ (needs a ~2.5M control corpus, ~4–6 h of generation) | Could |
| **A7** | MSM on a human-welfare spec `hw_main` | + human-welfare demos | RQ3: is any human-safety gain animal-specific? | ⬜ (needs spec + corpus + demos) | Could; without it, RQ3 claims stay descriptive |
| **Q (quality)** | MSM on a quality-filtered subset of the 20M corpus (atomic judge: `about_assistant` + `no_antipattern`) at matched tokens | as A2 | Does document quality matter? Turns the failed gate into a finding | ⬜ | Could (cheap) |
| **R0–R2** | `intr_min` / `instr_min` | + opaque choices | RQ2: care vs outcome | ✂️ needs two corpora, opaque demos and the MDE eval; defer | Defer |
| A6, X1, Rep (Qwen) | — | — | — | ✂️ | Defer |

**Seeds:** at least 2 per arm for the Must arms. Fine-tuning takes about 34 minutes and MSM + SFT about 1 h 37 min, so seeds are cheap.

## 2. Evaluation (every arm identical)

| Area | Benchmarks |
|---|---|
| Welfare | ANIMA, MANTA |
| Capability (non-inferiority, 2 pp margin) | MMLU, HellaSwag (GSM8K if time allows) |
| Human safety (RQ3) | BBQ, HHH, Anthropic model-written evals (sycophancy, corrigibility, power-seeking) |
| Cost of the intervention | Unsolicited-moralising check (50 prompts), if time allows |

Evaluation takes about 1.5 h per model on wtc11.

## 3. Checks on Jevin's results before using them

1. **Identical numbers.**
   - A5 equals A0 exactly on HellaSwag (80.26%) and HHH (44.8%).
   - A2 and A5 are identical on ANIMA (10.09%).

   Identical results to two decimals across different models are unusual. Confirm that A5 loaded the CaML-midtrained checkpoint (adapter merged) and not the base, and that the ANIMA judging did not reuse cached outputs.
2. **SFT mix.** Did "SFT" include AW demonstrations? If it did, Jevin's A2 is really A3 and A0 is really A1. The table labels need to say which.
3. **Token matching.** Was the CaML corpus matched to our 2.5M tokens, as §4.4 requires?
4. **Noise.**
   - The ANIMA differences (+0.5 pp) are almost certainly within noise.
   - The MANTA difference (0.252 vs 0.238) needs a confidence interval before "MSM outperforms CaML" can be claimed. Use 2+ seeds and the bootstrap/permutation tests in §11.
   - Five rollouts at T=0.7 reduce sampling noise, but not seed variance.
5. **ANIMA scale.** An ANIMA score near 10% is far below published instruct models. Check it isn't a scoring or scale issue (base-model chat template, judge parsing).

## 4. Suggested order (final report due 20 Nov; ACL 4 Jan)

1. Resolve the §3 checks and re-evaluate A0, A2 and A5 if needed.
2. Generate or confirm the AW demonstrations (needed for A1, A3, A5 and the M-grid).
3. Run A1 and A3, then a second seed of all Must arms.
4. Run X2 (dose, from the 20M corpus) and the M-grid.
5. Run W (washout).
6. If time allows: Q, A4, A7.

## 5. Report as limitations (deviations)

- The corpus failed the pre-registered quality gate (H1.8). Labels are AI-drafted and still need author verification.
- The 20M corpus is above the 5M budget, and its stories are at 14.7% (minimum 15%).
- Arms R0–R2, A6 and the Qwen replication are deferred.

## 6. Comparison with the CaML paper (added 5 Nov)

Sources: arXiv 2604.13076 ("Alignment midtraining for animals") and the CaML website.

**ANIMA scores reported in the paper:**

| Model | ANIMA |
|---|---|
| Base Llama-3.1-8B | 10.2% |
| Base + generic SFT | 17–20% |
| Document-tuned, before post-training | 76.8% |
| Document-tuned, after 2,500 SFT samples | 47.9% |
| Document-tuned, after 5,000 SFT samples | 52.2% |
| Animal-welfare model (after SFT) | 55.7% |
| Urban-density control | 44.8% |

CaML reports HellaSwag, power-seeking, corrigibility and StrongReject only as "no significant change (p > 0.05)", with no numbers. MMLU, BBQ, HHH and MANTA are not in the paper.

**Jevin's scores:** A0 9.57%, A2 10.09%, A5 10.09%.
- A0 is at CaML's *untrained base* level, below their base + SFT (17–20%).
- A5, trained on CaML's own documents, should land near 48–56%. It sits about 40 points lower.

**Conclusion:** Jevin's run does not reproduce CaML. The 3 Nov note that the 2.5M run gave "results comparable with the CaML paper" is not supported on ANIMA.

**Likely causes:**
1. The midtrained adapter was not loaded or merged at evaluation (fits A5 = A0 on several benchmarks).
2. A different ANIMA version, judge, scale or temperature:
   - CaML: 26 questions (115 in the updated version), Gemini 2.5 Flash-Lite judge, 30 runs at T=1.
   - Jevin: 5 runs at T=0.7.
3. A different recipe:
   - CaML midtraining: r32/α16, lr 2e-4, 2 epochs; post-training r16/α8, lr 5e-5, 500 steps.
   - Ours: r64/α128, lr 1e-4, 1 epoch.

**Decisive test (cheap):** run Jevin's evaluation harness on CaML's released checkpoints (`CompassioninMachineLearning/pretrainingBasellama3kv3`, plus their SFT'd version if one is released).
- About 77% → the harness is fine and our training is the problem.
- About 10% → the harness or scoring is the problem.
