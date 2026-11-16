
## `aw_main` v1 20M run (3–5 Nov 2026; Niraj's decision)

- **Why:** Jevin's fine-tune on the 2.5M v1 corpus gave results comparable with the CaML paper at matched tokens. The aim is to scale the same recipe and test whether the v1.1–v1.3 judging and rewriting were too harsh.
- **Recipe:** exactly v1 (git 4f5f607, tag `aw_main-v1`), in a separate worktree `$BIG/awmsm_v1`; the main repo is unchanged.
  - The spec is identical to the v1 run's `spec_used.md`. The S8 template is E08_filter_v3. Generator and judge are Qwen3.8-27B-FP8.
- **Config changes in the worktree only:**
  - `seeds.generation` 0 → 1 and `corpus.seed` 20261013 → 20261103, so ideas and requests do not repeat the v1 run;
  - `corpus.max_batches` 40 → 80;
  - `corpus.s8_unresolved: exclude`, as used in the v1 run (22 Oct decision, not committed at 4f5f607).
  - All other corpus settings match the v1 run's `manifest.json`.
- **Output:** a separate corpus `awmsm_v1/data/corpora/aw_main_v1_20m`. `aw_main_v1` (2.5M) is untouched and not mixed in.

## Run log

- **3 Nov, 01:37:** started. At 06:51 it halted at batch_06 S6 on 2 unresolved items.
- **Decision (Niraj, 3 Nov): S6/S7 items whose output is still invalid after the T=0 fallback are excluded and recorded.** This extends the 22 Oct S8 rule.
  - `exclude_unresolved.py` acts only on halts consisting purely of unresolved items, with backups.
  - Each exclusion is logged in `aw_main_v1_20m/unresolved_excluded.jsonl`.
  - Any other halt still stops the run.
- **Shared GPU from 3 Nov:** another user's job was on wtc11, so Qwen was served with `serve_qwen_shared.sh` (`--gpu-memory-utilization 0.75`, `--max-num-seqs 64`; original settings 0.92 / 256). This affects throughput only, apart from floating-point batching differences.
- **Server crash, 3 Nov 21:47:** "CUDA launch timed out" (error 702), as on 19 Oct; possibly the display (Xorg) watchdog on the shared GPU.
  - From 4 Nov the run is supervised by `awmsm_v1/run_20m.sh`, which keeps the crash log, restarts our server and resumes.
- **5 Nov, 18:09: corpus written. 16,684 documents, 20,000,827 tokens** (exact, `meta-llama/Llama-3.1-8B`), from 50 complete batches.
  - The run then halted in batch_51 at S6, with a T=0 fallback rate of 2.2% (limit 2%). Batch 51 is not used.
- **Quotas:** all pass except **stories 14.7% (minimum 15%)**.
  - **Decision (Niraj, 5 Nov): accepted as is, recorded as a deviation.** It is 0.3 pp short, about 50 stories.
- **Differences from the 2.5M v1 corpus:**
  - The §1-phrase deprioritising trim (24 Oct) was not applied. The corpus is the S12 output of the v1 code.
  - The S6/S7 exclusions above.
  - Stories 14.7% instead of 16.0%.
  - Generated on a shared GPU.
- **Near-duplicate check against `aw_main_v1` (5 Nov; 13-gram share of each new document found in the 2.5M corpus):**
  - 0 exact duplicates; **0 near-duplicates** at the S9 threshold (≥ 30%).
  - Distribution over the 16,684 new documents: 0% in 5,162; under 1% in 4,956; 1–10% in 6,357; 10–30% in 209.
  - The small overlaps are consistent with recited spec sentences, which both corpora share.
- **Status: the 20M corpus is final and ready for training** (`awmsm_v1/data/corpora/aw_main_v1_20m/corpus.jsonl`).
- **Methodology note:** 20M is above the 5M midtraining budget, and this runs ahead of the pending v1.3 decisions. It is a researcher decision.
