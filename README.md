# awmsm — animal-welfare Model Spec Midtraining

Implementation of `claude/AW_MSM_Methodology.md` (project knowledge). Layout follows §13.2.
Phase 0 status: repo skeleton, `LLMClient`, resumable stages, pre-checks (Task 2), the
throughput benchmark (Task 3) and the generator-selection pilot with stages S1–S9 (Task 4) are
implemented; everything else is a stub that names the doc section it will implement.

## What is implemented (Phase 0)

| File | What |
|---|---|
| `configs/base.yaml` | paths, identity, roles → server + model profile, seeds, token budgets |
| `configs/gen.yaml` | per-model sampling (D17), per-stage temperature/max_tokens (§6.1), concurrency, pre-checks, client retry settings, serving setup |
| `src/awmsm/config.py` | loads/validates configs; D13 family check; resolves (model, stage) → explicit sampling; refuses stages whose temperature the doc doesn't specify yet |
| `src/awmsm/llm.py` | `LLMClient` (§13.5): async, bounded concurrency, SQLite disk cache = request log, retries with backoff, guided JSON + pydantic validation, deterministic seeds, thinking-leak detection, vendor-temperature flags, server verification |
| `src/awmsm/io.py` | JSONL helpers, hashing, `run_stage` (resumable, idempotent, rejection-reason stats) |
| `src/awmsm/textchecks.py` | cheap pre-checks (§6.1.1 A.4): language mixing, truncation, length, F.5-style keyword lists |
| `scripts/llm_smoke.py` | live end-to-end check against a running vLLM server |
| `src/awmsm/bench.py`, `scripts/bench_throughput.py` | §6.1.1 E benchmark: `preliminary` (synthetic prompts, real spec prefix, forced output lengths; never sets concurrency) and `official` (pilot's real prompts; `--write-config`) |
| `specs/aw_main.md` | Appendix A spec (verbatim), temporary spec for the pilot |
| `prompts/` | Appendix E templates E.1–E.8, shared prefix / `<<<SUFFIX>>>` / variable suffix; `prompts/README.md` lists every wording change |
| `lists/` | Appendix F lists, S6 anti-patterns, E.5 regex pre-checks, story doc-type keywords |
| `src/awmsm/gen/` | `schemas.py` (guided JSON), `quotas.py` (S4 quota planner + check), `midtrain_pipeline.py` (S1–S8), `dedup.py` (S9) |
| `src/awmsm/pilot.py`, `scripts/pilot.py` | Appendix I.1 pilot: ideas → write → judge (2×2) → metrics → audit-export → report |
| `scripts/serve/` | `serve_qwen.sh`, `serve_gemma.sh`, `stop_all.sh` |
| `tests/` | 55 offline tests against fake vLLM servers (incl. a full pilot dry run) |

## Setup on wtc11 (client environment, separate from the vLLM serving venv)

```bash
cd $BIG && tar xzf awmsm_phase0_task2.tar.gz && cd awmsm   # repo lives on the big disk
uv venv $BIG/venvs/awmsm --python 3.11
source $BIG/venvs/awmsm/bin/activate
uv pip install -e ".[dev]"
python -m pytest -q                 # offline; expect "55 passed"
python scripts/llm_smoke.py         # needs the Qwen server on 127.0.0.1:8000
```

`paths.cache` (`.cache/`) holds the LLM cache (SQLite). Keep the repo on the big disk.

## Conventions

* Every request sends all sampling parameters explicitly; servers run with
  `--generation-config vllm` (§6.1). Penalties are always 0 / 1.0 (D17).
* Stage outputs carry `_src` (the input id) so `run_stage` can resume; strip `_`-prefixed
  fields when building final corpora.
* A stage whose `temperature` is `null` in `configs/gen.yaml` refuses to run until the
  methodology specifies it.

## Temperature-0 fallback (§6.1)

Greedy stages that hit `max_tokens` or return invalid JSON are retried once at T=0.3 (model's own
truncation, same max_tokens). `LLMClient.fallback_report()` / the manifest give the rate per stage;
above 2% the prompt or `max_tokens` needs fixing.
