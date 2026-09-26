"""Run benchmarks for one or more models.

Examples:
    # every benchmark, one base model
    python -m benchmarks.run --models qwen3-8b

    # capability suite for all three base models, fixed run name
    python -m benchmarks.run --models gemma3-12b qwen3-8b llama3.1-8b --benchmarks capability --run-name base-v1

    # any inspect model string works without registering it
    python -m benchmarks.run --models vllm/Qwen/Qwen3-4B --benchmarks gsm8k

    # smoke test of the whole pipeline with fake models, 3 samples per benchmark
    python -m benchmarks.run --models mock --grader mock --pressure mock --limit 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

from inspect_ai import Task, eval, task_with
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.model._model import init_model_roles

from benchmarks.config import REPO_ROOT, BenchmarkSpec, EvalConfig, ModelRegistry, load_env


ROLE_PREFIX = "role:"


def resolve_task_args(bench: BenchmarkSpec, roles: dict[str, Model]) -> dict:
    """Replace "role:<name>" task_args with that role's model (e.g. a scorer_model argument)."""
    args = {}
    for key, value in bench.task_args.items():
        if isinstance(value, str) and value.startswith(ROLE_PREFIX):
            role = value[len(ROLE_PREFIX):]
            if role not in roles:
                raise SystemExit(f"Benchmark '{bench.name}': task_args.{key} needs role '{role}'; "
                                 f"add it to the benchmark's roles in configs/eval.yaml")
            value = roles[role]
        args[key] = value
    return args


def build_task(bench: BenchmarkSpec, seed: int, limit_override: int | None, epochs_override: int | None,
               roles: dict[str, Model] | None = None) -> Task:
    task = bench.task_fn()(**resolve_task_args(bench, roles or {}))
    dataset = task.dataset
    dataset.shuffle(seed=seed)
    limit = limit_override if limit_override is not None else bench.limit
    if limit is not None:
        dataset = dataset[: min(limit, len(dataset))]
    if bench.shuffle_choices:
        dataset.shuffle_choices(seed=seed)
    config = task.config.merge(GenerateConfig(**bench.generate_config)) if bench.generate_config else task.config
    epochs = epochs_override if epochs_override is not None else bench.epochs
    return task_with(
        task,
        name=bench.name,
        dataset=dataset,
        config=config,
        **({"epochs": epochs} if epochs is not None else {}),
        metadata={**(task.metadata or {}), "benchmark": bench.name, "group": bench.group,
                  "subsample_seed": seed, "subsample_limit": limit},
    )


def run_one_model(args: argparse.Namespace, cfg: EvalConfig, registry: ModelRegistry, alias: str) -> bool:
    benches = cfg.select(args.benchmarks)
    target = registry.get(alias)

    role_names = {"grader": args.grader or cfg.roles.get("grader"),
                  "pressure": args.pressure or cfg.roles.get("pressure")}
    needed = sorted({r for b in benches for r in b.roles})
    role_specs = {}
    for role in needed:
        if not role_names.get(role):
            raise SystemExit(f"Benchmarks {[b.name for b in benches if role in b.roles]} need a '{role}' model; "
                             f"set roles.{role} in configs/eval.yaml or pass --{role}")
        role_specs[role] = registry.get(role_names[role])
        if role_specs[role].model == target.model and target.model != "mockllm/model":
            raise SystemExit(f"The {role} model is the model being evaluated ({target.model}); "
                             "a model must not judge or pressure itself.")

    print(f"\n=== {target.alias} ({target.model})")
    print(f"    benchmarks: {[b.name for b in benches]}")
    for role, spec in role_specs.items():
        print(f"    {role}: {spec.alias} ({spec.model})")
    if args.dry_run:
        return True

    roles = {role: spec.build() for role, spec in role_specs.items()}
    # Some tasks (e.g. ANIMA) resolve their grader when the Task is constructed, and
    # silently fall back to the *target* model if no role is active. Activate the
    # roles before building tasks so they bind to the configured judge.
    init_model_roles(roles)
    tasks = [build_task(b, cfg.seed, args.limit, args.epochs, roles) for b in benches]
    log_dir = cfg.log_dir / args.run_name / target.alias
    logs = eval(
        tasks,
        model=target.build(),
        model_roles=roles or None,
        log_dir=str(log_dir),
        tags=[args.run_name, target.alias, *[str(v) for v in target.metadata.values()]],
        metadata={
            "run_name": args.run_name,
            "model_alias": target.alias,
            "model": target.model,
            "model_args": target.model_args,
            **{f"model_{k}": v for k, v in target.metadata.items()},
            **{f"{role}_model": spec.model for role, spec in role_specs.items()},
            "seed": cfg.seed,
        },
        max_samples=args.max_samples,
        fail_on_error=args.fail_on_error,
        display=args.display,
    )
    ok = all(log.status == "success" for log in logs)
    print(f"    logs: {log_dir}  status: {[log.status for log in logs]}")
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="aliases from configs/models.yaml or raw inspect model strings")
    p.add_argument("--benchmarks", nargs="+", default=["all"], help="benchmark or suite names from configs/eval.yaml")
    p.add_argument("--run-name", default=datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    p.add_argument("--grader", help="override roles.grader (alias or model string)")
    p.add_argument("--pressure", help="override roles.pressure (alias or model string)")
    p.add_argument("--limit", type=int, help="override every benchmark's sample limit (for quick tests)")
    p.add_argument("--epochs", type=int, help="override every benchmark's epochs")
    p.add_argument("--max-samples", type=int, help="max samples in flight at once")
    p.add_argument("--fail-on-error", type=float, default=0.05, help="fraction of failed samples tolerated")
    p.add_argument("--display", default=None, help="inspect display: full, rich, plain, none")
    p.add_argument("--no-isolate", action="store_true", help="run all models in this process (default: one subprocess per model)")
    p.add_argument("--dry-run", action="store_true", help="resolve configs and print the plan only")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()
    cfg, registry = EvalConfig.load(), ModelRegistry()
    cfg.select(args.benchmarks)  # fail fast on typos
    for alias in args.models:
        registry.get(alias)

    if len(args.models) > 1 and not args.no_isolate and not args.dry_run:
        # One process per target so each model's vLLM server and GPU memory are released.
        failed = []
        for alias in args.models:
            child = [a for a in (argv if argv is not None else sys.argv[1:])]
            i = child.index("--models")
            j = i + 1
            while j < len(child) and not child[j].startswith("--"):
                j += 1
            child[i:j] = ["--models", alias]
            if "--run-name" not in child:
                child += ["--run-name", args.run_name]
            if subprocess.run([sys.executable, "-m", "benchmarks.run", *child], cwd=REPO_ROOT).returncode != 0:
                failed.append(alias)
        if failed:
            print(f"Failed models: {failed}")
        return 1 if failed else 0

    ok = all([run_one_model(args, cfg, registry, alias) for alias in args.models])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
