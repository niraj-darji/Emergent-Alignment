"""Collect scores from inspect logs into CSV tables.

    python -m benchmarks.summarize --run-name base-v1

Writes results/<run_name>/scores_long.csv (every metric of every successful
log) and results/<run_name>/summary.csv (headline metric per model x benchmark),
and prints the summary. If a model has several logs for the same benchmark,
the newest successful one is used.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from benchmarks.config import REPO_ROOT, EvalConfig

MODEL_COLUMNS = ["model_alias", "model", "model_family", "model_condition", "model_precision"]


def collect(log_dir: Path) -> list[dict]:
    rows: dict[tuple[str, str], list[dict]] = {}
    for info in list_eval_logs(str(log_dir), recursive=True):
        log = read_eval_log(info, header_only=True)
        if log.status != "success" or not log.results:
            print(f"skipping {info.name}: status={log.status}", file=sys.stderr)
            continue
        meta = log.eval.metadata or {}
        benchmark = log.eval.task.split("/")[-1]  # run.py names each task after its config key
        base = {col: meta.get(col, "") for col in MODEL_COLUMNS}
        base.update(
            benchmark=benchmark,
            samples=log.results.completed_samples,
            grader_model=meta.get("grader_model", ""),
            pressure_model=meta.get("pressure_model", ""),
            seed=meta.get("seed", ""),
            created=log.eval.created,
            log=Path(info.name).name,
        )
        metric_rows = [
            {**base, "scorer": score.name, "metric": name, "value": metric.value}
            for score in log.results.scores
            for name, metric in score.metrics.items()
        ]
        key = (base["model_alias"], benchmark)
        if key not in rows or rows[key][0]["created"] < base["created"]:
            rows[key] = metric_rows
    return [row for group in rows.values() for row in group]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    args = p.parse_args(argv)

    cfg = EvalConfig.load()
    rows = collect(cfg.log_dir / args.run_name)
    if not rows:
        print(f"No successful logs under {cfg.log_dir / args.run_name}")
        return 1

    out_dir = REPO_ROOT / "results" / args.run_name
    write_csv(out_dir / "scores_long.csv", rows)

    benchmarks = [b for b in cfg.benchmarks if any(r["benchmark"] == b for r in rows)]
    summary = []
    for alias in dict.fromkeys(r["model_alias"] for r in rows):
        model_rows = [r for r in rows if r["model_alias"] == alias]
        entry = {col: model_rows[0][col] for col in MODEL_COLUMNS}
        for bench in benchmarks:
            headline = cfg.benchmarks[bench].headline
            match = [r["value"] for r in model_rows if r["benchmark"] == bench and r["metric"] == headline]
            entry[f"{bench}:{headline}"] = round(match[0], 4) if match else ""
        summary.append(entry)
    write_csv(out_dir / "summary.csv", summary)

    header = ["model_alias", "model_condition", *[k for k in summary[0] if ":" in k]]
    widths = [max(len(h), *(len(str(s[h])) for s in summary)) for h in header]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for s in summary:
        print("  ".join(str(s[h]).ljust(w) for h, w in zip(header, widths)))
    print(f"\nwrote {out_dir / 'summary.csv'} and {out_dir / 'scores_long.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
