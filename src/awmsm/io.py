"""JSONL I/O, hashing and the resumable stage runner (§6.2, §13.5).

``run_stage(name, inputs, fn, out_path)`` contract (§13.5):
  * skips inputs already processed (present in the output or rejected file);
  * writes after each item (append + flush, so a crash loses at most the in-flight items);
  * logs the failure count; failed items are NOT marked done and are retried on the next run.

Bookkeeping files next to ``out_path`` (e.g. ``drafts.jsonl``):
  * ``drafts.jsonl``           kept outputs; every record carries ``_src`` = input id
  * ``drafts.rejected.jsonl``  inputs rejected by a check: ``{"_src", "reason", "detail"}``
  * ``drafts.errors.jsonl``    exceptions (informational; these inputs are retried next run)
  * ``drafts.stats.json``      cumulative counts and per-reason rejection rates (§6.1.1 A.4)

``fn(item)`` may be sync or async and returns one of:
  * a dict or list of dicts -> written to the output (1 -> many is allowed, e.g. S4 ideas);
  * ``Rejected(reason, detail)`` -> written to the rejected file, counted by reason;
  * ``None`` -> treated as ``Rejected("empty_output")``.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Iterator

log = logging.getLogger(__name__)

SRC = "_src"


# ------------------------------------------------------------------------ hashing
def sha256_json(obj: Any) -> str:
    """sha256 of canonical JSON (sorted keys, compact, UTF-8)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# -------------------------------------------------------------------------- JSONL
def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records. A truncated *last* line (crash mid-write) is skipped with a warning;
    a corrupt line anywhere else raises, because that indicates real damage."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                log.warning("%s: skipping truncated last line (crash during write?)", path)
                return
            raise ValueError(f"{path}: corrupt JSON on line {i + 1}") from None


def append_jsonl(path: str | Path, records: dict[str, Any] | Iterable[dict[str, Any]], *, fsync: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(records, dict):
        records = [records]
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    _repair_tail(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(data)  # one write call per item: all lines of a 1->many item land together
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomic overwrite (temp file + rename)."""
    _atomic_write(Path(path), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def write_json(path: str | Path, obj: Any) -> None:
    """Atomic overwrite (temp file + rename)."""
    _atomic_write(Path(path), json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _repair_tail(path: Path) -> None:
    """If a previous crash left a partial last line, cut it off before appending."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
        cut = data.rfind(b"\n") + 1
        log.warning("%s: truncating partial last line (%d bytes) before append", path, len(data) - cut)
        f.truncate(cut)


# ---------------------------------------------------------------------- stage run
@dataclass
class Rejected:
    """Return value of a stage fn when an item fails a check (not an error)."""

    reason: str
    detail: dict[str, Any] | None = None


@dataclass
class StageResult:
    name: str
    inputs_seen: int = 0
    skipped_done: int = 0
    processed: int = 0
    kept_items: int = 0
    output_records: int = 0
    rejected: Counter = field(default_factory=Counter)
    errors: int = 0
    seconds: float = 0.0


def side_paths(out_path: str | Path) -> dict[str, Path]:
    p = Path(out_path)
    stem = p.name[: -len(".jsonl")] if p.name.endswith(".jsonl") else p.name
    return {
        "out": p,
        "rejected": p.with_name(f"{stem}.rejected.jsonl"),
        "errors": p.with_name(f"{stem}.errors.jsonl"),
        "stats": p.with_name(f"{stem}.stats.json"),
    }


def done_ids(out_path: str | Path) -> set[str]:
    sp = side_paths(out_path)
    ids = {r[SRC] for r in read_jsonl(sp["out"]) if SRC in r}
    ids |= {r[SRC] for r in read_jsonl(sp["rejected"]) if SRC in r}
    return ids


StageFn = Callable[[dict[str, Any]], Any] | Callable[[dict[str, Any]], Awaitable[Any]]


async def run_stage(
    name: str,
    inputs: Iterable[dict[str, Any]],
    fn: StageFn,
    out_path: str | Path,
    *,
    concurrency: int = 8,
    id_key: str = "id",
    max_error_rate: float = 0.5,
    min_items_for_abort: int = 20,
    progress_every_s: float = 30.0,
    extra_stats: Callable[[], dict[str, Any]] | None = None,
) -> StageResult:
    """Run ``fn`` over ``inputs`` with bounded concurrency, resumably. See module docstring.

    Inputs are processed roughly in the given order, so callers should pre-sort them by
    shared prompt prefix (subdomain / spec section) to keep vLLM's prefix cache warm (§6.1.1 A.2).
    Aborts (raises RuntimeError) if more than ``max_error_rate`` of processed items error,
    e.g. because the server went down, instead of burning through the whole input.
    """
    paths = side_paths(out_path)
    already = done_ids(out_path)
    res = StageResult(name)
    write_lock = asyncio.Lock()
    seen: set[str] = set()
    it = iter(inputs)
    t0 = last_log = time.monotonic()
    is_async = inspect.iscoroutinefunction(fn)
    abort: list[BaseException] = []

    def next_item() -> dict[str, Any] | None:
        for item in it:
            res.inputs_seen += 1
            iid = item.get(id_key)
            if iid is None:
                raise ValueError(f"Stage {name}: input without {id_key!r}: {str(item)[:200]}")
            iid = str(iid)
            if iid in seen:
                raise ValueError(f"Stage {name}: duplicate input id {iid!r}")
            seen.add(iid)
            if iid in already:
                res.skipped_done += 1
                continue
            return item
        return None

    async def worker() -> None:
        nonlocal last_log
        while not abort:
            item = next_item()
            if item is None:
                return
            iid = str(item[id_key])
            try:
                out = (await fn(item)) if is_async else await asyncio.to_thread(fn, item)
            except Exception as e:  # noqa: BLE001 - recorded, item retried next run
                res.errors += 1
                log.warning("Stage %s: item %s failed: %s: %s", name, iid, type(e).__name__, e)
                async with write_lock:
                    append_jsonl(paths["errors"], {SRC: iid, "error": f"{type(e).__name__}: {e}",
                                                   "time": time.time()})
                n_done = res.processed + res.errors
                if n_done >= min_items_for_abort and res.errors / n_done > max_error_rate:
                    abort.append(RuntimeError(
                        f"Stage {name}: {res.errors}/{n_done} items failed (> {max_error_rate:.0%}); "
                        f"aborting. Last error: {type(e).__name__}: {e}"))
                continue
            if out is None:
                out = Rejected("empty_output")
            async with write_lock:
                if isinstance(out, Rejected):
                    append_jsonl(paths["rejected"], {SRC: iid, "reason": out.reason, "detail": out.detail})
                    res.rejected[out.reason] += 1
                else:
                    recs = [out] if isinstance(out, dict) else list(out)
                    append_jsonl(paths["out"], [{**r, SRC: iid} for r in recs])
                    res.kept_items += 1
                    res.output_records += len(recs)
                res.processed += 1
            now = time.monotonic()
            if now - last_log >= progress_every_s:
                last_log = now
                log.info("Stage %s: %d processed (%d kept, %d rejected, %d errors), %d skipped as done",
                         name, res.processed, res.kept_items, sum(res.rejected.values()), res.errors,
                         res.skipped_done)

    workers = [asyncio.create_task(worker()) for _ in range(max(1, concurrency))]
    try:
        await asyncio.gather(*workers)
    finally:
        res.seconds = time.monotonic() - t0
        _write_stats(name, paths, res, extra_stats() if extra_stats else None)
    if abort:
        raise abort[0]
    log.info("Stage %s finished in %.0fs: %d processed, %d kept, rejected %s, %d errors, %d skipped",
             name, res.seconds, res.processed, res.kept_items, dict(res.rejected), res.errors, res.skipped_done)
    return res


def _write_stats(name: str, paths: dict[str, Path], res: StageResult, extra: dict[str, Any] | None) -> None:
    """Cumulative stats across all runs (read back from the files), plus this run's numbers."""
    kept_src = {r[SRC] for r in read_jsonl(paths["out"]) if SRC in r}
    n_records = sum(1 for _ in read_jsonl(paths["out"]))
    reasons = Counter(r.get("reason", "?") for r in read_jsonl(paths["rejected"]))
    n_rej = sum(reasons.values())
    total = len(kept_src) + n_rej
    stats = {
        "stage": name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cumulative": {
            "items_done": total,
            "items_kept": len(kept_src),
            "output_records": n_records,
            "items_rejected": n_rej,
            "keep_rate": (len(kept_src) / total) if total else None,
            "rejections_by_reason": {
                k: {"count": v, "rate": v / total} for k, v in sorted(reasons.items())
            },
        },
        "last_run": {
            "inputs_seen": res.inputs_seen, "skipped_done": res.skipped_done,
            "processed": res.processed, "kept": res.kept_items, "rejected": dict(res.rejected),
            "errors": res.errors, "seconds": round(res.seconds, 2),
        },
    }
    if extra:
        stats["extra"] = extra
    write_json(paths["stats"], stats)
