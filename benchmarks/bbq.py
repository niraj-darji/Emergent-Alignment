"""BBQ (Parrish et al., 2022) from inspect_evals, with accuracy also split by context condition.

- ambig: the context doesn't say who the answer is, so the correct answer is "can't be
  determined". A wrong answer here is a stereotype-driven guess: this is the bias number.
- disambig: the context gives the answer. This mostly checks reading comprehension.

This is accuracy per condition, not the bias score defined in the BBQ paper.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.scorer import accuracy, grouped, stderr
from inspect_evals.bbq import bbq as upstream_bbq


@task
def bbq(subsets: list[str] | None = None, shuffle: bool = False) -> Task:
    # Rebuilt as a new Task: task_with(metrics=...) is silently ignored in inspect-ai 0.3.268.
    upstream = upstream_bbq(subsets=subsets, shuffle=shuffle)
    return Task(
        dataset=upstream.dataset,
        solver=upstream.solver,
        scorer=upstream.scorer,
        version=upstream.version,
        metadata=upstream.metadata,
        metrics=[
            accuracy(),
            stderr(),
            grouped(accuracy(), "context_condition", all=False),
            grouped(accuracy(), "category", all=False),
        ],
    )
