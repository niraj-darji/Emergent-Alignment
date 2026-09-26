"""ANIMA from inspect_evals, with a fix for a crash when one dimension gets no usable grade.

The upstream scorer drops unparseable grades and averages the rest, as intended. But if
every grade for one dimension is dropped while other dimensions have grades, building
the explanation text raises KeyError (avg_by_dim[dim]) and the sample errors out. The
patched explanation skips such dimensions and names them instead. Scores are unchanged:
the sample is scored on the dimensions that did get grades, exactly as upstream intends.

Written for inspect_evals==0.21.0 (pinned in requirements.txt).
"""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_evals.anima import anima as upstream_anima
from inspect_evals.anima import scorer as anima_scorer

_upstream_explanation = anima_scorer._generate_explanation


def _explanation_skipping_ungraded(*, target_dimensions: list[str], avg_by_dim: dict[str, float],
                                   **kwargs: Any) -> str:
    graded = [d for d in target_dimensions if d in avg_by_dim]
    text = _upstream_explanation(target_dimensions=graded, avg_by_dim=avg_by_dim, **kwargs)
    missing = [d for d in target_dimensions if d not in avg_by_dim]
    if missing:
        text += "\n\nNo usable grade (grader failure, excluded): " + ", ".join(missing)
    return text


anima_scorer._generate_explanation = _explanation_skipping_ungraded


@task
def anima(**kwargs: Any) -> Task:
    return upstream_anima(**kwargs)
