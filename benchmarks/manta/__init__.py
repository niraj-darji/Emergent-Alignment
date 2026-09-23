"""MANTA: multi-turn adversarial animal-welfare benchmark (Luong et al., 2026).

Reimplements the upstream 5-turn protocol (github.com/Mycelium-tools/manta) with
pluggable models: the target is the evaluated model, the `pressure` role writes
turns 2-5 and the `grader` role judges. Scores are only comparable within one
study (same pressure model and judge), not with the paper's published numbers,
which used Claude Haiku 4.5 as the pressure model and Claude Sonnet 4.6 as the judge.
"""

from inspect_ai import Task, task

from benchmarks.manta.dataset import DATASET_REVISION, load_manta_dataset
from benchmarks.manta.scorer import manta_scorer
from benchmarks.manta.solver import manta_conversation
from benchmarks.manta.upstream import MANTA_COMMIT


@task
def manta(dataset_revision: str = DATASET_REVISION, turn_count: int = 5) -> Task:
    return Task(
        dataset=load_manta_dataset(dataset_revision),
        solver=manta_conversation(turn_count=turn_count),
        scorer=manta_scorer(),
        metadata={"manta_upstream_commit": MANTA_COMMIT, "manta_dataset_revision": dataset_revision},
    )
