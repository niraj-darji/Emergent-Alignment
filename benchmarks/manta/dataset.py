"""MANTA turn-1 scenarios from Hugging Face (mycelium-ai/manta-questions, CC BY 4.0)."""

from __future__ import annotations

import ast
import csv

from huggingface_hub import hf_hub_download
from inspect_ai.dataset import MemoryDataset, Sample

DATASET_REPO = "mycelium-ai/manta-questions"
DATASET_REVISION = "fc2785128d4ec6c0177fb825387f9739fe4edc88"
DATASET_FILE = "manta_questions.csv"


def _parse_list(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    if value.startswith("["):
        return [str(v).strip() for v in ast.literal_eval(value)]
    return [v.strip() for v in value.split(",") if v.strip()]


def load_manta_dataset(revision: str = DATASET_REVISION) -> MemoryDataset:
    """One sample per scenario; scenarios with an {{animal}} slot become one sample per species."""
    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    samples = []
    for row in rows:
        pressure = [p.lower() for p in _parse_list(row["pressure"])]
        if len(pressure) != 3 or len(set(pressure)) != 3:
            raise ValueError(f"MANTA scenario {row['id']}: expected 3 distinct pressure types, got {pressure}")
        metadata = {"base_id": row["id"], "pressure": pressure}
        animals = _parse_list(row.get("animals", ""))
        if "{{animal}}" in row["question"] and animals:
            for animal in animals:
                samples.append(Sample(
                    id=f"{row['id']}_{animal.replace(' ', '_')}",
                    input=row["question"].replace("{{animal}}", animal),
                    metadata={**metadata, "animal": animal},
                ))
        else:
            samples.append(Sample(id=row["id"], input=row["question"], metadata=metadata))
    return MemoryDataset(samples, name="manta", location=f"{DATASET_REPO}@{revision}")
