"""Unit tests for Stage 2 Alignment Fine-Tuning (AFT) dataset, collator, and masking.

Reference: AW_MSM_Methodology.md (§8.4) and AFT_training_blueprint.md.
"""

import json
from pathlib import Path
import pytest
import torch
from jinja2 import Template

from awmsm.train.aft_dataset import AFTCollator, AFTDataset, get_aft_datasets


class DummyTokenizer:
    """Mock tokenizer for fast, isolated unit testing."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._vocab = {"<|end_of_text|>": 1}

    def encode(self, text: str, add_special_tokens: bool = False):
        tokens = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 2
            tokens.append(self._vocab[word])
        return tokens

    def decode(self, token_ids):
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(tid, f"tok_{tid}") for tid in token_ids)


@pytest.fixture
def dummy_template_path(tmp_path):
    template_file = tmp_path / "chat_template.jinja"
    template_content = """{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
<|system|>
{{ message['content'] }}
{% elif message['role'] == 'user' -%}
<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' -%}
<|assistant|>
{{ message['content'] }}<|end_of_text|>
{% endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|assistant|>
{% endif -%}"""
    template_file.write_text(template_content, encoding="utf-8")
    return template_file


def test_aft_loss_masking(dummy_template_path):
    tokenizer = DummyTokenizer()
    samples = [
        {
            "id": "sample_1",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital is Paris."},
            ],
        }
    ]

    dataset = AFTDataset(
        samples=samples,
        tokenizer=tokenizer,
        chat_template_path=dummy_template_path,
        max_seq_len=4096,
    )

    assert len(dataset) == 1
    item = dataset[0]
    input_ids = item["input_ids"]
    labels = item["labels"]

    assert len(input_ids) == len(labels)
    # Ensure there are both masked tokens (-100) and supervised target tokens
    masked_count = sum(1 for l in labels if l == -100)
    supervised_count = sum(1 for l in labels if l != -100)

    assert masked_count > 0, "Prompt tokens must be masked with -100"
    assert supervised_count > 0, "Assistant response tokens must have valid active labels"

    # Verify that supervised tokens match input_ids
    for in_id, lbl in zip(input_ids, labels):
        if lbl != -100:
            assert in_id == lbl


def test_aft_collator_dynamic_padding():
    collator = AFTCollator(pad_token_id=0, pad_to_multiple_of=8)
    batch = [
        {"input_ids": [10, 20, 30], "labels": [-100, 20, 30]},
        {"input_ids": [10, 20, 30, 40, 50], "labels": [-100, -100, 30, 40, 50]},
    ]

    collated = collator(batch)
    input_ids = collated["input_ids"]
    attention_mask = collated["attention_mask"]
    labels = collated["labels"]

    assert input_ids.shape == (2, 8), "Expected batch to be padded to multiple of 8"
    assert attention_mask.shape == (2, 8)
    assert labels.shape == (2, 8)

    # First row padded length is 5 tokens
    assert (input_ids[0, 3:] == 0).all()
    assert (attention_mask[0, 3:] == 0).all()
    assert (labels[0, 3:] == -100).all()


def test_aft_dataset_truncation(dummy_template_path):
    tokenizer = DummyTokenizer()
    long_content = "word " * 100
    samples = [
        {
            "id": "long_sample",
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": long_content},
            ],
        }
    ]

    max_len = 25
    dataset = AFTDataset(
        samples=samples,
        tokenizer=tokenizer,
        chat_template_path=dummy_template_path,
        max_seq_len=max_len,
    )

    item = dataset[0]
    assert len(item["input_ids"]) == max_len
    assert len(item["labels"]) == max_len


def test_deterministic_split_disjoint(tmp_path, dummy_template_path):
    jsonl_file = tmp_path / "test_sft.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for i in range(100):
            sample = {
                "id": f"doc_{i:03d}",
                "messages": [
                    {"role": "user", "content": f"Query {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ],
            }
            f.write(json.dumps(sample) + "\n")

    tokenizer = DummyTokenizer()
    config = {
        "data": {
            "sft_path": str(jsonl_file),
            "heldout_frac": 0.10,
            "split_seed": 42,
            "seq_len": 4096,
        },
        "model": {
            "chat_template": str(dummy_template_path),
        },
    }

    train_ds, eval_ds, _ = get_aft_datasets(config, tokenizer)

    train_ids = {item["id"] for item in train_ds}
    eval_ids = {item["id"] for item in eval_ds}

    # Ensure sets are disjoint
    intersection = train_ids.intersection(eval_ids)
    assert len(intersection) == 0, "Train and eval sets must be strictly disjoint"
    assert len(train_ds) + len(eval_ds) == 100

