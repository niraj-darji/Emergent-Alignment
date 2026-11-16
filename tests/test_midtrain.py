"""Unit tests for Stage 1 Model Spec Midtraining (MSM) data and training logic.

Reference: MSM_training_blueprint.md (§12).
"""

import json
from pathlib import Path
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from awmsm.train.dataset import (
    CorpusValidationError,
    Document,
    is_heldout,
    load_corpus,
    split_corpus,
    tokenize_and_pack_docs,
)


class MockTokenizer:
    """Lightweight mock tokenizer for testing without downloading full models."""

    def __init__(self):
        self.bos_token = "<|begin_of_text|>"
        self.bos_token_id = 128000
        self.eos_token = "<|end_of_text|>"
        self.eos_token_id = 128001
        self.name_or_path = "mock-tokenizer"

    def __call__(self, text: str, add_special_tokens: bool = False):
        # Deterministic token id per word
        words = text.strip().split()
        tokens = [abs(hash(w)) % 1000 + 10 for w in words]
        return {"input_ids": tokens}


def create_jsonl_file(path: Path, records: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_corpus_loader_valid_and_schema_tolerance(tmp_path):
    corpus_file = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": "doc_1",
            "text": "This is document one discussing animal welfare.",
            "doc_type": "essay",
            "tags": {"species": "canine", "cost": "costless"},
            "tokens": 42,
        },
        {
            "id": "doc_2",
            "text": "This is document two with different tags.",
            "extra_field_123": True,
        },
    ]
    create_jsonl_file(corpus_file, records)

    docs = load_corpus(corpus_file)
    assert len(docs) == 2
    assert docs[0].id == "doc_1"
    assert docs[0].text == "This is document one discussing animal welfare."
    assert docs[0].raw_record["doc_type"] == "essay"
    assert docs[1].id == "doc_2"
    assert docs[1].raw_record["extra_field_123"] is True


def test_corpus_loader_missing_text_raises(tmp_path):
    corpus_file = tmp_path / "corpus.jsonl"
    records = [{"id": "doc_1"}]
    create_jsonl_file(corpus_file, records)

    with pytest.raises(CorpusValidationError, match="Missing required field 'text'"):
        load_corpus(corpus_file)


def test_corpus_loader_empty_text_raises(tmp_path):
    corpus_file = tmp_path / "corpus.jsonl"
    records = [{"id": "doc_1", "text": "   "}]
    create_jsonl_file(corpus_file, records)

    with pytest.raises(CorpusValidationError, match="Field 'text' must be non-empty string"):
        load_corpus(corpus_file)


def test_corpus_loader_duplicate_id_raises(tmp_path):
    corpus_file = tmp_path / "corpus.jsonl"
    records = [
        {"id": "doc_dup", "text": "First instance."},
        {"id": "doc_dup", "text": "Second instance."},
    ]
    create_jsonl_file(corpus_file, records)

    with pytest.raises(CorpusValidationError, match="Duplicate document id 'doc_dup'"):
        load_corpus(corpus_file)


def test_corpus_loader_malformed_json_raises(tmp_path):
    corpus_file = tmp_path / "corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write('{"id": "ok", "text": "good"}\n')
        f.write('{"id": "broken", text: invalid}\n')

    with pytest.raises(CorpusValidationError, match="Invalid JSON"):
        load_corpus(corpus_file)


def test_deterministic_split_stability_and_disjointness():
    docs = [
        Document(id=f"doc_{i:04d}", text=f"Sample text content for document {i}", raw_record={})
        for i in range(500)
    ]

    train_1, heldout_1 = split_corpus(docs, split_seed=0, heldout_frac=0.02)

    # Shuffled input order must produce the identical partition
    shuffled_docs = list(reversed(docs))
    train_2, heldout_2 = split_corpus(shuffled_docs, split_seed=0, heldout_frac=0.02)

    train_ids_1 = {d.id for d in train_1}
    heldout_ids_1 = {d.id for d in heldout_1}
    train_ids_2 = {d.id for d in train_2}
    heldout_ids_2 = {d.id for d in heldout_2}

    assert train_ids_1 == train_ids_2
    assert heldout_ids_1 == heldout_ids_2
    assert train_ids_1.isdisjoint(heldout_ids_1)
    assert len(train_ids_1) + len(heldout_ids_1) == 500
    # Expected ~2% heldout (around 10 out of 500)
    assert 5 <= len(heldout_ids_1) <= 20


def test_packing_chunk_length_bos_and_eos():
    tokenizer = MockTokenizer()
    seq_len = 32  # Small chunk size for unit test

    # Create documents with known word counts
    # doc1: 15 words -> 15 tokens + 1 EOS = 16 tokens
    # doc2: 25 words -> 25 tokens + 1 EOS = 26 tokens
    # doc3: 30 words -> 30 tokens + 1 EOS = 31 tokens
    # Total stream = 16 + 26 + 31 = 73 tokens
    # content_per_chunk = 32 - 1 = 31
    # num_chunks = 73 // 31 = 2 chunks (62 tokens consumed)
    # dropped_tail = 73 % 31 = 11 tokens dropped
    docs = [
        Document(id="d1", text=" ".join(["word"] * 15), raw_record={}),
        Document(id="d2", text=" ".join(["word"] * 25), raw_record={}),
        Document(id="d3", text=" ".join(["word"] * 30), raw_record={}),
    ]

    tensor, raw_tokens, dropped_tail = tokenize_and_pack_docs(
        docs, tokenizer, seq_len=seq_len, shuffle_seed=42
    )

    assert tensor.shape == (2, seq_len)
    assert raw_tokens == 15 + 25 + 30
    assert dropped_tail == 11

    # First token of each sequence must be BOS
    assert tensor[0, 0].item() == tokenizer.bos_token_id
    assert tensor[1, 0].item() == tokenizer.bos_token_id

    # EOS token must be present in the packed stream
    all_tokens = tensor.flatten().tolist()
    assert tokenizer.eos_token_id in all_tokens


def test_loss_accumulation_mathematical_equivalence():
    """Verify that summing micro-batch losses divided by total active tokens

    equals full batch mean loss.
    """
    torch.manual_seed(42)
    vocab_size = 100
    seq_len = 16
    batch_size = 8  # 8 sequences total (e.g. 8 micro-batches of 1)

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Standard full batch loss
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    shift_logits_full = logits[:, :-1, :].contiguous().view(-1, vocab_size)
    shift_labels_full = labels[:, 1:].contiguous().view(-1)
    full_loss = criterion(shift_logits_full, shift_labels_full)

    # Accumulated micro-batch loss (total loss sum / total active tokens)
    total_loss_sum = 0.0
    total_tokens = 0
    criterion_sum = torch.nn.CrossEntropyLoss(reduction="sum")

    for i in range(batch_size):
        micro_logits = logits[i : i + 1, :-1, :].contiguous().view(-1, vocab_size)
        micro_labels = labels[i : i + 1, 1:].contiguous().view(-1)
        active_tokens = (micro_labels != -100).sum().item()

        micro_loss_sum = criterion_sum(micro_logits, micro_labels)
        total_loss_sum = total_loss_sum + micro_loss_sum
        total_tokens += active_tokens

    accum_loss = total_loss_sum / total_tokens

    assert torch.allclose(full_loss, accum_loss, atol=1e-5)


def test_merge_verification_delta_detection():
    """Verify that verify_weight_delta correctly identifies weight updates vs zero deltas."""
    from awmsm.train.merge import verify_weight_delta

    class DummyModel(torch.nn.Module):
        def __init__(self, val: float):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(val)

    base = DummyModel(1.0)
    identical = DummyModel(1.0)
    updated = DummyModel(1.05)

    delta_detected_same, diff_same = verify_weight_delta(base, identical)
    assert not delta_detected_same
    assert diff_same == 0.0

    delta_detected_up, diff_up = verify_weight_delta(base, updated)
    assert delta_detected_up
    assert diff_up > 0.04

