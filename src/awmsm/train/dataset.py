"""Dataset preparation, deterministic splitting, tokenization, packing, and caching.

Reference: MSM_training_blueprint.md (§2.1, §3.1-§3.5).
"""

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class Document:
    """A single document from the corpus JSONL file."""
    id: str
    text: str
    raw_record: Dict[str, Any]


class CorpusValidationError(Exception):
    """Raised when corpus JSONL violates the schema contract."""
    pass


def load_corpus(corpus_path: str | Path) -> List[Document]:
    """Load and validate documents from a corpus JSONL file.

    Schema contract (§2.1 of blueprint):
    - Required fields: 'id' (str, unique), 'text' (str, non-empty).
    - All other fields are passed through unused in raw_record.
    - Fails loudly with line numbers on validation errors.
    """
    path = Path(corpus_path)
    if not path.is_file():
        # Check relative to project root
        project_root = Path(__file__).resolve().parents[3]
        candidate = project_root / path
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(f"Corpus file not found: {path} (also checked {candidate})")

    documents: List[Document] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise CorpusValidationError(
                    f"Line {line_num}: Invalid JSON in {path}: {e}"
                ) from e

            if not isinstance(record, dict):
                raise CorpusValidationError(
                    f"Line {line_num}: Expected JSON object, got {type(record).__name__} in {path}"
                )

            if "id" not in record:
                raise CorpusValidationError(
                    f"Line {line_num}: Missing required field 'id' in {path}"
                )
            doc_id = str(record["id"]).strip()
            if not doc_id:
                raise CorpusValidationError(
                    f"Line {line_num}: Empty 'id' field in {path}"
                )

            if doc_id in seen_ids:
                raise CorpusValidationError(
                    f"Line {line_num}: Duplicate document id '{doc_id}' in {path}"
                )
            seen_ids.add(doc_id)

            if "text" not in record:
                raise CorpusValidationError(
                    f"Line {line_num}: Missing required field 'text' in {path}"
                )
            text = record["text"]
            if not isinstance(text, str) or not text.strip():
                raise CorpusValidationError(
                    f"Line {line_num}: Field 'text' must be non-empty string in {path}"
                )

            documents.append(Document(id=doc_id, text=text, raw_record=record))

    if not documents:
        raise CorpusValidationError(f"Corpus file {path} contained no valid documents.")

    return documents


def is_heldout(doc_id: str, split_seed: int = 0, heldout_frac: float = 0.02) -> bool:
    """Deterministic document-level heldout assignment (§3.1).

    Hash rule: sha256(f"{split_seed}:{id}").
    Assign to heldout if hash value falls in lowest heldout_frac fraction.
    """
    digest = hashlib.sha256(f"{split_seed}:{doc_id}".encode("utf-8")).hexdigest()
    # Use first 8 hex characters (32 bits)
    hash_int = int(digest[:8], 16)
    threshold = int(heldout_frac * 0xFFFFFFFF)
    return hash_int < threshold


def split_corpus(
    documents: List[Document],
    split_seed: int = 0,
    heldout_frac: float = 0.02,
) -> Tuple[List[Document], List[Document]]:
    """Split documents into (train_docs, heldout_docs) deterministically."""
    # Sort by id first for complete stability across file orderings
    sorted_docs = sorted(documents, key=lambda d: d.id)

    train_docs: List[Document] = []
    heldout_docs: List[Document] = []

    for doc in sorted_docs:
        if is_heldout(doc.id, split_seed=split_seed, heldout_frac=heldout_frac):
            heldout_docs.append(doc)
        else:
            train_docs.append(doc)

    return train_docs, heldout_docs


def tokenize_and_pack_docs(
    docs: List[Document],
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int = 4096,
    shuffle_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int, int]:
    """Tokenize documents, separate with EOS, pack into chunks of seq_len.

    Packing format (§3.2-§3.3):
    - Each sequence starts with one BOS token ([bos_token_id] + 4095 content tokens).
    - Documents are separated by eos_token_id.
    - Documents are shuffled according to shuffle_seed (if provided).
    - Trailing partial chunk is dropped (no padding).

    Returns:
        packed_tensor: Shape (num_sequences, seq_len), dtype torch.int64.
        raw_tokens_count: Total tokens across documents before packing.
        dropped_tail_tokens: Number of tokens dropped from final incomplete chunk.
    """
    if not docs:
        empty_tensor = torch.empty((0, seq_len), dtype=torch.int64)
        return empty_tensor, 0, 0

    ordered_docs = list(docs)
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(ordered_docs)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer must define a bos_token_id for packing.")
    if eos_id is None:
        raise ValueError("Tokenizer must define an eos_token_id for packing.")

    token_stream: List[int] = []
    raw_tokens_count = 0

    for doc in ordered_docs:
        token_ids = tokenizer(doc.text, add_special_tokens=False)["input_ids"]
        raw_tokens_count += len(token_ids)
        token_stream.extend(token_ids)
        token_stream.append(eos_id)

    # Each chunk gets 1 BOS + (seq_len - 1) tokens from the stream
    content_per_chunk = seq_len - 1
    num_chunks = len(token_stream) // content_per_chunk
    dropped_tail_tokens = len(token_stream) % content_per_chunk

    chunks: List[List[int]] = []
    for i in range(num_chunks):
        start = i * content_per_chunk
        end = start + content_per_chunk
        chunk_content = token_stream[start:end]
        chunk = [bos_id] + chunk_content
        chunks.append(chunk)

    if not chunks:
        packed_tensor = torch.empty((0, seq_len), dtype=torch.int64)
    else:
        packed_tensor = torch.tensor(chunks, dtype=torch.int64)

    return packed_tensor, raw_tokens_count, dropped_tail_tokens


class PackedDataset(Dataset):
    """PyTorch Dataset returning packed input_ids and labels for causal LM."""

    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        return {
            "input_ids": seq,
            "labels": seq.clone(),
        }


def compute_cache_hash(
    corpus_file_path: Path,
    tokenizer_name: str,
    seq_len: int,
    heldout_frac: float,
    split_seed: int,
    train_seed: int,
) -> str:
    """Compute deterministic SHA-256 hash for dataset caching (§3.4)."""
    hasher = hashlib.sha256()
    with open(corpus_file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    spec_str = (
        f"tok:{tokenizer_name}|seq:{seq_len}|heldout:{heldout_frac}|"
        f"split_seed:{split_seed}|train_seed:{train_seed}"
    )
    hasher.update(spec_str.encode("utf-8"))
    return hasher.hexdigest()[:16]


def get_midtrain_datasets(
    corpus_path: str | Path,
    corpus_id: str,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int = 4096,
    heldout_frac: float = 0.02,
    split_seed: int = 0,
    train_seed: int = 1,
    cache_dir: Optional[str | Path] = None,
    target_tokens: Optional[int] = None,
    tolerance: float = 0.02,
    enforce_budget: bool = False,
) -> Tuple[PackedDataset, PackedDataset, Dict[str, Any]]:
    """Build or load cached train and heldout packed datasets.

    Writes stats.json with complete token accounting (§3.5).
    """
    path = Path(corpus_path)
    if not path.is_file():
        project_root = Path(__file__).resolve().parents[3]
        candidate = project_root / path
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(f"Corpus file not found: {path} (also checked {candidate})")

    if cache_dir is None:
        project_root = Path(__file__).resolve().parents[3]
        cache_dir = project_root / "data" / "cache" / corpus_id
    else:
        cache_dir = Path(cache_dir) / corpus_id

    tok_identifier = getattr(tokenizer, "name_or_path", "tokenizer")
    cache_hash = compute_cache_hash(
        corpus_file_path=path,
        tokenizer_name=tok_identifier,
        seq_len=seq_len,
        heldout_frac=heldout_frac,
        split_seed=split_seed,
        train_seed=train_seed,
    )

    cache_path = cache_dir / cache_hash
    train_cache = cache_path / "train.pt"
    heldout_cache = cache_path / "heldout.pt"
    stats_cache = cache_path / "stats.json"

    if train_cache.is_file() and heldout_cache.is_file() and stats_cache.is_file():
        train_tensor = torch.load(train_cache, weights_only=True)
        heldout_tensor = torch.load(heldout_cache, weights_only=True)
        with open(stats_cache, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return PackedDataset(train_tensor), PackedDataset(heldout_tensor), stats

    # Cache miss: load and process
    all_docs = load_corpus(path)
    train_docs, heldout_docs = split_corpus(
        all_docs, split_seed=split_seed, heldout_frac=heldout_frac
    )

    train_tensor, train_raw, train_dropped = tokenize_and_pack_docs(
        docs=train_docs,
        tokenizer=tokenizer,
        seq_len=seq_len,
        shuffle_seed=train_seed,
    )

    heldout_tensor, heldout_raw, heldout_dropped = tokenize_and_pack_docs(
        docs=heldout_docs,
        tokenizer=tokenizer,
        seq_len=seq_len,
        shuffle_seed=split_seed,
    )

    tokens_trained = train_tensor.size(0) * seq_len
    tokens_heldout_packed = heldout_tensor.size(0) * seq_len

    stats: Dict[str, Any] = {
        "corpus_id": corpus_id,
        "corpus_path": str(path),
        "cache_hash": cache_hash,
        "n_docs_total": len(all_docs),
        "n_docs_train": len(train_docs),
        "n_docs_heldout": len(heldout_docs),
        "tokens_train_raw": train_raw,
        "tokens_heldout_raw": heldout_raw,
        "n_train_sequences": train_tensor.size(0),
        "n_heldout_sequences": heldout_tensor.size(0),
        "tokens_trained": tokens_trained,
        "tokens_heldout_packed": tokens_heldout_packed,
        "tokens_dropped_tail": train_dropped,
        "seq_len": seq_len,
        "heldout_frac": heldout_frac,
        "split_seed": split_seed,
        "train_seed": train_seed,
    }

    # Budget check (§3.5)
    if target_tokens is not None:
        delta = abs(tokens_trained - target_tokens)
        rel_diff = delta / target_tokens
        stats["target_tokens"] = target_tokens
        stats["relative_diff_from_target"] = rel_diff
        if rel_diff > tolerance:
            msg = (
                f"Trained tokens ({tokens_trained}) deviates from target ({target_tokens}) "
                f"by {rel_diff * 100:.2f}% (tolerance {tolerance * 100:.1f}%)."
            )
            if enforce_budget:
                raise ValueError(f"Token budget violation: {msg}")
            else:
                stats["budget_warning"] = msg

    # Save to disk cache
    cache_path.mkdir(parents=True, exist_ok=True)
    torch.save(train_tensor, train_cache)
    torch.save(heldout_tensor, heldout_cache)
    with open(stats_cache, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return PackedDataset(train_tensor), PackedDataset(heldout_tensor), stats
