"""S9 near-duplicate removal (§6.3): embedding cosine + exact 13-gram overlap.

* Embeddings: ``sentence-transformers/all-mpnet-base-v2`` [default]. A document is removed if its
  cosine similarity to any already-kept document exceeds 0.92 [default] (greedy, input order).
* 13-gram overlap: a document is removed if more than 30% [default] of its 13-grams already appear
  in the documents kept so far (checked against the union of all kept documents, not pair by pair).
* Documents are processed in a seeded, fixed order (``seeded_order``) so the result is reproducible.

Embedding needs the optional extra: ``uv pip install -e ".[embed]" --torch-backend=cpu``. On wtc11
run it on CPU (``CUDA_VISIBLE_DEVICES=""``) so it never competes with the vLLM server for the GPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

_TOK = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def ngram_seen_share(toks: list[str], seen: set[tuple[str, ...]], n: int = 13) -> float:
    """Share of the document's n-grams (one per position) that already appear in ``seen`` (§6.3 S9)."""
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return sum(g in seen for g in grams) / len(grams) if grams else 0.0


def seeded_order(ids: Sequence[str], seed: int) -> list[int]:
    """Indices of ``ids`` in a seeded, fixed order: sort by id first (independent of file order),
    then shuffle with ``seed``."""
    import random
    idx = sorted(range(len(ids)), key=lambda i: ids[i])
    random.Random(seed).shuffle(idx)
    return idx


def ngrams(toks: list[str], n: int = 13) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def embed(texts: Sequence[str], model_name: str = "sentence-transformers/all-mpnet-base-v2",
          device: str = "cpu", batch_size: int = 16):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError('S9 embeddings need: uv pip install -e ".[embed]" --torch-backend=cpu') from e
    model = SentenceTransformer(model_name, device=device)
    return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


@dataclass
class DedupResult:
    kept: list[int] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)

    @property
    def removal_rate(self) -> float:
        n = len(self.kept) + len(self.removed)
        return len(self.removed) / n if n else 0.0


def greedy_dedup(texts: Sequence[str], embeddings=None, *, cos_threshold: float = 0.92,
                 ngram_n: int = 13, ngram_max_share: float = 0.30) -> DedupResult:
    """Greedy in input order. ``embeddings`` (L2-normalised rows) may be None to skip the cosine test."""
    res = DedupResult()
    seen: set[tuple[str, ...]] = set()
    kept_vecs: list[Any] = []
    for i, text in enumerate(texts):
        toks = tokens(text)
        if embeddings is not None and kept_vecs:
            sims = [float((embeddings[i] * v).sum()) for v in kept_vecs]
            j = max(range(len(sims)), key=sims.__getitem__)
            if sims[j] > cos_threshold:
                res.removed.append({"index": i, "reason": "cosine", "value": round(sims[j], 4),
                                    "nearest_kept": res.kept[j]})
                continue
        share = ngram_seen_share(toks, seen, ngram_n)
        if share > ngram_max_share:
            res.removed.append({"index": i, "reason": "ngram13", "value": round(share, 4)})
            continue
        res.kept.append(i)
        seen |= ngrams(toks, ngram_n)
        if embeddings is not None:
            kept_vecs.append(embeddings[i])
    return res
