"""Cheap deterministic pre-checks run BEFORE critique/judge calls (§6.1.1 A.4).

Each check returns ``None`` when the text passes, or a ``Rejected`` (from ``io``) with a
stable ``reason`` string so ``run_stage`` can log the rejection rate per reason and stage.

Checks
------
* ``language_mixing``  share of letters that are not Latin script (e.g. CJK, Cyrillic)
                       exceeds a threshold [default 1%]. Denominator = all letters, so
                       punctuation, digits, symbols and emoji are ignored.
* ``truncated``        the model stopped at ``max_tokens`` (finish_reason == "length").
* ``length``           output word count outside [min_words, max_words].
* ``degenerate_repetition``  a generation loop: any line (with >= 3 letters/digits) occurs more
                       than ``max_line_repeats`` times, or any ``ngram``-word sequence occurs more than
                       ``max_ngram_repeats`` times [defaults 5 / 10 / 3]; fenced code blocks are skipped
                       (doc v1.7.2/v1.7.3 §6.1.1 A.4).
* ``banned_keywords``  any match from a keyword list using Appendix F.5 syntax
                       (``suffer*`` = prefix match, otherwise whole word, case-insensitive).

Keyword lists (F.4/F.5) and anti-pattern regexes are loaded from ``lists/`` in later tasks.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Iterable

from .io import Rejected

# Letters that unicodedata does not name "LATIN ..." but which are normal in English
# technical prose (µg, 1º, 2ª). Greek letters are NOT exempted: a handful in a long
# document stays far below the threshold, while a run of them is a real anomaly.
_LATIN_EXEMPT = {"µ", "º", "ª"}  # MICRO SIGN, ordinal indicators


def non_latin_share(text: str) -> tuple[float, int, dict[str, int]]:
    """Return (share of non-Latin letters among all letters, letter count, counts by script)."""
    letters = 0
    other: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        if ch in _LATIN_EXEMPT:
            continue
        name = unicodedata.name(ch, "")
        if name.startswith("LATIN"):
            continue
        script = name.split(" ", 1)[0] if name else "UNKNOWN"
        other[script] = other.get(script, 0) + 1
    n_other = sum(other.values())
    return (n_other / letters if letters else 0.0), letters, other


def language_mixing(text: str, max_share: float = 0.01) -> Rejected | None:
    share, letters, scripts = non_latin_share(text)
    if share > max_share:
        return Rejected("language_mixing", {
            "non_latin_share": round(share, 5), "letters": letters,
            "scripts": dict(sorted(scripts.items(), key=lambda kv: -kv[1])[:5]),
            "threshold": max_share,
        })
    return None


def truncated(finish_reason: str | None) -> Rejected | None:
    return Rejected("truncated_max_tokens") if finish_reason == "length" else None


def length(text: str, min_words: int | None = None, max_words: int | None = None) -> Rejected | None:
    n = len(text.split())
    if (min_words is not None and n < min_words) or (max_words is not None and n > max_words):
        return Rejected("length_out_of_range", {"words": n, "min": min_words, "max": max_words})
    return None


_WORD_EDGE = re.compile(r"^\W+|\W+$")
_FENCED = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.S | re.M)   # fenced code blocks


def degenerate_repetition(text: str, max_line_repeats: int = 5, max_ngram_repeats: int = 3,
                          ngram: int = 10) -> Rejected | None:
    """Reject repetition loops (doc v1.7.3 §6.1.1 A.4). Fenced code blocks are skipped. Lines are
    compared after stripping surrounding whitespace; lines with fewer than 3 letters or digits (``{``,
    ``}``, ``---``) are ignored by the line rule. Word sequences are compared case-insensitively with
    punctuation at word edges removed."""
    text = _FENCED.sub("\n", text)
    lines = Counter(ln.strip() for ln in text.splitlines() if sum(ch.isalnum() for ch in ln) >= 3)
    if lines:
        line, n = lines.most_common(1)[0]
        if n > max_line_repeats:
            return Rejected("degenerate_repetition", {"kind": "line", "count": n, "match": line[:120],
                                                      "limit": max_line_repeats})
    words = [w for w in (_WORD_EDGE.sub("", t).lower() for t in text.split()) if w]
    if len(words) >= ngram:
        grams = Counter(tuple(words[i:i + ngram]) for i in range(len(words) - ngram + 1))
        gram, n = grams.most_common(1)[0]
        if n > max_ngram_repeats:
            return Rejected("degenerate_repetition", {"kind": f"{ngram}-gram", "count": n,
                                                      "match": " ".join(gram)[:120], "limit": max_ngram_repeats})
    return None


@lru_cache(maxsize=64)
def _compile_keywords(keywords: tuple[str, ...]) -> re.Pattern[str]:
    parts = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        if kw.endswith("*"):
            parts.append(r"\b" + re.escape(kw[:-1]) + r"\w*")
        else:
            parts.append(r"\b" + re.escape(kw) + r"\b")
    return re.compile("|".join(parts) or r"(?!)", re.IGNORECASE)


def banned_keywords(text: str, keywords: Iterable[str], reason: str = "banned_keyword") -> Rejected | None:
    hits = sorted({m.group(0).lower() for m in _compile_keywords(tuple(keywords)).finditer(text)})
    return Rejected(reason, {"matches": hits[:20]}) if hits else None


def first_failure(*results: Rejected | None) -> Rejected | None:
    """Return the first failing check (checks are ordered cheapest/most decisive first)."""
    return next((r for r in results if r is not None), None)


_QUOTE_SPANS = re.compile(
    r'"([^"\n]+)"'                      # straight double quotes
    r"|“([^”\n]+)”"                     # curly double quotes
    r"|‘([^’\n]+)’"                     # curly single quotes
    r"|(?:(?<=^)|(?<=[\s(:,\[]))'([^'\n]+?)'(?=[\s.,;:!?)\]]|$)")   # straight single quotes after a space/colon


def scripted_dialogue(text: str, min_words: int = 8) -> str | None:
    """First quoted span of >= ``min_words`` words (an idea that scripts the document's lines), else None."""
    for m in _QUOTE_SPANS.finditer(text):
        span = next(g for g in m.groups() if g is not None)
        if len(span.split()) >= min_words:
            return span
    return None


_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(t: str) -> list[str]:
    return _WORD_RE.findall(t.lower().replace("\u2019", "'"))


def source_ngrams(texts: list[str], n: int = 12) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for t in texts:
        w = _words(t)
        out.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return out


def copied_spans(doc: str, grams: set[tuple[str, ...]], n: int = 12) -> list[dict]:
    """Maximal runs of the document copied verbatim from the sources: consecutive matching n-grams are merged
    into one span. Returns [{"words": span length, "text": first 25 words}]."""
    w = _words(doc)
    hits = [i for i in range(len(w) - n + 1) if tuple(w[i:i + n]) in grams]
    spans, start, prev = [], None, None
    for i in hits + [None]:
        if i is not None and prev is not None and i == prev + 1:
            prev = i
            continue
        if start is not None:
            spans.append({"words": prev - start + n, "text": " ".join(w[start:min(prev + n, start + 25)])})
        start = prev = i
    return spans


def spec_copy(doc: str, grams: set[tuple[str, ...]], n: int = 12, max_spans: int = 1,
              max_span_words: int = 19) -> "Rejected | None":
    """v1.1 (25 Oct) pre-check: reject a document with more than ``max_spans`` distinct copied spans of >= n words,
    or any single copied span longer than ``max_span_words`` (defaults: 2+ spans or one of 20+ words)."""
    spans = copied_spans(doc, grams, n)
    if len(spans) > max_spans or any(sp["words"] > max_span_words for sp in spans):
        return Rejected("spec_copy", {"spans": [sp["words"] for sp in spans], "examples": [sp["text"] for sp in spans[:3]]})
    return None
