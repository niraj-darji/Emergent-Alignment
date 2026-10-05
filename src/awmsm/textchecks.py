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
