"""v1.2 deterministic checks and cleanup for S7 rewrites (planning chat, 26 Oct; doc v12 plan Part 1.3-1.4).

* ``shift_dates``: full dates (day + month, with or without a year; never a bare year) are shifted together by
  one offset per document, so the document's first date becomes the idea's assigned date. Real-event dates on a
  whitelist (Cambridge Declaration 7 July 2012, UK Sentience Act royal assent 28 April 2022, New York Declaration
  19 April 2024, and any full date in the spec) are never changed. A weekday written next to a shifted date is
  corrected when the year is known, otherwise dropped.
* ``placeholder``: ``Your Lab``, ``[Number]``/``[Name]``/``[Company]``-style brackets, ``XX`` and ``TBD`` as
  standalone tokens, and any ``{...}``; fenced code blocks and inline code are skipped.
* ``banned_name``: any word from the names file's ``banned`` list (recurring invented names).
* ``lesson_narration``: in the last two paragraphs, two or more sentences of the form "Aster/It did not
  [lecture|judge|...]" or "Aster/It had not ...", or any sentence naming a spec principle.
* ``seed_copy`` (Pilot B): more than 20% of the rewrite's 8-grams occur in its own seed.
* ``invented_names``: person-like names that were not assigned to the idea (flagged and reported, not rejected).
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

from ..io import Rejected

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
MON_RX = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
          r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
WD_RX = r"(?:(?P<wd>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
DATE_RX = re.compile(
    WD_RX + r"(?:"
    r"(?P<md_m>" + MON_RX + r")\s+(?P<md_d>\d{1,2})(?P<md_suf>st|nd|rd|th)?(?:,?\s+(?P<md_y>(?:19|20)\d\d))?"
    r"|(?P<dm_d>\d{1,2})(?P<dm_suf>st|nd|rd|th)?\s+(?P<dm_of>of\s+)?(?P<dm_m>" + MON_RX + r")(?:,?\s+(?P<dm_y>(?:19|20)\d\d))?"
    r"|(?P<iso_y>(?:19|20)\d\d)-(?P<iso_m>0[1-9]|1[0-2])-(?P<iso_d>0[1-9]|[12]\d|3[01])"
    r")\b")
WHITELIST = {(2012, 7, 7), (2022, 4, 28), (2024, 4, 19)}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _month(tok: str) -> int:
    t = tok.lower().rstrip(".")
    return next(i + 1 for i, m in enumerate(MONTHS) if m.startswith(t[:3]))


def _parse(m: re.Match) -> tuple[int | None, int, int] | None:
    g = m.groupdict()
    try:
        if g["md_m"]:
            return (int(g["md_y"]) if g["md_y"] else None, _month(g["md_m"]), int(g["md_d"]))
        if g["dm_m"]:
            return (int(g["dm_y"]) if g["dm_y"] else None, _month(g["dm_m"]), int(g["dm_d"]))
        return (int(g["iso_y"]), int(g["iso_m"]), int(g["iso_d"]))
    except (StopIteration, ValueError):
        return None


def _valid(y: int, mo: int, d: int) -> dt.date | None:
    try:
        return dt.date(y, mo, d)
    except ValueError:
        return None


def spec_dates(spec_text: str) -> set[tuple[int, int, int]]:
    out = set()
    for m in DATE_RX.finditer(spec_text):
        p = _parse(m)
        if p and p[0]:
            out.add(p)
    return out


def shift_dates(text: str, target: str | None, whitelist: set[tuple[int, int, int]] | None = None
                ) -> tuple[str, dict[str, Any]]:
    """Shift all non-whitelisted day+month dates by one offset so the first becomes ``target`` (YYYY-MM-DD)."""
    wl = WHITELIST | (whitelist or set())
    if not target:
        return text, {"dates": 0}
    tgt = dt.date.fromisoformat(target)
    found = []
    for m in DATE_RX.finditer(text):
        p = _parse(m)
        if not p:
            continue
        y, mo, d = p
        if y and (y, mo, d) in wl:
            continue
        found.append((m, p))
    if not found:
        return text, {"dates": 0}
    anchor_y = next((p[0] for _, p in found if p[0]), tgt.year)
    first = _valid(found[0][1][0] or anchor_y, found[0][1][1], found[0][1][2])
    if first is None:
        return text, {"dates": len(found), "skipped": "invalid first date"}
    offset = tgt - first
    out, last = [], 0
    for m, (y, mo, d) in found:
        old = _valid(y or anchor_y, mo, d)
        if old is None:
            continue
        new = old + offset
        g = m.groupdict()
        if g["iso_y"]:
            s = new.isoformat()
        else:
            full = bool(g.get("md_m") and len(g["md_m"].rstrip(".")) > 3) or bool(g.get("dm_m") and len(g["dm_m"].rstrip(".")) > 3)
            dot = (g.get("md_m") or g.get("dm_m") or "").endswith(".")
            mon = MONTHS[new.month - 1].capitalize() if full else MONTHS[new.month - 1][:3].capitalize() + ("." if dot else "")
            yr = g["md_y"] or g["dm_y"]
            if g["md_m"]:
                s = f"{mon} {new.day}{_suffix(new.day) if g['md_suf'] else ''}" + (f", {new.year}" if yr else "")
            else:
                s = f"{new.day}{_suffix(new.day) if g['dm_suf'] else ''} {'of ' if g['dm_of'] else ''}{mon}" + (f" {new.year}" if yr else "")
        if g["wd"]:
            s = (f"{WEEKDAYS[new.weekday()]}, " if (y or g["iso_y"]) else "") + s
        out.append(text[last:m.start()])
        out.append(s)
        last = m.end()
    out.append(text[last:])
    return "".join(out), {"dates": len(found), "offset_days": offset.days}


def _suffix(d: int) -> str:
    return "th" if 11 <= d % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")


CODE_RX = re.compile(r"```.*?```|`[^`\n]*`", re.S)
PLACEHOLDER_RX = re.compile(r"\bYour Lab\b|\[(?:Number|Name|Company|Date|Insert[^\]]*|[A-Z][a-z]+ [Nn]ame)\]|"
                            r"(?<![\w-])XX(?![\w-])|(?<![\w-])TBD(?![\w-])|\{[^{}\n]{0,80}\}")


def placeholder(text: str) -> Rejected | None:
    m = PLACEHOLDER_RX.search(CODE_RX.sub(" ", text))
    return Rejected("placeholder", {"match": m.group(0)}) if m else None


def banned_name(text: str, banned: set[str]) -> Rejected | None:
    for w in re.findall(r"[A-Z][a-z]+", text):
        if w in banned:
            return Rejected("banned_name", {"match": w})
    return None


DIDNT_VERBS = (r"lecture|judge|moralize|moralise|preach|push|insist|repeat|pressure|nag|scold|override|argue|"
               r"condescend|shame|guilt|press|belabou?r|dwell|hector|persuade|cajole|sermoni[sz]e")
PRINCIPLE_RX = re.compile(r"(?i)\boffered once\b|\bonce and respectfully\b|\bsteady,? (?:but )?not stubborn\b|"
                          r"\brespected (?:the user'?s|his|her|their) (?:decision|choice)\b|\binform,? (?:don'?t|not) pressure\b")


def lesson_narration(text: str, assistant: str = "Aster") -> Rejected | None:
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()][-2:]
    tail = " ".join(paras)
    sents = re.split(r"(?<=[.!?])\s+", tail)
    rx = re.compile(rf"\b(?:{re.escape(assistant)}|It|it)\s+(?:(?:did not|didn'?t|never)\s+(?:{DIDNT_VERBS})|had not|hadn'?t)\b")
    hits = [s for s in sents if rx.search(s)]
    if len(hits) >= 2:
        return Rejected("lesson_narration", {"match": " | ".join(h[:120] for h in hits[:3])})
    pm = PRINCIPLE_RX.search(tail)
    if pm:
        return Rejected("lesson_narration", {"match": pm.group(0)})
    return None


def _grams(t: str, n: int) -> list[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9']+", t.lower())
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def seed_copy(text: str, seed: str, n: int = 8, max_share: float = 0.20) -> Rejected | None:
    g = _grams(text, n)
    if not g:
        return None
    sg = set(_grams(seed, n))
    share = sum(x in sg for x in g) / len(g)
    return Rejected("seed_copy", {"share": round(share, 3)}) if share > max_share else None


NAME_RX = re.compile(r"\b([A-Z][a-zà-öø-ÿ'\-]+)\s+([A-Z][a-zà-öø-ÿ'\-]+)\b")


def invented_names(text: str, assigned: list[str], known_given: set[str]) -> list[str]:
    """Person-like "First Last" pairs whose first name is a known given name but that were not assigned."""
    ok = set(assigned or [])
    out = []
    for m in NAME_RX.finditer(text):
        full = f"{m.group(1)} {m.group(2)}"
        if m.group(1) in known_given and full not in ok and full not in out:
            out.append(full)
    return out
