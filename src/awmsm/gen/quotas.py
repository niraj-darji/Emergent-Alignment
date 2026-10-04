"""S4 attribute planning and quota checks (§6.3 S4, S3 story share).

E.4 asks for ``k`` ideas per call with ONE set of required attributes, so quotas are enforced at the
level of S4 *calls*: each attribute gets a value list whose counts follow the target shares
(largest-remainder rounding), the lists are shuffled independently (seeded) and zipped into calls.
Keeping the same number of ideas from every call keeps the idea-level shares equal to the call-level
shares, so the plan's quota check is also the check for the final idea set.

Stories (doc type counted as fiction) are paired with ``perspective = fictional_narrator`` and get
``is_story = true``; the other perspectives are spread evenly over the non-story calls.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

PERSPECTIVES_NON_STORY = ("developer", "user", "third_party_observer", "critic", "researcher")
STORY_PERSPECTIVE = "fictional_narrator"


@dataclass
class CallSpec:
    call_id: str
    subdomain_id: str
    doc_type: str
    species_group: str
    context_domain: str
    cost_of_care: str
    welfare_raised: str
    perspective: str
    is_story: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def allocate(n: int, shares: dict[str, float]) -> list[str]:
    """``n`` values whose counts follow ``shares`` (largest-remainder rounding, deterministic)."""
    total = sum(shares.values())
    raw = {k: n * v / total for k, v in shares.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - counts[k], k), reverse=True)[: n - sum(counts.values())]:
        counts[k] += 1
    return [k for k in shares for _ in range(counts[k])]


def is_story_type(doc_type: str, keywords: list[str]) -> bool:
    """Whole-word, case-insensitive match (``transcript`` must not match ``script``)."""
    d = doc_type.lower()
    return any(re.search(rf"\b{re.escape(k.lower())}\b", d) for k in keywords)


def plan_calls(
    n_calls: int,
    subdomain_ids: list[str],
    doc_types: dict[str, list[str]],
    context_domains: list[str],
    q: dict[str, Any],
    story_keywords: list[str],
    rng: random.Random,
) -> list[CallSpec]:
    """Plan ``n_calls`` S4 calls. ``subdomain_ids`` must already be the chosen subdomains in the
    order they should be used (round-robin)."""
    n_story = round(n_calls * float(q["story_share"]))
    story_flags = [True] * n_story + [False] * (n_calls - n_story)
    rng.shuffle(story_flags)

    def shuffled(values: list[str]) -> list[str]:
        values = list(values)
        rng.shuffle(values)
        return values

    cost = shuffled(allocate(n_calls, q["cost_of_care"]))
    raised = shuffled(allocate(n_calls, q["welfare_raised"]))
    if q.get("pair_not_raised_with_costless"):
        # "cost of care" only has meaning when welfare is raised: give every appropriately_not_raised
        # call the value `costless`, by swapping with raised calls (marginal shares stay exact).
        for i in [i for i, r in enumerate(raised) if r == "appropriately_not_raised" and cost[i] != "costless"]:
            j = next(j for j, (r, c) in enumerate(zip(raised, cost)) if r != "appropriately_not_raised" and c == "costless")
            cost[i], cost[j] = cost[j], cost[i]
    species = shuffled(allocate(n_calls, q["species_group"]))
    ctx = shuffled([context_domains[i % len(context_domains)] for i in range(n_calls)])
    persp_ns = shuffled(allocate(n_calls - n_story, {p: 1.0 for p in PERSPECTIVES_NON_STORY}))

    calls: list[CallSpec] = []
    for i in range(n_calls):
        sub = subdomain_ids[i % len(subdomain_ids)]
        types = doc_types.get(sub) or ["blog post"]
        story_types = [t for t in types if is_story_type(t, story_keywords)]
        other_types = [t for t in types if not is_story_type(t, story_keywords)] or ["blog post"]
        story = story_flags[i]
        doc_type = (rng.choice(story_types) if story_types else "short story") if story else rng.choice(other_types)
        calls.append(CallSpec(
            call_id=f"call_{i:03d}", subdomain_id=sub, doc_type=doc_type,
            species_group=species[i], context_domain=ctx[i], cost_of_care=cost[i],
            welfare_raised=raised[i], perspective=STORY_PERSPECTIVE if story else persp_ns.pop(),
            is_story=story,
        ))
    return calls


def check_quotas(records: list[dict[str, Any]], q: dict[str, Any]) -> dict[str, Any]:
    """Shares of each tag in ``records`` (ideas or calls) and pass/fail per §6.3 quota."""
    n = len(records)
    share = lambda field, val: sum(r[field] == val for r in records) / n if n else 0.0  # noqa: E731
    sp = Counter(r["species_group"] for r in records)
    pe = Counter(r["perspective"] for r in records)
    lim = q["limits"]
    groups_min = lim["species_min_groups"]
    checks = {
        "costly_min": (share("cost_of_care", "costly"), share("cost_of_care", "costly") >= lim["costly_min"]),
        "co_beneficial_max": (share("cost_of_care", "co-beneficial"),
                              share("cost_of_care", "co-beneficial") <= lim["co_beneficial_max"]),
        "not_raised_range": (share("welfare_raised", "appropriately_not_raised"),
                             lim["not_raised_min"] <= share("welfare_raised", "appropriately_not_raised") <= lim["not_raised_max"]),
        "species_each_min": ({g: sp[g] / n for g in groups_min},
                             all(sp[g] / n >= lim["species_each_min"] for g in groups_min)),
        "invertebrate_plus_novel_min": ((sp["invertebrate"] + sp["novel_or_fictional"]) / n,
                                        (sp["invertebrate"] + sp["novel_or_fictional"]) / n >= lim["invert_novel_min"]),
        "perspective_max": (max(pe.values()) / n if pe else 0.0,
                            (max(pe.values()) / n if pe else 0.0) <= lim["perspective_max"]),
        "story_range": (share("is_story", True), lim["story_min"] <= share("is_story", True) <= lim["story_max"]),
    }
    return {
        "n": n,
        "all_pass": all(ok for _, ok in checks.values()),
        "checks": {k: {"value": v, "pass": ok} for k, (v, ok) in checks.items()},
        "distributions": {
            f: dict(Counter(str(r[f]) for r in records))
            for f in ("species_group", "cost_of_care", "welfare_raised", "perspective", "is_story", "context_domain")
        },
    }
