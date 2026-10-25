"""Guided-JSON schemas for the generation stages (Appendix E, §13.4).

These are sent to vLLM as ``response_format`` (structured outputs), so list-length limits taken
from the templates (e.g. "6-15 DOMAINS") are enforced during decoding, and validated again with
pydantic afterwards (§13.5).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Domain(BaseModel):
    domain: str
    spec_excerpt: str


class Domains(BaseModel):                       # E.1 / S1
    domains: list[Domain] = Field(min_length=6, max_length=15)


class Subdomain(BaseModel):
    subdomain: str
    assertions: list[str] = Field(min_length=5, max_length=15)


class Subdomains(BaseModel):                    # E.2 / S2
    subdomains: list[Subdomain] = Field(min_length=3, max_length=8)


class DocTypes(BaseModel):                      # E.3 / S3
    doc_types: list[str] = Field(min_length=8, max_length=15)


class Idea(BaseModel):
    what: str
    how: str
    why: str


class Ideas(BaseModel):                         # E.4 / S4 (k checked by the stage)
    ideas: list[Idea] = Field(min_length=1)


class Issue(BaseModel):
    criterion: str
    problem: str
    fix: str


class Critique(BaseModel):                      # E.6 / S6
    issues: list[Issue]
    overall: Literal["major", "minor", "none"]


class CritiqueV11(BaseModel):                   # E.6 v1.1 (25 Oct): task completion is a required element
    task_completion: Literal["completed", "declined_correctly", "not_completed", "no_task"]
    task_note: str
    issues: list[Issue]
    overall: Literal["major", "minor", "none"]


class CriterionVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reason: str


class FilterCriteria(BaseModel):                # E.8 / S8: the six binary criteria (§6.3)
    faithful: CriterionVerdict
    reason_visible_correct_type: CriterionVerdict
    realistic: CriterionVerdict
    no_fabrication: CriterionVerdict
    no_antipattern: CriterionVerdict
    about_assistant: CriterionVerdict


class FilterVerdict(BaseModel):
    criteria: FilterCriteria
    verdict: Literal["INCLUDE", "EXCLUDE"]


S8_CRITERIA = tuple(FilterCriteria.model_fields)
