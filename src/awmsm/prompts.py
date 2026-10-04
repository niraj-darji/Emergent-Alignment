"""Prompt templates (Appendix E) with a shared-prefix / variable-suffix split (§6.1.1 A.2).

Template files live in ``prompts/``. The line ``<<<SUFFIX>>>`` separates the shared prefix from the
variable suffix. ``{name}`` placeholders (lowercase identifiers) are substituted in one pass, so
literal JSON in the instructions (``{"domains": [...]}``) and braces inside inserted values are
never touched. Every placeholder in a template must be provided, otherwise ``KeyError``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import REPO_ROOT

MARKER = "<<<SUFFIX>>>"
_VAR = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


@dataclass(frozen=True)
class Rendered:
    prefix: str
    suffix: str

    @property
    def text(self) -> str:
        return self.prefix + self.suffix


@dataclass(frozen=True)
class Template:
    name: str
    prefix: str
    suffix: str

    @property
    def variables(self) -> set[str]:
        return set(_VAR.findall(self.prefix)) | set(_VAR.findall(self.suffix))

    def render(self, **values: Any) -> Rendered:
        missing = self.variables - values.keys()
        if missing:
            raise KeyError(f"Template {self.name}: missing values for {sorted(missing)}")
        return Rendered(fill(self.prefix, values), fill(self.suffix, values))


def fill(text: str, values: dict[str, Any]) -> str:
    return _VAR.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), text)


@lru_cache(maxsize=64)
def load(name: str, root: str | None = None) -> Template:
    path = Path(root or REPO_ROOT / "prompts") / f"{name}.txt"
    raw = path.read_text(encoding="utf-8")
    if MARKER in raw:
        prefix, suffix = raw.split(MARKER, 1)
        suffix = suffix.lstrip("\n")
    else:
        prefix, suffix = raw, ""
    return Template(name, prefix, suffix)


def load_text(name: str, root: str | None = None) -> str:
    return (Path(root or REPO_ROOT / "prompts") / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")
