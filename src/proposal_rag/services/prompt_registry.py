from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Optional
from jinja2 import Template

class PromptLoader:
    """File-based prompt registry with simple in-memory cache."""
    def __init__(self, base: Path):
        self.base = Path(base)
        self._cache: dict[str, str] = {}

    def _read(self, name: str) -> str:
        path = self.base / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def get(self, name: str, *, use_cache: bool = True) -> str:
        if use_cache and name in self._cache:
            return self._cache[name]
        text = self._read(name)
        if use_cache:
            self._cache[name] = text
        return text

    def render(self, name: str, context: Mapping[str, Any], *, use_cache: bool = True) -> str:
        src = self.get(name, use_cache=use_cache)
        return Template(src).render(**context)


