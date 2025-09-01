from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from ..config.settings import get_settings


class PromptRegistry:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    @lru_cache(maxsize=128)
    def get(self, key: str) -> str:
        path = (self.base_dir / f"{key}.txt").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"prompt not found: {path}")
        return path.read_text(encoding="utf-8")


def get_prompt_registry() -> PromptRegistry:
    s = get_settings()
    return PromptRegistry(base_dir=s.PROMPTS_DIR)
