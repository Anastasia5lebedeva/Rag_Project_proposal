from __future__ import annotations
import os
from importlib.resources import files as ir_files
from pathlib import Path
from typing import Optional

class PromptLoader:


    def __init__(self, base_dir: Optional[Path] = None, package: Optional[str] = "proposal_rag.resources.prompts"):
        self.base_dir = base_dir
        self.package = package



    def _from_package(self, name: str) -> Optional[str]:
        if not self.package:
            return None
        try:
            return (ir_files(self.package) / name).read_text(encoding="utf-8")
        except Exception:
            return None


    def _from_fs(self, name: str) -> Optional[str]:
        root = self.base_dir
        if root is None:
            root = Path(__file__).resolve().parents[3] / "resources" / "prompts"
        root = root.resolve()

        p = (root / name).resolve()
        if not str(p).startswith(str(root)):
            return None
        if p.is_file():
            return p.read_text(encoding="utf-8")
        return None


    def exists(self, name: str) -> bool:
        return (self._from_package(name) is not None) or (self._from_fs(name) is not None)

    def get_text(self, name: str) -> str:
        txt = self._from_package(name)
        if txt is None:
            txt = self._from_fs(name)
        if txt is None:
            raise FileNotFoundError(f"Prompt not found: {name}")
        return txt


def _env_base_dir() -> Optional[Path]:
    val = os.getenv("PROMPTS_DIR", "").strip()
    return Path(val).expanduser().resolve() if val else None

LOADER = PromptLoader(base_dir=_env_base_dir(), package="proposal_rag.resources.prompts")
