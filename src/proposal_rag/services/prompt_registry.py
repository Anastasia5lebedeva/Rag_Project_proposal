# src/proposal_rag/services/prompt_registry.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Iterable, Dict
from threading import RLock
import hashlib
import os

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound, select_autoescape

class PromptNotFound(FileNotFoundError):
    pass

def _safe_join(base: Path, name: str) -> Path:
    """
    Безопасно склеивает базовый каталог и имя файла, запрещая выход за пределы base.
    """
    candidate = (base / name).resolve()
    base_resolved = base.resolve()
    if not str(candidate).startswith(str(base_resolved) + os.sep) and candidate != base_resolved:
        raise PromptNotFound(f"Illegal prompt path: {name}")
    return candidate

@dataclass(frozen=True)
class _StatKey:
    size: int
    mtime_ns: int

def _stat_key(p: Path) -> _StatKey:
    st = p.stat()
    return _StatKey(size=st.st_size, mtime_ns=st.st_mtime_ns)

class PromptLoader:
    """
    Реестр файловых промптов с безопасным рендером и «умным» кэшированием.
    - База: каталог с промптами (обычно resources/prompts)
    - Кэш: текст и скомпилированные шаблоны, автоинвалидация при изменении файла
    - Jinja2: StrictUndefined (ошибка на опечатках ключей), поддержка include/extends
    """

    def __init__(
        self,
        base: Path,
        *,
        allowed_suffixes: Iterable[str] = (".j2", ".jinja", ".md", ".txt"),
        autoescape_for: Iterable[str] = (".html", ".xml"),
    ) -> None:
        self.base = Path(base).resolve()
        if not self.base.is_dir():
            raise NotADirectoryError(f"Prompts base is not a directory: {self.base}")

        self.allowed_suffixes = tuple(allowed_suffixes)
        self._lock = RLock()

        # Единый Jinja Environment с файловым загрузчиком
        self._env = Environment(
            loader=FileSystemLoader(str(self.base)),
            undefined=StrictUndefined,
            autoescape=select_autoescape(default=False, enabled_extensions=tuple(autoescape_for)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Кэши
        self._text_cache: Dict[str, str] = {}
        self._tpl_cache: Dict[str, Any] = {}  # compiled template
        self._stat_cache: Dict[str, _StatKey] = {}

    # ---------- discovery ----------

    def list_names(self) -> list[str]:
        """Вернёт относительные пути всех файлов с допустимыми суффиксами."""
        names: list[str] = []
        for p in self.base.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.allowed_suffixes:
                names.append(str(p.relative_to(self.base)))
        names.sort()
        return names

    def exists(self, name: str) -> bool:
        try:
            path = _safe_join(self.base, name)
        except PromptNotFound:
            return False
        return path.is_file() and path.suffix.lower() in self.allowed_suffixes

    # ---------- low-level file I/O with cache ----------

    def _read_text(self, name: str) -> str:
        path = _safe_join(self.base, name)
        if not path.exists() or not path.is_file():
            raise PromptNotFound(f"Prompt not found: {name}")

        if path.suffix.lower() not in self.allowed_suffixes:
            raise PromptNotFound(f"Unsupported prompt suffix for: {name}")

        sk = _stat_key(path)
        cached_stat = self._stat_cache.get(name)
        if cached_stat == sk:
            # стат совпал — можно вернуть кэшированный текст, если есть
            txt = self._text_cache.get(name)
            if txt is not None:
                return txt

        # читаем заново
        text = path.read_text(encoding="utf-8")
        self._stat_cache[name] = sk
        self._text_cache[name] = text
        return text

    def get_text(self, name: str) -> str:
        """Сырой текст шаблона без рендера."""
        with self._lock:
            return self._read_text(name)

    # ---------- template compile & render ----------

    def _get_compiled(self, name: str):
        """
        Возвращает скомпилированный шаблон, инвалидируя кэш при изменении файла.
        """
        path = _safe_join(self.base, name)
        if not path.exists() or not path.is_file():
            raise PromptNotFound(f"Prompt not found: {name}")

        sk = _stat_key(path)
        cached_stat = self._stat_cache.get(name)
        if cached_stat != sk or name not in self._tpl_cache:
            # Инвалидация: файл изменился или компиляции ещё не было
            try:
                tpl = self._env.get_template(name)
            except TemplateNotFound:
                # на случай, если Jinja не нашла; синхронизируем сообщение со стилем NotFound
                raise PromptNotFound(f"Prompt not found: {name}")
            self._tpl_cache[name] = tpl
            self._stat_cache[name] = sk

        return self._tpl_cache[name]

    def render(self, name: str, context: Optional[Mapping[str, Any]] = None) -> str:
        """
        Рендер шаблона. StrictUndefined заставит падать при опечатках ключей.
        """
        ctx = dict(context or {})
        with self._lock:
            tpl = self._get_compiled(name)
            return tpl.render(**ctx)

    # ---------- cache control ----------

    def invalidate(self, name: Optional[str] = None) -> None:
        """Инвалидировать один шаблон или весь кэш."""
        with self._lock:
            if name is None:
                self._text_cache.clear()
                self._tpl_cache.clear()
                self._stat_cache.clear()
            else:
                self._text_cache.pop(name, None)
                self._tpl_cache.pop(name, None)
                self._stat_cache.pop(name, None)

    def clear_cache(self) -> None:
        self.invalidate(None)



