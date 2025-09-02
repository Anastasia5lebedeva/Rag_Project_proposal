# src/proposal_rag/services/embedding_cache.py
from __future__ import annotations

import hashlib
import logging
import unicodedata
from functools import lru_cache
from typing import Awaitable, Callable, Iterable, List, Dict, Any, Optional, Sequence

import asyncpg

from proposal_rag.config.settings import get_settings

log = logging.getLogger(__name__)
s = get_settings()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def _resolve_embed_dim() -> int:
    # Единая точка правды: сначала settings.embeddings.dim, потом EMBED_DIM, иначе 1024
    dim = getattr(getattr(s, "embeddings", object()), "dim", None)
    if dim is None:
        dim = getattr(s, "EMBED_DIM", 1024)
    return int(dim)

EMBED_DIM: int = _resolve_embed_dim()

# kinds для прозрачности: где используем вектор
KIND_DOC = "doc"       # эмбеддинги чанков документов
KIND_QUERY = "query"   # эмбеддинги пользовательских запросов
KIND_OTHER = "other"   # прочее / служебное

# -----------------------------------------------------------------------------
# DDL (можно отключить ensure_ddl в проде, если миграции применяются отдельно)
# -----------------------------------------------------------------------------

DDL_EMBEDDING_CACHE = f"""
CREATE SCHEMA IF NOT EXISTS rag;

-- ENUM для вида эмбеддинга (делает данные выразительнее)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'embedding_kind') THEN
        CREATE TYPE rag.embedding_kind AS ENUM ('doc','query','other');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS rag.embedding_cache (
  id            BIGSERIAL PRIMARY KEY,
  model         TEXT NOT NULL,
  kind          rag.embedding_kind NOT NULL,
  content_hash  TEXT NOT NULL,
  content_norm  TEXT NOT NULL,
  embedding     vector({EMBED_DIM}) NOT NULL,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS embedding_cache_uq
  ON rag.embedding_cache (model, kind, content_hash);

-- Опционально, если вдруг захочешь ANN по кэшу (обычно не нужно):
-- CREATE INDEX IF NOT EXISTS embedding_cache_embedding_ivfflat
--   ON rag.embedding_cache USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

# -----------------------------------------------------------------------------
# Pool (lazy singleton)
# -----------------------------------------------------------------------------

_POOL: Optional[asyncpg.Pool] = None

async def _get_pool() -> asyncpg.Pool:
    global _POOL
    dsn = getattr(s, "DATABASE_URL", "") or getattr(s, "DB_DSN", "")
    if not dsn:
        raise RuntimeError("Database DSN is empty (DATABASE_URL/DB_DSN). Configure .env")
    if _POOL is None:
        _POOL = await asyncpg.create_pool(
            dsn=dsn,
            min_size=getattr(s, "DB_POOL_MIN", 1),
            max_size=getattr(s, "DB_POOL_MAX", 10),
            command_timeout=getattr(s, "LLM_TIMEOUT_S", 30),
        )
        log.info("embedding-cache pool initialized")
    return _POOL

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _normalizer() -> Callable[[str], str]:
    def _clean(x: str) -> str:
        # Детеминированная нормализация: Unicode NFKC + trim + схлопнуть пробелы
        x = unicodedata.normalize("NFKC", x or "")
        return " ".join(x.split())
    return _clean

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _to_vector_literal(vec: Sequence[float]) -> str:
    vals = [float(v) for v in vec]
    return "[" + ",".join(f"{v:.6f}" for v in vals) + "]"

async def ensure_table() -> None:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(DDL_EMBEDDING_CACHE)
    log.info("embedding-cache DDL ensured")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

ComputeFn = Callable[[List[str]], Awaitable[List[List[float]]]]

async def get_or_compute(
    texts: Iterable[str],
    *,
    model: str,
    kind: str = KIND_DOC,         # <--- явное разделение: 'doc' | 'query' | 'other'
    compute_fn: ComputeFn,
    use_cache: Optional[bool] = None,
    ensure_ddl: bool = True,
    validate_dim: bool = True,    # проверять длину вектора перед вставкой
) -> List[List[float]]:
    """
    Вернёт эмбеддинги для текстов, используя DB-кэш (rag.embedding_cache).
    Если кэш выключен или не сконфигурирован DSN — считает напрямую через compute_fn.

    Args:
        texts: iterable входных строк
        model: идентификатор модели эмбеддингов (часть ключа кэша)
        kind: 'doc' | 'query' | 'other' (для прозрачности в БД и будущей аналитики)
        compute_fn: async функция List[str] -> List[List[float]]
        use_cache: перекрыть флаг из настроек (EMBED_CACHE_ENABLED)
        ensure_ddl: выполнять DDL один раз за процесс (в проде можно False)
        validate_dim: проверять совпадение размерности эмбеддинга с EMBED_DIM

    Returns:
        Список эмбеддингов в исходном порядке входных текстов.
    """
    in_texts = list(texts)
    if not in_texts:
        return []

    # Bypass?
    flag_cache = s.EMBED_CACHE_ENABLED if use_cache is None else bool(use_cache)
    dsn_present = bool(getattr(s, "DATABASE_URL", "") or getattr(s, "DB_DSN", ""))
    if not flag_cache or not dsn_present:
        log.debug("embedding-cache bypassed (flag_cache=%s dsn_present=%s)", flag_cache, dsn_present)
        return await compute_fn(in_texts)

    normalizer = _normalizer()
    norm_texts = [normalizer(t) for t in in_texts]
    hashes = [_hash(t) for t in norm_texts]

    pool = await _get_pool()
    if ensure_ddl:
        await ensure_table()

    # 1) Чтение уже посчитанных
    have: Dict[str, List[float]] = {}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content_hash, embedding
            FROM rag.embedding_cache
            WHERE model = $1 AND kind = $2::rag.embedding_kind AND content_hash = ANY($3)
            """,
            model,
            kind,
            hashes,
        )
        for r in rows:
            emb = r["embedding"]
            if isinstance(emb, str):
                # если pgvector не отдает list[float], парсим текстовое представление
                log.warning("embedding returned as text; consider enabling pgvector adapter for asyncpg")
                emb = [float(x) for x in emb.strip("[]").split(",") if x]
            have[r["content_hash"]] = emb  # type: ignore[assignment]

    # 2) Досчитать недостающее
    missing_pairs: List[tuple[str, str]] = [(h, t) for h, t in zip(hashes, norm_texts) if h not in have]
    if missing_pairs:
        miss_texts = [t for _, t in missing_pairs]
        computed = await compute_fn(miss_texts)
        if len(computed) != len(miss_texts):
            raise RuntimeError("compute_fn returned unexpected number of embeddings")

        to_insert: List[Dict[str, Any]] = []
        for (h, t), emb in zip(missing_pairs, computed):
            if validate_dim and len(emb) != EMBED_DIM:
                raise ValueError(f"Embedding dim mismatch: got {len(emb)}, expected {EMBED_DIM}")
            have[h] = emb
            to_insert.append(
                {
                    "model": model,
                    "kind": kind,
                    "content_hash": h,
                    "content_norm": t,
                    "embedding_lit": _to_vector_literal(emb),
                }
            )

        if to_insert:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        f"""
                        INSERT INTO rag.embedding_cache(model, kind, content_hash, content_norm, embedding)
                        VALUES($1, $2::rag.embedding_kind, $3, $4, $5::vector({EMBED_DIM}))
                        ON CONFLICT (model, kind, content_hash) DO NOTHING
                        """,
                        [(r["model"], r["kind"], r["content_hash"], r["content_norm"], r["embedding_lit"]) for r in to_insert],
                    )
            log.info("embedding-cache inserted=%d (model=%s kind=%s)", len(to_insert), model, kind)

    # 3) Вернуть в исходном порядке
    result = [have[_hash(normalizer(t))] for t in in_texts]
    return result


# Удобная обёртка для одиночного текста
async def get_or_compute_one(
    text: str,
    *,
    model: str,
    kind: str = KIND_QUERY,
    compute_fn: Callable[[List[str]], Awaitable[List[List[float]]]],
    **kwargs: Any,
) -> List[float]:
    vec = await get_or_compute([text], model=model, kind=kind, compute_fn=compute_fn, **kwargs)
    return vec[0]
