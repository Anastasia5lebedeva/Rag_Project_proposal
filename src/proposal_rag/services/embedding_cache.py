from __future__ import annotations

import hashlib
import logging
import unicodedata
from functools import lru_cache
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)
import asyncpg
from proposal_rag.config.settings import get_settings
from proposal_rag.services.llm_client import LLMClient



log = logging.getLogger(__name__)
s = get_settings()


def _resolve_embed_dim() -> int:
    dim = getattr(getattr(s, "embeddings", object()), "dim", None)
    if dim is None:
        dim = getattr(s, "EMBED_DIM", 1024)
    return int(dim)

EMBED_DIM: int = _resolve_embed_dim()


KIND_DOC = "doc"
KIND_QUERY = "query"
KIND_OTHER = "other"


DDL_EMBEDDING_CACHE = f"""
CREATE SCHEMA IF NOT EXISTS rag;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typnamespace = 'rag'::regnamespace AND typname = 'embedding_kind') THEN
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

-- Уникальность и быстрый поиск по ключу
CREATE UNIQUE INDEX IF NOT EXISTS uidx_embedding_cache_model_kind_hash
  ON rag.embedding_cache (model, kind, content_hash);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_kind_hash
  ON rag.embedding_cache (model, kind, content_hash);
"""


_POOL: Optional[asyncpg.Pool] = None

async def _get_pool() -> asyncpg.Pool:

    global _POOL
    dsn = getattr(s, "DATABASE_URL", "") or getattr(s, "DB_DSN", "")
    if not dsn:
        raise RuntimeError("Database DSN is empty (DATABASE_URL/DB_DSN). Configure .env")

    if _POOL is None:
        _POOL = await asyncpg.create_pool(
            dsn=dsn,
            min_size=int(getattr(s, "DB_POOL_MIN", 1)),
            max_size=int(getattr(s, "DB_POOL_MAX", 10)),
            command_timeout=float(getattr(s, "LLM_TIMEOUT_S", 30)),
        )
        log.info("embedding-cache pool initialized")
    return _POOL


@lru_cache(maxsize=1)
def _normalizer() -> Callable[[str], str]:
    def _clean(x: str) -> str:
        x = unicodedata.normalize("NFKC", x or "")
        return " ".join(x.split())
    return _clean

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def to_vector_literal(vec: Sequence[float]) -> str:
    vals = [float(v) for v in vec]
    return "[" + ",".join(f"{v:.6f}" for v in vals) + "]"

def _parse_vector_if_needed(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",") if x]
    try:
        return [float(value)]
    except Exception as e:
        raise TypeError(f"Unsupported vector type from DB: {type(value)}") from e

def _runtime_ddl_enabled() -> bool:
    env = str(getattr(s, "ENV", "")).lower()
    allow = bool(getattr(s, "ALLOW_RUNTIME_DDL", False))
    return allow or env in {"dev", "local", "test"}

async def ensure_table() -> None:
    if not _runtime_ddl_enabled():
        log.debug("skip ensure_table: runtime DDL disabled (migrations are source of truth)")
        return
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(DDL_EMBEDDING_CACHE)
    log.info("embedding-cache DDL ensured (dev)")


ComputeFn = Callable[[List[str]], Awaitable[List[List[float]]]]


async def get_or_compute(
    texts: Iterable[str],
    *,
    model: str,
    kind: str = KIND_DOC,
    compute_fn: ComputeFn,
    use_cache: Optional[bool] = None,
    ensure_ddl: bool = True,
    validate_dim: bool = True,
) -> List[List[float]]:

    in_texts = list(texts)
    if not in_texts:
        return []

    flag_cache = bool(getattr(s, "EMBED_CACHE_ENABLED", True)) if use_cache is None else bool(use_cache)
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
            emb = _parse_vector_if_needed(r["embedding"])
            have[r["content_hash"]] = emb

    missing_pairs: List[Tuple[str, str]] = [(h, t) for h, t in zip(hashes, norm_texts) if h not in have]
    if missing_pairs:
        miss_texts = [t for _, t in missing_pairs]
        computed = await compute_fn(miss_texts)
        if len(computed) != len(miss_texts):
            raise RuntimeError("compute_fn returned unexpected number of embeddings")

        to_insert: List[Tuple[str, str, str, str, str]] = []
        for (h, t), emb in zip(missing_pairs, computed):
            if validate_dim and len(emb) != EMBED_DIM:
                raise ValueError(f"Embedding dim mismatch: got {len(emb)}, expected {EMBED_DIM}")
            have[h] = emb
            to_insert.append(
                (
                    model,
                    kind,
                    h,
                    t,
                    to_vector_literal(emb),
                )
            )

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    f"""
                    INSERT INTO rag.embedding_cache(model, kind, content_hash, content_norm, embedding)
                    VALUES($1, $2::rag.embedding_kind, $3, $4, $5::vector({EMBED_DIM}))
                    ON CONFLICT (model, kind, content_hash) DO NOTHING
                    """,
                    to_insert,
                )
        log.info("embedding-cache inserted=%d (model=%s kind=%s)", len(to_insert), model, kind)

    result = [have[_hash(normalizer(t))] for t in in_texts]
    return result


async def get_or_compute_one(
    text: str,
    *,
    model: str,
    kind: str = KIND_QUERY,
    compute_fn: ComputeFn,
    **kwargs: Any,
) -> List[float]:
    vecs = await get_or_compute([text], model=model, kind=kind, compute_fn=compute_fn, **kwargs)
    return vecs[0]

async def llm_embed_compute_fn(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    async with LLMClient(model=(model or getattr(s, "EMBED_MODEL", ""))) as llm:
        return await llm.embed(texts)
