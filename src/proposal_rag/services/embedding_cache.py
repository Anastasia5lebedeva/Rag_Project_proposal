# src/proposal_rag/services/embedding_cache.py
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Awaitable, Callable, Iterable, List, Dict, Any

import asyncpg

from proposal_rag.config.settings import get_settings

log = logging.getLogger(__name__)
s = get_settings()

# -----------------------------------------------------------------------------
# DDL (one-time safety net). If you already manage migrations elsewhere, you can
# disable ensure_table() calls in get_or_compute().
# -----------------------------------------------------------------------------

EMBED_DIM: int = int(getattr(s, "EMBED_DIM", 1024))

DDL_EMBEDDING_CACHE = f"""
CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.embedding_cache (
  id BIGSERIAL PRIMARY KEY,
  model TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  content_norm TEXT NOT NULL,
  embedding vector({EMBED_DIM}) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS embedding_cache_uq
  ON rag.embedding_cache (model, content_hash);

-- Optional index if you will do ANN over cache (usually not needed):
-- CREATE INDEX IF NOT EXISTS embedding_cache_embedding_ivfflat
--   ON rag.embedding_cache USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

# -----------------------------------------------------------------------------
# Pool (lazy singleton)
# -----------------------------------------------------------------------------

_POOL: asyncpg.Pool | None = None


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


@lru_cache(maxsize=1)
def _normalizer() -> Callable[[str], str]:
    def _clean(x: str) -> str:
        # keep it simple and deterministic: strip + collapse whitespace
        return " ".join((x or "").split())
    return _clean


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_vector_literal(vec: Iterable[float]) -> str:
    vals = [float(v) for v in vec]
    return "[" + ",".join(f"{v:.6f}" for v in vals) + "]"


async def ensure_table() -> None:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(DDL_EMBEDDING_CACHE)
        # ANALYZE is optional, do after massive loads if you enable ivfflat
        # await conn.execute("ANALYZE rag.embedding_cache;")
    log.info("embedding-cache DDL ensured")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

ComputeFn = Callable[[List[str]], Awaitable[List[List[float]]]]


async def get_or_compute(
    texts: Iterable[str],
    model: str,
    compute_fn: ComputeFn,
    *,
    use_cache: bool | None = None,
    ensure_ddl: bool = True,
) -> List[List[float]]:
    """
    Return embeddings for texts using DB-backed cache.
    If disabled or DB not configured, compute directly via compute_fn.

    Args:
        texts: iterable of input texts
        model: embedding model id (part of the cache key)
        compute_fn: async function that takes List[str] and returns List[List[float]]
        use_cache: overrides settings flag EMBED_CACHE_ENABLED if provided
        ensure_ddl: run table DDL once per process (safe in prod, can disable if migrations are guaranteed)

    Returns:
        embeddings in the same order as input texts
    """
    in_texts = list(texts)
    if not in_texts:
        return []

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

    # Step 1: fetch existing
    have: Dict[str, List[float]] = {}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content_hash, embedding FROM rag.embedding_cache WHERE model = $1 AND content_hash = ANY($2)",
            model,
            hashes,
        )
        for r in rows:
            # asyncpg returns 'vector' as Python list[float] if adapter present; else text. Cast to list when needed.
            emb = r["embedding"]
            if isinstance(emb, str):
                # Parsing text literal back is possible but not needed if DB returns list. Warn instead.
                log.warning("embedding returned as text; consider enabling pgvector adapter")
                # naive parse to float list:
                emb = [float(x) for x in emb.strip("[]").split(",") if x]
            have[r["content_hash"]] = emb

    # Step 2: compute missing
    missing_pairs: List[tuple[str, str]] = [
        (h, t) for h, t in zip(hashes, norm_texts) if h not in have
    ]
    if missing_pairs:
        miss_texts = [t for _, t in missing_pairs]
        computed = await compute_fn(miss_texts)

        if len(computed) != len(miss_texts):
            raise RuntimeError("compute_fn returned unexpected number of embeddings")

        to_insert: List[Dict[str, Any]] = []
        for (h, t), emb in zip(missing_pairs, computed):
            have[h] = emb
            to_insert.append(
                {
                    "model": model,
                    "content_hash": h,
                    "content_norm": t,
                    "embedding_lit": _to_vector_literal(emb),
                }
            )

        # Step 3: upsert batch
        if to_insert:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        f"""
                        INSERT INTO rag.embedding_cache(model, content_hash, content_norm, embedding)
                        VALUES($1,$2,$3, $4::vector({EMBED_DIM}))
                        ON CONFLICT (model, content_hash) DO NOTHING
                        """,
                        [(r["model"], r["content_hash"], r["content_norm"], r["embedding_lit"]) for r in to_insert],
                    )
            log.info("embedding-cache inserted=%d", len(to_insert))

    # Step 4: return in original order
    result = [have[_hash(normalizer(t))] for t in in_texts]
    return result
