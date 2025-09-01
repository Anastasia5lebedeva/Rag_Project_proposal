# src/proposal_rag/services/sync_segments.py
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from functools import lru_cache
from typing import Iterable, List, Tuple, Dict, Any, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from proposal_rag.config.logging_config import setup_logging
from proposal_rag.config.settings import get_settings
from sentence_transformers import SentenceTransformer

# Try to import your chunker; if not present, use a safe fallback
try:
    from proposal_rag.services.document_processor import smart_chunk as _smart_chunk
except Exception:  # noqa: BLE001
    _smart_chunk = None  # fallback defined below


# ----------------------------- Settings & Logger -----------------------------

s = get_settings()
setup_logging(s.LOG_LEVEL)
log = logging.getLogger(__name__)

EMBED_DIM: int = int(getattr(s, "EMBED_DIM", 1024))
DEFAULT_MAX_CHARS: int = int(getattr(s, "TOKEN_LIMIT_PER_CHUNK", 3000))  # conservative char budget
DEFAULT_OVERLAP_CHARS: int = 300
DEFAULT_MIN_CHARS: int = 400
DEFAULT_BATCH_SIZE: int = 200


# ----------------------------- Arg parsing ----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sync_segments",
        description="Chunk LLM contexts and upsert retriever segments with embeddings."
    )
    p.add_argument("--dsn", default=(getattr(s, "DATABASE_URL", "") or getattr(s, "DB_DSN", "")),
                   help="Postgres DSN. Falls back to Settings.DATABASE_URL / DB_DSN.")
    p.add_argument("--provider", default="sentence_transformers",
                   choices=["sentence_transformers"], help="Embedding provider.")
    p.add_argument("--model", default=s.EMBED_MODEL, help="Embedding model id.")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Max chars per chunk.")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_CHARS, help="Chars overlap between chunks.")
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="Minimum chars per chunk after cleanup.")
    p.add_argument("--filter-doc", default=None, help="Process only this document_id.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="UPSERT batch size.")
    p.add_argument("--dry-run", action="store_true", help="Do not write to DB.")
    return p.parse_args()


# ----------------------------- Embedding ------------------------------------

@lru_cache(maxsize=1)
def _get_embedder(provider: str, model_id: str) -> SentenceTransformer:
    if provider != "sentence_transformers":
        raise ValueError(f"Unsupported provider: {provider}")
    mdl = SentenceTransformer(model_id, device="cpu")
    return mdl


def embed_texts(texts: List[str], provider: str, model_id: str) -> List[List[float]]:
    if not texts:
        return []
    mdl = _get_embedder(provider, model_id)
    vecs = mdl.encode(texts, normalize_embeddings=True)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]


def vec_literal(vec: Iterable[float]) -> str:
    # Convert to PG vector literal; allows ::vector(EMBED_DIM) cast in SQL.
    vals = [float(x) for x in vec]
    return "[" + ",".join(f"{x:.6f}" for x in vals) + "]"


# ----------------------------- Chunking -------------------------------------

def smart_chunk_fallback(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int
) -> List[str]:
    # Token-agnostic, char-based conservative chunker
    if not text:
        return []
    maxc = max(200, max_tokens * 4)          # very rough chars per token ~0.25
    overc = max(0, overlap_tokens * 4)
    minc = max(100, min_tokens * 4)

    text = " ".join(text.split())
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + maxc)
        chunk = text[i:j]
        if len(chunk) >= minc:
            out.append(chunk)
        i = j - overc if j - overc > i else j
    return out


def smart_chunk(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int
) -> List[str]:
    if _smart_chunk is not None:
        return _smart_chunk(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens)
    return smart_chunk_fallback(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens)


# ----------------------------- SQL ------------------------------------------

DDL = f"""
CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.retriever_segments (
  id BIGSERIAL PRIMARY KEY,
  context_id BIGINT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text_norm TEXT NOT NULL,
  embedding_1024 vector({EMBED_DIM}),
  meta JSONB,
  section_key TEXT,
  lang TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS retriever_segments_uq
  ON rag.retriever_segments (context_id, chunk_index);

CREATE INDEX IF NOT EXISTS retriever_segments_meta_gin
  ON rag.retriever_segments USING GIN (meta);

-- Optional: ivfflat index requires ANALYZE after significant inserts
-- CREATE INDEX IF NOT EXISTS retriever_segments_embedding_ivfflat
--   ON rag.retriever_segments USING ivfflat (embedding_1024 vector_cosine_ops) WITH (lists = 100);
"""

SELECT_CONTEXTS = """
SELECT
  lc.id            AS context_id,
  lc.document_id   AS document_id,
  lc.section_title AS section_title,
  lc.content       AS content,
  lc.order_idx     AS order_idx,
  lc.section_key   AS section_key,
  lc.lang          AS lang,
  md5(coalesce(lc.content, '')) AS source_hash
FROM rag.llm_contexts lc
ORDER BY lc.id;
"""

# Using text literal + ::vector(EMBED_DIM) to avoid custom adapters
UPSERT_SEGMENT = f"""
INSERT INTO rag.retriever_segments
  (context_id, chunk_index, text_norm, embedding_1024, meta, section_key, lang)
VALUES
  (%(context_id)s, %(chunk_index)s, %(text_norm)s, %(embedding_lit)s::vector({EMBED_DIM}), %(meta)s, %(section_key)s, %(lang)s)
ON CONFLICT (context_id, chunk_index) DO UPDATE SET
  text_norm = EXCLUDED.text_norm,
  embedding_1024 = EXCLUDED.embedding_1024,
  meta = EXCLUDED.meta,
  section_key = EXCLUDED.section_key,
  lang = EXCLUDED.lang;
"""

DELETE_TAIL_SEGMENTS = """
DELETE FROM rag.retriever_segments
WHERE context_id = %(context_id)s AND chunk_index >= %(keep_from)s;
"""


# ----------------------------- Main -----------------------------------------

def _require_dsn(dsn: Optional[str]) -> str:
    if not dsn or not dsn.strip():
        raise ValueError("Database DSN is empty. Provide --dsn or set DATABASE_URL / DB_DSN in .env")
    return dsn


def _rows_filter_by_doc(rows: List[Dict[str, Any]], document_id: Optional[str]) -> List[Dict[str, Any]]:
    if not document_id:
        return rows
    return [r for r in rows if r.get("document_id") == document_id]


def _commit_batch(cur, batch: List[Dict[str, Any]]) -> None:
    if not batch:
        return
    cur.executemany(UPSERT_SEGMENT, batch)
    batch.clear()


def main() -> None:
    args = parse_args()
    trace_id = str(uuid.uuid4())
    log.info("sync_segments start trace_id=%s", trace_id)

    dsn = _require_dsn(args.dsn)

    try:
        with psycopg.connect(dsn, row_factory=dict_row, autocommit=False) as conn:
            with conn.cursor() as cur:
                # Ensure base DDL
                cur.execute(DDL)
                conn.commit()

                # Load all contexts
                cur.execute(SELECT_CONTEXTS)
                rows = list(cur.fetchall())
                rows = _rows_filter_by_doc(rows, args.filter_doc)
                log.info("contexts_to_process=%d filter_doc=%s", len(rows), args.filter_doc or "")

                total_chunks = 0
                upsert_batch: List[Dict[str, Any]] = []

                for r in rows:
                    ctx_id = r["context_id"]
                    content = r.get("content") or ""
                    section_key = r.get("section_key")
                    lang = r.get("lang")
                    source_hash = r.get("source_hash")

                    chunks = smart_chunk(
                        content,
                        max_tokens=max(128, args.max_chars // 4),
                        overlap_tokens=max(0, args.overlap // 4),
                        min_tokens=max(64, args.min_chars // 4),
                    )
                    if not chunks:
                        log.warning("no_chunks context_id=%s document_id=%s", ctx_id, r.get("document_id"))
                        # Still clean tail if any
                        cur.execute(DELETE_TAIL_SEGMENTS, {"context_id": ctx_id, "keep_from": 0})
                        continue

                    # Encode embeddings
                    vecs = embed_texts(chunks, provider=args.provider, model_id=args.model)

                    for idx, (txt, vec) in enumerate(zip(chunks, vecs)):
                        meta = {
                            "document_id": r.get("document_id"),
                            "order_idx": r.get("order_idx"),
                            "section_title": r.get("section_title"),
                            "source_hash": source_hash,
                        }
                        upsert_batch.append(
                            {
                                "context_id": ctx_id,
                                "chunk_index": idx,
                                "text_norm": txt,
                                "embedding_lit": vec_literal(vec),
                                "meta": Json(meta),
                                "section_key": section_key,
                                "lang": lang,
                            }
                        )

                        if len(upsert_batch) >= args.batch_size and not args.dry_run:
                            _commit_batch(cur, upsert_batch)
                            conn.commit()

                    total_chunks += len(chunks)

                    # Trim tail segments if we reduced chunk count
                    if not args.dry_run:
                        cur.execute(
                            DELETE_TAIL_SEGMENTS,
                            {"context_id": ctx_id, "keep_from": len(chunks)},
                        )
                        # Commit per-context to keep transactions short
                        if upsert_batch:
                            _commit_batch(cur, upsert_batch)
                        conn.commit()

                if not args.dry_run and upsert_batch:
                    _commit_batch(cur, upsert_batch)
                    conn.commit()

                log.info("sync_segments done trace_id=%s total_chunks=%d", trace_id, total_chunks)

    except psycopg.Error as e:
        log.exception("database error trace_id=%s", trace_id)
        # Avoid leaving open transaction
        raise
    except Exception:
        log.exception("fatal error trace_id=%s", trace_id)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as _e:  # noqa: N816
        # last line to make sure exit code != 0 and stacktrace is printed once
        sys.exit(1)
