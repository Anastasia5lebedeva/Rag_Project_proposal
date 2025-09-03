from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, List, Dict, Any, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from sentence_transformers import SentenceTransformer

from proposal_rag.config.settings import get_settings
from proposal_rag.services.document_processor import smart_chunk
from proposal_rag.services.vector_search import _vec_literal

log = logging.getLogger(__name__)
s = get_settings()


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(s.EMBED_MODEL, device=s.EMBED_DEVICE)


def _chunks(text: str) -> List[str]:
    return smart_chunk(
        text,
        profile=s.CHUNK_PROFILE,
        max_tokens=s.CHUNK_MAX_TOKENS,
        overlap_tokens=s.CHUNK_OVERLAP_TOKENS,
        min_tokens=s.CHUNK_MIN_TOKENS,
        max_chars=s.CHUNK_MAX_CHARS or None,
    )


def _embed_passages(passages: List[str]) -> List[List[float]]:
    if not passages:
        return []
    mdl = _get_embedder()
    inputs = [f"{s.PASSAGE_PREFIX}{p.strip()}" for p in passages]
    vecs = mdl.encode(inputs, normalize_embeddings=True)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]


def _vec_lits(vecs: Iterable[Iterable[float]]) -> List[str]:
    return [_vec_literal(v) for v in vecs]


def sync_segments(document_id: Optional[str] = None) -> int:
    total = 0
    with psycopg.connect(s.DSN, row_factory=dict_row, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SET search_path TO rag;")
            if document_id:
                cur.execute(
                    """
                    SELECT id, document_id, section_title, content, section_key, lang, order_idx
                    FROM rag.llm_contexts
                    WHERE document_id = %s
                    ORDER BY id
                    """,
                    (document_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, document_id, section_title, content, section_key, lang, order_idx
                    FROM rag.llm_contexts
                    ORDER BY id
                    """
                )
            rows = list(cur.fetchall())

        upsert_sql = f"""
        INSERT INTO rag.retriever_segments
          (context_id, chunk_index, text_norm, embedding, meta, section_key, lang)
        VALUES
          (%(context_id)s, %(chunk_index)s, %(text_norm)s, %(embedding_lit)s::vector({s.EMBED_DIM}), %(meta)s, %(section_key)s, %(lang)s)
        ON CONFLICT (context_id, chunk_index) DO UPDATE SET
          text_norm = EXCLUDED.text_norm,
          embedding = EXCLUDED.embedding,
          meta = EXCLUDED.meta,
          section_key = EXCLUDED.section_key,
          lang = EXCLUDED.lang
        """

        delete_tail_sql = """
        DELETE FROM rag.retriever_segments
        WHERE context_id = %(context_id)s AND chunk_index >= %(keep_from)s
        """

        batch: List[Dict[str, Any]] = []
        for r in rows:
            ctx_id = r["id"]
            content = r.get("content") or ""
            chunks = _chunks(content)
            if not chunks:
                with conn.cursor() as cur:
                    cur.execute(delete_tail_sql, {"context_id": ctx_id, "keep_from": 0})
                conn.commit()
                continue

            vecs = _embed_passages(chunks)
            lits = _vec_lits(vecs)
            for idx, (txt, lit) in enumerate(zip(chunks, lits)):
                meta = {
                    "document_id": r.get("document_id"),
                    "order_idx": r.get("order_idx"),
                    "section_title": r.get("section_title"),
                }
                batch.append(
                    {
                        "context_id": ctx_id,
                        "chunk_index": idx,
                        "text_norm": txt,
                        "embedding_lit": lit,
                        "meta": Json(meta),
                        "section_key": r.get("section_key"),
                        "lang": r.get("lang"),
                    }
                )
                if len(batch) >= s.SEGMENTS_BATCH_SIZE:
                    with conn.cursor() as cur:
                        cur.executemany(upsert_sql, batch)
                    conn.commit()
                    total += len(batch)
                    batch.clear()

            with conn.cursor() as cur:
                cur.execute(delete_tail_sql, {"context_id": ctx_id, "keep_from": len(chunks)})
            conn.commit()

        if batch:
            with conn.cursor() as cur:
                cur.executemany(upsert_sql, batch)
            conn.commit()
            total += len(batch)

    log.info("sync_segments done contexts=%d total_rows=%d", len(rows), total)
    return total
