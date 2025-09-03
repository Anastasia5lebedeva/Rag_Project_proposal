from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Iterable

import psycopg

from proposal_rag.config.settings import get_settings
from proposal_rag.repositories.search_repository import (
    search_all,
    enrich_rows_with_doc_and_ord,
)
from proposal_rag.services.embedding_cache import (
    get_or_compute_one,
    KIND_QUERY,
    llm_embed_compute_fn,
    to_vector_literal,
)

log = logging.getLogger(__name__)
s = get_settings()

MIN_QUERY_LEN: int = int(getattr(s, "MIN_QUERY_LEN", 3))
MAX_TOP_K: int = int(getattr(s, "MAX_TOP_K", 100))


def _check_vector_db_sync() -> bool:
    try:
        with psycopg.connect(s.DSN) as conn, conn.cursor() as cur:
            cur.execute(f'SELECT 1 FROM "{s.DB_SCHEMA}".retriever_segments LIMIT 1;')
            cur.fetchone()
            return True
    except Exception as e:
        log.error("Vector DB check failed: %s", e)
        return False


async def check_vector_db() -> bool:
    return await asyncio.to_thread(_check_vector_db_sync)


def _search_all_sync(q_text_short: str, q_vec_lit: str, top_k: int):
    return search_all(q_text_short=q_text_short, q_vec_lit=q_vec_lit, top_k=top_k)


def _enrich_rows_sync(rows):
    return enrich_rows_with_doc_and_ord(rows)


def _vec_literal(vec: Iterable[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _vec_lits(vecs: Iterable[Iterable[float]]) -> List[str]:
    return [_vec_literal(v) for v in vecs]



async def search_hybrid(query: str, top_k: int) -> List[Dict[str, Any]]:

    min_q = int(getattr(s, "MIN_QUERY_LEN", 3))
    max_k = int(getattr(s, "MAX_TOP_K", 100))

    if not isinstance(query, str) or len(query.strip()) < min_q:
        raise ValueError(f"query must be at least {min_q} characters")
    if not isinstance(top_k, int) or not (1 <= top_k <= max_k):
        raise ValueError(f"top_k must be in range [1..{max_k}]")

    q_text = query.strip()

    q_vec = await get_or_compute_one(
        q_text,
        model=s.EMBED_MODEL,
        kind=KIND_QUERY,
        compute_fn=lambda xs: llm_embed_compute_fn(xs, model=s.EMBED_MODEL),
    )
    q_vec_lit = to_vector_literal(q_vec)

    t0 = time.perf_counter()
    rows, mode = await asyncio.to_thread(_search_all_sync, q_text, q_vec_lit, top_k)
    elapsed = time.perf_counter() - t0
    log.info("retrieval mode=%s rows=%d elapsed=%.3fs", mode, len(rows or []), elapsed)

    rows = await asyncio.to_thread(_enrich_rows_sync, rows)
    rows = rows or []


    def _norm_score(r: Dict[str, Any]) -> float:
        if r.get("cosine_distance") is not None:
            try:
                v = 1.0 - float(r["cosine_distance"])
            except Exception:
                v = 0.0
        else:
            raw = r.get("score", r.get("cos_sim", 0.0))
            try:
                v = float(raw)
            except Exception:
                v = 0.0
        return max(-1.0, min(1.0, v))

    def _norm_preview(r: Dict[str, Any]) -> str:
        for k in ("preview", "section_title", "text", "content"):
            v = (r.get(k) or "").strip()
            if v:
                return v
        return "…"

    def _norm_chunk_index(r: Dict[str, Any]) -> int:
        try:
            return max(0, int(r.get("chunk_index", 0)))
        except Exception:
            return 0

    def _norm_meta(r: Dict[str, Any]) -> Dict[str, Any] | None:
        meta = {
            "id": r.get("id"),
            "context_id": r.get("context_id"),
            "document_id": r.get("document_id"),
            "section_key": r.get("section_key"),
            "section_title": r.get("section_title"),
            "order_idx": r.get("order_idx"),
        }
        meta = {k: v for k, v in meta.items() if v is not None}
        return meta or None

    hits: List[Dict[str, Any]] = []
    for r in rows[:top_k]:
        hits.append({
            "chunk_index": _norm_chunk_index(r),
            "score": _norm_score(r),
            "preview": _norm_preview(r),
            "source_meta": _norm_meta(r),
        })

    return hits

