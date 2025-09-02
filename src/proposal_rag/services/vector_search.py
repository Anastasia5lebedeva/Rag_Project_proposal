from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer

from proposal_rag.config.settings import get_settings
from proposal_rag.repositories.search_repository import (
    search_all,
    enrich_rows_with_doc_and_ord,
)

log = logging.getLogger(__name__)
s = get_settings()


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(s.EMBED_MODEL, device=s.EMBED_DEVICE)


def _vec_literal(vec: Any) -> str:
    data = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return "[" + ",".join(f"{float(x):.6f}" for x in data) + "]"


def _embed_query(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("query text is empty")
    embedder = _get_embedder()
    vec = embedder.encode([f"{s.QUERY_PREFIX}{text.strip()}"], normalize_embeddings=True)[0]
    if vec is None or not np.isfinite(vec).all():
        log.error("invalid embedding for query: %s", text)
        raise ValueError("invalid embedding generated")
    return _vec_literal(vec)


def check_vector_db() -> bool:
    try:
        with psycopg.connect(s.DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM rag.retriever_segments LIMIT 1;")
            cur.fetchone()
            return True
    except Exception as e:
        log.error("Vector DB check failed: %s", e)
        return False


def search_hybrid(query: str, top_k: int) -> List[Dict[str, Any]]:
    if not isinstance(query, str) or len(query.strip()) < s.MIN_QUERY_LEN:
        raise ValueError(f"query must be at least {s.MIN_QUERY_LEN} characters")
    if not isinstance(top_k, int) or top_k < 1 or top_k > s.MAX_TOP_K:
        raise ValueError(f"top_k must be in range [1..{s.MAX_TOP_K}]")

    q_vec_lit = _embed_query(query)

    t0 = time.perf_counter()
    rows, mode = search_all(q_text_short=query.strip(), q_vec_lit=q_vec_lit, top_k=top_k)
    elapsed = time.perf_counter() - t0
    log.info("retrieval mode=%s rows=%d elapsed=%.3fs", mode, len(rows), elapsed)

    rows = enrich_rows_with_doc_and_ord(rows)

    hits: List[Dict[str, Any]] = []
    for r in rows[:top_k]:
        score = r.get("score") or r.get("cos_sim")
        hit = {
            "chunk_index": r.get("chunk_index"),
            "score": float(score) if score is not None else None,
            "preview": r.get("preview") or "",
            "source_meta": {
                "id": r.get("id"),
                "context_id": r.get("context_id"),
                "document_id": r.get("document_id"),
                "section_key": r.get("section_key"),
                "section_title": r.get("section_title"),
                "order_idx": r.get("order_idx"),
            },
        }
        hits.append(hit)
    return hits
