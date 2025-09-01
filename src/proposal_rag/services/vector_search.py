# src/proposal_rag/services/vector_search.py
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from proposal_rag.config.settings import get_settings
from proposal_rag.repositories.search_repository import (
    search_all,
    enrich_rows_with_doc_and_ord,
)

log = logging.getLogger(__name__)
s = get_settings()


# ---- Model loading (cached) -------------------------------------------------

@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    """
    Lazy, cached embedder instance.
    """
    model = SentenceTransformer(s.EMBED_MODEL, device="cpu")
    # Keep max seq length configurable via settings if needed later.
    return model


# ---- Helpers ----------------------------------------------------------------

def _vec_literal(vec: Any) -> str:
    """
    Convert vector (numpy/list) to PostgreSQL vector literal: "[0.1,0.2,...]".
    """
    data = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return "[" + ",".join(f"{float(x):.6f}" for x in data) + "]"


def _embed_query(text: str) -> str:
    """
    Build embedding for query text using E5-style prefix and return as literal.
    """
    if not text or not text.strip():
        raise ValueError("query text is empty")
    embedder = _get_embedder()
    # E5-style prefix improves retrieval quality
    vec = embedder.encode([f"query: {text.strip()}"], normalize_embeddings=True)[0]
    return _vec_literal(vec)


# ---- Public API -------------------------------------------------------------

def search_hybrid(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    High-level retrieval:
    1) embed query,
    2) repository.search_all (tries rag.search_hybrid, falls back to ANN),
    3) enrich with doc/order fields,
    4) map to API-friendly dicts.
    """
    if not isinstance(query, str) or len(query.strip()) < s.MIN_QUERY_LEN:
        raise ValueError(f"query must be at least {s.MIN_QUERY_LEN} characters")

    if not isinstance(top_k, int) or top_k < 1 or top_k > s.MAX_TOP_K:
        raise ValueError(f"top_k must be in range [1..{s.MAX_TOP_K}]")

    q_vec_lit = _embed_query(query)
    rows, mode = search_all(q_text_short=query.strip(), q_vec_lit=q_vec_lit, top_k=top_k)
    log.info("retrieval mode=%s raw_rows=%d", mode, len(rows))

    rows = enrich_rows_with_doc_and_ord(rows)

    # Map repository rows -> API SearchHit
    hits: List[Dict[str, Any]] = []
    for r in rows[:top_k]:
        # repository returns dict-row with columns selected in SQL
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
