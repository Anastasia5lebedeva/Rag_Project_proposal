from __future__ import annotations
import logging
import time
from typing import Any, Dict, Iterable, List, Tuple

import psycopg
from psycopg.rows import dict_row

from proposal_rag.config.settings import get_settings
from proposal_rag.api.errors import VectorDBError, DatabaseError

log = logging.getLogger(__name__)
s = get_settings()

EMBED_DIM: int = s.EMBED_DIM
MIN_QUERY_LEN: int = s.MIN_QUERY_LEN
MAX_TOP_K: int = s.MAX_TOP_K
SEARCH_BOOST: int = s.SEARCH_VECTOR_BOOST
SEARCH_MIN_CANDIDATES: int = s.SEARCH_MIN_CANDIDATES
DB_SCHEMA: str = s.DB_SCHEMA
EMBED_COL: str = s.EMBED_COLUMN


def _get_dsn() -> str:
    dsn = getattr(s, "DSN", "") or getattr(s, "DATABASE_URL", "")
    if not dsn:
        raise DatabaseError("database dsn is empty")
    return dsn


def search_all(q_text_short: str, q_vec_lit: str, top_k: int) -> Tuple[List[Dict[str, Any]], str]:
    dsn = _get_dsn()
    cand = max(int(top_k) * SEARCH_BOOST, SEARCH_MIN_CANDIDATES)

    t0 = time.perf_counter()
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(f"SET search_path TO {DB_SCHEMA};")

            try:
                t_h0 = time.perf_counter()
                cur.execute(
                    f"""
                    SELECT id,
                           context_id,
                           chunk_index,
                           cos_sim,
                           preview,
                           document_id,
                           section_key,
                           section_title
                    FROM {DB_SCHEMA}.search_hybrid(%s, %s::vector({EMBED_DIM}), %s, NULL::text[]);
                    """,
                    (q_text_short, q_vec_lit, cand),
                )
                rows = cur.fetchall()
                if rows:
                    log.info(
                        "search_hybrid ok rows=%d cand=%d top_k=%d elapsed=%.3fs",
                        len(rows), cand, top_k, (time.perf_counter() - t_h0),
                    )
                    return _limit_and_format(rows, top_k), "HYBRID"
            except Exception as e:
                log.warning("search_hybrid failed: %s; falling back to ANN", e)

            t_a0 = time.perf_counter()
            cur.execute(
                f"""
                WITH params AS (
                  SELECT %s::vector({EMBED_DIM}) AS qv
                ),
                base AS (
                  SELECT id, context_id, chunk_index, text_norm, {EMBED_COL}, section_key
                  FROM {DB_SCHEMA}.retriever_segments
                  WHERE {EMBED_COL} IS NOT NULL
                ),
                scored AS (
                  SELECT b.*,
                         1 - ({EMBED_COL} <=> p.qv) AS cos_sim
                  FROM base b, params p
                )
                SELECT s.id,
                       s.context_id,
                       s.chunk_index,
                       s.cos_sim,
                       LEFT(s.text_norm, 300) AS preview,
                       lc.document_id,
                       COALESCE(s.section_key, lc.section_key) AS section_key,
                       lc.section_title
                FROM scored s
                LEFT JOIN {DB_SCHEMA}.llm_contexts lc ON lc.id = s.context_id
                ORDER BY s.cos_sim DESC
                LIMIT %s;
                """,
                (q_vec_lit, cand),
            )
            rows = cur.fetchall()
            log.info(
                "ANN fallback ok rows=%d cand=%d top_k=%d elapsed=%.3fs",
                len(rows), cand, top_k, (time.perf_counter() - t_a0),
            )
            return _limit_and_format(rows, top_k), "FALLBACK"


    except psycopg.Error as e:
        raise VectorDBError("database vector search failed", extra={"reason": str(e)}) from e
    finally:
        log.debug("search_all total_elapsed=%.3fs", (time.perf_counter() - t0))


def enrich_rows_with_doc_and_ord(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_list = list(rows)
    if not rows_list:
        return []

    ids = [r["id"] for r in rows_list if r.get("id") is not None]
    if not ids:
        return rows_list

    by_id: Dict[int, Dict[str, Any]] = {int(r["id"]): dict(r) for r in rows_list if r.get("id") is not None}

    try:
        t0 = time.perf_counter()
        dsn = _get_dsn()
        with psycopg.connect(dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(f"SET search_path TO {DB_SCHEMA};")
            cur.execute(
                f"""
                SELECT rs.id,
                       lc.document_id,
                       lc.order_idx,
                       COALESCE(rs.section_key, lc.section_key) AS section_key,
                       lc.section_title
                FROM {DB_SCHEMA}.retriever_segments rs
                LEFT JOIN {DB_SCHEMA}.llm_contexts lc ON lc.id = rs.context_id
                WHERE rs.id = ANY(%s)
                """,
                (ids,),
            )
            for rec in cur.fetchall():
                rid = rec["id"]
                row = by_id.get(rid)
                if row is not None:
                    row.setdefault("document_id", rec["document_id"])
                    row.setdefault("section_key", rec["section_key"])
                    row.setdefault("section_title", rec["section_title"])
                    row.setdefault("order_idx", rec["order_idx"])
        log.debug("enrich_rows elapsed=%.3fs count=%d", (time.perf_counter() - t0), len(ids))

    except psycopg.Error as e:
        raise DatabaseError("failed to enrich rows", extra={"reason": str(e)}) from e

    return [by_id[i] for i in ids if i in by_id]



def _limit_and_format(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    n = min(len(rows), top_k) if isinstance(top_k, int) and top_k > 0 else len(rows)
    out: List[Dict[str, Any]] = []
    for r in rows[:n]:
        cos = r.get("cos_sim")
        out.append(
            {
                "id": r.get("id"),
                "context_id": r.get("context_id"),
                "chunk_index": r.get("chunk_index"),
                "score": float(cos) if cos is not None else None,
                "preview": (r.get("preview") or "").strip(),
                "document_id": r.get("document_id"),
                "section_key": r.get("section_key"),
                "section_title": r.get("section_title"),
            }
        )
    return out
