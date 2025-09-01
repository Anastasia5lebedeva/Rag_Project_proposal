from __future__ import annotations
import logging
from typing import Any, Dict, Iterable, List, Tuple
import psycopg
from psycopg.rows import dict_row
from proposal_rag.config.settings import get_settings

log = logging.getLogger(__name__)
s = get_settings()

EMBED_DIM: int = int(getattr(s, "EMBED_DIM", 1024))

SEARCH_BOOST: int = int(getattr(s, "SEARCH_VECTOR_BOOST", 6))
SEARCH_MIN_CANDIDATES: int = int(getattr(s, "SEARCH_MIN_CANDIDATES", 200))

def _get_dsn() -> str:
    dsn = (
        getattr(s, "DATABASE_URL", "")
        or getattr(s, "DB_DSN", "")
        or ""
    )
    if not dsn:
        raise RuntimeError("Database DSN is empty (DATABASE_URL/DB_DSN). Configure .env")
    return dsn



def search_all(q_text_short: str, q_vec_lit: str, top_k: int) -> Tuple[List[Dict[str, Any]], str]:

    dsn = _get_dsn()
    cand = max(top_k * SEARCH_BOOST, SEARCH_MIN_CANDIDATES)

    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute("SET search_path TO rag;")

            try:
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
                    FROM rag.search_hybrid(%s, %s::vector({EMBED_DIM}), %s, NULL::text[]);
                    """,
                    (q_text_short, q_vec_lit, cand),
                )
                rows = cur.fetchall()
                if rows:
                    log.info("search_hybrid returned %d rows (cand=%d, top_k=%d)", len(rows), cand, top_k)
                    return _limit_and_format(rows, top_k), "HYBRID"
            except Exception as e:
                log.warning("search_hybrid unavailable/failed: %s. Fallback to ANN.", e)

            cur.execute(
                f"""
                WITH params AS (
                  SELECT %s::vector({EMBED_DIM}) AS qv
                ),
                base AS (
                  SELECT id, context_id, chunk_index, text_norm, embedding_1024, section_key
                  FROM rag.retriever_segments
                  WHERE embedding_1024 IS NOT NULL
                ),
                scored AS (
                  SELECT b.*,
                         1 - (b.embedding_1024 <=> p.qv) AS cos_sim
                  FROM base b, params p
                )
                SELECT s.id,
                       s.context_id,
                       s.chunk_index,
                       s.cos_sim,
                       left(s.text_norm, 300) AS preview,
                       lc.document_id,
                       COALESCE(s.section_key, lc.section_key) AS section_key,
                       lc.section_title
                FROM scored s
                LEFT JOIN rag.llm_contexts lc ON lc.id = s.context_id
                ORDER BY s.cos_sim DESC
                LIMIT %s;
                """,
                (q_vec_lit, cand),
            )
            rows = cur.fetchall()
            log.info("ANN fallback returned %d rows (cand=%d, top_k=%d)", len(rows), cand, top_k)
            return _limit_and_format(rows, top_k), "FALLBACK"

    except psycopg.Error as e:
        log.error("database search failed: %s", e)
        raise


def enrich_rows_with_doc_and_ord(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:

    rows_list = list(rows)
    if not rows_list:
        return rows_list

    ids = [r["id"] for r in rows_list if "id" in r]
    by_id: Dict[int, Dict[str, Any]] = {r["id"]: dict(r) for r in rows_list if "id" in r}

    try:
        dsn = _get_dsn()
        with psycopg.connect(dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute("SET search_path TO rag;")
            cur.execute(
                """
                SELECT rs.id,
                       lc.document_id,
                       lc.order_idx,
                       COALESCE(rs.section_key, lc.section_key) AS section_key,
                       lc.section_title
                FROM rag.retriever_segments rs
                LEFT JOIN rag.llm_contexts lc ON lc.id = rs.context_id
                WHERE rs.id = ANY(%s)
                """,
                (ids,),
            )
            for rec in cur.fetchall():
                rid = rec["id"]
                row = by_id.get(rid)
                if not row:
                    continue

                row.setdefault("document_id", rec["document_id"])
                row.setdefault("section_key", rec["section_key"])
                row.setdefault("section_title", rec["section_title"])
                row.setdefault("order_idx", rec["order_idx"])

    except psycopg.Error as e:
        log.error("failed to enrich rows: %s", e)
        raise

    return [by_id[i] for i in ids if i in by_id]


def _limit_and_format(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    for r in rows[:top_k]:
        out.append(
            {
                "id": r.get("id"),
                "context_id": r.get("context_id"),
                "chunk_index": r.get("chunk_index"),
                "score": float(r.get("cos_sim")) if r.get("cos_sim") is not None else None,
                "preview": r.get("preview") or "",
                "document_id": r.get("document_id"),
                "section_key": r.get("section_key"),
                "section_title": r.get("section_title"),
            }
        )
    return out
