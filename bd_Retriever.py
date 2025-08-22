import os, re, json, argparse, hashlib
from typing import List, Dict, Any
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json


#синхронизации чтобы rag.retriever_segments всегда содержала актуальные эмбеддинги для всех текстов/

DEFAULT_DSN = "postgresql://cortex:cortex123@127.0.0.1:5440/cortex_rag"
DEFAULT_MODEL = "intfloat/multilingual-e5-large"
EMB_DIM = 1024


WS_RE = re.compile(r"[ \t\u00A0]+")
MULTINL_RE = re.compile(r"\n{3,}")
HDR_RE = re.compile(r"(?m)^(#{1,6}\s+.+|[А-ЯA-Z0-9][^\n]{0,80}:$|(?:Раздел|Глава|Section)\s+\d+[^\n]*$)")
BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•]|[0-9]+\.)\s+")


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = WS_RE.sub(" ", s)
    s = MULTINL_RE.sub("\n\n", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    return s.strip()


def split_blocks(text: str) -> list[str]:
    lines = text.split("\n")
    blocks, buf = [], []
    for ln in lines:
        if HDR_RE.match(ln) and buf:
            blocks.append("\n".join(buf).strip()); buf = [ln]
        elif ln.strip() == "" and buf:
            blocks.append("\n".join(buf).strip()); buf = []
        else:
            buf.append(ln)
    if buf: blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b and len(b) >= 3]

def split_paragraphs(block: str) -> list[str]:
    paras, buf = [], []
    for ln in block.split("\n"):
        if BULLET_RE.match(ln):
            if buf: paras.append("\n".join(buf).strip()); buf = []
            buf.append(ln)
        else:
            buf.append(ln)
    if buf: paras.append("\n".join(buf).strip())
    return [p for p in paras if p]

def smart_chunk(text: str, max_tokens: int = 1600, overlap_tokens: int = 250, min_tokens: int = 150) -> list[str]:
    text = normalize_text(text)
    if not text: return []
    blocks = split_blocks(text) or [text]
    chunks, carry = [], ""
    buf, buf_toks = [], 0

    for blk in blocks:
        for p in split_paragraphs(blk):
            t = approx_tokens(p)
            if t >= max_tokens:
                if buf:
                    chunk = ((carry + "\n") if carry else "") + "\n\n".join(buf).strip()
                    if approx_tokens(chunk) >= min_tokens: chunks.append(chunk.strip())
                    carry = chunk[-int(overlap_tokens*4):] if chunk else ""
                    buf, buf_toks = [], 0
                chunks.append(p.strip())
                carry = p[-int(overlap_tokens*4):]
                continue
            if buf_toks + t > max_tokens:
                chunk = ((carry + "\n") if carry else "") + "\n\n".join(buf).strip()
                if approx_tokens(chunk) >= min_tokens: chunks.append(chunk.strip())
                carry = chunk[-int(overlap_tokens*4):] if chunk else ""
                buf, buf_toks = [p], t
            else:
                buf.append(p); buf_toks += t

    if buf:
        chunk = ((carry + "\n") if carry else "") + "\n\n".join(buf).strip()
        if approx_tokens(chunk) >= min_tokens or not chunks:
            chunks.append(chunk.strip())
    dedup, seen = [], set()
    for c in chunks:
        k = c[:200]
        if k in seen: continue
        seen.add(k); dedup.append(c)
    return dedup


class Encoder:
    def __init__(self, provider: str, model_name: str, dim: int):
        self.provider = provider
        self.dim = dim
        self.model = None
        if provider == "stub":
            return
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        test = self.model.encode(["probe"], normalize_embeddings=True)
        got = len(test[0])
        if got != dim:
            print(f"Модель даёт {got} dim, в БД vector({dim}).Отредактируй!")


    def encode(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "stub":
            return [[0.0]*self.dim for _ in texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

DDL = """
CREATE SCHEMA IF NOT EXISTS rag;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag.retriever_segments (
  d              BIGSERIAL PRIMARY KEY,
  context_id     BIGINT NOT NULL REFERENCES rag.llm_contexts(id) ON DELETE CASCADE,
  chunk_index    INT    NOT NULL,
  text_norm      TEXT   NOT NULL,
  embedding_1024 VECTOR(1024) NOT NULL,
  meta           JSONB  DEFAULT '{}'::jsonb,
  section_key    TEXT,
  lang           TEXT,
  UNIQUE (context_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS retr_segments_hnsw_cos
  ON rag.retriever_segments USING hnsw (embedding_1024 vector_cosine_ops);

CREATE INDEX IF NOT EXISTS retr_segments_ctx_chunk_uniq
  ON rag.retriever_segments (context_id, chunk_index);
"""


SELECT_CONTEXTS = """
WITH latest_hash AS (
  SELECT
    l.id AS context_id,
    md5(COALESCE(l.content,'')) AS src_hash
  FROM rag.llm_contexts l
  WHERE l.content IS NOT NULL AND btrim(l.content) <> ''
)
SELECT
  l.id           AS context_id,
  l.document_id  AS document_id,
  l.order_idx    AS order_idx,
  l.section_title,
  l.section_key,
  COALESCE(l.lang,'ru') AS lang,
  l.content      AS content,
  l.meta         AS meta,
  h.src_hash     AS source_hash
FROM rag.llm_contexts l
JOIN latest_hash h ON h.context_id = l.id
LEFT JOIN LATERAL (
  SELECT rs.meta->>'source_hash' AS stored_hash
  FROM rag.retriever_segments rs
  WHERE rs.context_id = l.id
  ORDER BY rs.chunk_index DESC
  LIMIT 1
) rs ON true
WHERE l.content IS NOT NULL
  AND btrim(l.content) <> ''
  AND COALESCE(rs.stored_hash, '') <> h.src_hash
ORDER BY l.id;
"""
UPSERT_SEGMENT = """
INSERT INTO rag.retriever_segments
  (context_id, chunk_index, text_norm, embedding_1024, meta, section_key, lang)
VALUES (%(context_id)s, %(chunk_index)s, %(text_norm)s, %(embedding)s, %(meta)s, %(section_key)s, %(lang)s)
ON CONFLICT (context_id, chunk_index) DO UPDATE
SET text_norm      = EXCLUDED.text_norm,
    embedding_1024 = EXCLUDED.embedding_1024,
    meta           = rag.retriever_segments.meta || EXCLUDED.meta,
    section_key    = EXCLUDED.section_key,
    lang           = EXCLUDED.lang;
"""

def parse_args():
    ap = argparse.ArgumentParser("Sync retriever_segments from llm_contexts (incremental)")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--provider", choices=["st","stub"], default="st")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chunk-chars", type=int, default=1800)
    ap.add_argument("--overlap", type=int, default=300)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--filter-doc", default=None, help="Обработать document_id")
    return ap.parse_args()

def main():
    args = parse_args()
    enc = Encoder(args.provider, args.model, EMB_DIM)

    with psycopg.connect(args.dsn, row_factory=dict_row, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)

            cur.execute(SELECT_CONTEXTS)
            rows = cur.fetchall()
            if args.filter_doc:
                rows = [r for r in rows if r["document_id"] == args.filter_doc]
            print(f"Контекстов к обновлению: {len(rows)}")

            total_chunks = 0

            for r in rows:
                ctx_id = r["context_id"]
                content = r["content"] or ""
                source_hash = r["source_hash"]
                chunks = smart_chunk(
                    content,
                    max_tokens=args.chunk_chars//4,
                    overlap_tokens=args.overlap//4,
                    min_tokens=args.min_chars//4
                )
                if not chunks:
                    continue

                vecs = enc.encode(chunks)
                for idx, (txt, vec) in enumerate(zip(chunks, vecs)):
                    meta = {
                        "document_id": r["document_id"],
                        "order_idx": r["order_idx"],
                        "section_title": r["section_title"],
                        "source_hash": source_hash
                    }
                    cur.execute(UPSERT_SEGMENT, {
                        "context_id": ctx_id,
                        "chunk_index": idx,
                        "text_norm": txt,
                        "embedding": vec,
                        "meta": Json(meta),
                        "section_key": r["section_key"],
                        "lang": r["lang"]
                    })
                total_chunks += len(chunks)

                cur.execute("DELETE FROM rag.retriever_segments WHERE context_id=%s AND chunk_index >= %s",
                            (ctx_id, len(chunks)))
            conn.commit()
            print(f"Синхронизировано. Всего: {total_chunks}")

if __name__ == "__main__":
    main()