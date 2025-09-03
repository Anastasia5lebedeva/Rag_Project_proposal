CREATE SCHEMA IF NOT EXISTS rag;
CREATE EXTENSION IF NOT EXISTS vector;


DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'rag' AND t.typname = 'embedding_kind'
  ) THEN
    CREATE TYPE rag.embedding_kind AS ENUM ('doc','query','other');
  END IF;
END $$;


CREATE TABLE IF NOT EXISTS rag.llm_contexts (
  id            BIGSERIAL PRIMARY KEY,
  document_id   TEXT NOT NULL,
  section_title TEXT,
  content       TEXT NOT NULL,
  meta          JSONB,
  order_idx     INTEGER,
  text_md       TEXT,
  title_project TEXT,
  section_key   TEXT,
  lang          TEXT
);

CREATE TABLE IF NOT EXISTS rag.llm_contexts_for_model (
  id          BIGINT,
  document_id TEXT,
  order_idx   INTEGER,
  text_md     TEXT
);

CREATE TABLE IF NOT EXISTS rag._stg_sections (
  document_id   TEXT NOT NULL,
  order_idx     INTEGER NOT NULL,
  section_title TEXT NOT NULL,
  content       TEXT,
  meta          JSONB,
  text_md       TEXT,
  title_project TEXT,
  section_key   TEXT NOT NULL,
  lang          TEXT
);


CREATE TABLE IF NOT EXISTS rag.retriever_segments (
  id             BIGSERIAL PRIMARY KEY,
  context_id     BIGINT  NOT NULL,
  chunk_index    INTEGER NOT NULL,
  text_norm      TEXT    NOT NULL,
  embedding      vector(1024),
  meta           JSONB,
  section_key    TEXT,
  lang           TEXT
);



CREATE TABLE IF NOT EXISTS rag.embedding_cache (
  id           BIGSERIAL PRIMARY KEY,
  model        TEXT NOT NULL,
  kind         rag.embedding_kind NOT NULL DEFAULT 'doc',
  content_hash TEXT NOT NULL,
  content_norm TEXT NOT NULL DEFAULT '',
  embedding    vector(1024) NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT embedding_cache_model_kind_hash_uidx
    UNIQUE (model, kind, content_hash)
);


CREATE TABLE IF NOT EXISTS rag.section_title_map (
  pattern    TEXT NOT NULL,
  canon_type TEXT NOT NULL
);


CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_hash
  ON rag.embedding_cache (model, content_hash);


CREATE UNIQUE INDEX IF NOT EXISTS uidx_retriever_segments_ctx_chunk
  ON rag.retriever_segments (context_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_retriever_segments_section
  ON rag.retriever_segments (section_key);

CREATE INDEX IF NOT EXISTS idx_retriever_segments_lang
  ON rag.retriever_segments (lang);


CREATE INDEX IF NOT EXISTS idx_llm_contexts_meta_gin
  ON rag.llm_contexts USING GIN (meta);

CREATE INDEX IF NOT EXISTS idx_retriever_segments_meta_gin
  ON rag.retriever_segments USING GIN (meta);


DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname='rag' AND indexname='idx_retriever_segments_embedding'
  ) THEN
    EXECUTE 'CREATE INDEX idx_retriever_segments_embedding
             ON rag.retriever_segments
             USING ivfflat (embedding vector_l2_ops)
             WITH (lists = 100);';
  END IF;
END $$;

COMMIT;