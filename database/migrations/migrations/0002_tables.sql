BEGIN;

CREATE SCHEMA IF NOT EXISTS rag;

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


CCREATE TABLE IF NOT EXISTS rag.llm_contexts (
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
  id          BIGSERIAL PRIMARY KEY,
  context_id  BIGINT  NOT NULL,
  chunk_index INTEGER NOT NULL,
  text_norm   TEXT    NOT NULL,
  embedding   vector(1024),
  meta        JSONB,
  section_key TEXT,
  lang        TEXT
);

CREATE TABLE IF NOT EXISTS rag.retriever_segments (
  id             BIGSERIAL PRIMARY KEY,
  context_id     BIGINT  NOT NULL,
  chunk_index    INTEGER NOT NULL,
  text_norm      TEXT    NOT NULL,
  embedding   vector(1024),
  meta           JSONB,
  section_key    TEXT,
  lang           TEXT
);

CREATE TABLE IF NOT EXISTS rag.section_title_map (
  pattern    TEXT NOT NULL,
  canon_type TEXT NOT NULL
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

CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_hash
  ON rag.embedding_cache (model, content_hash);

COMMIT;
