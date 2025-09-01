BEGIN;


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
  embedding_1024 vector(1024),
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
  content_hash TEXT NOT NULL,
  content_norm TEXT NOT NULL DEFAULT '',
  embedding    vector(1024) NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT embedding_cache_model_hash_uidx UNIQUE (model, content_hash)
);

COMMIT;
