CREATE SCHEMA IF NOT EXISTS rag;


CREATE TABLE IF NOT EXISTS rag.embedding_cache (
  id            BIGSERIAL PRIMARY KEY,
  model         TEXT NOT NULL,
  kind         rag.embedding_kind NOT NULL DEFAULT 'doc',
  content_hash  TEXT NOT NULL,
  content_norm  TEXT NOT NULL DEFAULT '',
  embedding     vector(1024) NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE UNIQUE INDEX IF NOT EXISTS uidx_embedding_cache_model_kind_hash
  ON rag.embedding_cache (model, kind, content_hash);


CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_kind_hash
  ON rag.embedding_cache (model, kind, content_hash);