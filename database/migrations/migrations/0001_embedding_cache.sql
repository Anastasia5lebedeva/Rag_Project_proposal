CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.embedding_cache (
  id BIGSERIAL PRIMARY KEY,
  model TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  content_norm TEXT NOT NULL DEFAULT '',
  embedding vector(1024) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT embedding_cache_model_hash_uidx UNIQUE (model, content_hash)
);

CREATE INDEX IF NOT EXISTS embedding_cache_model_idx
  ON rag.embedding_cache (model);