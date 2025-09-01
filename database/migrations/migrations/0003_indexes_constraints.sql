BEGIN;

CREATE UNIQUE INDEX IF NOT EXISTS retriever_segments_uq
  ON rag.retriever_segments (context_id, chunk_index);

CREATE INDEX IF NOT EXISTS retriever_segments_meta_gin
  ON rag.retriever_segments USING GIN (meta);


CREATE INDEX IF NOT EXISTS embedding_cache_model_idx
  ON rag.embedding_cache (model);

COMMIT;
