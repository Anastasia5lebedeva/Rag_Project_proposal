BEGIN;

CREATE INDEX IF NOT EXISTS idx_retriever_segments_section
  ON rag.retriever_segments (section_key);

CREATE INDEX IF NOT EXISTS idx_retriever_segments_lang
  ON rag.retriever_segments (lang);

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