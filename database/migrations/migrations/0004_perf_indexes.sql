BEGIN;

CREATE EXTENSION IF NOT EXISTS pg_trgm;


CREATE INDEX IF NOT EXISTS idx_llm_contexts_doc_ord
  ON rag.llm_contexts (document_id, order_idx);

CREATE INDEX IF NOT EXISTS idx_llm_contexts_section
  ON rag.llm_contexts (section_key);

DO $$
DECLARE
  has_section_idx boolean;
  has_lang_idx    boolean;
  has_ctx_chunk   boolean;
BEGIN
  SELECT EXISTS (
    SELECT 1
    FROM pg_index i
    JOIN pg_class c ON c.oid = i.indrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
    WHERE n.nspname = 'rag' AND c.relname = 'retriever_segments'
      AND i.indisvalid
      AND array_position(i.indkey, a.attnum) = 1
      AND a.attname = 'section_key'
      AND i.indnatts = 1
  ) INTO has_section_idx;

  IF NOT has_section_idx THEN
    EXECUTE 'CREATE INDEX idx_retriever_segments_section ON rag.retriever_segments (section_key)';
  END IF;

  SELECT EXISTS (
    SELECT 1
    FROM pg_index i
    JOIN pg_class c ON c.oid = i.indrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
    WHERE n.nspname = 'rag' AND c.relname = 'retriever_segments'
      AND i.indisvalid
      AND array_position(i.indkey, a.attnum) = 1
      AND a.attname = 'lang'
      AND i.indnatts = 1
  ) INTO has_lang_idx;

  IF NOT has_lang_idx THEN
    EXECUTE 'CREATE INDEX idx_retriever_segments_lang ON rag.retriever_segments (lang)';
  END IF;


  SELECT EXISTS (
    SELECT 1
    FROM pg_indexes
    WHERE schemaname='rag' AND tablename='retriever_segments'
      AND indexdef ILIKE '%(context_id, chunk_index)%'
  ) INTO has_ctx_chunk;

  IF NOT has_ctx_chunk THEN

    PERFORM 1;
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname='rag' AND tablename='retriever_segments'
      AND indexname='idx_retriever_segments_text_norm_trgm'
  ) THEN
    EXECUTE 'CREATE INDEX idx_retriever_segments_text_norm_trgm
             ON rag.retriever_segments
             USING GIN (text_norm gin_trgm_ops);';
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_embedding_cache_created_desc
  ON rag.embedding_cache (created_at DESC);

COMMIT;
