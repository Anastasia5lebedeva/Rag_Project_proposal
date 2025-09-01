BEGIN;

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX IF NOT EXISTS llm_contexts_doc_ord_idx
  ON rag.llm_contexts (document_id, order_idx);
CREATE INDEX IF NOT EXISTS llm_contexts_section_key_idx
  ON rag.llm_contexts (section_key);

CREATE INDEX IF NOT EXISTS retriever_segments_context_id_idx
  ON rag.retriever_segments (context_id);
CREATE INDEX IF NOT EXISTS retriever_segments_section_key_idx
  ON rag.retriever_segments (section_key);
CREATE INDEX IF NOT EXISTS retriever_segments_lang_idx
  ON rag.retriever_segments (lang);

CREATE INDEX IF NOT EXISTS embedding_cache_created_idx
  ON rag.embedding_cache (created_at DESC);

COMMIT;