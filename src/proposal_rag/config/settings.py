from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, NonNegativeInt, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = ROOT_DIR / ".env.dev"



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    ENV: str = Field("dev", description="Environment name")
    LOG_LEVEL: str = Field("INFO", description="Python logging level (DEBUG/INFO/WARNING/ERROR)")
    ALLOW_RUNTIME_DDL: bool = Field(False, description="Allow runtime DDL for dev/test")

    MAX_TOP_K: PositiveInt = Field(20, description="Hard upper bound for top_k")
    DEFAULT_TOP_K: PositiveInt = Field(10, description="Default top_k")
    MIN_QUERY_LEN: PositiveInt = Field(2, description="Minimal query length")
    TOKEN_LIMIT_PER_CHUNK: PositiveInt = Field(3000, description="Tokenizer-level chunk limit")
    SEARCH_VECTOR_BOOST: PositiveInt = Field(6, description="Boost factor for vector score in hybrid search")
    SEARCH_MIN_CANDIDATES: PositiveInt = Field(200, description="Minimal candidate set size before rerank")


    LLM_API_URL: str = Field("http://api.comrade.168.119.237.99.sslip.io/v1", description="Base URL for LLM API")
    LLM_API_KEY: SecretStr = Field("", description="API key (can be empty for local)")
    LLM_MODEL: str = Field("gpt-4.1", description="Default model id")
    LLM_TIMEOUT_S: PositiveInt = Field(120, description="HTTP timeout seconds")
    LLM_VERIFY_SSL: bool = Field(False, description="Verify TLS certificates for LLM client")
    LLM_CA_FILE: Optional[str] = Field(None, description="Custom CA bundle path for LLM client")

    EMBED_MODEL: str = Field("intfloat/multilingual-e5-large", description="Embedding model id")
    EMBED_DIM: PositiveInt = Field(1024, description="Embedding dimensionality")
    EMBED_DEVICE: str = Field("cpu", description="Device for embedding model (cpu/cuda)")
    EMBED_CACHE_ENABLED: bool = Field(True, description="Enable embedding cache")

    DATABASE_URL: Optional[str] = Field(None, description="Primary Postgres DSN")
    DB_DSN: Optional[str] = Field(None, description="Legacy Postgres DSN (fallback)")
    DB_POOL_MIN: NonNegativeInt = Field(1, description="Min pool size")
    DB_POOL_MAX: PositiveInt = Field(10, description="Max pool size")
    DB_SCHEMA: str = Field("rag", description="Postgres schema for RAG")
    EMBED_COLUMN: str = Field("embedding", description="Embedding column name")
    DB_CONNECT_TIMEOUT: PositiveInt = Field(5, description="DB connect timeout (seconds)")

    PROMPTS_DIR: Path = Field(default=ROOT_DIR / "resources" / "prompts", description="Directory with prompt files")
    PATTERNS_FILE: Path = Field(default=ROOT_DIR / "resources" / "prompts" / "hdr_pattern.txt",
                                description="Path to patterns file")
    PATTERN_HDR: Optional[str] = Field(None, description="Override header pattern")
    PATTERN_BULLET: Optional[str] = Field(None, description="Override bullet pattern")

    CHUNK_PROFILE: str = Field("strict", description="Chunking profile")
    CHUNK_MAX_TOKENS: PositiveInt = Field(2600, description="Max tokens per chunk")
    CHUNK_OVERLAP_TOKENS: PositiveInt = Field(250, description="Overlap tokens between chunks")
    CHUNK_MIN_TOKENS: PositiveInt = Field(150, description="Min tokens per chunk")
    CHUNK_MAX_CHARS: int = Field(0, ge=0, description="Max chars per chunk (0=disabled)")

    QUERY_PREFIX: str = Field("query:", description="Prefix for query embedding text")
    PASSAGE_PREFIX: str = Field("passage:", description="Prefix for passage embedding")

    SEGMENTS_BATCH_SIZE: PositiveInt = Field(200, description="Batch size for retriever_segments sync")

    FEATURE_PDF: bool = Field(True, description="Enable PDF parsing pipeline")
    FEATURE_DOCX: bool = Field(True, description="Enable DOCX parsing pipeline")


    DEFAULT_QUERY_PATH: Path = Field(Path("/app/data/query.docx"), description="Default path to a query docx")

    @property
    def DSN(self) -> str:
        return (self.DATABASE_URL or self.DB_DSN or "").strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()