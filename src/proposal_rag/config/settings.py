from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, AnyHttpUrl, SecretStr, NonNegativeInt, PositiveInt, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # App
    APP_ENV: str = Field("dev", description="Environment name")
    LOG_LEVEL: str = Field("INFO", description="Python logging level (DEBUG/INFO/WARNING/ERROR)")

    # Limits (no magic numbers in code)
    MAX_TOP_K: PositiveInt = Field(20, description="Hard upper bound for top_k")
    DEFAULT_TOP_K: PositiveInt = Field(10, description="Default top_k")
    MIN_QUERY_LEN: PositiveInt = Field(2, description="Minimal query length")
    TOKEN_LIMIT_PER_CHUNK: PositiveInt = Field(3000, description="Tokenizer-level chunk limit")

    # LLM
    LLM_API_URL: AnyHttpUrl = Field("http://localhost:11434/v1", description="Base URL for LLM API")
    LLM_API_KEY: SecretStr = Field("", description="API key (can be empty for local)")
    LLM_MODEL: str = Field("gpt-4.1", description="Default model id")
    LLM_TIMEOUT_S: PositiveInt = Field(30, description="HTTP timeout seconds")

    # Embeddings
    # Support both EMBED_MODEL and legacy EMB_MODEL
    EMBED_MODEL: str = Field(
        "intfloat/multilingual-e5-large",
        description="Embedding model id",
        validation_alias=AliasChoices("EMBED_MODEL", "EMB_MODEL"),
    )
    EMBED_DIM: PositiveInt = Field(1024, description="Embedding dimensionality")
    EMBED_CACHE_ENABLED: bool = Field(True, description="Enable embedding cache")

    # DB (prefer DATABASE_URL; DB_DSN kept for backward compatibility)
    DATABASE_URL: Optional[str] = Field(None, description="Primary Postgres DSN")
    DB_DSN: Optional[str] = Field(None, description="Legacy Postgres DSN (fallback)")
    DB_POOL_MIN: NonNegativeInt = Field(1, description="Min pool size")
    DB_POOL_MAX: PositiveInt = Field(10, description="Max pool size")

    # Prompts directory (no prompt hardcode)
    PROMPTS_DIR: Path = Field(default=ROOT_DIR / "resources" / "prompts", description="Directory with prompt files")

    # Feature flags
    FEATURE_PDF: bool = Field(True, description="Enable PDF parsing pipeline")
    FEATURE_DOCX: bool = Field(True, description="Enable DOCX parsing pipeline")

    # ---- Convenience properties (no secrets logged) -------------------------
    @property
    def DSN(self) -> str:
        """Effective DSN with DATABASE_URL preferred over DB_DSN."""
        return (self.DATABASE_URL or self.DB_DSN or "").strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()