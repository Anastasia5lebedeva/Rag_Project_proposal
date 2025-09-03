from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, conlist, ConfigDict, field_validator
from datetime import datetime
from typing import List


JsonDict = Dict[str, Any]
Vector1024 = conlist(float, min_length=1024, max_length=1024)


class DTO(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
    )


class ProposalRequestDTO(DTO):
    query: str
    context: List[str]
    temperature: Optional[float] = 0.1

    @field_validator("context", mode="before")
    @classmethod
    def _clean_context(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            raw = v.replace("---", "\n").replace(",", "\n")
            parts = [p.strip() for p in raw.splitlines()]
            return [p for p in parts if p]
        if isinstance(v, (list, tuple)):
            cleaned = [str(x).strip() for x in v if str(x).strip()]
            return cleaned
        return v



class ProposalResponseDTO(DTO):
    markdown: str
    sections: List[str] = Field(default_factory=list)





class DBModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        from_attributes=True,
        validate_assignment=True,
        frozen=False,
    )





class LlmContext(DBModel):
    id: int
    document_id: str
    section_title: Optional[str] = None
    content: str
    meta: Optional[JsonDict] = None
    order_idx: Optional[int] = None
    text_md: Optional[str] = None
    title_project: Optional[str] = None
    section_key: Optional[str] = None
    lang: Optional[str] = None



class LlmContextForModel(DBModel):
    id: Optional[int] = None
    document_id: Optional[str] = None
    order_idx: Optional[int] = None
    text_md: Optional[str] = None


class RetrieverSegment(DBModel):
    id: int
    context_id: int
    chunk_index: int
    text_norm: str
    embedding: Optional[Vector1024] = Field(default=None, description="pgvector dim=1024")
    meta: Optional[JsonDict] = None
    section_key: Optional[str] = None
    lang: Optional[str] = None



class SectionTitleMap(DBModel):
    pattern: str
    canon_type: str



class EmbeddingKind(str, Enum):
    doc = "doc"
    query = "query"
    other = "other"



class EmbeddingCache(DBModel):
    id: int
    model: str
    kind: EmbeddingKind
    content_hash: str
    content_norm: str
    embedding: Vector1024
    created_at: datetime


__all__ = [
    "JsonDict",
    "Vector1024",
    "DBModel",
    "LlmContext",
    "LlmContextForModel",
    "RetrieverSegment",
    "SectionTitleMap",
    "EmbeddingCache",
]