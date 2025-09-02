from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, conlist, ConfigDict
from datetime import datetime
from typing import List


JsonDict = Dict[str, Any]
Vector1024 = conlist(float, min_length=1024, max_length=1024)


class DTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

class ProposalRequestDTO(DTO):
    query: str
    context: List[str]
    temperature: float = 0.1

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
    embedding: Optional[Vector1024] = Field(default=None, alias="embedding_1024", description="pgvector dim=1024")
    meta: Optional[JsonDict] = None
    section_key: Optional[str] = None
    lang: Optional[str] = None


class SectionTitleMap(DBModel):
    pattern: str
    canon_type: str


class EmbeddingCache(DBModel):
    id: int
    model: str
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