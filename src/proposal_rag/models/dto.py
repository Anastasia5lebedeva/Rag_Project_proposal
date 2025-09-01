from __future__ import annotations
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field, conlist



JsonDict = Dict[str, Any]
Vector1024 = conlist(float, min_length=1)


class DBModel(BaseModel):
    class Config:
        populate_by_name = True
        str_strip_whitespace = True
        frozen = False


class StgSection(DBModel):
    document_id: str
    order_idx: int
    section_title: str
    content: Optional[str] = None
    meta: Optional[JsonDict] = None
    text_md: Optional[str] = None
    title_project: Optional[str] = None
    section_key: str
    lang: Optional[str] = None


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
    embedding_1024: Optional[Vector1024] = Field(
        default=None, description="pgvector embedding (dim=1024)"
    )
    meta: Optional[JsonDict] = None
    section_key: Optional[str] = None
    lang: Optional[str] = None


class SectionTitleMap(DBModel):
    pattern: str
    canon_type: str


__all__ = [
    "JsonDict",
    "Vector1024",
    "DBModel",
    "StgSection",
    "LlmContext",
    "LlmContextForModel",
    "RetrieverSegment",
    "SectionTitleMap",
]