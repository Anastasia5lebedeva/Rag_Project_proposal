from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    chunk_index: int = Field(..., ge=0)
    score: float = Field(..., ge=-1.0, le=1.0)
    preview: str = Field(..., min_length=1)
    source_meta: Optional[dict] = None


class SearchResponse(BaseModel):
    total_chunks: int = Field(..., ge=0)
    top_k: int = Field(..., ge=1)
    query: str = Field(..., min_length=1)
    results: List[SearchHit]