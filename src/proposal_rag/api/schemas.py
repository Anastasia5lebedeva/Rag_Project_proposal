from __future__ import annotations

from typing import Any, Dict, Optional, Literal, Annotated, List

from pydantic import BaseModel, Field


StrNonEmpty = Annotated[str, Field(min_length=1)]
ListNonEmptyStr = Annotated[list[StrNonEmpty], Field(min_length=1)]
PosInt_1_50 = Annotated[int, Field(ge=1, le=50)]
PosInt_1 = Annotated[int, Field(ge=1)]


class HealthResponse(BaseModel):
    status: Literal["ok", "fail"] = Field(..., description="Overall service status: ok | fail")
    db: Literal["ok", "fail"] = Field(..., description="PostgreSQL status")
    vectordb: Literal["ok", "fail"] = Field(..., description="Retriever segments / vector index status")
    llm: Literal["ok", "fail"] = Field(..., description="LLM service status (GPT-4.1)")


class AnalyzeResponse(BaseModel):
    problem: StrNonEmpty = Field(..., description="Client's problem description (normalized)")
    goals: list[str] = Field(default_factory=list, description="Client goals")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords for search")
    chunks: Optional[list[str]] = Field(default=None, description="Optional smart-chunks of the problem")


class SearchHit(BaseModel):
    chunk_index: Annotated[int, Field(ge=0)] = Field(..., description="Index of chunk in source")
    score: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(..., description="Similarity score")
    preview: StrNonEmpty = Field(..., description="Short text preview")
    source_meta: Optional[Dict[str, Any]] = Field(default=None, description="Original source metadata")


class SearchRequest(BaseModel):
    query: StrNonEmpty = Field(..., description="Search query")
    top_k: PosInt_1_50 = Field(10, description="Number of results to return")


class SearchResponse(BaseModel):
    total_chunks: Annotated[int, Field(ge=0)] = Field(..., description="Total available chunks")
    top_k: PosInt_1 = Field(..., description="Requested top_k")
    query: StrNonEmpty = Field(..., description="Echo of the query")
    results: List[SearchHit]


class GenerateRequest(BaseModel):
    queries: ListNonEmptyStr
    top_k: PosInt_1_50 = Field(10)


class GenerateResponse(BaseModel):
    text: StrNonEmpty
    used_chunks: Annotated[int, Field(ge=0)] = 0
    sources: list[Dict[str, Any]] = Field(default_factory=list)

