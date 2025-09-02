from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, conlist, constr




class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status: ok | error")
    db: str = Field(..., description="PostgreSQL status")
    vectordb: str = Field(..., description="Retriever segments status")
    llm: str = Field(..., description="LLM service status (GPT/Claude/DeepSeek)")




class AnalyzeRequest(BaseModel):
    text: Optional[str] = Field(None, description="Plain text query from client")


class AnalyzeResponse(BaseModel):
    problem: str = Field(..., description="Client's problem description")
    goals: List[str] = Field(..., description="Client goals")
    keywords: List[str] = Field(..., description="Extracted keywords for search")




class SearchHit(BaseModel):
    chunk_index: int = Field(..., ge=0)
    score: float = Field(..., ge=-1.0, le=1.0)
    preview: str = Field(..., min_length=1)
    source_meta: Optional[Dict[str, Any]] = None



class GenerateRequest(BaseModel):
    query: constr(min_length=1) = Field(..., description="Client query text")
    context_chunks: conlist(constr(min_length=1), min_items=1) = Field(
        ..., description="Ready-to-use text chunks for generation"
    )


class GenerateResponse(BaseModel):
    proposal_text: str = Field(..., description="Generated commercial proposal text")
    sections: Dict[str, str] = Field(
        ..., description="Structured sections of proposal: e.g. { 'Problem': '...', 'Solution': '...' }"
    )



class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")


class SearchResponse(BaseModel):
    total_chunks: int = Field(..., ge=0)
    top_k: int = Field(..., ge=1)
    query: str = Field(..., min_length=1)
    results: List[SearchHit]


