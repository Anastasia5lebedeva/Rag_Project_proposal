from __future__ import annotations

import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from proposal_rag.api.errors import (
    app_error_handler,
    unhandled_error_handler,
    BadRequest,
    UnsupportedMediaType,
    DatabaseError,
    VectorDBError,
    LLMServiceError,
)
from proposal_rag.api.schemas import (
    SearchRequest,
    SearchResponse,
    SearchHit,
    GenerateRequest,
    GenerateResponse,
    AnalyzeResponse,
    HealthResponse,
)


from proposal_rag.services import vector_search, llm_client


log = logging.getLogger(__name__)
app = FastAPI(title="Proposal RAG API")


# --------- Error Handlers ---------
app.add_exception_handler(Exception, unhandled_error_handler)
app.add_exception_handler(BadRequest, app_error_handler)
app.add_exception_handler(UnsupportedMediaType, app_error_handler)
app.add_exception_handler(DatabaseError, app_error_handler)
app.add_exception_handler(VectorDBError, app_error_handler)
app.add_exception_handler(LLMServiceError, app_error_handler)


# --------- Endpoints ---------

@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health() -> HealthResponse:
    """
    Проверка статуса: сервис + БД + векторка + LLM.
    """
    try:
        db_ok = db_utils.check_connection()
        vec_ok = vector_search.check_vector_db()
        llm_ok = await llm_client.check_health()
    except Exception as e:
        log.exception("Health check failed")
        raise DatabaseError("health check failed", extra={"reason": str(e)})

    return HealthResponse(
        service="ok",
        database="ok" if db_ok else "fail",
        vector_db="ok" if vec_ok else "fail",
        llm="ok" if llm_ok else "fail",
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze(
    text: str = Form(None),
    file: UploadFile = File(None)
) -> AnalyzeResponse:
    """
    Анализ входного запроса:
    - либо text (JSON/form-data),
    - либо файл (PDF/DOCX).
    """
    if not text and not file:
        raise BadRequest("Need either 'text' or 'file'")

    if file:
        if file.content_type not in [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ]:
            raise UnsupportedMediaType(f"Unsupported file type: {file.content_type}")
        content = await file.read()
        # TODO: реальный парсинг PDF/DOCX
        parsed_text = f"parsed content from {file.filename} ({len(content)} bytes)"
    else:
        parsed_text = text.strip()

    # TODO: передать в LLM для классификации / анализа
    return AnalyzeResponse(query=parsed_text, detected_needs=["draft_need"])


@app.post("/search", response_model=SearchResponse, tags=["search"])
async def search(req: SearchRequest) -> SearchResponse:
    """
    Поиск релевантных сегментов (BM25 + vector).
    """
    try:
        rows = vector_search.search_hybrid(req.query, req.top_k)
    except Exception as e:
        raise VectorDBError("vector search failed", extra={"reason": str(e)})

    hits = [
        SearchHit(
            chunk_index=row["chunk_index"],
            score=row["score"],
            preview=row["preview"],
            source_meta=row["source_meta"],
        )
        for row in rows
    ]

    return SearchResponse(
        total_chunks=len(hits),
        top_k=req.top_k,
        query=req.query,
        results=hits,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["generate"])
async def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Сгенерировать КП на основе найденных сегментов.
    """
    try:
        draft, sections = await llm_client.generate_proposal(
            query=req.query,
            context=req.context_chunks,
        )
    except Exception as e:
        raise LLMServiceError("LLM generation failed", extra={"reason": str(e)})

    return GenerateResponse(proposal_text=draft, sections=sections)

