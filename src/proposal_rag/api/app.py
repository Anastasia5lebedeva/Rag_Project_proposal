from __future__ import annotations
import logging
import uuid
import contextvars
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
from fastapi import FastAPI, Request, Form, UploadFile, File
from proposal_rag.models.dto import ProposalRequestDTO
from proposal_rag.services import vector_search
from proposal_rag.services.generate_service import generate_proposal_svc
from proposal_rag.services.health_service import check_health_svc
from proposal_rag.services.logger import setup_logging
from proposal_rag.services.document_processor import normalize_text, smart_chunk


log = logging.getLogger(__name__)
app = FastAPI(title="Proposal RAG API")
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


@app.on_event("startup")
async def _startup() -> None:
    setup_logging()


def get_request_id() -> str:
    return request_id_var.get()


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    try:
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        request_id_var.reset(token)


app.add_exception_handler(Exception, unhandled_error_handler)
app.add_exception_handler(BadRequest, app_error_handler)
app.add_exception_handler(UnsupportedMediaType, app_error_handler)
app.add_exception_handler(DatabaseError, app_error_handler)
app.add_exception_handler(VectorDBError, app_error_handler)
app.add_exception_handler(LLMServiceError, app_error_handler)



@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health() -> HealthResponse:
    try:
        status = await check_health_svc()
        db_ok, vec_ok, llm_ok = status["db_ok"], status["vec_ok"], status["llm_ok"]
    except Exception as e:
        log.exception("Health check failed")
        raise DatabaseError("health check failed", extra={"reason": str(e)})
    overall_ok = bool(db_ok and vec_ok and llm_ok)
    return HealthResponse(
        status="ok" if overall_ok else "fail",
        db="ok" if db_ok else "fail",
        vectordb="ok" if vec_ok else "fail",
        llm="ok" if llm_ok else "fail",
    )





@app.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze(
    text: str = Form(None),
    file: UploadFile = File(None)
) -> AnalyzeResponse:
    global chunks
    if not text and not file:
        raise BadRequest("Need either 'text' or 'file'")

    if file:
        ct = (file.content_type or "").lower()
        if ct not in {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }:
            raise UnsupportedMediaType(f"Unsupported file type: {file.content_type}")
        content = await file.read()
        parsed_text = f"parsed content from {file.filename} ({len(content)} bytes)"
    else:

        parsed_text = normalize_text(text.strip() or "")
        chunks = smart_chunk(parsed_text)

    return AnalyzeResponse(
        problem=parsed_text,
        goals=["draft_goal"],
        keywords=["draft_keyword"],
        chunks=chunks,
    )


@app.post("/search", response_model=SearchResponse, tags=["search"])
async def search(req: SearchRequest) -> SearchResponse:
    try:
        rows = await vector_search.search_hybrid(req.query, req.top_k)
    except ValueError as e:
        raise BadRequest(str(e))
    except Exception as e:
        # всё остальное → 5xx
        raise VectorDBError("vector search failed", extra={"reason": str(e)})

    hits = [
        SearchHit(
            chunk_index=int(row["chunk_index"]),
            score=float(row["score"]),
            preview=(row.get("preview") or "").strip(),
            source_meta=row.get("source_meta") or None,
        )
        for row in (rows or [])
    ]

    return SearchResponse(
        total_chunks=len(hits),
        top_k=req.top_k,
        query=req.query,
        results=hits,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["generate"])
async def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        svc_req = ProposalRequestDTO(
            query=req.query,
            context=req.context_chunks,
            temperature=0.1,
        )
        svc_resp = await generate_proposal_svc(svc_req)
    except Exception as e:
        raise LLMServiceError("LLM generation failed", extra={"reason": str(e)})

    return GenerateResponse(
        proposal_text=svc_resp.markdown,
        sections=svc_resp.sections,
    )




