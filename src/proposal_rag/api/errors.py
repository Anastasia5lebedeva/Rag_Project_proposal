from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class AppError(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "app_error"

    def __init__(self, message: str = "internal error", *, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.extra = extra or {}


class BadRequest(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"


class UnsupportedMediaType(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"


class ValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "validation_error"


class DocumentParseError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "document_parse_error"


class DocumentNotFound(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "document_not_found"


class DatabaseError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "database_error"


class DatabaseUnavailable(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "database_unavailable"


class VectorDBError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "vector_db_error"


class EmbeddingError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "embedding_error"


class LLMServiceError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "llm_service_error"


class LLMResponseError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "llm_response_error"


class ServiceError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "service_error"


def _get_request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    return rid or str(uuid.uuid4())


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = _get_request_id(request)
    log.error(
        "AppError: code=%s msg=%s",
        exc.code,
        exc.message,
        extra={"request_id": request_id, **(exc.extra or {})},
        exc_info=False,
    )
    payload = ErrorResponse(
        error=exc.code,
        message=exc.message,
        request_id=request_id,
        extra=exc.extra or None,
    ).model_dump()
    return JSONResponse(status_code=exc.status_code, content=payload)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = _get_request_id(request)
    log.exception("UnhandledError", extra={"request_id": request_id})
    payload = ErrorResponse(
        error="unhandled_error",
        message="internal server error",
        request_id=request_id,
    ).model_dump()
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

