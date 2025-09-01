from __future__ import annotations
import logging
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 500
    code = "app_error"
    def __init__(self, message: str = "internal error", *, extra: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.extra = extra or {}


class BadRequest(AppError):
    status_code = 400
    code = "bad_request"


class UnsupportedMediaType(AppError):
    status_code = 415
    code = "unsupported_media_type"


class DatabaseError(AppError):
    status_code = 500
    code = "database_error"


class ServiceError(AppError):
    status_code = 500
    code = "service_error"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    trace_id = str(uuid.uuid4())
    log.error("AppError: %s %s trace_id=%s", exc.code, exc.message, trace_id, exc_info=False, extra=exc.extra)
    payload = {"error": exc.code, "message": exc.message, "trace_id": trace_id}
    if exc.extra:
        payload["extra"] = exc.extra
    return JSONResponse(status_code=exc.status_code, content=payload)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    trace_id = str(uuid.uuid4())
    log.exception("UnhandledError trace_id=%s", trace_id)
    return JSONResponse(
        status_code=500,
        content={"error": "unhandled_error", "message": "internal server error", "trace_id": trace_id},
    )