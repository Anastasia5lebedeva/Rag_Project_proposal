# src/proposal_rag/api/errors.py
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)


# ===== Унифицированная схема ответа об ошибке =====

class ErrorResponse(BaseModel):
    error: str                 # машинное имя кода ошибки (snake_case)
    message: str               # человекочитаемое описание
    request_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None  # любые полезные поля: doc_id, dsn, model и т.п.


# ===== Базовый класс и доменные ошибки =====

class AppError(Exception):
    """Базовая доменная ошибка сервиса с HTTP-статусом и кодом."""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "app_error"

    def __init__(self, message: str = "internal error", *, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.extra = extra or {}


# --- Входные данные (от пользователя / API) ---

class BadRequest(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"


class UnsupportedMediaType(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"


class ValidationError(AppError):
    """Бизнес-валидация прошла в FastAPI/Pydantic, но не прошла доменная логика."""
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "validation_error"


# --- Документы / парсинг ---

class DocumentParseError(AppError):
    """Не удалось распарсить входной документ (docx/pdf/csv)."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "document_parse_error"


class DocumentNotFound(AppError):
    """Запрошенный документ/шаблон отсутствует в хранилище."""
    status_code = status.HTTP_404_NOT_FOUND
    code = "document_not_found"


# --- Хранилище / поиск ---

class DatabaseError(AppError):
    """Ошибка при работе с реляционной БД (метаданные КП и т.п.)."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "database_error"


class DatabaseUnavailable(AppError):
    """База данных недоступна (нет соединения/таймаут/инициализация)."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "database_unavailable"


class VectorDBError(AppError):
    """Ошибка векторного поиска (pgvector/FAISS/Weaviate/и т.п.)."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "vector_db_error"


# --- Модели / LLM ---

class EmbeddingError(AppError):
    """Не удалось сгенерировать эмбеддинг (модель/лимиты/таймаут)."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "embedding_error"


class LLMServiceError(AppError):
    """Внешний LLM API (GPT/Claude/DeepSeek) недоступен/ошибка сети."""
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "llm_service_error"


class LLMResponseError(AppError):
    """LLM вернул пустой/повреждённый/неожиданный ответ."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "llm_response_error"


# --- Общее ---

class ServiceError(AppError):
    """Прочие ошибки сервиса, не попавшие в конкретные категории."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "service_error"


# ===== Вспомогательные функции =====

def _get_request_id(request: Request) -> str:
    """Берём сквозной request_id из middleware; если его нет — генерируем."""
    rid = getattr(request.state, "request_id", None)
    return rid or str(uuid.uuid4())


# ===== Глобальные обработчики исключений =====

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = _get_request_id(request)
    # пишем в лог без stacktrace для ожидаемых доменных ошибок
    log.error(
        "AppError: code=%s msg=%s",
        exc.code, exc.message,
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
    """Удобная точка подключения — зови из create_app()."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
