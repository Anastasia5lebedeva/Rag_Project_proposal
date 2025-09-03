from __future__ import annotations
import logging
import os
import sys
from typing import Optional


def _get_default_level() -> int:
    level_name = os.getenv("LOG_LEVEL")
    if level_name:
        return logging.getLevelName(level_name.upper())
    try:
        from proposal_rag.config.settings import get_settings
        s = get_settings()
        level_name = (
            getattr(getattr(s, "App", s), "log_level", None)
            or getattr(s, "LOG_LEVEL", None)
            or "INFO"
        )
        return logging.getLevelName(str(level_name).upper())
    except Exception:
        return logging.INFO


class _PlainFormatter(logging.Formatter):
    default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:
        level = f"{record.levelname:>7s}"
        req_id = getattr(record, "request_id", None)
        rid = f" rid={req_id}" if req_id else ""
        fmt = f"%(asctime)s | {level} | %(name)s | %(message)s{rid}"
        if self._style._fmt != fmt:
            self._style._fmt = fmt
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json
        import time
        obj = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        req_id = getattr(record, "request_id", None)
        if req_id:
            obj["request_id"] = req_id
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


def setup_logging(level: Optional[int | str] = None, json_mode: Optional[bool] = None) -> None:
    if getattr(setup_logging, "_initialized", False):
        return

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    lvl = logging.getLevelName(str(level).upper()) if isinstance(level, str) else (level or _get_default_level())
    root.setLevel(lvl)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(lvl)

    use_json = bool(json_mode) if json_mode is not None else (os.getenv("LOG_FORMAT", "").lower() == "json")
    handler.setFormatter(_JsonFormatter() if use_json else _PlainFormatter("%(message)s", datefmt="%H:%M:%S"))

    root.addHandler(handler)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "asyncio", "psycopg", "asyncpg"):
        logging.getLogger(name).setLevel(lvl)
    logging.getLogger("uvicorn.access").propagate = True

    setup_logging._initialized = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
