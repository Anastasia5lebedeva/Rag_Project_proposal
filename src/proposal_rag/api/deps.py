from __future__ import annotations
import logging
from ..config.settings import get_settings, AppSettings


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def settings() -> AppSettings:
    return get_settings()