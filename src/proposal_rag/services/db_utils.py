from __future__ import annotations
import logging
import psycopg
from proposal_rag.config.settings import get_settings
from proposal_rag.api.errors import DatabaseError

log = logging.getLogger(__name__)
s = get_settings()





def check_connection() -> bool:
    try:
        with psycopg.connect(s.DSN, connect_timeout=s.DB_CONNECT_TIMEOUT) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True
    except Exception as e:
        log.error("Database connection failed: %s", e)
        raise DatabaseError("database connection failed", extra={"reason": str(e)})