from __future__ import annotations
import asyncio
from proposal_rag.services import db_utils, vector_search
from proposal_rag.services.llm_client import LLMClient
from proposal_rag.config.settings import get_settings

s = get_settings()


async def _db_check() -> bool:
    try:
        return await asyncio.to_thread(db_utils.check_connection)
    except Exception:
        return False


async def _vec_check() -> bool:
    try:
        return await vector_search.check_vector_db()
    except Exception:
        return False


async def _llm_check() -> bool:
    try:
        async with LLMClient() as llm:
            return await asyncio.wait_for(llm.check_health(), timeout=float(s.LLM_TIMEOUT_S))
    except Exception:
        return False


async def check_health_svc() -> dict[str, bool]:
    db_ok, vec_ok, llm_ok = await asyncio.gather(
        _db_check(),
        _vec_check(),
        _llm_check(),
        return_exceptions=False,
    )
    return {"db_ok": bool(db_ok), "vec_ok": bool(vec_ok), "llm_ok": bool(llm_ok)}