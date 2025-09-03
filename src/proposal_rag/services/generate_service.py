from __future__ import annotations

import logging
import re
from typing import List
from functools import lru_cache
from proposal_rag.models.dto import ProposalRequestDTO, ProposalResponseDTO
from proposal_rag.services.llm_client import LLMClient, LLMError
from proposal_rag.services.prompt_loader import LOADER
from jinja2 import Template

log = logging.getLogger(__name__)



@lru_cache(maxsize=1)
def _system_ru() -> str:
    return LOADER.get_text("kp_system_ru.txt")


@lru_cache(maxsize=1)
def _user_tpl_ru() -> Template:
    txt = LOADER.get_text("kp_user_ru.j2")
    return Template(txt)


def build_prompt(query: str, context_chunks: List[str]) -> List[dict]:
    sys = {"role": "system", "content": _system_ru()}
    ctx_text = "\n\n---\n".join((c or "").strip() for c in (context_chunks or []) if c and c.strip())
    user_text = _user_tpl_ru().render(query=query.strip(), context=ctx_text, chunks=context_chunks or [])

    user = {"role": "user", "content": user_text}
    return [sys, user]

_HDR_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$")

def _extract_sections(markdown: str) -> List[str]:
    if not markdown:
        return []
    return [m.group(1).strip() for m in _HDR_RE.finditer(markdown)]



async def generate_proposal_svc(req: ProposalRequestDTO) -> ProposalResponseDTO:
    """
    Сервис-обёртка:
    1) собирает messages из промптов и контекста,
    2) вызывает LLMClient.chat,
    3) парсит markdown и секции,
    4) возвращает ProposalResponseDTO.
    """
    messages = build_prompt(req.query, req.context)

    client = LLMClient()
    try:
        markdown = await client.chat(messages=messages, temperature=req.temperature)
    except LLMError as e:
        log.exception("LLM chat failed")
        raise

    sections = _extract_sections(markdown)
    return ProposalResponseDTO(markdown=markdown, sections=sections)