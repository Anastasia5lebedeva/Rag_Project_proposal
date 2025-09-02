from __future__ import annotations
import re
from typing import List

from proposal_rag.models.dto import ProposalRequestDTO, ProposalResponseDTO
from proposal_rag.services.llm_client import LLMClient


def _extract_sections(md: str) -> List[str]:
    """
    Разбивает markdown по подзаголовкам (##, ###).
    Возвращает список секций в виде текста.
    """
    if not md:
        return []

    # регулярка: ищем заголовки ## или ###
    parts = re.split(r"(?m)^#{2,3}\s+", md)
    # первый элемент — всё до первого заголовка (если есть)
    if not parts:
        return []

    # чистим пустые и лишние пробелы
    sections = [p.strip() for p in parts if p.strip()]
    return sections


async def generate_proposal_svc(req: ProposalRequestDTO) -> ProposalResponseDTO:
    async with LLMClient() as llm:
        md = await llm.generate_proposal(
            query=req.query,
            context=req.context,
            temperature=req.temperature,
        )

    sections = _extract_sections(md)
    return ProposalResponseDTO(markdown=md, sections=sections)