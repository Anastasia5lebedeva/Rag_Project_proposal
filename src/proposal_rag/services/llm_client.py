from __future__ import annotations
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt
import logging
from ..config.settings import get_settings

s = get_settings()

class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or s.LLM_API_URL).rstrip("/")
        self.api_key = api_key or s.LLM_API_KEY
        self.model = model or s.LLM_MODEL
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.timeout = s.LLM_TIMEOUT_S

    @retry(wait=wait_exponential(1, 10), stop=stop_after_attempt(3))
    async def chat(self, messages: list[dict]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages, "temperature": 0.1}
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as cli:
            r = await cli.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": s.EMBED_MODEL, "input": texts}
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as cli:
            r = await cli.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            # поддержка как батча, так и единичного входа
            if isinstance(texts, str):
                return data["data"][0]["embedding"]
            return [d["embedding"] for d in data["data"]]