from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ..config.settings import get_settings



s = get_settings()
log = logging.getLogger(__name__)



class LLMError(RuntimeError):
  def _pick(d: Mapping[str, Any], *keys: str) -> Dict[str, Any]:
    return {k: d[k] for k in keys if k in d and d[k] is not None}



class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.base_url = (base_url or s.LLM_API_URL).rstrip("/")
        self.api_key = api_key or s.LLM_API_KEY
        self.model = (model or s.LLM_MODEL).strip()
        self.timeout = float(timeout if timeout is not None else s.LLM_TIMEOUT_S)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(dict(extra_headers))

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )



    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, LLMError)),
        reraise=True,
    )
    async def _post(self, path: str, payload: Mapping[str, Any]) -> httpx.Response:
        try:
            resp = await self._client.post(path, json=payload)
        except httpx.TimeoutException as e:
            log.warning("LLM timeout: %s %s", path, e)
            raise
        except httpx.TransportError as e:
            log.warning("LLM transport error: %s %s", path, e)
            raise

        if resp.status_code >= 400:
            msg = f"HTTP {resp.status_code} for {path}"
            body = resp.text[:1000] if log.isEnabledFor(logging.DEBUG) else ""
            log.error("LLM error %s %s", msg, body)
            if resp.status_code in (408, 429, 500, 502, 503, 504):
                raise LLMError(msg)
            raise httpx.HTTPStatusError(msg, request=resp.request, response=resp)
        return resp




    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> str:

        payload: Dict[str, Any] = {
            "model": (model or self.model),
            "messages": list(messages),
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if extra_params:
            payload.update(extra_params)

        data = (await self._post("/chat/completions", payload)).json()

        try:

            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            try:
                content = data["choices"][0]["text"]
            except Exception as e:
                log.error("Unexpected chat response schema: %s", json.dumps(data)[:4000])
                raise LLMError(f"Invalid chat response schema: {e}") from e

        return (content or "").strip()


    async def embed(
        self,
        texts: Union[str, Sequence[str]],
        *,
        model: Optional[str] = None,
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> Union[List[float], List[List[float]]]:
        inputs = texts if isinstance(texts, str) else list(texts)
        payload: Dict[str, Any] = {"model": (model or s.EMBED_MODEL), "input": inputs}
        if extra_params:
            payload.update(extra_params)

        data = (await self._post("/embeddings", payload)).json()
        try:
            items = data["data"]
            if isinstance(texts, str):
                return items[0]["embedding"]
            return [item["embedding"] for item in items]
        except (KeyError, IndexError, TypeError) as e:
            log.error("Unexpected embeddings response schema: %s", json.dumps(data)[:4000])
            raise LLMError(f"Invalid embeddings response schema: {e}") from e




    async def check_health(self) -> bool:
        try:
            out = await self.chat([{"role": "user", "content": "ping"}], temperature=0.0, max_tokens=5)
            ok_chat = bool(out)
            _ = await self.embed("ping")
            ok_emb = True
            return ok_chat and ok_emb
        except Exception as e:
            log.warning("LLM health check failed: %s", e)
            return False