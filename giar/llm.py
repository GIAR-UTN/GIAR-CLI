"""Cliente LLM para endpoints OpenAI-compatible.

Usa el endpoint `{base_url}/chat/completions` (streaming SSE) con el formato
estándar de tool calls. Funciona con OpenAI, Ollama, vLLM, LM Studio,
OpenRouter, Groq, etc.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT = 180.0


class LLMError(Exception):
    pass


EFFORT_VALUES = {"low", "medium", "high"}


def _normalize_effort(value: Optional[str]) -> Optional[str]:
    """Normaliza reasoning_effort a uno de los valores estándar de OpenAI."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("", "off", "none", "auto"):
        return None
    if v not in EFFORT_VALUES:
        raise LLMError(
            f"reasoning_effort inválido: '{value}' (usa low, medium o high)"
        )
    return v


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        extra_headers: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url:
            raise LLMError("No hay base_url configurada. Ejecuta: giar config llm")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        self.base_url = base_url
        self.api_key = (api_key or "").strip()
        self.model = model
        self.reasoning_effort = _normalize_effort(reasoning_effort)
        self.temperature = temperature
        self.extra_headers = dict(extra_headers or {})
        self._client = httpx.AsyncClient(timeout=timeout)

    # ------------------------------------------------------------------ util
    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "giar-cli",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        stream: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if temperature is None:
            temperature = self.temperature
        if temperature is not None:
            payload["temperature"] = temperature
        effort = _normalize_effort(reasoning_effort)
        if effort is None:
            effort = self.reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort
        return payload

    async def close(self) -> None:
        await self._client.aclose()

    # -------------------------------------------------------------- streaming
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Genera eventos:
        {"type": "reasoning", "content": str}   (pensamiento del modelo, si lo envía)
        {"type": "text", "content": str}
        {"type": "tool_call", "index": int, "id": str|None,
         "name": str|None, "arguments": str}
        {"type": "finish", "reason": str|None}
        """
        payload = self._payload(
            messages, tools, model, temperature, reasoning_effort, stream=True
        )
        try:
            async with self._client.stream(
                "POST", self._url(), json=payload, headers=self._headers()
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise LLMError(
                        f"HTTP {response.status_code} desde {self.base_url}: {body[:600]}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        yield {"type": "reasoning", "content": reasoning}
                    content = delta.get("content")
                    if content:
                        yield {"type": "text", "content": content}
                    for tc in delta.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        yield {
                            "type": "tool_call",
                            "index": tc.get("index", 0),
                            "id": tc.get("id"),
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments") or "",
                        }
                    finish = choices[0].get("finish_reason")
                    if finish:
                        yield {"type": "finish", "reason": finish}
        except httpx.HTTPError as exc:
            raise LLMError(f"Error de red con {self.base_url}: {exc}") from exc

    # ------------------------------------------------------------ no streaming
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        payload = self._payload(
            messages, tools, model, temperature, reasoning_effort, stream=False
        )
        try:
            response = await self._client.post(
                self._url(), json=payload, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Error de red con {self.base_url}: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(
                f"HTTP {response.status_code} desde {self.base_url}: {response.text[:600]}"
            )
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Respuesta inesperada del endpoint: {data}") from exc
