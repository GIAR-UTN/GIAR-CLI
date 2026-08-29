"""Cliente MCP mínimo para transporte Streamable HTTP.

Implementa el protocolo JSON-RPC 2.0 sobre HTTP con soporte de respuestas
`application/json` y `text/event-stream` (SSE), manejo de `Mcp-Session-Id`
y negociación de `protocolVersion`.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional

import httpx

PROTOCOL_VERSIONS = ["2025-06-18", "2024-11-05"]
DEFAULT_TIMEOUT = 120.0


class MCPError(Exception):
    pass


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _parse_sse(text: str, request_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Parsea un cuerpo SSE y devuelve el mensaje con `id == request_id`
    (o el primero que tenga resultado/error)."""
    messages: List[Dict[str, Any]] = []
    data_lines: List[str] = []

    def flush() -> None:
        if not data_lines:
            return
        raw = "\n".join(data_lines)
        data_lines.clear()
        try:
            messages.append(json.loads(raw))
        except Exception:
            pass

    for line in text.splitlines():
        if line == "":
            flush()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    flush()

    if not messages and text.strip():
        try:
            messages.append(json.loads(text))
        except Exception:
            pass

    if request_id is not None:
        for msg in messages:
            if msg.get("id") == request_id:
                return msg
    for msg in messages:
        if "result" in msg or "error" in msg:
            return msg
    return None


class MCPClient:
    """Cliente de un único servidor MCP streamable-http."""

    def __init__(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.name = name
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.protocol_version: Optional[str] = None
        self.initialized = False
        self._extra_headers = dict(headers or {})
        self._client = httpx.AsyncClient(timeout=timeout)
        self._id = 0

    # ------------------------------------------------------------------ util
    def _new_id(self) -> int:
        self._id += 1
        return self._id

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "giar-cli",
        }
        headers.update(self._extra_headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = await self._client.post(
                self.url, json=payload, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise MCPError(f"No se pudo conectar con {self.name}: {exc}") from exc

        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self.session_id = session_id

        if response.status_code >= 400:
            body = response.text[:400]
            try:
                parsed = response.json()
                if isinstance(parsed, dict) and "error" in parsed:
                    return parsed
            except Exception:
                pass
            raise MCPError(f"HTTP {response.status_code} desde {self.name}: {body}")

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            msg = _parse_sse(response.text, payload.get("id"))
            if msg is None:
                raise MCPError(f"Respuesta SSE vacía o sin mensaje desde {self.name}")
            return msg

        try:
            return response.json()
        except Exception as exc:
            raise MCPError(f"Respuesta no JSON desde {self.name}: {response.text[:200]}") from exc

    async def _notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        await self._client.post(self.url, json=payload, headers=self._headers())

    # -------------------------------------------------------------- lifecycle
    async def initialize(self) -> None:
        errors: List[str] = []
        for version in PROTOCOL_VERSIONS:
            payload = {
                "jsonrpc": "2.0",
                "id": self._new_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": version,
                    "capabilities": {},
                    "clientInfo": {"name": "giar", "version": "0.1.0"},
                },
            }
            msg = await self._post(payload)
            if "error" in msg:
                message = str(msg.get("error", {}).get("message", ""))
                errors.append(message)
                m = re.search(r"(\d{4}-\d{2}-\d{2})", message)
                if m and m.group(1) in PROTOCOL_VERSIONS:
                    continue
                raise MCPError(f"{self.name}: {message}")
            self.protocol_version = version
            break
        else:
            raise MCPError(f"{self.name}: no se pudo negociar el protocolo MCP: {'; '.join(errors)}")

        await self._notify("notifications/initialized")
        self.initialized = True

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ tools
    async def list_tools(self) -> List[Dict[str, Any]]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._new_id(),
            "method": "tools/list",
            "params": {},
        }
        msg = await self._post(payload)
        if "error" in msg:
            raise MCPError(f"{self.name}: {msg['error']}")
        result = msg.get("result") or {}
        return list(result.get("tools", []))

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": self._new_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
        msg = await self._post(payload)
        if "error" in msg:
            return _format_error(msg["error"])
        result = msg.get("result") or {}
        return _format_result(result)

    # --------------------------------------------------------------- resources
    async def list_resources(self) -> List[Dict[str, Any]]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._new_id(),
            "method": "resources/list",
            "params": {},
        }
        msg = await self._post(payload)
        if "error" in msg:
            raise MCPError(f"{self.name}: {msg['error']}")
        result = msg.get("result") or {}
        return list(result.get("resources", []))

    def tool_public_name(self, tool_name: str) -> str:
        """Nombre público (sanitizado) para exponer al LLM."""
        return f"mcp__{_sanitize(self.name)}__{_sanitize(tool_name)}"


def _format_result(result: Dict[str, Any]) -> str:
    if result.get("isError"):
        prefix = "[error MCP] "
    else:
        prefix = ""
    parts: List[str] = []
    for block in result.get("content", []):
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "image":
            parts.append(f"[imagen MCP mediaType={block.get('mimeType', 'desconocido')}]")
        elif block_type == "resource":
            parts.append(str(block.get("resource", "")))
        else:
            parts.append(str(block))
    body = "\n".join(parts).strip()
    structured = result.get("structuredContent")
    if not body and structured is not None:
        body = json.dumps(structured, ensure_ascii=False)
    return prefix + (body or "(sin contenido)")


def _format_error(error: Dict[str, Any]) -> str:
    return f"[error MCP] {error.get('code', '')} {error.get('message', '')}".strip()


def new_request_id() -> int:
    return int(uuid.uuid4().int % 2**31)
