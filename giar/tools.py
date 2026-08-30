"""Registro de herramientas disponibles para el LLM.

Incluye herramientas builtin (skills, lectura de archivos del proyecto) y
las herramientas expuestas por los servidores MCP conectados.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from giar.mcp import MCPClient
from giar.skills import Skill

Handler = Callable[..., Awaitable[str]]


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Handler,
        source: str = "builtin",
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.source = source

    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def to_openai_list(self) -> List[Dict[str, Any]]:
        return [t.to_openai() for t in self._tools.values()]


def _project_tool_handler(cwd: Path):
    """Devuelve handlers de lectura de archivos restringidos al proyecto."""

    def resolve(path: str) -> Path:
        base = Path(path).expanduser()
        if not base.is_absolute():
            base = cwd / base
        base = base.resolve()
        if not base.is_relative_to(cwd.resolve()):
            raise ValueError(f"Ruta fuera del proyecto: {path}")
        return base

    async def read_file(path: str = "") -> str:
        p = resolve(path)
        if not p.is_file():
            raise FileNotFoundError(f"No existe el archivo: {path}")
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 60_000:
            content = content[:60_000] + "\n... [truncado]"
        return content

    async def list_dir(path: str = ".") -> str:
        p = resolve(path)
        if not p.is_dir():
            raise NotADirectoryError(f"No existe el directorio: {path}")
        entries = []
        for child in sorted(p.iterdir()):
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.name}{suffix}")
        return "\n".join(entries) if entries else "(directorio vacío)"

    return read_file, list_dir


def build_registry(
    cwd: Path,
    mcps: List[MCPClient],
    skills: List[Skill],
) -> ToolRegistry:
    registry = ToolRegistry()

    read_file, list_dir = _project_tool_handler(cwd)
    registry.register(
        Tool(
            name="read_file",
            description=(
                "Lee el contenido de un archivo del proyecto actual. "
                "Usa rutas relativas al proyecto."
            ),
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=read_file,
        )
    )
    registry.register(
        Tool(
            name="list_dir",
            description="Lista el contenido de un directorio del proyecto actual.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
            handler=list_dir,
        )
    )
    registry.register(
        Tool(
            name="list_skills",
            description="Lista los skills disponibles en este proyecto con su descripción.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: _list_skills(skills),
        )
    )
    registry.register(
        Tool(
            name="read_skill",
            description=(
                "Carga las instrucciones completas de un skill por su nombre. "
                "Usa esta herramienta antes de trabajar según un skill."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=lambda name: _read_skill(skills, name),
        )
    )

    for mcp in mcps:
        for tool in mcp.tools:
            registry.register(tool)
    return registry


async def _list_skills(skills: List[Skill]) -> str:
    if not skills:
        return "No hay skills disponibles en este proyecto."
    lines = [f"- {s.name}: {s.description or '(sin descripción)'}" for s in skills]
    return "\n".join(lines)


async def _read_skill(skills: List[Skill], name: str) -> str:
    for s in skills:
        if s.name == name:
            out = [f"# Skill: {s.name}", f"Origen: {s.source}", ""]
            if s.description:
                out.append(f"Descripción: {s.description}")
                out.append("")
            out.append("## Instrucciones")
            out.append(s.content or "(sin instrucciones)")
            for f in s.files:
                try:
                    out.append("")
                    out.append(f"## Archivo: {f.name}")
                    out.append(f.read_text(encoding="utf-8", errors="replace")[:30_000])
                except OSError:
                    pass
            return "\n".join(out)
    available = ", ".join(s.name for s in skills) or "ninguno"
    return f"Skill '{name}' no encontrado. Skills disponibles: {available}"


def wrap_mcp_tool(mcp: MCPClient, tool: Dict[str, Any]) -> Tool:
    name = mcp.tool_public_name(tool["name"])
    schema = tool.get("inputSchema") or {"type": "object", "properties": {}}

    async def handler(**kwargs: Any) -> str:
        return await mcp.call_tool(tool["name"], kwargs)

    return Tool(
        name=name,
        description=(tool.get("description") or f"Herramienta {tool['name']} de {mcp.name}")
        + f" (servidor MCP: {mcp.name})",
        parameters=schema,
        handler=handler,
        source=f"mcp:{mcp.name}",
    )


def arguments_to_kwargs(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
