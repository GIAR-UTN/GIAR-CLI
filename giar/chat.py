"""Sesión de chat interactiva estilo Claude Code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from giar import __version__
from giar.config import Config, get_history_path
from giar.latex import prepare_markdown
from giar.llm import LLMClient, LLMError
from giar.mcp import MCPClient, MCPError
from giar.skills import discover_skills, load_project_context
from giar.tools import (
    Tool,
    ToolRegistry,
    arguments_to_kwargs,
    build_registry,
    wrap_mcp_tool,
)
from giar.ui import banner, console, info, success, warn

MAX_TURNS = 20

PROMPT_STYLE = [
    ("class:giar-prompt", "❯ "),
]


def _render_live_text(title: str, content: str) -> Panel:
    text = Text(content or "…")
    text.append("▍", style="blink blue")
    return Panel(
        text,
        title=title,
        border_style="blue",
        title_align="left",
        padding=(0, 1),
    )


def _render_live_markdown(title: str, content: str) -> Panel:
    body: Any = Markdown(prepare_markdown(content)) if content else Text("…")
    return Panel(
        body,
        title=title,
        border_style="blue",
        title_align="left",
        padding=(0, 1),
    )


def _render_live_reasoning(content: str) -> Panel:
    text = Text(content[-2000:] or "…", style="dim italic")
    text.append("▍", style="blink blue")
    return Panel(
        text,
        title="🤔 Razonando…",
        border_style="dim",
        title_align="left",
        padding=(0, 1),
    )


def _render_reasoning_final(reasoning: str) -> None:
    console.print(
        Panel(
            Text(reasoning.strip(), style="dim italic"),
            title="🤔 Razonamiento",
            border_style="dim",
            title_align="left",
            padding=(0, 1),
        )
    )


def _render_final(title: str, content: str) -> None:
    console.print(
        Panel(
            Markdown(prepare_markdown(content)),
            title=title,
            border_style="blue",
            title_align="left",
            padding=(0, 1),
        )
    )


class ChatSession:
    def __init__(
        self,
        config: Config,
        cwd: Optional[Path] = None,
        model: Optional[str] = None,
        connect_mcps: bool = True,
        show_reasoning: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.cwd = Path.cwd() if cwd is None else Path(cwd)
        self.model = model or config.model or ""
        self.reasoning_effort = config.reasoning_effort or ""
        self.show_reasoning = config.show_reasoning if show_reasoning is None else show_reasoning
        self.connect_mcps = connect_mcps
        self.messages: List[Dict[str, Any]] = []
        self.skills = []
        self.mcp_clients: List[MCPClient] = []
        self.registry: ToolRegistry = ToolRegistry()
        self.llm: Optional[LLMClient] = None
        self._session: Optional[PromptSession] = None

    # -------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self.llm = LLMClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.model,
            extra_headers=self.config.extra_headers,
            reasoning_effort=self.reasoning_effort or None,
        )
        self.skills = [
            s
            for s in discover_skills(self.cwd, Path.home())
            if self.config.is_skill_enabled(s.name)
        ]
        if self.connect_mcps:
            await self._connect_mcps()
        self.registry = build_registry(self.cwd, self.mcp_clients, self.skills)
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        self._print_status()

    async def close(self) -> None:
        if self.llm:
            await self.llm.close()
        for mcp in self.mcp_clients:
            try:
                await mcp.close()
            except Exception:
                pass

    # ---------------------------------------------------------------- startup
    async def _connect_mcps(self) -> None:
        for entry in self.config.enabled_mcps():
            name = entry.get("name", "?")
            url = entry.get("url", "")
            client = MCPClient(name=name, url=url, headers=entry.get("headers") or {})
            try:
                await client.initialize()
                tools = await client.list_tools()
            except MCPError as exc:
                warn(f"MCP '{name}': {exc}")
                await client.close()
                continue
            client.tools = [wrap_mcp_tool(client, t) for t in tools]
            self.mcp_clients.append(client)

    def _print_status(self) -> None:
        console.print()
        info(f"[bold]Proyecto:[/] {self.cwd}")
        info(f"[bold]Modelo:[/] {self.model or '(sin definir)'}  [dim]({self.config.base_url or 'sin endpoint'})[/]")
        if self.reasoning_effort:
            info(f"[bold]Reasoning effort:[/] {self.reasoning_effort}  [dim](/effort para cambiar)[/]")
        info(
            f"[bold]Razonamiento:[/] {'visible' if self.show_reasoning else 'oculto'}  "
            f"[dim](/reasoning on|off)[/]"
        )
        if self.mcp_clients:
            for m in self.mcp_clients:
                info(f"[bold]MCP:[/] {m.name}  [dim]({len(m.tools)} herramientas)[/]")
        else:
            info("[bold]MCP:[/] [dim]ninguno conectado (giar config mcp)[/]")
        if self.skills:
            info(
                f"[bold]Skills:[/] {', '.join(s.name for s in self.skills)}"
            )
        else:
            info("[bold]Skills:[/] [dim]ninguno detectado en este proyecto[/]")
        console.print()
        console.print(
            "[dim]Escribe tu mensaje · /help para ayuda · /exit para salir (o Ctrl+D)[/]"
        )
        console.print()

    def _system_prompt(self) -> str:
        lines = [
            "Eres GIAR, un asistente de IA que trabaja dentro de un proyecto en la terminal, al estilo Claude Code.",
            "",
            f"Directorio de trabajo: {self.cwd}",
            "",
            "Reglas:",
            "- Responde en el idioma del usuario, de forma concisa y útil.",
            "- Usa las herramientas disponibles cuando aporten valor.",
            "- Si el usuario te pide trabajar con un skill, cárgalo primero con read_skill.",
            "- Las herramientas de servidores MCP llevan el prefijo mcp__<servidor>__<herramienta>.",
            "- Cuando termines, entrega el resultado final en texto plano.",
        ]
        project_context = load_project_context(self.cwd, Path.home())
        if project_context:
            lines += ["", "## Contexto del proyecto (AGENTS.md)", project_context]
        if self.skills:
            lines += ["", "## Skills disponibles en este proyecto"]
            for s in self.skills:
                lines.append(f"- {s.name}: {s.description or '(sin descripción)'}")
            lines.append("")
            lines.append("Usa read_skill(<nombre>) para cargar las instrucciones de un skill.")
        if self.mcp_clients:
            lines += ["", "## Servidores MCP conectados"]
            for m in self.mcp_clients:
                names = ", ".join(t.name for t in m.tools) or "(sin herramientas)"
                lines.append(f"- {m.name}: {names}")
        return "\n".join(lines)

    # ------------------------------------------------------------------- main
    def _redraw(self) -> None:
        """Limpia la pantalla y vuelve a mostrar el logo y el estado."""
        console.clear()
        banner()
        self._print_status()

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-l")
        def _(event: Any) -> None:
            self._redraw()

        return kb

    async def run(self) -> None:
        session = PromptSession(
            history=FileHistory(str(get_history_path())),
            message=PROMPT_STYLE,
            complete_while_typing=False,
            key_bindings=self._build_key_bindings(),
        )
        self._session = session
        while True:
            try:
                text = await session.prompt_async()
            except KeyboardInterrupt:
                console.print()
                continue
            except EOFError:
                console.print()
                break
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                action = await self.handle_command(text)
                if action == "exit":
                    success("Hasta luego 👋")
                    break
                continue
            try:
                await self.run_turn(text)
            except KeyboardInterrupt:
                warn("Turno interrumpido")
            except LLMError as exc:
                error_text = str(exc)
                warn(f"Error del modelo: {error_text}")
            console.print()

    # ------------------------------------------------------------ slash cmds
    async def handle_command(self, raw: str) -> Optional[str]:
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/salir"):
            return "exit"
        if cmd in ("/help", "/ayuda", "/?"):
            self._help()
        elif cmd == "/model":
            if arg:
                self.model = arg
                if self.llm:
                    self.llm.model = arg
                success(f"Modelo cambiado a: {self.model}")
            else:
                info(f"Modelo actual: {self.model}")
        elif cmd == "/effort":
            if arg:
                val = arg.strip().lower()
                if val in ("low", "medium", "high"):
                    self.reasoning_effort = val
                    if self.llm:
                        self.llm.reasoning_effort = val
                    success(f"Reasoning effort cambiado a: {val}")
                elif val in ("off", "none", "auto"):
                    self.reasoning_effort = ""
                    if self.llm:
                        self.llm.reasoning_effort = None
                    success("Reasoning effort desactivado")
                else:
                    warn("Usa: /effort low | medium | high | off")
            else:
                info(f"Reasoning effort actual: {self.reasoning_effort or '(desactivado)'}")
        elif cmd == "/reasoning":
            if arg:
                val = arg.strip().lower()
                if val in ("on", "si", "sí", "true", "1"):
                    self.show_reasoning = True
                    self.config.set_show_reasoning(True)
                    success("Mostrando el razonamiento del modelo")
                elif val in ("off", "no", "false", "0"):
                    self.show_reasoning = False
                    self.config.set_show_reasoning(False)
                    success("Ocultando el razonamiento del modelo")
                else:
                    warn("Usa: /reasoning on | off")
            else:
                info(f"Razonamiento: {'visible' if self.show_reasoning else 'oculto'}")
        elif cmd in ("/clear", "/nuevo", "/reset"):
            self.messages = [{"role": "system", "content": self._system_prompt()}]
            self._redraw()
            success("Conversación reiniciada")
        elif cmd == "/skills":
            if self.skills:
                for s in self.skills:
                    info(f"[bold]{s.name}[/]  [dim]{s.description or ''}[/]")
            else:
                info("No hay skills detectados en este proyecto.")
        elif cmd == "/tools":
            for t in self.registry.all():
                info(f"[bold]{t.name}[/]  [dim]({t.source})[/]")
        elif cmd == "/mcp":
            if not self.mcp_clients:
                info("No hay servidores MCP conectados.")
            for m in self.mcp_clients:
                info(
                    f"[bold]{m.name}[/]  [dim]{len(m.tools)} herramientas · {m.url}[/]"
                )
        elif cmd == "/config":
            info(
                "Configuración en: giar config  ·  LLM: giar config llm  ·  MCP: giar config mcp"
            )
        else:
            warn(f"Comando desconocido: {cmd} (usa /help)")
        return None

    def _help(self) -> None:
        info(
            "\n".join(
                [
                    "[bold]Comandos disponibles[/]",
                    "  /help            Muestra esta ayuda",
                    "  /model <name>    Cambia el modelo del LLM",
                    "  /effort <nivel>  Reasoning effort: low | medium | high | off",
                    "  /reasoning on|off  Mostrar/ocultar el pensamiento del modelo",
                    "  /clear           Reinicia la conversación (y muestra el logo)",
                    "  /skills          Lista los skills detectados",
                    "  /tools           Lista las herramientas disponibles",
                    "  /mcp             Estado de los servidores MCP",
                    "  /config          Dónde y cómo configurar",
                    "  /exit            Sale de GIAR (o Ctrl+D)",
                    "",
                    "Fuera de la CLI:",
                    "  giar config llm     Configurar endpoint LLM y API key",
                    "  giar config mcp     Gestionar servidores MCP",
                    "  giar skills         Ver skills del proyecto",
                    "  giar doctor         Comprobar la configuración",
                ]
            )
        )

    # -------------------------------------------------------------- turn
    async def run_turn(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        for _ in range(MAX_TURNS):
            assistant = await self.stream_assistant()
            if assistant["content"]:
                self.messages.append(
                    {"role": "assistant", "content": assistant["content"]}
                )
            calls = assistant["tool_calls"]
            if not calls:
                return
            self.messages.append(
                {
                    "role": "assistant",
                    "content": assistant["content"] or "",
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": c["arguments"] or "{}",
                            },
                        }
                        for c in calls
                    ],
                }
            )
            for call in calls:
                result = await self.execute_tool_call(call)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result or "(sin resultado)",
                    }
                )
            console.print()
        warn("Demasiados turnos de herramientas; deteniendo.")

    async def stream_assistant(self) -> Dict[str, Any]:
        if self.llm is None:
            raise LLMError("Cliente LLM no inicializado")
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []
        title = f"GIAR · {self.model}"

        with Live(
            _render_live_text(title, ""),
            console=console,
            refresh_per_second=20,
            transient=True,
        ) as live:
            async for event in self.llm.stream_chat(
                self.messages,
                tools=self.registry.to_openai_list(),
                model=self.model,
                reasoning_effort=self.reasoning_effort or None,
            ):
                etype = event["type"]
                if etype == "reasoning":
                    reasoning_parts.append(event["content"])
                    if self.show_reasoning and not content_parts:
                        live.update(
                            _render_live_reasoning("".join(reasoning_parts))
                        )
                elif etype == "text":
                    content_parts.append(event["content"])
                    live.update(_render_live_markdown(title, "".join(content_parts)))
                elif etype == "tool_call":
                    idx = event["index"]
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        order.append(idx)
                    tc = tool_calls[idx]
                    if event.get("id"):
                        tc["id"] = event["id"]
                    if event.get("name"):
                        tc["name"] = event["name"]
                    if event.get("arguments"):
                        tc["arguments"] += event["arguments"]
                    label = tc["name"] or "herramienta"
                    live.update(
                        _render_live_text(
                            title,
                            f"\n[dim]⚙ usando {label}…[/dim]",
                        )
                    )
                elif etype == "finish":
                    if content_parts:
                        live.update(_render_live_markdown(title, "".join(content_parts)))

        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        calls = [tool_calls[i] for i in order if tool_calls[i].get("id")]

        if reasoning and self.show_reasoning:
            _render_reasoning_final(reasoning)
        if content:
            _render_final(title, content)
        return {"content": content, "tool_calls": calls}

    async def execute_tool_call(self, call: Dict[str, Any]) -> str:
        tool: Optional[Tool] = self.registry.get(call["name"])
        args = arguments_to_kwargs(call.get("arguments"))
        if tool is None:
            ui_tool = Text(f"❓ herramienta desconocida: {call['name']}", style="yellow")
            console.print(ui_tool)
            return f"Error: la herramienta '{call['name']}' no está disponible."
        console.print(
            Text.assemble(
                ("⚙ ", "blue bold"),
                (f"{tool.name}", "bold cyan"),
                (f"  [{tool.source}]", "dim"),
            )
        )
        if args:
            console.print(Text(json.dumps(args, ensure_ascii=False, indent=2)[:400], style="dim"))
        try:
            result = await tool.handler(**args)
        except Exception as exc:
            result = f"Error ejecutando {tool.name}: {exc}"
        preview = " ".join(result.split())[:220]
        console.print(Text(f"└─ {preview}", style="dim"))
        return result
