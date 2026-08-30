"""Sesión de chat interactiva estilo Claude Code."""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    Dimension,
    FormattedTextControl,
    HSplit,
    Layout,
    ScrollablePane,
    Window,
)
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.widgets import TextArea
from rich.color import ColorType
from rich.console import Console as RichConsole
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style as RichStyle
from rich.text import Text

from giar import __version__
from giar.config import Config, get_history_path, get_home
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
from giar.ui import GIAR_ART, BLUE, banner, console

MAX_DEGENERATE_RETRIES = 3

_ELLIPSIS = re.compile(r"[.…]{2,}")


def _is_degenerate(content: str) -> bool:
    """Detecta salida degenerada del modelo: "..." en cadena o repetición excesiva."""
    text = content.strip()
    if len(text) < 4:
        return False
    ellipsis_chars = sum(len(m.group(0)) for m in _ELLIPSIS.finditer(text))
    if ellipsis_chars / len(text) > 0.25:
        return True
    words = re.findall(r"\w+", text.lower())
    if len(words) < 5:
        return False
    return len(set(words)) / len(words) < 0.4

PROMPT_STYLE = [
    ("class:giar-prompt", "❯ "),
]

_ANSI_NAMES = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]


def _rich_color_to_pt(color: Any) -> str:
    """Convierte un color de rich a un color de prompt_toolkit."""
    if color is None:
        return ""
    ctype = color.type
    if ctype in (ColorType.TRUECOLOR, ColorType.EIGHT_BIT):
        return color.get_truecolor().hex
    if ctype == ColorType.STANDARD:
        n = color.number
        if n < 8:
            return "ansi" + _ANSI_NAMES[n]
        return "ansi" + ("bright" if n < 16 else "") + _ANSI_NAMES[n % 8]
    return ""


def _rich_style_to_pt(style: RichStyle) -> str:
    """Convierte un estilo de rich a una cadena de estilo de prompt_toolkit."""
    parts = []
    if style.bold:
        parts.append("bold")
    if style.italic:
        parts.append("italic")
    if style.underline:
        parts.append("underline")
    if style.dim:
        parts.append("dim")
    if style.strike:
        parts.append("strike")
    if style.reverse:
        parts.append("reverse")
    fg = _rich_color_to_pt(style.color)
    if fg:
        parts.append("fg:" + fg)
    bg = _rich_color_to_pt(style.bgcolor)
    if bg:
        parts.append("bg:" + bg)
    return " ".join(parts)


def rich_to_fragments(renderable: Any, width: Optional[int] = None) -> StyleAndTextTuples:
    """Renderiza un objeto de rich como fragmentos de prompt_toolkit (con color)."""
    rc = RichConsole(width=width, color_system=None)
    out: StyleAndTextTuples = []
    for seg in rc.render(renderable):
        style = seg.style or RichStyle()
        out.append((_rich_style_to_pt(style), seg.text))
    return out


class _ScrollWindow(Window):
    """Ventana de conversación que avisa cuando el usuario intenta hacer scroll."""

    def __init__(self, *args: Any, on_scroll: Any = None, **kwargs: Any) -> None:
        self.on_scroll = on_scroll
        super().__init__(*args, **kwargs)

    def _mouse_handler(self, mouse_event: Any) -> Any:
        if mouse_event.event_type in (
            MouseEventType.SCROLL_UP,
            MouseEventType.SCROLL_DOWN,
        ):
            if self.on_scroll:
                self.on_scroll(mouse_event.event_type)
            return None
        return super()._mouse_handler(mouse_event)


class _Tui:
    """Pantalla completa con el logo de GIAR fijo arriba y la conversación abajo."""

    HEADER_HEIGHT = 8

    def __init__(self, session: "ChatSession") -> None:
        self.session = session
        self.frags: StyleAndTextTuples = []
        self.live_frags: StyleAndTextTuples = []
        self._follow = True
        self._busy = False
        self._current_task: Optional[asyncio.Task] = None

        self.control = FormattedTextControl([("", "")])
        self.conv_win = _ScrollWindow(
            self.control,
            wrap_lines=True,
            always_hide_cursor=True,
            on_scroll=self._manual_scroll,
        )
        self.pane = ScrollablePane(self.conv_win, show_scrollbar=True)
        self.input_area = TextArea(
            prompt=PROMPT_STYLE,
            multiline=False,
            history=FileHistory(str(get_history_path())),
        )
        self.input_area.accept_handler = self._on_accept

        # Ctrl+L lo captura por defecto `_default_bindings` (clear_screen) y
        # no repinta nada visible: lo redefinimos en el control del input, que
        # tiene máxima prioridad, para que repinte (limpia y muestra el estado).
        ctrl_l = KeyBindings()
        ctrl_l.add("c-l")(lambda e: self.session._redraw())
        self.input_area.control.key_bindings = ctrl_l

        kb = KeyBindings()
        kb.add("enter")(lambda e: self.input_area.buffer.validate_and_handle())
        kb.add("c-c")(self._on_ctrl_c)
        kb.add("c-d")(lambda e: e.app.exit())
        kb.add("c-l")(lambda e: self.session._redraw())
        kb.add("pageup")(lambda e: self._manual_scroll(MouseEventType.SCROLL_UP))
        kb.add("pagedown")(lambda e: self._manual_scroll(MouseEventType.SCROLL_DOWN))
        kb.add("up")(lambda e: self._history(e, backward=True))
        kb.add("down")(lambda e: self._history(e, backward=False))

        self.app = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(
                            FormattedTextControl(self._build_header()),
                            height=Dimension.exact(self.HEADER_HEIGHT),
                        ),
                        self.pane,
                        self.input_area,
                    ]
                )
            ),
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
        )

    def _build_header(self) -> StyleAndTextTuples:
        art = Text(GIAR_ART, style=BLUE)
        tag = Text(f"  v{__version__} · asistente IA por consola", style="dim italic")
        return rich_to_fragments(Group(art, tag), self._width())

    def _width(self) -> int:
        try:
            return get_app().output.get_size().columns - 2
        except Exception:
            return 80

    def _visible_height(self) -> int:
        try:
            return get_app().output.get_size().rows - self.HEADER_HEIGHT - 1
        except Exception:
            return 20

    def _content_lines(self) -> int:
        text = "".join(t for _, t in self.frags + self.live_frags)
        width = max(1, self._width())
        lines = 0
        for raw in text.split("\n"):
            lines += max(1, math.ceil(len(raw) / width))
        return lines

    def _max_scroll(self) -> int:
        return max(0, self._content_lines() - self._visible_height())

    def _refresh(self) -> None:
        self.control.text = self.frags + self.live_frags
        if self._follow:
            self.pane.vertical_scroll = self._max_scroll()
        self.app.invalidate()

    def _manual_scroll(self, event_type: str) -> None:
        self._follow = False
        delta = -3 if event_type == MouseEventType.SCROLL_UP else 3
        self.pane.vertical_scroll = min(
            self._max_scroll(), max(0, self.pane.vertical_scroll + delta)
        )
        self.app.invalidate()

    def emit(self, renderable: Any) -> None:
        """Añade un renderable de rich a la conversación y sigue el final."""
        if self.frags:
            self.frags.append(("", "\n"))
        self.frags.extend(rich_to_fragments(renderable, self._width()))
        self._follow = True
        self._refresh()

    def emit_blank(self) -> None:
        self.frags.append(("", "\n"))
        self._refresh()

    def set_live(self, renderable: Optional[Any]) -> None:
        """Actualiza el bloque en streaming; `None` lo quita."""
        if renderable is None:
            self.live_frags = []
        else:
            self.live_frags = rich_to_fragments(renderable, self._width())
        self._refresh()

    def clear(self) -> None:
        self.frags = []
        self.live_frags = []
        self._follow = True
        self._refresh()

    def _history(self, event: Any, backward: bool) -> None:
        buf = self.input_area.buffer
        if backward:
            buf.history_backward()
        else:
            buf.history_forward()

    def _on_ctrl_c(self, event: Any) -> None:
        if self._busy and self._current_task:
            self._current_task.cancel()
        else:
            self.input_area.buffer.reset()

    def _on_accept(self, buf: Buffer) -> bool:
        text = self.input_area.text.strip()
        if not text:
            return False
        if self._busy and not text.startswith("/"):
            self.session._emit_markup(
                "[bold yellow]⚠[/] Aún procesando el mensaje anterior…"
            )
            return False
        if not text.startswith("/"):
            self._busy = True
        self._current_task = get_app().create_background_task(self._handle(text))
        return False

    async def _handle(self, text: str) -> None:
        session = self.session
        if text.startswith("/"):
            action = await session.handle_command(text)
            if action == "exit":
                self.app.exit()
            return
        try:
            await session.run_turn(text)
        except asyncio.CancelledError:
            session._emit_markup("[bold yellow]⚠[/] Turno interrumpido")
        except KeyboardInterrupt:
            session._emit_markup("[bold yellow]⚠[/] Turno interrumpido")
        except LLMError as exc:
            session._emit_markup(f"[bold yellow]⚠[/] Error del modelo: {exc}")
        finally:
            self._busy = False
            self._current_task = None

    async def run(self) -> None:
        await self.app.run_async()


class _TuiLive:
    """Adaptador de `Live` de rich para la pantalla completa de GIAR."""

    def __init__(self, tui: _Tui) -> None:
        self._tui = tui

    def update(self, renderable: Any) -> None:
        self._tui.set_live(renderable)

    def __enter__(self) -> "_TuiLive":
        return self

    def __exit__(self, *args: Any) -> None:
        self._tui.set_live(None)


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


class ChatSession:
    def __init__(
        self,
        config: Config,
        cwd: Optional[Path] = None,
        model: Optional[str] = None,
        connect_mcps: bool = True,
        show_reasoning: Optional[bool] = None,
        tui: bool = False,
    ) -> None:
        self.config = config
        self.cwd = Path.cwd() if cwd is None else Path(cwd)
        self.model = model or config.model or ""
        self.reasoning_effort = config.reasoning_effort or ""
        self.show_reasoning = config.show_reasoning if show_reasoning is None else show_reasoning
        self.max_turns = config.max_turns
        self.connect_mcps = connect_mcps
        self.messages: List[Dict[str, Any]] = []
        self.skills = []
        self.mcp_clients: List[MCPClient] = []
        self.registry: ToolRegistry = ToolRegistry()
        self.llm: Optional[LLMClient] = None
        self._session: Optional[PromptSession] = None
        self.tui: Optional[_Tui] = None
        if tui:
            self.tui = _Tui(self)

    # -------------------------------------------------------------- output
    def _emit(self, renderable: Any) -> None:
        """Vuelca un renderable de rich a la conversación (o a la terminal)."""
        if self.tui is not None:
            self.tui.emit(renderable)
        else:
            console.print(renderable)

    def _emit_markup(self, markup: str) -> None:
        if self.tui is not None:
            self.tui.emit(Text.from_markup(markup))
        else:
            console.print(markup)

    def _emit_blank(self) -> None:
        if self.tui is not None:
            self.tui.emit_blank()
        else:
            console.print()

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
            for s in discover_skills(self.cwd, get_home())
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
                self._emit_markup(f"[bold yellow]⚠[/] MCP '{name}': {exc}")
                await client.close()
                continue
            client.tools = [wrap_mcp_tool(client, t) for t in tools]
            self.mcp_clients.append(client)

    def _print_status(self) -> None:
        lines = [f"[bold]Proyecto:[/] {self.cwd}"]
        lines.append(
            f"[bold]Modelo:[/] {self.model or '(sin definir)'}  "
            f"[dim]({self.config.base_url or 'sin endpoint'})[/]"
        )
        if self.reasoning_effort:
            lines.append(
                f"[bold]Reasoning effort:[/] {self.reasoning_effort}  "
                f"[dim](/effort para cambiar)[/]"
            )
        lines.append(
            f"[bold]Límite de turnos:[/] {self.max_turns}  "
            f"[dim](/turns para cambiar; deja que las secuencias largas terminen)[/]"
        )
        lines.append(
            f"[bold]Razonamiento:[/] {'visible' if self.show_reasoning else 'oculto'}  "
            f"[dim](/reasoning on|off)[/]"
        )
        if self.mcp_clients:
            for m in self.mcp_clients:
                lines.append(f"[bold]MCP:[/] {m.name}  [dim]({len(m.tools)} herramientas)[/]")
        else:
            lines.append("[bold]MCP:[/] [dim]ninguno conectado (giar config mcp)[/]")
        if self.skills:
            lines.append(f"[bold]Skills:[/] {', '.join(s.name for s in self.skills)}")
        else:
            lines.append("[bold]Skills:[/] [dim]ninguno detectado en este proyecto[/]")
        lines.append("")
        lines.append("[bright_green]Escribe tu mensaje · /help para ayuda · /exit para salir (o Ctrl+D)[/]")
        self._emit_markup("\n".join(lines))

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
            "- Puedes ejecutar secuencias largas de herramientas (bucles, movimientos, lecturas repetidas) "
            "hasta completar el objetivo: no hace falta dar la respuesta final hasta haber terminado. "
            "Por ejemplo, para mover un robot puedes avanzar poco a poco consultando la odometría "
            "en cada paso y parar al alcanzar el objetivo.",
            "- No repitas indefinidamente: si el objetivo no avanza o falla, detente y explica qué está pasando.",
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
        if self.tui is not None:
            self.tui.clear()
            self._print_status()
        else:
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
        if self.tui is not None:
            await self.tui.run()
            return
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
                    console.print("[bold green]✔[/] Hasta luego 👋")
                    break
                continue
            try:
                await self.run_turn(text)
            except KeyboardInterrupt:
                console.print("[bold yellow]⚠[/] Turno interrumpido")
            except LLMError as exc:
                console.print(f"[bold yellow]⚠[/] Error del modelo: {exc}")
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
                self._emit_markup(f"[bold green]✔[/] Modelo cambiado a: {self.model}")
            else:
                self._emit_markup(f"[bold]Modelo actual:[/] {self.model}")
        elif cmd == "/effort":
            if arg:
                val = arg.strip().lower()
                if val in ("low", "medium", "high"):
                    self.reasoning_effort = val
                    if self.llm:
                        self.llm.reasoning_effort = val
                    self._emit_markup(f"[bold green]✔[/] Reasoning effort cambiado a: {val}")
                elif val in ("off", "none", "auto"):
                    self.reasoning_effort = ""
                    if self.llm:
                        self.llm.reasoning_effort = None
                    self._emit_markup("[bold green]✔[/] Reasoning effort desactivado")
                else:
                    self._emit_markup("[bold yellow]⚠[/] Usa: /effort low | medium | high | off")
            else:
                self._emit_markup(f"[bold]Reasoning effort actual:[/] {self.reasoning_effort or '(desactivado)'}")
        elif cmd == "/reasoning":
            if arg:
                val = arg.strip().lower()
                if val in ("on", "si", "sí", "true", "1"):
                    self.show_reasoning = True
                    self.config.set_show_reasoning(True)
                    self._emit_markup("[bold green]✔[/] Mostrando el razonamiento del modelo")
                elif val in ("off", "no", "false", "0"):
                    self.show_reasoning = False
                    self.config.set_show_reasoning(False)
                    self._emit_markup("[bold green]✔[/] Ocultando el razonamiento del modelo")
                else:
                    self._emit_markup("[bold yellow]⚠[/] Usa: /reasoning on | off")
            else:
                self._emit_markup(f"[bold]Razonamiento:[/] {'visible' if self.show_reasoning else 'oculto'}")
        elif cmd in ("/clear", "/nuevo", "/reset"):
            self.messages = [{"role": "system", "content": self._system_prompt()}]
            self._redraw()
            self._emit_markup("[bold green]✔[/] Conversación reiniciada")
        elif cmd == "/turns":
            if arg:
                try:
                    value = int(arg.strip())
                except ValueError:
                    self._emit_markup("[bold yellow]⚠[/] Usa: /turns <número> (mínimo 1)")
                else:
                    if value < 1:
                        self._emit_markup("[bold yellow]⚠[/] El mínimo es 1 turno.")
                    else:
                        self.max_turns = value
                        self.config.set_max_turns(value)
                        self._emit_markup(
                            f"[bold green]✔[/] Límite de turnos por mensaje: {value} "
                            "(permite secuencias largas de herramientas)"
                        )
            else:
                self._emit_markup(f"[bold]Límite de turnos actual:[/] {self.max_turns}  [dim](/turns <n> para cambiar)[/]")
        elif cmd == "/skills":
            if self.skills:
                for s in self.skills:
                    self._emit_markup(f"[bold]{s.name}[/]  [dim]{s.description or ''}[/]")
            else:
                self._emit_markup("No hay skills detectados en este proyecto.")
        elif cmd == "/tools":
            for t in self.registry.all():
                self._emit_markup(f"[bold]{t.name}[/]  [dim]({t.source})[/]")
        elif cmd == "/mcp":
            if not self.mcp_clients:
                self._emit_markup("No hay servidores MCP conectados.")
            for m in self.mcp_clients:
                self._emit_markup(
                    f"[bold]{m.name}[/]  [dim]{len(m.tools)} herramientas · {m.url}[/]"
                )
        elif cmd == "/config":
            self._emit_markup(
                "Configuración en: giar config  ·  LLM: giar config llm  ·  MCP: giar config mcp"
            )
        else:
            self._emit_markup(f"[bold yellow]⚠[/] Comando desconocido: {cmd} (usa /help)")
        return None

    def _help(self) -> None:
        self._emit_markup(
            "\n".join(
                [
                    "[bold]Comandos disponibles[/]",
                    "  /help            Muestra esta ayuda",
                    "  /model <name>    Cambia el modelo del LLM",
                    "  /effort <nivel>  Reasoning effort: low | medium | high | off",
                    "  /reasoning on|off  Mostrar/ocultar el pensamiento del modelo",
                    "  /turns <n>       Límite de turnos de herramientas por mensaje",
                    "  /clear           Reinicia la conversación",
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
        if self.tui is not None:
            self._emit(
                Panel(
                    Text(text),
                    title="Tú",
                    border_style="green",
                    title_align="left",
                    padding=(0, 1),
                )
            )
        self.messages.append({"role": "user", "content": text})
        degenerate_retries = 0
        for _ in range(self.max_turns):
            assistant = await self.stream_assistant()
            content = assistant["content"]
            calls = assistant["tool_calls"]
            if assistant.get("degenerate"):
                degenerate_retries += 1
                if degenerate_retries <= MAX_DEGENERATE_RETRIES:
                    self._emit_markup("[bold yellow]⚠[/] Salida degenerada del modelo; reintentando…")
                    continue
                self._emit_markup(
                    "[bold yellow]⚠[/] El modelo sigue generando salida degenerada; "
                    "no se guardó la respuesta. Inténtalo de nuevo."
                )
                return
            if not calls:
                if content:
                    self.messages.append(
                        {"role": "assistant", "content": content}
                    )
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
            self._emit_blank()
        self._emit_markup("[bold yellow]⚠[/] Demasiados turnos de herramientas; deteniendo.")

    async def stream_assistant(self) -> Dict[str, Any]:
        if self.llm is None:
            raise LLMError("Cliente LLM no inicializado")
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []
        title = f"GIAR · {self.model}"

        if self.tui is not None:
            live: Any = _TuiLive(self.tui)
        else:
            live = Live(
                _render_live_text(title, ""),
                console=console,
                refresh_per_second=20,
                transient=True,
            )
        with live:
            live.update(_render_live_text(title, ""))
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

        degenerate = bool(content) and not calls and _is_degenerate(content)

        if reasoning and self.show_reasoning:
            self._emit(
                Panel(
                    Text(reasoning.strip(), style="dim italic"),
                    title="🤔 Razonamiento",
                    border_style="dim",
                    title_align="left",
                    padding=(0, 1),
                )
            )
        if content and not degenerate:
            self._emit(
                Panel(
                    Markdown(prepare_markdown(content)),
                    title=title,
                    border_style="blue",
                    title_align="left",
                    padding=(0, 1),
                )
            )
        return {"content": content, "tool_calls": calls, "degenerate": degenerate}

    async def execute_tool_call(self, call: Dict[str, Any]) -> str:
        tool: Optional[Tool] = self.registry.get(call["name"])
        args = arguments_to_kwargs(call.get("arguments"))
        if tool is None:
            self._emit(Text(f"❓ herramienta desconocida: {call['name']}", style="yellow"))
            return f"Error: la herramienta '{call['name']}' no está disponible."
        self._emit(
            Text.assemble(
                ("⚙ ", "blue bold"),
                (f"{tool.name}", "bold cyan"),
                (f"  [{tool.source}]", "dim"),
            )
        )
        if args:
            self._emit(Text(json.dumps(args, ensure_ascii=False, indent=2)[:400], style="dim"))
        try:
            result = await tool.handler(**args)
        except Exception as exc:
            result = f"Error ejecutando {tool.name}: {exc}"
        preview = " ".join(result.split())[:220]
        self._emit(Text(f"└─ {preview}", style="dim"))
        return result
