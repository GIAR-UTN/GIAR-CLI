"""Elementos de interfaz: banner, estilos y helpers de renderizado."""

from __future__ import annotations

from typing import Iterable, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from giar import __version__

console = Console(highlight=False)

GIAR_ART = r"""██████╗  ██╗ █████╗ ██████╗ 
██╔════╝ ██║██╔══██╗██╔══██╗
██║  ███╗██║███████║██████╔╝
██║   ██║██║██╔══██║██╔══██╗
╚██████╔╝██║██║  ██║██║  ██║
 ╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝"""

BLUE = "bold #4FC3F7"
DIM = "dim"


def banner() -> None:
    """Imprime el arte ASCII de GIAR en azul."""
    art = Text(GIAR_ART, style=BLUE)
    tag = Text(
        f"  v{__version__} · asistente IA por consola",
        style="dim italic",
    )
    console.print(Group(art, tag), justify="center")
    console.print()


def hline(title: Optional[str] = None) -> None:
    if title:
        console.rule(f"[bold]{title}[/bold]", style="blue")
    else:
        console.rule(style="blue")


def info(msg: str) -> None:
    console.print(msg, style="cyan")


def success(msg: str) -> None:
    console.print(f"[bold green]✔[/] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]⚠[/] {msg}")


def error(msg: str) -> None:
    console.print(f"[bold red]✖[/] {msg}")


def assistant_panel(title: str, content: str) -> None:
    """Panel final para una respuesta del asistente."""
    console.print(
        Panel(
            Text(content),
            title=title,
            border_style="blue",
            title_align="left",
            padding=(0, 1),
        )
    )


def tool_started(name: str, args: str, source: str) -> None:
    label = f"[bold blue]🔧 {name}[/bold blue]"
    if source:
        label += f" [dim]({source})[/dim]"
    console.print(f"{label}")
    if args:
        console.print(f"[dim]  {args[:400]}[/dim]")


def tool_result_preview(kind: str, preview: str) -> None:
    console.print(f"[dim]  └─ {kind}: {preview[:200]}[/dim]")


def render_tool_call_block(
    lines: Iterable[str],
    title: str = "Herramienta",
    border: str = "cyan",
) -> None:
    console.print(
        Panel(
            "\n".join(lines),
            title=title,
            border_style=border,
            title_align="left",
            padding=(0, 1),
        )
    )
