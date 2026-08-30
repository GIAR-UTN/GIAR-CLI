"""Punto de entrada de la CLI de GIAR: comandos y asistentes de configuración."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.prompt import Confirm, Prompt
from rich.table import Table

from giar import __version__
from giar.config import Config, get_config_path, get_home
from giar.llm import LLMClient, LLMError
from giar.mcp import MCPClient, MCPError
from giar.skills import discover_skills, find_agents_md
from giar.ui import banner, console, error, hline, info, success, warn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="giar",
        description="GIAR - Asistente IA por consola (OpenAI-compatible + MCP + Skills).",
    )
    parser.add_argument("--version", action="version", version=f"giar {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="comando")

    p_chat = sub.add_parser("chat", help="Inicia la sesión de chat interactiva (por defecto)")
    p_chat.add_argument("-p", "--prompt", help="Pregunta única, sin modo interactivo")
    p_chat.add_argument("-m", "--model", help="Modelo a usar en esta sesión")
    p_chat.add_argument("--mcp-off", action="store_true", help="No conectar servidores MCP")
    p_chat.add_argument("--no-reasoning", action="store_true",
                        help="Ocultar el pensamiento (reasoning) del modelo")
    p_chat.add_argument("-c", "--cwd", default=None, help="Directorio de trabajo (default: actual)")

    p_config = sub.add_parser("config", help="Configuración (endpoint LLM, api key, MCPs)")
    p_config.add_argument("what", nargs="?", choices=["llm", "mcp", "show"],
                          help="llm | mcp | show")

    p_mcp = sub.add_parser("mcp", help="Gestionar servidores MCP streamable-http")
    p_mcp_sub = p_mcp.add_subparsers(dest="mcp_action", metavar="accion")
    p_add = p_mcp_sub.add_parser("add", help="Añadir un servidor MCP")
    p_add.add_argument("name", help="Nombre del servidor")
    p_add.add_argument("url", help="URL del endpoint streamable-http")
    p_add.add_argument("--token", default=None, help="Token Bearer para autenticación")
    p_add.add_argument("--header", action="append", default=[], metavar="Clave: Valor",
                       help="Cabecera adicional (repetible)")
    p_mcp_sub.add_parser("list", help="Listar servidores MCP configurados")
    p_mcp_sub.add_parser("test", help="Probar la conexión con los MCPs configurados")
    p_rm = p_mcp_sub.add_parser("remove", help="Eliminar un servidor MCP")
    p_rm.add_argument("name")
    p_toggle = p_mcp_sub.add_parser("toggle", help="Activar/desactivar un servidor MCP")
    p_toggle.add_argument("name")
    p_toggle.add_argument("state", choices=["on", "off"])

    p_skills = sub.add_parser("skills", help="Listar skills detectados en el proyecto")
    p_skills.add_argument("-c", "--cwd", default=None)

    p_doctor = sub.add_parser("doctor", help="Comprobar configuración, LLM y MCPs")
    p_doctor.add_argument("--llm", action="store_true", help="Solo comprobar LLM")
    p_doctor.add_argument("--mcp", action="store_true", help="Solo comprobar MCPs")

    sub.add_parser("version", help="Mostrar versión")

    return parser


# ------------------------------------------------------------------- wizards
def wizard_llm(cfg: Config) -> None:
    hline("Configuración del LLM (endpoint OpenAI-compatible)")
    console.print(
        "[dim]Ejemplos de base_url:\n"
        "  https://api.openai.com/v1\n"
        "  https://api.groq.com/openai/v1\n"
        "  https://openrouter.ai/api/v1\n"
        "  http://localhost:11434/v1   (Ollama)\n"
        "  http://localhost:1234/v1    (LM Studio)[/]\n"
    )
    current_url = cfg.base_url or "https://api.openai.com/v1"
    base_url = Prompt.ask("Base URL", default=current_url).strip().rstrip("/")
    if not base_url:
        warn("Base URL vacía; cancelando.")
        return

    current_model = cfg.model or "gpt-4o-mini"
    model = Prompt.ask("Modelo", default=current_model).strip()
    if not model:
        warn("Modelo vacío; cancelando.")
        return

    api_key = cfg.api_key
    if api_key:
        console.print(f"API key actual: [dim]{api_key[:6]}******[/]")
        if not Confirm.ask("¿Cambiar la API key?", default=False):
            api_key = cfg.api_key
        else:
            api_key = getpass.getpass("API key (oculta): ").strip()
    else:
        api_key = getpass.getpass("API key (oculta, Enter para omitir): ").strip()

    current_effort = cfg.reasoning_effort or "medium"
    effort = Prompt.ask(
        "Reasoning effort (low | medium | high | none)",
        choices=["low", "medium", "high", "none"],
        default=current_effort,
    )
    effort_value = "" if effort == "none" else effort

    cfg.set_provider(base_url, model, api_key or None, reasoning_effort=effort_value)
    success("Configuración guardada.")

    if Confirm.ask("¿Probar la conexión ahora?", default=True):
        test_llm(cfg)


def wizard_mcp(cfg: Config) -> None:
    while True:
        hline("Servidores MCP (streamable-http)")
        table = Table(box=None)
        table.add_column("Nombre", style="bold cyan")
        table.add_column("URL")
        table.add_column("Estado")
        table.add_column("Cabeceras", style="dim")
        if not cfg.mcps:
            console.print("[dim]No hay servidores MCP configurados.[/]")
        for m in cfg.mcps:
            table.add_row(
                m.get("name", "?"),
                m.get("url", ""),
                "on" if m.get("enabled", True) else "off",
                ", ".join(m.get("headers", {}).keys()) or "—",
            )
        console.print(table)
        console.print()
        console.print(
            "[bold cyan]1[/] Añadir    [bold cyan]2[/] Eliminar    "
            "[bold cyan]3[/] Activar/Desactivar    [bold cyan]4[/] Probar    "
            "[bold cyan]5[/] Salir"
        )
        choice = Prompt.ask("Opción", choices=["1", "2", "3", "4", "5"], default="5")
        if choice == "1":
            wizard_mcp_add(cfg)
        elif choice == "2":
            name = Prompt.ask("Nombre del servidor a eliminar")
            if cfg.remove_mcp(name):
                success(f"'{name}' eliminado.")
            else:
                warn(f"'{name}' no existe.")
        elif choice == "3":
            name = Prompt.ask("Nombre del servidor")
            m = cfg.get_mcp(name)
            if m is None:
                warn(f"'{name}' no existe.")
                continue
            enabled = Confirm.ask("¿Activar?", default=not m.get("enabled", True))
            cfg.set_mcp_enabled(name, enabled)
            success(f"'{name}' {'activado' if enabled else 'desactivado'}.")
        elif choice == "4":
            asyncio.run(test_mcps(cfg, only=Prompt.ask("Servidor (Enter = todos)", default="")))
        else:
            return
        console.print()


def wizard_mcp_add(cfg: Config) -> None:
    name = Prompt.ask("Nombre del servidor").strip()
    url = Prompt.ask("URL del endpoint streamable-http").strip()
    if not name or not url:
        warn("Nombre y URL son obligatorios.")
        return
    headers: Dict[str, str] = {}
    token = getpass.getpass("Token Bearer (opcional, Enter para omitir): ").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    while True:
        extra = Prompt.ask("Cabecera adicional 'Clave: Valor' (Enter para terminar)", default="")
        if not extra.strip():
            break
        if ":" in extra:
            k, v = extra.split(":", 1)
            headers[k.strip()] = v.strip()
        else:
            warn("Formato inválido, usa 'Clave: Valor'.")
    cfg.add_mcp(name, url, headers)
    success(f"Servidor MCP '{name}' añadido.")


# --------------------------------------------------------------------- tests
def test_llm(cfg: Config) -> bool:
    if not cfg.is_configured():
        error("LLM no configurado. Ejecuta: giar config llm")
        return False
    console.print(f"Probando LLM en [bold]{cfg.base_url}[/] (modelo [bold]{cfg.model}[/])…")

    async def _probe() -> str:
        client = LLMClient(
            cfg.base_url,
            cfg.api_key,
            cfg.model,
            cfg.extra_headers,
            reasoning_effort=cfg.reasoning_effort or None,
        )
        try:
            parts = []
            async for event in client.stream_chat(
                [{"role": "user", "content": "Responde únicamente con: OK"}]
            ):
                if event["type"] == "text":
                    parts.append(event["content"])
            return "".join(parts).strip()
        finally:
            await client.close()

    try:
        reply = asyncio.run(_probe())
    except LLMError as exc:
        error(f"Conexión fallida: {exc}")
        return False
    success(f"LLM responde: {reply[:80] or '(vacío)'}")
    return True


async def _test_one_mcp(entry: Dict[str, Any]) -> bool:
    name = entry.get("name", "?")
    url = entry.get("url", "")
    console.print(f"Probando MCP [bold]{name}[/] en {url}…")
    client = MCPClient(name=name, url=url, headers=entry.get("headers") or {})
    try:
        await client.initialize()
        tools = await client.list_tools()
    except MCPError as exc:
        error(f"{name}: {exc}")
        await client.close()
        return False
    n = len(tools)
    names = ", ".join(t.get("name", "?") for t in tools[:6])
    if n > 6:
        names += ", …"
    success(f"{name}: conectado, {n} herramientas ({names})")
    await client.close()
    return True


async def test_mcps(cfg: Config, only: str = "") -> bool:
    entries = cfg.mcps if not only else [m for m in cfg.mcps if m.get("name") == only]
    if not entries:
        warn("No hay servidores MCP configurados.")
        return False
    results = []
    for entry in entries:
        results.append(await _test_one_mcp(entry))
    return all(results)


# --------------------------------------------------------------- subcommands
def cmd_chat(args: argparse.Namespace) -> None:
    cfg = Config.load()
    args_cwd = getattr(args, "cwd", None)
    args_model = getattr(args, "model", None)
    args_prompt = getattr(args, "prompt", None)
    args_mcp_off = getattr(args, "mcp_off", False)
    args_no_reasoning = getattr(args, "no_reasoning", False)

    cwd = Path(args_cwd).resolve() if args_cwd else Path.cwd()

    if not cfg.is_configured():
        warn("GIAR aún no está configurado.")
        wizard_llm(cfg)
        if not cfg.is_configured():
            error("Configuración incompleta; no se puede iniciar el chat.")
            sys.exit(1)

    # En el modo interactivo el logo ya está fijo en el header del TUI.
    if args_prompt:
        banner()

    if args_model:
        cfg.set_provider(cfg.base_url, args_model, cfg.api_key)

    from giar.chat import ChatSession

    session = ChatSession(
        config=cfg,
        cwd=cwd,
        model=cfg.model,
        connect_mcps=not args_mcp_off,
        show_reasoning=None if not args_no_reasoning else False,
        tui=not bool(args_prompt),
    )

    async def _run() -> None:
        await session.start()
        try:
            if args_prompt:
                try:
                    await session.run_turn(args_prompt)
                except LLMError as exc:
                    error(str(exc))
            else:
                await session.run()
        finally:
            await session.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print()


def cmd_config(args: argparse.Namespace) -> None:
    cfg = Config.load()
    if args.what == "llm":
        wizard_llm(cfg)
    elif args.what == "mcp":
        wizard_mcp(cfg)
    elif args.what == "show":
        console.print(json.dumps(cfg.to_redacted_dict(), indent=2, ensure_ascii=False))
    else:
        hline("Configuración de GIAR")
        table = Table(box=None)
        table.add_column("Sección", style="bold cyan")
        table.add_column("Descripción")
        table.add_row("llm", "Endpoint OpenAI-compatible + API key + modelo")
        table.add_row("mcp", "Servidores MCP streamable-http")
        table.add_row("show", "Mostrar configuración actual")
        console.print(table)
        console.print()
        choice = Prompt.ask(
            "¿Qué quieres configurar?", choices=["llm", "mcp", "show"], default="llm"
        )
        if choice == "llm":
            wizard_llm(cfg)
        elif choice == "mcp":
            wizard_mcp(cfg)
        else:
            console.print(json.dumps(cfg.to_redacted_dict(), indent=2, ensure_ascii=False))


def cmd_mcp(args: argparse.Namespace) -> None:
    """Gestionar servidores MCP: add <nombre> <url>, list, remove, toggle, test."""
    cfg = Config.load()
    action = args.mcp_action or "list"

    if action == "add":
        headers: Dict[str, str] = {}
        if args.token:
            headers["Authorization"] = f"Bearer {args.token}"
        for h in args.header:
            if ":" in h:
                k, v = h.split(":", 1)
                headers[k.strip()] = v.strip()
        cfg.add_mcp(args.name, args.url, headers)
        success(f"Servidor MCP '{args.name}' añadido.")
    elif action == "list":
        if not cfg.mcps:
            info("No hay servidores MCP configurados. Usa: giar mcp add <nombre> <url>")
            return
        table = Table(title="Servidores MCP")
        table.add_column("Nombre", style="bold cyan")
        table.add_column("URL")
        table.add_column("Estado")
        for m in cfg.mcps:
            table.add_row(
                m.get("name", "?"),
                m.get("url", ""),
                "on" if m.get("enabled", True) else "off",
            )
        console.print(table)
    elif action == "remove":
        if cfg.remove_mcp(args.name):
            success(f"Servidor '{args.name}' eliminado.")
        else:
            warn(f"Servidor '{args.name}' no existe.")
    elif action == "toggle":
        if cfg.set_mcp_enabled(args.name, args.state == "on"):
            success(f"Servidor '{args.name}' {'activado' if args.state == 'on' else 'desactivado'}.")
        else:
            warn(f"Servidor '{args.name}' no existe.")
    elif action == "test":
        asyncio.run(test_mcps(cfg))
    else:
        console.print(cmd_mcp.__doc__ or "")


def cmd_skills(args: argparse.Namespace) -> None:
    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    banner(cmd=True)
    skills = discover_skills(cwd, get_home())
    console.print(f"[bold]Skills detectados en {cwd}:[/]")
    if not skills:
        console.print("[dim]  (ninguno)[/]")
    for s in skills:
        console.print(f"  [bold cyan]{s.name}[/]  [dim]{s.source}[/]")
        if s.description:
            console.print(f"      {s.description}")
        if s.files:
            console.print(f"      [dim]archivos: {', '.join(f.name for f in s.files)}[/]")
    agents = find_agents_md(cwd, Path.home())
    if agents:
        console.print(f"\n[bold]Contexto de proyecto:[/] {agents}")
    else:
        console.print("\n[dim]No se encontró AGENTS.md en el proyecto.[/]")


def cmd_doctor(args: argparse.Namespace) -> None:
    cfg = Config.load()
    banner(cmd=True)
    hline("Diagnóstico de GIAR")

    table = Table(box=None)
    table.add_column("Sección", style="bold cyan")
    table.add_column("Valor")
    table.add_row("Config", str(get_config_path()))
    table.add_row("Base URL", cfg.base_url or "—")
    table.add_row("Modelo", cfg.model or "—")
    table.add_row("Reasoning effort", cfg.reasoning_effort or "—")
    table.add_row("API key", "✔ configurada" if cfg.api_key else "✖ ausente")
    table.add_row("MCPs", f"{len(cfg.mcps)} configurados / {len(cfg.enabled_mcps())} activos")
    console.print(table)
    console.print()

    ok = True
    if not args.mcp:
        ok = test_llm(cfg) and ok
    if not args.llm:
        ok = asyncio.run(test_mcps(cfg)) and ok

    hline()
    if ok:
        success("Todo parece correcto. ¡Disfruta de GIAR!")
    else:
        error("Hay problemas que revisar (ver arriba).")


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "chat"):
        cmd_chat(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "mcp":
        cmd_mcp(args)
    elif args.command == "skills":
        cmd_skills(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "version":
        console.print(f"giar {__version__}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
