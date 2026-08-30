# AGENTS.md

## Qué es esto

`giar-cli` (paquete `giar`) es un asistente de IA por consola estilo Claude
Code: llama a **cualquier endpoint OpenAI-compatible** (`/chat/completions`),
se conecta a **servidores MCP** (transporte streamable-http, JSON-RPC 2.0) y
detecta **skills** (convención `SKILL.md`). Toda la interfaz, la doc y los
mensajes son en **español**.

## Convenciones

- **Idioma**: docstrings, comentarios, mensajes de UI y tests en español.
  Mantener el mismo estilo al añadir código.
- **Python 3.10+** con `from __future__ import annotations`. El paquete es
  *typed* (incluye `py.typed`).
- No hay linter/typechecker configurado (ni ruff ni mypy). No introducir una
  configuración nueva sin motivo.
- Dependencias: `rich`, `httpx`, `prompt_toolkit`, `pyyaml` (ver `pyproject.toml`).
- Todo el I/O de red es **async** (`httpx.AsyncClient`); la CLI arranca con
  `asyncio.run()`.

## Comandos

- Tests: `python3 -m unittest discover tests` (usa **unittest**, no pytest).
  Los tests son sin red; corren con `GIAR_HOME` apuntando a un dir temporal.
- Instalación en editable: `pip install -e .` (script `giar` → `giar.cli:main`).
- Para probar a mano: `giar doctor`, `giar skills`, `giar config show`.

## Arquitectura (giar/)

- `cli.py` — parser argparse, subcomandos y asistentes de configuración
  (`config llm|mcp`, `mcp add/list/...`, `doctor`). Punto de entrada `main()`.
  En `chat` el banner solo se imprime en modo `-p` (en la TUI el logo ya está
  fijo en el header); se activa con `ChatSession(..., tui=not bool(args_prompt))`.
- `chat.py` — sesión interactiva (`ChatSession`). En modo interactivo arranca
  una **TUI de pantalla completa** (`_Tui`, prompt_toolkit `Application`) con el
  logo de GIAR **fijo arriba** (`HEADER_HEIGHT = 8`), la conversación en un
  `ScrollablePane` con scroll automático al final, y el input abajo (`TextArea`
  con historial). Salida enriquecida con `rich.live` en el fallback no-TUI.
  El enrutado de salida pasa por `_emit`/`_emit_markup`/`_emit_blank` (a la TUI
  o a la consola rich según el modo). Conversión rich→fragmentos de
  prompt_toolkit en `rich_to_fragments` (colores TRUECOLOR/EIGHT_BIT → `#hex`,
  STANDARD → `ansi<nombre>`; la app NO acepta `fg:ansi1` numerado).
  Los mensajes del usuario se muestran como panel `Tú` (borde verde) y las
  respuestas como panel `GIAR · modelo` (borde azul). Comandos `/...`, historial,
  y bucle de herramientas limitado a `max_turns` (configurable en `config.json`
  con `chat.max_turns` o en caliente con `/turns <n>`; por defecto 200).
- `config.py` — config persistente en `~/.giar/config.json` (o `$GIAR_HOME`).
  Guardado **atómico** (tmp + `os.replace`, permisos `0600`); si el fichero
  está corrupto se respalda como `config.json.corrupt-<fecha>` sin perderlo.
- `llm.py` — cliente OpenAI-compatible. `stream_chat()` emite eventos dict:
  `reasoning`, `text`, `tool_call` (con `index`), `finish`. Soporta
  `reasoning_effort` (solo se envía si está definido) y lee el pensamiento de
  `reasoning_content`/`reasoning`. `base_url` tolera acabar en `/chat/completions`.
- `mcp.py` — cliente MCP streamable-http: negocia `protocolVersion` probando
  `PROTOCOL_VERSIONS` en orden, maneja `Mcp-Session-Id`, respuestas JSON y SSE.
  El nombre público de cada herramienta es
  `mcp__<servidor_sanitizado>__<herramienta_sanitizada>`.
- `skills.py` — descubre skills en `.giar/skills`, `.claude/skills`, `skills/`
  y `~/.giar/skills` (frontmatter YAML `name`/`description` + cuerpo Markdown).
  `find_agents_md()` busca `AGENTS.md` subiendo directorios hasta el home.
- `tools.py` — registro de herramientas. Builtins: `read_file` y `list_dir`
  (rutas **restringidas al proyecto**: resuelven y validan dentro de `cwd`),
  `list_skills`, `read_skill`. Las herramientas MCP se envuelven con
  `wrap_mcp_tool()` y se registran con `source=f"mcp:{name}"`.
- `ui.py` — `rich.console` compartida, banner ASCII y helpers
  (`info`, `success`, `warn`, `error`, `hline`).

## Notas de comportamiento

- La sesión inyecta en el system prompt el contenido de `AGENTS.md` (contexto
  de proyecto), la lista de skills y los servidores MCP conectados.
- `Config.api_key` cae a `GIAR_API_KEY` / `OPENAI_API_KEY` si no hay clave en
  el fichero. Nunca imprimir la api key (usar `to_redacted_dict()`).
- Errores de red/LLM/MCP se propagan como `LLMError` / `MCPError`; el chat los
  muestra como avisos, no rompe la sesión.
- El chat detecta salidas degeneradas del modelo (cadenas de `...` o repetición
  excesiva) con `_is_degenerate()` y reintenta el turno hasta
  `MAX_DEGENERATE_RETRIES` veces sin guardar la respuesta basura en el historial.
- Atajos de la TUI: `Enter` envía, `Ctrl+C` interrumpe el turno en curso (o
  limpia el input si está ocioso), `Ctrl+D` sale, `Ctrl+L` limpia la pantalla y
  vuelve a mostrar el estado, `↑`/`↓` historial, `AvPág`/`RePág`/rueda scroll.
  `Ctrl+L` lo captura por defecto `_default_bindings` de prompt_toolkit
  (clear_screen, sin efecto visible); por eso se redefine en el control del
  input (`input_area.control.key_bindings`), que tiene prioridad sobre la app.
- El estado inicial (`_print_status`) se emite como un único bloque contiguo
  (sin líneas en blanco entre cada dato) con la línea de ayuda final en
  `bright_green`; es lo que se vuelve a mostrar tras `Ctrl+L`/`/clear`.
- El venv de desarrollo puede vivir dentro de `giar/` (ver `.gitignore`:
  `giar/bin/`, `giar/lib/`…); solo se trackean los fuentes `giar/*.py`.
- `README.md` documenta todos los comandos de la CLI y del chat; mantenerlo
  sincronizado al añadir comandos o flags.
