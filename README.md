# GIAR

Asistente de IA por consola, estilo **Claude Code**, que trabaja con **cualquier
endpoint OpenAI-compatible** y se conecta a **servidores MCP** (transporte
**streamable-http**) para que cualquier usuario común pueda chatear con sus
herramientas directamente desde la terminal.

```
██████╗  ██╗ █████╗ ██████╗ 
██╔════╝ ██║██╔══██╗██╔══██╗
██║  ███╗██║███████║██████╔╝
██║   ██║██║██╔══██║██╔══██╗
╚██████╔╝██║██║  ██║██║  ██║
 ╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
```

## Características

- **LLM OpenAI-compatible**: apunta a cualquier endpoint que implemente
  `/v1/chat/completions` (OpenAI, Groq, OpenRouter, Ollama, LM Studio, vLLM…).
- **API key** configurable (también vía `GIAR_API_KEY` o `OPENAI_API_KEY`).
- **MCP streamable-http**: conecta servidores MCP y expone sus herramientas al
  LLM; tú chateas y GIAR llama a las herramientas por ti.
- **Skills**: detecta automáticamente `SKILL.md` en el proyecto (`.giar/skills`,
  `.claude/skills`, `skills/` o `~/.giar/skills`) y respeta `AGENTS.md`.
- **Respuestas en markdown**: las respuestas se renderizan en la terminal
  (negritas, listas, bloques de código…).
- **Modelos razonadores**: soporta `reasoning_effort` y muestra el pensamiento
  (`reasoning_content`) en un bloque atenuado, con opción de verlo u ocultarlo.
- Interfaz interactiva estilo Claude Code: streaming, tool calls visibles,
  historial, comandos `/`, `Ctrl+L` y `/clear` vuelven a mostrar el logo.

## Instalación

```bash
pip install .
```

o desde el código:

```bash
pip install -e .
```

Requiere Python 3.10+.

## Primeros pasos

```bash
giar          # primer arranque: te guía por la configuración
```

1. **Configura el LLM**: `giar config llm`
   - Base URL (ej. `https://api.openai.com/v1`, `http://localhost:11434/v1` para Ollama)
   - Modelo (ej. `gpt-4o-mini`, `qwen2.5:7b`)
   - API key (oculta)
   - Reasoning effort (opcional, para modelos de razonamiento: `low` | `medium` | `high`)
2. **Añade un servidor MCP**: `giar config mcp` (o `giar mcp add <nombre> <url>`)
3. **¡Chatea!**: `giar`

```bash
giar                                  # chat interactivo
giar -p "resume el README del proyecto"   # pregunta única
giar doctor                           # comprueba LLM y MCPs
```

## Comandos de la CLI

| Comando | Descripción |
|---|---|
| `giar` / `giar chat` | Sesión de chat interactiva |
| `giar chat -p "pregunta"` | Pregunta única (no interactivo) |
| `giar chat --no-reasoning` | Ocultar el pensamiento del modelo |
| `giar config llm` | Configurar endpoint LLM + API key + modelo + reasoning effort |
| `giar config mcp` | Menú para gestionar servidores MCP |
| `giar config show` | Mostrar configuración (sin api keys) |
| `giar mcp add <nombre> <url> [--token X] [--header "K: V"]` | Añadir MCP |
| `giar mcp list` / `remove` / `toggle` / `test` | Gestionar MCPs |
| `giar skills` | Listar skills detectados en el proyecto |
| `giar doctor` | Diagnóstico completo (LLM + MCPs) |

### MCP con autenticación

```bash
giar mcp add mi-servidor https://ejemplo.com/mcp --token "sk-..."
giar mcp add otro http://localhost:3000/mcp --header "X-API-Key: abc"
```

## Comandos dentro del chat

```
/help             Ayuda
/model <name>     Cambiar modelo
/effort <nivel>   Reasoning effort: low | medium | high | off
/reasoning on|off Mostrar/ocultar el pensamiento del modelo
/clear            Reinicia la conversación y vuelve a mostrar el logo
/skills           Ver skills detectados
/tools            Ver herramientas disponibles
/mcp              Estado de servidores MCP
/config           Dónde y cómo configurar
/exit             Salir (o Ctrl+D)
```

`Ctrl+L` limpia la pantalla y vuelve a mostrar el banner de GIAR (igual que
`/clear`).

## Modelos razonadores

- **`reasoning_effort`** (estándar OpenAI: `low` | `medium` | `high`): controla
  cuánto "piensa" el modelo antes de responder. Se configura en
  `giar config llm` o en caliente con `/effort`. Solo se envía al endpoint
  cuando está definido, así que es seguro con cualquier modelo.
- **Pensamiento en streaming**: si el modelo incluye su razonamiento en el
  streaming (`reasoning_content`/`reasoning`), GIAR lo muestra en un bloque
  atenuado "🤔 Razonamiento" encima de la respuesta. Ocúltalo con
  `/reasoning off` (o `giar chat --no-reasoning`); la preferencia se guarda en
  la configuración y se cambia con `/reasoning on|off`.

## Skills

GIAR busca skills (convención Claude Code) al abrirse en un proyecto:

```
proyecto/
├── AGENTS.md                       # contexto de proyecto
├── .giar/skills/
│   └── mi-skill/
│       ├── SKILL.md                # frontmatter YAML + instrucciones
│       └── otros-archivos...
├── .claude/skills/…                # también se soporta
└── skills/…                        # y esto
```

Formato de `SKILL.md`:

```markdown
---
name: mi-skill
description: Qué hace este skill.
---
Instrucciones detalladas para el agente…
```

El LLM ve la lista de skills disponibles y usa la herramienta `read_skill`
para cargar las instrucciones cuando hace falta.

## Configuración

Todo se guarda en `~/.giar/config.json` (o `$GIAR_HOME`):

```json
{
  "provider": {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "api_key": "sk-...",
    "headers": {},
    "reasoning_effort": "medium",
    "show_reasoning": true
  },
  "mcps": [
    { "name": "mi-servidor", "url": "https://…/mcp", "headers": {}, "enabled": true }
  ],
  "skills": { "disabled": [] }
}
```

> El guardado es **atómico** (temp + `os.replace()`) y con permisos `0600`.
> Si `config.json` se encuentra corrupto o ilegible, GIAR lo respalda como
> `config.json.corrupt-<fecha>` (avisa por stderr) y usa valores por defecto:
> nunca se sobrescribe en silencio ni se pierde la configuración anterior.

## Arquitectura

```
giar/
├── cli.py      # comandos, asistentes de configuración y doctor
├── chat.py     # sesión interactiva: markdown, razonamiento, tool calls, slash commands
├── config.py   # configuración persistente (guardado atómico + respaldo)
├── llm.py      # cliente OpenAI-compatible (streaming, tool calls, reasoning)
├── mcp.py      # cliente MCP streamable-http (JSON-RPC + SSE)
├── skills.py   # detección de SKILL.md y AGENTS.md
├── tools.py    # registro de herramientas (builtins + MCP)
└── ui.py       # banner azul y helpers de interfaz
```
