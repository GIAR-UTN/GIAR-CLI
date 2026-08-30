"""Pruebas unitarias de GIAR (sin red). Ejecutar: python -m unittest discover tests"""

import asyncio
import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

from giar.mcp import _format_error, _parse_sse, _sanitize
from giar.llm import LLMClient, LLMError, _normalize_effort
from giar.skills import (
    _split_frontmatter,
    discover_skills,
    find_agents_md,
    parse_skill_dir,
)
from giar.tools import _project_tool_handler, arguments_to_kwargs
from giar.config import Config, get_config_path
from giar.latex import prepare_markdown
from giar.chat import (
    ChatSession,
    _is_degenerate,
    _rich_style_to_pt,
    _Tui,
    rich_to_fragments,
)


class TestSSE(unittest.TestCase):
    def test_single_json(self):
        body = '{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}'
        msg = _parse_sse(body, 1)
        self.assertIsNotNone(msg)
        self.assertIn("result", msg)

    def test_sse_stream(self):
        body = (
            'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
            'event: message\ndata: {"jsonrpc":"2.0","id":2,"result":{"ok":false}}\n\n'
        )
        msg = _parse_sse(body, 2)
        self.assertEqual(msg["result"]["ok"], False)

    def test_no_match_returns_result_message(self):
        body = 'event: message\ndata: {"jsonrpc":"2.0","result":{"ok":true}}\n\n'
        msg = _parse_sse(body, 999)
        self.assertIsNotNone(msg)

    def test_empty(self):
        self.assertIsNone(_parse_sse("", 1))

    def test_multiline_data(self):
        body = 'event: message\ndata: {"a":1}\ndata: ,"b":2}\n\n'
        msg = _parse_sse(body, 1)
        self.assertIsNone(msg)


class TestSanitize(unittest.TestCase):
    def test_sanitize(self):
        self.assertEqual(_sanitize("add numbers"), "add_numbers")
        self.assertEqual(_sanitize("a.b/c"), "a_b_c")


class TestFrontmatter(unittest.TestCase):
    def test_split(self):
        text = "---\nname: x\ndescription: y\n---\n# Cuerpo\n"
        meta, body = _split_frontmatter(text)
        self.assertIn("name: x", meta)
        self.assertEqual(body, "# Cuerpo\n")

    def test_no_frontmatter(self):
        meta, body = _split_frontmatter("# solo cuerpo")
        self.assertEqual(meta, "")
        self.assertEqual(body, "# solo cuerpo")


class TestSkills(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_skill(self):
        d = self.root / ".giar" / "skills" / "mi-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: mi-skill\ndescription: hace cosas\n---\nInstrucciones."
        )
        (d / "extra.py").write_text("x = 1")
        skill = parse_skill_dir(d)
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "mi-skill")
        self.assertEqual(skill.description, "hace cosas")
        self.assertEqual(skill.content, "Instrucciones.")
        self.assertEqual([f.name for f in skill.files], ["extra.py"])

    def test_parse_skill_fallback_name(self):
        d = self.root / "skills" / "otro"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\ndescription: sin nombre\n---\nCuerpo")
        skill = parse_skill_dir(d)
        self.assertEqual(skill.name, "otro")

    def test_discover_locations(self):
        for sub in (".giar", ".claude", "",):
            d = self.root / sub / "skills" / f"s-{sub or 'root'}"
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: s-{sub or 'root'}\n---\nCuerpo")
        user = self.root / "home" / "skills" / "u1"
        user.mkdir(parents=True)
        (user / "SKILL.md").write_text("---\nname: u1\n---\nCuerpo")
        found = discover_skills(self.root, self.root / "home")
        names = {s.name for s in found}
        self.assertEqual(names, {"s-.giar", "s-.claude", "s-root", "u1"})


class TestArgs(unittest.TestCase):
    def test_dict(self):
        self.assertEqual(arguments_to_kwargs({"a": 1}), {"a": 1})

    def test_json_string(self):
        self.assertEqual(arguments_to_kwargs('{"a": 1}'), {"a": 1})

    def test_invalid(self):
        self.assertEqual(arguments_to_kwargs("no-json"), {})


class TestReasoningEffort(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(_normalize_effort("High"), "high")
        self.assertEqual(_normalize_effort(" medium "), "medium")
        self.assertIsNone(_normalize_effort(""))
        self.assertIsNone(_normalize_effort(None))
        self.assertIsNone(_normalize_effort("off"))
        self.assertIsNone(_normalize_effort("none"))
        with self.assertRaises(LLMError):
            _normalize_effort("ultra")

    def test_payload_includes_effort(self):
        client = LLMClient("http://x/v1", model="o3", reasoning_effort="high")
        payload = client._payload([{"role": "user", "content": "hola"}])
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_payload_omits_effort_when_unset(self):
        client = LLMClient("http://x/v1", model="o3")
        payload = client._payload([{"role": "user", "content": "hola"}])
        self.assertNotIn("reasoning_effort", payload)

    def test_payload_per_call_override(self):
        client = LLMClient("http://x/v1", model="o3", reasoning_effort="low")
        payload = client._payload(
            [{"role": "user", "content": "hola"}], reasoning_effort="high"
        )
        self.assertEqual(payload["reasoning_effort"], "high")


class TestShowReasoning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(
            os.environ, {"GIAR_HOME": self.tmp.name}, clear=False
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_default_true(self):
        cfg = Config()
        self.assertTrue(cfg.show_reasoning)

    def test_toggle(self):
        cfg = Config()
        cfg.set_show_reasoning(False)
        self.assertFalse(cfg.show_reasoning)
        cfg.set_show_reasoning(True)
        self.assertTrue(cfg.show_reasoning)


class TestMaxTurns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(
            os.environ, {"GIAR_HOME": self.tmp.name}, clear=False
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_default(self):
        cfg = Config()
        self.assertEqual(cfg.max_turns, 200)

    def test_set_and_persist(self):
        cfg = Config()
        cfg.set_max_turns(500)
        loaded = Config.load()
        self.assertEqual(loaded.max_turns, 500)

    def test_min_clamped(self):
        cfg = Config()
        cfg.set_max_turns(0)
        self.assertEqual(Config.load().max_turns, 1)

    def test_invalid_value_uses_default(self):
        cfg = Config({"chat": {"max_turns": "abc"}})
        self.assertEqual(cfg.max_turns, 200)

    def test_session_uses_config(self):
        cfg = Config()
        cfg.set_max_turns(7)
        self.assertEqual(ChatSession(cfg).max_turns, 7)

    def test_turns_command_sets_and_persists(self):
        cfg = Config()
        session = ChatSession(cfg)
        asyncio.run(session.handle_command("/turns 42"))
        self.assertEqual(session.max_turns, 42)
        self.assertEqual(Config.load().max_turns, 42)

    def test_turns_command_rejects_bad_value(self):
        cfg = Config()
        session = ChatSession(cfg)
        asyncio.run(session.handle_command("/turns abc"))
        self.assertEqual(session.max_turns, cfg.max_turns)

    def test_temperature_command_sets_and_persists(self):
        cfg = Config()
        session = ChatSession(cfg)
        asyncio.run(session.handle_command("/temperature 0.7"))
        self.assertEqual(session.temperature, 0.7)
        self.assertEqual(Config.load().temperature, 0.7)

    def test_temperature_command_off(self):
        cfg = Config()
        cfg.set_temperature(0.7)
        session = ChatSession(cfg)
        asyncio.run(session.handle_command("/temperature off"))
        self.assertIsNone(session.temperature)
        self.assertIsNone(Config.load().temperature)

    def test_temperature_command_rejects_bad_value(self):
        cfg = Config()
        session = ChatSession(cfg)
        asyncio.run(session.handle_command("/temperature abc"))
        self.assertEqual(session.temperature, cfg.temperature)


class TestConfigPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(
            os.environ, {"GIAR_HOME": self.tmp.name}, clear=False
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_roundtrip(self):
        cfg = Config()
        cfg.set_provider("https://x/v1", "m1", api_key="sk-123",
                         reasoning_effort="high")
        cfg.add_mcp("srv", "http://localhost:1/mcp")
        loaded = Config.load()
        self.assertEqual(loaded.base_url, "https://x/v1")
        self.assertEqual(loaded.model, "m1")
        self.assertEqual(loaded.api_key, "sk-123")
        self.assertEqual(loaded.reasoning_effort, "high")
        self.assertEqual(len(loaded.mcps), 1)
        self.assertEqual(loaded.mcps[0]["name"], "srv")

    def test_corrupt_file_backed_up_not_lost(self):
        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{esto no es json", encoding="utf-8")
        cfg = Config.load()
        self.assertFalse(cfg.is_configured())
        backups = list(path.parent.glob("config.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertFalse(path.exists())

    def test_empty_file_returns_defaults(self):
        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        cfg = Config.load()
        self.assertFalse(cfg.is_configured())
        backups = list(path.parent.glob("config.json.corrupt-*"))
        self.assertEqual(len(backups), 1)

    def test_valid_json_but_not_dict_is_backed_up(self):
        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('["esto", "no", "es", "un", "dict"]', encoding="utf-8")
        cfg = Config.load()
        self.assertFalse(cfg.is_configured())
        backups = list(path.parent.glob("config.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertFalse(path.exists())
        self.assertIn("esto", backups[0].read_text(encoding="utf-8"))

    def test_atomic_save_leaves_no_temp(self):
        cfg = Config()
        cfg.set_provider("https://x/v1", "m1")
        leftovers = list(get_config_path().parent.glob("config.json.tmp"))
        self.assertEqual(leftovers, [])
        self.assertTrue(get_config_path().exists())


class TestDegenerate(unittest.TestCase):
    def test_ellipsis_chain(self):
        self.assertTrue(_is_degenerate("Se ha ... ... ... ... ... ... ..."))

    def test_repetitive_words(self):
        self.assertTrue(_is_degenerate("adelante adelante adelante adelante adelante"))

    def test_normal_text(self):
        self.assertFalse(
            _is_degenerate("El resultado es correcto y se guardó en el fichero.")
        )

    def test_short_text_not_flagged(self):
        self.assertFalse(_is_degenerate("..." ))


class TestLatex(unittest.TestCase):
    def test_inline_math(self):
        self.assertEqual(
            prepare_markdown("$x_i + y_j = z^2$"),
            "xᵢ + yⱼ = z²",
        )

    def test_display_math(self):
        self.assertEqual(
            prepare_markdown("$$E = mc^2$$"),
            "E = mc²",
        )

    def test_nested_frac(self):
        self.assertEqual(
            prepare_markdown(r"$$\frac{\frac{1}{2}+\frac{3}{4}}{x}$$"),
            "((1)/(2)+(3)/(4))/(x)",
        )

    def test_sqrt_and_frac(self):
        self.assertEqual(
            prepare_markdown(r"$$\frac{a}{b} = \sqrt{c}$$"),
            "(a)/(b) = √(c)",
        )

    def test_greek_and_symbols(self):
        self.assertEqual(
            prepare_markdown(r"$\alpha + \beta \times \gamma \geq 0$"),
            "α + β × γ ≥ 0",
        )

    def test_superscript_subscript_unicode(self):
        self.assertEqual(prepare_markdown(r"$\lambda_i = \sum_{j=1}^{n} x_{ij}$"),
                         "λᵢ = ∑ⱼ₌₁ⁿ xᵢⱼ")

    def test_paren_and_bracket_forms(self):
        self.assertEqual(
            prepare_markdown(r"\( a^2 + b^2 = c^2 \)"),
            "a² + b² = c²",
        )

    def test_text_command(self):
        self.assertEqual(
            prepare_markdown(r"$$\text{hola } x \geq 0$$"),
            "hola x ≥ 0",
        )

    def test_escapes_markdown_sensitive_chars(self):
        self.assertEqual(
            prepare_markdown(r"$\lim_{x \to 0} f(x)$"),
            r"lim\_(x → 0) f(x)",
        )

    def test_code_spans_untouched(self):
        self.assertEqual(
            prepare_markdown("`echo $HOME` y `x=$((a+b))`"),
            "`echo $HOME` y `x=$((a+b))`",
        )

    def test_code_blocks_untouched(self):
        src = "```bash\nrm -rf $HOME/*\n```"
        self.assertEqual(prepare_markdown(src), src)

    def test_prices_untouched(self):
        self.assertEqual(
            prepare_markdown("cuesta $5 y $10, o $5.99"),
            "cuesta $5 y $10, o $5.99",
        )

    def test_single_char_math(self):
        self.assertEqual(prepare_markdown("variable $x$"), "variable x")

    def test_math_inside_code_is_not_touched(self):
        src = "`$x_i$` se queda"
        self.assertEqual(prepare_markdown(src), src)


class TestPathRestriction(unittest.TestCase):
    """Bug 1: las rutas deben quedar dentro del proyecto (sin escape por prefijo)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "proj").mkdir()
        (self.root / "proj-evil").mkdir()
        (self.root / "proj-evil" / "secret.txt").write_text("DATOS SECRETOS")
        self.read_file, self.list_dir = _project_tool_handler(self.root / "proj")

    def tearDown(self):
        self.tmp.cleanup()

    def test_sibling_dir_blocked(self):
        with self.assertRaises(ValueError):
            asyncio.run(self.read_file("../proj-evil/secret.txt"))

    def test_absolute_outside_blocked(self):
        with self.assertRaises(ValueError):
            asyncio.run(self.read_file(str(self.root / "proj-evil" / "secret.txt")))

    def test_list_dir_outside_blocked(self):
        with self.assertRaises(ValueError):
            asyncio.run(self.list_dir("../proj-evil"))

    def test_inside_project_allowed(self):
        (self.root / "proj" / "a.txt").write_text("hola")
        self.assertEqual(asyncio.run(self.read_file("a.txt")), "hola")

    def test_absolute_inside_allowed(self):
        (self.root / "proj" / "b.txt").write_text("x")
        p = str(self.root / "proj" / "b.txt")
        self.assertEqual(asyncio.run(self.read_file(p)), "x")


class TestMCPFormatError(unittest.TestCase):
    """Bug 5: errores MCP en forma de string no deben romper el cliente."""

    def test_string_error(self):
        self.assertEqual(_format_error("boom"), "[error MCP] boom")

    def test_dict_error(self):
        self.assertEqual(
            _format_error({"code": -32601, "message": "método no encontrado"}),
            "[error MCP] -32601 método no encontrado",
        )


class TestAgentsStop(unittest.TestCase):
    """Bug 6: la búsqueda de AGENTS.md debe parar en el home del usuario."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.home = self.base / "home"
        (self.home / "proj").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ignores_agents_above_home(self):
        (self.base / "AGENTS.md").write_text("contexto raíz")
        self.assertIsNone(find_agents_md(self.home / "proj", self.home))

    def test_finds_within_home(self):
        (self.home / "AGENTS.md").write_text("contexto home")
        self.assertEqual(
            find_agents_md(self.home / "proj", self.home), self.home / "AGENTS.md"
        )

    def test_project_takes_precedence(self):
        (self.home / "proj" / "AGENTS.md").write_text("contexto proyecto")
        (self.home / "AGENTS.md").write_text("contexto home")
        self.assertEqual(
            find_agents_md(self.home / "proj", self.home),
            self.home / "proj" / "AGENTS.md",
        )


class TestRunTurnMessages(unittest.TestCase):
    """Bug 2: content + tool_calls no debe duplicar el mensaje assistant."""

    def test_content_and_calls_single_assistant_message(self):
        session = ChatSession(Config())
        session.messages = [{"role": "system", "content": "x"}]
        n_calls = 0

        async def fake_stream(*args, **kwargs):
            nonlocal n_calls
            n_calls += 1
            if n_calls == 1:
                return {
                    "content": "texto",
                    "tool_calls": [
                        {"id": "1", "name": "read_file", "arguments": "{}"}
                    ],
                    "degenerate": False,
                }
            return {
                "content": "respuesta final",
                "tool_calls": [],
                "degenerate": False,
            }

        async def fake_exec(call):
            return "ok"

        session.stream_assistant = fake_stream
        session.execute_tool_call = fake_exec
        asyncio.run(session.run_turn("hola"))

        roles = [m["role"] for m in session.messages]
        for a, b in zip(roles, roles[1:]):
            self.assertFalse(a == "assistant" and b == "assistant")
        tc_msgs = [m for m in session.messages if m.get("tool_calls")]
        self.assertEqual(len(tc_msgs), 1)
        self.assertEqual(tc_msgs[0]["content"], "texto")
        self.assertEqual(
            tc_msgs[0]["tool_calls"][0]["function"]["name"], "read_file"
        )

    def test_content_only_appends_once(self):
        session = ChatSession(Config())
        session.messages = [{"role": "system", "content": "x"}]

        async def fake_stream(*args, **kwargs):
            return {
                "content": "solo texto",
                "tool_calls": [],
                "degenerate": False,
            }

        session.stream_assistant = fake_stream
        asyncio.run(session.run_turn("hola"))
        roles = [m["role"] for m in session.messages]
        self.assertEqual(roles.count("assistant"), 1)


class TestShowReasoningInit(unittest.TestCase):
    """Bug 4: show_reasoning=None debe respetar la config persistida."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(
            os.environ, {"GIAR_HOME": self.tmp.name}, clear=False
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_none_uses_config(self):
        cfg = Config()
        cfg.set_show_reasoning(False)
        self.assertFalse(ChatSession(cfg, show_reasoning=None).show_reasoning)

    def test_false_forces_hidden(self):
        cfg = Config()
        cfg.set_show_reasoning(True)
        self.assertFalse(ChatSession(cfg, show_reasoning=False).show_reasoning)


class TestUserSkillsAtGiarHome(unittest.TestCase):
    """Bug 3: discover_skills(cwd, get_home()) encuentra ~/.giar/skills."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.giar_home = self.root / ".giar"
        d = self.giar_home / "skills" / "global"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: global\n---\nCuerpo")

    def tearDown(self):
        self.tmp.cleanup()

    def test_skill_in_giar_home_found(self):
        found = discover_skills(self.root / "proyecto", self.giar_home)
        self.assertIn("global", {s.name for s in found})


class TestRichToFragments(unittest.TestCase):
    """Conversión de renderables rich a fragmentos de prompt_toolkit."""

    def test_markup_conserva_color(self):
        from rich.text import Text

        frags = rich_to_fragments(Text.from_markup("[bold yellow]⚠[/] aviso"), 80)
        styled = [s for s, _ in frags if "⚠" in _]
        self.assertTrue(any("bold" in s and "yellow" in s for s in styled))

    def test_panel_borde_azul(self):
        from rich.panel import Panel

        frags = rich_to_fragments(Panel("hola", border_style="blue"), 80)
        borders = [s for s, t in frags if t.startswith("╭")]
        self.assertTrue(all("blue" in s for s in borders))

    def test_markdown_convierte_texto(self):
        from rich.markdown import Markdown

        frags = rich_to_fragments(Markdown("## Título\n\n**negrita** y `código`"), 80)
        text = "".join(t for _, t in frags)
        self.assertIn("Título", text)
        self.assertIn("código", text)

    def test_estilo_vacio_para_texto_plano(self):
        from rich.text import Text

        frags = rich_to_fragments(Text("plano"), 80)
        contenido = [(s, t) for s, t in frags if t.strip()]
        self.assertEqual(contenido, [("", "plano")])


class TestTuiHelpers(unittest.TestCase):
    """Comportamiento de la pantalla completa sin necesidad de una terminal."""

    def test_style_to_pt_truecolor(self):
        from rich.style import Style

        self.assertEqual(
            _rich_style_to_pt(Style(color="#4FC3F7")), "fg:#4fc3f7"
        )

    def test_style_to_pt_ansi(self):
        from rich.style import Style

        self.assertEqual(_rich_style_to_pt(Style(color="blue")), "fg:ansiblue")
        self.assertEqual(
            _rich_style_to_pt(Style(color="bright_red")), "fg:ansibrightred"
        )

    def test_emit_acumula_bloques(self):
        from rich.text import Text

        session = ChatSession(Config())
        session.tui = _Tui(session)
        tui = session.tui
        tui.emit(Text("primero"))
        tui.emit(Text("segundo"))
        text = "".join(t for _, t in tui.frags)
        self.assertIn("primero", text)
        self.assertIn("segundo", text)
        self.assertIn("\n", text)

    def test_set_live_y_clear(self):
        from rich.text import Text

        session = ChatSession(Config())
        tui = _Tui(session)
        tui.emit(Text("base"))
        tui.set_live(Text("en vivo"))
        text = "".join(t for _, t in tui.frags + tui.live_frags)
        self.assertIn("en vivo", text)
        tui.set_live(None)
        self.assertEqual(tui.live_frags, [])

    def test_visible_height_por_defecto(self):
        session = ChatSession(Config())
        tui = _Tui(session)
        # Sin aplicación en marcha cae al valor por defecto.
        self.assertGreaterEqual(tui._visible_height(), 10)


class TestTurnEmiteMensajeUsuario(unittest.TestCase):
    """En modo TUI el mensaje del usuario se muestra en la conversación."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(
            os.environ, {"GIAR_HOME": self.tmp.name}, clear=False
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_panel_tu_en_tui(self):
        session = ChatSession(Config(), tui=True)
        session.messages = [{"role": "system", "content": "x"}]

        async def fake_stream(*args, **kwargs):
            return {"content": "respuesta", "tool_calls": [], "degenerate": False}

        session.stream_assistant = fake_stream
        asyncio.run(session.run_turn("mi pregunta"))

        text = "".join(t for _, t in session.tui.frags)
        self.assertIn("mi pregunta", text)
        roles = [m["role"] for m in session.messages]
        self.assertEqual(roles.count("user"), 1)


if __name__ == "__main__":
    unittest.main()
