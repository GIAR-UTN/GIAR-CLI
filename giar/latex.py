r"""Conversión de sintaxis LaTeX a texto Unicode legible en la terminal.

`rich.markdown.Markdown` no entiende LaTeX: los delimitadores ``$`` se
imprimen tal cual, la barra de ``\alpha`` se interpreta como escape de
markdown (y se come la ``a``), y los guiones bajos de ``x_i`` se toman como
énfasis. Este módulo detecta las expresiones matemáticas del markdown
(``$...$``, ``$$...$$``, ``\\(...\\)``, ``\\[...\\]``) y las convierte a una
representación Unicode aproximada, escapando los caracteres que el parser de
markdown interpretaría (``_``, ``*``, ``~``, `` ` ``, ``[``, ``]``) para que
no queden cortadas.

Para no romper código con variables ``$...$`` ni precios, solo se convierte
una expresión inline si "parece LaTeX" (contiene comandos ``\``, ``^``, ``_``,
``{}``, operadores o dígitos). Los bloques de código y los spans de código se
protegen y no se tocan.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# ---------------------------------------------------------------- símbolos
GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο",
    "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ",
    "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω",
}

SYMBOLS = {
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠",
    "approx": "≈", "equiv": "≡", "propto": "∝", "sim": "∼",
    "ll": "≪", "gg": "≫", "in": "∈", "notin": "∉", "ni": "∋",
    "subset": "⊂", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅",
    "forall": "∀", "exists": "∃", "nexists": "∄", "neg": "¬",
    "land": "∧", "lor": "∨", "wedge": "∧", "vee": "∨",
    "oplus": "⊕", "otimes": "⊗", "star": "⋆", "ast": "∗",
    "to": "→", "rightarrow": "→", "leftarrow": "←",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "leftrightarrow": "↔",
    "Leftrightarrow": "⇔", "mapsto": "↦", "uparrow": "↑",
    "downarrow": "↓", "infty": "∞", "partial": "∂", "nabla": "∇",
    "sum": "∑", "prod": "∏", "int": "∫", "oint": "∮",
    "iint": "∬", "iiint": "∭",
    "lim": "lim", "min": "min", "max": "max", "sup": "sup", "inf": "inf",
    "log": "log", "ln": "ln", "exp": "exp", "sin": "sin", "cos": "cos",
    "tan": "tan", "cot": "cot", "sec": "sec", "csc": "csc",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "ldots": "…", "cdots": "⋯", "dots": "…", "vdots": "⋮", "ddots": "⋱",
    "perp": "⊥", "parallel": "∥", "mid": "|", "lvert": "|", "rvert": "|",
    "Vert": "‖", "angle": "∠", "triangle": "△", "circ": "∘",
    "degree": "°", "prime": "′", "hbar": "ℏ", "ell": "ℓ",
    "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋",
    "lceil": "⌈", "rceil": "⌉", "dagger": "†", "ddagger": "‡",
    "aleph": "ℵ", "Re": "ℜ", "Im": "ℑ", "checkmark": "✓", "surd": "√",
}

# Superíndices y subíndices Unicode (por carácter, para ^ y _ simples)
SUPERSCRIPT = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
    "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ",
    "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ",
    "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ",
    "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
    "α": "ᵅ", "β": "ᵝ", "γ": "ᵞ", "δ": "ᵟ", "φ": "ᵠ", "θ": "ᶿ",
}

SUBSCRIPT = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ",
}

# ---------------------------------------------------------------- regexs
_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_BRACKET = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_PAREN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_INLINE = re.compile(r"\$(?![ \t])(.+?)(?<![ \t])\$")

_NEWLINE = re.compile(r"\\\\")
_ENV = re.compile(r"\\begin\{[^{}]*\}|\\end\{[^{}]*\}")
_PMOD = re.compile(r"\\pmod\{([^{}]*)\}")
_TEXTLIKE = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|operatorname|mbox|textrm|textit|textbf|underline)\{([^{}]*)\}"
)
_LEFT = re.compile(r"\\left")
_RIGHT = re.compile(r"\\right")
_LIMITS = re.compile(r"\\limits|\\nolimits")
_BMOD = re.compile(r"\\bmod")
_SPACE = re.compile(r"\\[,;:!]|\\quad|\\qquad|\\enspace|\\ ")
_CMD = re.compile(r"\\([A-Za-z]+)")
_SUPB = re.compile(r"\^\{([^{}]*)\}")
_SUBB = re.compile(r"_\{([^{}]*)\}")
_SUP1 = re.compile(r"\^([A-Za-z0-9αβγδεφθ])")
_SUB1 = re.compile(r"_([A-Za-z0-9])")

_STRUCT = re.compile(r"\\(frac|dfrac|tfrac|binom|sqrt)")

# Acentos y vectores con marcas Unicode combinables
_ACCENT = [
    (re.compile(r"\\vec\{([^{}]*)\}"), "\u20D7"),
    (re.compile(r"\\hat\{([^{}]*)\}"), "\u0302"),
    (re.compile(r"\\bar\{([^{}]*)\}"), "\u0304"),
    (re.compile(r"\\overline\{([^{}]*)\}"), "\u0305"),
    (re.compile(r"\\tilde\{([^{}]*)\}"), "\u0303"),
    (re.compile(r"\\dot\{([^{}]*)\}"), "\u0307"),
    (re.compile(r"\\ddot\{([^{}]*)\}"), "\u0308"),
]

_CODE_PH = "\uE000{}#\uE001"


# ------------------------------------------------------------- protección
def _protect_code(content: str) -> Tuple[str, List[str]]:
    """Sustituye bloques y spans de código por marcadores temporales."""
    snippets: List[str] = []

    def _repl(match: re.Match) -> str:
        idx = len(snippets)
        snippets.append(match.group(0))
        return _CODE_PH.format(idx)

    protected = re.sub(r"```.*?```", _repl, content, flags=re.DOTALL)
    protected = re.sub(r"`[^`\n]*`", _repl, protected)
    return protected, snippets


def _restore_code(content: str, snippets: List[str]) -> str:
    for idx, snippet in enumerate(snippets):
        content = content.replace(_CODE_PH.format(idx), snippet)
    return content


# -------------------------------------------------------------- conversión
def _split_braced(text: str) -> List[str]:
    """Extrae los grupos ``{...}`` balanceados consecutivos de ``text``."""
    groups: List[str] = []
    i, n = 0, len(text)
    while i < n and text[i] == "{":
        depth = 0
        j = i
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    groups.append(text[i + 1 : j])
                    i = j + 1
                    break
            j += 1
        else:
            break
    return groups


def _to_sup(expr: str) -> Optional[str]:
    try:
        return "".join(SUPERSCRIPT[ch] for ch in expr)
    except KeyError:
        return None


def _to_sub(expr: str) -> Optional[str]:
    try:
        return "".join(SUBSCRIPT[ch] for ch in expr)
    except KeyError:
        return None


def _replace_frac(num: str, den: str) -> str:
    return f"({_convert_core(num)})/({_convert_core(den)})"


def _replace_binom(a: str, b: str) -> str:
    return f"C({_convert_core(a)},{_convert_core(b)})"


def _replace_sqrt(index: Optional[str], body: str) -> str:
    body_c = _convert_core(body)
    if index is None:
        return f"√({body_c})"
    idx = _convert_core(index)
    return f"√[{idx}]({body_c})"


def _replace_structures(expr: str) -> str:
    r"""Procesa \frac, \binom y \sqrt con argumentos balanceados."""
    out = ""
    i, n = 0, len(expr)
    while i < n:
        m = _STRUCT.search(expr, i)
        if not m:
            out += expr[i:]
            break
        out += expr[i : m.start()]
        cmd = m.group(1)
        j = m.end()
        index: Optional[str] = None
        if cmd == "sqrt" and j < n and expr[j] == "[":
            k = expr.find("]", j)
            if k != -1:
                index = expr[j + 1 : k]
                j = k + 1
        groups = _split_braced(expr[j:])
        if cmd == "frac" and len(groups) >= 2:
            out += _replace_frac(groups[0], groups[1])
            j += len(groups[0]) + len(groups[1]) + 4
        elif cmd == "binom" and len(groups) >= 2:
            out += _replace_binom(groups[0], groups[1])
            j += len(groups[0]) + len(groups[1]) + 4
        elif cmd == "sqrt" and len(groups) >= 1:
            out += _replace_sqrt(index, groups[0])
            j += len(groups[0]) + 2
        else:
            out += "\\" + cmd
            j = m.end()
        i = j
    return out


def _replace_textlike(match: re.Match) -> str:
    return _convert_core(match.group(1))


def _replace_pmod(match: re.Match) -> str:
    return f"(mod {_convert_core(match.group(1))})"


def _replace_supb(match: re.Match) -> str:
    inner = _convert_core(match.group(1))
    t = _to_sup(inner)
    return t if t is not None else f"^({inner})"


def _replace_subb(match: re.Match) -> str:
    inner = _convert_core(match.group(1))
    t = _to_sub(inner)
    return t if t is not None else f"_({inner})"


def _replace_sup1(match: re.Match) -> str:
    ch = match.group(1)
    t = SUPERSCRIPT.get(ch)
    return t if t is not None else f"^{ch}"


def _replace_sub1(match: re.Match) -> str:
    ch = match.group(1)
    t = SUBSCRIPT.get(ch)
    return t if t is not None else f"_{ch}"


def _replace_cmd(match: re.Match) -> str:
    name = match.group(1)
    if name in GREEK:
        return GREEK[name]
    if name in SYMBOLS:
        return SYMBOLS[name]
    return name


def _convert_core(expr: str) -> str:
    """Convierte una expresión LaTeX a texto Unicode aproximado."""
    expr = expr.strip()
    expr = _NEWLINE.sub(" ", expr)
    expr = _ENV.sub(" ", expr)
    expr = expr.replace("&", " ")
    expr = _PMOD.sub(_replace_pmod, expr)
    expr = _TEXTLIKE.sub(_replace_textlike, expr)
    for pattern, mark in _ACCENT:
        expr = pattern.sub(
            lambda m, mark=mark: _convert_core(m.group(1)) + mark, expr
        )
    expr = _LEFT.sub("", expr)
    expr = _RIGHT.sub("", expr)
    expr = _LIMITS.sub("", expr)
    expr = _BMOD.sub(" mod ", expr)
    expr = _SPACE.sub("", expr)
    expr = _replace_structures(expr)
    expr = _SUPB.sub(_replace_supb, expr)
    expr = _SUP1.sub(_replace_sup1, expr)
    expr = _SUBB.sub(_replace_subb, expr)
    expr = _SUB1.sub(_replace_sub1, expr)
    expr = _CMD.sub(_replace_cmd, expr)
    expr = expr.replace("{", "").replace("}", "")
    # Escapa lo que el parser de markdown interpretaría como formato
    for ch in ("_", "*", "~", "`", "[", "]"):
        expr = expr.replace(ch, "\\" + ch)
    return expr


def _is_mathish(text: str) -> bool:
    """¿Parece LaTeX de verdad? Evita romper ``$var`` de shell o precios."""
    if len(text) == 1 and (text.isalnum() or text in "+-*/"):
        return True
    return bool(re.search(r"[\\^_{}=<>]", text))


def _convert_inline(match: re.Match) -> str:
    if not _is_mathish(match.group(1)):
        return match.group(0)
    return _convert_core(match.group(1))


# ------------------------------------------------------------ API pública
def prepare_markdown(content: str) -> str:
    """Prepara markdown para renderizar: convierte LaTeX y protege el código.

    Los bloques y spans de código se dejan intactos; el resto del texto se
    procesa en busca de expresiones matemáticas.
    """
    content, snippets = _protect_code(content)
    content = _DISPLAY.sub(lambda m: _convert_core(m.group(1)), content)
    content = _BRACKET.sub(lambda m: _convert_core(m.group(1)), content)
    content = _PAREN.sub(lambda m: _convert_core(m.group(1)), content)
    content = _INLINE.sub(_convert_inline, content)
    return _restore_code(content, snippets)
