"""Detección de Skills y contexto de proyecto (AGENTS.md).

Convención (compatible con Claude Code):
  - Un skill es un directorio que contiene un `SKILL.md` con frontmatter YAML
    (`name`, `description`) seguido de instrucciones en Markdown.
  - Se buscan en: `<proyecto>/.giar/skills`, `<proyecto>/.claude/skills`,
    `<proyecto>/skills` y `~/.giar/skills`.
  - El contexto de proyecto se lee de `AGENTS.md` (y sube por directorios
    padres hasta el home).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

SKILL_FILENAME = "SKILL.md"


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    content: str
    files: List[Path] = field(default_factory=list)

    @property
    def source(self) -> str:
        return str(self.path.parent)


def _split_frontmatter(text: str) -> tuple[str, str]:
    if text.startswith("---"):
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
        if m:
            return m.group(1), m.group(2)
    return "", text


def parse_skill_dir(skill_dir: Path) -> Optional[Skill]:
    skill_file = skill_dir / SKILL_FILENAME
    if not skill_file.is_file():
        return None
    try:
        text = skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    frontmatter, body = _split_frontmatter(text)
    meta: dict = {}
    if frontmatter:
        try:
            parsed = yaml.safe_load(frontmatter)
            if isinstance(parsed, dict):
                meta = parsed
        except Exception:
            meta = {}
    name = str(meta.get("name") or skill_dir.name).strip()
    description = str(meta.get("description") or "").strip()
    files = sorted(
        p
        for p in skill_dir.iterdir()
        if p.is_file() and p.name != SKILL_FILENAME and not p.name.startswith(".")
    )
    return Skill(
        name=name,
        description=description,
        path=skill_file,
        content=(body or "").strip(),
        files=files,
    )


def _scan_dir(dirpath: Path, out: List[Skill]) -> None:
    if not dirpath.is_dir():
        return
    for child in sorted(dirpath.iterdir()):
        if child.is_dir() and (child / SKILL_FILENAME).is_file():
            skill = parse_skill_dir(child)
            if skill is not None:
                out.append(skill)


def discover_skills(cwd: Path, user_home: Path) -> List[Skill]:
    """Busca skills en el proyecto actual y en el home del usuario."""
    found: List[Skill] = []
    candidates = [
        cwd / ".giar" / "skills",
        cwd / ".claude" / "skills",
        cwd / "skills",
        user_home / "skills",
    ]
    for loc in candidates:
        _scan_dir(loc, found)
    return found


def find_agents_md(cwd: Path, stop: Path) -> Optional[Path]:
    """Busca AGENTS.md desde cwd hacia arriba (hasta `stop` inclusive)."""
    current = cwd.resolve()
    stop = stop.resolve()
    while True:
        candidate = current / "AGENTS.md"
        if candidate.is_file():
            return candidate
        if current == stop or current.parent == current:
            return None
        current = current.parent


def load_project_context(cwd: Path, user_home: Path) -> str:
    agents = find_agents_md(cwd, user_home)
    if agents is None:
        return ""
    try:
        return agents.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
