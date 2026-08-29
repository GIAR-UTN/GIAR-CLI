"""Gestión de la configuración de GIAR.

La configuración vive en `~/.giar/config.json` (o `$GIAR_HOME/config.json`).
Contiene el proveedor LLM (endpoint OpenAI-compatible + api key) y la lista
de servidores MCP tipo streamable-http.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_HOME = "~/.giar"
CONFIG_FILE = "config.json"
HISTORY_FILE = "history.txt"


def get_home() -> Path:
    return Path(os.environ.get("GIAR_HOME", DEFAULT_HOME)).expanduser()


def get_config_path() -> Path:
    return get_home() / CONFIG_FILE


def get_history_path() -> Path:
    return get_home() / HISTORY_FILE


def default_config() -> Dict[str, Any]:
    return {
        "provider": {
            "base_url": "",
            "model": "",
            "api_key": "",
            "headers": {},
            "reasoning_effort": "",
            "show_reasoning": True,
        },
        "mcps": [],
        "skills": {"disabled": []},
    }


class Config:
    """Capa sobre el dict de configuración con helpers de acceso."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self.data: Dict[str, Any] = data if data is not None else default_config()

    # ------------------------------------------------------------- persistence
    @classmethod
    def load(cls) -> "Config":
        path = get_config_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                backup = cls._backup_corrupt(path)
                print(
                    f"giar: config.json corrupto ({exc}); se respaldó en {backup} "
                    "y se usará la configuración por defecto.",
                    file=sys.stderr,
                )
                return cls()
            if isinstance(data, dict):
                return cls(data)
            backup = cls._backup_corrupt(path)
            print(
                f"giar: config.json con formato inesperado; se respaldó en {backup} "
                "y se usará la configuración por defecto.",
                file=sys.stderr,
            )
        return cls()

    @staticmethod
    def _backup_corrupt(path: Path) -> Path:
        import time

        backup = path.with_name(
            f"config.json.corrupt-{time.strftime('%Y%m%d%H%M%S')}"
        )
        try:
            path.rename(backup)
        except OSError:
            pass
        return backup

    def save(self) -> None:
        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)

    # ----------------------------------------------------------------- provider
    @property
    def provider(self) -> Dict[str, Any]:
        return self.data.setdefault("provider", {})

    @property
    def base_url(self) -> str:
        return (self.provider.get("base_url") or "").strip().rstrip("/")

    @property
    def model(self) -> str:
        return (self.provider.get("model") or "").strip()

    @property
    def api_key(self) -> str:
        key = (self.provider.get("api_key") or "").strip()
        if not key:
            key = os.environ.get("GIAR_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        return key.strip()

    @property
    def extra_headers(self) -> Dict[str, str]:
        return dict(self.provider.get("headers") or {})

    @property
    def reasoning_effort(self) -> str:
        return (self.provider.get("reasoning_effort") or "").strip()

    @property
    def show_reasoning(self) -> bool:
        return bool(self.provider.get("show_reasoning", True))

    def set_show_reasoning(self, value: bool) -> None:
        self.provider["show_reasoning"] = bool(value)
        self.save()

    def set_provider(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        base_url = base_url.strip().rstrip("/")
        self.provider["base_url"] = base_url
        self.provider["model"] = model.strip()
        if api_key is not None:
            self.provider["api_key"] = api_key.strip()
        if extra_headers is not None:
            self.provider["headers"] = dict(extra_headers)
        if reasoning_effort is not None:
            self.provider["reasoning_effort"] = reasoning_effort.strip().lower()
        self.save()

    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)

    # --------------------------------------------------------------------- mcp
    @property
    def mcps(self) -> List[Dict[str, Any]]:
        return self.data.setdefault("mcps", [])

    def enabled_mcps(self) -> List[Dict[str, Any]]:
        return [m for m in self.mcps if m.get("enabled", True)]

    def add_mcp(
        self, name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> None:
        entry = {
            "name": name.strip(),
            "url": url.strip(),
            "headers": dict(headers or {}),
            "enabled": True,
        }
        self.remove_mcp(entry["name"])
        self.mcps.append(entry)
        self.save()

    def remove_mcp(self, name: str) -> bool:
        before = len(self.mcps)
        self.data["mcps"] = [m for m in self.mcps if m.get("name") != name]
        if len(self.mcps) != before:
            self.save()
            return True
        return False

    def get_mcp(self, name: str) -> Optional[Dict[str, Any]]:
        for m in self.mcps:
            if m.get("name") == name:
                return m
        return None

    def set_mcp_enabled(self, name: str, enabled: bool) -> bool:
        m = self.get_mcp(name)
        if m is None:
            return False
        m["enabled"] = bool(enabled)
        self.save()
        return True

    # ------------------------------------------------------------------- skills
    @property
    def skills_config(self) -> Dict[str, Any]:
        return self.data.setdefault("skills", {})

    def is_skill_enabled(self, name: str) -> bool:
        return name not in set(self.skills_config.get("disabled", []))

    def set_skill_enabled(self, name: str, enabled: bool) -> None:
        disabled = set(self.skills_config.get("disabled", []))
        if enabled:
            disabled.discard(name)
        else:
            disabled.add(name)
        self.skills_config["disabled"] = sorted(disabled)
        self.save()

    # -------------------------------------------------------------------- misc
    def to_redacted_dict(self) -> Dict[str, Any]:
        """Representación segura (sin api keys) para mostrar."""
        redacted = json.loads(json.dumps(self.data))
        if redacted.get("provider", {}).get("api_key"):
            redacted["provider"]["api_key"] = "******"
        return redacted
