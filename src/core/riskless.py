"""Persistent runtime setting for terminal confirmation behavior."""

from __future__ import annotations

import json
from pathlib import Path

_CONFIG_PATH = Path.home() / ".hpagent" / "config.json"
_riskless_enabled: bool = False


def _load_config() -> None:
    """Load riskless mode from the shared CLI config file."""
    global _riskless_enabled
    try:
        if not _CONFIG_PATH.exists():
            return
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        value = cfg.get("riskless_mode")
        if isinstance(value, bool):
            _riskless_enabled = value
    except Exception:
        pass


def _save_config() -> None:
    """Save riskless mode to the shared CLI config file."""
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cfg = {}
        if _CONFIG_PATH.exists():
            cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        cfg["riskless_mode"] = _riskless_enabled
        _CONFIG_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def is_riskless_enabled() -> bool:
    return _riskless_enabled


def set_riskless_enabled(enabled: bool) -> None:
    global _riskless_enabled
    _riskless_enabled = bool(enabled)
    _save_config()


_load_config()


__all__ = ["is_riskless_enabled", "set_riskless_enabled"]
