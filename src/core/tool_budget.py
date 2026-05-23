"""Persistent tool-loop budget mode for CLI requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.llm.tool_context import ToolContextPolicy

ToolBudgetMode = Literal["normal", "extended", "web"]

_CONFIG_PATH = Path.home() / ".hpagent" / "config.json"


@dataclass(frozen=True)
class ToolBudgetProfile:
    mode: ToolBudgetMode
    max_rounds: int
    max_tool_calls: int
    tool_context_policy: ToolContextPolicy = "auto"


PROFILES: dict[ToolBudgetMode, ToolBudgetProfile] = {
    "normal": ToolBudgetProfile(
        mode="normal",
        max_rounds=30,
        max_tool_calls=80,
        tool_context_policy="auto",
    ),
    "extended": ToolBudgetProfile(
        mode="extended",
        max_rounds=60,
        max_tool_calls=160,
        tool_context_policy="auto",
    ),
    "web": ToolBudgetProfile(
        mode="web",
        max_rounds=120,
        max_tool_calls=300,
        tool_context_policy="auto",
    ),
}

_budget_mode: ToolBudgetMode = "normal"


def _load_config() -> None:
    global _budget_mode
    try:
        if not _CONFIG_PATH.exists():
            return
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        value = cfg.get("tool_budget_mode")
        if value in PROFILES:
            _budget_mode = value
    except Exception:
        pass


def _save_config() -> None:
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cfg = {}
        if _CONFIG_PATH.exists():
            cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        cfg["tool_budget_mode"] = _budget_mode
        _CONFIG_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def get_tool_budget_mode() -> ToolBudgetMode:
    return _budget_mode


def get_tool_budget_profile() -> ToolBudgetProfile:
    return PROFILES[_budget_mode]


def set_tool_budget_mode(mode: str) -> ToolBudgetProfile:
    global _budget_mode
    normalized = mode.strip().lower()
    if normalized not in PROFILES:
        raise ValueError(f"Unknown tool budget mode: {mode}")
    _budget_mode = normalized  # type: ignore[assignment]
    _save_config()
    return PROFILES[_budget_mode]


_load_config()


__all__ = [
    "PROFILES",
    "ToolBudgetMode",
    "ToolBudgetProfile",
    "get_tool_budget_mode",
    "get_tool_budget_profile",
    "set_tool_budget_mode",
]
