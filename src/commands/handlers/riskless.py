"""Handler for /riskless terminal confirmation mode."""

from __future__ import annotations

from src.agents import QueryAgent
from src.cli import get_renderer
from src.core.riskless import is_riskless_enabled, set_riskless_enabled


def run(raw: str, agent: QueryAgent) -> bool:
    """Toggle terminal confirmation mode."""
    _ = agent
    renderer = get_renderer()
    parts = raw.strip().split()

    if len(parts) == 1:
        mode = "on" if is_riskless_enabled() else "off"
        renderer.info(f"riskless mode: {mode}")
        return False

    sub = parts[1].lower()
    if sub in ("on", "1", "enable", "true"):
        set_riskless_enabled(True)
        renderer.success(
            "riskless mode: on. Normal terminal commands run without confirmation; extreme commands still ask."
        )
    elif sub in ("off", "0", "disable", "false"):
        set_riskless_enabled(False)
        renderer.warning(
            "riskless mode: off. Non-read-only terminal commands require confirmation."
        )
    else:
        renderer.error("用法: /riskless [on|off]")
        return False

    return False


__all__ = ["run", "is_riskless_enabled"]
