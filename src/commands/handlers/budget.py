"""Handler for /budget tool-loop budget mode."""

from __future__ import annotations

from src.agents import QueryAgent
from src.cli import get_renderer
from src.core.tool_budget import (
    PROFILES,
    get_tool_budget_profile,
    set_tool_budget_mode,
)


def _profile_line(mode: str) -> str:
    profile = PROFILES[mode]  # type: ignore[index]
    return (
        f"{profile.mode}: rounds={profile.max_rounds}, "
        f"tool_calls={profile.max_tool_calls}, context_policy={profile.tool_context_policy}"
    )


def run(raw: str, agent: QueryAgent) -> bool:
    """Show or set the active tool-loop budget."""
    _ = agent
    renderer = get_renderer()
    parts = raw.strip().split()

    if len(parts) == 1:
        profile = get_tool_budget_profile()
        renderer.info(
            "tool budget: "
            f"{profile.mode} "
            f"(rounds={profile.max_rounds}, tool_calls={profile.max_tool_calls}, "
            f"context_policy={profile.tool_context_policy})"
        )
        return False

    mode = parts[1].lower()
    if mode not in PROFILES:
        renderer.error(
            "Usage: /budget [normal|extended|web]\n"
            + "\n".join(_profile_line(item) for item in PROFILES)
        )
        return False

    profile = set_tool_budget_mode(mode)
    renderer.success(
        "tool budget: "
        f"{profile.mode} "
        f"(rounds={profile.max_rounds}, tool_calls={profile.max_tool_calls}). "
        "Tool-loop context gate remains enabled."
    )
    return False


__all__ = ["run"]
