"""Commands package — single entry point for all CLI command handling.

Registry format:
    { "/name": handler }

Each handler has signature: (raw: str, agent: QueryAgent) -> bool
    raw  — full command string, e.g. "/context -cd *"
    agent — QueryAgent instance (for commands that need state)

Returns True  → run loop should break (e.g. /exit).
Returns False → run loop continues.
"""

from src.agents import QueryAgent
from src.commands.handlers import (
    run_context,
    run_exit,
    run_help,
    run_love,
    run_model,
    run_sessions,
    run_skim,
    run_summary,
    run_tokens,
)

from src.commands.completer import CommandCompleter, get_completer

__all__ = [
    "COMMAND_HANDLERS",
    "handle_command",
    "CommandCompleter",
    "get_completer",
]

COMMAND_HANDLERS: dict[str, callable] = {
    "/context":    run_context,
    "/exit":       run_exit,
    "/help":       run_help,
    "/I!Love'You!": run_love,
    "/model":      run_model,
    "/sessions":   run_sessions,
    "/skim":       run_skim,
    "/summary":    run_summary,
    "/tokens":     run_tokens,
}


def handle_command(raw: str, agent: QueryAgent) -> bool:
    """Dispatch a /command. Returns True if the loop should break."""
    stripped = raw.lstrip("/").strip()
    name = "/" + stripped.split()[0] if stripped else ""

    handler = COMMAND_HANDLERS.get(name)
    if handler is None:
        print(f"Unknown command: {name.lstrip('/')}")
        return False

    return handler(raw, agent)
