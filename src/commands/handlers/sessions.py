"""Handler for the /sessions command — manage sessions for the current project.

Sub-commands:
    /sessions list                → list all sessions for the current project
    /sessions create             → create a new session and switch to it
    /sessions switch <id>       → switch to a session by id
    /sessions delete <id>       → delete a session by id
"""

import uuid

from src.agents import QueryAgent
from src.cli import get_renderer
from src.memory import ConversationContext


VALID_SUBS = ("list", "create", "switch", "delete")


def _run_list(agent: QueryAgent) -> None:
    from src.memory.session_store import _store_dir
    ph = getattr(agent, "_project_hash", "unknown")
    store_path = _store_dir(ph)
    current = agent._current_session
    rows = [
        (sid, len(agent._contexts[sid].messages), sid == current)
        for sid in sorted(agent._contexts)
    ]
    get_renderer().render_sessions(ph, store_path, rows)


def _run_create(agent: QueryAgent) -> None:
    new_id = uuid.uuid4().hex[:8]
    agent._contexts[new_id] = ConversationContext()
    agent._current_session = new_id
    get_renderer().success(f"New session created: [{new_id}]")


def _run_switch(agent: QueryAgent, target: str) -> None:
    if target not in agent._contexts:
        get_renderer().error(f"Session not found: [{target}]")
        return
    agent._current_session = target
    msg_count = len(agent._get_context(target).messages)
    get_renderer().success(f"Switched to session [{target}] ({msg_count} messages)")


def _run_delete(agent: QueryAgent, target: str) -> None:
    deleted = agent.delete_session(target)
    if deleted:
        get_renderer().success(f"Session [{target}] deleted.")
    else:
        get_renderer().error(f"Session [{target}] not found.")


def run(raw: str, agent: QueryAgent) -> bool:
    """Dispatch to the appropriate sub-command."""
    parts = raw.strip().split()

    if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "list"):
        _run_list(agent)
        return False

    sub = parts[1].lower()

    if sub not in VALID_SUBS:
        get_renderer().error(f"Unknown sub-command: '{sub}'")
        get_renderer().info(f"Valid sub-commands: {', '.join(VALID_SUBS)}")
        return False

    if sub == "list":
        _run_list(agent)
        return False

    if sub == "create":
        _run_create(agent)
        return False

    if sub == "switch":
        if len(parts) < 3:
            get_renderer().error("Usage: /sessions switch <session-id>")
            return False
        _run_switch(agent, parts[2])
        return False

    if sub == "delete":
        if len(parts) < 3:
            get_renderer().error("Usage: /sessions delete <session-id>")
            return False
        _run_delete(agent, parts[2])
        return False

    return False
