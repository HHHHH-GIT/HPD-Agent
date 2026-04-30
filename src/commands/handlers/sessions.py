"""Handler for the /sessions command — manage sessions for the current project.

Sub-commands:
    /sessions list                → list all sessions for the current project
    /sessions create             → create a new session and switch to it
    /sessions switch <id>       → switch to a session by id
    /sessions delete <id>       → delete a session by id
"""

import uuid

from src.agents import QueryAgent
from src.memory import ConversationContext


VALID_SUBS = ("list", "create", "switch", "delete")


def _run_list(agent: QueryAgent) -> None:
    from src.memory.session_store import _store_dir
    ph = getattr(agent, "_project_hash", "unknown")
    store_path = _store_dir(ph)
    print(f"Sessions for current project [{ph}] (stored at {store_path}):\n")

    if not agent._contexts:
        print("  No active sessions in this project.")
        return

    current = agent._current_session
    for sid in sorted(agent._contexts):
        marker = " <-- current" if sid == current else ""
        count = len(agent._contexts[sid].messages)
        print(f"  [{sid}]  ({count} messages){marker}")
    print()


def _run_create(agent: QueryAgent) -> None:
    new_id = uuid.uuid4().hex[:8]
    agent._contexts[new_id] = ConversationContext()
    agent._current_session = new_id
    print(f"New session created: [{new_id}]")


def _run_switch(agent: QueryAgent, target: str) -> None:
    if target not in agent._contexts:
        print(f"Session not found: [{target}]")
        return
    agent._current_session = target
    msg_count = len(agent._get_context(target).messages)
    print(f"Switched to session [{target}] ({msg_count} messages)")


def _run_delete(agent: QueryAgent, target: str) -> None:
    deleted = agent.delete_session(target)
    if deleted:
        print(f"Session [{target}] deleted.")
    else:
        print(f"Session [{target}] not found.")


def run(raw: str, agent: QueryAgent) -> bool:
    """Dispatch to the appropriate sub-command."""
    parts = raw.strip().split()

    if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "list"):
        _run_list(agent)
        return False

    sub = parts[1].lower()

    if sub not in VALID_SUBS:
        print(f"Unknown sub-command: '{sub}'")
        print(f"Valid sub-commands: {', '.join(VALID_SUBS)}\n")
        return False

    if sub == "list":
        _run_list(agent)
        return False

    if sub == "create":
        _run_create(agent)
        return False

    if sub == "switch":
        if len(parts) < 3:
            print("Usage: /sessions switch <session-id>\n")
            return False
        _run_switch(agent, parts[2])
        return False

    if sub == "delete":
        if len(parts) < 3:
            print("Usage: /sessions delete <session-id>\n")
            return False
        _run_delete(agent, parts[2])
        return False

    return False
