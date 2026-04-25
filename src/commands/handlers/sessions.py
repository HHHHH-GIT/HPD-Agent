"""Handler for the /sessions command — list or switch sessions."""

from src.agents import QueryAgent


def run(raw: str, agent: QueryAgent) -> bool:
    """List all sessions, or switch to a given session."""
    parts = raw.strip().split()
    target = parts[1] if len(parts) > 1 else None

    if target:
        if target not in agent._contexts:
            print(f"Session not found: {target}")
        else:
            agent._current_session = target
            msg_count = len(agent._get_context(target).messages)
            print(f"Switched to session [{target}] ({msg_count} messages)")
    else:
        if not agent._contexts:
            print("No active sessions.")
            return False

        current = getattr(agent, "_current_session", None) or (
            list(agent._contexts.keys())[0] if agent._contexts else None
        )
        print("Active sessions:")
        for sid in sorted(agent._contexts):
            marker = " <-- current" if sid == current else ""
            count = len(agent._contexts[sid].messages)
            print(f"  [{sid}]  ({count} messages){marker}")

    return False
