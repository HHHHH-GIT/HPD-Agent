"""Handler for the /sessions command — list, switch, or delete sessions for the current project."""

from src.agents import QueryAgent


def run(raw: str, agent: QueryAgent) -> bool:
    """List sessions, switch to a session, or delete a session."""
    parts = raw.strip().split()
    ph = getattr(agent, "_project_hash", "unknown")

    if len(parts) > 1 and parts[1] == "delete":
        if len(parts) > 2:
            target = parts[2]
            deleted = agent.delete_session(target)
            if deleted:
                print(f"Session [{target}] deleted.")
            else:
                print(f"Session [{target}] not found.")
        else:
            print("Usage: /sessions delete <session_id>")
        return False

    if len(parts) > 1:
        target = parts[1]
        if target not in agent._contexts:
            print(f"Session not found: {target}")
        else:
            agent._current_session = target
            msg_count = len(agent._get_context(target).messages)
            print(f"Switched to session [{target}] ({msg_count} messages)")
        return False

    from src.memory.session_store import _store_dir
    store_path = _store_dir(ph)
    print(f"Sessions for current project [{ph}] (stored at {store_path}):\n")

    if not agent._contexts:
        print("  No active sessions in this project.")
        return False

    current = agent._current_session
    for sid in sorted(agent._contexts):
        marker = " <-- current" if sid == current else ""
        count = len(agent._contexts[sid].messages)
        print(f"  [{sid}]  ({count} messages){marker}")

    return False
