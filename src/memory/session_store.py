"""File-based session persistence — one JSON file per session under ~/.hpagent/sessions/{project_hash}/.

Each project directory maps to a hash derived from its canonical path, so sessions
from different projects never collide. The hash is stable across CLI restarts for the
same path (symlinks / real paths are resolved).
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.memory.context import ConversationContext, Message


def _project_hash(cwd: str | None = None) -> str:
    """Compute a URL-safe hash from the canonical project path.

    Uses the real resolved path so that symlinks and trailing slashes don't
    produce different hashes for the same physical directory.
    """
    path = Path(cwd or os.getcwd()).resolve()
    key = str(path).encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


def _store_dir(project_hash: str) -> Path:
    base = Path.home() / ".hpagent" / "sessions" / project_hash
    base.mkdir(parents=True, exist_ok=True)
    return base


def _session_path(session_id: str, project_hash: str) -> Path:
    safe = session_id.replace("/", "_").replace("\\", "_")
    return _store_dir(project_hash) / f"{safe}.json"


def save(ctx: ConversationContext, session_id: str, project_hash: str | None = None) -> None:
    """Write session state to disk under the given project hash.

    Atomically replaces any existing data.
    """
    ph = project_hash or _project_hash()
    path = _session_path(session_id, ph)
    now = datetime.now(timezone.utc)
    payload = {
        "session_id": session_id,
        "project_hash": ph,
        "messages": [m.model_dump(mode="json") for m in ctx.messages],
        "sub_task_outputs": ctx.sub_task_outputs,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load(session_id: str, project_hash: str | None = None) -> ConversationContext | None:
    """Restore a session from disk for the given project hash. Returns None if not found."""
    ph = project_hash or _project_hash()
    path = _session_path(session_id, ph)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    msgs = [
        Message.model_validate(m) for m in raw.get("messages", [])
    ]
    ctx = ConversationContext(
        messages=msgs,
        sub_task_outputs=raw.get("sub_task_outputs", []),
    )
    return ctx


def list_sessions(project_hash: str | None = None) -> list[dict[str, Any]]:
    """Return metadata for all saved sessions under the given project hash (sorted newest first)."""
    ph = project_hash or _project_hash()
    items = []
    store = _store_dir(ph)
    if not store.exists():
        return items

    for p in store.glob("*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            items.append({
                "session_id": raw.get("session_id", p.stem),
                "message_count": len(raw.get("messages", [])),
                "updated_at": raw.get("updated_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return items


def delete(session_id: str, project_hash: str | None = None) -> bool:
    """Remove a session file for the given project. Returns True if deleted, False if not found."""
    ph = project_hash or _project_hash()
    path = _session_path(session_id, ph)
    if path.exists():
        path.unlink()
        return True
    return False


def has_session(session_id: str, project_hash: str | None = None) -> bool:
    """Return True if a session exists for the given project hash."""
    ph = project_hash or _project_hash()
    return _session_path(session_id, ph).exists()
