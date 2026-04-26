from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import _create_msgpack_ext_hook

# Modules that need explicit allowlist registration for msgpack deserialization.
# Format: (module_name, class_name) — must match what _check_allowed does internally.
_MSGPACK_ALLOWLIST = [
    ("src.core.enums", "TaskDifficulty"),
    ("src.core.models", "SubTask"),
    ("src.core.models", "SubTaskOutput"),
    ("src.core.models", "PlannerResult"),
    ("src.core.models", "TaskOutput"),
    ("src.core.models", "AgentMeta"),
    ("src.memory.context", "Message"),
    ("src.memory.context", "MessageRole"),
    ("src.memory.context", "ConversationContext"),
]


def _patch_serde(checkpointer: MemorySaver) -> MemorySaver:
    """Patch the checkpointer's serializer to allow our custom types.

    The default base allowlist is True (warn-but-allow everything). We set an
    explicit allowlist so langgraph doesn't warn on our pydantic models.
    _check_allowed inside langgraph uses (module, name) tuples as keys.
    """
    serde = checkpointer.serde
    serde._allowed_msgpack_modules = _MSGPACK_ALLOWLIST
    serde._unpack_ext_hook = _create_msgpack_ext_hook(_MSGPACK_ALLOWLIST)
    return checkpointer


def get_checkpointer(backend: str = "memory") -> MemorySaver:
    """Return a checkpointer for the given backend.

    Args:
        backend: One of "memory", "sqlite", or "postgres".
                 Falls back to "memory" for unknown values.
    """
    match backend:
        case "sqlite":
            from langgraph.checkpoint.sqlite.aio import SqliteSaver

            return _patch_serde(SqliteSaver.from_conn_string("checkpoints.db"))
        case "postgres":
            from langgraph.checkpoint.postgres.aio import PostgresSaver

            conn_string = "postgresql://user:password@localhost:5432/agent_db"
            return _patch_serde(PostgresSaver.from_conn_string(conn_string))
        case _:
            return _patch_serde(MemorySaver())
