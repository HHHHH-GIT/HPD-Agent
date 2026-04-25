"""Handler for the /new command — create a new session."""

import uuid

from src.agents import QueryAgent
from src.memory import ConversationContext


def run(raw: str, agent: QueryAgent) -> bool:
    """Create a new session and switch to it."""
    new_id = uuid.uuid4().hex[:8]
    agent._contexts[new_id] = ConversationContext()
    agent._current_session = new_id
    print(f"New session created: [{new_id}]")
    return False
