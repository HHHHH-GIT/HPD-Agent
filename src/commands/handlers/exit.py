"""Handler for the /exit command."""

from src.agents import QueryAgent


def run(raw: str, agent: QueryAgent) -> bool:
    """Exit the agent.

    Returns True to signal the run loop should break.
    """
    print("Goodbye!")
    return True
