"""Handler for the /I!Love'You! command."""

from src.agents import QueryAgent


def run(raw: str, agent: QueryAgent) -> bool:
    """Show some love."""
    print("I love you too!")
    return False
