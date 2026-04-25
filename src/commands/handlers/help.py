"""Handler for the /help command."""

from src.agents import QueryAgent
from src.commands.details import COMMAND_DETAILS


def run(raw: str, agent: QueryAgent) -> bool:
    """Print the help table."""
    print("Available commands:")
    for cmd, meta in COMMAND_DETAILS.items():
        print(f"  {cmd} - {meta}")
    return False
