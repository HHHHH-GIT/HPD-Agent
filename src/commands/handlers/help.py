"""Handler for the /help command."""

from src.agents import QueryAgent
from src.commands.details import COMMAND_DETAILS


def run(raw: str, agent: QueryAgent) -> bool:
    """Print the help table."""
    print("Available commands:")
    for cmd, meta in COMMAND_DETAILS.items():
        first_line = meta.split("\n")[0]
        print(f"  {cmd} - {first_line}")
        for line in meta.split("\n")[1:]:
            print(f"        {line}")
    return False
