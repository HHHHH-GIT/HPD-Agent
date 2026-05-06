"""Handler for the /help command."""

from src.agents import QueryAgent
from src.cli import get_renderer
from src.commands.details import COMMAND_DETAILS


def run(raw: str, agent: QueryAgent) -> bool:
    """Print the help table."""
    get_renderer().render_help(COMMAND_DETAILS)
    return False
