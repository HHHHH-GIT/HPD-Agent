"""Handler for the /context command."""

from src.agents import QueryAgent
from src.cli import get_renderer
from src.memory.context import ConversationContext, MessageRole


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def _parse_context_args(raw: str) -> tuple[int, bool]:
    """Parse /context arguments.

    Returns:
        (count, full_content):
            - count: number of messages to show (0 means "all")
            - full_content: whether to show the full message (no truncation)
    """
    count = 5
    full_content = False

    raw = raw.strip()
    if not raw:
        return count, full_content

    for token in raw.split():
        if token == "-c":
            pass
        elif token == "-d":
            full_content = True
        elif token == "-cd" or token == "-dc":
            full_content = True
        elif token == "*":
            count = 0
        elif token.lstrip("-").isdigit():
            count = int(token.lstrip("-"))

    return count, full_content


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------
def _format_message(msg_obj: object, idx: int, full_content: bool) -> str:
    """Format a single message for display.

    For assistant messages with answer_content, shows the clean answer instead of
    the full internal content (which may include sub-task details and synthesis
    instructions).
    """
    role_label = "用户" if msg_obj.role == MessageRole.USER else "助手"
    ts = msg_obj.timestamp.strftime("%H:%M:%S")

    # Prefer answer_content for clean display; fall back to content for full detail.
    display_content = (
        msg_obj.answer_content if msg_obj.answer_content else msg_obj.content
    )
    if not full_content and len(display_content) > 100:
        display_content = display_content[:100] + "..."

    return f"  [{idx}] {role_label} @ {ts}\n    {display_content}"


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def run(raw: str, agent: QueryAgent) -> bool:
    """Handle /context command.

    Args:
        raw: Full command string, e.g. "/context -cd *" or "/context clear".
        agent: QueryAgent instance (provides access to conversation contexts).

    Returns:
        False (never exits).
    """
    sub_args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""

    if sub_args.strip().lower() == "clear":
        ctx = agent._get_context()
        ctx.messages.clear()
        ctx.sub_task_outputs.clear()
        get_renderer().success("上下文已清空。")
        return False

    ctx = agent._get_context()
    _print_context(ctx, sub_args=sub_args, session_id=agent._current_session)
    return False


def _print_context(
    context: ConversationContext,
    sub_args: str = "",
    session_id: str = "default",
) -> None:
    """Print the context window to stdout.

    Args:
        context: The ConversationContext to display.
        sub_args: Raw arguments after the command name, e.g. "-cd *".
        session_id: Session identifier (shown in header).
    """
    count, full_content = _parse_context_args(sub_args)
    messages = context.messages if context else []

    if count == 0:
        to_show = list(enumerate(messages))
    else:
        recent = list(enumerate(messages[-count * 2:]))
        to_show = [(len(messages) - len(recent) + i, pair[1]) for i, pair in enumerate(recent)]
    entries = []
    for idx, msg in reversed(to_show):
        role = "用户" if msg.role == MessageRole.USER else "助手"
        ts = msg.timestamp.strftime("%H:%M:%S")
        content = _format_message(msg, idx + 1, full_content).split("\n", 1)[1].strip()
        entries.append((role, ts, content))
    get_renderer().render_context(session_id, len(messages), context.max_turns, entries)
