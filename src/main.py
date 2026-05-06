import argparse
import asyncio
import os
import sys
import warnings

# Fix terminal encoding BEFORE any I/O — must be at the very top
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding and sys.stdout.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stderr.reconfigure(encoding="utf-8")

# Suppress LangChain / pydantic noise before any other imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*stream.*")
warnings.filterwarnings("ignore", message=".*pydantic.*")
warnings.filterwarnings("ignore", message=".*serialized.*")
warnings.filterwarnings("ignore", message=".*WARNING.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.agents import QueryAgent
from src.cli import get_renderer
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_llm
from src.commands import COMMAND_HANDLERS, handle_command, get_completer
from src.commands.handlers.tokens import get_used_tokens, estimate_next_request_tokens, MAX_TOKENS
from src.commands.handlers.trace import is_trace_enabled, is_trace_save_enabled, get_trace_mode
from src.models import get_store

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style


_renderer = get_renderer()
_active_agent: QueryAgent | None = None


# ----------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------


def _build_session() -> PromptSession:
    """Build a PromptSession: Enter submits, Ctrl+J inserts newline."""
    kb = KeyBindings()

    @kb.add(Keys.ControlC, eager=True)
    def _ctrl_c(_):
        raise KeyboardInterrupt

    @kb.add(Keys.ControlJ)
    def _newline(event):
        event.app.current_buffer.insert_text("\n")

    return PromptSession(
        multiline=False,
        key_bindings=kb,
        enable_history_search=False,
        complete_while_typing=True,
        prompt_continuation="  │ ",
        completer=get_completer(),
        bottom_toolbar=_build_toolbar,
        style=Style.from_dict(
            {
                "bottom-toolbar": "bg:#0f172a #cbd5e1",
            }
        ),
    )


def _build_toolbar() -> HTML:
    agent = _active_agent
    session_id = getattr(agent, "_current_session", "default")
    trace_mode = get_trace_mode().upper()
    try:
        model_name = get_store().active_profile().model
    except Exception:
        model_name = "unknown"
    return HTML(
        "  <b fg='#22d3ee'>Session</b> "
        f"<style fg='#e2e8f0'>{session_id}</style>  "
        "<b fg='#22d3ee'>Model</b> "
        f"<style fg='#e2e8f0'>{model_name}</style>  "
        "<b fg='#22d3ee'>Trace</b> "
        f"<style fg='#e2e8f0'>{trace_mode}</style>  "
        "<style fg='#94a3b8'>Ctrl+J newline • /help commands</style>"
    )


def _build_prompt_message(agent: QueryAgent) -> HTML:
    width = _renderer.prompt_width()
    session_id = agent._current_session
    top = "─" * width
    return HTML(
        f"<style fg='#475569'>{top}</style>\n"
        f"<b fg='#22d3ee'>Input</b> "
        f"<style fg='#e2e8f0'>[{session_id}]</style>\n"
        "<style fg='#94a3b8'>╰─› </style>"
    )


def _read_input(prompt: HTML) -> str:
    """Read input via prompt_toolkit — true multi-line editing with full cursor control."""
    try:
        return _session.prompt(prompt)
    except KeyboardInterrupt:
        raise
    except UnicodeDecodeError:
        sys.stdin.readline()
        return _session.prompt(prompt)


# Shared session (created once, reused)
_session = _build_session()


async def _read_line(prompt: HTML) -> str:
    """Run the blocking prompt_toolkit call in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _read_input, prompt)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


async def run_loop():
    tracer = get_tracer()
    _renderer.install_print_hook()
    _renderer.render_banner()

    agent = QueryAgent()
    global _active_agent
    _active_agent = agent
    get_completer().set_agent(agent)

    while True:
        try:
            query = await _read_line(_build_prompt_message(agent))
        except (KeyboardInterrupt, EOFError):
            _renderer.blank()
            _renderer.info("Goodbye!")
            break

        if not query:
            continue
        if query.startswith("/"):
            if await handle_command(query, agent):
                break
            continue

        _renderer.blank()

        tokens = estimate_next_request_tokens(agent, query)
        if tokens > MAX_TOKENS:
            occupied = get_used_tokens(agent)
            _renderer.warning(
                "Token limit reached for the next request. "
                f"occupied={occupied}, next_estimate={tokens}. "
                "Please create a new conversation or run /summary to summarize the conversation."
            )
            continue

        _tracing = is_trace_enabled()
        if _tracing:
            trace_id = tracer.start_trace(query=query, session_id=agent._current_session)
            TokenTrackerCallback.reset()
            _renderer.rule(f"Run • trace:{trace_id}", style="trace")
            _renderer.info("开始处理...")

        try:
            result = await agent.ainvoke(query, agent._current_session)

            # Complex path: stream synthesis, record tokens
            if result.get("synthesis_prompt"):
                _renderer.blank()
                _renderer.rule("Final Answer", style="accent")
                if _tracing:
                    with tracer.span("synthesis_stream") as synthesis_span_id:
                        llm = get_llm(stream=True)
                        streamed_answer = ""
                        async for chunk in llm.astream(result["synthesis_prompt"]):
                            t = getattr(chunk, "content", "") or ""
                            if t:
                                streamed_answer += t
                                _renderer.stream_answer(t)
                        agent.store_streamed_answer(streamed_answer)
                        tin, tout, model = TokenTrackerCallback.snapshot()
                        tracer.record_tokens(synthesis_span_id, tokens_in=tin, tokens_out=tout, model=model)
        finally:
            # ── Observability: end trace ───────────────────────────────
            if _tracing:
                record = tracer.end_trace()
                if record is not None:
                    _renderer.blank()
                    record.print_console()
                    if is_trace_save_enabled():
                        path = record.save()
                        _renderer.success(f"Trace saved: {path}")

        _renderer.blank()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="HPD-Agent CLI — AI coding assistant with multi-agent planning."
    )
    parser.add_argument(
        "-p", "--project",
        metavar="PATH",
        help="Set the working directory (cwd) for this session. "
             "All git/project detection and tool operations use this path. "
             "Example: python -m src.main -p /path/to/project",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.project:
        os.chdir(os.path.expanduser(args.project))
    asyncio.run(run_loop())


if __name__ == "__main__":
    main()
