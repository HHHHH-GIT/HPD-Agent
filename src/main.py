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
from src.llm import get_llm
from src.commands import COMMAND_HANDLERS, handle_command, get_completer
from src.commands.handlers.tokens import get_used_tokens, MAX_TOKENS

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


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
        prompt_continuation=" | ",
        completer=get_completer(),
    )


def _read_input(prompt: str) -> str:
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


async def _read_line(prompt: str) -> str:
    """Run the blocking prompt_toolkit call in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _read_input, prompt)


async def run_loop():
    print(r"""
            ██╗  ██╗ ██████╗  ██████╗
            ██║  ██║ ██╔══██╗ ██╔══██╗
            ███████║ ██████╔╝ ██║  ██║
            ██╔══██║ ██╔═══╝  ██║  ██║
            ██║  ██║ ██║      ██████╔╝
            ╚═╝  ╚═╝ ╚═╝      ╚═════╝
    """)
    print("  Ctrl+Enter for new line, Enter to submit, Ctrl+C to cancel.\n")
    print("  Type /exit to quit.\n")

    agent = QueryAgent()
    get_completer().set_agent(agent)

    while True:
        try:
            query = await _read_line(f"[{agent._current_session}] > ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.startswith("/"):
            if handle_command(query, agent):
                break
            continue

        print()

        tokens = get_used_tokens(agent)
        if tokens > MAX_TOKENS:
            print("Token limit reached. Please create a new conversation or run /summary to summarize the conversation.")
            continue
        result = await agent.ainvoke(query, agent._current_session)

        if result.get("synthesis_prompt"):
            print("\n─── 最终回答 ───\n")
            llm = get_llm(stream=True)
            streamed_answer = ""
            async for chunk in llm.astream(result["synthesis_prompt"]):
                t = getattr(chunk, "content", "") or ""
                if t:
                    streamed_answer += t
                    print(t, end="", flush=True)
            agent.store_streamed_answer(streamed_answer)

        print()


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
