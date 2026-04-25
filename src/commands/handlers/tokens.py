"""Handler for the /tokens command — shows context window token usage."""

import concurrent.futures

from src.agents import QueryAgent
from src.memory.context import ConversationContext


MAX_TOKENS = 10_000


def _load_encoder():
    """Run in thread pool: download vocab + compile encoder."""
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


# tiktoken.get_encoding() blocks on first call (network + compilation).
# Warm it in a background thread pool so it never blocks the asyncio loop.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_encoder_future = _executor.submit(_load_encoder)


def _get_encoder():
    """Return the cached encoder, blocking only the background thread if not yet ready."""
    return _encoder_future.result()


def _count_tokens(text: str) -> int:
    """Return the number of tokens in a text string."""
    enc = _get_encoder()
    return len(enc.encode(text))


def _count_context_tokens(ctx: ConversationContext) -> int:
    """Count total tokens across all messages in the context."""
    enc = _get_encoder()
    total = 0
    for msg in ctx.messages:
        text = f"{msg.role.value}: {msg.content}"
        total += len(enc.encode(text))
    return total


def _format_bar(pct: float, width: int = 20) -> str:
    """Render a simple text progress bar."""
    filled = int(pct * width)
    return "[" + "=" * filled + " " * (width - filled) + "]"

def get_used_tokens(agent: QueryAgent) -> int:
    """Get the number of used tokens in the current session."""
    ctx = agent._get_context()
    return _count_context_tokens(ctx)

def run(raw: str, agent: QueryAgent) -> bool:
    """Handle /tokens command — print context window token usage."""
    ctx = agent._get_context()
    total = _count_context_tokens(ctx)

    pct = min(total / MAX_TOKENS, 1.0)
    bar = _format_bar(pct)

    print(f"=== Context Token Usage ===")
    print(f"  Used:      {total:>6} tokens")
    print(f"  Max:       {MAX_TOKENS:>6} tokens")
    print(f"  Usage:     {bar} {pct * 100:.1f}%")
    print()

    if total >= MAX_TOKENS:
        print("  [WARNING] 上下文窗口已满，请开启新对话")

    return False
