"""Handler for the /tokens command — focuses on context window occupancy.

The headline number is the resident context that will be injected into the
next turn before the user's next query is added. This is the closest analogue
to Codex/Claude Code's "Context window used" indicator.

Secondary rows show rough next-call estimates so users can tell whether a new
query is likely to overflow even when the resident context itself still fits.
"""

import concurrent.futures
import json

from src.agents import QueryAgent
from src.cli import get_renderer
from src.llm import ASSESSMENT_PROMPT
from src.memory.context import ConversationContext
from src.system_info import build_boot_prompt


# cl100k_base context window for the model (conservative: 128k for deepseek-v4)
MAX_TOKENS = 128_000


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


def _count_tool_schema_tokens() -> int:
    """Count tokens consumed by tool schemas from bind_tools().

    These are injected into every tool-bound API request but not tracked
    by the conversation context.
    """
    from src.tools import tool_list
    from langchain_core.utils.function_calling import convert_to_openai_function

    enc = _get_encoder()
    total = 0
    for t in tool_list:
        fn = convert_to_openai_function(t)
        total += len(enc.encode(json.dumps(fn, ensure_ascii=False)))
    return total


def _count_context_tokens(ctx: ConversationContext) -> int:
    """Count resident tokens for the history block as actually injected."""
    return _count_tokens(_build_history_section(ctx))


def _build_history_text(ctx: ConversationContext) -> str:
    return ctx.to_summary()


def _build_history_section(ctx: ConversationContext) -> str:
    history_text = _build_history_text(ctx)
    return f"【对话历史】\n{history_text}\n\n" if history_text else ""


def _build_history_section_for_agent(agent: QueryAgent) -> str:
    ctx = agent._get_context()
    history_text = _build_history_text(ctx)
    sid = getattr(agent, "_current_session", "default")
    if sid not in getattr(agent, "_session_boot_done", set()):
        boot_line = f"助手: {build_boot_prompt()}"
        history_text = f"{boot_line}\n{history_text}" if history_text else boot_line
    return f"【对话历史】\n{history_text}\n\n" if history_text else ""


def _build_direct_answer_prompt(history_section: str, query: str) -> str:
    if history_section:
        return (
            f"{history_section}"
            f"【当前问题】\n{query}\n\n"
            "【重要规则】\n"
            "1. 如果问题需要获取实时信息或执行操作，你必须调用可用工具，不允许自己输出命令或猜测结果。\n"
            "2. 请基于对话历史，直接、简洁地回答用户的问题。如果问题是对之前话题的追问，"
            "请结合历史上下文作答。如果不知道答案，请诚实说明。"
        )
    return (
        f"【用户问题】\n{query}\n\n"
        "【重要规则】\n"
        "1. 如果问题需要获取实时信息或执行操作，你必须调用可用工具，不允许自己输出命令或猜测结果。\n"
        "2. 请直接、简洁地回答。如果不知道答案，请诚实说明。"
    )


def get_resident_context_tokens(agent: QueryAgent) -> int:
    """Return the current resident context occupying the next turn's window."""
    return _count_tokens(_build_history_section_for_agent(agent))


def estimate_next_request_tokens(agent: QueryAgent, query: str) -> int:
    """Conservative estimate for the next request before it is sent."""
    history_section = _build_history_section_for_agent(agent)
    assessment_prompt = ASSESSMENT_PROMPT.format(
        history_section=history_section,
        query=query,
    )
    direct_prompt = _build_direct_answer_prompt(history_section, query)
    return max(
        _count_tokens(assessment_prompt),
        _count_tokens(direct_prompt) + _count_tool_schema_tokens(),
    )


def get_used_tokens(agent: QueryAgent) -> int:
    """Backward-compatible alias for resident context usage."""
    return get_resident_context_tokens(agent)


def _format_bar(pct: float, width: int = 20) -> str:
    """Render a simple text progress bar."""
    filled = int(pct * width)
    return "[" + "=" * filled + " " * (width - filled) + "]"


def run(raw: str, agent: QueryAgent) -> bool:
    """Handle /tokens command — print context window token usage."""
    ctx = agent._get_context()

    resident_tokens = get_resident_context_tokens(agent)
    left_tokens = max(MAX_TOKENS - resident_tokens, 0)

    # Count analysis cache kept only for /summary and bookkeeping
    sub_task_tokens = 0
    for output in ctx.sub_task_outputs:
        sub_task_tokens += _count_tokens(
            f"[子任务 {output['id']}: {output['name']}]\n{output['detail']}"
        )

    # Count tool schemas (bind_tools overhead, sent with every tool-bound call)
    tool_schema_tokens = _count_tool_schema_tokens()

    # Count tool summaries (informational — already included in messages)
    tool_tokens = 0
    for msg in ctx.messages:
        if msg.tool_summary:
            tool_tokens += _count_tokens(msg.tool_summary)

    history_section = _build_history_section(ctx)
    query_placeholder = "<next user query>"
    next_simple_estimate = _count_tokens(
        _build_direct_answer_prompt(history_section, query_placeholder)
    ) + tool_schema_tokens
    next_assessment_estimate = _count_tokens(
        ASSESSMENT_PROMPT.format(
            history_section=history_section,
            query=query_placeholder,
        )
    )

    pct = min(resident_tokens / MAX_TOKENS, 1.0)
    get_renderer().render_tokens(
        resident_tokens=resident_tokens,
        left_tokens=left_tokens,
        max_tokens=MAX_TOKENS,
        pct=pct,
        next_simple_estimate=next_simple_estimate,
        next_assessment_estimate=next_assessment_estimate,
        tool_schema_tokens=tool_schema_tokens,
        sub_task_tokens=sub_task_tokens,
        tool_tokens=tool_tokens,
    )

    if resident_tokens >= MAX_TOKENS:
        get_renderer().warning("上下文窗口已满，请开启新对话或使用 /summary 总结上下文")

    return False
