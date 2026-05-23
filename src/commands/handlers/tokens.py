"""Handler for /tokens and shared prompt-size estimation."""

from __future__ import annotations

import concurrent.futures
import json

from src.agents import QueryAgent
from src.agents.query_agent import parse_query_control_flags
from src.cli import get_renderer
from src.memory.budget import build_history_section, build_prompt_preview
from src.memory.context import ConversationContext
from src.llm.tool_context import load_tool_context_config


MAX_TOKENS = 128_000


def _load_encoder():
    """Run in a background thread because tiktoken can block on first use."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_encoder_future = _executor.submit(_load_encoder)


def _get_encoder():
    return _encoder_future.result()


def _count_tokens(text: str) -> int:
    enc = _get_encoder()
    return len(enc.encode(text))


def _count_tool_schema_tokens() -> int:
    """Count bind_tools schema overhead included in tool-enabled calls."""
    from langchain_core.utils.function_calling import convert_to_openai_function

    from src.tools import tool_list

    enc = _get_encoder()
    total = 0
    for tool in tool_list:
        fn = convert_to_openai_function(tool)
        total += len(enc.encode(json.dumps(fn, ensure_ascii=False)))
    return total


def get_model_context_window() -> int:
    """Return active model context window, defaulting conservatively to 128k."""
    try:
        from src.models import get_store

        profile = get_store().active_profile()
        extra_body = getattr(profile, "extra_body", {}) if profile else {}
        if isinstance(extra_body, dict):
            for key in ("context_window", "max_context_tokens", "max_input_tokens"):
                value = extra_body.get(key)
                if value:
                    return int(value)
    except Exception:
        pass
    return MAX_TOKENS


def _build_history_text(ctx: ConversationContext) -> str:
    return ctx.to_summary()


def _build_history_section(ctx: ConversationContext) -> str:
    return build_history_section(ctx, include_boot_prompt=False)


def _build_history_section_for_agent(agent: QueryAgent) -> str:
    ctx = agent._get_context()
    sid = getattr(agent, "_current_session", "default")
    include_boot = sid not in getattr(agent, "_session_boot_done", set())
    return build_history_section(ctx, include_boot_prompt=include_boot)


def _count_context_tokens(ctx: ConversationContext) -> int:
    return _count_tokens(_build_history_section(ctx))


def get_resident_context_tokens(agent: QueryAgent) -> int:
    return _count_tokens(_build_history_section_for_agent(agent))


def estimate_next_request_tokens(agent: QueryAgent, query: str) -> int:
    """Conservative estimate for the next request before it is sent."""
    normalized_query, force_complex = parse_query_control_flags(query)
    ctx = agent._get_context()
    sid = getattr(agent, "_current_session", "default")
    preview = build_prompt_preview(
        ctx,
        query=normalized_query,
        force_complex=force_complex,
        token_counter=_count_tokens,
        max_tokens=get_model_context_window(),
        tool_schema_tokens=_count_tool_schema_tokens(),
        include_boot_prompt=sid not in getattr(agent, "_session_boot_done", set()),
    )
    return preview.next_request_tokens


def get_used_tokens(agent: QueryAgent) -> int:
    return get_resident_context_tokens(agent)


def run(raw: str, agent: QueryAgent) -> bool:
    """Handle /tokens: print resident context and next-call estimates."""
    ctx = agent._get_context()
    sid = getattr(agent, "_current_session", "default")
    max_tokens = get_model_context_window()
    tool_schema_tokens = _count_tool_schema_tokens()
    tool_context_config = load_tool_context_config()
    preview = build_prompt_preview(
        ctx,
        query="<next user query>",
        force_complex=False,
        token_counter=_count_tokens,
        max_tokens=max_tokens,
        tool_schema_tokens=tool_schema_tokens,
        include_boot_prompt=sid not in getattr(agent, "_session_boot_done", set()),
    )

    resident_tokens = preview.resident_history_tokens
    left_tokens = max(max_tokens - resident_tokens, 0)
    sub_task_tokens = _count_tokens(ctx.to_sub_tasks_summary())

    tool_tokens = 0
    for msg in ctx.messages:
        if msg.tool_summary:
            tool_tokens += _count_tokens(msg.tool_summary)

    get_renderer().render_tokens(
        resident_tokens=resident_tokens,
        left_tokens=left_tokens,
        max_tokens=max_tokens,
        pct=min(resident_tokens / max_tokens, 1.0),
        next_simple_estimate=preview.next_direct_tokens,
        next_assessment_estimate=preview.next_assessment_tokens,
        tool_schema_tokens=tool_schema_tokens,
        sub_task_tokens=sub_task_tokens,
        tool_tokens=tool_tokens,
        preview_breakdown={
            "Task summary": preview.task_summary_tokens,
            "Toolchain summary": preview.toolchain_summary_tokens,
            "Next planner": preview.next_planner_tokens,
            "Synthesis estimate": preview.synthesis_estimate_tokens,
            "Tool result inline limit": tool_context_config.max_inline_tool_result_tokens,
            "Artifacts (non-resident)": preview.nonresident_artifact_tokens,
        },
    )

    if resident_tokens >= max_tokens:
        get_renderer().warning(
            "Context window is full. Start a new conversation or run /summary."
        )

    return False
