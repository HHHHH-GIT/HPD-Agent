"""Shared prompt preview and context budget helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.llm import ASSESSMENT_PROMPT, PLANNER_PROMPT
from src.memory.context import ConversationContext
from src.system_info import build_boot_prompt

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class PromptPreview:
    """Token breakdown for the next prompt path."""

    resident_history_tokens: int
    task_summary_tokens: int
    toolchain_summary_tokens: int
    tool_schema_tokens: int
    next_assessment_tokens: int
    next_direct_tokens: int
    next_planner_tokens: int
    synthesis_estimate_tokens: int
    nonresident_artifact_tokens: int
    next_request_tokens: int
    max_tokens: int

    @property
    def resident_tokens(self) -> int:
        return self.resident_history_tokens

    @property
    def usage_ratio(self) -> float:
        if self.max_tokens <= 0:
            return 0.0
        return self.next_request_tokens / self.max_tokens


def build_prompt_preview(
    ctx: ConversationContext,
    *,
    query: str,
    force_complex: bool,
    token_counter: TokenCounter,
    max_tokens: int = 128_000,
    tool_schema_tokens: int = 0,
    include_boot_prompt: bool = False,
) -> PromptPreview:
    """Build the shared token preview used by /tokens and request gating."""
    history_section = build_history_section(ctx, include_boot_prompt=include_boot_prompt)
    resident_history_tokens = token_counter(history_section)
    task_summary_tokens = token_counter(ctx.session_summary)
    toolchain_summary_tokens = token_counter(ctx.toolchain_summary)

    assessment_prompt = ASSESSMENT_PROMPT.format(
        history_section=history_section,
        query=query,
    )
    direct_prompt = _build_direct_answer_prompt(history_section, query)
    planner_prompt = PLANNER_PROMPT.format(query=query)
    synthesis_estimate = _estimate_synthesis_tokens(ctx, token_counter)

    next_assessment_tokens = token_counter(assessment_prompt)
    next_direct_tokens = token_counter(direct_prompt) + tool_schema_tokens
    next_planner_tokens = token_counter(planner_prompt) + synthesis_estimate
    next_request_tokens = (
        next_planner_tokens
        if force_complex
        else max(next_assessment_tokens, next_direct_tokens)
    )

    return PromptPreview(
        resident_history_tokens=resident_history_tokens,
        task_summary_tokens=task_summary_tokens,
        toolchain_summary_tokens=toolchain_summary_tokens,
        tool_schema_tokens=tool_schema_tokens,
        next_assessment_tokens=next_assessment_tokens,
        next_direct_tokens=next_direct_tokens,
        next_planner_tokens=next_planner_tokens,
        synthesis_estimate_tokens=synthesis_estimate,
        nonresident_artifact_tokens=ctx.artifact_total_tokens,
        next_request_tokens=next_request_tokens,
        max_tokens=max_tokens,
    )


def build_history_section(ctx: ConversationContext, *, include_boot_prompt: bool = False) -> str:
    """Render resident history exactly as the agent injects it."""
    history_text = ctx.to_summary()
    if include_boot_prompt:
        boot_line = f"助手: {build_boot_prompt()}"
        history_text = f"{boot_line}\n{history_text}" if history_text else boot_line
    return f"【对话历史】\n{history_text}\n\n" if history_text else ""


def should_compact(preview: PromptPreview, *, precompact_ratio: float = 0.70) -> bool:
    """Return True when the next request is over the pre-compaction threshold."""
    if preview.max_tokens <= 0:
        return False
    return preview.next_request_tokens >= int(preview.max_tokens * precompact_ratio)


def _build_direct_answer_prompt(history_section: str, query: str) -> str:
    if history_section:
        return (
            f"{history_section}"
            f"【当前问题】\n{query}\n\n"
            "【重要规则】\n"
            "1. 如果问题需要获取实时信息或执行操作，必须调用可用工具。\n"
            "2. 基于对话历史直接、简洁地回答。"
        )
    return (
        f"【用户问题】\n{query}\n\n"
        "【重要规则】\n"
        "1. 如果问题需要获取实时信息或执行操作，必须调用可用工具。\n"
        "2. 直接、简洁地回答。"
    )


def _estimate_synthesis_tokens(ctx: ConversationContext, token_counter: TokenCounter) -> int:
    sub_task_tokens = token_counter(ctx.to_sub_tasks_summary())
    if not sub_task_tokens:
        return 0
    # Synthesis carries resident history plus a compact task-output summary.
    return token_counter(ctx.to_summary()) + sub_task_tokens
