"""Reviewer node: LLM-based quality assessment for sub-task outputs.

This is a pure business-logic node — no round control, no decision mapping.
The reviewer agent (src/agents/reviewer_agent.py) handles those concerns.
"""

import re

from src.core.models import ReviewerDecision
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_structured_llm, REVIEW_PROMPT

_MAX_DETAIL_CHARS = 2000


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at a sentence boundary near max_chars."""
    if len(text) <= max_chars:
        return text
    # Try to cut at a sentence-ending punctuation near the limit
    cut = text[:max_chars]
    # Look for the last sentence boundary in the cut range
    match = re.search(r"[。！？.!?]\s*", cut[::-1])
    if match:
        # match.start() is the position from the end
        boundary = max_chars - match.start()
        if boundary > max_chars * 0.6:  # Don't cut too early
            return cut[:boundary] + "\n…（已截断）"
    # Fallback: hard cut with marker
    return cut + "…（已截断）"


def _format_sub_task_results(outputs) -> str:
    lines = []
    for o in outputs:
        status = "失败" if o.summary.startswith("[失败]") else "成功"
        detail_preview = _truncate_at_sentence(o.detail, _MAX_DETAIL_CHARS)
        lines.append(
            f"子任务 {o.id} ({o.name}) — {status}\n"
            f"  摘要: {o.summary}\n"
            f"  详情: {detail_preview}"
        )
    return "\n\n".join(lines)


async def review(query: str, outputs: list, current_round: int, max_rounds: int) -> ReviewerDecision:
    """Call the LLM to evaluate sub-task quality.

    Args:
        query: The original user question.
        outputs: List of SubTaskOutput objects to evaluate.
        current_round: Current review round (0-based).
        max_rounds: Maximum allowed review rounds.

    Returns:
        ReviewerDecision with overall_quality, task_reviews, re_execute_ids, etc.
    """
    tracer = get_tracer()
    with tracer.span("reviewer_llm") as span_id:
        results_text = _format_sub_task_results(outputs)
        prompt = REVIEW_PROMPT.format(
            query=query,
            sub_task_results=results_text,
            round=current_round + 1,
            max_rounds=max_rounds,
        )

        llm = get_structured_llm(ReviewerDecision)
        decision: ReviewerDecision = await llm.ainvoke(prompt)

        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

        return decision
