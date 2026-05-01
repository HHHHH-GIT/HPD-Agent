"""Reflector node: analyzes evaluation issues and generates improved strategies.

This is a pure business-logic node — no graph coupling, no state mutation.
It takes a sub-task result's evaluation (score + issues) and produces
an improved prompt/strategy for the next execution attempt.
"""

from src.core.models import EvaluatorScore, ReflectionResult
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_structured_llm, REFLECTION_PROMPT


async def reflect(
    task_name: str,
    context: str,
    result_detail: str,
    score_result: EvaluatorScore,
) -> ReflectionResult:
    """Analyze evaluation issues and generate an improved execution strategy.

    Args:
        task_name:     Human-readable sub-task name.
        context:       The original user query (shared background context).
        result_detail: The full detail text of the current result.
        score_result:  The evaluator's score and issues for this result.

    Returns:
        ReflectionResult with improved_prompt, strategy, and reasoning.
    """
    tracer = get_tracer()
    with tracer.span("reflector") as span_id:
        issues_text = "\n".join(f"- {issue}" for issue in score_result.issues)
        if not issues_text:
            issues_text = "- 未列出具体问题"

        prompt = REFLECTION_PROMPT.format(
            task_name=task_name,
            context=context[:2000],
            result_detail=result_detail[:2000],
            issues=issues_text,
            score=f"{score_result.score:.2f}",
        )

        llm = get_structured_llm(ReflectionResult)
        result: ReflectionResult = await llm.ainvoke(prompt)

        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

        return result
