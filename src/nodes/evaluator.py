"""Evaluator node: 1:1 quality scoring for a single sub-task result.

This is a pure business-logic node — no graph coupling, no state mutation.
It takes a single result's detail text and produces an independent quality score.
"""

from src.core.models import EvaluatorScore
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_structured_llm, EVALUATOR_PROMPT


async def evaluate_single(
    result_detail: str,
    task_name: str,
    context: str,
) -> EvaluatorScore:
    """Evaluate a single sub-task result independently (1:1 scoring).

    Args:
        result_detail: The full detail text of the result to evaluate.
        task_name:     Human-readable sub-task name.
        context:       The original user query (shared background context).

    Returns:
        EvaluatorScore with score (0.0-1.0), reasoning, and issues list.
    """
    tracer = get_tracer()
    with tracer.span("evaluator") as span_id:
        prompt = EVALUATOR_PROMPT.format(
            task_name=task_name,
            context=context[:2000],
            result_detail=result_detail[:3000],
        )

        llm = get_structured_llm(EvaluatorScore)
        result: EvaluatorScore = await llm.ainvoke(prompt)

        # Clamp score to valid range
        result.score = max(0.0, min(1.0, result.score))

        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

        return result
