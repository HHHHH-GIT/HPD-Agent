"""Rewriter node: generates N diverse prompt angles for a sub-task.

This is a pure business-logic node — no graph coupling, no state mutation.
It takes a sub-task and produces N different angles/approaches that can be
used to generate diverse candidate answers for evaluation.
"""

from src.core.models import RewriteResult
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_structured_llm, REWRITE_PROMPT


async def rewrite_prompt(
    task_id: int,
    task_name: str,
    context: str,
    n: int = 3,
) -> RewriteResult:
    """Generate N diverse prompt angles for a sub-task.

    Args:
        task_id:   Sub-task ID from the DAG.
        task_name: Human-readable sub-task name.
        context:   The original user query (shared background context).
        n:         Number of angles to generate (default 3).

    Returns:
        RewriteResult with N angle descriptions.
    """
    tracer = get_tracer()
    with tracer.span(f"rewriter[#{task_id}]") as span_id:
        prompt = REWRITE_PROMPT.format(
            n=n,
            task_id=task_id,
            task_name=task_name,
            context=context[:2000],
        )

        llm = get_structured_llm(RewriteResult)
        result: RewriteResult = await llm.ainvoke(prompt)

        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

        return result
