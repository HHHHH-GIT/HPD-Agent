"""Sub-task executor: assesses difficulty then executes a single sub-task."""

import re

from src.core.enums import SubTaskDifficulty
from src.core.models import SubTaskAssessmentResult, SubTaskOutput
from src.llm import (
    get_llm,
    get_structured_llm,
    SUB_TASK_ASSESSMENT_PROMPT,
    SUB_TASK_PROMPT,
)


def _extract_summary(detail: str) -> str:
    """Extract concise summary from LLM output, handling JSON-fragment edge cases."""
    import json

    try:
        parsed = json.loads(detail.strip())
        if isinstance(parsed, dict) and parsed.get("summary"):
            s = parsed["summary"].strip()
            if s:
                return s
    except (json.JSONDecodeError, TypeError):
        pass

    sentences = re.split(r"(?<=[。！？.!?\n])", detail)
    candidates = [s.strip() for s in sentences if 5 < len(s.strip()) < 120]
    return candidates[-1] if candidates else detail[:80].strip()


async def run(task_id: int, task_name: str, context: str) -> SubTaskOutput:
    """Assess difficulty and execute a single sub-task.

    Args:
        task_id:  Sub-task ID from the DAG.
        task_name: Human-readable sub-task name.
        context:  The original user query (shared background context).

    Returns:
        SubTaskOutput with detail, summary, and expert_mode flag.
    """
    # ── Difficulty assessment ─────────────────────────────────────────────────
    classifier = get_structured_llm(SubTaskAssessmentResult)
    assessment: SubTaskAssessmentResult = await classifier.ainvoke(
        SUB_TASK_ASSESSMENT_PROMPT.format(task_id=task_id, task_name=task_name)
    )

    is_expert = assessment.difficulty == SubTaskDifficulty.HARD
    if is_expert:
        print(
            f"\n[DEBUG] 子任务 {task_id} ({task_name}) 是困难子任务，后续调用专家模式"
        )

    # ── Execution ─────────────────────────────────────────────────────────────
    llm = get_llm()
    prompt = SUB_TASK_PROMPT.format(
        context=context, task_id=task_id, task_name=task_name
    )
    response = await llm.ainvoke(prompt)
    
    content = getattr(response, "content", "") or str(response)

    return SubTaskOutput(
        id=task_id,
        name=task_name,
        detail=content,
        summary=_extract_summary(content),
        expert_mode=is_expert,
    )
