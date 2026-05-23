from src.core.enums import TaskDifficulty
from src.core.models import AssessmentResult, TaskOutput
from src.core.state import AgentState
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_structured_llm, ASSESSMENT_PROMPT


async def first_level_assessment(state: AgentState) -> AgentState:
    """Classify the user's query into simple | complex and write analysis to state.

    Includes conversation history for context-aware classification,
    and retries once on failure before falling back to SIMPLE.
    """
    tracer = get_tracer()
    with tracer.span("first_level_assessment") as span_id:
        if state.get("force_complex"):
            print("[Assessment] 检测到末尾 &, 本轮请求强制进入复杂任务路由")
            result = AssessmentResult(
                difficulty=TaskDifficulty.COMPLEX,
                reasoning="User forced complex routing via trailing '&'.",
            )
        else:
            # Build history section (same pattern as direct_answer)
            history = state.get("conversation_history")
            history_text = history.to_summary() if history else ""
            history_section = f"【对话历史】\n{history_text}\n\n" if history_text else ""

            prompt = ASSESSMENT_PROMPT.format(
                history_section=history_section,
                query=state["input"],
            )

            result = await _classify_with_retry(prompt)

        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

    output = TaskOutput(
        node="first_level_assessment",
        result={
            "difficulty": result.difficulty.value,
            "forced": bool(state.get("force_complex")),
        },
    )

    return {
        "analysis": result.difficulty,
        "outputs": [*state.get("outputs", []), output],
        "parent_span_id": span_id,
    }


async def _classify_with_retry(prompt: str) -> AssessmentResult:
    """Call the assessment LLM with one retry, falling back to SIMPLE on double failure."""
    classifier = get_structured_llm(AssessmentResult)

    for attempt in range(2):
        try:
            return await classifier.ainvoke(prompt)
        except Exception as exc:
            if attempt == 0:
                print(f"[Assessment] 分类失败，重试中... ({exc})")
                continue
            print(f"[Assessment] 分类两次失败，降级为 simple: {exc}")
            return AssessmentResult(
                difficulty=TaskDifficulty.SIMPLE,
                reasoning=f"分类失败，降级为 simple: {exc}",
            )
