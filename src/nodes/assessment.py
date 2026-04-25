from src.core.models import AssessmentResult, TaskOutput
from src.core.state import AgentState
from src.llm import get_structured_llm, ASSESSMENT_PROMPT


async def first_level_assessment(state: AgentState) -> AgentState:
    """Classify the user's query into simple | complex and write analysis to state."""
    classifier = get_structured_llm(AssessmentResult)
    prompt = ASSESSMENT_PROMPT.format(query=state["input"])
    result: AssessmentResult = await classifier.ainvoke(prompt)

    output = TaskOutput(
        node="first_level_assessment",
        result={
            "difficulty": result.difficulty.value,
            "reasoning": result.reasoning,
        },
    )

    return {
        "analysis": result.difficulty,
        "outputs": [*state.get("outputs", []), output],
    }
