from src.core.models import TaskOutput
from src.core.state import AgentState
from src.llm import get_llm, DIRECT_ANSWER_PROMPT


async def direct_answer(state: AgentState) -> AgentState:
    """Handle simple tasks: stream response chunks directly to stdout."""
    print()

    history = state.get("conversation_history")
    history_text = history.to_summary() if history else ""

    if history_text:
        prompt = (
            f"【对话历史】\n{history_text}\n\n"
            f"【当前问题】\n{state['input']}\n\n"
            "请基于对话历史，直接、简洁地回答用户的问题。如果问题是对之前话题的追问，"
            "请结合历史上下文作答。如果不知道答案，请诚实说明。"
        )
    else:
        prompt = (
            f"【用户问题】\n{state['input']}\n\n"
            "请直接、简洁地回答。如果不知道答案，请诚实说明。"
        )

    llm = get_llm(stream=True)
    full_content = ""

    async for chunk in llm.astream(prompt):
        t = getattr(chunk, "content", "") or ""
        if t:
            print(t, end="", flush=True)
            full_content += t

    output = TaskOutput(
        node="direct_answer",
        result={"content": full_content},
    )

    return {
        "outputs": [*state.get("outputs", []), output],
        "final_response": full_content,
    }
