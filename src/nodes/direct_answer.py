from src.cli import get_renderer
from src.core.models import TaskOutput
from src.core.state import AgentState
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import invoke_with_tools
from src.tools import tool_list


def _build_direct_prompt(state: AgentState) -> str:
    """Build the LLM prompt for a direct-answer call."""
    history = state.get("conversation_history")
    history_text = history.to_summary() if history else ""

    if history_text:
        return (
            f"【对话历史】\n{history_text}\n\n"
            f"【当前问题】\n{state['input']}\n\n"
            "【重要规则】\n"
            "1. 如果问题需要获取实时信息或执行操作，你必须调用可用工具，不允许自己输出命令或猜测结果。\n"
            "2. 请基于对话历史，直接、简洁地回答用户的问题。如果问题是对之前话题的追问，"
            "请结合历史上下文作答。如果不知道答案，请诚实说明。"
        )
    else:
        return (
            f"【用户问题】\n{state['input']}\n\n"
            "【重要规则】\n"
            "1. 如果问题需要获取实时信息或执行操作，你必须调用可用工具，不允许自己输出命令或猜测结果。\n"
            "2. 请直接、简洁地回答。如果不知道答案，请诚实说明。"
        )


async def direct_answer(state: AgentState) -> AgentState:
    """Handle simple tasks: ainvoke with tool-calling support.

    Uses the shared invoke_with_tools() path so simple mode and sub-task mode
    share the same tool budget, confirmation, and budget-exhaustion behavior.
    """
    renderer = get_renderer()
    renderer.blank()
    tracer = get_tracer()
    parent_id = state.get("parent_span_id") or None

    with tracer.span("direct_answer", parent_id=parent_id) as span_id:
        prompt = _build_direct_prompt(state)
        content, tool_results = await invoke_with_tools(prompt, tools=tool_list)
        if content:
            renderer.stream_answer(content)

        # Record span tokens (all LLM calls were tracked by the monkey-patch)
        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

    output = TaskOutput(
        node="direct_answer",
        result={"content": content, "tool_calls": tool_results},
    )

    return {
        "outputs": [*state.get("outputs", []), output],
        "final_response": content,
        "parent_span_id": span_id,
    }
