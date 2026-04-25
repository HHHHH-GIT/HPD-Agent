from src.core.models import TaskOutput
from src.core.state import AgentState


async def synthesizer(state: AgentState) -> AgentState:
    """Build the synthesis prompt that combines all sub-task results.

    Reads ``sub_task_outputs``, ``input``, and ``conversation_history`` from state;
    writes ``synthesis_prompt`` for streaming output downstream.
    """
    done = state.get("sub_task_outputs", [])

    sub_task_section = "\n".join(
        f"### 子任务 {o.id}: {o.name}\n"
        f"{'[专家模式] ' if o.expert_mode else ''}"
        f"{o.detail}\n结论: {o.summary}"
        for o in sorted(done, key=lambda x: x.id)
    )

    history = state.get("conversation_history")
    history_section = ""
    if history:
        history_text = history.to_summary()
        if history_text:
            history_section = f"【对话历史】\n{history_text}\n\n"

    synthesis_prompt = (
        f"{history_section}"
        f"【用户原始问题】\n{state['input']}\n\n"
        f"【所有子任务执行结果】\n{sub_task_section}\n\n"
        "请基于以上信息，用流畅自然的语言给出完整、专业的回答。"
        "直接输出回答内容，无需额外说明。"
    )

    output = TaskOutput(
        node="synthesizer",
        result={
            "sub_task_count": len(done),
            "expert_mode_count": sum(1 for o in done if o.expert_mode),
            "failed_count": sum(1 for o in done if o.summary.startswith("[失败]")),
        },
    )

    return {
        "synthesis_prompt": synthesis_prompt,
        "outputs": [*state.get("outputs", []), output],
    }
