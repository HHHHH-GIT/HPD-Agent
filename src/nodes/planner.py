"""Task planner: decomposes a complex query into a DAG and orchestrates execution."""

from src.core.models import PlannerResult
from src.core.state import AgentState
from src.llm import get_structured_llm, PLANNER_PROMPT
from src.nodes import executor, scheduler


async def planner(state: AgentState) -> AgentState:
    """Decompose a complex task into a DAG and execute sub-tasks in parallel.

    Responsibilities:
      1. LLM-based task decomposition (planning)
      2. Delegate execution to the scheduler (scheduling + execution)
      3. Build synthesis prompt for the final answer
    """

    # ── Step 1: Decompose into DAG ─────────────────────────────────────────────
    classifier = get_structured_llm(PlannerResult)
    result: PlannerResult = await classifier.ainvoke(
        PLANNER_PROMPT.format(query=state["input"])
    )
    tasks = result.sub_tasks
    total = len(tasks)

    print(f"\n[Planner] 任务已拆解，共 {total} 个子任务:")
    for t in tasks:
        deps = ", ".join(str(d) for d in t.depends) if t.depends else "无"
        print(f"  [{t.id}] {t.name}  ← 依赖: {deps}")

    # ── Step 2: Execute via scheduler ────────────────────────────────────────────
    statuses, done = await scheduler.run_all(
        tasks=tasks,
        executor=executor,
        context=state["input"],
    )

    # ── Step 3: Build synthesis prompt ───────────────────────────────────────────
    sub_task_section = "\n".join(
        f"### 子任务 {o.id}: {o.name}\n"
        f"{'[专家模式] ' if o.expert_mode else ''}"
        f"{o.detail}\n结论: {o.summary}"
        for o in sorted(done, key=lambda x: x.id)
    )
    synthesis_prompt = (
        f"【用户原始问题】\n{state['input']}\n\n"
        f"【所有子任务执行结果】\n{sub_task_section}\n\n"
        "请基于以上子任务结果，用流畅自然的语言给出完整、专业的回答。"
        "直接输出回答内容，无需额外说明。"
    )

    return {
        "tasks": tasks,
        "sub_task_statuses": statuses,
        "sub_task_outputs": done,
        "synthesis_prompt": synthesis_prompt,
        "outputs": [
            *state.get("outputs", []),
            {
                "node": "planner",
                "result": {
                    "total_sub_task_count": result.total_sub_task_count,
                    "sub_tasks": [
                        {"id": t.id, "name": t.name, "depends": t.depends}
                        for t in tasks
                    ],
                },
            },
        ],
    }
