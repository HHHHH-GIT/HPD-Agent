"""Decomposer: decomposes a complex query into a DAG of sub-tasks."""

from src.core.models import PlannerResult
from src.core.state import AgentState
from src.llm import get_structured_llm, PLANNER_PROMPT
from src.nodes.scheduler import check_circle


async def decomposer(state: AgentState) -> AgentState:
    """Use the LLM to decompose a complex task into a structured DAG.

    Produces ``decomposition_result`` and ``tasks`` in state for downstream nodes.
    Retries decomposition on cycle-detected output (up to 3 attempts).
    """
    for attempt in range(1, 4):
        classifier = get_structured_llm(PlannerResult)
        result: PlannerResult = await classifier.ainvoke(
            PLANNER_PROMPT.format(query=state["input"])
        )
        tasks = result.sub_tasks

        if check_circle(tasks):
            print(
                f"[Decomposer] 尝试 {attempt}/3 检测到 DAG 循环，重新生成..."
            )
            if attempt == 3:
                raise RuntimeError(
                    f"DAG 分解在 3 次尝试后仍产生循环图，请检查原始查询是否合理。\n"
                    f"子任务列表: {[(t.id, t.name, t.depends) for t in tasks]}"
                )
            continue

        total = len(tasks)
        print(f"\n[Decomposer] 任务已拆解，共 {total} 个子任务:")
        for t in tasks:
            deps = ", ".join(str(d) for d in t.depends) if t.depends else "无"
            print(f"  [{t.id}] {t.name}  ← 依赖: {deps}")

        return {
            "decomposition_result": result,
            "tasks": tasks,
            "outputs": [
                *state.get("outputs", []),
                {
                    "node": "decomposer",
                    "result": {
                        "total_sub_task_count": result.total_sub_task_count,
                        "actual_count": total,
                        "sub_tasks": [
                            {"id": t.id, "name": t.name, "depends": t.depends}
                            for t in tasks
                        ],
                    },
                },
            ],
        }

    # unreachable — kept for static analysis
    raise RuntimeError("Decomposer exhausted retry attempts without returning.")
