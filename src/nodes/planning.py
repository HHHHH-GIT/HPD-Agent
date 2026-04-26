"""Planning node: LLM-based DAG decomposition for complex tasks.

This is a pure business-logic node — no graph coupling, no state mutation.
It handles:
  - LLM-based task decomposition
  - Cycle detection with retry (up to 3 attempts)
  - Structured output (PlannerResult + SubTask list)

The coordinator agent calls this to make the "plan" decision.
"""

from src.core.models import PlannerResult
from src.llm import get_structured_llm, PLANNER_PROMPT
from src.nodes.scheduler import check_circle


async def decompose(query: str) -> tuple[list, PlannerResult]:
    """Decompose a query into a DAG of sub-tasks.

    Retries up to 3 times on cycle detection before raising RuntimeError.

    Returns:
        Tuple of (tasks list, PlannerResult).
    """
    for attempt in range(1, 4):
        classifier = get_structured_llm(PlannerResult)
        result: PlannerResult = await classifier.ainvoke(
            PLANNER_PROMPT.format(query=query)
        )
        tasks = result.sub_tasks

        if check_circle(tasks):
            print(f"[Planning] 尝试 {attempt}/3 检测到 DAG 循环，重新生成...")
            if attempt == 3:
                raise RuntimeError(
                    f"DAG 分解在 3 次尝试后仍产生循环图，请检查原始查询是否合理。\n"
                    f"子任务列表: {[(t.id, t.name, t.depends) for t in tasks]}"
                )
            continue

        _log_tasks(tasks)
        return list(tasks), result

    raise RuntimeError("Decomposer exhausted retry attempts without returning.")


def _log_tasks(tasks: list) -> None:
    print(f"\n[Planning] 任务已拆解，共 {len(tasks)} 个子任务:")
    for t in tasks:
        deps = ", ".join(str(d) for d in t.depends) if t.depends else "无"
        print(f"  [{t.id}] {t.name}  ← 依赖: {deps}")
