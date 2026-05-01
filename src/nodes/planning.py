"""Planning node: LLM-based DAG decomposition for complex tasks.

This is a pure business-logic node — no graph coupling, no state mutation.
It handles:
  - LLM-based task decomposition
  - Cycle detection with retry (up to 3 attempts)
  - Structured output (PlannerResult + SubTask list)

The coordinator agent calls this to make the "plan" decision.
"""

from src.core.models import PlannerResult, SubTask, SubTaskOutput
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_structured_llm, PLANNER_PROMPT, REPLAN_PROMPT
from src.nodes.scheduler import check_circle


async def decompose(query: str) -> tuple[list, PlannerResult]:
    """Decompose a query into a DAG of sub-tasks.

    Retries up to 3 times on cycle detection before raising RuntimeError.

    Returns:
        Tuple of (tasks list, PlannerResult).
    """
    tracer = get_tracer()
    with tracer.span("decompose") as span_id:
        for attempt in range(1, 4):
            classifier = get_structured_llm(PlannerResult)
            result: PlannerResult = await classifier.ainvoke(
                PLANNER_PROMPT.format(query=query)
            )
            if result is None:
                print(f"[Planning] 尝试 {attempt}/3 LLM 返回空结果，重试...")
                if attempt == 3:
                    raise RuntimeError("LLM 在 3 次尝试后均未返回有效分解结果。")
                continue
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
            tin, tout, model = TokenTrackerCallback.snapshot()
            tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)
            return list(tasks), result

        raise RuntimeError("Decomposer exhausted retry attempts without returning.")


async def replan(
    query: str,
    existing_tasks: list[SubTask],
    existing_outputs: list[SubTaskOutput],
    feedback: str,
    suggestions: list[str],
    next_id: int,
) -> tuple[list[SubTask], PlannerResult]:
    """Add new sub-tasks based on reviewer feedback.

    Returns only the NEW sub-tasks (caller merges with existing).
    """
    tracer = get_tracer()
    with tracer.span("replan") as span_id:
        # Format existing task info
        existing_lines = []
        for t in existing_tasks:
            output = next((o for o in existing_outputs if o.id == t.id), None)
            summary = output.summary if output else "(未执行)"
            existing_lines.append(f"  [{t.id}] {t.name} → {summary}")
        existing_text = "\n".join(existing_lines)

        suggestions_text = "\n".join(f"  - {s}" for s in suggestions) if suggestions else "  （无）"

        prompt = REPLAN_PROMPT.format(
            query=query,
            existing_tasks=existing_text,
            feedback=feedback,
            suggestions=suggestions_text,
            next_id=next_id,
        )

        llm = get_structured_llm(PlannerResult)
        result: PlannerResult = await llm.ainvoke(prompt)
        if result is None:
            raise RuntimeError("Replan LLM 返回空结果。")
        new_tasks = result.sub_tasks

        # Cycle check on full graph
        all_tasks = existing_tasks + new_tasks
        if check_circle(all_tasks):
            print("[Replan] 检测到循环，尝试修复...")
            # Strip depends that reference non-existent IDs
            existing_ids = {t.id for t in existing_tasks}
            for t in new_tasks:
                t.depends = [d for d in t.depends if d in existing_ids or d in {nt.id for nt in new_tasks}]
            if check_circle(all_tasks):
                raise RuntimeError("Replan produced a cyclic DAG after fix attempt.")

        _log_tasks(new_tasks, prefix="[Replan]")
        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)
        return new_tasks, result


def _log_tasks(tasks: list, prefix: str = "[Planning]") -> None:
    print(f"\n{prefix} 任务已拆解，共 {len(tasks)} 个子任务:")
    for t in tasks:
        deps = ", ".join(str(d) for d in t.depends) if t.depends else "无"
        print(f"  [{t.id}] {t.name}  ← 依赖: {deps}")
