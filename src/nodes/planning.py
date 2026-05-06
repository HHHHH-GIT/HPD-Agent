"""Planning node: LLM-based DAG decomposition for complex tasks.

This is a pure business-logic node — no graph coupling, no state mutation.
It handles:
  - LLM-based task decomposition
  - Cycle detection with retry (up to 3 attempts)
  - Structured output (PlannerResult + SubTask list)

The coordinator agent calls this to make the "plan" decision.
"""

import re

from src.core.models import PlannerResult, RewriteResult, SubTask, SubTaskOutput
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_structured_llm, PLANNER_PROMPT, REPLAN_PROMPT, REWRITE_PROMPT
from src.nodes.scheduler import check_circle

_VARIANT_KEYWORDS = (
    "解法",
    "方案",
    "思路",
    "角度",
    "策略",
    "方法",
    "approach",
    "solution",
    "method",
)
_COUNT_PATTERN = re.compile(
    r"([0-9]+|[一二两三四五六七八九十百]+)\s*(?:种|个|类|套)\s*(?:不同的?)?\s*"
    r"(?:解法|方案|思路|角度|策略|方法)"
)
_CN_NUMERAL = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


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
            tasks, expansion_reason = await _maybe_expand_parallel_variants(query, tasks)
            if expansion_reason:
                result.sub_tasks = tasks
                result.total_sub_task_count = len(tasks)
                if result.reasoning:
                    result.reasoning = f"{result.reasoning} {expansion_reason}"
                else:
                    result.reasoning = expansion_reason

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


async def _maybe_expand_parallel_variants(query: str, tasks: list[SubTask]) -> tuple[list[SubTask], str]:
    """Expand coarse single-task plans when the user explicitly wants multiple alternatives."""
    variant_count = _extract_parallel_variant_count(query)
    if variant_count is None or len(tasks) != 1:
        return tasks, ""

    print(f"[Planning] 检测到多方案请求，尝试展开为 {variant_count} 个并行子任务...")
    original_task = tasks[0]
    angles = await _generate_variant_angles(query, original_task.name, variant_count)
    expanded_tasks = _build_parallel_variant_tasks(original_task.name, variant_count, angles)
    reason = (
        f"检测到用户明确要求至少 {variant_count} 种不同方案，"
        f"已将粗粒度任务展开为 {len(expanded_tasks)} 个可并行执行的独立子任务。"
    )
    return expanded_tasks, reason


def _extract_parallel_variant_count(query: str) -> int | None:
    lowered = query.lower()
    if not any(keyword in query or keyword in lowered for keyword in _VARIANT_KEYWORDS):
        return None

    matches = _COUNT_PATTERN.findall(query)
    if not matches:
        return None

    counts = [_parse_number_token(token) for token in matches]
    counts = [count for count in counts if count and count >= 2]
    if not counts:
        return None
    return max(counts)


def _parse_number_token(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    return _parse_chinese_numeral(token)


def _parse_chinese_numeral(token: str) -> int | None:
    if token == "十":
        return 10
    if "百" in token:
        return None
    if "十" not in token:
        return _CN_NUMERAL.get(token)

    left, _, right = token.partition("十")
    tens = _CN_NUMERAL.get(left, 1 if left == "" else None)
    ones = _CN_NUMERAL.get(right, 0 if right == "" else None)
    if tens is None or ones is None:
        return None
    return tens * 10 + ones


async def _generate_variant_angles(query: str, task_name: str, variant_count: int) -> list[str]:
    """Use the rewriter prompt to generate distinct parallel solution angles."""
    prompt = REWRITE_PROMPT.format(
        n=variant_count,
        task_id=1,
        task_name=task_name,
        context=(
            f"用户原始问题：{query}\n"
            "当前目标：把一个过粗的总任务拆成多个彼此独立、互不依赖、可并行执行的分析子任务。"
            "每个角度最终都应对应一种不同的解法/方案，并尽量避免重复。"
        )[:2000],
    )

    try:
        llm = get_structured_llm(RewriteResult)
        result: RewriteResult = await llm.ainvoke(prompt)
        if result is None:
            return []
        return [angle.strip() for angle in result.angles if angle and angle.strip()]
    except Exception:
        return []


def _build_parallel_variant_tasks(task_name: str, variant_count: int, angles: list[str]) -> list[SubTask]:
    seen: set[str] = set()
    normalized_angles: list[str] = []

    for angle in angles:
        if angle in seen:
            continue
        seen.add(angle)
        normalized_angles.append(angle)
        if len(normalized_angles) == variant_count:
            break

    while len(normalized_angles) < variant_count:
        normalized_angles.append(f"第{len(normalized_angles) + 1}种独立方案")

    return [
        SubTask(
            id=index + 1,
            name=f"{task_name}（角度：{angle}）",
            depends=[],
        )
        for index, angle in enumerate(normalized_angles)
    ]
