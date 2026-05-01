"""DAG scheduler: Kahn's-algorithm executor with retry, cycle detection, and progress tracking."""

import asyncio
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.core.models import SubTask, SubTaskOutput


_print_lock = threading.Lock()

_CHARS_PER_TOKEN = 3.5


@dataclass
class RetryConfig:
    """Configuration for the per-task retry mechanism."""

    max_attempts: int = 3
    """Maximum number of execution attempts per sub-task."""

    base_delay: float = 1.0
    """Initial backoff delay in seconds (doubles after each retry)."""

    max_delay: float = 10.0
    """Cap the exponential delay at this value."""


@dataclass
class ContextConfig:
    """Configuration for how upstream results are passed to downstream tasks."""

    max_total_chars: int = 3000
    """Hard cap on total context length (chars) passed to any executor call."""

    max_tools_per_task: int = 20
    """Maximum number of tools_used entries to include."""

    max_findings_per_task: int = 15
    """Maximum number of key_findings entries to include."""


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base), falling back to char/3.5."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        return int(len(text) / _CHARS_PER_TOKEN)


def _build_dep_guidance(
    dep_ids: list[int],
    completed_cache: dict[int, dict],
    task_map: dict[int, SubTask],
    config: ContextConfig,
) -> tuple[list[str], list[str]]:
    """Extract tools_used and key_findings from transitive dependencies (deduplicated)."""
    seen_tools: set[str] = set()
    seen_findings: set[str] = set()
    all_tools: list[str] = []
    all_findings: list[str] = []

    queue: list[int] = list(dep_ids)
    visited: set[int] = set()

    while queue and len(all_tools) < config.max_tools_per_task:
        dep_id = queue.pop(0)
        if dep_id in visited or dep_id not in completed_cache:
            continue
        visited.add(dep_id)

        c = completed_cache[dep_id]

        for tool in (c.get("tools_used") or [])[: config.max_tools_per_task]:
            if tool not in seen_tools:
                seen_tools.add(tool)
                all_tools.append(tool)

        for finding in (c.get("key_findings") or [])[: config.max_findings_per_task]:
            if finding not in seen_findings:
                seen_findings.add(finding)
                all_findings.append(finding)

        for transitively_dep in task_map[dep_id].depends:
            if transitively_dep not in visited:
                queue.append(transitively_dep)

    return all_tools, all_findings


def _build_context(
    original_context: str,
    dep_block: str,
    tools_used: list[str],
    key_findings: list[str],
    config: ContextConfig,
) -> str:
    """Assemble a downstream task's full context within the token budget.

    Layout:
      1. Original question (always first, so LLM sees it without scrolling)
      2. Do-not-replay guidance block (only if tools or findings exist)
      3. Upstream narrative details — truncated to fit remaining budget
    """
    # No deps at all — return original context unchanged
    if not tools_used and not key_findings and not dep_block.strip():
        return original_context

    # Build guidance block
    guidance_lines = ["【上游任务已覆盖的内容】（请勿重复执行以下操作）"]
    if tools_used:
        guidance_lines.append(f"  已读取文件: {', '.join(tools_used)}")
    if key_findings:
        guidance_lines.append("  已知关键事实:")
        for f in key_findings[:10]:
            guidance_lines.append(f"    - {f}")
    if not tools_used and not key_findings:
        guidance_lines.append("  （暂无）")

    guidance_block = "\n".join(guidance_lines)

    # Reserve budget: guidance + original_context + overhead
    overhead_chars = 100
    budget_chars = config.max_total_chars - overhead_chars - len(guidance_block)
    budget_chars = max(budget_chars, 500)

    truncated_dep = dep_block[:budget_chars]
    if len(dep_block) > budget_chars:
        truncated_dep += "\n…（上游结果已被截断）"

    return (
        f"{original_context}\n\n"
        f"{guidance_block}\n\n"
        f"【上游任务详细结果】\n{truncated_dep}"
    )


async def run_all(
    tasks: list[SubTask],
    executor: Any,
    context: str,
    retry: RetryConfig = RetryConfig(),
    ctx_config: ContextConfig | None = None,
    execute_only: set[int] | None = None,
    existing_outputs: list[SubTaskOutput] | None = None,
) -> tuple[dict[int, str], list[SubTaskOutput]]:
    """Execute all sub-tasks respecting DAG dependencies (parallel within each topological layer).

    Args:
        tasks:     Ordered DAG sub-tasks from the planner.
        executor:  Async callable with signature ``(task_id, task_name, context)``.
                   Pass None to use the ExpertAgent from src.agents.expert_agent.
        context:  Passed through to each executor call.
        retry:    Retry configuration (max_attempts, base_delay, max_delay).
        ctx_config: Context configuration (max_total_chars, max_tools, max_findings).
                   Defaults to ContextConfig().
        execute_only: If set, only execute tasks with these IDs. All other tasks
                      are pre-populated from existing_outputs (skipped).
        existing_outputs: Previous run outputs. Used with execute_only to carry
                          forward results of tasks that don't need re-execution.

    Returns:
        A tuple of (statuses, outputs) where:
          - statuses maps task-id → "pending" | "running" | "done" | "failed"
          - outputs is a list of SubTaskOutput in completion order

    Raises:
        RuntimeError: if a cycle / deadlock is detected.
    """
    if ctx_config is None:
        ctx_config = ContextConfig()

    if executor is None:
        from src.agents.expert_agent import execute as expert_execute
        executor = expert_execute

    statuses: dict[int, str] = {t.id: "pending" for t in tasks}
    done: list[SubTaskOutput] = []
    in_degree: dict[int, int] = {t.id: len(t.depends) for t in tasks}
    running: set[int] = set()

    task_map: dict[int, SubTask] = {t.id: t for t in tasks}

    # Cache: detail, summary, tools_used, key_findings
    completed_cache: dict[int, dict] = {}
    # Audit trail: original question + one-line summary per completed task.
    accumulated_context = context

    # Pre-populate completed_cache and done from existing_outputs when filtering
    skip_ids: set[int] = set()
    if execute_only is not None and existing_outputs:
        for o in existing_outputs:
            if o.id not in execute_only and o.id in task_map:
                skip_ids.add(o.id)
                statuses[o.id] = "done"
                done.append(o)
                completed_cache[o.id] = {
                    "detail": o.detail,
                    "summary": o.summary,
                    "tools_used": o.tools_used,
                    "key_findings": o.key_findings,
                }
                in_degree[o.id] = -1
                accumulated_context += (
                    f"\n\n[子任务 {o.id} ({o.name})]\n{o.summary}"
                )
        # Decrement in-degree of dependents for skipped tasks
        for sid in skip_ids:
            for t in tasks:
                if sid in t.depends:
                    in_degree[t.id] -= 1

    total = len(tasks)
    target_count = len(execute_only) if execute_only is not None else total

    _print_progress(len(done), total)

    while len(done) < total:
        ready_ids = [
            i for i, d in in_degree.items()
            if d == 0 and i not in running and i not in skip_ids
        ]

        if not ready_ids:
            raise RuntimeError(f"Deadlock: {total - len(done)} tasks stuck.")

        ready_tasks = [t for t in tasks if t.id in ready_ids]
        running.update(ready_ids)

        # BFS transitive closure: every upstream result, exactly once.
        dep_results: list[str] = []
        for task in ready_tasks:
            if task.depends:
                seen: set[int] = set()
                queue: list[int] = list(task.depends)
                while queue:
                    dep_id = queue.pop(0)
                    if dep_id in seen or dep_id not in completed_cache:
                        continue
                    seen.add(dep_id)
                    c = completed_cache[dep_id]
                    dep_results.append(
                        f"[子任务 {dep_id} ({task_map[dep_id].name}) 结果]\n"
                        f"{c.get('detail', '')[:1500]}"
                    )
                    for transitively_dep in task_map[dep_id].depends:
                        if transitively_dep not in seen and transitively_dep not in queue:
                            queue.append(transitively_dep)

        dep_block = "\n\n".join(dep_results)

        # Collect structured guidance from all transitive deps (deduplicated globally)
        all_tools: list[str] = []
        all_findings: list[str] = []
        for task in ready_tasks:
            if task.depends:
                tools, findings = _build_dep_guidance(
                    task.depends, completed_cache, task_map, ctx_config
                )
                all_tools.extend(tools)
                all_findings.extend(findings)

        # Deduplicate across all ready tasks
        seen_t = set()
        unique_tools = [t for t in all_tools if not (t in seen_t or seen_t.add(t))]
        seen_f = set()
        unique_findings = [f for f in all_findings if not (f in seen_f or seen_f.add(f))]

        task_contexts: dict[int, str] = {}
        for task in ready_tasks:
            task_contexts[task.id] = _build_context(
                accumulated_context,
                dep_block,
                unique_tools,
                unique_findings,
                ctx_config,
            )

        results = await asyncio.gather(
            *[
                _run_with_retry(task, executor, task_contexts[task.id], retry, statuses)
                for task in ready_tasks
            ],
            return_exceptions=True,
        )

        for task, res in zip(ready_tasks, results):
            running.discard(task.id)
            if isinstance(res, Exception):
                statuses[task.id] = "failed"
                done.append(
                    SubTaskOutput(
                        id=task.id,
                        name=task.name,
                        detail=str(res),
                        summary=f"[失败] {res}",
                    )
                )
                with _print_lock:
                    print(
                        f"\n[Scheduler] ✗ 子任务 {task.id} ({task.name}) 最终失败: {res}"
                    )
            else:
                statuses[task.id] = "done"
                done.append(res)
                completed_cache[task.id] = {
                    "detail": res.detail,
                    "summary": res.summary,
                    "tools_used": res.tools_used,
                    "key_findings": res.key_findings,
                }
                with _print_lock:
                    print(f"[Scheduler] ✓ 子任务 {task.id}: {task.name}")
                    print(f"             → {res.summary}")

        for task in ready_tasks:
            in_degree[task.id] = -1
            for t in tasks:
                if task.id in t.depends:
                    in_degree[t.id] -= 1

        # Audit trail: summaries only (not full details)
        for task in ready_tasks:
            if task.id in completed_cache:
                accumulated_context += (
                    f"\n\n[子任务 {task.id} ({task.name})]\n"
                    f"{completed_cache[task.id]['summary']}"
                )

        _print_progress(len(done), total)

    print()
    return statuses, done


async def _run_with_retry(
    task: SubTask,
    executor: Any,
    context: str,
    retry: RetryConfig,
    statuses: dict[int, str],
) -> SubTaskOutput:
    """Execute a single task with exponential-backoff retry on failure."""
    last_error: Exception | None = None

    for attempt in range(1, retry.max_attempts + 1):
        statuses[task.id] = "running"
        try:
            result: SubTaskOutput = await executor(task.id, task.name, context)
            statuses[task.id] = "done"
            if result.expert_mode:
                print(f"[Scheduler]   子任务 {task.id} ({task.name}) → 专家模式")
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            statuses[task.id] = "pending"
            if attempt < retry.max_attempts:
                delay = min(retry.base_delay * (2 ** (attempt - 1)), retry.max_delay)
                with _print_lock:
                    print(
                        f"\n[Scheduler] ! 子任务 {task.id} ({task.name}) "
                        f"尝试 {attempt}/{retry.max_attempts} 失败: {exc}, "
                        f"{delay:.1f}s 后重试..."
                    )
                await asyncio.sleep(delay)

    statuses[task.id] = "failed"
    raise RuntimeError(
        f"子任务 {task.id} ({task.name}) 在 {retry.max_attempts} 次尝试后仍然失败"
    ) from last_error


def check_circle(tasks: list[SubTask]) -> bool:
    """Detect whether the task list contains a cycle using Kahn's algorithm.

    Returns True if a cycle exists (i.e. the graph is NOT a valid DAG),
    False if the graph is acyclic (valid DAG).
    """
    if not tasks:
        return False

    in_degree: dict[int, int] = defaultdict(int)
    all_ids: set[int] = set()
    adjacency: dict[int, list[int]] = defaultdict(list)

    for t in tasks:
        all_ids.add(t.id)
        in_degree[t.id]
        for pre_id in t.depends:
            if pre_id not in all_ids:
                return True
            adjacency[pre_id].append(t.id)
            in_degree[t.id] += 1

    queue: list[int] = [tid for tid in all_ids if in_degree[tid] == 0]
    visited: int = 0

    while queue:
        cur = queue.pop(0)
        visited += 1
        for nxt in adjacency[cur]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    return visited != len(all_ids)


def _print_progress(completed: int, total: int) -> None:
    bar_len = 30
    filled = int(bar_len * completed / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = f"{100 * completed / total:.0f}%" if total else "0%"
    with _print_lock:
        print(f"\r  [{bar}] {pct}  ({completed}/{total})   ", flush=True)
