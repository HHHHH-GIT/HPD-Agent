"""DAG scheduler: Kahn's-algorithm executor with retry, cycle detection, and progress tracking."""

import asyncio
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.core.models import SubTask, SubTaskOutput


_print_lock = threading.Lock()


@dataclass
class RetryConfig:
    """Configuration for the per-task retry mechanism."""

    max_attempts: int = 3
    """Maximum number of execution attempts per sub-task."""

    base_delay: float = 1.0
    """Initial backoff delay in seconds (doubles after each retry)."""

    max_delay: float = 10.0
    """Cap the exponential delay at this value."""


async def run_all(
    tasks: list[SubTask],
    executor: Any,
    context: str,
    retry: RetryConfig = RetryConfig(),
) -> tuple[dict[int, str], list[SubTaskOutput]]:
    """Execute all sub-tasks respecting DAG dependencies (parallel within each topological layer).

    Args:
        tasks:    Ordered DAG sub-tasks from the planner.
        executor: Async callable with signature ``(task_id, task_name, context)``.
                  Pass None to use the ExpertAgent from src.agents.expert_agent.
        context:  Passed through to each executor call.
        retry:   Retry configuration (max_attempts, base_delay, max_delay).

    Returns:
        A tuple of (statuses, outputs) where:
          - statuses maps task-id → "pending" | "running" | "done" | "failed"
          - outputs is a list of SubTaskOutput in completion order

    Raises:
        RuntimeError: if a cycle / deadlock is detected.
    """
    # Resolve executor: None means use ExpertAgent
    if executor is None:
        from src.agents.expert_agent import execute as expert_execute
        executor = expert_execute
    statuses: dict[int, str] = {t.id: "pending" for t in tasks}
    done: list[SubTaskOutput] = []
    in_degree: dict[int, int] = {t.id: len(t.depends) for t in tasks}
    running: set[int] = set()
    total = len(tasks)

    _print_progress(0, total)

    while len(done) < total:
        ready_ids = [
            i for i, d in in_degree.items() if d == 0 and i not in running
        ]

        if not ready_ids:
            raise RuntimeError(f"Deadlock: {total - len(done)} tasks stuck.")

        ready_tasks = [t for t in tasks if t.id in ready_ids]
        running.update(ready_ids)

        results = await asyncio.gather(
            *[
                _run_with_retry(
                    task,
                    executor,
                    context,
                    retry,
                    statuses,
                )
                for t, task in enumerate(ready_tasks)
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
                with _print_lock:
                    print(f"[Scheduler] ✓ 子任务 {task.id}: {task.name}")
                    # print(f"             → {res.detail}")
                    print(f"             → {res.summary}")

        for task in ready_tasks:
            in_degree[task.id] = -1
            for t in tasks:
                if task.id in t.depends:
                    in_degree[t.id] -= 1

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
        in_degree[t.id]  # ensure all ids are present even if no dependencies
        for pre_id in t.depends:
            if pre_id not in all_ids:
                return True  # reference to non-existent task is a cycle-like error
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
