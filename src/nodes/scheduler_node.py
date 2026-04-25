from src.core.models import TaskOutput
from src.core.state import AgentState
from src.nodes.scheduler import run_all as scheduler
from .executor import run as executor
from .scheduler import RetryConfig


async def scheduler_node(state: AgentState) -> AgentState:
    """Execute all sub-tasks via Kahn's algorithm + asyncio parallel execution.

    Reads ``tasks`` from state; writes ``sub_task_statuses`` and ``sub_task_outputs``.
    """
    tasks = state.get("tasks", [])
    retry = RetryConfig(max_attempts=3, base_delay=1.0, max_delay=10.0)
    statuses, done = await scheduler(
        tasks=tasks,
        executor=executor,
        context=state["input"],
        retry=retry,
    )

    failed_count = sum(1 for o in done if o.summary.startswith("[失败]"))
    completed_count = len(done) - failed_count

    output = TaskOutput(
        node="scheduler",
        result={
            "total": len(tasks),
            "completed": completed_count,
            "failed": failed_count,
            "task_summaries": [
                {"id": o.id, "name": o.name, "summary": o.summary, "expert_mode": o.expert_mode}
                for o in done
            ],
        },
    )

    return {
        "sub_task_statuses": statuses,
        "sub_task_outputs": done,
        "outputs": [*state.get("outputs", []), output],
    }
