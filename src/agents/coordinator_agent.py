"""Coordinator Agent: decides whether to decompose a complex task.

This is a thin decision shell — all business logic lives in nodes/planning.py.
The agent only decides "yes/no" and delegates actual decomposition to the planning node.

The coordinator writes ``tasks`` and ``decomposition_result`` to state;
the graph edges then drive ``scheduler_node`` (execution) and ``synthesizer``
(streaming) as separate nodes.
"""

import uuid

from src.core.models import AgentMeta, TaskOutput
from src.core.state import AgentState
from src.nodes.planning import decompose as decompose_node


async def coordinate(state: AgentState) -> AgentState:
    """Decide to decompose a complex task, delegating to the planning node.

    Reads from state:
        - ``input``: the original user query

    Writes to state:
        - ``tasks``: the decomposed DAG
        - ``decomposition_result``: raw LLM planner result
        - ``agent_history``: metadata about coordinator invocation
    """
    coordinator_id = f"coordinator-{uuid.uuid4().hex[:8]}"

    print(f"\n[CoordinatorAgent {coordinator_id}] 收到复杂任务，开始规划...")

    tasks, result = await decompose_node(state["input"])

    coordinator_meta = AgentMeta(
        role="coordinator",
        agent_id=coordinator_id,
        sub_task_id=None,
        result_summary=f"分解为 {len(tasks)} 个子任务",
    )

    return {
        "tasks": tasks,
        "decomposition_result": result,
        "outputs": [
            *state.get("outputs", []),
            TaskOutput(
                node="coordinator",
                result={
                    "coordinator_id": coordinator_id,
                    "total_sub_task_count": result.total_sub_task_count,
                    "sub_tasks": [
                        {"id": t.id, "name": t.name, "depends": t.depends}
                        for t in tasks
                    ],
                },
            ),
        ],
        "agent_history": [
            *state.get("agent_history", []),
            coordinator_meta,
        ],
    }
