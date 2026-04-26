"""Expert Agent: decides whether to execute a sub-task.

This is a thin decision shell — all business logic lives in nodes/execution.py.
The agent only decides "yes/no" and delegates actual execution to the execution node.

When expert mode is implemented, the agent will:
  - Use higher temperature for multi-path generation
  - Apply dynamic weighting for different task types
  - Perform multi-dimensional scoring
  - Self-reflect and iterate
"""

import uuid

from src.core.models import AgentMeta, SubTaskOutput
from src.nodes.execution import execute as _execute_node


AGENT_ROLE = "expert"


async def execute(task_id: int, task_name: str, context: str) -> SubTaskOutput:
    """Decide to execute a sub-task, delegating to the execution node.

    The decision logic (difficulty assessment + LLM call) is in nodes/execution.py.
    This shell only adds agent metadata.
    """
    return await _execute_node(task_id, task_name, context)


def make_meta(task_id: int, task_name: str, expert_mode: bool) -> AgentMeta:
    """Return agent metadata for a completed expert invocation."""
    return AgentMeta(
        role=AGENT_ROLE,
        agent_id=f"expert-{uuid.uuid4().hex[:8]}",
        sub_task_id=task_id,
        result_summary=f"expert mode={'是' if expert_mode else '否'}，执行 {task_name}",
    )
