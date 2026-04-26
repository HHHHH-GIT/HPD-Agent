from typing import TypedDict, Sequence

from .enums import TaskDifficulty
from .models import AgentMeta, PlannerResult, SubTask, SubTaskOutput, TaskOutput
from src.memory.context import ConversationContext


class AgentState(TypedDict):
    """Shared memory for the entire agent pipeline.

    All nodes read from and write to this state. Fields are designed
    to accumulate across node executions so the graph can be extended
    without reshaping existing fields.
    """

    input: str
    """User's original question."""

    analysis: TaskDifficulty | None
    """First-level assessment result (simple | complex). Set by first_level_assessment node."""

    tasks: Sequence[SubTask]
    """DAG of sub-tasks. Populated by decomposer; consumed by scheduler."""

    decomposition_result: PlannerResult | None
    """Raw planner LLM result containing sub-task metadata and total count."""

    sub_task_statuses: dict[int, str]
    """Real-time status of each sub-task (pending | running | done | failed)."""

    sub_task_outputs: Sequence[SubTaskOutput]
    """Accumulated results from completed sub-tasks."""

    outputs: Sequence[TaskOutput]
    """Sequential record of each node's output, keyed by node name."""

    final_response: str
    """The final answer returned to the user."""

    synthesis_prompt: str
    """Full prompt for the synthesis LLM. Set by synthesizer; consumed by main.py for streaming output."""

    conversation_history: ConversationContext
    """Short-term rolling context window of the current conversation thread.

    Tracks recent user/assistant message pairs so the agent can understand
    follow-up questions, pronouns, and conversational flow.
    """

    agent_history: Sequence[AgentMeta]
    """Record of which agents were invoked and on which sub-tasks."""
