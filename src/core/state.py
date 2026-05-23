from collections.abc import Sequence
from typing import Literal, TypedDict

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

    force_complex: bool
    """Whether the user explicitly forced complex-task routing for this query."""

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

    parent_span_id: str | None
    """Parent span ID for tracing. Propagated from the caller so child spans form a tree."""

    review_round: int
    """Current review round (0 = initial execution, 1+ = feedback loops)."""

    review_decision: Literal["proceed", "re-execute", "add_tasks", "repair"] | None
    """Reviewer's last decision: 'proceed' | 're-execute' | 'add_tasks' | 'repair' | None."""

    re_execute_task_ids: list[int]
    """IDs of sub-tasks the reviewer wants re-executed."""

    review_feedback: str
    """Free-text feedback from the reviewer for re-execution or new task planning."""

    new_sub_tasks: list[SubTask]
    """New sub-tasks proposed by the coordinator during re-planning."""
