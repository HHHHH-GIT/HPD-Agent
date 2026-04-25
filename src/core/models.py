from pydantic import BaseModel, Field

from .enums import TaskDifficulty, SubTaskDifficulty


class TaskOutput(BaseModel):
    """Standardized output record for every node in the graph."""

    node: str = Field(description="Name of the node that produced this output.")
    result: dict = Field(
        default_factory=dict,
        description="Structured result payload. Shape depends on the node.",
    )


class AssessmentResult(BaseModel):
    """Structured output from the first-level assessment LLM call."""

    difficulty: TaskDifficulty = Field(
        description="Classification: 'simple' for single-step/common-knowledge tasks, "
        "'complex' for multi-step/deep-research/logic-dependent tasks."
    )
    reasoning: str = Field(
        description="Brief explanation of why this difficulty level was assigned."
    )


class SubTask(BaseModel):
    """A single node in the DAG decomposition of a complex task."""

    id: int = Field(description="Unique integer ID for this sub-task.")
    name: str = Field(description="Human-readable name of the sub-task.")
    depends: list[int] = Field(
        default_factory=list,
        description="List of sub-task IDs that must complete before this one runs.",
    )


class SubTaskOutput(BaseModel):
    """Result produced by executing a single sub-task."""

    id: int = Field(description="ID matching the originating SubTask.")
    name: str = Field(description="Name of the sub-task.")
    summary: str = Field(
        default="",
        description="Short conclusion or key finding after executing this sub-task.",
    )
    detail: str = Field(
        default="",
        description="Full reasoning / detailed output from this sub-task (internal use).",
    )
    expert_mode: bool = Field(
        default=False,
        description="Whether this sub-task was executed in expert mode (hard sub-tasks).",
    )


class SubTaskAssessmentResult(BaseModel):
    """Structured output from second-level sub-task difficulty assessment."""

    difficulty: SubTaskDifficulty = Field(
        description="Classification: 'easy' for common-knowledge single-step subtasks, "
        "'hard' for multi-step/deep-research/logic-dependent subtasks."
    )
    reasoning: str = Field(
        description="Brief explanation of why this difficulty level was assigned."
    )


class PlannerResult(BaseModel):
    """Structured output from the planner LLM call — a DAG of sub-tasks."""

    total_sub_task_count: int = Field(
        description="Total number of sub-tasks returned."
    )
    sub_tasks: list[SubTask] = Field(
        description="Ordered list of sub-tasks with DAG dependency edges."
    )
    reasoning: str = Field(
        description="Brief explanation of why the query was broken down this way."
    )
