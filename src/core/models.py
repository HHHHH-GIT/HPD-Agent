from pydantic import BaseModel, Field

from .enums import TaskDifficulty, SubTaskDifficulty


class AgentMeta(BaseModel):
    """Metadata about an agent invocation — used to track multi-agent execution."""

    role: str = Field(
        description="Agent role: 'coordinator' or 'expert'."
    )
    agent_id: str = Field(
        description="Unique identifier for this agent instance."
    )
    sub_task_id: int | None = Field(
        default=None,
        description="Sub-task ID this agent is responsible for. None for the coordinator.",
    )
    result_summary: str = Field(
        default="",
        description="Brief human-readable summary of what this agent did.",
    )


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
    tools_used: list[str] = Field(
        default_factory=list,
        description="Paths/identifiers of files or resources read by this sub-task's tools. "
        "Used by downstream tasks to avoid re-reading the same resource.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Concise key facts discovered by this sub-task. "
        "Each entry is a standalone fact (e.g. 'port=8080', 'version=2.1.0'). "
        "Used by downstream tasks to inherit knowledge without re-executing.",
    )
    tool_log: str = Field(
        default="",
        description="Full tool execution log (tool calls + results). "
        "Kept separate from detail so detail stays compact for context passing, "
        "but the synthesizer can access full tool output when needed.",
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


class RewriteResult(BaseModel):
    """Structured output from the prompt rewriter."""

    angles: list[str] = Field(
        description="N different angles/approaches for tackling the task. "
        "Each angle is a concise one-line description of a strategy."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why these angles were chosen.",
    )


class CandidateResult(BaseModel):
    """Result from executing a single prompt variation (multi-path)."""

    variation_index: int = Field(description="Index of the variation (0-based).")
    prompt_angle: str = Field(description="The angle/approach used for this candidate.")
    detail: str = Field(description="Full reasoning output from this candidate.")
    summary: str = Field(description="One-sentence summary of this candidate.")


class EvaluatorScore(BaseModel):
    """1:1 evaluation score for a single result."""

    score: float = Field(description="Quality score (0.0-1.0).")
    reasoning: str = Field(description="Explanation of the score.")
    issues: list[str] = Field(
        default_factory=list,
        description="Specific issues found in the result.",
    )


class ReflectionResult(BaseModel):
    """Structured output from the reflector."""

    improved_prompt: str = Field(
        description="Improved prompt/strategy for the next execution attempt."
    )
    strategy: str = Field(description="Summary of the improvement strategy.")
    reasoning: str = Field(description="Why this improvement should help.")


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


class ReviewTaskResult(BaseModel):
    """Per-sub-task quality assessment from the reviewer."""

    sub_task_id: int = Field(description="ID of the assessed sub-task.")
    quality: str = Field(
        description="Quality level: 'good' (sufficient), 'weak' (needs improvement), 'failed' (no useful output)."
    )
    reasoning: str = Field(description="Brief explanation of the quality assessment.")


class ReviewerDecision(BaseModel):
    """Structured output from the reviewer LLM call."""

    overall_quality: str = Field(
        description="'sufficient' (proceed to synthesis), 'needs_improvement' (re-execute), 'needs_more_tasks' (add new sub-tasks)."
    )
    task_reviews: list[ReviewTaskResult] = Field(
        description="Per-sub-task quality assessments."
    )
    re_execute_ids: list[int] = Field(
        default_factory=list,
        description="IDs of sub-tasks to re-execute (when overall_quality is 'needs_improvement')."
    )
    new_task_suggestions: list[str] = Field(
        default_factory=list,
        description="Descriptions of new sub-tasks needed (when overall_quality is 'needs_more_tasks')."
    )
    feedback: str = Field(
        default="",
        description="Free-text guidance for re-execution or new task planning."
    )
