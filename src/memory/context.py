from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single turn in the conversation history."""

    role: MessageRole = Field(description="Sender role: user or assistant.")
    content: str = Field(
        description="Full internal content stored in the context window "
        "(for assistant messages: includes sub-task results, synthesis prompts, etc.)."
    )
    answer_content: str | None = Field(
        default=None,
        description="Clean answer shown to the user. "
        "Only set for assistant messages; None means use content as fallback."
    )
    tool_summary: str | None = Field(
        default=None,
        description="Concise summary of tool calls made during this assistant turn "
        "(e.g. 'read_file: /path/to/file, terminal: cat config'). "
        "Included in summaries so downstream context doesn't lose tool usage info."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this message was recorded.",
    )


class ContextArtifact(BaseModel):
    """A non-resident context artifact stored on disk."""

    artifact_id: str = Field(description="Stable artifact id within a session.")
    kind: str = Field(description="Artifact category, such as history or sub_task.")
    title: str = Field(description="Short human-readable title.")
    content_ref: str = Field(description="Path relative to the artifact store root.")
    token_count: int = Field(default=0, description="Estimated token count.")
    created_at: datetime = Field(default_factory=datetime.now)


class ConversationContext(BaseModel):
    """Short-term contextual memory — rolling window of the current conversation.

    Captures recent turns so the agent can understand follow-up questions,
    references, and conversational flow within a thread.
    """

    messages: list[Message] = Field(
        default_factory=list,
        description="Ordered list of recent message turns.",
    )

    max_turns: int = Field(
        default=10,
        description="Maximum number of turns to retain in the rolling window.",
    )

    sub_task_outputs: list[dict] = Field(
        default_factory=list,
        description="All completed sub-task outputs (id, name, detail, summary, "
        "tools_used, key_findings, expert_mode). Used to track token cost of DAG results "
        "that are NOT stored in messages but DO consume context window in the synthesizer.",
    )
    """Accumulated sub-task outputs — tracked separately from messages because the
    synthesizer reads them from state (not from conversation_history), but they still
    consume the LLM context window when synthesizing the final answer."""

    session_summary: str = Field(
        default="",
        description="Long-lived structured summary kept resident after compaction.",
    )

    toolchain_summary: str = Field(
        default="",
        description="Compact resident summary of prior tool calls and validations.",
    )

    artifacts: list[ContextArtifact] = Field(
        default_factory=list,
        description="References to non-resident context artifacts stored on disk.",
    )

    artifact_total_tokens: int = Field(
        default=0,
        description="Estimated tokens stored in non-resident artifacts.",
    )

    def add_user_message(self, content: str) -> None:
        """Append a user message and trim to max_turns."""
        self.messages.append(
            Message(role=MessageRole.USER, content=content)
        )
        self._trim()

    def add_assistant_message(
        self,
        content: str,
        answer_content: str | None = None,
        tool_summary: str | None = None,
    ) -> None:
        """Append an assistant message and trim to max_turns.

        Args:
            content: Full internal content (sub-task results, synthesis prompts, etc.).
            answer_content: Clean user-facing answer. If None, content is used for display.
            tool_summary: Concise summary of tool calls made (e.g. 'read_file: /path/to/file').
        """
        self.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=content,
                answer_content=answer_content,
                tool_summary=tool_summary,
            )
        )
        self._trim()

    def _trim(self) -> None:
        """Keep at most max_turns pairs (user + assistant = 1 turn)."""
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2 :]
        # Cap sub-task outputs to prevent unbounded growth
        if len(self.sub_task_outputs) > 50:
            self.sub_task_outputs = self.sub_task_outputs[-50:]

    def to_summary(self) -> str:
        """Render conversation history as a readable string for prompt injection.

        Assistant messages show their clean answer (answer_content) when available,
        falling back to full internal content.  Tool summaries are included as a
        separate block so important file/resource operations are not lost.
        """
        if not self.messages and not self.session_summary and not self.toolchain_summary and not self.artifacts:
            return ""
        lines = []
        if self.session_summary:
            lines.append(f"[session_summary]\n{self.session_summary}")
        if self.toolchain_summary:
            lines.append(f"[toolchain_summary]\n{self.toolchain_summary}")
        for artifact in self.artifacts:
            lines.append(
                "artifact: "
                f"{artifact.artifact_id} "
                f"kind={artifact.kind} "
                f"tokens={artifact.token_count} "
                f"ref={artifact.content_ref}"
            )
        for msg in self.messages:
            prefix = "用户" if msg.role == MessageRole.USER else "助手"
            text = msg.answer_content if msg.answer_content else msg.content
            lines.append(f"{prefix}: {text}")
            if msg.tool_summary:
                lines.append(f"  [工具记录] {msg.tool_summary}")
        return "\n".join(lines)

    def to_sub_tasks_summary(self) -> str:
        """Render accumulated sub-task outputs as a readable string for prompt injection.

        Each sub-task is formatted as its summary (key finding) along with its name
        and the tools it used, so the next session retains the important conclusions
        without carrying the full reasoning detail.
        """
        if not self.sub_task_outputs:
            return ""
        lines = []
        for output in self.sub_task_outputs:
            lines.append(
                f"子任务{output['id']}: {output['name']}"
                + (f" [expert模式]" if output.get("expert_mode") else "")
            )
            if output.get("summary"):
                lines.append(f"  摘要: {output['summary']}")
            if output.get("tools_used"):
                lines.append(f"  工具: {', '.join(output['tools_used'])}")
        return "\n".join(lines)
