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
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this message was recorded.",
    )


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
    ) -> None:
        """Append an assistant message and trim to max_turns.

        Args:
            content: Full internal content (sub-task results, synthesis prompts, etc.).
            answer_content: Clean user-facing answer. If None, content is used for display.
        """
        self.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=content,
                answer_content=answer_content,
            )
        )
        self._trim()

    def _trim(self) -> None:
        """Keep at most max_turns pairs (user + assistant = 1 turn)."""
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2 :]

    def to_summary(self) -> str:
        """Render conversation history as a readable string for prompt injection.

        Assistant messages show their clean answer (answer_content) when available,
        falling back to full internal content.
        """
        if not self.messages:
            return ""
        lines = []
        for msg in self.messages:
            prefix = "用户" if msg.role == MessageRole.USER else "助手"
            text = msg.answer_content if msg.answer_content else msg.content
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)
