from pathlib import Path

from src.commands.handlers import summary as summary_handler
from src.memory.compaction import CompactionSummary, SessionArtifactStore
from src.memory.context import ConversationContext


class _Renderer:
    def __init__(self) -> None:
        self.summary_text = ""

    def info(self, _message: str) -> None:
        pass

    def render_summary(self, **kwargs) -> None:
        self.summary_text = kwargs["summary_text"]


class _Agent:
    _current_session = "default"
    _project_hash = "project"

    def __init__(self, ctx: ConversationContext) -> None:
        self.ctx = ctx
        self.saved = False

    def _get_context(self) -> ConversationContext:
        return self.ctx

    def _model_context_window(self) -> int:
        return 128_000

    def save_current_session(self) -> None:
        self.saved = True


def test_manual_summary_uses_llm_summarizer_for_all_resident_messages(tmp_path: Path, monkeypatch) -> None:
    ctx = ConversationContext(max_turns=10)
    ctx.add_user_message("short resident question")
    ctx.add_assistant_message("short resident answer")
    agent = _Agent(ctx)
    renderer = _Renderer()
    calls = []

    def fake_summarizer(payload):
        calls.append(payload)
        assert "short resident question" in payload.history_text
        return CompactionSummary("LLM manual summary", "LLM toolchain")

    monkeypatch.setattr(summary_handler, "get_renderer", lambda: renderer)
    monkeypatch.setattr(summary_handler, "default_artifact_store", lambda _project_hash: SessionArtifactStore(tmp_path))
    monkeypatch.setattr(summary_handler, "summarize_compaction_with_llm", fake_summarizer)

    should_exit = summary_handler.run("/summary", agent)

    assert should_exit is False
    assert calls
    assert ctx.messages == []
    assert ctx.session_summary == "LLM manual summary"
    assert agent.saved is True
    assert "LLM manual summary" in renderer.summary_text
