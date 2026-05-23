from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.memory.compaction import (
    CompactionConfig,
    CompactionSummary,
    SessionArtifactStore,
    compact_context,
)
from src.memory.context import ConversationContext
from src.memory.session_store import load, save


def _count_chars(text: str) -> int:
    return len(text)


def test_compaction_preserves_recent_turns_and_writes_artifacts(tmp_path: Path) -> None:
    ctx = ConversationContext(max_turns=20)
    for i in range(6):
        ctx.add_user_message(f"user-{i} " + ("u" * 20))
        ctx.add_assistant_message(f"assistant-{i} " + ("a" * 40))
    ctx.sub_task_outputs.append(
        {
            "id": 1,
            "name": "Read large file",
            "detail": "DETAIL-" + ("x" * 300),
            "summary": "large file inspected",
            "tools_used": ["src/app.py"],
            "tool_log": "[Tool: read_file(path='src/app.py')]\n" + ("file" * 100),
            "expert_mode": False,
            "key_findings": ["file=src/app.py"],
        }
    )

    result = compact_context(
        ctx,
        session_id="default",
        project_hash="project",
        artifact_store=SessionArtifactStore(tmp_path),
        token_counter=_count_chars,
        config=CompactionConfig(
            max_tokens=240,
            precompact_ratio=0.7,
            force_ratio=0.9,
            preserve_recent_turns=3,
        ),
        force=True,
    )

    assert result.compacted is True
    assert [m.content.split()[0] for m in ctx.messages] == [
        "user-3",
        "assistant-3",
        "user-4",
        "assistant-4",
        "user-5",
        "assistant-5",
    ]
    assert "当前目标" in ctx.session_summary
    assert "read_file" in ctx.toolchain_summary
    assert len(ctx.artifacts) >= 2
    assert ctx.artifact_total_tokens > 0

    artifact_paths = [tmp_path / artifact.content_ref for artifact in ctx.artifacts]
    assert all(path.exists() for path in artifact_paths)
    assert any("DETAIL-" in path.read_text(encoding="utf-8") for path in artifact_paths)


def test_context_summary_references_artifacts_without_embedding_content(tmp_path: Path) -> None:
    ctx = ConversationContext(max_turns=10)
    ctx.add_user_message("old question " + ("q" * 200))
    ctx.add_assistant_message("old answer " + ("a" * 200))
    ctx.add_user_message("recent question")
    ctx.add_assistant_message("recent answer")

    compact_context(
        ctx,
        session_id="s1",
        project_hash="p1",
        artifact_store=SessionArtifactStore(tmp_path),
        token_counter=_count_chars,
        config=CompactionConfig(max_tokens=80, preserve_recent_turns=1),
        force=True,
    )

    rendered = ctx.to_summary()

    assert "old question" not in rendered
    assert "old answer" not in rendered
    assert "recent question" in rendered
    assert "recent answer" in rendered
    assert "artifact:" in rendered


def test_compacted_session_metadata_survives_save_and_load(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ctx = ConversationContext(max_turns=10)
    ctx.add_user_message("old question " + ("q" * 100))
    ctx.add_assistant_message("old answer " + ("a" * 100), tool_summary="read_file: src/main.py")
    ctx.add_user_message("recent")
    ctx.add_assistant_message("answer")

    compact_context(
        ctx,
        session_id="s1",
        project_hash="p1",
        artifact_store=SessionArtifactStore(tmp_path / "artifacts"),
        token_counter=_count_chars,
        config=CompactionConfig(max_tokens=80, preserve_recent_turns=1),
        force=True,
    )
    save(ctx, "s1", project_hash="p1")

    loaded = load("s1", project_hash="p1")

    assert loaded is not None
    assert loaded.session_summary == ctx.session_summary
    assert loaded.toolchain_summary == ctx.toolchain_summary
    assert [artifact.content_ref for artifact in loaded.artifacts] == [
        artifact.content_ref for artifact in ctx.artifacts
    ]
    assert loaded.artifact_total_tokens == ctx.artifact_total_tokens


def test_compaction_uses_llm_summarizer_when_provided(tmp_path: Path) -> None:
    ctx = ConversationContext(max_turns=10)
    ctx.add_user_message("old question")
    ctx.add_assistant_message("old answer", tool_summary="read_file: src/main.py")
    ctx.add_user_message("recent question")
    ctx.add_assistant_message("recent answer")
    calls = []

    def summarizer(payload):
        calls.append(payload)
        assert "old question" in payload.history_text
        assert "read_file" in payload.toolchain_text
        return CompactionSummary(
            session_summary="LLM session summary",
            toolchain_summary="LLM toolchain summary",
        )

    compact_context(
        ctx,
        session_id="s1",
        project_hash="p1",
        artifact_store=SessionArtifactStore(tmp_path),
        token_counter=_count_chars,
        config=CompactionConfig(max_tokens=80, preserve_recent_turns=1),
        force=True,
        summarizer=summarizer,
    )

    assert calls
    assert ctx.session_summary == "LLM session summary"
    assert ctx.toolchain_summary == "LLM toolchain summary"


def test_compaction_marks_summary_when_llm_summarizer_fails(tmp_path: Path) -> None:
    ctx = ConversationContext(max_turns=10)
    ctx.add_user_message("old question")
    ctx.add_assistant_message("old answer")
    ctx.add_user_message("recent question")
    ctx.add_assistant_message("recent answer")

    def broken_summarizer(_payload):
        raise RuntimeError("llm down")

    compact_context(
        ctx,
        session_id="s1",
        project_hash="p1",
        artifact_store=SessionArtifactStore(tmp_path),
        token_counter=_count_chars,
        config=CompactionConfig(max_tokens=80, preserve_recent_turns=1),
        force=True,
        summarizer=broken_summarizer,
    )

    assert "LLM summary failed" in ctx.session_summary
    assert "llm down" in ctx.session_summary
