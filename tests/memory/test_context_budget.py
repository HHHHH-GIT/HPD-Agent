import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.query_agent import parse_query_control_flags
from src.memory.budget import PromptPreview, build_prompt_preview
from src.memory.context import ConversationContext


def _count_chars(text: str) -> int:
    return len(text)


def test_prompt_preview_breaks_down_resident_and_next_request_tokens() -> None:
    ctx = ConversationContext()
    ctx.session_summary = "长期摘要"
    ctx.toolchain_summary = "工具链: read_file src/main.py"
    ctx.artifact_total_tokens = 1234
    ctx.add_user_message("hello")
    ctx.add_assistant_message("world", tool_summary="read_file: src/main.py")

    preview = build_prompt_preview(
        ctx,
        query="分析一下 &",
        force_complex=parse_query_control_flags("分析一下 &")[1],
        token_counter=_count_chars,
        max_tokens=1000,
        tool_schema_tokens=50,
        include_boot_prompt=False,
    )

    assert isinstance(preview, PromptPreview)
    assert preview.resident_history_tokens > 0
    assert preview.task_summary_tokens == len("长期摘要")
    assert preview.toolchain_summary_tokens == len("工具链: read_file src/main.py")
    assert preview.tool_schema_tokens == 50
    assert preview.next_planner_tokens > 0
    assert preview.next_request_tokens == preview.next_planner_tokens
    assert preview.nonresident_artifact_tokens == 1234


def test_prompt_preview_for_simple_path_includes_tool_schema_overhead() -> None:
    ctx = ConversationContext()
    ctx.add_user_message("previous")

    preview = build_prompt_preview(
        ctx,
        query="current",
        force_complex=False,
        token_counter=_count_chars,
        max_tokens=1000,
        tool_schema_tokens=77,
        include_boot_prompt=False,
    )

    assert preview.next_direct_tokens >= 77
    assert preview.next_request_tokens == max(
        preview.next_assessment_tokens,
        preview.next_direct_tokens,
    )
