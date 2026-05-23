from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.compaction import SessionArtifactStore


class _FailingArtifactStore:
    def write(self, **_kwargs):
        raise OSError("disk full")


@pytest.mark.asyncio
async def test_large_tool_result_is_artifacted_before_next_llm_call(tmp_path: Path) -> None:
    from src.llm.client import invoke_with_tools

    llm = AsyncMock()
    first_response = MagicMock(content="")
    first_response.tool_calls = [
        {"id": "tool-1", "name": "terminal", "args": {"cmd": "cat huge.log"}}
    ]
    second_response = MagicMock(content="done")
    second_response.tool_calls = []
    llm.ainvoke.side_effect = [first_response, second_response]

    tool = MagicMock()
    tool.name = "terminal"
    tool.ainvoke = AsyncMock(return_value="alpha " * 5000)

    with patch("src.llm.client.get_llm_with_tools", return_value=llm):
        content, tool_log = await invoke_with_tools(
            "prompt",
            tools=[tool],
            context_window=20_000,
            artifact_store=SessionArtifactStore(tmp_path),
        )

    second_messages = llm.ainvoke.call_args_list[1].args[0]
    tool_message = second_messages[-1]
    assert content == "done"
    assert "[artifact:" in tool_message.content
    assert len(tool_message.content) < 2000
    assert "[artifact:" in tool_log
    assert list(tmp_path.rglob("tool_result-*.txt"))


@pytest.mark.asyncio
async def test_artifact_write_failure_falls_back_to_truncated_tool_message() -> None:
    from src.llm.client import invoke_with_tools

    llm = AsyncMock()
    first_response = MagicMock(content="")
    first_response.tool_calls = [
        {"id": "tool-1", "name": "read_file", "args": {"path": "large.txt"}}
    ]
    second_response = MagicMock(content="done")
    second_response.tool_calls = []
    llm.ainvoke.side_effect = [first_response, second_response]

    tool = MagicMock()
    tool.name = "read_file"
    tool.ainvoke = AsyncMock(return_value="echo " * 5000)

    with patch("src.llm.client.get_llm_with_tools", return_value=llm):
        await invoke_with_tools(
            "prompt",
            tools=[tool],
            context_window=20_000,
            artifact_store=_FailingArtifactStore(),
        )

    second_messages = llm.ainvoke.call_args_list[1].args[0]
    assert "[artifact_write_failed: OSError]" in second_messages[-1].content
    assert len(second_messages[-1].content) < 6000


@pytest.mark.asyncio
async def test_tool_loop_compacts_old_rounds_when_context_crosses_threshold(tmp_path: Path) -> None:
    from src.llm.client import invoke_with_tools

    llm = AsyncMock()
    responses = []
    for index in range(5):
        response = MagicMock(content="")
        response.tool_calls = [
            {"id": f"tool-{index}", "name": "read_file", "args": {"path": f"f{index}.txt"}}
        ]
        responses.append(response)
    final_response = MagicMock(content="done")
    final_response.tool_calls = []
    responses.append(final_response)
    llm.ainvoke.side_effect = responses

    tool = MagicMock()
    tool.name = "read_file"
    tool.ainvoke = AsyncMock(return_value="beta " * 1200)

    with patch("src.llm.client.get_llm_with_tools", return_value=llm):
        await invoke_with_tools(
            "prompt",
            tools=[tool],
            max_rounds=10,
            context_window=5000,
            artifact_store=SessionArtifactStore(tmp_path),
        )

    last_messages = llm.ainvoke.call_args_list[-1].args[0]
    message_contents = [
        getattr(message, "content", "")
        for message in last_messages
    ]
    tool_contents = [
        content
        for message, content in zip(last_messages, message_contents)
        if message.__class__.__name__ == "ToolMessage"
    ]
    assert any("[compacted_tool_round]" in content for content in message_contents)
    assert len(tool_contents) <= 3


@pytest.mark.asyncio
async def test_tool_loop_stops_and_finalizes_when_strict_context_still_exhausted(
    tmp_path: Path,
) -> None:
    from src.llm.client import invoke_with_tools

    tool_llm = AsyncMock()
    response = MagicMock(content="")
    response.tool_calls = [
        {"id": "tool-1", "name": "read_file", "args": {"path": "huge.txt"}}
    ]
    tool_llm.ainvoke.return_value = response

    final_llm = AsyncMock()
    final_response = MagicMock(content="best effort")
    final_llm.ainvoke.return_value = final_response

    tool = MagicMock()
    tool.name = "read_file"
    tool.ainvoke = AsyncMock(return_value="charlie " * 5000)

    with patch("src.llm.client.get_llm_with_tools", return_value=tool_llm), patch(
        "src.llm.client.get_llm",
        return_value=final_llm,
    ):
        content, tool_log = await invoke_with_tools(
            "prompt",
            tools=[tool],
            context_window=30,
            tool_context_policy="strict",
            artifact_store=SessionArtifactStore(tmp_path),
        )

    assert content == "best effort"
    assert "[ToolContextExhausted]" in tool_log
    final_llm.ainvoke.assert_awaited_once()
    final_messages = final_llm.ainvoke.call_args.args[0]
    assert "charlie charlie charlie" not in "\n".join(
        getattr(message, "content", "") for message in final_messages
    )


@pytest.mark.asyncio
async def test_tool_loop_records_context_metadata_on_active_trace(tmp_path: Path) -> None:
    from src.core.observability import get_tracer
    from src.llm.client import invoke_with_tools

    tracer = get_tracer()
    _ = tracer.end_trace()
    _ = tracer.start_trace(query="tool context", session_id="s")

    llm = AsyncMock()
    first_response = MagicMock(content="")
    first_response.tool_calls = [
        {"id": "tool-1", "name": "read_file", "args": {"path": "large.txt"}}
    ]
    second_response = MagicMock(content="done")
    second_response.tool_calls = []
    llm.ainvoke.side_effect = [first_response, second_response]

    tool = MagicMock()
    tool.name = "read_file"
    tool.ainvoke = AsyncMock(return_value="delta " * 5000)

    with tracer.span("direct_answer") as span_id:
        with patch("src.llm.client.get_llm_with_tools", return_value=llm):
            await invoke_with_tools(
                "prompt",
                tools=[tool],
                context_window=20_000,
                artifact_store=SessionArtifactStore(tmp_path),
            )
        span = tracer.get_span(span_id)

    record = tracer.end_trace()
    assert record is not None
    assert span is not None
    assert span.metadata["tool_artifact_tokens"] > 0
    assert span.metadata["tool_context_tokens"] > 0
