import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document


class _Renderer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def success(self, message: str) -> None:
        self.messages.append(("success", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))


def test_budget_command_is_registered_documented_and_completed() -> None:
    from src.commands import COMMAND_HANDLERS, CommandCompleter
    from src.commands.details import COMMAND_DETAILS

    assert "/budget" in COMMAND_HANDLERS
    assert "/budget" in COMMAND_DETAILS

    completions = list(
        CommandCompleter().get_completions(
            Document("/budget e"),
            CompleteEvent(),
        )
    )
    assert {completion.text for completion in completions} == {"extended"}


def test_budget_mode_persists_to_config(tmp_path: Path, monkeypatch) -> None:
    from src.commands.handlers import budget as budget_handler
    from src.core import tool_budget

    config_path = tmp_path / "home" / ".hpagent" / "config.json"
    renderer = _Renderer()
    monkeypatch.setattr(tool_budget, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(tool_budget, "_budget_mode", "normal")
    monkeypatch.setattr(budget_handler, "get_renderer", lambda: renderer)

    assert budget_handler.run("/budget web", object()) is False
    assert tool_budget.get_tool_budget_mode() == "web"
    assert json.loads(config_path.read_text(encoding="utf-8"))["tool_budget_mode"] == "web"
    assert tool_budget.get_tool_budget_profile().max_tool_calls > tool_budget.PROFILES["normal"].max_tool_calls
    assert renderer.messages[-1][0] == "success"

    assert budget_handler.run("/budget", object()) is False
    assert renderer.messages[-1][0] == "info"
    assert "web" in renderer.messages[-1][1]


@pytest.mark.asyncio
async def test_invoke_with_tools_uses_selected_budget_by_default(monkeypatch) -> None:
    from src.core import tool_budget
    from src.llm.client import invoke_with_tools

    monkeypatch.setattr(tool_budget, "_budget_mode", "web")

    first_response = MagicMock()
    first_response.content = ""
    first_response.tool_calls = [
        {"id": f"tool-{index}", "name": "read_file", "args": {"path": f"{index}.txt"}}
        for index in range(tool_budget.PROFILES["normal"].max_tool_calls + 1)
    ]
    second_response = MagicMock()
    second_response.content = "done"
    second_response.tool_calls = []

    llm = AsyncMock()
    llm.ainvoke.side_effect = [first_response, second_response]
    tool = AsyncMock()
    tool.name = "read_file"
    tool.ainvoke = AsyncMock(return_value="ok")

    with patch("src.llm.client.get_llm_with_tools", return_value=llm):
        content, _tool_log = await invoke_with_tools("prompt", tools=[tool])

    assert content == "done"
    assert tool.ainvoke.await_count == tool_budget.PROFILES["normal"].max_tool_calls + 1
