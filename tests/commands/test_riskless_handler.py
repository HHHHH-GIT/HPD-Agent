import json
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document


class _Renderer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def success(self, message: str) -> None:
        self.messages.append(("success", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))


def test_riskless_command_is_registered_and_documented() -> None:
    from src.commands import COMMAND_HANDLERS, CommandCompleter
    from src.commands.details import COMMAND_DETAILS

    assert "/riskless" in COMMAND_HANDLERS
    assert "/riskless" in COMMAND_DETAILS

    completions = list(
        CommandCompleter().get_completions(
            Document("/riskless o"),
            CompleteEvent(),
        )
    )
    assert {completion.text for completion in completions} == {"on", "off"}


def test_riskless_on_off_persists_to_config(tmp_path: Path, monkeypatch) -> None:
    from src.commands.handlers import riskless as riskless_handler
    from src.core import riskless as riskless_state

    config_path = tmp_path / "home" / ".hpagent" / "config.json"
    renderer = _Renderer()
    monkeypatch.setattr(riskless_state, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(riskless_state, "_riskless_enabled", False)
    monkeypatch.setattr(riskless_handler, "get_renderer", lambda: renderer)

    assert riskless_handler.run("/riskless on", object()) is False
    assert riskless_handler.is_riskless_enabled() is True
    assert json.loads(config_path.read_text(encoding="utf-8"))["riskless_mode"] is True

    assert riskless_handler.run("/riskless off", object()) is False
    assert riskless_handler.is_riskless_enabled() is False
    assert json.loads(config_path.read_text(encoding="utf-8"))["riskless_mode"] is False
    assert any(level == "success" for level, _ in renderer.messages)


def test_riskless_show_current_status(monkeypatch) -> None:
    from src.commands.handlers import riskless as riskless_handler
    from src.core import riskless as riskless_state

    renderer = _Renderer()
    monkeypatch.setattr(riskless_state, "_riskless_enabled", True)
    monkeypatch.setattr(riskless_handler, "get_renderer", lambda: renderer)

    assert riskless_handler.run("/riskless", object()) is False

    assert renderer.messages
    assert renderer.messages[-1][0] == "info"
    assert "on" in renderer.messages[-1][1]
