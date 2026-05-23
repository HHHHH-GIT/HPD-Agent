from src.agents.query_agent import QueryAgent
from src.main import (
    _build_toolbar,
    _format_context_window,
    _format_project_directory,
    _refresh_toolbar_context_window,
)


def test_context_window_formatter_uses_compact_units() -> None:
    assert _format_context_window(12_345, 128_000) == "Context 12.3k/128k 9.6%"


def test_toolbar_shows_cached_context_window_label() -> None:
    agent = QueryAgent.__new__(QueryAgent)
    agent._current_session = "default"
    agent._toolbar_context_window = "Context 12.3k/128k 9.6%"

    global_agent = __import__("src.main", fromlist=["_active_agent"])
    setattr(global_agent, "_active_agent", agent)

    toolbar = _build_toolbar()
    text = getattr(toolbar, "value", str(toolbar))

    assert "Context" in text
    assert "12.3k/128k 9.6%" in text


def test_project_directory_formatter_uses_basename_for_long_paths() -> None:
    assert _format_project_directory("/root/projects/evo_agent") == "evo_agent"


def test_toolbar_shows_current_project_directory() -> None:
    agent = QueryAgent.__new__(QueryAgent)
    agent._current_session = "default"
    agent._toolbar_context_window = "Context 12.3k/128k 9.6%"
    agent._project_dir = "/root/projects/evo_agent"

    global_agent = __import__("src.main", fromlist=["_active_agent"])
    setattr(global_agent, "_active_agent", agent)

    toolbar = _build_toolbar()
    text = getattr(toolbar, "value", str(toolbar))

    assert "Project" in text
    assert "evo_agent" in text


def test_refresh_toolbar_context_window_updates_cache() -> None:
    agent = QueryAgent.__new__(QueryAgent)
    agent._current_session = "default"
    agent._toolbar_context_window = ""
    agent._get_context = lambda thread_id=None: None

    from src import main as main_module

    original_get_used_tokens = main_module.get_used_tokens
    original_get_model_context_window = main_module.get_model_context_window
    try:
        main_module.get_used_tokens = lambda _agent: 4096
        main_module.get_model_context_window = lambda: 128000
        _refresh_toolbar_context_window(agent)
    finally:
        main_module.get_used_tokens = original_get_used_tokens
        main_module.get_model_context_window = original_get_model_context_window

    assert agent._toolbar_context_window == "Context 4.1k/128k 3.2%"
