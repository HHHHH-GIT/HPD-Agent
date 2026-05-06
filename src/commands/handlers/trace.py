"""Handler for the /trace command — enable or disable tracing.

Sub-commands:
    /trace          → show current status
    /trace on       → full tracing (console + file)
    /trace half     → console only, no file saved
    /trace off      → disable tracing

Trace mode is persisted to ~/.hpagent/config.json and restored on startup.
"""

import json
from pathlib import Path

from src.agents import QueryAgent
from src.cli import get_renderer

_CONFIG_PATH = Path.home() / ".hpagent" / "config.json"

# Module-level tracing toggle — read by main.py
# "on" = full (console + file), "half" = console only, "off" = disabled
_trace_mode: str = "on"


def _load_config() -> None:
    """Load trace mode from config file."""
    global _trace_mode
    try:
        if _CONFIG_PATH.exists():
            cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            mode = cfg.get("trace_mode")
            if mode in ("on", "half", "off"):
                _trace_mode = mode
    except Exception:
        pass


def _save_config() -> None:
    """Save current trace mode to config file."""
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cfg = {}
        if _CONFIG_PATH.exists():
            cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        cfg["trace_mode"] = _trace_mode
        _CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def is_trace_enabled() -> bool:
    return _trace_mode != "off"


def is_trace_save_enabled() -> bool:
    return _trace_mode == "on"


def get_trace_mode() -> str:
    return _trace_mode


# Load persisted config on import
_load_config()


def run(raw: str, agent: QueryAgent) -> bool:
    global _trace_mode
    renderer = get_renderer()

    parts = raw.strip().split()
    if len(parts) == 1:
        labels = {"on": "启用（完整）", "half": "仅控制台", "off": "禁用"}
        renderer.render_trace_mode(f"链路追踪当前: {labels.get(_trace_mode, _trace_mode)}")
        return False

    sub = parts[1].lower()
    if sub in ("on", "1", "enable", "true"):
        _trace_mode = "on"
        _save_config()
        renderer.success("链路追踪: 已启用（控制台 + 文件）")
    elif sub in ("half", "console"):
        _trace_mode = "half"
        _save_config()
        renderer.info("链路追踪: 仅控制台输出，不保存文件")
    elif sub in ("off", "0", "disable", "false"):
        _trace_mode = "off"
        _save_config()
        renderer.warning("链路追踪: 已禁用")
    else:
        renderer.error("用法: /trace [on|half|off]")
        return False

    return False
