"""Context-aware completer with support for sub-commands.

Dynamically updates model names and session IDs after the agent is initialized.
"""

import sys
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, cast

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from typing_extensions import override

if TYPE_CHECKING:
    from src.agents import QueryAgent


class CommandCompleter(Completer):

    _agent_ref: "QueryAgent | None" = None

    @classmethod
    def set_agent(cls, agent: "QueryAgent") -> None:
        cls._agent_ref = agent

    @override
    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterator[Completion]:
        _ = complete_event
        command_names = _command_names()

        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        stripped = text.strip()
        if not stripped:
            return

        trailing_space = text != stripped or text.endswith(" ")

        parts = stripped.split()

        # ── Level 1: top-level command ──────────────────────────────
        if len(parts) == 1 and not trailing_space:
            for cmd in command_names:
                if cmd.startswith(parts[0]):
                    yield Completion(
                        cmd,
                        start_position=-len(parts[0]),
                        display=cmd.lstrip("/"),
                    )
            return

        cmd = parts[0]
        sub = parts[1].lower() if len(parts) > 1 else ""

        # ── /model sub-commands ────────────────────────────────────
        if cmd == "/model":
            model_subs = ("list", "create", "switch")
            if len(parts) == 1:
                for s in model_subs:
                    yield Completion(s, start_position=0)
            elif len(parts) == 2:
                for s in model_subs:
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub))
                if self._agent_ref:
                    for p in self._model_names():
                        if p.startswith(sub):
                            yield Completion(p, start_position=-len(sub))
            elif sub == "switch" and len(parts) == 3:
                for name in self._model_names():
                    if name.startswith(parts[2].lower()):
                        yield Completion(name, start_position=-len(parts[2]))
            return

        # ── /sessions sub-commands ─────────────────────────────────
        if cmd == "/sessions":
            session_subs = ("list", "create", "switch", "delete")
            if len(parts) == 1:
                for s in session_subs:
                    yield Completion(s, start_position=0)
            elif len(parts) == 2:
                for s in session_subs:
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub))
            elif sub in ("switch", "delete") and len(parts) == 3:
                for sid in self._session_ids():
                    if sid.startswith(parts[2]):
                        yield Completion(sid, start_position=-len(parts[2]))
            return

        # ── /context sub-commands ─────────────────────────────────
        if cmd == "/context":
            context_subs = ("clear",)
            if len(parts) == 1:
                for s in context_subs:
                    yield Completion(s, start_position=0)
            elif len(parts) == 2:
                for s in context_subs:
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub))
            return


        # ── /index sub-commands ───────────────────────────────────
        if cmd == "/index":
            index_subs = {
                "status": "查看索引状态",
                "build": "构建 symbol index",
                "clear": "清理索引缓存",
            }
            if len(parts) == 1:
                for s, meta in index_subs.items():
                    yield Completion(s, start_position=0, display_meta=meta)
            elif len(parts) == 2:
                for s, meta in index_subs.items():
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub), display_meta=meta)
            return

        # ── /lsp sub-commands ─────────────────────────────────────
        if cmd == "/lsp":
            lsp_subs = {
                "status": "查看 LSP 状态",
                "stop": "停止 LSP server",
                "restart": "重启指定语言 server",
            }
            lsp_languages = ("python", "typescript", "javascript")
            if len(parts) == 1:
                for s, meta in lsp_subs.items():
                    yield Completion(s, start_position=0, display_meta=meta)
            elif len(parts) == 2:
                for s, meta in lsp_subs.items():
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub), display_meta=meta)
            elif sub in ("restart", "stop") and len(parts) == 3:
                language_prefix = parts[2].lower()
                for language in lsp_languages:
                    if language.startswith(language_prefix):
                        yield Completion(language, start_position=-len(parts[2]), display_meta="语言标识")
            return

        # ── /trace sub-commands ───────────────────────────────────
        if cmd == "/trace":
            trace_subs = ("on", "half", "off")
            if len(parts) == 1:
                for s in trace_subs:
                    yield Completion(s, start_position=0)
            elif len(parts) == 2:
                for s in trace_subs:
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub))
            return

        # ── /riskless sub-commands ────────────────────────────────
        if cmd == "/riskless":
            riskless_subs = ("on", "off")
            if len(parts) == 1:
                for s in riskless_subs:
                    yield Completion(s, start_position=0)
            elif len(parts) == 2:
                for s in riskless_subs:
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub))
            return

        # ── /budget sub-commands ─────────────────────────────────
        if cmd == "/budget":
            budget_subs = ("normal", "extended", "web")
            if len(parts) == 1:
                for s in budget_subs:
                    yield Completion(s, start_position=0)
            elif len(parts) == 2:
                for s in budget_subs:
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub))
            return

        # ── Generic top-level command completion ────────────────────
        for c in command_names:
            if c.startswith(text) and c != cmd:
                yield Completion(c, start_position=-len(text), display=c.lstrip("/"))

    @staticmethod
    def _model_names() -> list[str]:
        if CommandCompleter._agent_ref is None:
            return []
        try:
            from src.models import get_store
            return [p.name for p in get_store().all()]
        except Exception:
            return []

    @staticmethod
    def _session_ids() -> list[str]:
        if CommandCompleter._agent_ref is None:
            return []
        contexts = cast(object | None, getattr(CommandCompleter._agent_ref, "_contexts", None))
        if not isinstance(contexts, Mapping):
            return []
        session_map = cast(Mapping[str, object], contexts)
        return list(session_map.keys())


def _command_names() -> tuple[str, ...]:
    commands_module = cast(object | None, sys.modules.get("src.commands"))
    handlers = cast(object | None, getattr(commands_module, "COMMAND_HANDLERS", None))
    if not isinstance(handlers, Mapping):
        return ()
    handler_map = cast(Mapping[str, object], handlers)
    return tuple(handler_map.keys())


def get_completer() -> CommandCompleter:
    return _completer_instance


_completer_instance = CommandCompleter()
