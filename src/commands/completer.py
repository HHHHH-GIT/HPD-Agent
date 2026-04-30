"""Context-aware completer with support for sub-commands.

Dynamically updates model names and session IDs after the agent is initialized.
"""

from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion

if TYPE_CHECKING:
    from src.agents import QueryAgent


class CommandCompleter(Completer):

    _agent_ref: "QueryAgent | None" = None

    @classmethod
    def set_agent(cls, agent: "QueryAgent") -> None:
        cls._agent_ref = agent

    def get_completions(self, document, complete_event):
        from src.commands import COMMAND_HANDLERS

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
            for cmd in COMMAND_HANDLERS:
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

        # ── Generic top-level command completion ────────────────────
        for c in COMMAND_HANDLERS:
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
        agent = CommandCompleter._agent_ref
        return list(agent._contexts.keys())


def get_completer() -> CommandCompleter:
    return _completer_instance


_completer_instance = CommandCompleter()
