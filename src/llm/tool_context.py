"""Context budgeting for iterative tool-calling loops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.memory.compaction import SessionArtifactStore

ToolContextPolicy = Literal["auto", "compact", "strict"]

_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_CONFIG_PATH = Path.home() / ".hpagent" / "config.json"
_DEFAULT_ARTIFACT_ROOT = Path.home() / ".hpagent" / "tool_artifacts"


@dataclass(frozen=True)
class ToolContextConfig:
    compact_ratio: float = 0.70
    force_ratio: float = 0.90
    preserve_recent_tool_rounds: int = 3
    max_inline_tool_result_tokens: int = 4000


@dataclass
class ToolContextStats:
    tool_context_tokens: int = 0
    tool_artifact_tokens: int = 0
    tool_messages_compacted: int = 0
    tool_context_exhausted: bool = False


@dataclass
class _ToolRound:
    ai_message: AIMessage
    tool_message: ToolMessage
    summary: str
    token_count: int
    compacted: bool = False


class ToolLoopContextManager:
    """Keep tool-loop messages under the active model context window."""

    def __init__(
        self,
        *,
        context_window: int | None = None,
        policy: ToolContextPolicy = "auto",
        artifact_store: SessionArtifactStore | None = None,
        token_counter: Any | None = None,
        config: ToolContextConfig | None = None,
    ) -> None:
        self.context_window = int(context_window or _DEFAULT_CONTEXT_WINDOW)
        self.policy: ToolContextPolicy = policy
        self.artifact_store = artifact_store or SessionArtifactStore(_DEFAULT_ARTIFACT_ROOT)
        self.token_counter = token_counter or count_tokens
        self.config = config or load_tool_context_config()
        self.stats = ToolContextStats()
        self._rounds: list[_ToolRound] = []

    def append_tool_interaction(
        self,
        messages: list[Any],
        *,
        call: dict,
        name: str,
        args: dict,
        result: object,
    ) -> tuple[bool, str]:
        """Append one tool interaction and compact if needed.

        Returns:
            (ok, reason).  If ok is False, callers should stop tool-calling and
            finalize from the already-compacted evidence.
        """
        ai_message = AIMessage(content="", tool_calls=[call])
        raw_result = str(result)
        content, summary, artifact_tokens = self._prepare_tool_content(
            name=name,
            args=args,
            raw_result=raw_result,
        )
        self.stats.tool_artifact_tokens += artifact_tokens
        tool_message = ToolMessage(
            name=name,
            content=content,
            tool_call_id=call.get("id", ""),
        )
        round_token_count = self.token_counter(_message_text(ai_message)) + self.token_counter(content)
        self._rounds.append(
            _ToolRound(
                ai_message=ai_message,
                tool_message=tool_message,
                summary=summary,
                token_count=round_token_count,
            )
        )
        self._rebuild_messages(messages)
        ok, reason = self._compact_if_needed(messages)
        self.stats.tool_context_tokens = self.count_messages(messages)
        return ok, reason

    def count_messages(self, messages: list[Any]) -> int:
        return sum(self.token_counter(_message_text(message)) for message in messages)

    def prepare_final_messages(self, messages: list[Any], reason: str) -> None:
        """Replace oversized tool-loop messages with compact evidence summaries."""
        base = messages[:1] if messages else []
        summary_lines = [reason, "Retained tool evidence summary:"]
        for index, round_item in enumerate(self._rounds, start=1):
            summary_lines.append(f"\n[{index}] {_final_evidence_summary(round_item.summary)}")
        summary = _clip_text(
            "\n".join(summary_lines),
            max(1000, self.context_window * 3),
        )
        messages[:] = [*base, HumanMessage(content=summary)]
        self.stats.tool_context_tokens = self.count_messages(messages)

    def _prepare_tool_content(
        self,
        *,
        name: str,
        args: dict,
        raw_result: str,
    ) -> tuple[str, str, int]:
        token_count = self.token_counter(raw_result)
        summary = _summarize_tool_result(name, args, raw_result, token_count)
        if token_count <= self.config.max_inline_tool_result_tokens:
            return raw_result, summary, 0

        try:
            artifact = self.artifact_store.write(
                session_id="tool_loop",
                project_hash="tool_loop",
                kind="tool_result",
                title=f"{name} result",
                content=raw_result,
                token_count=token_count,
            )
            artifact_ref = (
                f"[artifact: {artifact.artifact_id} "
                f"kind={artifact.kind} tokens={artifact.token_count} ref={artifact.content_ref}]"
            )
            return f"{summary}\n{artifact_ref}", f"{summary}\n{artifact_ref}", token_count
        except Exception as exc:
            clipped = _clip_text(raw_result, 4000)
            failed = (
                f"{summary}\n[artifact_write_failed: {exc.__class__.__name__}]\n"
                f"{clipped}"
            )
            return failed, failed, 0

    def _compact_if_needed(self, messages: list[Any]) -> tuple[bool, str]:
        current_tokens = self.count_messages(messages)
        compact_threshold = int(self.context_window * self.config.compact_ratio)
        force_threshold = int(self.context_window * self.config.force_ratio)

        if current_tokens < compact_threshold and self.policy == "auto":
            return True, ""
        if current_tokens < force_threshold and self.policy != "compact":
            return True, ""

        self._compact_old_rounds(messages)
        current_tokens = self.count_messages(messages)
        if current_tokens <= self.context_window:
            return True, ""

        # Strict final safety: even after compaction, do not call the model with
        # an oversized tool-loop prompt.
        self.stats.tool_context_exhausted = True
        return False, (
            "Tool loop context exceeded model window after compaction: "
            f"{current_tokens}/{self.context_window} tokens."
        )

    def _compact_old_rounds(self, messages: list[Any]) -> None:
        preserve = max(0, self.config.preserve_recent_tool_rounds)
        keep_start = max(0, len(self._rounds) - preserve)
        changed = False
        for index, round_item in enumerate(self._rounds):
            if index >= keep_start or round_item.compacted:
                continue
            compacted_content = f"[compacted_tool_round]\n{round_item.summary}"
            round_item.tool_message = ToolMessage(
                name=round_item.tool_message.name,
                content=compacted_content,
                tool_call_id=round_item.tool_message.tool_call_id,
            )
            round_item.token_count = self.token_counter(compacted_content)
            round_item.compacted = True
            self.stats.tool_messages_compacted += 1
            changed = True
        if changed:
            self._rebuild_messages(messages)

    def _rebuild_messages(self, messages: list[Any]) -> None:
        base = messages[:1] if messages else []
        rebuilt = list(base)
        for round_item in self._rounds:
            if round_item.compacted:
                rebuilt.append(HumanMessage(content=round_item.tool_message.content))
            else:
                rebuilt.append(round_item.ai_message)
                rebuilt.append(round_item.tool_message)
        messages[:] = rebuilt


def load_tool_context_config(config_path: str | Path | None = None) -> ToolContextConfig:
    path = Path(config_path).expanduser() if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.exists():
        return ToolContextConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        section = raw.get("tool_context", {}) if isinstance(raw, dict) else {}
        if not isinstance(section, dict):
            return ToolContextConfig()
        return ToolContextConfig(
            compact_ratio=_bounded_float(section.get("compact_ratio"), 0.70, 0.10, 0.95),
            force_ratio=_bounded_float(section.get("force_ratio"), 0.90, 0.20, 0.99),
            preserve_recent_tool_rounds=max(0, int(section.get("preserve_recent_tool_rounds", 3))),
            max_inline_tool_result_tokens=max(1, int(section.get("max_inline_tool_result_tokens", 4000))),
        )
    except Exception:
        return ToolContextConfig()


def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
    except Exception:
        return max(1, len(text) // 4)


def _bounded_float(value: object, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return min(max(parsed, low), high)


def _message_text(message: Any) -> str:
    parts = [message.__class__.__name__, str(getattr(message, "content", "") or "")]
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        parts.append(json.dumps(tool_calls, ensure_ascii=False, default=str))
    return "\n".join(parts)


def _summarize_tool_result(name: str, args: dict, raw_result: str, token_count: int) -> str:
    lines = [
        f"[tool_result_summary] tool={name} tokens={token_count}",
    ]
    if args:
        lines.append(f"args={json.dumps(args, ensure_ascii=False, default=str)[:800]}")

    stripped = raw_result.strip()
    if stripped.startswith("[Command failed"):
        lines.append(_extract_labeled_block(stripped, "[STDERR]") or stripped[:1000])
    elif stripped.startswith("{") or stripped.startswith("["):
        lines.append(_json_shape_summary(stripped))
    else:
        first = _clip_text(stripped, 600)
        last = _clip_text(stripped[-600:], 600) if len(stripped) > 1200 else ""
        lines.append(f"head:\n{first}")
        if last:
            lines.append(f"tail:\n{last}")
    return "\n".join(line for line in lines if line)


def _json_shape_summary(text: str) -> str:
    try:
        data = json.loads(text)
    except Exception:
        return f"json/text head:\n{_clip_text(text, 1200)}"
    if isinstance(data, dict):
        keys = list(data)[:20]
        title = data.get("title") or data.get("url") or data.get("path") or data.get("error")
        return f"json object keys={keys} highlight={title!r}"
    if isinstance(data, list):
        return f"json list length={len(data)} head={json.dumps(data[:3], ensure_ascii=False, default=str)[:1200]}"
    return f"json scalar={data!r}"


def _extract_labeled_block(text: str, label: str) -> str:
    if label not in text:
        return ""
    return f"{label}\n{_clip_text(text.split(label, 1)[1].strip(), 1200)}"


def _final_evidence_summary(summary: str) -> str:
    lines = []
    for line in summary.splitlines():
        if line in {"head:", "tail:"}:
            break
        lines.append(line)
    return "\n".join(lines) or summary.splitlines()[0]


def _clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit].rstrip() + f"\n...[truncated {omitted} chars]"


__all__ = [
    "ToolContextConfig",
    "ToolContextPolicy",
    "ToolContextStats",
    "ToolLoopContextManager",
    "count_tokens",
    "load_tool_context_config",
]
