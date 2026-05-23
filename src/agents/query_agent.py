import asyncio
import os
import re
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING, cast

from langgraph.checkpoint.base import BaseCheckpointSaver

from src.core.state import AgentState
from src.memory.budget import build_prompt_preview, should_compact
from src.memory.compaction import (
    CompactionConfig,
    compact_context,
    default_artifact_store,
    summarize_compaction_with_llm,
)
from src.memory import ConversationContext, MessageRole, get_checkpointer
from src.memory.session_store import (
    _project_hash,
    delete as delete_session,
    list_sessions,
    load as load_session,
    save as save_session,
)
from src.system_info import build_boot_prompt
from src.workflow import build_graph

if TYPE_CHECKING:
    from src.code_intel.runtime import CodeIntelRuntime


def parse_query_control_flags(raw_query: str) -> tuple[str, bool]:
    """Parse user-level control suffixes from a raw query string.

    Current controls:
      - trailing ``&`` forces the query into the complex DAG route
      - trailing ``\\&`` keeps a literal ampersand with no force flag
    """
    stripped = raw_query.rstrip()
    if not stripped:
        return raw_query, False
    if stripped.endswith("\\&"):
        return stripped[:-2] + "&", False
    if not stripped.endswith("&"):
        return raw_query, False

    content = stripped[:-1].rstrip()
    if not content:
        return raw_query, False
    return content, True


class QueryAgent:
    """High-level agent facade.

    Encapsulates graph compilation, checkpointer wiring,
    conversation context threading, and the public `run` / `ainvoke` interface.
    """

    def __init__(self, checkpointer: BaseCheckpointSaver[Any] | None = None):
        graph = build_graph()
        checkpointer = checkpointer or get_checkpointer()
        self._app = graph.compile(checkpointer=checkpointer)
        self._contexts: dict[str, ConversationContext] = {}
        self._current_session: str = "default"
        self._session_boot_done: set[str] = set()
        self._auto_save_enabled: bool = True
        self._project_hash: str = _project_hash()
        self._project_dir: str = os.getcwd()
        self.code_intel_runtime: "CodeIntelRuntime | None" = None
        self._load_all()

    def _count_context_tokens(self, text: str) -> int:
        """Best-effort token counter for request gating and compaction."""
        try:
            import tiktoken

            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            return max(1, len(text) // 4) if text else 0

    def _model_context_window(self) -> int:
        """Read context window from active model profile if configured."""
        try:
            from src.models import get_store

            profile = get_store().active_profile()
            extra_body = getattr(profile, "extra_body", {}) if profile else {}
            if isinstance(extra_body, dict):
                for key in ("context_window", "max_context_tokens", "max_input_tokens"):
                    value = extra_body.get(key)
                    if value:
                        return int(value)
        except Exception:
            pass
        return 128_000

    def _count_tool_schema_tokens(self) -> int:
        """Best-effort tool schema token count used by prompt preview."""
        try:
            import json

            from langchain_core.utils.function_calling import convert_to_openai_function

            from src.tools import tool_list

            return sum(
                self._count_context_tokens(
                    json.dumps(convert_to_openai_function(tool), ensure_ascii=False)
                )
                for tool in tool_list
            )
        except Exception:
            return 0

    def maybe_compact_context(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        force_complex: bool = False,
    ) -> bool:
        """Compact resident context before sending an oversized next request."""
        sid = thread_id or self._current_session
        ctx = self._get_context(sid)
        max_tokens = self._model_context_window()
        config = CompactionConfig(max_tokens=max_tokens)
        preview = build_prompt_preview(
            ctx,
            query=query,
            force_complex=force_complex,
            token_counter=self._count_context_tokens,
            max_tokens=max_tokens,
            tool_schema_tokens=self._count_tool_schema_tokens(),
            include_boot_prompt=sid not in self._session_boot_done,
        )
        if not should_compact(preview, precompact_ratio=config.precompact_ratio):
            return False
        result = compact_context(
            ctx,
            session_id=sid,
            project_hash=self._project_hash,
            artifact_store=default_artifact_store(self._project_hash),
            token_counter=self._count_context_tokens,
            config=config,
            force=preview.usage_ratio >= config.force_ratio,
            summarizer=summarize_compaction_with_llm,
        )
        if result.compacted and self._auto_save_enabled:
            save_session(ctx, sid, self._project_hash)
        return result.compacted

    def _load_all(self) -> None:
        """Restore all persisted sessions for the current project from disk."""
        for meta in list_sessions(self._project_hash):
            sid = meta["session_id"]
            ctx = load_session(sid, self._project_hash)
            if ctx is not None:
                self._contexts[sid] = ctx
                self._session_boot_done.add(sid)

    def save_current_session(self) -> None:
        """Persist the current session to disk under the current project hash."""
        save_session(self._get_context(), self._current_session, self._project_hash)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from memory and disk for the current project."""
        self._contexts.pop(session_id, None)
        self._session_boot_done.discard(session_id)
        if self._current_session == session_id:
            self._current_session = "default"
            if "default" not in self._contexts:
                self._contexts["default"] = ConversationContext()
        return delete_session(session_id, self._project_hash)

    def _get_context(self, thread_id: str | None = None) -> ConversationContext:
        """Return the rolling context for this thread, creating if needed."""
        sid = thread_id or self._current_session
        if sid not in self._contexts:
            self._contexts[sid] = ConversationContext()
        return self._contexts[sid]

    def _summarize_tools_from_subtasks(self, outputs: Sequence[Any]) -> str:
        """Build a compact tool summary from SubTaskOutput objects or cached dicts."""
        seen: set[str] = set()
        parts: list[str] = []

        def add(item: str) -> None:
            value = item.strip()
            if not value or value in seen:
                return
            seen.add(value)
            parts.append(value)

        for output in outputs:
            if isinstance(output, dict):
                tool_log = output.get("tool_log", "") or ""
                tools_used = output.get("tools_used", [])
            else:
                tool_log = getattr(output, "tool_log", "") or ""
                tools_used = getattr(output, "tools_used", [])

            if tool_log:
                extracted = self._extract_tool_summary(str(tool_log))
                for piece in extracted.split(","):
                    if piece.strip():
                        add(piece)

            for path in tools_used or []:
                if not isinstance(path, str):
                    continue
                path = path.strip()
                if not path or path == "...":
                    continue
                if "/" in path or path.endswith((".py", ".md", ".json", ".toml", ".yaml", ".yml")):
                    add(f"read_file: {path}")
                else:
                    add(f"tool_input: {path}")

        return ", ".join(parts)

    def _backfill_missing_tool_summary(self, ctx: ConversationContext) -> None:
        """Best-effort migration for older sessions that saved no tool_summary.

        If the session contains cached sub-task outputs but no assistant message
        has tool metadata, attach a compact summary to the most recent assistant
        turn so follow-up questions about previous tool usage can be answered.
        """
        if not ctx.sub_task_outputs:
            return
        if any(msg.tool_summary for msg in ctx.messages if msg.role == MessageRole.ASSISTANT):
            return

        summary = self._summarize_tools_from_subtasks(ctx.sub_task_outputs)
        if not summary:
            return

        for msg in reversed(ctx.messages):
            if msg.role == MessageRole.ASSISTANT and not msg.tool_summary:
                msg.tool_summary = summary
                break

    def _extract_tool_summary(self, tool_log_text: str) -> str:
        """Parse tool names and paths from a tool log section.

        The log contains tool results in the form:
          [Tool: tool_name]
result
          [Tool: read_file(path='/...')]
content
          [Tool: terminal(cmd='cat /...')]
output

        We extract unique (tool_name, path/cmd) pairs for the tool_summary field.
        """
        seen: set[str] = set()
        parts: list[str] = []

        for match in re.finditer(r"\[Tool:\s*(\w+)(?:\([^)]*\))?\]", tool_log_text):
            key = match.group(1)
            if key not in seen:
                seen.add(key)
                parts.append(key)

        for match in re.finditer(r"read_file\s*\([^'\"]*['\"]([^'\"]+)['\"]", tool_log_text):
            path = match.group(1).strip()
            key = f"read_file: {path}"
            if key not in seen:
                seen.add(key)
                parts.append(key)

        for match in re.finditer(r"terminal\s*\(\s*cmd\s*=\s*'([^']+)'", tool_log_text):
            cmd = match.group(1).strip()
            key = f"terminal: {cmd}"
            if key not in seen:
                seen.add(key)
                parts.append(key)

        return ", ".join(parts) if parts else ""

    async def ainvoke(
        self,
        query: str,
        thread_id: str | None = None,
        force_complex: bool | None = None,
    ) -> AgentState:
        """Run the graph and return the final state.

        For simple tasks: final_response is populated — caller prints it.
        For complex tasks: synthesis_prompt is populated — caller must stream
        a synthesizer LLM call using that prompt.
        """
        normalized_query, suffix_forced_complex = parse_query_control_flags(query)
        if force_complex is None:
            force_complex = suffix_forced_complex
        else:
            force_complex = force_complex or suffix_forced_complex

        sid = thread_id or self._current_session
        ctx = self._get_context(sid)
        self._backfill_missing_tool_summary(ctx)
        self.maybe_compact_context(
            normalized_query,
            thread_id=sid,
            force_complex=force_complex,
        )
        ctx.add_user_message(normalized_query)

        if sid not in self._session_boot_done:
            self._session_boot_done.add(sid)
            from src.memory.context import Message

            ctx.messages.insert(
                0,
                Message(role=MessageRole.ASSISTANT, content=build_boot_prompt()),
            )

        initial_state: AgentState = {
            "input": normalized_query,
            "force_complex": force_complex,
            "analysis": None,
            "tasks": [],
            "decomposition_result": None,
            "sub_task_statuses": {},
            "sub_task_outputs": [],
            "outputs": [],
            "final_response": "",
            "synthesis_prompt": "",
            "conversation_history": ctx,
            "parent_span_id": "",
            "review_round": 0,
            "review_decision": None,
            "re_execute_task_ids": [],
            "review_feedback": "",
            "new_sub_tasks": [],
            "agent_history": [],
        }
        config = {"configurable": {"thread_id": sid}}
        result = cast(
            AgentState, await self._app.ainvoke(initial_state, config=cast(Any, config))
        )

        synthesis = result.get("synthesis_prompt", "")
        final_text = result.get("final_response") or synthesis

        outputs = result.get("sub_task_outputs", [])
        for o in outputs:
            o_dict = {
                "id": o.id,
                "name": o.name,
                "detail": o.detail,
                "summary": o.summary,
                "tools_used": o.tools_used,
                "expert_mode": o.expert_mode,
                "key_findings": getattr(o, "key_findings", []),
                "tool_log": getattr(o, "tool_log", ""),
            }
            ctx.sub_task_outputs.append(o_dict)
        if len(ctx.sub_task_outputs) > 50:
            ctx.sub_task_outputs = ctx.sub_task_outputs[-50:]

        tool_summary = self._extract_tool_summary(synthesis) if synthesis else ""
        if not tool_summary and outputs:
            tool_summary = self._summarize_tools_from_subtasks(outputs)
        if not tool_summary:
            for output in reversed(result.get("outputs", [])):
                if output.node == "direct_answer":
                    tool_log = output.result.get("tool_calls", "")
                    if isinstance(tool_log, list):
                        tool_log = "\n\n".join(str(item) for item in tool_log)
                    if tool_log:
                        tool_summary = self._extract_tool_summary(str(tool_log))
                    break
        if final_text:
            compact = final_text[:2000] if synthesis else final_text
            ctx.add_assistant_message(
                content=compact,
                answer_content=final_text[:5000] if synthesis else None,
                tool_summary=tool_summary or None,
            )

        if self._auto_save_enabled:
            save_session(ctx, sid, self._project_hash)

        return cast(AgentState, result)

    def store_streamed_answer(self, answer: str, thread_id: str | None = None) -> None:
        """Backfill the answer_content for the most recent assistant message."""
        sid = thread_id or self._current_session
        ctx = self._get_context(sid)
        if ctx.messages and ctx.messages[-1].role == MessageRole.ASSISTANT:
            ctx.messages[-1].answer_content = answer

    def invoke(self, query: str, thread_id: str | None = None) -> AgentState:
        """Sync wrapper."""
        return asyncio.run(self.ainvoke(query, thread_id))
