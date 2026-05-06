import asyncio
import re

from src.core.observability import get_tracer

from langgraph.checkpoint.base import BaseCheckpointSaver

from src.core.state import AgentState
from src.memory import ConversationContext, MessageRole, get_checkpointer
from src.memory.session_store import _project_hash, delete as delete_session, list_sessions, load as load_session, save as save_session
from src.system_info import build_boot_prompt
from src.workflow import build_graph


class QueryAgent:
    """High-level agent facade.

    Encapsulates graph compilation, checkpointer wiring,
    conversation context threading, and the public `run` / `ainvoke` interface.
    """

    def __init__(self, checkpointer: BaseCheckpointSaver | None = None):
        graph = build_graph()
        checkpointer = checkpointer or get_checkpointer()
        self._app = graph.compile(checkpointer=checkpointer)
        self._contexts: dict[str, ConversationContext] = {}
        self._current_session: str = "default"
        self._session_boot_done: set[str] = set()
        self._auto_save_enabled: bool = True
        self._project_hash: str = _project_hash()
        self._load_all()

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

    def _summarize_tools_from_subtasks(self, outputs: list) -> str:
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
            tool_log = getattr(output, "tool_log", "") or output.get("tool_log", "") if isinstance(output, dict) else ""
            if tool_log:
                extracted = self._extract_tool_summary(tool_log)
                for piece in extracted.split(","):
                    if piece.strip():
                        add(piece)

            tools_used = getattr(output, "tools_used", None)
            if tools_used is None and isinstance(output, dict):
                tools_used = output.get("tools_used", [])

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
          [Tool: tool_name]\nresult
          [Tool: read_file(path='/...')]\ncontent
          [Tool: terminal(cmd='cat /...')]\noutput

        We extract unique (tool_name, path/cmd) pairs for the tool_summary field.
        """
        seen: set[str] = set()
        parts: list[str] = []

        # Pattern: [Tool: name(args)]\n or [Tool: name]\n
        for match in re.finditer(r"\[Tool:\s*(\w+)(?:\([^)]*\))?\]", tool_log_text):
            key = match.group(1)
            if key not in seen:
                seen.add(key)
                parts.append(key)

        # Also capture specific paths from read_file(path='...') and terminal(cmd='...')
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

    async def ainvoke(self, query: str, thread_id: str | None = None) -> AgentState:
        """Run the graph and return the final state.

        For simple tasks: final_response is populated — caller prints it.
        For complex tasks: synthesis_prompt is populated — caller must stream
        a synthesizer LLM call using that prompt.
        """
        sid = thread_id or self._current_session
        ctx = self._get_context(sid)
        self._backfill_missing_tool_summary(ctx)
        ctx.add_user_message(query)

        if sid not in self._session_boot_done:
            self._session_boot_done.add(sid)
            from src.memory.context import Message
            ctx.messages.insert(
                0,
                Message(role=MessageRole.ASSISTANT, content=build_boot_prompt()),
            )

        initial_state: AgentState = {
            "input": query,
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
        result = await self._app.ainvoke(initial_state, config=config)

        synthesis = result.get("synthesis_prompt", "")
        final_text = result.get("final_response") or synthesis

        # Track sub-task outputs for token accounting (not stored in messages,
        # but consumed by the synthesizer's final LLM call).
        # Only keep summary-level info to avoid context bloat — full details
        # are in the synthesis prompt which is ephemeral.
        outputs = result.get("sub_task_outputs", [])
        for o in outputs:
            o_dict = {
                "id": o.id,
                "name": o.name,
                "detail": o.detail,  # now compressed: reasoning + tool chain only
                "summary": o.summary,
                "tools_used": o.tools_used,
                "expert_mode": o.expert_mode,
            }
            ctx.sub_task_outputs.append(o_dict)
        # Cap accumulated sub-task outputs to the most recent 50
        if len(ctx.sub_task_outputs) > 50:
            ctx.sub_task_outputs = ctx.sub_task_outputs[-50:]

        tool_summary = self._extract_tool_summary(synthesis) if synthesis else ""
        if not tool_summary and outputs:
            tool_summary = self._summarize_tools_from_subtasks(outputs)
        if not tool_summary:
            for output in reversed(result.get("outputs", [])):
                if output.node == "direct_answer":
                    tool_log = output.result.get("tool_calls", "")
                    if tool_log:
                        tool_summary = self._extract_tool_summary(tool_log)
                    break
        if final_text:
            # Store compact content — the full synthesis prompt is ephemeral.
            # to_summary() already prefers answer_content, so content is just a fallback.
            compact = final_text[:2000] if synthesis else final_text
            ctx.add_assistant_message(
                content=compact,
                answer_content=final_text[:5000] if synthesis else None,
                tool_summary=tool_summary or None,
            )

        if self._auto_save_enabled:
            save_session(ctx, sid, self._project_hash)

        return result

    def store_streamed_answer(self, answer: str, thread_id: str | None = None) -> None:
        """Backfill the answer_content for the most recent assistant message."""
        sid = thread_id or self._current_session
        ctx = self._get_context(sid)
        if ctx.messages and ctx.messages[-1].role == MessageRole.ASSISTANT:
            ctx.messages[-1].answer_content = answer

    def invoke(self, query: str, thread_id: str | None = None) -> AgentState:
        """Sync wrapper."""
        return asyncio.run(self.ainvoke(query, thread_id))
