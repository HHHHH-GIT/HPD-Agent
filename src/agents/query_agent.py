import asyncio
import re

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

    def _extract_tool_summary(self, synthesis_prompt: str) -> str:
        """Parse tool names and paths from a synthesis prompt's tool log section.

        The synthesis prompt contains tool results in the form:
          [Tool: tool_name]\nresult
          [Tool: read_file(path='/...')]\ncontent
          [Tool: terminal(cmd='cat /...')]\noutput

        We extract unique (tool_name, path/cmd) pairs for the tool_summary field.
        """
        seen: set[str] = set()
        parts: list[str] = []

        # Pattern: [Tool: name(args)]\n or [Tool: name]\n
        for match in re.finditer(r"\[Tool:\s*(\w+)(?:\([^)]*\))?\]", synthesis_prompt):
            key = match.group(1)
            if key not in seen:
                seen.add(key)
                parts.append(key)

        # Also capture specific paths from read_file(path='...') and terminal(cmd='...')
        for match in re.finditer(r"read_file\s*\([^'\"]*['\"]([^'\"]+)['\"]", synthesis_prompt):
            path = match.group(1).strip()
            key = f"read_file: {path}"
            if key not in seen:
                seen.add(key)
                parts.append(key)

        for match in re.finditer(r"terminal\s*\(\s*cmd\s*=\s*'([^']+)'", synthesis_prompt):
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
        }
        config = {"configurable": {"thread_id": sid}}
        result = await self._app.ainvoke(initial_state, config=config)

        synthesis = result.get("synthesis_prompt", "")
        final_text = result.get("final_response") or synthesis

        # Track sub-task outputs for token accounting (not stored in messages,
        # but consumed by the synthesizer's final LLM call)
        outputs = result.get("sub_task_outputs", [])
        for o in outputs:
            o_dict = {
                "id": o.id,
                "name": o.name,
                "detail": o.detail,
                "summary": o.summary,
                "tools_used": o.tools_used,
                "expert_mode": o.expert_mode,
            }
            ctx.sub_task_outputs.append(o_dict)

        tool_summary = self._extract_tool_summary(synthesis) if synthesis else ""
        if final_text:
            ctx.add_assistant_message(
                content=synthesis if synthesis else final_text,
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
