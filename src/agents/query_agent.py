import asyncio

from langgraph.checkpoint.base import BaseCheckpointSaver

from src.core.state import AgentState
from src.memory import ConversationContext, MessageRole, get_checkpointer
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
        if "default" not in self._contexts:
            self._contexts["default"] = ConversationContext()

    def _get_context(self, thread_id: str | None = None) -> ConversationContext:
        """Return the rolling context for this thread, creating if needed."""
        sid = thread_id or self._current_session
        if sid not in self._contexts:
            self._contexts[sid] = ConversationContext()
        return self._contexts[sid]

    async def ainvoke(self, query: str, thread_id: str | None = None) -> AgentState:
        """Run the graph and return the final state.

        For simple tasks: final_response is populated — caller prints it.
        For complex tasks: synthesis_prompt is populated — caller must stream
        a synthesizer LLM call using that prompt.
        """
        sid = thread_id or self._current_session
        ctx = self._get_context(sid)
        ctx.add_user_message(query)

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
        if final_text:
            ctx.add_assistant_message(
                content=synthesis if synthesis else final_text,
                answer_content=final_text[:5000] if synthesis else None,
            )

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
