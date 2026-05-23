"""Handler for the /summary command."""

from __future__ import annotations

from src.agents import QueryAgent
from src.cli import get_renderer
from src.commands.handlers.tokens import _count_context_tokens, _count_tokens
from src.memory.compaction import (
    CompactionConfig,
    compact_context,
    default_artifact_store,
    summarize_compaction_with_llm,
)
from src.memory.context import ConversationContext


def _count_sub_task_tokens(ctx: ConversationContext) -> int:
    total = 0
    for output in ctx.sub_task_outputs:
        total += _count_tokens(f"[sub-task {output['id']}: {output['name']}]\n{output['detail']}")
    return total


def run(raw: str, agent: QueryAgent) -> bool:
    """Handle /summary command by reusing the compaction pipeline."""
    renderer = get_renderer()
    ctx = agent._get_context()
    old_msg_tokens = _count_context_tokens(ctx)
    old_sub_task_tokens = _count_sub_task_tokens(ctx)
    old_total = old_msg_tokens + old_sub_task_tokens

    renderer.info("Generating summary...")
    compact_context(
        ctx,
        session_id=agent._current_session,
        project_hash=agent._project_hash,
        artifact_store=default_artifact_store(agent._project_hash),
        token_counter=_count_tokens,
        config=CompactionConfig(
            max_tokens=agent._model_context_window(),
            preserve_recent_turns=0,
        ),
        force=True,
        summarizer=summarize_compaction_with_llm,
    )

    full_content = ctx.session_summary
    if ctx.toolchain_summary:
        full_content = f"Toolchain\n{ctx.toolchain_summary}\n\n{full_content}"
    agent.save_current_session()

    new_msg_tokens = _count_context_tokens(ctx)
    saved_total = old_total - new_msg_tokens

    renderer.render_summary(
        summary_text=full_content,
        old_msg_tokens=old_msg_tokens,
        new_msg_tokens=new_msg_tokens,
        old_sub_task_tokens=old_sub_task_tokens,
        old_total=old_total,
        saved_total=saved_total,
    )

    return False
