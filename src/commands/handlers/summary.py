"""Handler for the /summary command — summarize context and reset window."""

import concurrent.futures

from src.agents import QueryAgent
from src.commands.handlers.tokens import _count_context_tokens, _count_tokens
from src.llm import get_llm
from src.memory.context import ConversationContext, MessageRole
from src.llm.prompts import SUMMARY_SYSTEM_PROMPT
from src.llm.prompts import TOOLCHAIN_SUMMARY_PROMPT
from src.system_info import build_boot_prompt


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _build_toolchain(ctx: ConversationContext) -> str:
    """Build a concise tool chain summary from sub-task outputs via LLM."""
    if not ctx.sub_task_outputs:
        return ""

    lines = []
    for output in ctx.sub_task_outputs:
        parts = [f"子任务{output['id']}: {output['name']}"]
        if output.get("tools_used"):
            parts.append(f"  工具: {', '.join(output['tools_used'])}")
        if output.get("summary"):
            parts.append(f"  摘要: {output['summary']}")
        lines.append("\n".join(parts))

    toolchain_text = "\n\n".join(lines)
    prompt = f"{TOOLCHAIN_SUMMARY_PROMPT}\n\n【子任务执行记录】\n{toolchain_text}"
    llm = get_llm(temperature=0.3)
    response = llm.invoke(prompt)
    return getattr(response, "content", "") or str(response)


def _count_sub_task_tokens(ctx: ConversationContext) -> int:
    """Count tokens for accumulated sub-task outputs."""
    total = 0
    for output in ctx.sub_task_outputs:
        total += _count_tokens(f"[子任务 {output['id']}: {output['name']}]\n{output['detail']}")
    return total


def _summarize_sync(ctx: ConversationContext) -> str:
    """Call LLM synchronously in a thread pool to avoid blocking the asyncio loop."""
    history = ctx.to_summary()
    sub_tasks = ctx.to_sub_tasks_summary()

    if not history and not sub_tasks:
        return "（空对话，无内容可总结）"

    sections = []
    if history:
        sections.append(f"【对话历史】\n{history}")
    if sub_tasks:
        sections.append(f"【子任务结果】\n{sub_tasks}")

    prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\n{'='*40}\n".join(sections)
    llm = get_llm(temperature=0.3)
    response = llm.invoke(prompt)
    return getattr(response, "content", "") or str(response)


def run(raw: str, agent: QueryAgent) -> bool:
    """Handle /summary command."""
    ctx = agent._get_context()
    old_msg_tokens = _count_context_tokens(ctx)
    old_sub_task_tokens = _count_sub_task_tokens(ctx)
    old_total = old_msg_tokens + old_sub_task_tokens

    print("正在生成摘要...")
    toolchain = _executor.submit(_build_toolchain, ctx).result()
    summary_text = _executor.submit(_summarize_sync, ctx).result()

    ctx.messages.clear()
    ctx.sub_task_outputs.clear()

    # Re-inject boot prompt so it survives the summary reset
    from src.memory.context import Message
    ctx.messages.insert(
        0,
        Message(role=MessageRole.ASSISTANT, content=build_boot_prompt()),
    )

    full_content = summary_text
    if toolchain:
        full_content = f"【工具调用链】\n{toolchain}\n\n【对话摘要】\n {summary_text}"

    ctx.add_assistant_message(
        content=full_content,
        answer_content=summary_text,
    )

    new_msg_tokens = _count_context_tokens(ctx)
    saved_total = old_total - new_msg_tokens
    saved_msg = old_msg_tokens - new_msg_tokens

    print(f"\n=== 摘要已生成 ===")
    print(f"\n{full_content}")
    print(f"\n────────────────────")
    msg_saved_pct = (saved_msg / old_msg_tokens * 100) if old_msg_tokens else 0
    sub_saved_pct = (old_sub_task_tokens / old_total * 100) if old_total else 0
    print(f"  Conversation history:  {old_msg_tokens:>6} -> {new_msg_tokens:>6}  (节省 ~{saved_msg}, {msg_saved_pct:.1f}%)")
    if old_sub_task_tokens > 0:
        print(f"  Sub-task results:     {old_sub_task_tokens:>6} -> {0:>6}  (节省 ~{old_sub_task_tokens}, {sub_saved_pct:.1f}%)")
    print(f"  Total:               {old_total:>6} -> {new_msg_tokens:>6}  (节省 ~{saved_total})")

    return False
