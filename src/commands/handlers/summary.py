"""Handler for the /summary command — summarize context and reset window."""

import concurrent.futures

from src.agents import QueryAgent
from src.commands.handlers.tokens import _count_context_tokens
from src.llm import get_llm
from src.memory.context import ConversationContext, MessageRole
from src.llm.prompts import SUMMARY_SYSTEM_PROMPT
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)





def _summarize_sync(ctx: ConversationContext) -> str:
    """Call LLM synchronously in a thread pool to avoid blocking the asyncio loop."""
    history = ctx.to_summary()
    if not history:
        return "（空对话，无内容可总结）"

    prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\n【对话历史】\n{history}"
    llm = get_llm(temperature=0.3)
    response = llm.invoke(prompt)
    return getattr(response, "content", "") or str(response)


def run(raw: str, agent: QueryAgent) -> bool:
    """Handle /summary command."""
    ctx = agent._get_context()
    old_tokens = _count_context_tokens(ctx)

    print("正在生成摘要...")
    summary_text = _executor.submit(_summarize_sync, ctx).result()

    ctx.messages.clear()
    ctx.add_assistant_message(
        content=summary_text,
        answer_content=summary_text,
    )

    new_tokens = _count_context_tokens(ctx)
    saved = old_tokens - new_tokens

    print(f"\n=== 摘要已生成 ===")
    print(f"\n{summary_text}")
    print(f"\n────────────────────")
    print(f"  Token: {old_tokens} -> {new_tokens}  (节省 ~{saved})")

    return False
