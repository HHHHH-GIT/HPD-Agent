"""LLM factory — all ChatOpenAI instances are created here.

Active model is read from the profile store singleton so that /model
switches take effect for every LLM call without code changes.

Token usage is tracked via monkey-patching so every LLM call (including
structured-output and streaming via astream_events) contributes tokens to
the global TokenTrackerCallback accumulator.
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel

from src.cli import get_renderer
from src.core.observability import TokenTrackerCallback

# ----------------------------------------------------------------------
# Token-tracker (global singleton, populated by monkey-patches below)
# ----------------------------------------------------------------------
_token_callback = TokenTrackerCallback()


# ----------------------------------------------------------------------
# Monkey-patch 1: _agenerate — captures non-streaming + structured-output calls
# ----------------------------------------------------------------------
def _install_token_tracker() -> None:
    original_agenerate = ChatOpenAI._agenerate

    async def tracked_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await original_agenerate(self, messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            msg: AIMessage = gen.message
            usage = getattr(msg, "usage_metadata", None) or {}
            tin  = usage.get("input_tokens", 0)
            tout = usage.get("output_tokens", 0)
            if tin or tout:
                model = getattr(self, "model_name", "") or getattr(self, "model", "") or ""
                _token_callback._accumulate(tin, tout, model)
        return result

    ChatOpenAI._agenerate = tracked_agenerate

    # ------------------------------------------------------------------
    # Monkey-patch 2: _convert_chunk_to_generation_chunk — captures streaming tokens
    # This method is called per chunk during astream_events, and usage
    # metadata (including final totals) is embedded in the last chunk.
    # ------------------------------------------------------------------
    original_convert = BaseChatOpenAI._convert_chunk_to_generation_chunk

    def tracked_convert(self, chunk, default_chunk_class, base_generation_info=None):
        result = original_convert(self, chunk, default_chunk_class, base_generation_info)
        chunk_usage = chunk.get("usage") if isinstance(chunk, dict) else None
        if chunk_usage:
            tin = chunk_usage.get("input_tokens", 0)
            tout = chunk_usage.get("output_tokens", 0)
            if tin or tout:
                model = getattr(self, "model_name", "") or getattr(self, "model", "") or ""
                _token_callback._accumulate(tin, tout, model)
        return result

    BaseChatOpenAI._convert_chunk_to_generation_chunk = tracked_convert


_install_token_tracker()


# ----------------------------------------------------------------------
# Rest of client.py
# ----------------------------------------------------------------------
_seen_tool_call_ids: set[str] = set()
"""Track which tool_call IDs have already been executed to prevent duplicate calls."""

DEFAULT_MAX_TOOL_ROUNDS = 30
DEFAULT_MAX_TOOL_CALLS = 80


def _reset_seen_tool_call_ids() -> None:
    """Reset the seen call-ID set before a new invoke_with_tools session."""
    _seen_tool_call_ids.clear()


def _resolve_api_key(profile_key: str) -> str:
    """Prefer the per-profile key; fall back to the legacy env var."""
    if profile_key:
        return profile_key
    for env_key in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        if env_key in os.environ:
            return os.environ[env_key]
    raise RuntimeError(
        "No API key configured. "
        "Set DEEPSEEK_API_KEY (or DASHSCOPE_API_KEY), "
        "or configure an api_key in your model profile via /model create."
    )


def _active_profile() -> object:
    """Import lazily to avoid circular imports at module startup."""
    from src.models import get_store
    return get_store().active_profile()


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    stream: bool = False,
) -> ChatOpenAI:
    """
    Build a ChatOpenAI client from the active profile.

    Any explicitly-passed parameter overrides the profile (for backwards
    compatibility with callers that pass model=/temperature=/base_url=).
    """
    profile = _active_profile()

    actual_model = model if model is not None else (getattr(profile, "model", "deepseek-v4-flash") if profile else "deepseek-v4-flash")
    actual_temp  = temperature if temperature is not None else (getattr(profile, "temperature", 0.0) if profile else 0.0)
    actual_base  = base_url if base_url is not None else (getattr(profile, "base_url", "https://api.deepseek.com") if profile else "https://api.deepseek.com")
    actual_key   = _resolve_api_key(getattr(profile, "api_key", "") if profile else "")
    actual_eb    = getattr(profile, "extra_body", {}) if profile else {}

    return ChatOpenAI(
        model=actual_model,
        temperature=actual_temp,
        api_key=actual_key,
        base_url=actual_base,
        stream=stream,
        extra_body=actual_eb,
        request_timeout=60,
    )


def get_structured_llm(
    schema: type[BaseModel],
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
) -> ChatOpenAI:
    """Return an LLM bound to output a specific Pydantic schema."""
    llm = get_llm(model=model, temperature=temperature, base_url=base_url)
    return llm.with_structured_output(schema, method="function_calling")


def get_llm_with_tools(
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    tools: list[BaseTool] | None = None,
    stream: bool = False,
) -> ChatOpenAI:
    """Build a ChatOpenAI client and bind tools to it."""
    llm = get_llm(model=model, temperature=temperature, base_url=base_url)
    if tools:
        llm = llm.bind_tools(tools)
    return llm


async def _finalize_with_available_context(
    messages: list,
    reason: str,
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
) -> str:
    """Produce a best-effort final answer after tool budget exhaustion.

    Uses the already-collected tool results in the conversation and performs
    one last no-tool model call so the current task can still conclude with a
    reasoned answer instead of returning only partial tool traces.
    """
    llm = get_llm(model=model, temperature=temperature, base_url=base_url)
    finalize_prompt = (
        "工具调用预算已经耗尽，不能再调用任何工具。\n"
        f"原因：{reason}\n\n"
        "请基于本轮对话中已经存在的工具返回结果，给出当前任务的最佳可得结论。"
        "如果信息不完整或无法完全确认，请明确指出哪些部分仍未验证。"
        "不要再请求工具，也不要假装已获取未出现的信息。"
    )
    response = await llm.ainvoke([*messages, HumanMessage(content=finalize_prompt)])
    return getattr(response, "content", "") or ""


async def invoke_with_tools(
    prompt: str,
    tools: list[BaseTool] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    stream: bool = False,
    max_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    on_budget_exceeded: str = "finalize",
) -> tuple[str, str]:
    """Invoke the LLM with tool-calling support.

    Args:
        prompt: The user prompt to send.
        tools: List of tools available to the LLM.
        model / temperature / base_url: Override active profile settings.
        stream: If True, yields chunks via print; returns full content.

    Returns:
        (final_text, tool_calls_log) — the LLM's final text response and a
        concatenated string of all tool results for context.
    """
    _reset_seen_tool_call_ids()
    llm = get_llm_with_tools(
        model=model, temperature=temperature, base_url=base_url,
        tools=tools, stream=stream,
    )

    messages = [HumanMessage(content=prompt)]
    tool_results: list[str] = []
    full_content = ""
    active_model = ""

    tool_call_count = 0
    for _ in range(max_rounds):
        response = await llm.ainvoke(messages)

        usage = getattr(response, "usage_metadata", None) or {}
        if usage:
            input_tokens  = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            if input_tokens or output_tokens:
                active_model = getattr(llm, "model_name", "")
                _token_callback._accumulate(input_tokens, output_tokens, active_model)

        content = getattr(response, "content", "") or ""
        tool_calls = getattr(response, "tool_calls", []) or []

        if not tool_calls:
            full_content = content
            break

        if tool_call_count + len(tool_calls) > max_tool_calls:
            reason = (
                f"Tool budget exceeded: {tool_call_count + len(tool_calls)} calls "
                f"would exceed limit {max_tool_calls}."
            )
            if on_budget_exceeded == "raise":
                raise RuntimeError(reason)
            print(f"[DEBUG] {reason}")
            full_content = await _finalize_with_available_context(
                messages,
                reason=reason,
                model=model,
                temperature=temperature,
                base_url=base_url,
            )
            break

        for call in tool_calls:
            tool_call_count += 1
            call_id = call.get("id", "")
            name = call.get("name") or ""
            args = call.get("args") or {}
            tool = None
            if tools:
                for t in tools:
                    if t.name == name:
                        tool = t
                        break
            if tool is None:
                result = f"[Error] Tool '{name}' not found"
                print(f"[DEBUG] Tool '{name}' not found")
            elif call_id in _seen_tool_call_ids:
                result = (
                    f"[Error] Duplicate tool call detected for '{name}' "
                    f"with id={call_id}. This operation was previously "
                    f"cancelled or failed. Do not retry the same action."
                )
                print(f"[DEBUG] Tool '{name}' duplicate call ignored (id={call_id})")
                break
            else:
                _seen_tool_call_ids.add(call_id)
                if name == "terminal":
                    cmd = args.get("cmd", "")
                    print(f"[DEBUG] Tool '{name}' → terminal command: {cmd}")
                    safe_prefixes = (
                        "pwd", "ls", "cat", "echo", "date", "whoami",
                        "head", "tail", "less", "file ", "stat ",
                        "uname", "id", "hostname", "env", "printenv",
                    )
                    if cmd.strip().startswith(safe_prefixes) or cmd.strip().startswith("cd "):
                        result = tool.invoke(args)
                        success = not str(result).startswith("[Error]")
                    else:
                        confirm = get_renderer().confirm(f"Allow terminal command? {cmd}")
                        if not confirm:
                            result = "[Cancelled] User declined to execute the terminal command"
                            print(f"[DEBUG] Tool '{name}' cancelled by user")
                        else:
                            result = tool.invoke(args)
                            success = not str(result).startswith("[Error]") and not str(result).startswith("[Cancelled]")
                            if not success:
                                print(f"[DEBUG] Tool '{name}' failed: {result}")
                else:
                    result = tool.invoke(args)
                    success = not str(result).startswith("[Error]")
                    if success:
                        print(f"[DEBUG] Tool '{name}' succeeded")
                    else:
                        print(f"[DEBUG] Tool '{name}' failed:\n{str(result)[:4000]}")

            messages.append(AIMessage(content="", tool_calls=[call]))
            messages.append(ToolMessage(name=name, content=str(result), tool_call_id=call_id))
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
            tool_results.append(f"[Tool: {name}({args_str})]\n{result}")

    else:
        reason = f"Tool interaction exceeded max rounds ({max_rounds}) without final response."
        if on_budget_exceeded == "raise":
            raise RuntimeError(reason)
        print(f"[DEBUG] {reason}")
        full_content = await _finalize_with_available_context(
            messages,
            reason=reason,
            model=model,
            temperature=temperature,
            base_url=base_url,
        )

    return full_content, "\n\n".join(tool_results)
