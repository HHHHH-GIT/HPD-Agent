"""LLM factory — all ChatOpenAI instances are created here.

Active model is read from the profile store singleton so that /model
switches take effect for every LLM call without code changes.

Token usage is tracked via monkey-patching so every LLM call (including
structured-output and streaming via astream_events) contributes tokens to
the global TokenTrackerCallback accumulator.
"""

import json
import os
from collections.abc import Callable
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel

from src.cli import get_renderer
from src.core.observability import TokenTrackerCallback
from src.core.riskless import is_riskless_enabled
from src.llm.tool_context import (
    ToolContextPolicy,
    ToolLoopContextManager,
    count_tokens,
)
from src.memory.compaction import SessionArtifactStore
from src.tools.invocation import invoke_tool
from src.tools.terminal_policy import should_confirm_terminal_command

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
            tin = usage.get("input_tokens", 0)
            tout = usage.get("output_tokens", 0)
            if tin or tout:
                model = (
                    getattr(self, "model_name", "") or getattr(self, "model", "") or ""
                )
                _token_callback._accumulate(tin, tout, model, source="api")
        return result

    ChatOpenAI._agenerate = tracked_agenerate

    # ------------------------------------------------------------------
    # Monkey-patch 2: _convert_chunk_to_generation_chunk — captures streaming tokens
    # This method is called per chunk during astream_events, and usage
    # metadata (including final totals) is embedded in the last chunk.
    # ------------------------------------------------------------------
    original_convert = BaseChatOpenAI._convert_chunk_to_generation_chunk

    def tracked_convert(self, chunk, default_chunk_class, base_generation_info=None):
        result = original_convert(
            self, chunk, default_chunk_class, base_generation_info
        )
        chunk_usage = chunk.get("usage") if isinstance(chunk, dict) else None
        if chunk_usage:
            tin = chunk_usage.get("input_tokens", 0)
            tout = chunk_usage.get("output_tokens", 0)
            if tin or tout:
                model = (
                    getattr(self, "model_name", "") or getattr(self, "model", "") or ""
                )
                _token_callback._accumulate(tin, tout, model, source="api")
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


_SOCKS_PROXY_KEYS = ("ALL_PROXY", "all_proxy")


def _strip_socks_proxy() -> dict[str, str]:
    """remove any SOCKS proxy settings from the environment, returning the stripped values for later restoration."""
    stripped: dict[str, str] = {}
    for key in _SOCKS_PROXY_KEYS:
        val = os.environ.get(key, "")
        if val.startswith(("socks://", "socks5://")):
            stripped[key] = os.environ.pop(key)
    return stripped


def _restore_socks_proxy(stripped: dict[str, str]) -> None:
    """restore any previously-stripped SOCKS proxy settings back into the environment."""
    os.environ.update(stripped)


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

    actual_model = (
        model
        if model is not None
        else (
            getattr(profile, "model", "deepseek-v4-flash")
            if profile
            else "deepseek-v4-flash"
        )
    )
    actual_temp = (
        temperature
        if temperature is not None
        else (getattr(profile, "temperature", 0.0) if profile else 0.0)
    )
    actual_base = (
        base_url
        if base_url is not None
        else (
            getattr(profile, "base_url", "https://api.deepseek.com")
            if profile
            else "https://api.deepseek.com"
        )
    )
    actual_key = _resolve_api_key(getattr(profile, "api_key", "") if profile else "")
    actual_eb = getattr(profile, "extra_body", {}) if profile else {}

    stripped = _strip_socks_proxy()
    try:
        return cast(
            ChatOpenAI,
            ChatOpenAI(
                model=actual_model,
                temperature=actual_temp,
                api_key=actual_key,
                base_url=actual_base,
                streaming=stream,
                extra_body=actual_eb,
                timeout=60,
            ),
        )
    finally:
        _restore_socks_proxy(stripped)


def get_structured_llm(
    schema: type[BaseModel],
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
) -> ChatOpenAI:
    """Return an LLM bound to output a specific Pydantic schema."""
    llm = get_llm(model=model, temperature=temperature, base_url=base_url)
    return cast(
        ChatOpenAI, llm.with_structured_output(schema, method="function_calling")
    )


def get_llm_with_tools(
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    tools: list[BaseTool] | None = None,
    stream: bool = False,
) -> ChatOpenAI:
    """Build a ChatOpenAI client and bind tools to it."""
    llm = get_llm(
        model=model,
        temperature=temperature,
        base_url=base_url,
        stream=stream,
    )
    if tools:
        llm = cast(ChatOpenAI, llm.bind_tools(tools))
    return llm


async def _stream_llm_response(
    llm: Any,
    messages: list,
    on_token: Callable[[str], None] | None = None,
) -> AIMessage:
    """Stream a chat response and return the merged message for tool handling."""
    merged = None
    content_parts: list[str] = []

    async for chunk in llm.astream(messages):
        if merged is None:
            merged = chunk
        else:
            try:
                merged = merged + chunk
            except TypeError:
                merged = chunk

        token = getattr(chunk, "content", "") or ""
        if isinstance(token, str) and token:
            content_parts.append(token)
            if on_token:
                on_token(token)

    if merged is None:
        return AIMessage(content="")

    content = getattr(merged, "content", "") or "".join(content_parts)
    tool_calls = getattr(merged, "tool_calls", []) or []
    return AIMessage(content=content, tool_calls=tool_calls)


async def _invoke_llm_response(
    llm: Any,
    messages: list,
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> AIMessage:
    """Invoke the model, optionally streaming text chunks to the caller."""
    if stream:
        return await _stream_llm_response(llm, messages, on_token=on_token)
    return cast(AIMessage, await llm.ainvoke(messages))


def _active_span_token_snapshot() -> tuple[str, int, int]:
    """Return the active span id and current token counters for fallback estimation."""
    from src.core.observability import get_tracer

    tracer = get_tracer()
    span_id = tracer._active_span_id.get() or ""
    span = tracer.get_span(span_id) if span_id else None
    if span is None:
        return "", 0, 0
    return span_id, span.tokens_in, span.tokens_out


def _estimate_llm_usage(messages: list, response: Any, llm: Any) -> tuple[int, int]:
    """Estimate LLM usage when the provider does not return usage metadata."""
    prompt_text = "\n".join(
        str(getattr(message, "content", "") or "") for message in messages
    )
    response_text = str(getattr(response, "content", "") or "")
    tool_calls = getattr(response, "tool_calls", []) or []
    if tool_calls:
        response_text = f"{response_text}\n{json.dumps(tool_calls, ensure_ascii=False, default=str)}"
    input_tokens = count_tokens(prompt_text)
    output_tokens = count_tokens(response_text)
    return input_tokens, output_tokens


def _record_estimated_usage_if_missing(
    *,
    llm: Any,
    messages: list,
    response: Any,
    before: tuple[str, int, int],
) -> None:
    """Fallback to estimated usage when the active span did not receive API usage."""
    span_id, before_in, before_out = before
    if not span_id:
        return

    from src.core.observability import get_tracer

    tracer = get_tracer()
    span = tracer.get_span(span_id)
    if span is None:
        return
    if span.tokens_in != before_in or span.tokens_out != before_out:
        return

    input_tokens, output_tokens = _estimate_llm_usage(messages, response, llm)
    if input_tokens or output_tokens:
        model = getattr(llm, "model_name", "") or getattr(llm, "model", "") or ""
        tracer.record_tokens(
            span_id,
            tokens_in=input_tokens,
            tokens_out=output_tokens,
            model=model,
            token_source="estimated",
        )


async def _finalize_with_available_context(
    messages: list,
    reason: str,
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Produce a best-effort final answer after tool budget exhaustion.

    Uses the already-collected tool results in the conversation and performs
    one last no-tool model call so the current task can still conclude with a
    reasoned answer instead of returning only partial tool traces.
    """
    llm = get_llm(
        model=model,
        temperature=temperature,
        base_url=base_url,
        stream=stream,
    )
    finalize_prompt = (
        "工具调用预算已经耗尽，不能再调用任何工具。\n"
        f"原因：{reason}\n\n"
        "请基于本轮对话中已经存在的工具返回结果，给出当前任务的最佳可得结论。"
        "如果信息不完整或无法完全确认，请明确指出哪些部分仍未验证。"
        "不要再请求工具，也不要假装已获取未出现的信息。"
    )
    final_messages = [*messages, HumanMessage(content=finalize_prompt)]
    before_tokens = _active_span_token_snapshot()
    response = await _invoke_llm_response(
        llm,
        final_messages,
        stream=stream,
        on_token=on_token,
    )
    _record_estimated_usage_if_missing(
        llm=llm,
        messages=final_messages,
        response=response,
        before=before_tokens,
    )
    return getattr(response, "content", "") or ""


async def invoke_with_tools(
    prompt: str,
    tools: list[BaseTool] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    stream: bool = False,
    max_rounds: int | None = None,
    max_tool_calls: int | None = None,
    on_budget_exceeded: str = "finalize",
    on_token: Callable[[str], None] | None = None,
    context_window: int | None = None,
    tool_context_policy: ToolContextPolicy | None = None,
    artifact_store: SessionArtifactStore | None = None,
) -> tuple[str, str]:
    """Invoke the LLM with tool-calling support.

    Args:
        prompt: The user prompt to send.
        tools: List of tools available to the LLM.
        model / temperature / base_url: Override active profile settings.
        stream: If True, streams text chunks through on_token; returns full content.

    Returns:
        (final_text, tool_calls_log) — the LLM's final text response and a
        concatenated string of all tool results for context.
    """
    _reset_seen_tool_call_ids()
    llm = get_llm_with_tools(
        model=model,
        temperature=temperature,
        base_url=base_url,
        tools=tools,
        stream=stream,
    )

    messages: list[Any] = [HumanMessage(content=prompt)]
    tool_results: list[str] = []
    full_content = ""
    effective_max_rounds = DEFAULT_MAX_TOOL_ROUNDS if max_rounds is None else max_rounds
    effective_max_tool_calls = (
        DEFAULT_MAX_TOOL_CALLS if max_tool_calls is None else max_tool_calls
    )
    effective_tool_context_policy: ToolContextPolicy = tool_context_policy or "auto"
    if max_rounds is None or max_tool_calls is None or tool_context_policy is None:
        try:
            from src.core.tool_budget import get_tool_budget_profile

            budget = get_tool_budget_profile()
            if max_rounds is None:
                effective_max_rounds = budget.max_rounds
            if max_tool_calls is None:
                effective_max_tool_calls = budget.max_tool_calls
            if tool_context_policy is None:
                effective_tool_context_policy = budget.tool_context_policy
        except Exception:
            pass
    tool_context = ToolLoopContextManager(
        context_window=context_window or _model_context_window(),
        policy=effective_tool_context_policy,
        artifact_store=artifact_store,
        token_counter=count_tokens,
    )

    tool_call_count = 0
    for _ in range(effective_max_rounds):
        before_tokens = _active_span_token_snapshot()
        response = await _invoke_llm_response(
            llm,
            messages,
            stream=stream,
            on_token=on_token,
        )
        _record_estimated_usage_if_missing(
            llm=llm,
            messages=messages,
            response=response,
            before=before_tokens,
        )

        content = getattr(response, "content", "") or ""
        tool_calls = getattr(response, "tool_calls", []) or []

        if not tool_calls:
            full_content = content
            break

        if tool_call_count + len(tool_calls) > effective_max_tool_calls:
            reason = (
                f"Tool budget exceeded: {tool_call_count + len(tool_calls)} calls "
                f"would exceed limit {effective_max_tool_calls}."
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
                stream=stream,
                on_token=on_token,
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
                    decision = should_confirm_terminal_command(
                        str(cmd),
                        riskless_enabled=is_riskless_enabled(),
                        project_dir=os.getcwd(),
                    )
                    if not decision.requires_confirmation:
                        result = await invoke_tool(tool, args)
                        success = not str(result).startswith("[Error]")
                    else:
                        confirm = get_renderer().confirm(
                            f"Allow terminal command? {cmd} ({decision.reason})"
                        )
                        if not confirm:
                            result = "[Cancelled] User declined to execute the terminal command"
                            print(f"[DEBUG] Tool '{name}' cancelled by user")
                        else:
                            result = await invoke_tool(tool, args)
                            success = not str(result).startswith("[Error]") and not str(
                                result
                            ).startswith("[Cancelled]")
                            if not success:
                                print(f"[DEBUG] Tool '{name}' failed: {result}")
                else:
                    result = await invoke_tool(tool, args)
                    success = not str(result).startswith("[Error]")
                    if success:
                        print(f"[DEBUG] Tool '{name}' succeeded")
                    else:
                        print(f"[DEBUG] Tool '{name}' failed:\n{str(result)[:4000]}")

            ok, context_reason = tool_context.append_tool_interaction(
                messages,
                call=call,
                name=name,
                args=dict(args),
                result=result,
            )
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
            tool_results.append(
                f"[Tool: {name}({args_str})]\n{messages[-1].content if messages else result}"
            )
            _record_tool_context_metadata(tool_context)
            if not ok:
                reason = f"[ToolContextExhausted] {context_reason}"
                tool_results.append(reason)
                if on_budget_exceeded == "raise":
                    raise RuntimeError(reason)
                tool_context.prepare_final_messages(messages, reason)
                full_content = await _finalize_with_available_context(
                    messages,
                    reason=reason,
                    model=model,
                    temperature=temperature,
                    base_url=base_url,
                    stream=stream,
                    on_token=on_token,
                )
                _record_tool_context_metadata(tool_context)
                return full_content, "\n\n".join(tool_results)

    else:
        reason = (
            f"Tool interaction exceeded max rounds ({effective_max_rounds}) "
            "without final response."
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
            stream=stream,
            on_token=on_token,
        )

    _record_tool_context_metadata(tool_context)
    return full_content, "\n\n".join(tool_results)


def _model_context_window() -> int:
    """Return active model context window, defaulting to 128k."""
    try:
        profile = _active_profile()
        extra_body = getattr(profile, "extra_body", {}) if profile else {}
        if isinstance(extra_body, dict):
            for key in ("context_window", "max_context_tokens", "max_input_tokens"):
                value = extra_body.get(key)
                if value:
                    return int(value)
    except Exception:
        pass
    return 128_000


def _record_tool_context_metadata(manager: ToolLoopContextManager) -> None:
    from src.core.observability import get_tracer

    tracer = get_tracer()
    span_id = tracer._active_span_id.get()
    span = tracer.get_span(span_id or "")
    if span is None:
        return
    span.metadata.update(
        {
            "tool_context_tokens": manager.stats.tool_context_tokens,
            "tool_artifact_tokens": manager.stats.tool_artifact_tokens,
            "tool_messages_compacted": manager.stats.tool_messages_compacted,
            "tool_context_exhausted": manager.stats.tool_context_exhausted,
        }
    )
