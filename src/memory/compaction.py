"""Budget-aware context compaction and artifact persistence."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.memory.context import ContextArtifact, ConversationContext, Message

TokenCounter = Callable[[str], int]


@dataclass
class CompactionConfig:
    """Controls when and how resident context is compacted."""

    max_tokens: int = 128_000
    precompact_ratio: float = 0.70
    force_ratio: float = 0.90
    preserve_recent_turns: int = 3


@dataclass
class CompactionResult:
    """Result metadata returned by a compaction run."""

    compacted: bool
    before_tokens: int
    after_tokens: int
    artifact_count: int
    reason: str = ""


@dataclass
class CompactionPayload:
    """Resident and non-resident material sent to a summarizer."""

    existing_session_summary: str
    existing_toolchain_summary: str
    history_text: str
    sub_tasks_text: str
    toolchain_text: str
    artifact_refs: list[str]


@dataclass
class CompactionSummary:
    """Structured summary produced by an LLM or fallback summarizer."""

    session_summary: str
    toolchain_summary: str


CompactionSummarizer = Callable[[CompactionPayload], CompactionSummary]


class SessionArtifactStore:
    """Writes non-resident session artifacts under a project-local root."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        session_id: str,
        project_hash: str,
        kind: str,
        title: str,
        content: str,
        token_count: int,
    ) -> ContextArtifact:
        artifact_id = f"{kind}-{uuid.uuid4().hex[:10]}"
        safe_session = _safe_part(session_id)
        safe_project = _safe_part(project_hash)
        rel = Path(safe_project) / safe_session / f"{artifact_id}.txt"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ContextArtifact(
            artifact_id=artifact_id,
            kind=kind,
            title=title,
            content_ref=rel.as_posix(),
            token_count=token_count,
        )


def compact_context(
    ctx: ConversationContext,
    *,
    session_id: str,
    project_hash: str,
    artifact_store: SessionArtifactStore,
    token_counter: TokenCounter,
    config: CompactionConfig | None = None,
    force: bool = False,
    summarizer: CompactionSummarizer | None = None,
) -> CompactionResult:
    """Compact old resident context and task details into session artifacts."""
    config = config or CompactionConfig()
    before_tokens = _resident_tokens(ctx, token_counter)
    should_compact = force or before_tokens >= int(config.max_tokens * config.precompact_ratio)
    if not should_compact:
        return CompactionResult(False, before_tokens, before_tokens, len(ctx.artifacts), "below_threshold")

    preserve_count = max(0, config.preserve_recent_turns) * 2
    old_messages = ctx.messages[:-preserve_count] if preserve_count else list(ctx.messages)
    recent_messages = ctx.messages[-preserve_count:] if preserve_count else []
    old_sub_tasks = list(ctx.sub_task_outputs)

    created: list[ContextArtifact] = []
    if old_messages:
        content = _render_messages(old_messages)
        created.append(
            artifact_store.write(
                session_id=session_id,
                project_hash=project_hash,
                kind="history",
                title="Compacted conversation history",
                content=content,
                token_count=token_counter(content),
            )
        )

    for output in old_sub_tasks:
        content = _render_sub_task(output)
        title = f"Sub-task {output.get('id', '?')}: {output.get('name', '')}".strip()
        created.append(
            artifact_store.write(
                session_id=session_id,
                project_hash=project_hash,
                kind="sub_task",
                title=title,
                content=content,
                token_count=token_counter(content),
            )
        )

    if old_messages or old_sub_tasks:
        summary = _summarize_compacted_material(
            ctx,
            old_messages,
            old_sub_tasks,
            created,
            summarizer,
        )
        ctx.session_summary = summary.session_summary
        ctx.toolchain_summary = summary.toolchain_summary
        ctx.messages = recent_messages
        ctx.sub_task_outputs.clear()
        ctx.artifacts.extend(created)
        ctx.artifact_total_tokens = sum(artifact.token_count for artifact in ctx.artifacts)

    after_tokens = _resident_tokens(ctx, token_counter)
    return CompactionResult(
        compacted=bool(old_messages or old_sub_tasks),
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        artifact_count=len(created),
        reason="forced" if force else "threshold",
    )


def default_artifact_store(project_hash: str) -> SessionArtifactStore:
    """Return the default artifact store used by CLI sessions."""
    return SessionArtifactStore(Path.home() / ".hpagent" / "sessions" / project_hash / "artifacts")


def summarize_compaction_with_llm(payload: CompactionPayload) -> CompactionSummary:
    """Generate structured compaction summaries with the active LLM."""
    from src.llm import get_llm
    from src.llm.prompts import SUMMARY_SYSTEM_PROMPT, TOOLCHAIN_SUMMARY_PROMPT

    llm = get_llm(temperature=0.2)
    sections = [
        SUMMARY_SYSTEM_PROMPT,
        "\n请生成新的长期会话摘要，必须覆盖：当前目标、任务状态、已决策事项、未完成事项、关键约束。",
    ]
    if payload.existing_session_summary:
        sections.append(f"\n【已有长期摘要】\n{payload.existing_session_summary}")
    if payload.history_text:
        sections.append(f"\n【被压缩的历史消息】\n{payload.history_text}")
    if payload.sub_tasks_text:
        sections.append(f"\n【被压缩的子任务细节】\n{payload.sub_tasks_text}")
    if payload.artifact_refs:
        sections.append(f"\n【artifact 引用】\n" + "\n".join(payload.artifact_refs))
    session_response = llm.invoke("\n".join(sections))
    session_summary = getattr(session_response, "content", "") or str(session_response)

    tool_sections = [
        TOOLCHAIN_SUMMARY_PROMPT,
    ]
    if payload.existing_toolchain_summary:
        tool_sections.append(f"\n【已有工具链摘要】\n{payload.existing_toolchain_summary}")
    if payload.toolchain_text:
        tool_sections.append(f"\n【工具调用记录】\n{payload.toolchain_text}")
    if payload.sub_tasks_text:
        tool_sections.append(f"\n【子任务执行摘要】\n{payload.sub_tasks_text}")
    tool_response = llm.invoke("\n".join(tool_sections))
    toolchain_summary = getattr(tool_response, "content", "") or str(tool_response)

    return CompactionSummary(
        session_summary=session_summary.strip(),
        toolchain_summary=toolchain_summary.strip(),
    )


def _resident_tokens(ctx: ConversationContext, token_counter: TokenCounter) -> int:
    return token_counter(ctx.to_summary()) + token_counter(ctx.to_sub_tasks_summary())


def _render_messages(messages: list[Message]) -> str:
    rows = []
    for message in messages:
        rows.append(message.model_dump(mode="json"))
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _render_sub_task(output: dict) -> str:
    return json.dumps(output, ensure_ascii=False, indent=2, default=str)


def _summarize_compacted_material(
    ctx: ConversationContext,
    old_messages: list[Message],
    old_sub_tasks: list[dict],
    artifacts: list[ContextArtifact],
    summarizer: CompactionSummarizer | None,
) -> CompactionSummary:
    fallback_summary = _merge_summary(ctx.session_summary, old_messages, old_sub_tasks)
    fallback_toolchain = _merge_toolchain(ctx.toolchain_summary, old_messages, old_sub_tasks)
    if summarizer is None:
        return CompactionSummary(fallback_summary, fallback_toolchain)

    payload = CompactionPayload(
        existing_session_summary=ctx.session_summary,
        existing_toolchain_summary=ctx.toolchain_summary,
        history_text=_render_messages(old_messages) if old_messages else "",
        sub_tasks_text="\n\n".join(_render_sub_task(item) for item in old_sub_tasks),
        toolchain_text=fallback_toolchain,
        artifact_refs=[
            f"artifact: {artifact.artifact_id} kind={artifact.kind} "
            f"tokens={artifact.token_count} ref={artifact.content_ref}"
            for artifact in artifacts
        ],
    )
    try:
        summary = summarizer(payload)
    except Exception as exc:
        return CompactionSummary(
            session_summary=f"LLM summary failed: {exc}\n\n{fallback_summary}",
            toolchain_summary=fallback_toolchain,
        )
    if not summary.session_summary.strip():
        return CompactionSummary(fallback_summary, summary.toolchain_summary or fallback_toolchain)
    return CompactionSummary(
        session_summary=summary.session_summary.strip(),
        toolchain_summary=summary.toolchain_summary.strip() or fallback_toolchain,
    )


def _merge_summary(existing: str, old_messages: list[Message], old_sub_tasks: list[dict]) -> str:
    parts = [existing.strip()] if existing.strip() else []
    task_summaries = [str(item.get("summary", "")).strip() for item in old_sub_tasks if item.get("summary")]
    lines = [
        "当前目标: 保留已压缩会话状态，继续处理用户后续请求。",
        f"任务状态: {len(old_messages)} 条旧消息和 {len(old_sub_tasks)} 个子任务细节已移入 artifact。",
    ]
    if task_summaries:
        lines.append("已决策事项/发现: " + "；".join(task_summaries)[:1000])
    lines.append("未完成事项: 继续结合最近原文消息和 artifact 引用回答后续问题。")
    parts.append("\n".join(lines))
    return "\n\n".join(part for part in parts if part)


def _merge_toolchain(existing: str, old_messages: list[Message], old_sub_tasks: list[dict]) -> str:
    parts: list[str] = []
    if existing.strip():
        parts.append(existing.strip())
    for message in old_messages:
        if message.tool_summary:
            parts.extend(_split_tool_items(message.tool_summary))
    for output in old_sub_tasks:
        tool_log = str(output.get("tool_log", "") or "")
        tools_used = output.get("tools_used", []) or []
        parts.extend(_extract_tool_log_items(tool_log))
        for item in tools_used:
            text = str(item).strip()
            if text:
                parts.append(text if ":" in text else f"read_file: {text}")
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            deduped.append(part)
    return ", ".join(deduped)


def _split_tool_items(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _extract_tool_log_items(tool_log: str) -> list[str]:
    items: list[str] = []
    for match in re.finditer(r"\[Tool:\s*(\w+)", tool_log):
        items.append(match.group(1))
    for match in re.finditer(r"read_file\s*\([^'\"]*['\"]([^'\"]+)['\"]", tool_log):
        items.append(f"read_file: {match.group(1).strip()}")
    for match in re.finditer(r"terminal\s*\(\s*cmd\s*=\s*'([^']+)'", tool_log):
        items.append(f"terminal: {match.group(1).strip()}")
    return items


def _safe_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "default"
