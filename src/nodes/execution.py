"""Execution node: difficulty assessment + LLM execution for a single sub-task.

This is a pure business-logic node — no graph coupling, no state mutation.
It handles:
  - Sub-task difficulty assessment (easy / hard)
  - Easy tasks: single LLM call (no tools)
  - Hard tasks: multi-path generation + evaluate→reflect iteration
  - Summary extraction
  - Tool-usage tracking (which files/resources were read)
  - Key-finding extraction (structured facts for downstream context)

The expert agent calls this to make the "execute" decision.
"""

import asyncio
import json
import re

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.core.enums import SubTaskDifficulty
from src.core.models import (
    SubTaskAssessmentResult,
    SubTaskOutput,
    CandidateResult,
    EvaluatorScore,
)
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import (
    get_llm,
    invoke_with_tools,
    get_structured_llm,
    SUB_TASK_ASSESSMENT_PROMPT,
    SUB_TASK_PROMPT,
    KEY_FINDINGS_PROMPT,
)
from src.nodes.rewriter import rewrite_prompt
from src.nodes.evaluator import evaluate_single
from src.nodes.reflector import reflect
from src.tools import tool_list

NUM_CANDIDATES = 3
EXPERT_SCORE_THRESHOLD = 0.7
MAX_EXPERT_ITERATIONS = 2


def _extract_summary(detail: str) -> str:
    """Extract concise summary from LLM output, handling JSON-fragment edge cases."""
    import json

    try:
        parsed = json.loads(detail.strip())
        if isinstance(parsed, dict) and parsed.get("summary"):
            s = parsed["summary"].strip()
            if s:
                return s
    except (json.JSONDecodeError, TypeError):
        pass

    sentences = re.split(r"(?<=[。！？.!?\n])", detail)
    candidates = [s.strip() for s in sentences if 5 < len(s.strip()) < 120]
    return candidates[-1] if candidates else detail[:80].strip()


class KeyFindingsResult(BaseModel):
    """Structured key findings extracted from a sub-task's output."""

    findings: list[str] = Field(
        default_factory=list,
        description=(
            "Key facts discovered, one per entry. Format: concise, machine-readable "
            "facts (e.g. 'port=8080', 'python=3.11', 'error=connection refused'). "
            "Only include genuinely new facts — do NOT repeat the original question. "
            "Maximum 10 entries."
        ),
    )


def _parse_tools_used(tool_log: str) -> list[str]:
    """Extract file/resource identifiers from a tool call log.

    Parses both direct read_file calls and terminal commands (cat, ls, find)
    so downstream tasks know exactly which files were already read.
    """
    paths: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(r"read_file\s*\([^'\"]*['\"]([^'\"]+)['\"]", tool_log):
        path = match.group(1).strip()
        if path and path not in seen:
            seen.add(path)
            paths.append(path)

    for match in re.finditer(r"terminal\s*\(\s*cmd\s*=\s*'([^']+)'", tool_log):
        cmd = match.group(1).strip()
        for path in _extract_paths_from_terminal_cmd(cmd):
            if path and path not in seen:
                seen.add(path)
                paths.append(path)

    for match in re.finditer(r'terminal\s*\(\s*cmd\s*=\s*"([^"]+)"', tool_log):
        cmd = match.group(1).strip()
        for path in _extract_paths_from_terminal_cmd(cmd):
            if path and path not in seen:
                seen.add(path)
                paths.append(path)

    return paths


def _extract_paths_from_terminal_cmd(cmd: str) -> list[str]:
    """Extract file/directory paths from a shell command string."""
    paths: list[str] = []
    if not cmd:
        return paths

    if cmd.lstrip().startswith("cat "):
        rest = cmd.lstrip()[4:]
        tokens = re.split(r"\s+", rest)
        for token in tokens:
            if token and not token.startswith("-") and "|" not in token:
                paths.append(token)

    ls_match = re.search(r"\bls\s+(?:-\w+\s+)*([^\s|;&]+)", cmd)
    if ls_match:
        target = ls_match.group(1).strip().rstrip("/")
        if target and target != ".":
            paths.append(target)

    find_match = re.search(r"\bfind\s+(?:[^\s|;&]+(?:\s+-(?!name|type|exec))?)*\s+([^\s|;&]+)", cmd)
    if find_match:
        target = find_match.group(1).strip().rstrip("/")
        if target and target not in ("-name", "-type", "-exec", "-ok", "-delete", "-print"):
            paths.append(target)

    return paths


def _build_tool_chain(tool_log: str) -> str:
    """Build a compact tool call chain summary — tool names + key args, no output.

    Example output:
      [1] read_file('src/main.py')
      [2] terminal('ls -la')
      [3] read_file('src/config.py')
    """
    lines: list[str] = []
    idx = 0
    # Match [Tool: name(arg_string)] — capture the full args including quotes
    for match in re.finditer(r"\[Tool:\s*(\w+)\(([^)]*)\)\]", tool_log):
        idx += 1
        name = match.group(1)
        args_raw = match.group(2) or ""
        # Extract the primary value: path='...' or cmd='...'
        # Handle nested quotes: cmd='grep -r "TODO" src/'
        val_match = re.search(r"""(?:path|cmd)\s*=\s*'([^']*(?:'[^']*')*[^']*)'""", args_raw)
        if not val_match:
            val_match = re.search(r'(?:path|cmd)\s*=\s*"([^"]*)"', args_raw)
        if val_match:
            arg_str = val_match.group(1)
            # Truncate long commands
            if len(arg_str) > 60:
                arg_str = arg_str[:57] + "..."
        else:
            arg_str = args_raw[:60] if args_raw else ""
        lines.append(f"  [{idx}] {name}({arg_str})")
    return "\n".join(lines) if lines else ""


def _extract_key_findings_llm(detail: str) -> list[str]:
    """Use an LLM to extract structured key findings from detail text."""
    findings: list[str] = []
    try:
        classifier = get_structured_llm(KeyFindingsResult)
        result: KeyFindingsResult = classifier.invoke(KEY_FINDINGS_PROMPT.format(detail=detail[:3000]))
        findings = result.findings
    except Exception:
        pass
    return findings


async def _execute_candidate(
    index: int,
    angle: str,
    task_id: int,
    task_name: str,
    context: str,
) -> CandidateResult:
    """Execute a single prompt variation without tools (multi-path candidate)."""
    augmented_name = f"{task_name}\n\n【特定角度】请从以下角度切入：{angle}"
    prompt = SUB_TASK_PROMPT.format(
        context=context, task_id=task_id, task_name=augmented_name
    )

    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    content = response.content or ""

    detail = content
    summary = ""
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict):
            detail = parsed.get("detail", content)
            summary = parsed.get("summary", "")
    except (json.JSONDecodeError, TypeError):
        pass

    if not summary:
        summary = _extract_summary(content)

    return CandidateResult(
        variation_index=index,
        prompt_angle=angle,
        detail=detail,
        summary=summary,
    )


async def _execute_multipath(
    task_id: int,
    task_name: str,
    context: str,
) -> tuple[list[CandidateResult], list[EvaluatorScore]]:
    """Multi-path generation: rewrite → parallel candidates → 1:1 evaluation.

    Returns:
        (candidates, scores) — lists of equal length.
    """
    # Step 1: Generate N prompt angles
    rewrite_result = await rewrite_prompt(task_id, task_name, context, n=NUM_CANDIDATES)
    angles = rewrite_result.angles[:NUM_CANDIDATES]

    if len(angles) < 2:
        raise RuntimeError(f"Rewriter produced only {len(angles)} angle(s), need >= 2")

    # Step 2: Execute all candidates in parallel (no tools)
    candidate_results = await asyncio.gather(
        *[_execute_candidate(i, angle, task_id, task_name, context)
          for i, angle in enumerate(angles)],
        return_exceptions=True,
    )

    valid_candidates: list[CandidateResult] = []
    for i, result in enumerate(candidate_results):
        if isinstance(result, Exception):
            print(f"[TOT] Candidate {i} failed: {result}")
        else:
            valid_candidates.append(result)

    if len(valid_candidates) < 2:
        raise RuntimeError(f"Only {len(valid_candidates)} valid candidate(s), need >= 2")

    # Step 3: Evaluate each candidate independently (1:1 scoring)
    scores = await asyncio.gather(
        *[evaluate_single(c.detail, task_name, context) for c in valid_candidates],
        return_exceptions=True,
    )

    valid_scores: list[EvaluatorScore] = []
    for i, result in enumerate(scores):
        if isinstance(result, Exception):
            print(f"[TOT] Evaluation {i} failed: {result}")
            valid_scores.append(EvaluatorScore(score=0.0, reasoning="evaluation failed"))
        else:
            valid_scores.append(result)

    return valid_candidates, valid_scores


async def _expert_loop(
    task_id: int,
    task_name: str,
    context: str,
) -> SubTaskOutput:
    """Expert mode for hard tasks: multi-path generation + evaluate→reflect iteration.

    Flow:
      1. Multi-path: rewrite → N candidates → 1:1 evaluate each → pick best
      2. Evaluate best result
      3. If score < threshold: reflect → re-execute with tools → re-evaluate
      4. Repeat up to MAX_EXPERT_ITERATIONS
    """
    tracer = get_tracer()
    span_id = None  # Will be set by caller's span

    # ── Step A: Multi-path generation ────────────────────────────────
    best_detail = ""
    best_score_obj: EvaluatorScore | None = None

    try:
        candidates, scores = await _execute_multipath(task_id, task_name, context)
        best_idx = max(range(len(scores)), key=lambda i: scores[i].score)
        best_detail = candidates[best_idx].detail
        best_score_obj = scores[best_idx]

        print(f"[TOT] Task {task_id}: {len(candidates)} candidates, "
              f"best score={best_score_obj.score:.2f} (candidate {best_idx})")
    except Exception as e:
        print(f"[TOT] Task {task_id}: multi-path failed ({e}), using single-path")
        # Fall back: single execution with tools
        prompt = SUB_TASK_PROMPT.format(
            context=context, task_id=task_id, task_name=task_name
        )
        full_content, tool_log = await invoke_with_tools(prompt, tools=tool_list)
        best_detail = full_content or ""

    # ── Step B: Evaluate→Reflect iteration (all hard tasks) ──────────
    current_detail = best_detail
    current_score = best_score_obj

    for iteration in range(MAX_EXPERT_ITERATIONS):
        # Evaluate if not already scored (single-path fallback)
        if current_score is None:
            current_score = await evaluate_single(current_detail, task_name, context)

        print(f"[Expert] Task {task_id}: score={current_score.score:.2f} "
              f"{'(iteration ' + str(iteration + 1) + '/' + str(MAX_EXPERT_ITERATIONS) + ')' if iteration > 0 or current_score.score < EXPERT_SCORE_THRESHOLD else ''}")

        if current_score.score >= EXPERT_SCORE_THRESHOLD:
            print(f"[Expert] Task {task_id}: score={current_score.score:.2f} >= "
                  f"threshold={EXPERT_SCORE_THRESHOLD}, accepted")
            break

        if iteration >= MAX_EXPERT_ITERATIONS - 1:
            print(f"[Expert] Task {task_id}: max iterations reached, using current result")
            break

        # Reflect
        print(f"[Expert] Task {task_id}: score={current_score.score:.2f} < "
              f"threshold={EXPERT_SCORE_THRESHOLD}, reflecting... (iteration {iteration + 1}/{MAX_EXPERT_ITERATIONS})")
        reflection = await reflect(task_name, context, current_detail, current_score)

        # Re-execute with improved prompt (with tools)
        improved_prompt = SUB_TASK_PROMPT.format(
            context=context,
            task_id=task_id,
            task_name=f"{task_name}\n\n【改进策略】{reflection.strategy}",
        )
        full_content, tool_log = await invoke_with_tools(improved_prompt, tools=tool_list)
        current_detail = full_content or ""
        current_score = None  # Force re-evaluation

    return current_detail


def _build_output(
    task_id: int,
    task_name: str,
    detail: str,
    is_expert: bool,
    tool_log: str = "",
) -> SubTaskOutput:
    """Build SubTaskOutput from execution result."""
    tools_used = _parse_tools_used(tool_log) if tool_log else []
    tool_chain = _build_tool_chain(tool_log) if tool_log else ""
    if tool_chain:
        full_detail = f"{detail}\n\n[工具调用链]\n{tool_chain}"
    else:
        full_detail = detail

    key_findings = _extract_key_findings_llm(full_detail)

    return SubTaskOutput(
        id=task_id,
        name=task_name,
        detail=full_detail,
        summary=_extract_summary(full_detail),
        expert_mode=is_expert,
        tools_used=tools_used,
        key_findings=key_findings,
        tool_log=tool_log or "",
    )


async def execute(task_id: int, task_name: str, context: str) -> SubTaskOutput:
    """Assess difficulty and execute a single sub-task.

    - Easy tasks: single LLM call, no tools (like direct_answer).
    - Hard tasks: multi-path generation + evaluate→reflect iteration.

    Args:
        task_id:   Sub-task ID from the DAG.
        task_name: Human-readable sub-task name.
        context:   The original user query (shared background context).

    Returns:
        SubTaskOutput with detail, summary, tools_used, key_findings, and expert_mode flag.
    """
    tracer = get_tracer()
    with tracer.span(f"execution[#{task_id}]", metadata={"task_name": task_name}) as span_id:
        # Step 1: Difficulty assessment
        classifier = get_structured_llm(SubTaskAssessmentResult)
        assessment: SubTaskAssessmentResult = await classifier.ainvoke(
            SUB_TASK_ASSESSMENT_PROMPT.format(task_id=task_id, task_name=task_name)
        )
        is_expert = assessment.difficulty == SubTaskDifficulty.HARD

        # Record assessment tokens
        tin0, tout0, model0 = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin0, tokens_out=tout0, model=model0)

        # Step 2: Branch by difficulty
        if not is_expert:
            # Easy: single LLM call, no tools
            print(f"[Expert] Task {task_id}: easy, single call")
            prompt = SUB_TASK_PROMPT.format(
                context=context, task_id=task_id, task_name=task_name
            )
            llm = get_llm()
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content or ""

            tin, tout, model = TokenTrackerCallback.snapshot()
            tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

            return _build_output(task_id, task_name, content, is_expert=False)

        # Hard: Expert mode — multi-path + evaluate→reflect
        print(f"[Expert] Task {task_id}: hard, entering expert loop")
        detail = await _expert_loop(task_id, task_name, context)

        # Record tokens
        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

        return _build_output(task_id, task_name, detail, is_expert=True)
