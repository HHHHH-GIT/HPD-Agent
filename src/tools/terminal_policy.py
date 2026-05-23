"""Risk classification for terminal commands before CLI execution."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TerminalRisk = Literal["safe", "normal", "extreme"]


@dataclass(frozen=True)
class TerminalConfirmationDecision:
    requires_confirmation: bool
    risk: TerminalRisk
    reason: str


_READ_ONLY_COMMANDS = {
    "pwd",
    "ls",
    "cat",
    "echo",
    "date",
    "whoami",
    "head",
    "tail",
    "less",
    "file",
    "stat",
    "uname",
    "id",
    "hostname",
    "env",
    "printenv",
}

_REDIRECT_PATTERN = re.compile(r"(^|[^<])>>?|2>|&>")


def should_confirm_terminal_command(
    cmd: str,
    *,
    riskless_enabled: bool,
    project_dir: str | os.PathLike[str] | None = None,
) -> TerminalConfirmationDecision:
    """Return whether a terminal command needs manual confirmation."""
    stripped = (cmd or "").strip()
    if not stripped:
        return TerminalConfirmationDecision(False, "safe", "empty command")

    extreme_reason = _extreme_reason(stripped, project_dir)
    if extreme_reason:
        return TerminalConfirmationDecision(True, "extreme", extreme_reason)

    if _is_read_only_command(stripped):
        return TerminalConfirmationDecision(False, "safe", "read-only command")

    if riskless_enabled:
        return TerminalConfirmationDecision(False, "normal", "riskless mode enabled")

    return TerminalConfirmationDecision(True, "normal", "riskless mode disabled")


def _is_read_only_command(cmd: str) -> bool:
    if _REDIRECT_PATTERN.search(cmd):
        return False
    segments = _split_shell_segments(cmd)
    if len(segments) != 1:
        return False
    tokens = _split_tokens(segments[0])
    if not tokens:
        return True
    program = _program_name(tokens)
    if program == "cd":
        return True
    return program in _READ_ONLY_COMMANDS


def _extreme_reason(cmd: str, project_dir: str | os.PathLike[str] | None) -> str:
    for segment in _split_shell_segments(cmd):
        tokens = _split_tokens(segment)
        if not tokens:
            continue
        program = _program_name(tokens)
        if program == "git" and _git_force_or_destructive(tokens):
            return "git force/destructive operation"
        if program == "rm":
            outside = _rm_outside_project_target(tokens, project_dir)
            if outside:
                return f"rm target outside project: {outside}"
    return ""


def _split_shell_segments(cmd: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*(?:&&|\|\||;)\s*", cmd) if part.strip()]


def _split_tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _program_name(tokens: list[str]) -> str:
    if not tokens:
        return ""
    index = 0
    while index < len(tokens) and tokens[index] in {"sudo", "doas", "command"}:
        index += 1
        while index < len(tokens) and tokens[index].startswith("-"):
            index += 1
    if index >= len(tokens):
        return ""
    return Path(tokens[index]).name


def _git_force_or_destructive(tokens: list[str]) -> bool:
    normalized = [token.lower() for token in tokens]
    if any(token.startswith("--force") or token == "-f" for token in normalized):
        return True
    if normalized[:3] == ["git", "reset", "--hard"]:
        return True
    if len(normalized) >= 3 and normalized[0] == "git" and normalized[1] == "clean":
        return any("f" in token.lstrip("-") for token in normalized[2:] if token.startswith("-"))
    return False


def _rm_outside_project_target(
    tokens: list[str],
    project_dir: str | os.PathLike[str] | None,
) -> str:
    project_root = Path(project_dir or Path.cwd()).expanduser().resolve()
    program_index = _program_index(tokens)
    operands = tokens[program_index + 1 :]
    for token in operands:
        if not token or token == "--" or token.startswith("-"):
            continue
        if any(marker in token for marker in ("$", "`")):
            return token
        target = Path(os.path.expanduser(token))
        if not target.is_absolute():
            target = project_root / target
        try:
            resolved = target.resolve(strict=False)
            if resolved == project_root:
                return str(resolved)
            resolved.relative_to(project_root)
        except ValueError:
            return str(resolved)
    return ""


def _program_index(tokens: list[str]) -> int:
    index = 0
    while index < len(tokens) and tokens[index] in {"sudo", "doas", "command"}:
        index += 1
        while index < len(tokens) and tokens[index].startswith("-"):
            index += 1
    return min(index, len(tokens) - 1)


__all__ = ["TerminalConfirmationDecision", "should_confirm_terminal_command"]
