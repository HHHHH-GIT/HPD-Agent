"""Collect system, git, and project information for boot prompt injection."""

import datetime
import os
import platform
import subprocess
from pathlib import Path


def _run(cmd: str, default: str = "") -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except Exception:
        return default


def collect() -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cwd = os.getcwd()
    lines = [
        f"## 系统环境 [{now}]",
    ]

    uname = platform.platform()
    py_version = platform.python_version()
    lines.append(f"- OS: {uname}")
    lines.append(f"- Python: {py_version}")
    lines.append(f"- 工作目录: {cwd}")

    git_branch = _run("git rev-parse --abbrev-ref HEAD")
    git_status = _run("git status --porcelain").strip()
    git_ahead = _run("git rev-list --left-right --count HEAD...origin/main")
    if git_ahead:
        behind, ahead = git_ahead.split()
        if int(behind) or int(ahead):
            lines.append(
                f"- Git: 分支 {git_branch} | "
                f"ahead={ahead} behind={behind}"
            )
        else:
            lines.append(f"- Git: 分支 {git_branch} (已同步)")
    else:
        lines.append(f"- Git: 分支 {git_branch} (无远程)")

    if git_status:
        modified = [f for f in git_status.splitlines() if f]
        lines.append(f"  未提交变更 ({len(modified)} 个文件):")
        for m in modified[:10]:
            lines.append(f"    {m}")
        if len(modified) > 10:
            lines.append(f"    ... 还有 {len(modified) - 10} 个文件")
    else:
        lines.append("  无未提交变更")

    proj_root = _find_project_root(cwd)
    if proj_root and proj_root != cwd:
        lines.append(f"- 项目根目录: {proj_root}")
        pyproject = Path(proj_root) / "pyproject.toml"
        reqs = Path(proj_root) / "requirements.txt"
        if pyproject.exists():
            lines.append(f"  依赖: pyproject.toml")
        elif reqs.exists():
            lines.append(f"  依赖: requirements.txt")

    gitignore = Path(cwd) / ".gitignore"
    if gitignore.exists():
        gi = gitignore.read_text()
        if gi:
            parts = [p.strip() for p in gi.splitlines() if p.strip() and not p.startswith("#")]
            if parts:
                lines.append(f"- .gitignore: {', '.join(parts[:6])}")
                if len(parts) > 6:
                    lines[-1] += f" ... (+{len(parts)-6})"

    return "\n".join(lines)


def _find_project_root(cwd: str) -> str | None:
    current = Path(cwd)
    markers = (
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "Cargo.toml",
        ".git",
    )
    for parent in [current] + list(current.parents):
        if any(parent.joinpath(m).exists() for m in markers):
            return str(parent)
    return None


_HPD_MD_MAX_CHARS = 3000
"""Maximum characters of HPD.MD content to embed in the boot prompt."""


def read_hpdm() -> str | None:
    """Read the project's HPD.MD file if it exists in cwd."""
    hpdm = Path.cwd() / "HPD.MD"
    if not hpdm.exists():
        return None
    try:
        content = hpdm.read_text(encoding="utf-8", errors="replace")
        if len(content) > _HPD_MD_MAX_CHARS:
            return content[:_HPD_MD_MAX_CHARS] + f"\n\n...（HPD.MD 内容已被截断，完整内容见项目根目录 HPD.MD）"
        return content
    except Exception:
        return None


_SKIM_INJECT_TEMPLATE = (
    "\n\n"
    "## 项目知识摘要 (HPD.MD)\n"
    "当前项目目录下存在 `HPD.MD` 文件，以下是其中的内容摘要：\n\n"
    "{hpdm_content}\n\n"
    "【重要】在回答涉及本项目的技术问题时，请优先参考上述 HPD.MD 中的项目结构、"
    "技术栈、配置和运行方式等信息。如果用户的问题与项目直接相关，"
    "请先阅读 HPD.MD 获取必要的上下文。"
)


def build_boot_prompt() -> str:
    """Build the full boot prompt, including HPD.MD content if available."""
    info = collect()
    hpdm = read_hpdm()
    from src.llm.prompts import BOOT_PROMPT
    prompt = BOOT_PROMPT.format(system_info=info)
    if hpdm:
        prompt += _SKIM_INJECT_TEMPLATE.format(hpdm_content=hpdm)
    return prompt
