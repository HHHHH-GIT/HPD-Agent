"""Handler for the /skim command — scan the project and write HPD.MD."""

import traceback

from pathlib import Path

from src.agents import QueryAgent
from src.llm.prompts import SKIM_PROMPT, SKIM_ERROR_PROMPT

HPD_MD_HEADER = """# HPD — Project Knowledge Summary

> 由 /skim 命令自动生成，请勿手动编辑，下次运行 /skim 时会被覆盖。
> 更新时间：{timestamp}

---

"""


def _skim_sync(project_root: Path | None) -> tuple[str, str, str]:
    """Run the full skim pipeline synchronously.

    Returns (scan_result, llm_summary, hpdm_path).
    """
    import traceback as tb

    from src.llm import get_llm
    from src.tools.project_scanner import scan_project

    root = project_root or Path.cwd()

    # 1. Scan
    try:
        scan_result = scan_project(str(root))
        error_count = 0
    except Exception as exc:
        scan_result = f"[扫描失败] {exc}"
        error_count = 1

    # 2. Summarize with LLM
    project_name = root.name
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if error_count > 0:
        prompt = SKIM_ERROR_PROMPT.format(
            error_count=error_count,
            scan_result=scan_result,
        )
    else:
        prompt = SKIM_PROMPT.format(
            project_name=project_name,
            timestamp=timestamp,
            scan_result=scan_result,
        )

    llm = get_llm(temperature=0.3)
    try:
        response = llm.invoke(prompt)
        llm_summary = getattr(response, "content", "") or str(response)
    except Exception as exc:
        tb.print_exc()
        llm_summary = f"[LLM 调用失败] {exc}\n\n请手动运行 /skim 重试。"

    # 3. Write HPD.MD
    hpdm_path = root / "HPD.MD"
    header = HPD_MD_HEADER.format(timestamp=timestamp)
    hpdm_path.write_text(header + llm_summary.strip() + "\n", encoding="utf-8")

    return scan_result, llm_summary, str(hpdm_path)


def run(raw: str, agent: QueryAgent | None = None) -> bool:
    """Handle /skim command."""
    import os

    project_root = None
    parts = raw.strip().split()
    if len(parts) > 1:
        path = parts[1].strip()
        expanded = os.path.expanduser(path)
        if Path(expanded).exists():
            project_root = Path(expanded)
        else:
            print(f"[/skim] 路径不存在: {expanded}")
            return False

    if project_root is None:
        cwd = os.getcwd()
        project_root = Path(cwd)

    print(f"[/skim] 正在扫描项目: {project_root.resolve()}")
    print("[/skim] 这可能需要几秒钟...\n")

    try:
        scan_result, llm_summary, hpdm_path = _skim_sync(project_root)
    except Exception as exc:
        print(f"[/skim] 出错: {exc}")
        traceback.print_exc()
        return False

    print(f"\n[/skim] 生成完毕！文件已保存至: {hpdm_path}")
    print("\n─────────────────────────────────────────────")
    print(llm_summary.strip())
    print("─────────────────────────────────────────────\n")
    return False
