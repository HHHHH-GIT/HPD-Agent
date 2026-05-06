from __future__ import annotations

import builtins
import shutil
import sys
from pathlib import Path
from typing import Iterable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree


_THEME = Theme(
    {
        "banner": "bold bright_cyan",
        "answer": "white",
        "accent": "bold cyan",
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "debug": "dim",
        "muted": "grey62",
        "agent": "magenta",
        "trace": "bright_blue",
    }
)


class CliRenderer:
    def __init__(self) -> None:
        self.console = Console(theme=_THEME, soft_wrap=True, highlight=False)
        self._original_print = builtins.print
        self._print_hook_installed = False

    def install_print_hook(self) -> None:
        if self._print_hook_installed:
            return
        builtins.print = self._print_hook
        self._print_hook_installed = True

    def render_banner(self) -> None:
        logo = "\n".join(
            [
                "██╗  ██╗ ██████╗  ██████╗",
                "██║  ██║ ██╔══██╗ ██╔══██╗",
                "███████║ ██████╔╝ ██║  ██║",
                "██╔══██║ ██╔═══╝  ██║  ██║",
                "██║  ██║ ██║      ██████╔╝",
                "╚═╝  ╚═╝ ╚═╝      ╚═════╝",
            ]
        )
        subtitle = "Hierarchical Parallel Dynamic Agent"
        body = Text()
        body.append(f"{logo}\n", style="banner")
        body.append(f"{subtitle}\n", style="accent")
        body.append("Ctrl+J newline  •  Enter submit  •  Ctrl+C cancel  •  /exit quit", style="muted")
        self.console.print(
            Panel(
                body,
                border_style="accent",
                title="Repository Agent",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        self.console.print(Rule("Output", style="muted"))

    def rule(self, title: str, style: str = "muted") -> None:
        self.console.print(Rule(title, style=style))

    def blank(self) -> None:
        self.console.print("")

    def info(self, message: str) -> None:
        self.console.print(message, style="info", markup=False, highlight=False)

    def success(self, message: str) -> None:
        self.console.print(message, style="success", markup=False, highlight=False)

    def warning(self, message: str) -> None:
        self.console.print(message, style="warning", markup=False, highlight=False)

    def error(self, message: str) -> None:
        self.console.print(message, style="error", markup=False, highlight=False)

    def stream_answer(self, chunk: str) -> None:
        self.console.print(chunk, style="answer", markup=False, highlight=False, end="")

    def confirm(self, prompt: str) -> bool:
        return Confirm.ask(prompt, console=self.console, default=False)

    def render_help(self, command_details: dict[str, str]) -> None:
        table = Table(box=box.SIMPLE_HEAVY, header_style="accent")
        table.add_column("Command", style="accent", no_wrap=True)
        table.add_column("Description", style="white")
        for command, detail in command_details.items():
            lines = detail.splitlines()
            summary = lines[0]
            extra = "\n".join(lines[1:])
            table.add_row(command, f"{summary}\n{extra}" if extra else summary)
        self.console.print(Panel(table, border_style="accent", title="Commands"))

    def render_sessions(
        self,
        project_hash: str,
        store_path: Path,
        rows: Iterable[tuple[str, int, bool]],
    ) -> None:
        table = Table(box=box.SIMPLE_HEAVY, header_style="accent")
        table.add_column("Session", style="accent")
        table.add_column("Messages", justify="right")
        table.add_column("Active", justify="center")
        count = 0
        for session_id, messages, is_active in rows:
            count += 1
            table.add_row(session_id, str(messages), "●" if is_active else "")
        if count == 0:
            self.warning("No active sessions in this project.")
            return
        title = f"Sessions • {project_hash}"
        subtitle = str(store_path)
        self.console.print(Panel(table, title=title, subtitle=subtitle, border_style="accent"))

    def render_models(self, rows: Iterable[tuple[str, str, bool]]) -> None:
        table = Table(box=box.SIMPLE_HEAVY, header_style="accent")
        table.add_column("Name", style="accent")
        table.add_column("Model")
        table.add_column("Active", justify="center")
        total = 0
        for name, model, is_active in rows:
            total += 1
            table.add_row(name, model, "●" if is_active else "")
        self.console.print(Panel(table, title=f"Models • {total}", border_style="accent"))

    def render_tokens(
        self,
        resident_tokens: int,
        left_tokens: int,
        max_tokens: int,
        pct: float,
        next_simple_estimate: int,
        next_assessment_estimate: int,
        tool_schema_tokens: int,
        sub_task_tokens: int,
        tool_tokens: int,
    ) -> None:
        table = Table(box=box.SIMPLE_HEAVY, header_style="accent")
        table.add_column("Category", style="accent")
        table.add_column("Tokens", justify="right")
        table.add_column("Notes", style="muted")
        table.add_row("Context used", str(resident_tokens), "resident context injected on next turn")
        table.add_row("Context left", str(left_tokens), f"of {max_tokens}")
        table.add_row("Next simple call", str(next_simple_estimate), "resident context + query + rules + tool schemas")
        table.add_row("Next assessment", str(next_assessment_estimate), "resident context + query + classifier prompt")
        table.add_row("Tool schemas", str(tool_schema_tokens), "included only on tool-enabled calls")
        table.add_row("Analysis cache", str(sub_task_tokens), "saved for /summary, not resident context")
        if tool_tokens > 0:
            table.add_row("Tool summaries", str(tool_tokens), "already included in resident context")
        body = Text()
        body.append(f"Context window: {left_tokens} left / {resident_tokens} used / {max_tokens}\n", style="accent")
        body.append(self._bar(pct), style="accent")
        self.console.print(Panel(table, title="Context Window", subtitle=body.plain, border_style="accent"))

    def render_context(
        self,
        session_id: str,
        total_messages: int,
        max_turns: int,
        entries: Iterable[tuple[str, str, str]],
    ) -> None:
        table = Table(box=box.SIMPLE_HEAVY, header_style="accent")
        table.add_column("Role", style="accent", no_wrap=True)
        table.add_column("Time", style="muted", no_wrap=True)
        table.add_column("Content")
        count = 0
        for role, ts, content in entries:
            count += 1
            table.add_row(role, ts, content)
        title = f"Context • {session_id}"
        subtitle = f"{total_messages} messages • max_turns={max_turns}"
        if count == 0:
            self.console.print(Panel("(no messages in context)", title=title, subtitle=subtitle, border_style="accent"))
            return
        self.console.print(Panel(table, title=title, subtitle=subtitle, border_style="accent"))

    def render_summary(
        self,
        summary_text: str,
        old_msg_tokens: int,
        new_msg_tokens: int,
        old_sub_task_tokens: int,
        old_total: int,
        saved_total: int,
    ) -> None:
        body = Text(summary_text)
        self.console.print(Panel(body, title="Session Summary", border_style="accent"))
        msg_saved = old_msg_tokens - new_msg_tokens
        msg_saved_pct = (msg_saved / old_msg_tokens * 100) if old_msg_tokens else 0
        footer = Table(box=box.SIMPLE_HEAVY, header_style="accent")
        footer.add_column("Metric", style="accent")
        footer.add_column("Before", justify="right")
        footer.add_column("After", justify="right")
        footer.add_column("Saved", justify="right")
        footer.add_row("Conversation", str(old_msg_tokens), str(new_msg_tokens), f"{msg_saved} ({msg_saved_pct:.1f}%)")
        if old_sub_task_tokens > 0:
            footer.add_row("Sub-task results", str(old_sub_task_tokens), "0", str(old_sub_task_tokens))
        footer.add_row("Total", str(old_total), str(new_msg_tokens), str(saved_total))
        self.console.print(Panel(footer, border_style="accent", title="Compression"))

    def render_trace_mode(self, label: str) -> None:
        self.console.print(Panel(label, border_style="trace", title="Trace"))

    def render_command_result(self, title: str, message: str, style: str = "accent") -> None:
        self.console.print(Panel(message, title=title, border_style=style))

    def render_trace_record(self, record: object, roots: list[object], children: dict[str, list[str]], id_map: dict[str, object]) -> None:
        title = f"Trace • {getattr(record, 'trace_id', '')}"
        subtitle = f"session {getattr(record, 'session_id', '') or 'default'}"
        tree = Tree(self._trace_label("query", getattr(record, "query", "") or "(empty query)"), guide_style="trace")

        for root in roots:
            self._add_trace_branch(tree, root, children, id_map)

        self.console.print(Panel(tree, title=title, subtitle=subtitle, border_style="trace"))

        summary = Table(box=box.SIMPLE_HEAVY, header_style="trace")
        summary.add_column("Metric", style="trace")
        summary.add_column("Value", style="white")
        total_dur = getattr(record, "duration_ms", 0.0) / 1000
        total_in = getattr(record, "total_tokens_in", 0)
        total_out = getattr(record, "total_tokens_out", 0)
        cost = getattr(record, "estimate_cost", lambda: 0.0)()
        summary.add_row("Duration", f"{total_dur:.2f}s")
        summary.add_row("Tokens In", str(total_in))
        summary.add_row("Tokens Out", str(total_out))
        summary.add_row("Estimated Cost", f"${cost:.6f}")
        self.console.print(Panel(summary, title="Trace Summary", border_style="trace"))

    def _add_trace_branch(self, branch: Tree, span: object, children: dict[str, list[str]], id_map: dict[str, object]) -> None:
        node = branch.add(self._format_span(span), guide_style="trace")
        error_msg = getattr(span, "error_msg", "")
        if error_msg:
            node.add(Text(f"error: {error_msg}", style="error"))
        metadata = getattr(span, "metadata", {}) or {}
        if metadata:
            meta_text = ", ".join(f"{k}={v}" for k, v in metadata.items())
            node.add(Text(f"meta: {meta_text}", style="muted"))
        for child_id in children.get(getattr(span, "span_id", ""), []):
            child = id_map.get(child_id)
            if child is not None:
                self._add_trace_branch(node, child, children, id_map)

    def _format_span(self, span: object) -> Text:
        status = getattr(span, "status", "ok")
        status_icon = "●" if status == "ok" else "✕"
        status_style = "success" if status == "ok" else "error"
        name = getattr(span, "name", "")
        duration_ms = getattr(span, "duration_ms", 0.0)
        tokens_in = getattr(span, "tokens_in", 0)
        tokens_out = getattr(span, "tokens_out", 0)
        model = getattr(span, "model", "")

        text = Text()
        text.append(f"{status_icon} ", style=status_style)
        text.append(name, style="trace")
        text.append(f"  {duration_ms / 1000:.2f}s", style="accent")
        if tokens_in or tokens_out:
            text.append(f"  in {tokens_in} / out {tokens_out}", style="muted")
        if model:
            text.append(f"  {model}", style="muted")
        return text

    def _trace_label(self, prefix: str, value: str) -> Text:
        text = Text()
        text.append(f"{prefix}: ", style="trace")
        text.append(value, style="white")
        return text

    def _bar(self, pct: float, width: int = 28) -> str:
        filled = max(0, min(width, int(pct * width)))
        return "[" + "■" * filled + "·" * (width - filled) + "]"

    def _print_hook(self, *args, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
        if file not in (None, sys.stdout):
            self._original_print(*args, sep=sep, end=end, file=file, flush=flush)
            return

        message = sep.join(str(arg) for arg in args)
        self._render_captured_message(message, end=end)
        if flush:
            self.console.file.flush()

    def _render_captured_message(self, message: str, end: str = "\n") -> None:
        stripped = message.strip()
        if not stripped and end:
            self.console.print("")
            return
        if stripped.startswith("=== ") and stripped.endswith(" ==="):
            title = stripped.strip("=").strip()
            self.console.print(Rule(title, style="muted"))
            return
        if "─── 最终回答 ───" in stripped:
            self.console.print(Rule("Final Answer", style="accent"))
            return

        style = self._style_for_text(stripped)
        self.console.print(message, style=style, markup=False, highlight=False, end=end)

    def _style_for_text(self, text: str) -> str | None:
        if not text:
            return None
        if text.startswith("[DEBUG]"):
            return "debug"
        if text.startswith("[trace:") or text.startswith("[/skim]"):
            return "trace"
        if text.startswith("[Scheduler]") or text.startswith("[Assessment]") or text.startswith("[Planning]") or text.startswith("[Replan]"):
            return "info"
        if text.startswith("[CoordinatorAgent") or text.startswith("[ReviewerAgent") or text.startswith("[Expert]") or text.startswith("[TOT]"):
            return "agent"
        if "WARNING" in text or "Token limit reached" in text:
            return "warning"
        if text.startswith("Unknown") or text.startswith("Error:") or "[Error]" in text:
            return "error"
        if text.startswith("Goodbye"):
            return "muted"
        return None

    def prompt_width(self) -> int:
        return max(48, min(96, shutil.get_terminal_size((100, 20)).columns - 4))


_renderer: CliRenderer | None = None


def get_renderer() -> CliRenderer:
    global _renderer
    if _renderer is None:
        _renderer = CliRenderer()
    return _renderer
