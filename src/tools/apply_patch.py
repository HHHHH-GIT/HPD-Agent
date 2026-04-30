# pyright: reportUnknownVariableType=false
"""安全的 v2 apply_patch 工具。

v2 协议设计目标:让 LLM 一次写对补丁。

格式概览
~~~~~~~~

::

    *** Begin Patch
    *** Update File: src/app.py
    <<<<<<< SEARCH
    def greet(name):
        return "hello"
    =======
    def greet(name: str) -> str:
        return f"hello {name}"
    >>>>>>> REPLACE
    *** Add File: src/util.py
    <<<<<<< CONTENT
    def noop():
        pass
    >>>>>>> END
    *** Replace File: README.md
    <<<<<<< CONTENT
    # New README
    >>>>>>> END
    *** Delete File: legacy.py
    *** End Patch

四种文件操作:

* ``*** Add File: path`` —— 新建文件;目标必须不存在。
* ``*** Update File: path`` —— 局部修改;一个或多个 SEARCH/REPLACE 块。
* ``*** Replace File: path`` —— 整体覆盖现有文件;目标必须存在。
* ``*** Delete File: path`` —— 删除文件;目标必须存在。

SEARCH 必须在目标文件中**按行精确且唯一**出现。空 SEARCH 仅在目标文件
为空时合法(等价于整体写入空文件,等价于 Replace File)。多个块允许乱序
提供,工具内部按行位置排序;块之间不能重叠。

安全边界
~~~~~~~~

仅允许工作区内的安全相对路径;拒绝 ``..``、``~``、绝对路径、``.git``、敏感
凭据命名、符号链接(包括父目录)、目录、非 UTF-8 目标、超限补丁(256 KiB)
和超过 1 MiB 的最终输出。写入前重新校验目标快照;失败时同进程回滚。

非目标(non-goals)
~~~~~~~~~~~~~~~~~~

* no git diff compatibility:不接受 ``diff --git`` / ``---`` / ``+++`` 片段。
* no fuzzy matching:SEARCH 必须按行精确匹配;不会模糊查找替代位置。
* no terminal hardening:terminal 调用的安全边界仍在 ``src.llm.client``,
  本工具不修改 terminal 逻辑。
"""

from dataclasses import dataclass, replace
import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import threading
from typing import Literal, cast

from langchain_core.tools import tool


BEGIN_MARKER = "*** Begin Patch"
END_MARKER = "*** End Patch"
SEARCH_OPEN = "<<<<<<< SEARCH"
SEARCH_DIVIDER = "======="
REPLACE_CLOSE = ">>>>>>> REPLACE"
CONTENT_OPEN = "<<<<<<< CONTENT"
CONTENT_CLOSE = ">>>>>>> END"

UTF8_BOM = b"\xef\xbb\xbf"
PATCH_TEXT_LIMIT_BYTES = 256 * 1024
TARGET_FILE_LIMIT_BYTES = 1024 * 1024

# ──────────────── 错误码 ────────────────
INVALID_PATCH = "INVALID_PATCH"
INVALID_ENVELOPE = "INVALID_ENVELOPE"
MALFORMED_SECTION = "MALFORMED_SECTION"
MALFORMED_BLOCK = "MALFORMED_BLOCK"
DUPLICATE_FILE_OPERATION = "DUPLICATE_FILE_OPERATION"

INVALID_PATH = "INVALID_PATH"
SENSITIVE_PATH = "SENSITIVE_PATH"
TARGET_IS_SYMLINK = "TARGET_IS_SYMLINK"
TARGET_TOO_LARGE = "TARGET_TOO_LARGE"
PATCH_TOO_LARGE = "PATCH_TOO_LARGE"
INVALID_UTF8 = "INVALID_UTF8"
BINARY_CONTENT = "BINARY_CONTENT"

TARGET_EXISTS = "TARGET_EXISTS"
TARGET_MISSING = "TARGET_MISSING"
TARGET_IS_DIRECTORY = "TARGET_IS_DIRECTORY"

FINAL_TOO_LARGE = "FINAL_TOO_LARGE"
MIXED_NEWLINES = "MIXED_NEWLINES"

# v2 匹配错误(替代 v1 的 HUNK_*)
SEARCH_NOT_FOUND = "SEARCH_NOT_FOUND"
AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
BLOCK_OVERLAP = "BLOCK_OVERLAP"
EMPTY_SEARCH_NON_EMPTY_FILE = "EMPTY_SEARCH_NON_EMPTY_FILE"

NOOP_PATCH = "NOOP_PATCH"
APPLY_FAILED = "APPLY_FAILED"
CLEANUP_FAILED = "CLEANUP_FAILED"
TARGET_CHANGED = "TARGET_CHANGED"
ROLLBACK_FAILED = "ROLLBACK_FAILED"

ERROR_CODES = (
    INVALID_PATCH,
    INVALID_ENVELOPE,
    MALFORMED_SECTION,
    MALFORMED_BLOCK,
    DUPLICATE_FILE_OPERATION,
    INVALID_PATH,
    SENSITIVE_PATH,
    TARGET_IS_SYMLINK,
    TARGET_TOO_LARGE,
    PATCH_TOO_LARGE,
    INVALID_UTF8,
    BINARY_CONTENT,
    TARGET_EXISTS,
    TARGET_MISSING,
    TARGET_IS_DIRECTORY,
    FINAL_TOO_LARGE,
    MIXED_NEWLINES,
    SEARCH_NOT_FOUND,
    AMBIGUOUS_MATCH,
    BLOCK_OVERLAP,
    EMPTY_SEARCH_NON_EMPTY_FILE,
    NOOP_PATCH,
    APPLY_FAILED,
    CLEANUP_FAILED,
    TARGET_CHANGED,
    ROLLBACK_FAILED,
)

OperationKind = Literal["add", "update", "delete", "replace"]

_SECTION_RE = re.compile(
    r"^\*\*\* (Add File|Update File|Delete File|Replace File): (.+)$"
)
_GIT_DIFF_HINT_RE = re.compile(r"^(diff --git |--- |\+\+\+ |@@ )")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_WINDOWS_DRIVE_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_PATH_TOKEN_RE = re.compile(r"[._\-\s]+")
_ALLOWED_TEXT_CONTROL_BYTES = frozenset({0x08, 0x09, 0x0A, 0x0C, 0x0D, 0x1B})
_FILE_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[str, threading.Lock] = {}
_SENSITIVE_PATH_TOKENS = frozenset(
    {
        "secret",
        "secrets",
        "credential",
        "credentials",
        "key",
        "keys",
        "apikey",
        "apikeys",
        "token",
        "tokens",
        "password",
        "passwords",
        "passwd",
    }
)


# ──────────────── 数据类 ────────────────
@dataclass(frozen=True)
class SearchReplaceBlock:
    """Update File 中的一个搜索-替换块。

    ``search_text`` 与 ``replace_text`` 是块内容的纯文本,内部统一使用
    ``\\n`` 作为换行;实际写入时会被规整化为目标文件的换行风格。
    """

    search_text: str
    replace_text: str


@dataclass(frozen=True)
class PatchOperation:
    """单个文件操作的解析结果。

    Add / Replace 操作使用 ``content`` 字段;Update 使用 ``blocks`` 字段;
    Delete 两个字段都为空。
    """

    kind: OperationKind
    path: str
    content: str = ""
    blocks: tuple[SearchReplaceBlock, ...] = ()


@dataclass(frozen=True)
class PatchDocument:
    operations: tuple[PatchOperation, ...]


@dataclass(frozen=True)
class ExistingFileContent:
    text: str
    byte_size: int
    has_utf8_bom: bool = False
    newline: str = "\n"
    has_trailing_newline: bool = True


@dataclass(frozen=True)
class ValidatedOperation:
    operation: PatchOperation
    relative_path: Path
    target_path: Path
    existing_content: ExistingFileContent | None = None


@dataclass(frozen=True)
class ValidatedPatch:
    workspace_root: Path
    operations: tuple[ValidatedOperation, ...]
    patch_text_size: int | None = None
    final_output_size_limit: int = TARGET_FILE_LIMIT_BYTES


@dataclass(frozen=True)
class LineStats:
    old_line_count: int
    new_line_count: int
    added_lines: int
    deleted_lines: int


@dataclass(frozen=True)
class UpdateApplyResult:
    final_bytes: bytes
    stats: LineStats
    newline: str
    has_utf8_bom: bool


@dataclass(frozen=True)
class TargetSnapshot:
    exists: bool
    is_symlink: bool
    is_directory: bool
    size: int | None = None
    mtime_ns: int | None = None
    mode: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class PlannedOperation:
    kind: OperationKind
    path: str
    final_size: int
    stats: LineStats
    target_path: Path
    relative_path: Path
    final_bytes: bytes | None
    original_snapshot: TargetSnapshot


@dataclass(frozen=True)
class PlannedPatch:
    operations: tuple[PlannedOperation, ...]
    total_final_size: int
    workspace_root: Path
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BlockMatch:
    start_line: int
    end_line: int  # exclusive
    block: SearchReplaceBlock


@dataclass(frozen=True)
class _AppliedOperation:
    kind: OperationKind
    target_path: Path
    workspace_root: Path
    relative_path: Path
    backup_path: Path | None = None
    added_path: Path | None = None


@dataclass
class _ApplyState:
    applied_operations: list[_AppliedOperation]
    temp_paths: list[Path]
    created_dirs: list[Path]


class PatchError(ValueError):
    """可被模型修复的 apply_patch 失败。

    错误结构包含 code / message / phase / line / file / hint / expected /
    actual 字段,便于 LLM 根据信息直接重写补丁,无需猜测。
    """

    code: str
    message: str
    line: int | None
    file: str | None
    block: str | None
    expected: str | None
    actual: str | None
    hint: str | None

    def __init__(
        self,
        code: str,
        message: str,
        *,
        line: int | None = None,
        file: str | None = None,
        block: str | None = None,
        expected: str | None = None,
        actual: str | None = None,
        hint: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.line = line
        self.file = file
        self.block = block
        self.expected = expected
        self.actual = actual
        self.hint = hint
        super().__init__(self.display_message)

    @property
    def display_message(self) -> str:
        if self.line is None:
            return self.message
        return f"{self.message} (line {self.line})"

    def to_error_result(self, *, phase: str | None = None) -> str:
        return _format_error(
            self.code,
            self.message,
            phase=phase,
            line=self.line,
            file=self.file,
            block=self.block,
            expected=self.expected,
            actual=self.actual,
            hint=self.hint or _error_hint(self.code),
        )


# ──────────────── 错误格式化 ────────────────
def _clip_debug_text(text: str, limit: int = 1600) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n...[truncated {omitted} chars]"


def _indent_block(text: str, prefix: str = "  ") -> str:
    if text == "":
        return prefix + "<empty>"
    return "\n".join(prefix + line for line in text.splitlines())


def _format_error(
    code: str,
    detail: str,
    *,
    phase: str | None = None,
    line: int | None = None,
    file: str | None = None,
    block: str | None = None,
    expected: str | None = None,
    actual: str | None = None,
    hint: str | None = None,
) -> str:
    lines = [
        f"[Error][APPLY_PATCH][{code}]",
        f"message: {detail}",
    ]
    if phase:
        lines.append(f"phase: {phase}")
    if line is not None:
        lines.append(f"line: {line}")
    if file:
        lines.append(f"file: {file}")
    if block:
        lines.append(f"block: {block}")
    if expected is not None:
        lines.append("expected:")
        lines.append(_indent_block(_clip_debug_text(expected)))
    if actual is not None:
        lines.append("actual:")
        lines.append(_indent_block(_clip_debug_text(actual)))
    if hint:
        lines.append(f"hint: {hint}")
    return "\n".join(lines)


def _format_warning(code: str, detail: str) -> str:
    return f"[Warning][APPLY_PATCH][{code}] {detail}"


def _error_hint(code: str) -> str:
    hints = {
        TARGET_EXISTS: (
            "如果要修改现有文件,请使用 *** Update File:;"
            "如果要整体覆盖,请使用 *** Replace File:。"
        ),
        TARGET_MISSING: "请确认目标文件存在,或改用 *** Add File:。",
        TARGET_IS_DIRECTORY: "请指定普通文件路径,不能指定目录。",
        FINAL_TOO_LARGE: "请拆分补丁或缩小目标文件内容。",
        SEARCH_NOT_FOUND: (
            "请重新读取目标文件对应区域,使用精确原文(包含空格、缩进、标点)"
            "构造 SEARCH 块。"
        ),
        AMBIGUOUS_MATCH: (
            "SEARCH 在文件中出现多次。请扩大 SEARCH 的上下文范围,直到它在"
            "目标文件中只出现一次。"
        ),
        BLOCK_OVERLAP: (
            "多个 SEARCH 块在文件中的匹配区域重叠。请合并为一个块,"
            "或重新选择不重叠的上下文。"
        ),
        EMPTY_SEARCH_NON_EMPTY_FILE: (
            "空 SEARCH 块只能用于空文件。要整体覆盖现有文件,请使用 "
            "*** Replace File:。"
        ),
        MALFORMED_BLOCK: (
            "SEARCH/REPLACE 块格式应为:<<<<<<< SEARCH / 内容 / ======= / "
            "内容 / >>>>>>> REPLACE。CONTENT 块格式应为:<<<<<<< CONTENT / "
            "内容 / >>>>>>> END。"
        ),
        MALFORMED_SECTION: (
            "请检查 *** Add/Update/Delete/Replace File: 行;Add 与 Replace "
            "需要 CONTENT 块,Update 需要至少一个 SEARCH/REPLACE 块,Delete 不需要正文。"
        ),
        INVALID_ENVELOPE: (
            "请输出完整 v2 patch:以 *** Begin Patch 开始,以 *** End Patch 结束。"
            "不要使用 git diff 格式(diff --git / --- / +++ / @@)。"
        ),
        TARGET_CHANGED: "请重新读取目标文件后再生成补丁。",
        ROLLBACK_FAILED: "请检查残留备份或临时文件后手动处理。",
        APPLY_FAILED: "请检查权限、磁盘空间和目标路径状态。",
        CLEANUP_FAILED: "补丁已成功写入,请检查并手动删除残留临时或备份文件。",
        INVALID_PATH: "请使用工作区内的安全相对路径。",
        SENSITIVE_PATH: "请不要通过补丁写入密钥、令牌或环境配置。",
        BINARY_CONTENT: "apply_patch 只处理 UTF-8 文本文件;请不要补丁二进制内容。",
        NOOP_PATCH: (
            "补丁不会改变目标内容。如果目标已满足要求,无需修改;"
            "否则请检查 SEARCH 块和 REPLACE 内容是否一致。"
        ),
        MIXED_NEWLINES: "请确保目标文件统一使用 LF 或 CRLF 换行。",
        DUPLICATE_FILE_OPERATION: "每个文件在补丁中只能出现一次操作。",
    }
    return hints.get(code, "请检查补丁格式、路径和目标文件状态后重试。")


# ──────────────── 解析器 ────────────────
def parse_patch_text(patch_text: str) -> PatchDocument:
    """把 v2 patch 文本解析为 PatchDocument(块感知,支持标签)。

    解析器一次性按行从上到下走读,关键不变量:**进入 CONTENT 或 SEARCH/
    REPLACE 块后,只看自己的 close 标记**。这意味着块内可以含字面的
    ``*** End Patch`` / ``*** Add File:`` / 其它 ``<<<<<<<`` 等,都视为内容。

    标签化块(可选):为了让块内容含字面 ``>>>>>>> END`` / ``>>>>>>> REPLACE``
    时也能解析,允许在打开标记后写一个非空白 tag,close 必须带相同 tag::

        <<<<<<< CONTENT mydoc
        content with >>>>>>> END inside
        >>>>>>> END mydoc

    严格模式:不接受 git diff 片段(``diff --git`` / ``---`` / ``+++`` /
    ``@@``)。
    """
    _ = validate_patch_text_size(patch_text)
    lines = patch_text.splitlines()
    if not lines or lines[0] != BEGIN_MARKER:
        raise PatchError(
            INVALID_ENVELOPE,
            "patch must start with the begin marker",
            line=1,
            expected=BEGIN_MARKER,
            actual=lines[0] if lines else "<empty patch>",
            hint="请输出完整 v2 patch 信封,不要输出 git diff。",
        )

    operations: list[PatchOperation] = []
    seen_paths: set[str] = set()
    cursor = 1
    end_marker_index: int | None = None

    while cursor < len(lines):
        line = lines[cursor]
        line_number = cursor + 1

        if _is_blank_separator(line):
            cursor += 1
            continue

        if line == END_MARKER:
            end_marker_index = cursor
            cursor += 1
            break

        if _GIT_DIFF_HINT_RE.match(line):
            raise PatchError(
                MALFORMED_SECTION,
                "v2 patch does not accept git diff syntax",
                line=line_number,
                actual=line,
            )

        section_match = _SECTION_RE.match(line)
        if section_match is None:
            # 提前关闭诊断:回扫近 30 行,看是否有刚关闭过的 CONTENT/REPLACE 块。
            # 如果当前 line 离最近的 close 标记不远(典型 markdown
            # 文档场景),很可能是块内含字面 close 标记,导致块被提前关闭。
            diagnostic_hint = _diagnose_premature_close(lines, cursor)
            raise PatchError(
                MALFORMED_SECTION,
                "expected a file operation section (Add/Update/Delete/Replace File)",
                line=line_number,
                actual=line,
                hint=diagnostic_hint,
            )

        section_name = section_match.group(1)
        raw_path = section_match.group(2)
        if not raw_path.strip():
            raise PatchError(
                MALFORMED_SECTION,
                "section path must not be empty",
                line=line_number,
            )
        if raw_path in seen_paths:
            raise PatchError(
                DUPLICATE_FILE_OPERATION,
                "patch contains repeated file operation path",
                line=line_number,
            )
        seen_paths.add(raw_path)

        cursor += 1
        if section_name == "Add File":
            operation, cursor = _parse_content_section(
                "add", raw_path, lines, cursor, line_number
            )
        elif section_name == "Replace File":
            operation, cursor = _parse_content_section(
                "replace", raw_path, lines, cursor, line_number
            )
        elif section_name == "Update File":
            operation, cursor = _parse_update_section(
                raw_path, lines, cursor, line_number
            )
        else:  # Delete File
            operation, cursor = _parse_delete_section(
                raw_path, lines, cursor, line_number
            )
        operations.append(operation)

    if end_marker_index is None:
        raise PatchError(
            INVALID_ENVELOPE,
            "patch is missing the end marker",
            expected=END_MARKER,
            hint="请确保补丁最后一行是 *** End Patch。",
        )

    while cursor < len(lines):
        if lines[cursor].strip():
            raise PatchError(
                INVALID_ENVELOPE,
                "only whitespace is allowed after the end marker",
                line=cursor + 1,
                actual=lines[cursor],
            )
        cursor += 1

    if not operations:
        raise PatchError(
            INVALID_ENVELOPE, "patch must contain at least one file operation"
        )

    return PatchDocument(operations=tuple(operations))


def validate_patch_text_size(patch_text: str) -> int:
    patch_size = len(patch_text.encode("utf-8"))
    if patch_size > PATCH_TEXT_LIMIT_BYTES:
        raise PatchError(PATCH_TOO_LARGE, "补丁文本超过 256 KiB 限制")
    return patch_size


def _is_blank_separator(line: str) -> bool:
    return not line.strip()


def _parse_block_tag(line: str, expected_open: str) -> str | None:
    """返回 tag 字符串(空字符串表示未带 tag);不匹配则返回 None。

    合法形式:
      ``<<<<<<< CONTENT``         → tag = ""
      ``<<<<<<< CONTENT mydoc``   → tag = "mydoc"

    tag 必须是单个非空白 token,不含控制字符。
    """
    if line == expected_open:
        return ""
    prefix = expected_open + " "
    if line.startswith(prefix):
        tag = line[len(prefix) :].strip()
        if not tag:
            return None
        if _CONTROL_CHAR_RE.search(tag):
            return None
        return tag
    return None


def _close_marker(base: str, tag: str) -> str:
    return f"{base} {tag}" if tag else base


def _diagnose_premature_close(lines: list[str], cursor: int) -> str | None:
    """当 envelope 主循环遇到非法 section 行时,检查是否是块被字面 close 提前关闭。

    回扫最近 30 行,如果发现近期(<= 5 行内)出现过 CONTENT/REPLACE close 标记,
    且当前行不像合法 section header,说明 LLM 很可能在块内字面写了 close。
    """
    look_back = max(0, cursor - 30)
    for probe in range(cursor - 1, look_back - 1, -1):
        candidate = lines[probe]
        if (
            candidate == CONTENT_CLOSE
            or candidate == REPLACE_CLOSE
            or candidate.startswith(CONTENT_CLOSE + " ")
            or candidate.startswith(REPLACE_CLOSE + " ")
        ):
            distance = cursor - probe
            if distance <= 5:
                return (
                    f"上一个块在第 {probe + 1} 行被关闭(距当前行 {distance} 行);"
                    "如果这是文档/教程内容,块里很可能含字面 '>>>>>>> END' 或 "
                    "'>>>>>>> REPLACE'。请给该块加 tag,例如 '<<<<<<< CONTENT doc' / "
                    "'>>>>>>> END doc',这样块内字面 close 标记不会被误识别。"
                )
            break
    return (
        "请检查 *** Add/Update/Delete/Replace File: 行;Add 与 Replace 需要 "
        "CONTENT 块,Update 需要至少一个 SEARCH/REPLACE 块,Delete 不需要正文。"
        "如果文件内容里需要含字面 '>>>>>>> END' / '>>>>>>> REPLACE' / '=======',"
        "请给块加 tag,例如 '<<<<<<< CONTENT doc' / '>>>>>>> END doc'。"
    )


def _parse_content_section(
    kind: OperationKind,
    path: str,
    lines: list[str],
    start: int,
    section_line_number: int,
) -> tuple[PatchOperation, int]:
    """解析 Add File / Replace File 的 CONTENT 块。"""
    cursor = start
    while cursor < len(lines) and _is_blank_separator(lines[cursor]):
        cursor += 1

    section_label = "Add" if kind == "add" else "Replace"

    if cursor >= len(lines):
        raise PatchError(
            MALFORMED_SECTION,
            f"{section_label} File requires a CONTENT block",
            line=section_line_number,
            expected=f"{CONTENT_OPEN} ... {CONTENT_CLOSE}",
            actual="<missing>",
        )

    open_line = lines[cursor]
    tag = _parse_block_tag(open_line, CONTENT_OPEN)
    if tag is None:
        # 把"另一个 section header / End Patch 直接跟在 Add File 后"的情况
        # 识别为缺少 CONTENT 块,而不是无关 section 错误
        if open_line == END_MARKER or _SECTION_RE.match(open_line):
            raise PatchError(
                MALFORMED_SECTION,
                f"{section_label} File requires a CONTENT block",
                line=section_line_number,
                expected=f"{CONTENT_OPEN} ... {CONTENT_CLOSE}",
                actual=open_line,
            )
        raise PatchError(
            MALFORMED_SECTION,
            f"{section_label} File requires a CONTENT block",
            line=cursor + 1,
            expected=f"{CONTENT_OPEN} ... {CONTENT_CLOSE}",
            actual=open_line,
        )

    close_target = _close_marker(CONTENT_CLOSE, tag)
    content_start = cursor + 1
    close_index: int | None = None
    for probe in range(content_start, len(lines)):
        if lines[probe] == close_target:
            close_index = probe
            break
    if close_index is None:
        raise PatchError(
            MALFORMED_BLOCK,
            f"CONTENT block missing closing marker {close_target}",
            line=cursor + 1,
            hint=(
                "若内容中含字面 '>>>>>>> END',请在打开标记后加一个唯一 tag,"
                "例如 '<<<<<<< CONTENT doc' / '>>>>>>> END doc'。"
            ),
        )

    content = "\n".join(lines[content_start:close_index])
    return PatchOperation(kind=kind, path=path, content=content), close_index + 1


def _parse_update_section(
    path: str,
    lines: list[str],
    start: int,
    section_line_number: int,
) -> tuple[PatchOperation, int]:
    """解析 Update File 的一个或多个 SEARCH/REPLACE 块,直到下一个 section 或 End Patch。"""
    blocks: list[SearchReplaceBlock] = []
    cursor = start

    while cursor < len(lines):
        while cursor < len(lines) and _is_blank_separator(lines[cursor]):
            cursor += 1
        if cursor >= len(lines):
            break

        line = lines[cursor]
        if line == END_MARKER or _SECTION_RE.match(line):
            break

        tag = _parse_block_tag(line, SEARCH_OPEN)
        if tag is None:
            diagnostic_hint = _diagnose_premature_close(lines, cursor) or (
                "Update File 必须由一个或多个 SEARCH/REPLACE 块组成。"
                "格式:<<<<<<< SEARCH / 内容 / ======= / 内容 / >>>>>>> REPLACE。"
            )
            raise PatchError(
                MALFORMED_BLOCK,
                f"expected {SEARCH_OPEN}",
                line=cursor + 1,
                actual=line,
                hint=diagnostic_hint,
            )

        block, cursor = _parse_search_replace_block(lines, cursor, tag)
        blocks.append(block)

    if not blocks:
        raise PatchError(
            MALFORMED_SECTION,
            "Update File requires at least one SEARCH/REPLACE block",
            line=section_line_number,
            expected=f"{SEARCH_OPEN} ... {SEARCH_DIVIDER} ... {REPLACE_CLOSE}",
        )

    return PatchOperation(kind="update", path=path, blocks=tuple(blocks)), cursor


def _parse_search_replace_block(
    lines: list[str],
    open_index: int,
    tag: str,
) -> tuple[SearchReplaceBlock, int]:
    divider_target = _close_marker(SEARCH_DIVIDER, tag)
    close_target = _close_marker(REPLACE_CLOSE, tag)

    divider_index: int | None = None
    close_index: int | None = None

    probe = open_index + 1
    while probe < len(lines):
        line = lines[probe]
        if line == close_target:
            close_index = probe
            break
        if line == divider_target and divider_index is None:
            divider_index = probe
        elif divider_index is None and _parse_block_tag(line, SEARCH_OPEN) is not None:
            # 在还没看到 divider 之前就出现新的 SEARCH 打开标记 → 大概率是漏了 divider
            raise PatchError(
                MALFORMED_BLOCK,
                "nested SEARCH block before previous block was closed",
                line=probe + 1,
            )
        probe += 1

    if divider_index is None:
        raise PatchError(
            MALFORMED_BLOCK,
            f"SEARCH block missing divider {divider_target}",
            line=open_index + 1,
            hint=(
                "若 SEARCH 内容含字面 '=======',请在打开标记后加一个唯一 tag,"
                "例如 '<<<<<<< SEARCH x' / '======= x' / '>>>>>>> REPLACE x'。"
            ),
        )
    if close_index is None:
        raise PatchError(
            MALFORMED_BLOCK,
            f"SEARCH block missing closing marker {close_target}",
            line=open_index + 1,
            hint=(
                "若 REPLACE 内容含字面 '>>>>>>> REPLACE',请在打开标记后加一个唯一 tag,"
                "例如 '<<<<<<< SEARCH x' 与 '>>>>>>> REPLACE x'。"
            ),
        )

    search_lines = lines[open_index + 1 : divider_index]
    replace_lines = lines[divider_index + 1 : close_index]
    return (
        SearchReplaceBlock(
            search_text="\n".join(search_lines),
            replace_text="\n".join(replace_lines),
        ),
        close_index + 1,
    )


def _parse_delete_section(
    path: str,
    lines: list[str],
    start: int,
    section_line_number: int,
) -> tuple[PatchOperation, int]:
    _ = section_line_number
    cursor = start
    while cursor < len(lines):
        line = lines[cursor]
        if _is_blank_separator(line):
            cursor += 1
            continue
        if line == END_MARKER or _SECTION_RE.match(line):
            break
        raise PatchError(
            MALFORMED_SECTION,
            "Delete File section does not accept a body",
            line=cursor + 1,
            actual=line,
        )
    return PatchOperation(kind="delete", path=path), cursor


# ──────────────── SEARCH/REPLACE 应用核心 ────────────────
def apply_update_blocks(
    existing_content: ExistingFileContent,
    blocks: tuple[SearchReplaceBlock, ...],
) -> UpdateApplyResult:
    """把若干 SEARCH/REPLACE 块按行对齐应用到现有文本。

    工作方式:

    1. 把现有文本按行切分(保留空行)。
    2. 对每个块的 SEARCH 在文件行序列中查找精确且唯一的连续匹配。
    3. 验证不同块的匹配区间互不重叠。
    4. 按行位置排序后,从后往前替换(避免位置失效)。

    空 SEARCH 仅在文件本身为空时合法,等价于"在空文件中写入内容";
    其它情况应使用 ``*** Replace File:``。
    """
    if not blocks:
        raise PatchError(NOOP_PATCH, "更新补丁没有可应用的块")

    file_lines = _split_lines_preserve(existing_content.text)
    matches = _find_block_matches(file_lines, blocks)
    _ensure_no_overlap(matches)

    new_file_lines = _apply_matches(file_lines, matches)
    final_text = _join_lines_preserve(
        new_file_lines,
        newline=existing_content.newline,
        trailing_newline=existing_content.has_trailing_newline,
    )
    final_payload = final_text.encode("utf-8")
    final_bytes = (
        UTF8_BOM + final_payload if existing_content.has_utf8_bom else final_payload
    )

    # Re-render the original through the same join logic so the NOOP
    # comparison is not fooled by the stripped trailing newline in
    # existing_content.text.
    original_text = _join_lines_preserve(
        file_lines,
        newline=existing_content.newline,
        trailing_newline=existing_content.has_trailing_newline,
    )
    original_payload = original_text.encode("utf-8")
    original_bytes = (
        UTF8_BOM + original_payload
        if existing_content.has_utf8_bom
        else original_payload
    )
    _ensure_likely_text_bytes(final_bytes, subject="计划输出")
    if final_bytes == original_bytes:
        raise PatchError(NOOP_PATCH, "更新补丁不会改变目标内容")

    added_lines = sum(len(_split_block_lines(b.replace_text)) for b in blocks)
    deleted_lines = sum(len(_split_block_lines(b.search_text)) for b in blocks)
    return UpdateApplyResult(
        final_bytes=final_bytes,
        stats=LineStats(
            old_line_count=len(file_lines),
            new_line_count=len(new_file_lines),
            added_lines=added_lines,
            deleted_lines=deleted_lines,
        ),
        newline=existing_content.newline,
        has_utf8_bom=existing_content.has_utf8_bom,
    )


def _split_lines_preserve(text: str) -> list[str]:
    """按 \\n 切行,保留空行。空字符串返回空列表。"""
    if text == "":
        return []
    return text.split("\n")


def _join_lines_preserve(
    file_lines: list[str], *, newline: str, trailing_newline: bool
) -> str:
    if not file_lines:
        return ""
    joined = newline.join(file_lines)
    if trailing_newline:
        joined += newline
    return joined


def _split_block_lines(text: str) -> list[str]:
    """块内容按行切分(空字符串视为空列表)。"""
    if text == "":
        return []
    return text.split("\n")


def _detect_existing_newline(raw_text: str) -> tuple[str, bool, str]:
    """识别现有文本的换行风格,并返回 (规整化后的 \\n 文本, trailing_newline, 原 newline)。

    规整化逻辑:把 CRLF 统一替换为 LF;混合换行报错。
    """
    if raw_text == "":
        return "", False, "\n"

    has_lf = "\n" in raw_text
    crlf_count = raw_text.count("\r\n")
    lone_lf_count = raw_text.count("\n") - crlf_count
    # 检查孤立 \r
    stripped_crlf = raw_text.replace("\r\n", "\n")
    if "\r" in stripped_crlf:
        raise PatchError(MIXED_NEWLINES, "目标文件包含不支持的孤立 CR 换行")

    if crlf_count > 0 and lone_lf_count > 0:
        raise PatchError(MIXED_NEWLINES, "目标文件包含 LF 与 CRLF 混合换行")

    newline = "\r\n" if crlf_count > 0 else "\n"
    normalized = stripped_crlf
    trailing_newline = has_lf and normalized.endswith("\n")
    if trailing_newline:
        normalized = normalized[:-1]
    return normalized, trailing_newline, newline


def _find_block_matches(
    file_lines: list[str], blocks: tuple[SearchReplaceBlock, ...]
) -> list[_BlockMatch]:
    matches: list[_BlockMatch] = []
    for index, block in enumerate(blocks):
        search_lines = _split_block_lines(block.search_text)
        if not search_lines:
            if file_lines:
                raise PatchError(
                    EMPTY_SEARCH_NON_EMPTY_FILE,
                    "空 SEARCH 块只能用于空文件",
                    block=_block_label(index, block),
                    hint=("要整体覆盖现有文件,请改用 *** Replace File:。"),
                )
            matches.append(_BlockMatch(start_line=0, end_line=0, block=block))
            continue

        positions = _find_all_line_matches(file_lines, search_lines)
        if not positions:
            raise PatchError(
                SEARCH_NOT_FOUND,
                "SEARCH 内容在目标文件中不存在",
                block=_block_label(index, block),
                expected=block.search_text,
                actual=_clip_debug_text(
                    _join_lines_preserve(
                        file_lines, newline="\n", trailing_newline=False
                    ),
                    limit=800,
                ),
            )
        if len(positions) > 1:
            preview = ", ".join(f"line {p + 1}" for p in positions[:5])
            raise PatchError(
                AMBIGUOUS_MATCH,
                f"SEARCH 在文件中匹配多次({len(positions)} 处)",
                block=_block_label(index, block),
                actual=preview,
                hint=(
                    "请扩大 SEARCH 的上下文,使它在目标文件中只出现一次;"
                    "通常只需要在前后各加一两行原文即可。"
                ),
            )
        start = positions[0]
        matches.append(
            _BlockMatch(
                start_line=start, end_line=start + len(search_lines), block=block
            )
        )
    return matches


def _find_all_line_matches(file_lines: list[str], search_lines: list[str]) -> list[int]:
    if not search_lines:
        return []
    n = len(search_lines)
    if n > len(file_lines):
        return []
    positions: list[int] = []
    first = search_lines[0]
    for index in range(len(file_lines) - n + 1):
        if file_lines[index] != first:
            continue
        if file_lines[index : index + n] == search_lines:
            positions.append(index)
    return positions


def _ensure_no_overlap(matches: list[_BlockMatch]) -> None:
    sorted_matches = sorted(matches, key=lambda m: m.start_line)
    for left, right in zip(sorted_matches, sorted_matches[1:]):
        if left.start_line == left.end_line and right.start_line == right.end_line:
            # 两个空 SEARCH 块,理论上只允许出现一次(空文件只有一个匹配)
            raise PatchError(
                BLOCK_OVERLAP,
                "多个空 SEARCH 块不能同时使用",
                hint="一个 Update File 中只能有一个空 SEARCH 块。",
            )
        if left.end_line > right.start_line:
            raise PatchError(
                BLOCK_OVERLAP,
                "多个 SEARCH 块的匹配区域重叠",
                actual=(
                    f"block A: lines {left.start_line + 1}-{left.end_line}; "
                    f"block B: lines {right.start_line + 1}-{right.end_line}"
                ),
            )


def _apply_matches(file_lines: list[str], matches: list[_BlockMatch]) -> list[str]:
    sorted_matches = sorted(matches, key=lambda m: m.start_line)
    new_lines: list[str] = []
    cursor = 0
    for match in sorted_matches:
        new_lines.extend(file_lines[cursor : match.start_line])
        new_lines.extend(_split_block_lines(match.block.replace_text))
        cursor = match.end_line
    new_lines.extend(file_lines[cursor:])
    return new_lines


def _block_label(index: int, block: SearchReplaceBlock) -> str:
    preview = block.search_text.replace("\n", " ⏎ ")
    if len(preview) > 60:
        preview = preview[:57] + "..."
    return f"block #{index + 1} (search: {preview})"


def build_updated_file_bytes(
    existing_content: ExistingFileContent, blocks: tuple[SearchReplaceBlock, ...]
) -> bytes:
    return apply_update_blocks(existing_content, blocks).final_bytes


# ──────────────── 校验:路径 / 敏感性 / 工作区 ────────────────
def validate_patch_document(
    document: PatchDocument,
    *,
    patch_text: str | None = None,
    workspace_root: Path | None = None,
) -> ValidatedPatch:
    workspace_root = (
        Path.cwd().resolve() if workspace_root is None else workspace_root.resolve()
    )
    patch_text_size = (
        validate_patch_text_size(patch_text) if patch_text is not None else None
    )
    validated_operations: list[ValidatedOperation] = []
    seen_targets: set[Path] = set()

    for operation in document.operations:
        relative_path = _validate_operation_path_syntax(
            operation.path, workspace_root=workspace_root
        )
        _validate_sensitive_path(relative_path)
        _validate_existing_ancestors(workspace_root, relative_path)
        target_path = _resolve_workspace_target(workspace_root, relative_path)

        if target_path in seen_targets:
            raise PatchError(DUPLICATE_FILE_OPERATION, "补丁包含重复的规范化目标路径")
        seen_targets.add(target_path)

        existing_content = _validate_target_state(operation.kind, target_path)
        validated_operations.append(
            ValidatedOperation(
                operation=operation,
                relative_path=relative_path,
                target_path=target_path,
                existing_content=existing_content,
            )
        )

    return ValidatedPatch(
        workspace_root=workspace_root,
        operations=tuple(validated_operations),
        patch_text_size=patch_text_size,
    )


def _validate_operation_path_syntax(
    raw_path: str, *, workspace_root: Path | None = None
) -> Path:
    if not raw_path or not raw_path.strip():
        raise PatchError(INVALID_PATH, "路径不能为空")
    if raw_path != raw_path.strip():
        raise PatchError(INVALID_PATH, "路径不能包含首尾空白")
    if _CONTROL_CHAR_RE.search(raw_path):
        raise PatchError(INVALID_PATH, "路径不能包含控制字符")
    if raw_path.startswith("~"):
        raise PatchError(INVALID_PATH, "路径不能使用用户目录缩写")

    workspace_root = (
        Path.cwd().resolve() if workspace_root is None else workspace_root.resolve()
    )
    if _WINDOWS_DRIVE_RE.match(raw_path):
        return _validate_windows_drive_path(raw_path, workspace_root)

    normalized_raw_path = raw_path.replace("\\", "/")
    if normalized_raw_path.endswith("/"):
        raise PatchError(INVALID_PATH, "路径不能以目录分隔符结尾")

    return _normalize_relative_operation_path(normalized_raw_path)


def _validate_windows_drive_path(raw_path: str, workspace_root: Path) -> Path:
    if not _WINDOWS_DRIVE_ABSOLUTE_RE.match(raw_path):
        raise PatchError(INVALID_PATH, "Windows drive-relative 路径不安全")
    if not _is_windows_runtime():
        raise PatchError(INVALID_PATH, "Windows 盘符绝对路径必须位于当前工作区内")
    relative_path = _windows_drive_absolute_to_relative(raw_path, workspace_root)
    return _normalize_relative_operation_path(
        relative_path.as_posix().replace("\\", "/")
    )


def _is_windows_runtime() -> bool:
    return os.name == "nt"


def _windows_drive_absolute_to_relative(raw_path: str, workspace_root: Path) -> Path:
    target_path = Path(raw_path).resolve(strict=False)
    try:
        return target_path.relative_to(workspace_root)
    except ValueError as exc:
        raise PatchError(INVALID_PATH, "Windows 盘符路径位于工作区之外") from exc


def _normalize_relative_operation_path(raw_path: str) -> Path:
    pure_path = PurePosixPath(raw_path)
    if pure_path.is_absolute():
        raise PatchError(INVALID_PATH, "路径必须相对工作区")

    parts = pure_path.parts
    if not parts:
        raise PatchError(INVALID_PATH, "路径不能为空")

    normalized_parts: list[str] = []
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise PatchError(INVALID_PATH, "路径不能包含上级目录片段")
        if part == ".git":
            raise PatchError(INVALID_PATH, "路径不能指向 .git 目录")
        normalized_parts.append(part)

    if not normalized_parts:
        raise PatchError(INVALID_PATH, "路径不能为空")

    return Path(*normalized_parts)


def _validate_sensitive_path(relative_path: Path) -> None:
    for component in relative_path.parts:
        lower_component = component.lower()
        if lower_component in {".env", ".envrc"}:
            raise PatchError(SENSITIVE_PATH, "路径指向敏感环境配置")
        if lower_component.startswith(".env.") and lower_component != ".env.example":
            raise PatchError(SENSITIVE_PATH, "路径指向敏感环境配置")

        tokens = [token for token in _PATH_TOKEN_RE.split(lower_component) if token]
        if any(token in _SENSITIVE_PATH_TOKENS for token in tokens):
            raise PatchError(SENSITIVE_PATH, "路径包含敏感凭据命名片段")


def _validate_existing_ancestors(workspace_root: Path, relative_path: Path) -> None:
    current = workspace_root
    for part in relative_path.parts[:-1]:
        current = current / part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            return

        if stat.S_ISLNK(current_stat.st_mode):
            raise PatchError(TARGET_IS_SYMLINK, "路径父级不能是符号链接")
        if not stat.S_ISDIR(current_stat.st_mode):
            raise PatchError(INVALID_PATH, "路径父级必须是目录")


def _resolve_workspace_target(workspace_root: Path, relative_path: Path) -> Path:
    target_path = workspace_root / relative_path
    try:
        target_lstat = target_path.lstat()
    except FileNotFoundError:
        target_lstat = None

    if target_lstat is not None and stat.S_ISLNK(target_lstat.st_mode):
        raise PatchError(TARGET_IS_SYMLINK, "目标不能是符号链接")

    resolved_target = target_path.resolve(strict=False)
    try:
        _ = resolved_target.relative_to(workspace_root)
    except ValueError as exc:
        raise PatchError(INVALID_PATH, "路径解析后位于工作区之外") from exc
    return resolved_target


def _validate_target_state(
    kind: OperationKind, target_path: Path
) -> ExistingFileContent | None:
    try:
        target_stat = target_path.lstat()
    except FileNotFoundError:
        if kind == "add":
            return None
        raise PatchError(TARGET_MISSING, "更新、覆盖或删除目标必须存在") from None

    if stat.S_ISLNK(target_stat.st_mode):
        raise PatchError(TARGET_IS_SYMLINK, "目标不能是符号链接")
    if stat.S_ISDIR(target_stat.st_mode):
        raise PatchError(TARGET_IS_DIRECTORY, "目标不能是目录")
    if not stat.S_ISREG(target_stat.st_mode):
        raise PatchError(INVALID_PATH, "目标必须是普通文件")

    if kind == "add":
        raise PatchError(TARGET_EXISTS, "新增目标已经存在")
    if target_stat.st_size > TARGET_FILE_LIMIT_BYTES:
        raise PatchError(TARGET_TOO_LARGE, "目标文件超过 1 MiB 限制")
    if kind == "update":
        return _read_existing_utf8(target_path, target_stat.st_size)
    # replace / delete 不需要在此返回内容(replace 整体覆盖,delete 只需备份)
    return None


def _read_existing_utf8(target_path: Path, byte_size: int) -> ExistingFileContent:
    existing_bytes, descriptor_stat = _read_regular_file_bytes_no_follow(target_path)
    if descriptor_stat.st_size != byte_size:
        raise PatchError(TARGET_CHANGED, "目标文件在读取期间发生变化")
    return _decode_existing_utf8_bytes(existing_bytes)


def _decode_existing_utf8_bytes(existing_bytes: bytes) -> ExistingFileContent:
    has_utf8_bom = existing_bytes.startswith(UTF8_BOM)
    payload = existing_bytes[len(UTF8_BOM) :] if has_utf8_bom else existing_bytes
    try:
        raw_text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError(INVALID_UTF8, "目标文件不是有效的 UTF-8 文本") from exc
    _ensure_likely_text_bytes(payload, subject="目标文件")
    normalized_text, trailing_newline, newline = _detect_existing_newline(raw_text)
    return ExistingFileContent(
        text=normalized_text,
        byte_size=len(existing_bytes),
        has_utf8_bom=has_utf8_bom,
        newline=newline,
        has_trailing_newline=trailing_newline,
    )


def _ensure_likely_text_bytes(content: bytes, *, subject: str) -> None:
    for byte in content:
        if (byte < 0x20 or byte == 0x7F) and byte not in _ALLOWED_TEXT_CONTROL_BYTES:
            raise PatchError(
                BINARY_CONTENT, f"{subject}包含疑似二进制控制字节 0x{byte:02x}"
            )


def _ensure_final_size(final_bytes: bytes) -> None:
    if len(final_bytes) > TARGET_FILE_LIMIT_BYTES:
        raise PatchError(FINAL_TOO_LARGE, "计划输出超过 1 MiB 限制")


# ──────────────── 文件锁 / 安全读取 ────────────────
def _acquire_document_locks(
    document: PatchDocument, workspace_root: Path
) -> list[threading.Lock]:
    lock_keys = _document_target_lock_keys(document, workspace_root)
    return _acquire_file_locks(lock_keys)


def _document_target_lock_keys(
    document: PatchDocument, workspace_root: Path
) -> tuple[str, ...]:
    lock_keys: list[str] = []
    for operation in document.operations:
        relative_path = _validate_operation_path_syntax(
            operation.path, workspace_root=workspace_root
        )
        _validate_existing_ancestors(workspace_root, relative_path)
        target_path = _resolve_workspace_target(workspace_root, relative_path)
        lock_keys.append(_target_lock_key(target_path))
    return tuple(lock_keys)


def _target_lock_key(target_path: Path) -> str:
    return os.path.normcase(str(target_path.resolve(strict=False)))


def _acquire_file_locks(lock_keys: tuple[str, ...]) -> list[threading.Lock]:
    acquired_locks: list[threading.Lock] = []
    for lock_key in sorted(set(lock_keys)):
        file_lock = _get_file_lock(lock_key)
        _ = file_lock.acquire()
        acquired_locks.append(file_lock)
    return acquired_locks


def _get_file_lock(lock_key: str) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        file_lock = _FILE_LOCKS.get(lock_key)
        if file_lock is None:
            file_lock = threading.Lock()
            _FILE_LOCKS[lock_key] = file_lock
        return file_lock


def _release_file_locks(acquired_locks: list[threading.Lock]) -> None:
    for file_lock in reversed(acquired_locks):
        file_lock.release()


# ──────────────── 测试钩子 ────────────────
def _pre_write_revalidation_hook(planned_patch: "PlannedPatch") -> None:
    _ = planned_patch


def _before_target_open_hook(target_path: Path) -> None:
    _ = target_path


def _after_existing_bytes_read_hook(target_path: Path) -> None:
    _ = target_path


def _before_apply_operation_hook(operation: "PlannedOperation") -> None:
    _ = operation


# ──────────────── 安全读取 ────────────────
def _read_regular_file_bytes_no_follow(
    target_path: Path,
) -> tuple[bytes, os.stat_result]:
    nofollow = cast(int | None, getattr(os, "O_NOFOLLOW", None))
    if nofollow is None:
        raise PatchError(TARGET_IS_SYMLINK, "当前平台不支持安全的无跟随读取")

    _before_target_open_hook(target_path)
    try:
        file_descriptor = os.open(target_path, os.O_RDONLY | nofollow)
    except FileNotFoundError as exc:
        raise PatchError(TARGET_MISSING, "目标文件不存在") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PatchError(TARGET_IS_SYMLINK, "目标不能是符号链接") from exc
        if exc.errno == errno.ENOTDIR:
            raise PatchError(INVALID_PATH, "目标父路径不是目录") from exc
        raise PatchError(
            APPLY_FAILED, f"无法安全读取目标文件:{type(exc).__name__}"
        ) from exc

    try:
        descriptor_stat = os.fstat(file_descriptor)
        if stat.S_ISDIR(descriptor_stat.st_mode):
            raise PatchError(TARGET_IS_DIRECTORY, "目标不能是目录")
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise PatchError(INVALID_PATH, "目标必须是普通文件")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), descriptor_stat
    finally:
        os.close(file_descriptor)


def _read_existing_utf8_with_snapshot(
    target_path: Path,
) -> tuple[ExistingFileContent, TargetSnapshot]:
    try:
        before_stat = target_path.lstat()
    except FileNotFoundError as exc:
        raise PatchError(TARGET_MISSING, "目标文件在读取前已不存在") from exc

    if stat.S_ISLNK(before_stat.st_mode):
        raise PatchError(TARGET_IS_SYMLINK, "目标不能是符号链接")
    if stat.S_ISDIR(before_stat.st_mode):
        raise PatchError(TARGET_IS_DIRECTORY, "目标不能是目录")
    if not stat.S_ISREG(before_stat.st_mode):
        raise PatchError(INVALID_PATH, "目标必须是普通文件")
    if before_stat.st_size > TARGET_FILE_LIMIT_BYTES:
        raise PatchError(TARGET_TOO_LARGE, "目标文件超过 1 MiB 限制")

    existing_bytes, descriptor_stat = _read_regular_file_bytes_no_follow(target_path)
    if _stat_identity(before_stat) != _stat_identity(descriptor_stat):
        raise PatchError(TARGET_CHANGED, "目标文件在读取期间发生变化")

    _after_existing_bytes_read_hook(target_path)
    after_bytes, after_stat = _read_regular_file_bytes_no_follow(target_path)
    if (
        _stat_identity(descriptor_stat) != _stat_identity(after_stat)
        or after_bytes != existing_bytes
    ):
        raise PatchError(TARGET_CHANGED, "目标文件在读取期间发生变化")

    snapshot = TargetSnapshot(
        exists=True,
        is_symlink=False,
        is_directory=False,
        size=descriptor_stat.st_size,
        mtime_ns=descriptor_stat.st_mtime_ns,
        mode=descriptor_stat.st_mode,
        sha256=hashlib.sha256(existing_bytes).hexdigest(),
    )
    return _decode_existing_utf8_bytes(existing_bytes), snapshot


def _stat_identity(path_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        path_stat.st_mode,
        path_stat.st_size,
        path_stat.st_mtime_ns,
        path_stat.st_ino,
        path_stat.st_dev,
    )


def _capture_target_snapshot(target_path: Path) -> TargetSnapshot:
    try:
        target_stat = target_path.lstat()
    except FileNotFoundError:
        return TargetSnapshot(exists=False, is_symlink=False, is_directory=False)

    is_symlink = stat.S_ISLNK(target_stat.st_mode)
    is_directory = stat.S_ISDIR(target_stat.st_mode)
    file_hash = None
    snapshot_stat = target_stat
    if stat.S_ISREG(target_stat.st_mode) and not is_symlink:
        existing_bytes, descriptor_stat = _read_regular_file_bytes_no_follow(
            target_path
        )
        if _stat_identity(target_stat) != _stat_identity(descriptor_stat):
            raise PatchError(TARGET_CHANGED, "目标文件在快照读取期间发生变化")
        file_hash = hashlib.sha256(existing_bytes).hexdigest()
        snapshot_stat = descriptor_stat
    return TargetSnapshot(
        exists=True,
        is_symlink=is_symlink,
        is_directory=is_directory,
        size=snapshot_stat.st_size,
        mtime_ns=snapshot_stat.st_mtime_ns,
        mode=snapshot_stat.st_mode,
        sha256=file_hash,
    )


# ──────────────── 计划 / 应用 ────────────────
def plan_patch_dry_run(patch_text: str) -> PlannedPatch:
    document = parse_patch_text(patch_text)
    workspace_root = Path.cwd().resolve()
    acquired_locks = _acquire_document_locks(document, workspace_root)
    try:
        return _plan_patch_document_locked(document, patch_text, workspace_root)
    finally:
        _release_file_locks(acquired_locks)


def _plan_patch_document_locked(
    document: PatchDocument,
    patch_text: str,
    workspace_root: Path,
) -> PlannedPatch:
    validated_patch = validate_patch_document(
        document,
        patch_text=patch_text,
        workspace_root=workspace_root,
    )
    add_newline = _detect_patch_newline_for_add(patch_text)
    planned_operations: list[PlannedOperation] = []

    for validated_operation in validated_patch.operations:
        try:
            planned_operations.append(_plan_operation(validated_operation, add_newline))
        except PatchError as exc:
            raise PatchError(
                exc.code,
                exc.message,
                line=exc.line,
                file=exc.file or validated_operation.relative_path.as_posix(),
                block=exc.block,
                expected=exc.expected,
                actual=exc.actual,
                hint=exc.hint,
            ) from exc

    return PlannedPatch(
        operations=tuple(planned_operations),
        total_final_size=sum(operation.final_size for operation in planned_operations),
        workspace_root=validated_patch.workspace_root,
    )


def _detect_patch_newline_for_add(patch_text: str) -> str:
    crlf_count = patch_text.count("\r\n")
    lf_only_count = patch_text.count("\n") - crlf_count
    return "\r\n" if crlf_count and not lf_only_count else "\n"


def _plan_operation(
    validated_operation: ValidatedOperation, add_newline: str
) -> PlannedOperation:
    operation = validated_operation.operation
    if operation.kind == "add":
        return _plan_add_operation(validated_operation, add_newline)
    if operation.kind == "update":
        return _plan_update_operation(validated_operation)
    if operation.kind == "replace":
        return _plan_replace_operation(validated_operation, add_newline)
    return _plan_delete_operation(validated_operation)


def _plan_add_operation(
    validated_operation: ValidatedOperation, add_newline: str
) -> PlannedOperation:
    original_snapshot = _capture_target_snapshot(validated_operation.target_path)
    if original_snapshot.exists:
        raise PatchError(
            TARGET_EXISTS,
            "新增目标已经存在",
            hint="如果要整体覆盖现有文件,请使用 *** Replace File:。",
        )
    final_bytes = _serialize_new_file_bytes(
        validated_operation.operation.content, add_newline
    )
    _ensure_final_size(final_bytes)
    _ensure_likely_text_bytes(final_bytes, subject="新增输出")
    line_count = (
        len(_split_block_lines(validated_operation.operation.content))
        if validated_operation.operation.content
        else 0
    )
    return PlannedOperation(
        kind="add",
        path=validated_operation.relative_path.as_posix(),
        final_size=len(final_bytes),
        stats=LineStats(
            old_line_count=0,
            new_line_count=line_count,
            added_lines=line_count,
            deleted_lines=0,
        ),
        target_path=validated_operation.target_path,
        relative_path=validated_operation.relative_path,
        final_bytes=final_bytes,
        original_snapshot=original_snapshot,
    )


def _plan_update_operation(validated_operation: ValidatedOperation) -> PlannedOperation:
    existing_content, original_snapshot = _read_existing_utf8_with_snapshot(
        validated_operation.target_path
    )
    result = apply_update_blocks(existing_content, validated_operation.operation.blocks)
    _ensure_final_size(result.final_bytes)
    _ensure_likely_text_bytes(result.final_bytes, subject="更新输出")
    return PlannedOperation(
        kind="update",
        path=validated_operation.relative_path.as_posix(),
        final_size=len(result.final_bytes),
        stats=result.stats,
        target_path=validated_operation.target_path,
        relative_path=validated_operation.relative_path,
        final_bytes=result.final_bytes,
        original_snapshot=original_snapshot,
    )


def _plan_replace_operation(
    validated_operation: ValidatedOperation, patch_newline: str
) -> PlannedOperation:
    existing_content, original_snapshot = _read_existing_utf8_with_snapshot(
        validated_operation.target_path
    )
    # 复用原文件的换行风格,避免无意义的换行翻转
    new_content = validated_operation.operation.content
    final_text = _serialize_replace_text(new_content, existing_content)
    final_payload = final_text.encode("utf-8")
    final_bytes = (
        UTF8_BOM + final_payload if existing_content.has_utf8_bom else final_payload
    )
    _ensure_final_size(final_bytes)
    _ensure_likely_text_bytes(final_bytes, subject="覆盖输出")

    original_text = _join_lines_preserve(
        _split_lines_preserve(existing_content.text),
        newline=existing_content.newline,
        trailing_newline=existing_content.has_trailing_newline,
    )
    original_payload = original_text.encode("utf-8")
    original_bytes = (
        UTF8_BOM + original_payload
        if existing_content.has_utf8_bom
        else original_payload
    )
    if final_bytes == original_bytes:
        raise PatchError(NOOP_PATCH, "覆盖内容与目标完全一致")

    new_lines = _split_block_lines(new_content) if new_content else []
    old_lines = _split_lines_preserve(existing_content.text)
    return PlannedOperation(
        kind="replace",
        path=validated_operation.relative_path.as_posix(),
        final_size=len(final_bytes),
        stats=LineStats(
            old_line_count=len(old_lines),
            new_line_count=len(new_lines),
            added_lines=len(new_lines),
            deleted_lines=len(old_lines),
        ),
        target_path=validated_operation.target_path,
        relative_path=validated_operation.relative_path,
        final_bytes=final_bytes,
        original_snapshot=original_snapshot,
    )


def _plan_delete_operation(validated_operation: ValidatedOperation) -> PlannedOperation:
    existing_content, original_snapshot = _read_existing_utf8_with_snapshot(
        validated_operation.target_path
    )
    original_lines = _split_lines_preserve(existing_content.text)
    deleted_lines = len(original_lines)
    return PlannedOperation(
        kind="delete",
        path=validated_operation.relative_path.as_posix(),
        final_size=0,
        stats=LineStats(
            old_line_count=deleted_lines,
            new_line_count=0,
            added_lines=0,
            deleted_lines=deleted_lines,
        ),
        target_path=validated_operation.target_path,
        relative_path=validated_operation.relative_path,
        final_bytes=None,
        original_snapshot=original_snapshot,
    )


def _serialize_new_file_bytes(content: str, newline: str) -> bytes:
    """新建文件:始终以 newline 结尾(如果有内容)。"""
    if content == "":
        return b""
    lines = content.split("\n")
    text = newline.join(lines) + newline
    return text.encode("utf-8")


def _serialize_replace_text(
    new_content: str, existing_content: ExistingFileContent
) -> str:
    """整体覆盖:沿用原文件的换行风格与末尾换行习惯。"""
    if new_content == "":
        return ""
    lines = new_content.split("\n")
    text = existing_content.newline.join(lines)
    if existing_content.has_trailing_newline:
        text += existing_content.newline
    return text


# ──────────────── 真实写入与回滚 ────────────────
def apply_patch_to_files(patch_text: str) -> PlannedPatch:
    document = parse_patch_text(patch_text)
    workspace_root = Path.cwd().resolve()
    acquired_locks = _acquire_document_locks(document, workspace_root)
    try:
        planned_patch = _plan_patch_document_locked(
            document, patch_text, workspace_root
        )
        return _apply_planned_patch_locked(planned_patch)
    finally:
        _release_file_locks(acquired_locks)


def _apply_planned_patch_locked(planned_patch: PlannedPatch) -> PlannedPatch:
    _pre_write_revalidation_hook(planned_patch)
    _revalidate_planned_patch(planned_patch)
    state = _ApplyState(applied_operations=[], temp_paths=[], created_dirs=[])

    try:
        for operation in planned_patch.operations:
            _apply_planned_operation(operation, state, planned_patch.workspace_root)
    except PatchError:
        _rollback_apply_state(state)
        raise
    except Exception as exc:
        try:
            _rollback_apply_state(state)
        except PatchError as rollback_error:
            raise rollback_error from exc
        raise PatchError(APPLY_FAILED, f"应用补丁失败:{type(exc).__name__}") from exc

    cleanup_warnings = _cleanup_success_paths(state)
    return replace(planned_patch, warnings=cleanup_warnings)


def _revalidate_planned_patch(planned_patch: PlannedPatch) -> None:
    for operation in planned_patch.operations:
        _revalidate_planned_operation(planned_patch.workspace_root, operation)


def _revalidate_planned_operation(
    workspace_root: Path, operation: PlannedOperation
) -> None:
    _validate_existing_ancestors(workspace_root, operation.relative_path)
    target_path = _resolve_workspace_target(workspace_root, operation.relative_path)
    if target_path != operation.target_path:
        raise PatchError(TARGET_CHANGED, f"{operation.path}: 目标路径解析结果已变化")
    current_snapshot = _capture_target_snapshot(operation.target_path)
    if current_snapshot != operation.original_snapshot:
        raise PatchError(TARGET_CHANGED, f"{operation.path}: 目标文件在验证后发生变化")


def _validate_workspace_mutation_path(workspace_root: Path, path: Path) -> None:
    try:
        relative_path = path.relative_to(workspace_root)
    except ValueError as exc:
        raise PatchError(INVALID_PATH, "路径位于工作区之外") from exc

    _validate_existing_ancestors(workspace_root, relative_path)
    resolved_path = _resolve_workspace_target(workspace_root, relative_path)
    if resolved_path != path.resolve(strict=False):
        raise PatchError(TARGET_CHANGED, "路径解析结果已变化")


def _apply_planned_operation(
    operation: PlannedOperation, state: _ApplyState, workspace_root: Path
) -> None:
    _before_apply_operation_hook(operation)
    _revalidate_planned_operation(workspace_root, operation)
    if operation.kind == "add":
        _apply_add_operation(operation, state, workspace_root)
    elif operation.kind == "update":
        _apply_update_or_replace_operation(operation, state, workspace_root)
    elif operation.kind == "replace":
        _apply_update_or_replace_operation(operation, state, workspace_root)
    else:
        _apply_delete_operation(operation, state, workspace_root)


def _apply_add_operation(
    operation: PlannedOperation, state: _ApplyState, workspace_root: Path
) -> None:
    if operation.final_bytes is None:
        raise PatchError(APPLY_FAILED, f"{operation.path}: 新增内容缺失")
    _create_missing_parent_dirs(operation.target_path.parent, state, workspace_root)
    temp_path = _write_temp_file(
        operation.target_path, operation.final_bytes, 0o644, state, workspace_root
    )
    _validate_workspace_mutation_path(workspace_root, operation.target_path)
    os.replace(temp_path, operation.target_path)
    state.applied_operations.append(
        _AppliedOperation(
            kind="add",
            target_path=operation.target_path,
            workspace_root=workspace_root,
            relative_path=operation.relative_path,
            added_path=operation.target_path,
        )
    )


def _apply_update_or_replace_operation(
    operation: PlannedOperation, state: _ApplyState, workspace_root: Path
) -> None:
    if operation.final_bytes is None:
        raise PatchError(APPLY_FAILED, f"{operation.path}: 输出内容缺失")
    backup_path = _make_sidecar_path(
        operation.target_path, "backup", state, workspace_root
    )
    _revalidate_planned_operation(workspace_root, operation)
    os.replace(operation.target_path, backup_path)
    state.applied_operations.append(
        _AppliedOperation(
            kind=operation.kind,
            target_path=operation.target_path,
            workspace_root=workspace_root,
            relative_path=operation.relative_path,
            backup_path=backup_path,
        )
    )
    mode = (
        operation.original_snapshot.mode
        if operation.original_snapshot.mode is not None
        else 0o644
    )
    temp_path = _write_temp_file(
        operation.target_path, operation.final_bytes, mode, state, workspace_root
    )
    _validate_workspace_mutation_path(workspace_root, operation.target_path)
    os.replace(temp_path, operation.target_path)


def _apply_delete_operation(
    operation: PlannedOperation, state: _ApplyState, workspace_root: Path
) -> None:
    backup_path = _make_sidecar_path(
        operation.target_path, "backup", state, workspace_root
    )
    _revalidate_planned_operation(workspace_root, operation)
    os.replace(operation.target_path, backup_path)
    state.applied_operations.append(
        _AppliedOperation(
            kind="delete",
            target_path=operation.target_path,
            workspace_root=workspace_root,
            relative_path=operation.relative_path,
            backup_path=backup_path,
        )
    )


def _create_missing_parent_dirs(
    parent_path: Path, state: _ApplyState, workspace_root: Path
) -> None:
    try:
        relative_parent = parent_path.relative_to(workspace_root)
    except ValueError as exc:
        raise PatchError(INVALID_PATH, "父目录位于工作区之外") from exc

    current = workspace_root
    for part in relative_parent.parts:
        current = current / part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            state.created_dirs.append(current)
            continue
        if stat.S_ISLNK(current_stat.st_mode):
            raise PatchError(TARGET_IS_SYMLINK, "父目录不能是符号链接")
        if not stat.S_ISDIR(current_stat.st_mode):
            raise PatchError(INVALID_PATH, "父路径组件不是目录")


def _make_sidecar_path(
    target_path: Path, label: str, state: _ApplyState, workspace_root: Path
) -> Path:
    _validate_workspace_mutation_path(workspace_root, target_path)
    file_descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target_path.name}.apply-patch-{label}-",
        dir=str(target_path.parent),
    )
    os.close(file_descriptor)
    sidecar_path = Path(raw_path)
    state.temp_paths.append(sidecar_path)
    return sidecar_path


def _write_temp_file(
    target_path: Path,
    content: bytes,
    mode: int,
    state: _ApplyState,
    workspace_root: Path,
) -> Path:
    _validate_workspace_mutation_path(workspace_root, target_path)
    file_descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target_path.name}.apply-patch-temp-",
        dir=str(target_path.parent),
    )
    temp_path = Path(raw_path)
    state.temp_paths.append(temp_path)
    try:
        with os.fdopen(file_descriptor, "wb") as temp_file:
            _ = temp_file.write(content)
        _validate_workspace_mutation_path(workspace_root, temp_path)
        os.chmod(temp_path, stat.S_IMODE(mode))
    except Exception:
        _remove_path_if_exists(temp_path)
        raise
    return temp_path


def _rollback_apply_state(state: _ApplyState) -> None:
    failures: list[str] = []
    for operation in reversed(state.applied_operations):
        try:
            _rollback_operation(operation)
        except Exception:
            failures.append(_safe_display_path(operation.target_path))

    for temp_path in reversed(state.temp_paths):
        try:
            _remove_path_if_exists(temp_path)
        except Exception:
            failures.append(_safe_display_path(temp_path))

    for directory_path in reversed(state.created_dirs):
        try:
            if directory_path.exists():
                directory_path.rmdir()
        except OSError:
            pass
        except Exception:
            failures.append(_safe_display_path(directory_path))

    if failures:
        residuals = ", ".join(failures[:5])
        raise PatchError(ROLLBACK_FAILED, f"回滚失败;残留路径: {residuals}")


def _rollback_operation(operation: _AppliedOperation) -> None:
    if operation.kind == "add":
        if operation.added_path is not None:
            _validate_workspace_mutation_path(
                operation.workspace_root, operation.added_path
            )
            _remove_path_if_exists(operation.added_path)
        return

    if operation.backup_path is None:
        raise PatchError(ROLLBACK_FAILED, "备份文件缺失,无法回滚")
    _validate_workspace_mutation_path(operation.workspace_root, operation.backup_path)
    _validate_workspace_mutation_path(operation.workspace_root, operation.target_path)
    if not operation.backup_path.exists():
        raise PatchError(ROLLBACK_FAILED, "备份文件缺失,无法回滚")
    _remove_path_if_exists(operation.target_path)
    _validate_workspace_mutation_path(operation.workspace_root, operation.backup_path)
    _validate_workspace_mutation_path(operation.workspace_root, operation.target_path)
    os.replace(operation.backup_path, operation.target_path)


def _cleanup_success_paths(state: _ApplyState) -> tuple[str, ...]:
    failures: list[str] = []
    for temp_path in reversed(state.temp_paths):
        try:
            _remove_path_if_exists(temp_path)
        except Exception:
            failures.append(_safe_display_path(temp_path))
    if failures:
        residuals = ", ".join(failures[:5])
        return (f"补丁已应用,但清理临时或备份文件失败;残留路径: {residuals}",)
    return ()


def _remove_path_if_exists(path: Path) -> None:
    _validate_workspace_mutation_path(Path.cwd().resolve(), path)
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _safe_display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


# ──────────────── 结果格式化 ────────────────
_OPERATION_LABELS: dict[OperationKind, str] = {
    "add": "Add",
    "update": "Update",
    "replace": "Replace",
    "delete": "Delete",
}


def format_dry_run_result(planned_patch: PlannedPatch) -> str:
    counts = _count_operations(planned_patch.operations)
    lines = [
        (
            "[DRY-RUN OK] 补丁验证通过,计划 "
            + f"{len(planned_patch.operations)} 个文件;"
            + f"Add {counts['add']}、Update {counts['update']}、"
            + f"Replace {counts['replace']}、Delete {counts['delete']}。"
            "未写入文件。"
        ),
        "验证: 路径、安全、大小、UTF-8 与 SEARCH 唯一性检查已通过。",
    ]
    for operation in planned_patch.operations:
        lines.append(_format_planned_operation_line(operation))
    return "\n".join(lines)


def format_apply_result(planned_patch: PlannedPatch) -> str:
    counts = _count_operations(planned_patch.operations)
    lines = [
        (
            "[OK] Applied patch: "
            + f"{len(planned_patch.operations)} 个文件;"
            + f"Add {counts['add']}、Update {counts['update']}、"
            + f"Replace {counts['replace']}、Delete {counts['delete']}。"
        )
    ]
    for operation in planned_patch.operations:
        lines.append(_format_planned_operation_line(operation))
    for warning in planned_patch.warnings:
        lines.append(_format_warning(CLEANUP_FAILED, warning))
    return "\n".join(lines)


def _count_operations(
    operations: tuple[PlannedOperation, ...],
) -> dict[OperationKind, int]:
    counts: dict[OperationKind, int] = {
        "add": 0,
        "update": 0,
        "replace": 0,
        "delete": 0,
    }
    for operation in operations:
        counts[operation.kind] += 1
    return counts


def _format_planned_operation_line(operation: PlannedOperation) -> str:
    label = _OPERATION_LABELS[operation.kind]
    return (
        f"- {label}: {operation.path} "
        f"(+{operation.stats.added_lines}/-{operation.stats.deleted_lines}, "
        f"final {operation.final_size} bytes)"
    )


def _dry_run_patch_result(patch_text: str) -> str:
    try:
        return format_dry_run_result(plan_patch_dry_run(patch_text))
    except PatchError as exc:
        return _format_patch_error(exc, phase="dry-run")


def _apply_patch_result(patch_text: str) -> str:
    try:
        planned_patch = apply_patch_to_files(patch_text)
        return format_apply_result(planned_patch)
    except PatchError as exc:
        return _format_patch_error(exc, phase="apply")


def _format_patch_error(error: PatchError, *, phase: str) -> str:
    return error.to_error_result(phase=phase)


# ──────────────── 工具入口 ────────────────
@tool
def apply_patch(patch_text: str, dry_run: bool = True) -> str:
    """解析并应用 v2 apply_patch 补丁(SEARCH/REPLACE 风格)。

    协议要点
    ~~~~~~~~

    每个补丁以 ``*** Begin Patch`` 开始,以 ``*** End Patch`` 结束。中间是
    一个或多个文件操作:

    * ``*** Add File: path`` —— 新建文件(目标必须不存在),用 CONTENT 块。
    * ``*** Update File: path`` —— 局部修改,用一个或多个 SEARCH/REPLACE 块。
    * ``*** Replace File: path`` —— 整体覆盖现有文件,用 CONTENT 块。
    * ``*** Delete File: path`` —— 删除文件,无正文。

    SEARCH/REPLACE 块::

        <<<<<<< SEARCH
        old line(s) — 必须在文件中按行精确且唯一出现
        =======
        new line(s)
        >>>>>>> REPLACE

    CONTENT 块::

        <<<<<<< CONTENT
        file body — 原样写入,不需要 + 前缀
        >>>>>>> END

    不支持 ``diff --git`` / ``---`` / ``+++`` / ``@@`` 等 git diff 格式;
    不做 fuzzy matching;SEARCH 不唯一时报 AMBIGUOUS_MATCH,不存在时报
    SEARCH_NOT_FOUND——按错误信息扩大上下文或重读目标即可。

    Args:
        patch_text: v2 补丁文本。
        dry_run: 为 True 时只校验并返回规划摘要,不写入文件;为 False 时
            真实写入,并在失败时尝试回滚。
    """
    if dry_run:
        return _dry_run_patch_result(patch_text)
    return _apply_patch_result(patch_text)


__all__ = [
    "apply_patch",
    "parse_patch_text",
    "validate_patch_document",
    "validate_patch_text_size",
    "plan_patch_dry_run",
    "format_dry_run_result",
    "apply_patch_to_files",
    "format_apply_result",
    "apply_update_blocks",
    "build_updated_file_bytes",
    "PatchDocument",
    "PatchOperation",
    "SearchReplaceBlock",
    "ExistingFileContent",
    "ValidatedOperation",
    "ValidatedPatch",
    "LineStats",
    "UpdateApplyResult",
    "TargetSnapshot",
    "PlannedOperation",
    "PlannedPatch",
    "PatchError",
    "INVALID_PATCH",
    "INVALID_ENVELOPE",
    "MALFORMED_SECTION",
    "MALFORMED_BLOCK",
    "DUPLICATE_FILE_OPERATION",
    "INVALID_PATH",
    "SENSITIVE_PATH",
    "TARGET_IS_SYMLINK",
    "TARGET_TOO_LARGE",
    "PATCH_TOO_LARGE",
    "INVALID_UTF8",
    "BINARY_CONTENT",
    "TARGET_EXISTS",
    "TARGET_MISSING",
    "TARGET_IS_DIRECTORY",
    "FINAL_TOO_LARGE",
    "MIXED_NEWLINES",
    "SEARCH_NOT_FOUND",
    "AMBIGUOUS_MATCH",
    "BLOCK_OVERLAP",
    "EMPTY_SEARCH_NON_EMPTY_FILE",
    "NOOP_PATCH",
    "APPLY_FAILED",
    "CLEANUP_FAILED",
    "TARGET_CHANGED",
    "ROLLBACK_FAILED",
    "PATCH_TEXT_LIMIT_BYTES",
    "TARGET_FILE_LIMIT_BYTES",
    "ERROR_CODES",
    "UTF8_BOM",
    "BEGIN_MARKER",
    "END_MARKER",
    "SEARCH_OPEN",
    "SEARCH_DIVIDER",
    "REPLACE_CLOSE",
    "CONTENT_OPEN",
    "CONTENT_CLOSE",
]
