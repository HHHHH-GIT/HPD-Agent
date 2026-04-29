# pyright: reportUnknownVariableType=false
"""更安全的补丁工具骨架。

本模块定义更安全补丁工具的 LangChain 入口。v1 解析器、校验器、
dry-run 规划器与文件系统应用逻辑会在后续任务中逐步补齐。
"""

from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Literal

from langchain_core.tools import tool


BEGIN_MARKER = "*** Begin Patch"
END_MARKER = "*** End Patch"
EOF_MARKER = "\\ No newline at end of file"
UTF8_BOM = b"\xef\xbb\xbf"
PATCH_TEXT_LIMIT_BYTES = 256 * 1024
TARGET_FILE_LIMIT_BYTES = 1024 * 1024

NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
INVALID_PATCH = "INVALID_PATCH"
INVALID_ENVELOPE = "INVALID_ENVELOPE"
MALFORMED_SECTION = "MALFORMED_SECTION"
MALFORMED_HUNK = "MALFORMED_HUNK"
DUPLICATE_FILE_OPERATION = "DUPLICATE_FILE_OPERATION"
INVALID_PATH = "INVALID_PATH"
SENSITIVE_PATH = "SENSITIVE_PATH"
TARGET_IS_SYMLINK = "TARGET_IS_SYMLINK"
TARGET_TOO_LARGE = "TARGET_TOO_LARGE"
PATCH_TOO_LARGE = "PATCH_TOO_LARGE"
INVALID_UTF8 = "INVALID_UTF8"
TARGET_EXISTS = "TARGET_EXISTS"
TARGET_MISSING = "TARGET_MISSING"
TARGET_IS_DIRECTORY = "TARGET_IS_DIRECTORY"
FINAL_TOO_LARGE = "FINAL_TOO_LARGE"
MIXED_NEWLINES = "MIXED_NEWLINES"
HUNK_MISMATCH = "HUNK_MISMATCH"
HUNK_ORDER_ERROR = "HUNK_ORDER_ERROR"
INVALID_EOF_MARKER = "INVALID_EOF_MARKER"
NOOP_PATCH = "NOOP_PATCH"
UNSAFE_PATH = "UNSAFE_PATH"
APPLY_FAILED = "APPLY_FAILED"

ERROR_CODES = (
    NOT_IMPLEMENTED,
    INVALID_PATCH,
    INVALID_ENVELOPE,
    MALFORMED_SECTION,
    MALFORMED_HUNK,
    DUPLICATE_FILE_OPERATION,
    INVALID_PATH,
    SENSITIVE_PATH,
    TARGET_IS_SYMLINK,
    TARGET_TOO_LARGE,
    PATCH_TOO_LARGE,
    INVALID_UTF8,
    TARGET_EXISTS,
    TARGET_MISSING,
    TARGET_IS_DIRECTORY,
    FINAL_TOO_LARGE,
    MIXED_NEWLINES,
    HUNK_MISMATCH,
    HUNK_ORDER_ERROR,
    INVALID_EOF_MARKER,
    NOOP_PATCH,
    UNSAFE_PATH,
    APPLY_FAILED,
)

OperationKind = Literal["add", "update", "delete"]
HunkLineKind = Literal["context", "add", "delete"]

_SECTION_RE = re.compile(r"^\*\*\* (Add File|Update File|Delete File): (.+)$")
_HUNK_RE = re.compile(r"^@@ -([0-9]+),([0-9]+) \+([0-9]+),([0-9]+) @@$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_PATH_TOKEN_RE = re.compile(r"[._\-\s]+")
_SENSITIVE_PATH_TOKENS = frozenset(
    {
        "secret",
        "secrets",
        "token",
        "tokens",
        "key",
        "keys",
        "credential",
        "credentials",
    }
)


@dataclass(frozen=True)
class PatchHunkLine:
    kind: HunkLineKind
    content: str
    no_newline_at_end: bool = False


@dataclass(frozen=True)
class PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[PatchHunkLine, ...]


@dataclass(frozen=True)
class PatchOperation:
    kind: OperationKind
    path: str
    added_lines: tuple[PatchHunkLine, ...] = ()
    hunks: tuple[PatchHunk, ...] = ()


@dataclass(frozen=True)
class PatchDocument:
    operations: tuple[PatchOperation, ...]


@dataclass(frozen=True)
class ExistingFileContent:
    text: str
    byte_size: int
    has_utf8_bom: bool = False


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
class PlannedOperation:
    kind: OperationKind
    path: str
    final_size: int
    stats: LineStats


@dataclass(frozen=True)
class PlannedPatch:
    operations: tuple[PlannedOperation, ...]
    total_final_size: int


@dataclass(frozen=True)
class _FileLine:
    content: str
    has_newline: bool = True
    from_eof_marker: bool = False


@dataclass(frozen=True)
class _HunkPlan:
    start_index: int
    end_index: int
    output_lines: tuple[_FileLine, ...]
    added_lines: int
    deleted_lines: int


class PatchError(ValueError):
    code: str
    message: str
    line: int | None

    def __init__(self, code: str, message: str, *, line: int | None = None) -> None:
        self.code = code
        self.message = message
        self.line = line
        super().__init__(self.display_message)

    @property
    def display_message(self) -> str:
        if self.line is None:
            return self.message
        return f"{self.message} (line {self.line})"

    def to_error_result(self) -> str:
        return _format_error(self.code, self.display_message)


def _format_error(code: str, detail: str) -> str:
    return f"[Error] {code}: {detail}"


def parse_patch_text(patch_text: str) -> PatchDocument:
    lines = patch_text.splitlines()
    if not lines or lines[0] != BEGIN_MARKER:
        raise PatchError(
            INVALID_ENVELOPE, "patch must start with the begin marker", line=1
        )

    end_index = _find_end_marker(lines)
    for offset, trailing_line in enumerate(lines[end_index + 1 :], start=end_index + 2):
        if trailing_line.strip():
            raise PatchError(
                INVALID_ENVELOPE,
                "only whitespace is allowed after the end marker",
                line=offset,
            )

    body = lines[1:end_index]
    operations: list[PatchOperation] = []
    seen_paths: set[str] = set()
    index = 0

    while index < len(body):
        line = body[index]
        line_number = index + 2
        if _is_blank_separator(line):
            index += 1
            continue

        section_match = _SECTION_RE.match(line)
        if section_match is None:
            raise PatchError(
                MALFORMED_SECTION, "expected a file operation section", line=line_number
            )

        section_name = section_match.group(1)
        raw_path = section_match.group(2)
        if not raw_path.strip():
            raise PatchError(
                MALFORMED_SECTION, "section path must not be empty", line=line_number
            )
        if raw_path in seen_paths:
            raise PatchError(
                DUPLICATE_FILE_OPERATION,
                "patch contains repeated file operation path",
                line=line_number,
            )
        seen_paths.add(raw_path)

        index += 1
        if section_name == "Add File":
            operation, index = _parse_add_file(raw_path, body, index)
        elif section_name == "Update File":
            operation, index = _parse_update_file(raw_path, body, index)
        else:
            operation, index = _parse_delete_file(raw_path, body, index)
        operations.append(operation)

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


def validate_patch_document(
    document: PatchDocument,
    *,
    patch_text: str | None = None,
) -> ValidatedPatch:
    workspace_root = Path.cwd().resolve()
    patch_text_size = (
        validate_patch_text_size(patch_text) if patch_text is not None else None
    )
    validated_operations: list[ValidatedOperation] = []
    seen_targets: set[Path] = set()

    for operation in document.operations:
        relative_path = _validate_operation_path_syntax(operation.path)
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


def apply_update_hunks(
    existing_content: ExistingFileContent,
    hunks: tuple[PatchHunk, ...],
) -> UpdateApplyResult:
    if not hunks:
        raise PatchError(NOOP_PATCH, "更新补丁没有可应用的 hunk")

    original_lines, newline = _split_existing_lines(existing_content.text)
    plans = _validate_update_hunks(original_lines, hunks)
    output_lines = _build_output_lines(original_lines, plans)
    final_text = _serialize_lines(output_lines, newline)
    final_payload = final_text.encode("utf-8")
    final_bytes = (
        UTF8_BOM + final_payload if existing_content.has_utf8_bom else final_payload
    )
    original_payload = existing_content.text.encode("utf-8")
    original_bytes = (
        UTF8_BOM + original_payload
        if existing_content.has_utf8_bom
        else original_payload
    )

    if final_bytes == original_bytes:
        raise PatchError(NOOP_PATCH, "更新补丁不会改变目标内容")

    return UpdateApplyResult(
        final_bytes=final_bytes,
        stats=LineStats(
            old_line_count=len(original_lines),
            new_line_count=len(output_lines),
            added_lines=sum(plan.added_lines for plan in plans),
            deleted_lines=sum(plan.deleted_lines for plan in plans),
        ),
        newline=newline,
        has_utf8_bom=existing_content.has_utf8_bom,
    )


def build_updated_file_bytes(
    existing_content: ExistingFileContent, hunks: tuple[PatchHunk, ...]
) -> bytes:
    return apply_update_hunks(existing_content, hunks).final_bytes


def _split_existing_lines(text: str) -> tuple[tuple[_FileLine, ...], str]:
    newline = _detect_newline_style(text)
    if text == "":
        return (), newline

    parts = text.split(newline)
    if text.endswith(newline):
        return (
            tuple(_FileLine(content=part, has_newline=True) for part in parts[:-1]),
            newline,
        )

    lines = [_FileLine(content=part, has_newline=True) for part in parts[:-1]]
    lines.append(_FileLine(content=parts[-1], has_newline=False))
    return tuple(lines), newline


def _detect_newline_style(text: str) -> str:
    crlf_count = 0
    lf_count = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\r":
            if index + 1 < len(text) and text[index + 1] == "\n":
                crlf_count += 1
                index += 2
                continue
            raise PatchError(MIXED_NEWLINES, "目标文件包含不支持的混合换行")
        if char == "\n":
            lf_count += 1
        index += 1

    if crlf_count and lf_count:
        raise PatchError(MIXED_NEWLINES, "目标文件包含 LF 与 CRLF 混合换行")
    return "\r\n" if crlf_count else "\n"


def _validate_update_hunks(
    original_lines: tuple[_FileLine, ...],
    hunks: tuple[PatchHunk, ...],
) -> tuple[_HunkPlan, ...]:
    plans: list[_HunkPlan] = []
    previous_end_index = 0
    cumulative_delta = 0

    for hunk in hunks:
        _validate_hunk_accounting(hunk)
        start_index, end_index = _hunk_original_span(hunk, len(original_lines))
        if start_index < previous_end_index:
            raise PatchError(HUNK_ORDER_ERROR, "hunk 必须按原文件位置排序且不能重叠")

        expected_new_start = hunk.old_start + cumulative_delta
        if hunk.old_count == 0:
            if hunk.old_start == 0 and cumulative_delta != 0:
                raise PatchError(
                    HUNK_ORDER_ERROR, "文件开头插入只能出现在首个未偏移位置"
                )
            expected_new_start = hunk.old_start + cumulative_delta + 1
        if hunk.new_start != expected_new_start:
            raise PatchError(HUNK_ORDER_ERROR, "hunk 的 new_start 与累计行偏移不一致")

        plan = _validate_hunk_against_original(
            original_lines, hunk, start_index, end_index
        )
        plans.append(plan)
        previous_end_index = max(previous_end_index, end_index)
        cumulative_delta += hunk.new_count - hunk.old_count

    return tuple(plans)


def _validate_hunk_accounting(hunk: PatchHunk) -> None:
    old_side_count = sum(1 for line in hunk.lines if line.kind != "add")
    new_side_count = sum(1 for line in hunk.lines if line.kind != "delete")
    if old_side_count != hunk.old_count or new_side_count != hunk.new_count:
        raise PatchError(HUNK_MISMATCH, "hunk 行数与头部计数不一致")
    if hunk.old_count == 0 and any(line.kind != "add" for line in hunk.lines):
        raise PatchError(HUNK_MISMATCH, "零长度插入 hunk 只能包含新增行")


def _hunk_original_span(hunk: PatchHunk, original_line_count: int) -> tuple[int, int]:
    if hunk.old_count == 0:
        if hunk.old_start < 0 or hunk.old_start > original_line_count:
            raise PatchError(HUNK_MISMATCH, "插入位置超出原文件行数")
        return hunk.old_start, hunk.old_start

    if hunk.old_start < 1:
        raise PatchError(HUNK_MISMATCH, "非插入 hunk 的 old_start 必须从 1 开始")
    start_index = hunk.old_start - 1
    end_index = start_index + hunk.old_count
    if end_index > original_line_count:
        raise PatchError(HUNK_MISMATCH, "hunk 指向的原文件范围不存在")
    return start_index, end_index


def _validate_hunk_against_original(
    original_lines: tuple[_FileLine, ...],
    hunk: PatchHunk,
    start_index: int,
    end_index: int,
) -> _HunkPlan:
    old_index = start_index
    output_lines: list[_FileLine] = []
    added_lines = 0
    deleted_lines = 0

    for hunk_line in hunk.lines:
        if hunk_line.kind != "add":
            if old_index >= end_index:
                raise PatchError(HUNK_MISMATCH, "hunk 旧侧行数超过声明范围")
            original_line = original_lines[old_index]
            if original_line.content != hunk_line.content:
                raise PatchError(HUNK_MISMATCH, "hunk 旧侧内容与目标文件不匹配")
            _validate_old_eof_marker(original_lines, old_index, hunk_line)
            old_index += 1

        if hunk_line.kind != "delete":
            output_lines.append(
                _FileLine(
                    content=hunk_line.content,
                    has_newline=not hunk_line.no_newline_at_end,
                    from_eof_marker=hunk_line.no_newline_at_end,
                )
            )
            if hunk_line.kind == "add":
                added_lines += 1
        else:
            deleted_lines += 1

    if old_index != end_index:
        raise PatchError(HUNK_MISMATCH, "hunk 旧侧行数少于声明范围")

    return _HunkPlan(
        start_index=start_index,
        end_index=end_index,
        output_lines=tuple(output_lines),
        added_lines=added_lines,
        deleted_lines=deleted_lines,
    )


def _validate_old_eof_marker(
    original_lines: tuple[_FileLine, ...],
    old_index: int,
    hunk_line: PatchHunkLine,
) -> None:
    original_line = original_lines[old_index]
    is_final_original_line = old_index == len(original_lines) - 1
    if hunk_line.no_newline_at_end:
        if original_line.has_newline or not is_final_original_line:
            raise PatchError(
                INVALID_EOF_MARKER, "EOF 标记必须匹配原文件最后一个无换行行"
            )
        return
    if not original_line.has_newline:
        raise PatchError(
            INVALID_EOF_MARKER, "原文件无末尾换行时 hunk 必须显式 EOF 标记"
        )


def _build_output_lines(
    original_lines: tuple[_FileLine, ...],
    plans: tuple[_HunkPlan, ...],
) -> tuple[_FileLine, ...]:
    output_lines: list[_FileLine] = []
    cursor = 0
    for plan in plans:
        output_lines.extend(original_lines[cursor : plan.start_index])
        output_lines.extend(plan.output_lines)
        cursor = plan.end_index
    output_lines.extend(original_lines[cursor:])
    _validate_output_eof_markers(output_lines)
    return tuple(output_lines)


def _validate_output_eof_markers(output_lines: list[_FileLine]) -> None:
    for index, line in enumerate(output_lines):
        is_final_line = index == len(output_lines) - 1
        if not line.has_newline and not is_final_line:
            raise PatchError(INVALID_EOF_MARKER, "无换行行只能出现在输出末尾")
        if line.from_eof_marker and not is_final_line:
            raise PatchError(INVALID_EOF_MARKER, "EOF 标记只能作用于输出末尾行")


def _serialize_lines(lines: tuple[_FileLine, ...], newline: str) -> str:
    chunks: list[str] = []
    for line in lines:
        chunks.append(line.content)
        if line.has_newline:
            chunks.append(newline)
    return "".join(chunks)


def plan_patch_dry_run(patch_text: str) -> PlannedPatch:
    document = parse_patch_text(patch_text)
    validated_patch = validate_patch_document(document, patch_text=patch_text)
    add_newline = _detect_patch_newline_for_add(patch_text)
    planned_operations: list[PlannedOperation] = []

    for validated_operation in validated_patch.operations:
        try:
            planned_operations.append(_plan_operation(validated_operation, add_newline))
        except PatchError as exc:
            raise PatchError(
                exc.code,
                f"{validated_operation.relative_path.as_posix()}: {exc.display_message}",
            ) from exc

    return PlannedPatch(
        operations=tuple(planned_operations),
        total_final_size=sum(operation.final_size for operation in planned_operations),
    )


def format_dry_run_result(planned_patch: PlannedPatch) -> str:
    add_count = sum(
        1 for operation in planned_patch.operations if operation.kind == "add"
    )
    update_count = sum(
        1 for operation in planned_patch.operations if operation.kind == "update"
    )
    delete_count = sum(
        1 for operation in planned_patch.operations if operation.kind == "delete"
    )
    lines = [
        (
            "[DRY-RUN OK] 补丁验证通过，计划 "
            + f"{len(planned_patch.operations)} 个文件；Add {add_count}、"
            + f"Update {update_count}、Delete {delete_count}。未写入文件。"
        ),
        "验证: 路径、安全、大小、UTF-8 与 hunk 检查已通过。",
    ]
    labels = {"add": "Add", "update": "Update", "delete": "Delete"}
    for operation in planned_patch.operations:
        lines.append(
            f"- {labels[operation.kind]}: {operation.path} "
            + f"(+{operation.stats.added_lines}/-{operation.stats.deleted_lines}, "
            + f"final {operation.final_size} bytes)"
        )
    return "\n".join(lines)


def _dry_run_patch_result(patch_text: str) -> str:
    try:
        return format_dry_run_result(plan_patch_dry_run(patch_text))
    except PatchError as exc:
        return _format_patch_error(exc)


def _plan_operation(
    validated_operation: ValidatedOperation, add_newline: str
) -> PlannedOperation:
    operation = validated_operation.operation
    if operation.kind == "add":
        return _plan_add_operation(validated_operation, add_newline)
    if operation.kind == "update":
        return _plan_update_operation(validated_operation)
    return _plan_delete_operation(validated_operation)


def _plan_add_operation(
    validated_operation: ValidatedOperation, add_newline: str
) -> PlannedOperation:
    if validated_operation.target_path.exists():
        raise PatchError(TARGET_EXISTS, "新增目标已经存在；请改用 Update File")
    final_bytes = _build_add_file_bytes(
        validated_operation.operation.added_lines, add_newline
    )
    _ensure_final_size(final_bytes)
    line_count = len(validated_operation.operation.added_lines)
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
    )


def _plan_update_operation(validated_operation: ValidatedOperation) -> PlannedOperation:
    existing_content = validated_operation.existing_content
    if existing_content is None:
        raise PatchError(TARGET_MISSING, "更新目标缺少可读取的现有内容")
    result = apply_update_hunks(existing_content, validated_operation.operation.hunks)
    _ensure_final_size(result.final_bytes)
    return PlannedOperation(
        kind="update",
        path=validated_operation.relative_path.as_posix(),
        final_size=len(result.final_bytes),
        stats=result.stats,
    )


def _plan_delete_operation(validated_operation: ValidatedOperation) -> PlannedOperation:
    target_stat = validated_operation.target_path.stat()
    existing_content = _read_existing_utf8(
        validated_operation.target_path, target_stat.st_size
    )
    original_lines, _newline = _split_existing_lines(existing_content.text)
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
    )


def _build_add_file_bytes(
    added_lines: tuple[PatchHunkLine, ...], newline: str
) -> bytes:
    if not added_lines:
        return b""
    text = "".join(line.content + newline for line in added_lines)
    return text.encode("utf-8")


def _detect_patch_newline_for_add(patch_text: str) -> str:
    crlf_count = patch_text.count("\r\n")
    lf_only_count = patch_text.count("\n") - crlf_count
    return "\r\n" if crlf_count and not lf_only_count else "\n"


def _ensure_final_size(final_bytes: bytes) -> None:
    if len(final_bytes) > TARGET_FILE_LIMIT_BYTES:
        raise PatchError(FINAL_TOO_LARGE, "计划输出超过 1 MiB 限制")


def _format_patch_error(error: PatchError) -> str:
    return f"{error.to_error_result()} 上下文: dry-run 阶段。提示: {_error_hint(error.code)}"


def _error_hint(code: str) -> str:
    hints = {
        TARGET_EXISTS: "如果要修改现有文件，请使用 Update File。",
        TARGET_MISSING: "请确认目标文件存在，或改用 Add File。",
        TARGET_IS_DIRECTORY: "请指定普通文件路径，不能指定目录。",
        FINAL_TOO_LARGE: "请拆分补丁或缩小目标文件内容。",
        HUNK_MISMATCH: "请根据目标文件当前内容重新生成 hunk。",
        HUNK_ORDER_ERROR: "请按原文件行号排序 hunk，且不要重叠。",
        INVALID_PATH: "请使用工作区内的安全相对路径。",
        SENSITIVE_PATH: "请不要通过补丁写入密钥、令牌或环境配置。",
    }
    return hints.get(code, "请检查补丁格式、路径和目标文件状态后重试。")


def _validate_operation_path_syntax(raw_path: str) -> Path:
    if not raw_path or not raw_path.strip():
        raise PatchError(INVALID_PATH, "路径不能为空")
    if raw_path != raw_path.strip():
        raise PatchError(INVALID_PATH, "路径不能包含首尾空白")
    if _CONTROL_CHAR_RE.search(raw_path):
        raise PatchError(INVALID_PATH, "路径不能包含控制字符")
    if raw_path.startswith("~"):
        raise PatchError(INVALID_PATH, "路径不能使用用户目录缩写")
    if "\\" in raw_path:
        raise PatchError(INVALID_PATH, "路径必须使用正斜杠分隔")
    if raw_path.endswith("/"):
        raise PatchError(INVALID_PATH, "路径不能以目录分隔符结尾")

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
        raise PatchError(TARGET_MISSING, "更新或删除目标必须存在") from None

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
    return None


def _read_existing_utf8(target_path: Path, byte_size: int) -> ExistingFileContent:
    existing_bytes = target_path.read_bytes()
    has_utf8_bom = existing_bytes.startswith(UTF8_BOM)
    payload = existing_bytes[len(UTF8_BOM) :] if has_utf8_bom else existing_bytes
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError(INVALID_UTF8, "目标文件不是有效的 UTF-8 文本") from exc
    return ExistingFileContent(
        text=text, byte_size=byte_size, has_utf8_bom=has_utf8_bom
    )


def _find_end_marker(lines: list[str]) -> int:
    for index, line in enumerate(lines[1:], start=1):
        if line == END_MARKER:
            return index
    raise PatchError(INVALID_ENVELOPE, "patch is missing the end marker")


def _parse_add_file(
    path: str,
    lines: list[str],
    start_index: int,
) -> tuple[PatchOperation, int]:
    end_index = _find_next_section(lines, start_index)
    body_start, body_end = _trim_blank_separators(lines, start_index, end_index)
    added_lines: list[PatchHunkLine] = []

    for index in range(body_start, body_end):
        line = lines[index]
        if _is_blank_separator(line) or not line.startswith("+"):
            raise PatchError(
                MALFORMED_SECTION,
                "Add File body lines must start with '+'",
                line=index + 2,
            )
        added_lines.append(PatchHunkLine(kind="add", content=line[1:]))

    return (
        PatchOperation(kind="add", path=path, added_lines=tuple(added_lines)),
        end_index,
    )


def _parse_update_file(
    path: str,
    lines: list[str],
    start_index: int,
) -> tuple[PatchOperation, int]:
    end_index = _find_next_section(lines, start_index)
    body_start, body_end = _trim_blank_separators(lines, start_index, end_index)
    if body_start == body_end:
        raise PatchError(MALFORMED_SECTION, "Update File requires at least one hunk")

    hunks: list[PatchHunk] = []
    index = body_start
    while index < body_end:
        line = lines[index]
        if _is_blank_separator(line):
            index += 1
            continue
        if _HUNK_RE.match(line) is None:
            raise PatchError(
                MALFORMED_HUNK, "expected a well-formed hunk header", line=index + 2
            )
        hunk, index = _parse_hunk(lines, index, body_end)
        hunks.append(hunk)

    if not hunks:
        raise PatchError(MALFORMED_SECTION, "Update File requires at least one hunk")

    return PatchOperation(kind="update", path=path, hunks=tuple(hunks)), end_index


def _parse_delete_file(
    path: str,
    lines: list[str],
    start_index: int,
) -> tuple[PatchOperation, int]:
    end_index = _find_next_section(lines, start_index)
    body_start, body_end = _trim_blank_separators(lines, start_index, end_index)
    if body_start != body_end:
        raise PatchError(
            MALFORMED_SECTION,
            "Delete File section does not accept a body",
            line=body_start + 2,
        )
    return PatchOperation(kind="delete", path=path), end_index


def _parse_hunk(
    lines: list[str], header_index: int, section_end: int
) -> tuple[PatchHunk, int]:
    header = lines[header_index]
    match = _HUNK_RE.match(header)
    if match is None:
        raise PatchError(MALFORMED_HUNK, "malformed hunk header", line=header_index + 2)

    old_start = int(match.group(1))
    old_count = int(match.group(2))
    new_start = int(match.group(3))
    new_count = int(match.group(4))
    old_seen = 0
    new_seen = 0
    hunk_lines: list[PatchHunkLine] = []
    last_body_line_index: int | None = None
    index = header_index + 1

    while index < section_end:
        line = lines[index]
        if line == EOF_MARKER:
            if not hunk_lines or last_body_line_index != index - 1:
                raise PatchError(
                    MALFORMED_HUNK,
                    "EOF marker must immediately follow a hunk body line",
                    line=index + 2,
                )
            hunk_lines[-1] = replace(hunk_lines[-1], no_newline_at_end=True)
            index += 1
            continue

        if old_seen == old_count and new_seen == new_count:
            break

        if not line or line[0] not in " +-":
            raise PatchError(
                MALFORMED_HUNK,
                "hunk body lines must start with space, '-', or '+'",
                line=index + 2,
            )

        prefix = line[0]
        content = line[1:]
        if prefix == " ":
            old_seen += 1
            new_seen += 1
            kind: HunkLineKind = "context"
        elif prefix == "-":
            old_seen += 1
            kind = "delete"
        else:
            new_seen += 1
            kind = "add"

        if old_seen > old_count or new_seen > new_count:
            raise PatchError(
                MALFORMED_HUNK, "hunk body exceeds header counts", line=index + 2
            )

        hunk_lines.append(PatchHunkLine(kind=kind, content=content))
        last_body_line_index = index
        index += 1

    if old_seen != old_count or new_seen != new_count:
        raise PatchError(
            MALFORMED_HUNK,
            "hunk body does not match header counts",
            line=header_index + 2,
        )

    return (
        PatchHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            lines=tuple(hunk_lines),
        ),
        index,
    )


def _find_next_section(lines: list[str], start_index: int) -> int:
    index = start_index
    while index < len(lines):
        if lines[index].startswith("*** "):
            return index
        index += 1
    return index


def _trim_blank_separators(
    lines: list[str], start_index: int, end_index: int
) -> tuple[int, int]:
    while start_index < end_index and _is_blank_separator(lines[start_index]):
        start_index += 1
    while end_index > start_index and _is_blank_separator(lines[end_index - 1]):
        end_index -= 1
    return start_index, end_index


def _is_blank_separator(line: str) -> bool:
    return not line.strip()


def _placeholder_result(patch_text: str, dry_run: bool) -> str:
    mode = "dry-run" if dry_run else "apply"
    detail = (
        f"apply_patch 的 {mode} 模式尚未实现；"
        + f"已收到 {len(patch_text)} 个字符的补丁文本。"
    )
    return _format_error(NOT_IMPLEMENTED, detail)


@tool
def apply_patch(patch_text: str, dry_run: bool = False) -> str:
    """安全地将补丁应用到仓库文件系统。

    Args:
        patch_text: 后续实现中需要校验并应用的补丁文本。
        dry_run: 为 True 时，后续版本只报告计划变更而不写入文件。
    """
    if dry_run:
        return _dry_run_patch_result(patch_text)
    return _placeholder_result(patch_text=patch_text, dry_run=dry_run)


__all__ = [
    "apply_patch",
    "parse_patch_text",
    "validate_patch_document",
    "validate_patch_text_size",
    "plan_patch_dry_run",
    "format_dry_run_result",
    "apply_update_hunks",
    "build_updated_file_bytes",
    "PatchDocument",
    "PatchOperation",
    "PatchHunk",
    "PatchHunkLine",
    "ExistingFileContent",
    "ValidatedOperation",
    "ValidatedPatch",
    "LineStats",
    "UpdateApplyResult",
    "PlannedOperation",
    "PlannedPatch",
    "PatchError",
    "NOT_IMPLEMENTED",
    "INVALID_PATCH",
    "INVALID_ENVELOPE",
    "MALFORMED_SECTION",
    "MALFORMED_HUNK",
    "DUPLICATE_FILE_OPERATION",
    "INVALID_PATH",
    "SENSITIVE_PATH",
    "TARGET_IS_SYMLINK",
    "TARGET_TOO_LARGE",
    "PATCH_TOO_LARGE",
    "INVALID_UTF8",
    "TARGET_EXISTS",
    "TARGET_MISSING",
    "TARGET_IS_DIRECTORY",
    "FINAL_TOO_LARGE",
    "MIXED_NEWLINES",
    "HUNK_MISMATCH",
    "HUNK_ORDER_ERROR",
    "INVALID_EOF_MARKER",
    "NOOP_PATCH",
    "UNSAFE_PATH",
    "APPLY_FAILED",
    "PATCH_TEXT_LIMIT_BYTES",
    "TARGET_FILE_LIMIT_BYTES",
    "ERROR_CODES",
]
