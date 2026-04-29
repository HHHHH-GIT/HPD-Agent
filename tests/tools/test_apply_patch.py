# pyright: reportUnknownMemberType=false
import importlib
import inspect
import os
from pathlib import Path
import stat
import threading
from typing import Callable, Protocol, cast

import pytest

from src.tools.apply_patch import (
    DUPLICATE_FILE_OPERATION,
    FINAL_TOO_LARGE,
    HUNK_MISMATCH,
    HUNK_ORDER_ERROR,
    INVALID_ENVELOPE,
    INVALID_EOF_MARKER,
    INVALID_PATH,
    INVALID_UTF8,
    MALFORMED_HUNK,
    MALFORMED_SECTION,
    MIXED_NEWLINES,
    NOOP_PATCH,
    PATCH_TEXT_LIMIT_BYTES,
    PATCH_TOO_LARGE,
    SENSITIVE_PATH,
    TARGET_EXISTS,
    TARGET_FILE_LIMIT_BYTES,
    TARGET_IS_DIRECTORY,
    TARGET_IS_SYMLINK,
    TARGET_MISSING,
    TARGET_TOO_LARGE,
    TARGET_CHANGED,
    APPLY_FAILED,
    BINARY_CONTENT,
    CLEANUP_FAILED,
    ERROR_CODES,
    UTF8_BOM,
    ExistingFileContent,
    PatchDocument,
    PatchError,
    PatchOperation,
    PlannedOperation,
    PlannedPatch,
    apply_patch,
    apply_update_hunks,
    parse_patch_text,
    validate_patch_document,
)


class _NamedTool(Protocol):
    name: str


apply_patch_module = importlib.import_module("src.tools.apply_patch")


def _patch(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _assert_parser_error(patch_text: str, code: str) -> None:
    with pytest.raises(PatchError) as exc_info:
        _ = parse_patch_text(patch_text)

    error = exc_info.value
    assert error.code == code
    assert error.to_error_result().startswith(f"[Error] {code}:")


def _assert_validation_error(document: PatchDocument, code: str) -> None:
    with pytest.raises(PatchError) as exc_info:
        _ = validate_patch_document(document)

    error = exc_info.value
    assert error.code == code
    assert error.to_error_result().startswith(f"[Error] {code}:")


def _add_document(path: str) -> PatchDocument:
    return PatchDocument(operations=(PatchOperation(kind="add", path=path),))


def _update_document(path: str) -> PatchDocument:
    return parse_patch_text(
        _patch(
            "*** Begin Patch",
            f"*** Update File: {path}",
            "@@ -0,0 +0,0 @@",
            "*** End Patch",
        )
    )


def test_path_allows_gitignore_and_github_and_env_example(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    document = parse_patch_text(
        _patch(
            "*** Begin Patch",
            "*** Add File: .gitignore",
            "+*.pyc",
            "*** Add File: .github/workflows/test.yml",
            "+name: test",
            "*** Add File: .env.example",
            "+EXAMPLE=1",
            "*** Add File: monkey.py",
            "+print('ok')",
            "*** End Patch",
        )
    )

    validated = validate_patch_document(document)

    assert [
        operation.relative_path.as_posix() for operation in validated.operations
    ] == [
        ".gitignore",
        ".github/workflows/test.yml",
        ".env.example",
        "monkey.py",
    ]


def test_relative_windows_separators_normalize_to_posix_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    document = parse_patch_text(
        _patch(
            "*** Begin Patch",
            "*** Add File: dir\\file.txt",
            "+created",
            "*** End Patch",
        )
    )

    validated = validate_patch_document(document)
    result = cast(
        str,
        apply_patch.invoke(
            {
                "patch_text": _patch(
                    "*** Begin Patch",
                    "*** Add File: nested\\child.txt",
                    "+created",
                    "*** End Patch",
                ),
                "dry_run": True,
            }
        ),
    )

    assert validated.operations[0].relative_path.as_posix() == "dir/file.txt"
    assert "nested/child.txt" in result


@pytest.mark.parametrize(
    "raw_path",
    [
        "C:/outside.txt",
        "C:\\outside.txt",
        "C:relative.txt",
        "C:dir\\file.txt",
    ],
)
def test_windows_drive_paths_reject_unsafe_or_non_windows_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_path: str,
):
    monkeypatch.chdir(tmp_path)

    _assert_validation_error(_add_document(raw_path), INVALID_PATH)


def test_windows_drive_path_under_workspace_normalizes_when_windows_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)

    def fake_drive_relative(raw_path: str, workspace_root: Path) -> Path:
        assert raw_path == "C:\\workspace\\dir\\file.txt"
        assert workspace_root == tmp_path.resolve()
        return Path("dir\\file.txt")

    monkeypatch.setattr(apply_patch_module, "_is_windows_runtime", lambda: True)
    monkeypatch.setattr(
        apply_patch_module,
        "_windows_drive_absolute_to_relative",
        fake_drive_relative,
    )
    document = parse_patch_text(
        _patch(
            "*** Begin Patch",
            "*** Add File: C:\\workspace\\dir\\file.txt",
            "+created",
            "*** End Patch",
        )
    )

    validated = validate_patch_document(document)

    assert validated.operations[0].relative_path.as_posix() == "dir/file.txt"


@pytest.mark.parametrize(
    "raw_path",
    [
        "/absolute.txt",
        "../escape.txt",
        "safe/../escape.txt",
        "~/secret.txt",
        "dir/",
        "bad\x00name.txt",
        "bad\x1fname.txt",
        ".git/config",
        "",
        "   ",
    ],
)
def test_rejects_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_path: str,
):
    monkeypatch.chdir(tmp_path)

    _assert_validation_error(_add_document(raw_path), INVALID_PATH)


@pytest.mark.parametrize(
    "raw_path",
    [
        ".env",
        ".env.local",
        ".env.example.local",
        ".envrc",
        ".env/config",
        "api_key.py",
        "service-token.txt",
        "credentials/config.json",
    ],
)
def test_rejects_sensitive_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_path: str,
):
    monkeypatch.chdir(tmp_path)

    _assert_validation_error(_add_document(raw_path), SENSITIVE_PATH)


def test_rejects_symlink_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (tmp_path / "link").symlink_to(real_dir, target_is_directory=True)

    _assert_validation_error(_add_document("link/new.txt"), TARGET_IS_SYMLINK)


def test_rejects_symlink_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    real_file = tmp_path / "real.txt"
    _ = real_file.write_text("ok", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(real_file)

    _assert_validation_error(_update_document("link.txt"), TARGET_IS_SYMLINK)


def test_rejects_canonical_duplicate_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    document = parse_patch_text(
        _patch(
            "*** Begin Patch",
            "*** Add File: a.txt",
            "+content",
            "*** Delete File: ./a.txt",
            "*** End Patch",
        )
    )

    _assert_validation_error(document, DUPLICATE_FILE_OPERATION)


def test_rejects_patch_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    document = _add_document("safe.txt")
    oversized_patch = "x" * (PATCH_TEXT_LIMIT_BYTES + 1)

    with pytest.raises(PatchError) as exc_info:
        _ = validate_patch_document(document, patch_text=oversized_patch)

    assert exc_info.value.code == PATCH_TOO_LARGE


def test_parse_rejects_oversized_patch_before_envelope_scan():
    _assert_parser_error("x" * (PATCH_TEXT_LIMIT_BYTES + 1), PATCH_TOO_LARGE)


def test_rejects_existing_target_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "large.txt"
    _ = target.write_bytes(b"x" * (TARGET_FILE_LIMIT_BYTES + 1))

    _assert_validation_error(_update_document("large.txt"), TARGET_TOO_LARGE)


def test_rejects_invalid_utf8_update_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "bad.txt"
    _ = target.write_bytes(b"\xff")

    _assert_validation_error(_update_document("bad.txt"), INVALID_UTF8)


def test_rejects_valid_utf8_binary_like_update_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "binary.txt"
    _ = target.write_bytes(b"old\x00\n")

    _assert_validation_error(_update_document("binary.txt"), BINARY_CONTENT)


def test_dry_run_rejects_binary_like_final_add_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: binary.txt",
        "+safe\x00binary",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith(f"[Error] {BINARY_CONTENT}:")
    assert not (tmp_path / "binary.txt").exists()


def test_dry_run_rejects_binary_like_final_update_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "target.txt"
    _ = target.write_text("old\n", encoding="utf-8")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: target.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new\x01",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith(f"[Error] {BINARY_CONTENT}:")
    assert target.read_text(encoding="utf-8") == "old\n"


def test_binary_and_cleanup_codes_are_exported():
    assert BINARY_CONTENT in ERROR_CODES
    assert CLEANUP_FAILED in ERROR_CODES


def test_update_validation_preserves_utf8_bom_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "bom.txt"
    _ = target.write_bytes(b"\xef\xbb\xbfhello")

    validated = validate_patch_document(_update_document("bom.txt"))

    existing_content = validated.operations[0].existing_content
    assert existing_content is not None
    assert existing_content.has_utf8_bom is True
    assert existing_content.text == "hello"
    assert existing_content.byte_size == 8


def test_rejects_missing_update_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    _assert_validation_error(_update_document("missing.txt"), TARGET_MISSING)


def test_rejects_directory_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dir.txt").mkdir()

    _assert_validation_error(_update_document("dir.txt"), TARGET_IS_DIRECTORY)


def _update_hunks(*lines: str):
    document = parse_patch_text(
        _patch(
            "*** Begin Patch",
            "*** Update File: file.txt",
            *lines,
            "*** End Patch",
        )
    )
    return document.operations[0].hunks


def _existing_content(raw_bytes: bytes) -> ExistingFileContent:
    has_bom = raw_bytes.startswith(UTF8_BOM)
    payload = raw_bytes[len(UTF8_BOM) :] if has_bom else raw_bytes
    return ExistingFileContent(
        text=payload.decode("utf-8"),
        byte_size=len(raw_bytes),
        has_utf8_bom=has_bom,
    )


def _assert_apply_error(raw_bytes: bytes, code: str, *hunk_lines: str) -> None:
    with pytest.raises(PatchError) as exc_info:
        _ = apply_update_hunks(_existing_content(raw_bytes), _update_hunks(*hunk_lines))

    error = exc_info.value
    assert error.code == code
    assert error.to_error_result().startswith(f"[Error] {code}:")


def test_update_preserves_crlf_and_applies_exact_hunk():
    result = apply_update_hunks(
        _existing_content(b"one\r\ntwo\r\nthree\r\n"),
        _update_hunks(
            "@@ -1,3 +1,4 @@",
            " one",
            "-two",
            "+TWO",
            "+inserted",
            " three",
        ),
    )

    assert result.final_bytes == b"one\r\nTWO\r\ninserted\r\nthree\r\n"
    assert result.newline == "\r\n"
    assert result.stats.old_line_count == 3
    assert result.stats.new_line_count == 4
    assert result.stats.added_lines == 2
    assert result.stats.deleted_lines == 1


def test_hunk_apply_replaces_exact_lf_bytes():
    result = apply_update_hunks(
        _existing_content(b"alpha\nbeta\ngamma\n"),
        _update_hunks(
            "@@ -2,1 +2,1 @@",
            "-beta",
            "+BETA",
        ),
    )

    assert result.final_bytes == b"alpha\nBETA\ngamma\n"
    assert result.newline == "\n"


def test_insertion_before_first_line_and_append():
    result = apply_update_hunks(
        _existing_content(b"middle\n"),
        _update_hunks(
            "@@ -0,0 +1,1 @@",
            "+start",
            "@@ -1,0 +3,1 @@",
            "+end",
        ),
    )

    assert result.final_bytes == b"start\nmiddle\nend\n"
    assert result.stats.added_lines == 2
    assert result.stats.deleted_lines == 0


def test_newline_rejects_mixed_newlines():
    _assert_apply_error(
        b"one\ntwo\r\n",
        MIXED_NEWLINES,
        "@@ -1,1 +1,1 @@",
        "-one",
        "+ONE",
    )


def test_eof_marker_controls_no_final_newline_bytes():
    result = apply_update_hunks(
        _existing_content(b"one\ntwo"),
        _update_hunks(
            "@@ -2,1 +2,1 @@",
            "-two",
            "\\ No newline at end of file",
            "+TWO",
            "\\ No newline at end of file",
        ),
    )

    assert result.final_bytes == b"one\nTWO"
    assert result.stats.new_line_count == 2


def test_hunk_apply_preserves_utf8_bom():
    result = apply_update_hunks(
        _existing_content(UTF8_BOM + b"one\n"),
        _update_hunks(
            "@@ -1,1 +1,1 @@",
            "-one",
            "+uno",
        ),
    )

    assert result.final_bytes == UTF8_BOM + b"uno\n"
    assert result.has_utf8_bom is True


def test_hunk_mismatch_rejects_non_matching_old_slice():
    _assert_apply_error(
        b"one\n",
        HUNK_MISMATCH,
        "@@ -1,1 +1,1 @@",
        "-two",
        "+TWO",
    )


def test_invalid_eof_marker_rejects_non_final_output_line():
    _assert_apply_error(
        b"one\n",
        INVALID_EOF_MARKER,
        "@@ -1,1 +1,2 @@",
        "+zero",
        "\\ No newline at end of file",
        " one",
    )


def test_invalid_eof_marker_requires_old_final_marker():
    _assert_apply_error(
        b"one",
        INVALID_EOF_MARKER,
        "@@ -1,1 +1,1 @@",
        "-one",
        "+ONE",
    )


@pytest.mark.parametrize(
    "hunk_lines",
    [
        (
            "@@ -2,1 +2,1 @@",
            "-b",
            "+B",
            "@@ -1,1 +1,1 @@",
            "-a",
            "+A",
        ),
        (
            "@@ -1,2 +1,2 @@",
            " a",
            "-b",
            "+B",
            "@@ -2,1 +2,1 @@",
            "-b",
            "+bee",
        ),
    ],
)
def test_hunk_apply_rejects_order_and_overlap_errors(hunk_lines: tuple[str, ...]):
    _assert_apply_error(b"a\nb\n", HUNK_ORDER_ERROR, *hunk_lines)


def test_hunk_apply_rejects_noop_update():
    _assert_apply_error(
        b"same\n",
        NOOP_PATCH,
        "@@ -1,1 +1,1 @@",
        " same",
    )


def test_dry_run_validates_without_mutating_or_creating_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "existing.txt"
    old_file = tmp_path / "old.txt"
    _ = existing.write_text("old\n", encoding="utf-8")
    _ = old_file.write_text("remove\n", encoding="utf-8")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: nested/new.txt",
        "+created",
        "*** Update File: existing.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new",
        "*** Delete File: old.txt",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith("[DRY-RUN OK]")
    assert "Add 1" in result
    assert "Update 1" in result
    assert "Delete 1" in result
    assert "nested/new.txt" in result
    assert "+1/-0" in result
    assert "+1/-1" in result
    assert "+0/-1" in result
    assert existing.read_text(encoding="utf-8") == "old\n"
    assert old_file.read_text(encoding="utf-8") == "remove\n"
    assert not (tmp_path / "nested").exists()


def test_dry_run_update_result_format_preserves_target_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "file.txt"
    _ = target.write_bytes(b"one\r\ntwo\r\n")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: file.txt",
        "@@ -2,1 +2,1 @@",
        "-two",
        "+TWO",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith("[DRY-RUN OK]")
    assert "验证:" in result
    assert "- Update: file.txt" in result
    assert target.read_bytes() == b"one\r\ntwo\r\n"


def test_error_output_has_code_context_hint_without_full_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    _ = (tmp_path / "exists.txt").write_text(
        "SECRET_CONTENT_SHOULD_NOT_APPEAR", encoding="utf-8"
    )
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: exists.txt",
        "+FULL_PATCH_CONTENT_SHOULD_NOT_APPEAR",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith(f"[Error] {TARGET_EXISTS}:")
    assert "上下文:" in result
    assert "提示:" in result
    assert "SECRET_CONTENT_SHOULD_NOT_APPEAR" not in result
    assert "FULL_PATCH_CONTENT_SHOULD_NOT_APPEAR" not in result


def test_error_code_result_format_for_public_dry_run_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: api_key.py",
        "+value = 'redacted'",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith(f"[Error] {SENSITIVE_PATH}:")
    assert "提示:" in result
    assert "redacted" not in result


def test_dry_run_rejects_final_too_large(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(apply_patch_module, "TARGET_FILE_LIMIT_BYTES", 5)
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: too-large.txt",
        "+123456",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith(f"[Error] {FINAL_TOO_LARGE}:")


def test_dry_run_delete_rejects_invalid_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    _ = (tmp_path / "bad.txt").write_bytes(b"\xff")
    patch_text = _patch("*** Begin Patch", "*** Delete File: bad.txt", "*** End Patch")

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith(f"[Error] {INVALID_UTF8}:")


def test_apply_valid_multi_file_patch_add_update_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "existing.txt"
    old_file = tmp_path / "old.txt"
    _ = existing.write_bytes(b"old\n")
    _ = old_file.write_bytes(b"remove\n")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: nested/new.txt",
        "+created",
        "*** Update File: existing.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new",
        "*** Delete File: old.txt",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith("[OK] Applied patch")
    assert "Add 1" in result
    assert "Update 1" in result
    assert "Delete 1" in result
    assert (tmp_path / "nested/new.txt").read_bytes() == b"created\n"
    assert existing.read_bytes() == b"new\n"
    assert not old_file.exists()
    assert not list(tmp_path.rglob(".old.txt.apply-patch-*"))


def test_apply_parent_dir_component_file_fails_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    parent_file = tmp_path / "parent"
    _ = parent_file.write_text("still file", encoding="utf-8")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: parent/child.txt",
        "+content",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(f"[Error] {INVALID_PATH}:")
    assert parent_file.read_text(encoding="utf-8") == "still file"
    assert not (tmp_path / "parent/child.txt").exists()


def test_rollback_after_partial_failure_removes_added_file_and_parent_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "target.txt"
    _ = target.write_bytes(b"old\n")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: new/dir/added.txt",
        "+created",
        "*** Update File: target.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new",
        "*** End Patch",
    )

    def fail_on_update(operation: PlannedOperation) -> None:
        if operation.path == "target.txt":
            raise PatchError(APPLY_FAILED, "测试注入失败")

    monkeypatch.setattr(
        apply_patch_module, "_before_apply_operation_hook", fail_on_update
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(f"[Error] {APPLY_FAILED}:")
    assert target.read_bytes() == b"old\n"
    assert not (tmp_path / "new").exists()


def test_revalidate_target_changed_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "target.txt"
    _ = target.write_bytes(b"old\n")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: target.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new",
        "*** End Patch",
    )

    def change_after_validation(planned_patch: PlannedPatch) -> None:
        _ = planned_patch
        _ = target.write_bytes(b"changed\n")

    monkeypatch.setattr(
        apply_patch_module, "_pre_write_revalidation_hook", change_after_validation
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(f"[Error] {TARGET_CHANGED}:")
    assert target.read_bytes() == b"changed\n"


def test_apply_preserves_permission_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "script.sh"
    _ = script.write_bytes(b"old\n")
    os.chmod(script, 0o755)
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: script.sh",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith("[OK] Applied patch")
    assert script.read_bytes() == b"new\n"
    assert stat.S_IMODE(script.stat().st_mode) == 0o755


def test_delete_rollback_after_partial_failure_restores_bytes_and_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "victim.txt"
    _ = victim.write_bytes(b"victim\n")
    os.chmod(victim, 0o640)
    patch_text = _patch(
        "*** Begin Patch",
        "*** Delete File: victim.txt",
        "*** Add File: after.txt",
        "+after",
        "*** End Patch",
    )

    def fail_on_add(operation: PlannedOperation) -> None:
        if operation.path == "after.txt":
            raise PatchError(APPLY_FAILED, "测试注入失败")

    monkeypatch.setattr(apply_patch_module, "_before_apply_operation_hook", fail_on_add)

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(f"[Error] {APPLY_FAILED}:")
    assert victim.read_bytes() == b"victim\n"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o640
    assert not (tmp_path / "after.txt").exists()


def test_update_change_during_planning_rejects_stale_content_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "target.txt"
    _ = target.write_bytes(b"old\n")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: target.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new",
        "*** End Patch",
    )
    changed = False

    def change_during_read(path: Path) -> None:
        nonlocal changed
        if path.name == "target.txt" and not changed:
            changed = True
            _ = target.write_bytes(b"race\n")

    monkeypatch.setattr(
        apply_patch_module, "_after_existing_bytes_read_hook", change_during_read
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(f"[Error] {TARGET_CHANGED}:")
    assert target.read_bytes() == b"race\n"


def test_same_file_concurrent_apply_is_serialized_by_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "target.txt"
    _ = target.write_bytes(b"old\n")
    patch_first = _patch(
        "*** Begin Patch",
        "*** Update File: target.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+first",
        "*** End Patch",
    )
    patch_second = _patch(
        "*** Begin Patch",
        "*** Update File: target.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+second",
        "*** End Patch",
    )
    first_holds_lock = threading.Event()
    first_can_finish = threading.Event()
    second_attempted_lock = threading.Event()
    results: dict[str, str] = {}
    errors: dict[str, BaseException] = {}
    original_pre_write = cast(
        Callable[[PlannedPatch], None],
        apply_patch_module._pre_write_revalidation_hook,
    )
    original_acquire = cast(
        Callable[[tuple[str, ...]], list[threading.Lock]],
        apply_patch_module._acquire_file_locks,
    )

    def gate_first_apply(planned_patch: PlannedPatch) -> None:
        original_pre_write(planned_patch)
        if threading.current_thread().name == "apply-first":
            first_holds_lock.set()
            if not first_can_finish.wait(5):
                raise AssertionError("first apply was not released")

    def record_second_lock_attempt(lock_keys: tuple[str, ...]) -> list[threading.Lock]:
        if threading.current_thread().name == "apply-second":
            second_attempted_lock.set()
        return original_acquire(lock_keys)

    def run_apply(name: str, patch_text: str) -> None:
        try:
            results[name] = cast(
                str,
                apply_patch.invoke({"patch_text": patch_text, "dry_run": False}),
            )
        except BaseException as exc:
            errors[name] = exc

    monkeypatch.setattr(
        apply_patch_module, "_pre_write_revalidation_hook", gate_first_apply
    )
    monkeypatch.setattr(
        apply_patch_module, "_acquire_file_locks", record_second_lock_attempt
    )

    first_thread = threading.Thread(
        target=run_apply,
        args=("first", patch_first),
        name="apply-first",
    )
    first_thread.start()
    assert first_holds_lock.wait(5)

    second_thread = threading.Thread(
        target=run_apply,
        args=("second", patch_second),
        name="apply-second",
    )
    second_thread.start()
    assert second_attempted_lock.wait(5)
    assert second_thread.is_alive()

    first_can_finish.set()
    first_thread.join(5)
    second_thread.join(5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == {}
    assert results["first"].startswith("[OK] Applied patch")
    assert results["second"].startswith(f"[Error] {HUNK_MISMATCH}:")
    assert target.read_bytes() == b"first\n"


def test_delete_change_during_planning_rejects_stale_content_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "delete-me.txt"
    _ = target.write_bytes(b"delete\n")
    patch_text = _patch(
        "*** Begin Patch", "*** Delete File: delete-me.txt", "*** End Patch"
    )
    changed = False

    def change_during_read(path: Path) -> None:
        nonlocal changed
        if path.name == "delete-me.txt" and not changed:
            changed = True
            _ = target.write_bytes(b"race\n")

    monkeypatch.setattr(
        apply_patch_module, "_after_existing_bytes_read_hook", change_during_read
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(f"[Error] {TARGET_CHANGED}:")
    assert target.read_bytes() == b"race\n"


def test_success_cleanup_failure_returns_cleanup_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: cleanup.txt",
        "+created",
        "*** End Patch",
    )

    def fail_cleanup(path: Path) -> None:
        if "apply-patch-temp" in path.name:
            raise OSError("cleanup failed")
        original_remove(path)

    original_remove = cast(
        Callable[[Path], None], apply_patch_module._remove_path_if_exists
    )
    monkeypatch.setattr(apply_patch_module, "_remove_path_if_exists", fail_cleanup)

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith("[OK] Applied patch")
    assert f"[Warning] {CLEANUP_FAILED}:" in result
    assert "残留路径" in result
    assert (tmp_path / "cleanup.txt").read_bytes() == b"created\n"


def test_update_symlink_swap_during_target_read_is_not_followed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "target.txt"
    symlink_source = tmp_path / "symlink-source.txt"
    _ = target.write_bytes(b"old\n")
    _ = symlink_source.write_bytes(b"old\n")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: target.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new",
        "*** End Patch",
    )
    open_count = 0

    def swap_before_planning_open(path: Path) -> None:
        nonlocal open_count
        if path.name != "target.txt":
            return
        open_count += 1
        if open_count == 2:
            target.unlink()
            target.symlink_to(symlink_source)

    monkeypatch.setattr(
        apply_patch_module, "_before_target_open_hook", swap_before_planning_open
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(f"[Error] {TARGET_IS_SYMLINK}:")
    assert target.is_symlink()
    assert symlink_source.read_bytes() == b"old\n"


def test_delete_symlink_swap_during_target_read_is_not_followed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "delete-me.txt"
    symlink_source = tmp_path / "symlink-source.txt"
    _ = target.write_bytes(b"delete\n")
    _ = symlink_source.write_bytes(b"delete\n")
    patch_text = _patch(
        "*** Begin Patch", "*** Delete File: delete-me.txt", "*** End Patch"
    )
    swapped = False

    def swap_before_delete_open(path: Path) -> None:
        nonlocal swapped
        if path.name == "delete-me.txt" and not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(symlink_source)

    monkeypatch.setattr(
        apply_patch_module, "_before_target_open_hook", swap_before_delete_open
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(f"[Error] {TARGET_IS_SYMLINK}:")
    assert target.is_symlink()
    assert symlink_source.read_bytes() == b"delete\n"


def _swap_dir_to_outside_symlink(directory_path: Path, outside_path: Path) -> Path:
    real_dir = directory_path.with_name(f"{directory_path.name}-real")
    _ = directory_path.rename(real_dir)
    directory_path.symlink_to(outside_path, target_is_directory=True)
    return real_dir


def test_update_parent_symlink_swap_after_revalidation_does_not_escape_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    workspace_dir = tmp_path / "dir"
    outside_dir = tmp_path / "outside"
    workspace_dir.mkdir()
    outside_dir.mkdir()
    target = workspace_dir / "target.txt"
    outside_target = outside_dir / "target.txt"
    _ = target.write_bytes(b"old\n")
    _ = outside_target.write_bytes(b"outside\n")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: dir/target.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new",
        "*** End Patch",
    )
    real_dirs: list[Path] = []

    def swap_parent_after_revalidation(operation: PlannedOperation) -> None:
        if operation.path == "dir/target.txt" and not real_dirs:
            real_dirs.append(_swap_dir_to_outside_symlink(workspace_dir, outside_dir))

    monkeypatch.setattr(
        apply_patch_module,
        "_before_apply_operation_hook",
        swap_parent_after_revalidation,
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(
        (f"[Error] {TARGET_IS_SYMLINK}:", f"[Error] {TARGET_CHANGED}:")
    )
    assert real_dirs
    assert workspace_dir.is_symlink()
    assert outside_target.read_bytes() == b"outside\n"
    assert (real_dirs[0] / "target.txt").read_bytes() == b"old\n"
    assert not list(outside_dir.glob(".target.txt.apply-patch-*"))


def test_delete_parent_symlink_swap_after_revalidation_does_not_escape_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    workspace_dir = tmp_path / "dir"
    outside_dir = tmp_path / "outside"
    workspace_dir.mkdir()
    outside_dir.mkdir()
    target = workspace_dir / "target.txt"
    outside_target = outside_dir / "target.txt"
    _ = target.write_bytes(b"delete\n")
    _ = outside_target.write_bytes(b"outside\n")
    patch_text = _patch(
        "*** Begin Patch", "*** Delete File: dir/target.txt", "*** End Patch"
    )
    real_dirs: list[Path] = []

    def swap_parent_after_revalidation(operation: PlannedOperation) -> None:
        if operation.path == "dir/target.txt" and not real_dirs:
            real_dirs.append(_swap_dir_to_outside_symlink(workspace_dir, outside_dir))

    monkeypatch.setattr(
        apply_patch_module,
        "_before_apply_operation_hook",
        swap_parent_after_revalidation,
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False}))

    assert result.startswith(
        (f"[Error] {TARGET_IS_SYMLINK}:", f"[Error] {TARGET_CHANGED}:")
    )
    assert real_dirs
    assert workspace_dir.is_symlink()
    assert outside_target.read_bytes() == b"outside\n"
    assert (real_dirs[0] / "target.txt").read_bytes() == b"delete\n"
    assert not list(outside_dir.glob(".target.txt.apply-patch-*"))


def test_tool_list_exposes_apply_patch_and_not_write_file():
    tools_module = importlib.import_module("src.tools")
    exported_tool_list = cast(list[_NamedTool], tools_module.tool_list)
    tool_names = [tool.name for tool in exported_tool_list]

    assert tool_names == ["read_file", "apply_patch", "terminal"]
    assert "write_file" not in tool_names


def test_write_file_module_importable_after_tool_list_deexposure():
    write_file_module = importlib.import_module("src.tools.write_file")
    write_file_tool = cast(_NamedTool, write_file_module.write_file)

    assert write_file_tool.name == "write_file"


def test_apply_patch_tool_langchain_invoke_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: new.txt",
        "+created",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith("[DRY-RUN OK]")
    assert "Add 1" in result
    assert not (tmp_path / "new.txt").exists()


def test_apply_patch_tool_schema_only_patch_text_and_dry_run():
    tool_args = cast(dict[str, object], apply_patch.args)

    assert set(tool_args) == {"patch_text", "dry_run"}


def test_apply_patch_tool_name_is_stable():
    assert apply_patch.name == "apply_patch"


def test_apply_patch_tool_args_include_public_scaffold_parameters():
    tool_args = cast(dict[str, object], apply_patch.args)

    assert "patch_text" in tool_args
    assert "dry_run" in tool_args


def test_apply_patch_documentation_describes_v1_scope_boundaries():
    module_docs = apply_patch_module.__doc__ or ""
    tool_description = apply_patch.description or ""
    docs = f"{module_docs}\n{tool_description}"

    assert "v1" in docs
    assert "*** Begin Patch" in docs
    assert "*** Add File" in docs
    assert "*** Update File" in docs
    assert "*** Delete File" in docs
    assert "dry_run=True" in docs
    assert "no git diff compatibility" in docs
    assert "diff --git" in docs or "git diff" in docs
    assert "no fuzzy matching" in docs
    assert "no terminal hardening" in docs
    assert "terminal" in docs.lower()


def test_scope_boundaries_keep_terminal_logic_and_write_file_available():
    from src.llm import client as llm_client
    from src.tools import tool_list
    from src.tools.write_file import write_file

    tool_names = [tool.name for tool in tool_list]
    client_source = inspect.getsource(llm_client)

    assert tool_names == ["read_file", "apply_patch", "terminal"]
    assert "safe_prefixes" in client_source
    assert "terminal" in client_source
    assert write_file.name == "write_file"


def test_apply_patch_tool_direct_function_call_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: direct.txt",
        "+created",
        "*** End Patch",
    )
    direct_apply = getattr(apply_patch, "func", None)

    assert direct_apply is not None
    result = cast(Callable[..., str], direct_apply)(patch_text=patch_text, dry_run=True)

    assert result.startswith("[DRY-RUN OK]")
    assert "direct.txt" in result
    assert not (tmp_path / "direct.txt").exists()


def test_success_output_summarizes_without_full_file_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    hidden_content = "SUCCESS_CONTENT_SHOULD_NOT_APPEAR"
    patch_text = _patch(
        "*** Begin Patch",
        "*** Add File: summary.txt",
        f"+{hidden_content}",
        "*** End Patch",
    )

    dry_result = cast(
        str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True})
    )
    apply_result = cast(
        str, apply_patch.invoke({"patch_text": patch_text, "dry_run": False})
    )

    for result in (dry_result, apply_result):
        assert "summary.txt" in result
        assert hidden_content not in result
        assert patch_text not in result
    assert (tmp_path / "summary.txt").read_text(
        encoding="utf-8"
    ) == f"{hidden_content}\n"


def test_dry_run_rejects_update_final_too_large(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(apply_patch_module, "TARGET_FILE_LIMIT_BYTES", 5)
    target = tmp_path / "file.txt"
    _ = target.write_text("old\n", encoding="utf-8")
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: file.txt",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+123456",
        "*** End Patch",
    )

    result = cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": True}))

    assert result.startswith(f"[Error] {FINAL_TOO_LARGE}:")
    assert target.read_text(encoding="utf-8") == "old\n"


def test_apply_patch_non_dry_run_parse_error_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    result = cast(
        str,
        apply_patch.invoke(
            {
                "patch_text": "*** Begin Patch\n*** End Patch\n",
                "dry_run": False,
            }
        ),
    )

    assert result.startswith(f"[Error] {INVALID_ENVELOPE}:")
    assert "apply" in result
    assert "提示:" in result


def test_parse_valid_add_update_delete_envelope():
    document = parse_patch_text(
        _patch(
            "*** Begin Patch",
            "*** Add File: docs/new.txt",
            "+hello",
            "+",
            "++literal",
            "",
            "*** Update File: src/example.py",
            "@@ -1,2 +1,3 @@",
            " keep",
            "-old",
            "+new",
            "+added",
            "\\ No newline at end of file",
            "",
            "@@ -10,1 +11,0 @@",
            "-remove",
            "*** Delete File: docs/old.txt",
            "*** End Patch",
        )
    )

    assert [operation.kind for operation in document.operations] == [
        "add",
        "update",
        "delete",
    ]

    add_operation = document.operations[0]
    assert add_operation.path == "docs/new.txt"
    assert [line.content for line in add_operation.added_lines] == [
        "hello",
        "",
        "+literal",
    ]

    update_operation = document.operations[1]
    assert update_operation.path == "src/example.py"
    assert len(update_operation.hunks) == 2
    first_hunk = update_operation.hunks[0]
    assert (first_hunk.old_start, first_hunk.old_count) == (1, 2)
    assert (first_hunk.new_start, first_hunk.new_count) == (1, 3)
    assert [line.kind for line in first_hunk.lines] == [
        "context",
        "delete",
        "add",
        "add",
    ]
    assert [line.content for line in first_hunk.lines] == [
        "keep",
        "old",
        "new",
        "added",
    ]
    assert first_hunk.lines[-1].no_newline_at_end is True

    second_hunk = update_operation.hunks[1]
    assert (second_hunk.old_start, second_hunk.old_count) == (10, 1)
    assert (second_hunk.new_start, second_hunk.new_count) == (11, 0)
    assert [line.kind for line in second_hunk.lines] == ["delete"]

    assert document.operations[2].path == "docs/old.txt"


@pytest.mark.parametrize(
    ("patch_text", "code"),
    [
        (
            _patch("*** Add File: a.txt", "+content", "*** End Patch"),
            INVALID_ENVELOPE,
        ),
        (
            _patch("*** Begin Patch", "*** Add File: a.txt", "+content"),
            INVALID_ENVELOPE,
        ),
        (
            _patch(
                "*** Begin Patch",
                "*** Add File: a.txt",
                "+content",
                "*** End Patch",
                "trailing",
            ),
            INVALID_ENVELOPE,
        ),
        (
            _patch("*** Begin Patch", "*** End Patch"),
            INVALID_ENVELOPE,
        ),
        (
            _patch("*** Begin Patch", "*** Move File: a.txt", "*** End Patch"),
            MALFORMED_SECTION,
        ),
        (
            _patch("*** Begin Patch", "diff --git a/a.txt b/a.txt", "*** End Patch"),
            MALFORMED_SECTION,
        ),
        (
            _patch("*** Begin Patch", "--- a/a.txt", "*** End Patch"),
            MALFORMED_SECTION,
        ),
        (
            _patch("*** Begin Patch", "+++ b/a.txt", "*** End Patch"),
            MALFORMED_SECTION,
        ),
        (
            _patch(
                "*** Begin Patch",
                "*** Add File: a.txt",
                "+content",
                "*** Delete File: a.txt",
                "*** End Patch",
            ),
            DUPLICATE_FILE_OPERATION,
        ),
    ],
)
def test_parser_rejects_invalid_envelope_and_section_errors(patch_text: str, code: str):
    _assert_parser_error(patch_text, code)


@pytest.mark.parametrize(
    "patch_text",
    [
        _patch("*** Begin Patch", "diff --git a/a.txt b/a.txt", "*** End Patch"),
        _patch("*** Begin Patch", "--- a/a.txt", "*** End Patch"),
        _patch("*** Begin Patch", "+++ b/a.txt", "*** End Patch"),
    ],
)
def test_parser_rejects_git_diff_snippets(patch_text: str):
    _assert_parser_error(patch_text, MALFORMED_SECTION)


@pytest.mark.parametrize(
    "patch_text",
    [
        _patch(
            "*** Begin Patch",
            "*** Update File: a.txt",
            "@@ -1 +1 @@",
            "*** End Patch",
        ),
        _patch(
            "*** Begin Patch",
            "*** Update File: a.txt",
            "@@ -1,1 +1,1 @@",
            "body without prefix",
            "*** End Patch",
        ),
        _patch(
            "*** Begin Patch",
            "*** Update File: a.txt",
            "@@ -1,1 +1,1 @@",
            " same",
            "+extra",
            "*** End Patch",
        ),
    ],
)
def test_parser_rejects_malformed_hunk_syntax(patch_text: str):
    _assert_parser_error(patch_text, MALFORMED_HUNK)


def test_parser_rejects_eof_marker_without_previous_hunk_body_line():
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: a.txt",
        "@@ -0,0 +0,0 @@",
        "\\ No newline at end of file",
        "*** End Patch",
    )

    _assert_parser_error(patch_text, MALFORMED_HUNK)
