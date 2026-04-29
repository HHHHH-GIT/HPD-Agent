# pyright: reportUnknownMemberType=false
import importlib
from pathlib import Path
from typing import cast

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
    NOT_IMPLEMENTED,
    PATCH_TEXT_LIMIT_BYTES,
    PATCH_TOO_LARGE,
    SENSITIVE_PATH,
    TARGET_EXISTS,
    TARGET_FILE_LIMIT_BYTES,
    TARGET_IS_DIRECTORY,
    TARGET_IS_SYMLINK,
    TARGET_MISSING,
    TARGET_TOO_LARGE,
    UTF8_BOM,
    ExistingFileContent,
    PatchDocument,
    PatchError,
    PatchOperation,
    apply_patch,
    apply_update_hunks,
    parse_patch_text,
    validate_patch_document,
)


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


@pytest.mark.parametrize(
    "raw_path",
    [
        "/absolute.txt",
        "../escape.txt",
        "safe/../escape.txt",
        "~/secret.txt",
        "dir\\file.txt",
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


def test_apply_patch_tool_name_is_stable():
    assert apply_patch.name == "apply_patch"


def test_apply_patch_tool_args_include_public_scaffold_parameters():
    tool_args = cast(dict[str, object], apply_patch.args)

    assert "patch_text" in tool_args
    assert "dry_run" in tool_args


def test_apply_patch_placeholder_output_is_structured_error():
    result = cast(
        str,
        apply_patch.invoke(
            {
                "patch_text": "*** Begin Patch\n*** End Patch\n",
                "dry_run": False,
            }
        ),
    )

    assert result.startswith(f"[Error] {NOT_IMPLEMENTED}:")
    assert "尚未实现" in result
    assert "apply" in result


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
