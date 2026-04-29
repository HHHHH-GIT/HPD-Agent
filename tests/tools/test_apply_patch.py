# pyright: reportUnknownMemberType=false
"""apply_patch v2 完整测试套件。

 1. 解析器:信封 / section / SEARCH/REPLACE 块 / CONTENT 块
 2. 路径校验:相对路径 / Windows 盘符 / 敏感词
 3. Symlink / 目录 / 类型校验
 4. 大小限制
 5. UTF-8 / BOM / 二进制
 6. SEARCH 匹配核心:唯一性 / 不存在 / 多匹配 / 重叠 / 空 SEARCH
 7. 换行处理:LF / CRLF / 混合 / 末尾换行
 8. NOOP 检测
 9. 多块乱序应用
10. Dry-run 安全性
11. 真实 apply:add / update / replace / delete / 多文件
12. 权限保留
13. 回滚机制
14. 并发锁
15. CLEANUP_FAILED 警告
16. 错误输出结构
17. Tool 元信息
18. 完整端到端场景重放(对应 v1 翻车场景)
"""

import importlib
import inspect
import os
from pathlib import Path
import stat
import threading
from typing import Callable, Protocol, cast

import pytest

from src.tools.apply_patch import (
    AMBIGUOUS_MATCH,
    APPLY_FAILED,
    BINARY_CONTENT,
    BLOCK_OVERLAP,
    CLEANUP_FAILED,
    DUPLICATE_FILE_OPERATION,
    EMPTY_SEARCH_NON_EMPTY_FILE,
    ERROR_CODES,
    FINAL_TOO_LARGE,
    INVALID_ENVELOPE,
    INVALID_PATH,
    INVALID_UTF8,
    MALFORMED_BLOCK,
    MALFORMED_SECTION,
    MIXED_NEWLINES,
    NOOP_PATCH,
    PATCH_TEXT_LIMIT_BYTES,
    PATCH_TOO_LARGE,
    ROLLBACK_FAILED,
    SEARCH_NOT_FOUND,
    SENSITIVE_PATH,
    TARGET_CHANGED,
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
    PlannedOperation,
    PlannedPatch,
    SearchReplaceBlock,
    apply_patch,
    apply_update_blocks,
    parse_patch_text,
    validate_patch_document,
)


class _NamedTool(Protocol):
    name: str


apply_patch_module = importlib.import_module("src.tools.apply_patch")


# ============================================================================
# Helpers
# ============================================================================


def _patch(*lines: str) -> str:
    """组合一段 patch 文本(自动拼接 \\n 并以 \\n 收尾)。"""
    return "\n".join(lines) + "\n"


def _envelope(*body: str) -> str:
    """组合完整信封(*** Begin Patch ... *** End Patch)。"""
    return _patch("*** Begin Patch", *body, "*** End Patch")


def _add_block(path: str, *content_lines: str) -> tuple[str, ...]:
    """构造 Add File 段(返回信封 body 行 tuple)。"""
    return (
        f"*** Add File: {path}",
        "<<<<<<< CONTENT",
        *content_lines,
        ">>>>>>> END",
    )


def _replace_block(path: str, *content_lines: str) -> tuple[str, ...]:
    """构造 Replace File 段。"""
    return (
        f"*** Replace File: {path}",
        "<<<<<<< CONTENT",
        *content_lines,
        ">>>>>>> END",
    )


def _update_block(path: str, search: list[str], replace: list[str]) -> tuple[str, ...]:
    """构造一个 Update File + 单个 SEARCH/REPLACE 块。"""
    return (
        f"*** Update File: {path}",
        "<<<<<<< SEARCH",
        *search,
        "=======",
        *replace,
        ">>>>>>> REPLACE",
    )


def _assert_parser_error(patch_text: str, code: str) -> None:
    """断言 parse_patch_text 抛出指定错误码。"""
    with pytest.raises(PatchError) as exc_info:
        _ = parse_patch_text(patch_text)

    error = exc_info.value
    assert error.code == code, f"expected {code}, got {error.code}: {error.message}"
    assert error.to_error_result().startswith(f"[Error][APPLY_PATCH][{code}]")


def _assert_validation_error(document: PatchDocument, code: str) -> None:
    """断言 validate_patch_document 抛出指定错误码。"""
    with pytest.raises(PatchError) as exc_info:
        _ = validate_patch_document(document)

    error = exc_info.value
    assert error.code == code, f"expected {code}, got {error.code}: {error.message}"
    assert error.to_error_result().startswith(f"[Error][APPLY_PATCH][{code}]")


def _add_document(path: str, *content_lines: str) -> PatchDocument:
    """构造一个只包含 Add File 操作的 PatchDocument(用于路径校验测试)。"""
    return PatchDocument(
        operations=(
            PatchOperation(kind="add", path=path, content="\n".join(content_lines)),
        )
    )


def _update_document_simple(path: str) -> PatchDocument:
    """构造一个简单的 Update File 文档(用于目标状态校验测试)。"""
    return parse_patch_text(_envelope(*_update_block(path, ["X"], ["Y"])))


def _delete_document(path: str) -> PatchDocument:
    return PatchDocument(operations=(PatchOperation(kind="delete", path=path),))


def _invoke(patch_text: str, dry_run: bool = True) -> str:
    return cast(str, apply_patch.invoke({"patch_text": patch_text, "dry_run": dry_run}))


def _assert_error_result(result: str, code: str) -> None:
    assert result.startswith(
        f"[Error][APPLY_PATCH][{code}]"
    ), f"expected {code} error, got: {result[:200]}"


def _assert_ok_result(result: str, dry_run: bool = True) -> None:
    expected = "[DRY-RUN OK]" if dry_run else "[OK]"
    assert result.startswith(expected), f"expected {expected}, got: {result[:200]}"


# ============================================================================
# 1. 解析器:信封 / section / 块结构
# ============================================================================


class TestEnvelope:
    """*** Begin Patch / *** End Patch 信封校验。"""

    def test_parse_minimal_add(self):
        document = parse_patch_text(_envelope(*_add_block("a.txt", "hello")))
        assert len(document.operations) == 1
        op = document.operations[0]
        assert op.kind == "add"
        assert op.path == "a.txt"
        assert op.content == "hello"

    def test_parse_minimal_update(self):
        document = parse_patch_text(
            _envelope(*_update_block("a.txt", ["old"], ["new"]))
        )
        assert len(document.operations) == 1
        op = document.operations[0]
        assert op.kind == "update"
        assert len(op.blocks) == 1
        assert op.blocks[0].search_text == "old"
        assert op.blocks[0].replace_text == "new"

    def test_parse_minimal_replace(self):
        document = parse_patch_text(_envelope(*_replace_block("a.txt", "fresh")))
        assert document.operations[0].kind == "replace"
        assert document.operations[0].content == "fresh"

    def test_parse_minimal_delete(self):
        document = parse_patch_text(_envelope("*** Delete File: a.txt"))
        assert document.operations[0].kind == "delete"
        assert document.operations[0].path == "a.txt"

    def test_parse_all_four_operations(self):
        document = parse_patch_text(
            _envelope(
                *_add_block("new.txt", "hello"),
                *_update_block("u.txt", ["old"], ["new"]),
                *_replace_block("r.txt", "replaced"),
                "*** Delete File: d.txt",
            )
        )
        kinds = [op.kind for op in document.operations]
        assert kinds == ["add", "update", "replace", "delete"]

    def test_blank_separators_between_sections_allowed(self):
        document = parse_patch_text(
            _envelope(
                *_add_block("a.txt", "x"),
                "",
                "",
                *_update_block("u.txt", ["o"], ["n"]),
            )
        )
        assert len(document.operations) == 2

    def test_missing_begin_marker(self):
        _assert_parser_error("not begin\n*** End Patch\n", INVALID_ENVELOPE)

    def test_missing_end_marker(self):
        _assert_parser_error(
            "*** Begin Patch\n*** Add File: a.txt\n<<<<<<< CONTENT\nx\n>>>>>>> END\n",
            INVALID_ENVELOPE,
        )

    def test_empty_envelope(self):
        _assert_parser_error("*** Begin Patch\n*** End Patch\n", INVALID_ENVELOPE)

    def test_trailing_content_after_end_marker(self):
        _assert_parser_error(
            "*** Begin Patch\n"
            "*** Add File: a.txt\n"
            "<<<<<<< CONTENT\nx\n>>>>>>> END\n"
            "*** End Patch\n"
            "trailing junk\n",
            INVALID_ENVELOPE,
        )

    def test_only_whitespace_after_end_marker_is_ok(self):
        document = parse_patch_text(
            "*** Begin Patch\n"
            "*** Add File: a.txt\n"
            "<<<<<<< CONTENT\nx\n>>>>>>> END\n"
            "*** End Patch\n"
            "\n\n   \n"
        )
        assert len(document.operations) == 1


class TestParserGitDiffRejection:
    """v2 拒绝 git diff 片段并提示用户切换格式。"""

    @pytest.mark.parametrize(
        "snippet",
        [
            "diff --git a/a.txt b/a.txt",
            "--- a/a.txt",
            "+++ b/a.txt",
            "@@ -1,1 +1,1 @@",
        ],
    )
    def test_git_diff_snippet_rejected(self, snippet: str):
        _assert_parser_error(_envelope(snippet), MALFORMED_SECTION)


class TestParserSection:
    """*** Add/Update/Replace/Delete File: 行的格式与重复检测。"""

    def test_unknown_section_rejected(self):
        _assert_parser_error(_envelope("*** Move File: a.txt"), MALFORMED_SECTION)

    def test_empty_section_path(self):
        _assert_parser_error(
            _envelope("*** Add File: "),
            MALFORMED_SECTION,
        )

    def test_duplicate_path_in_envelope(self):
        _assert_parser_error(
            _envelope(
                *_add_block("a.txt", "x"),
                "*** Delete File: a.txt",
            ),
            DUPLICATE_FILE_OPERATION,
        )

    def test_add_without_content_block_rejected(self):
        _assert_parser_error(
            _envelope("*** Add File: a.txt", "+just plus"),
            MALFORMED_SECTION,
        )

    def test_replace_without_content_block_rejected(self):
        _assert_parser_error(
            _envelope("*** Replace File: a.txt"),
            MALFORMED_SECTION,
        )

    def test_update_without_blocks_rejected(self):
        _assert_parser_error(
            _envelope("*** Update File: a.txt"),
            MALFORMED_SECTION,
        )

    def test_delete_with_body_rejected(self):
        _assert_parser_error(
            _envelope("*** Delete File: a.txt", "<<<<<<< CONTENT", "x", ">>>>>>> END"),
            MALFORMED_SECTION,
        )

    def test_content_followed_by_unexpected_lines(self):
        _assert_parser_error(
            _envelope(
                "*** Add File: a.txt",
                "<<<<<<< CONTENT",
                "x",
                ">>>>>>> END",
                "garbage",
            ),
            MALFORMED_SECTION,
        )


class TestParserBlocks:
    """SEARCH/REPLACE 与 CONTENT 块结构错误。"""

    def test_content_missing_close(self):
        _assert_parser_error(
            _envelope("*** Add File: a.txt", "<<<<<<< CONTENT", "x"),
            MALFORMED_BLOCK,
        )

    def test_search_missing_divider(self):
        _assert_parser_error(
            _envelope(
                "*** Update File: a.txt",
                "<<<<<<< SEARCH",
                "old",
                ">>>>>>> REPLACE",
            ),
            MALFORMED_BLOCK,
        )

    def test_search_missing_close(self):
        _assert_parser_error(
            "*** Begin Patch\n"
            "*** Update File: a.txt\n"
            "<<<<<<< SEARCH\n"
            "old\n"
            "=======\n"
            "new\n"
            "*** End Patch\n",
            MALFORMED_BLOCK,
        )

    def test_nested_search_open_rejected(self):
        _assert_parser_error(
            _envelope(
                "*** Update File: a.txt",
                "<<<<<<< SEARCH",
                "<<<<<<< SEARCH",
                "=======",
                "new",
                ">>>>>>> REPLACE",
            ),
            MALFORMED_BLOCK,
        )

    def test_update_expects_search_open(self):
        _assert_parser_error(
            _envelope(
                "*** Update File: a.txt",
                "this is not a SEARCH open marker",
            ),
            MALFORMED_BLOCK,
        )

    def test_search_with_empty_search_body_parses(self):
        """空 SEARCH 在解析器层面合法,语义错误在 apply 时报。"""
        document = parse_patch_text(
            _envelope(
                "*** Update File: a.txt",
                "<<<<<<< SEARCH",
                "=======",
                "content",
                ">>>>>>> REPLACE",
            )
        )
        assert document.operations[0].blocks[0].search_text == ""

    def test_search_with_multiline_content(self):
        document = parse_patch_text(
            _envelope(
                "*** Update File: a.txt",
                "<<<<<<< SEARCH",
                "line 1",
                "line 2",
                "line 3",
                "=======",
                "new line a",
                "new line b",
                ">>>>>>> REPLACE",
            )
        )
        block = document.operations[0].blocks[0]
        assert block.search_text == "line 1\nline 2\nline 3"
        assert block.replace_text == "new line a\nnew line b"

    def test_multiple_search_replace_blocks_in_one_update(self):
        document = parse_patch_text(
            _envelope(
                "*** Update File: a.txt",
                "<<<<<<< SEARCH",
                "A",
                "=======",
                "AA",
                ">>>>>>> REPLACE",
                "<<<<<<< SEARCH",
                "B",
                "=======",
                "BB",
                ">>>>>>> REPLACE",
            )
        )
        assert len(document.operations[0].blocks) == 2

    def test_content_block_preserves_blank_lines(self):
        document = parse_patch_text(
            _envelope(
                "*** Add File: a.txt",
                "<<<<<<< CONTENT",
                "first",
                "",
                "third",
                ">>>>>>> END",
            )
        )
        assert document.operations[0].content == "first\n\nthird"

    def test_content_block_with_empty_body(self):
        document = parse_patch_text(
            _envelope(
                "*** Add File: empty.txt",
                "<<<<<<< CONTENT",
                ">>>>>>> END",
            )
        )
        assert document.operations[0].content == ""

    def test_no_plus_prefix_munging(self):
        """v2 不再把 + 当作前缀,内容原样保留。"""
        document = parse_patch_text(
            _envelope(
                "*** Add File: a.txt",
                "<<<<<<< CONTENT",
                "+leading",
                "++double",
                ">>>>>>> END",
            )
        )
        assert document.operations[0].content == "+leading\n++double"


# ============================================================================
# 2. 路径校验
# ============================================================================


class TestPathSyntax:
    """相对路径校验:..、绝对路径、控制字符、空白、~ 等。"""

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
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        raw_path: str,
    ):
        monkeypatch.chdir(tmp_path)
        _assert_validation_error(_add_document(raw_path, "x"), INVALID_PATH)

    def test_allows_gitignore_and_github_and_env_example(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        document = parse_patch_text(
            _envelope(
                *_add_block(".gitignore", "*.pyc"),
                *_add_block(".github/workflows/test.yml", "name: test"),
                *_add_block(".env.example", "EXAMPLE=1"),
                *_add_block("monkey.py", "print('ok')"),
            )
        )
        validated = validate_patch_document(document)
        assert [op.relative_path.as_posix() for op in validated.operations] == [
            ".gitignore",
            ".github/workflows/test.yml",
            ".env.example",
            "monkey.py",
        ]

    def test_relative_windows_separators_normalize(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        document = parse_patch_text(_envelope(*_add_block("dir\\file.txt", "x")))
        validated = validate_patch_document(document)
        assert validated.operations[0].relative_path.as_posix() == "dir/file.txt"

    @pytest.mark.parametrize(
        "raw_path",
        [
            "C:/outside.txt",
            "C:\\outside.txt",
            "C:relative.txt",
            "C:dir\\file.txt",
        ],
    )
    def test_windows_drive_paths_reject_unsafe_or_non_windows(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        raw_path: str,
    ):
        monkeypatch.chdir(tmp_path)
        _assert_validation_error(_add_document(raw_path, "x"), INVALID_PATH)

    def test_windows_drive_path_under_workspace_normalizes_when_windows(
        self,
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
            _envelope(*_add_block("C:\\workspace\\dir\\file.txt", "x"))
        )
        validated = validate_patch_document(document)
        assert validated.operations[0].relative_path.as_posix() == "dir/file.txt"


class TestSensitivePath:
    """敏感凭据路径词表(env / secret / credential / key / token / password 等)。"""

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
            "config/secrets.yml",
            "my-password-store.txt",
        ],
    )
    def test_rejects_sensitive_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        raw_path: str,
    ):
        monkeypatch.chdir(tmp_path)
        _assert_validation_error(_add_document(raw_path, "x"), SENSITIVE_PATH)

    def test_env_example_is_allowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        document = parse_patch_text(_envelope(*_add_block(".env.example", "FOO=1")))
        validated = validate_patch_document(document)
        assert validated.operations[0].relative_path.name == ".env.example"


class TestDuplicateCanonicalPath:
    """规范化后路径相同 → 拒绝。"""

    def test_canonical_duplicate_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        document = parse_patch_text(
            _envelope(
                *_add_block("a.txt", "x"),
                "*** Delete File: ./a.txt",
            )
        )
        _assert_validation_error(document, DUPLICATE_FILE_OPERATION)


# ============================================================================
# 3. Symlink / 目录 / 类型校验
# ============================================================================


class TestSymlinkRejection:
    """所有 symlink 路径(目标 / 父级 / 中途 swap)都拒绝。"""

    def test_rejects_symlink_parent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (tmp_path / "link").symlink_to(real_dir, target_is_directory=True)
        _assert_validation_error(_add_document("link/new.txt", "x"), TARGET_IS_SYMLINK)

    def test_rejects_symlink_target_for_update(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        real_file = tmp_path / "real.txt"
        _ = real_file.write_text("ok", encoding="utf-8")
        (tmp_path / "link.txt").symlink_to(real_file)
        _assert_validation_error(_update_document_simple("link.txt"), TARGET_IS_SYMLINK)

    def test_rejects_symlink_target_for_delete(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        real_file = tmp_path / "real.txt"
        _ = real_file.write_text("ok", encoding="utf-8")
        (tmp_path / "link.txt").symlink_to(real_file)
        _assert_validation_error(_delete_document("link.txt"), TARGET_IS_SYMLINK)

    def test_update_symlink_swap_during_target_read_is_not_followed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "target.txt"
        symlink_source = tmp_path / "symlink-source.txt"
        _ = target.write_bytes(b"old\n")
        _ = symlink_source.write_bytes(b"old\n")
        patch_text = _envelope(*_update_block("target.txt", ["old"], ["new"]))
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
        result = _invoke(patch_text, dry_run=False)
        _assert_error_result(result, TARGET_IS_SYMLINK)
        assert target.is_symlink()
        assert symlink_source.read_bytes() == b"old\n"


class TestDirectoryRejection:
    def test_rejects_directory_target_for_update(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "dir.txt").mkdir()
        _assert_validation_error(
            _update_document_simple("dir.txt"), TARGET_IS_DIRECTORY
        )


class TestTargetState:
    """目标存在性校验。"""

    def test_add_existing_target_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _ = (tmp_path / "x.txt").write_text("anything", encoding="utf-8")
        result = _invoke(_envelope(*_add_block("x.txt", "y")))
        _assert_error_result(result, TARGET_EXISTS)
        # hint 应该提示用 Replace
        assert "Replace File" in result

    def test_update_missing_target_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _assert_validation_error(_update_document_simple("nope.txt"), TARGET_MISSING)

    def test_replace_missing_target_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(_envelope(*_replace_block("nope.txt", "stuff")))
        _assert_error_result(result, TARGET_MISSING)

    def test_delete_missing_target_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(_envelope("*** Delete File: ghost.txt"))
        _assert_error_result(result, TARGET_MISSING)


# ============================================================================
# 4. 大小限制
# ============================================================================


class TestSizeLimits:
    def test_patch_too_large_short_circuits(self):
        _assert_parser_error("x" * (PATCH_TEXT_LIMIT_BYTES + 1), PATCH_TOO_LARGE)

    def test_validate_rejects_oversized_patch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        document = _add_document("safe.txt", "x")
        oversized_patch = "x" * (PATCH_TEXT_LIMIT_BYTES + 1)
        with pytest.raises(PatchError) as exc_info:
            _ = validate_patch_document(document, patch_text=oversized_patch)
        assert exc_info.value.code == PATCH_TOO_LARGE

    def test_existing_target_too_large(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "big.txt"
        _ = target.write_bytes(b"x" * (TARGET_FILE_LIMIT_BYTES + 1))
        _assert_validation_error(_update_document_simple("big.txt"), TARGET_TOO_LARGE)

    def test_final_too_large_for_add(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(apply_patch_module, "TARGET_FILE_LIMIT_BYTES", 5)
        result = _invoke(_envelope(*_add_block("too-large.txt", "123456")))
        _assert_error_result(result, FINAL_TOO_LARGE)

    def test_final_too_large_for_update(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(apply_patch_module, "TARGET_FILE_LIMIT_BYTES", 5)
        target = tmp_path / "f.txt"
        _ = target.write_text("old\n", encoding="utf-8")
        result = _invoke(_envelope(*_update_block("f.txt", ["old"], ["123456"])))
        _assert_error_result(result, FINAL_TOO_LARGE)
        assert target.read_text(encoding="utf-8") == "old\n"


# ============================================================================
# 5. UTF-8 / BOM / 二进制
# ============================================================================


class TestUtf8AndBinary:
    def test_rejects_invalid_utf8_for_update(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _ = (tmp_path / "bad.txt").write_bytes(b"\xff")
        _assert_validation_error(_update_document_simple("bad.txt"), INVALID_UTF8)

    def test_rejects_invalid_utf8_for_delete(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _ = (tmp_path / "bad.txt").write_bytes(b"\xff")
        result = _invoke(_envelope("*** Delete File: bad.txt"))
        _assert_error_result(result, INVALID_UTF8)

    def test_rejects_binary_like_target(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "bin.txt"
        _ = target.write_bytes(b"old\x00\n")
        _assert_validation_error(_update_document_simple("bin.txt"), BINARY_CONTENT)

    def test_rejects_binary_in_add_content(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(_envelope(*_add_block("bin.txt", "safe\x00binary")))
        _assert_error_result(result, BINARY_CONTENT)
        assert not (tmp_path / "bin.txt").exists()

    def test_rejects_binary_in_update_replace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "f.txt"
        _ = target.write_text("old\n", encoding="utf-8")
        result = _invoke(_envelope(*_update_block("f.txt", ["old"], ["new\x01"])))
        _assert_error_result(result, BINARY_CONTENT)
        assert target.read_text(encoding="utf-8") == "old\n"

    def test_preserves_utf8_bom_in_validation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _ = (tmp_path / "bom.txt").write_bytes(UTF8_BOM + b"hello\n")
        # Update validation reads the existing content
        validated = validate_patch_document(_update_document_simple("bom.txt"))
        existing = validated.operations[0].existing_content
        assert existing is not None
        assert existing.has_utf8_bom is True
        assert existing.text == "hello"  # newline 已被规整化剥离
        assert existing.byte_size == 9  # 3 + 5 + 1


class TestErrorCodesExposed:
    def test_all_v2_codes_in_error_codes(self):
        for code in [
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
        ]:
            assert code in ERROR_CODES, f"{code} missing from ERROR_CODES"

    def test_v1_only_codes_no_longer_exposed(self):
        """v1 的 hunk 相关错误码已经废弃。"""
        v1_only = {
            "HUNK_MISMATCH",
            "HUNK_ORDER_ERROR",
            "INVALID_EOF_MARKER",
            "MALFORMED_HUNK",
        }
        for code in v1_only:
            assert not hasattr(
                apply_patch_module, code
            ), f"v1-only code {code} should not exist in v2"


# ============================================================================
# 6. SEARCH 匹配核心:apply_update_blocks
# ============================================================================


def _existing_content(raw_bytes: bytes) -> ExistingFileContent:
    """构造与 _read_existing_utf8_with_snapshot 等价的 ExistingFileContent。"""
    has_bom = raw_bytes.startswith(UTF8_BOM)
    payload = raw_bytes[len(UTF8_BOM) :] if has_bom else raw_bytes
    text = payload.decode("utf-8")
    # 通过模块内的私有工具规整化换行,保持 fixture 与生产路径一致
    normalized, trailing_newline, newline = apply_patch_module._detect_existing_newline(
        text
    )
    return ExistingFileContent(
        text=normalized,
        byte_size=len(raw_bytes),
        has_utf8_bom=has_bom,
        newline=newline,
        has_trailing_newline=trailing_newline,
    )


def _blocks(*pairs: tuple[str, str]) -> tuple[SearchReplaceBlock, ...]:
    """从 (search, replace) 元组列表构造块 tuple。"""
    return tuple(SearchReplaceBlock(search_text=s, replace_text=r) for s, r in pairs)


def _assert_apply_error(
    raw_bytes: bytes,
    code: str,
    blocks: tuple[SearchReplaceBlock, ...],
) -> None:
    with pytest.raises(PatchError) as exc_info:
        _ = apply_update_blocks(_existing_content(raw_bytes), blocks)
    error = exc_info.value
    assert error.code == code, f"expected {code}, got {error.code}: {error.message}"


class TestSearchMatching:
    def test_simple_unique_match(self):
        result = apply_update_blocks(
            _existing_content(b"alpha\nbeta\ngamma\n"),
            _blocks(("beta", "BETA")),
        )
        assert result.final_bytes == b"alpha\nBETA\ngamma\n"
        assert result.stats.added_lines == 1
        assert result.stats.deleted_lines == 1

    def test_multiline_match(self):
        result = apply_update_blocks(
            _existing_content(b"a\nb\nc\nd\n"),
            _blocks(("b\nc", "B\nC\nC2")),
        )
        assert result.final_bytes == b"a\nB\nC\nC2\nd\n"

    def test_search_not_found(self):
        _assert_apply_error(
            b"hello\n",
            SEARCH_NOT_FOUND,
            _blocks(("missing", "x")),
        )

    def test_ambiguous_match_rejected(self):
        _assert_apply_error(
            b"foo\nfoo\nfoo\n",
            AMBIGUOUS_MATCH,
            _blocks(("foo", "X")),
        )

    def test_ambiguous_resolved_by_more_context(self):
        result = apply_update_blocks(
            _existing_content(b"a\nfoo\nb\nfoo\nc\n"),
            _blocks(("b\nfoo\nc", "b\nBAR\nc")),
        )
        assert result.final_bytes == b"a\nfoo\nb\nBAR\nc\n"

    def test_line_aligned_no_substring_match(self):
        """SEARCH 必须按行对齐,不能匹配子串。"""
        _assert_apply_error(
            b"abcdef\n",
            SEARCH_NOT_FOUND,
            _blocks(("bcd", "X")),
        )

    def test_no_blocks_is_noop(self):
        _assert_apply_error(b"x\n", NOOP_PATCH, ())

    def test_search_at_file_start(self):
        result = apply_update_blocks(
            _existing_content(b"first\nsecond\n"),
            _blocks(("first", "FIRST")),
        )
        assert result.final_bytes == b"FIRST\nsecond\n"

    def test_search_at_file_end(self):
        result = apply_update_blocks(
            _existing_content(b"first\nlast\n"),
            _blocks(("last", "LAST")),
        )
        assert result.final_bytes == b"first\nLAST\n"


class TestEmptySearch:
    def test_empty_search_on_empty_file_writes_content(self):
        result = apply_update_blocks(
            _existing_content(b""),
            _blocks(("", "new content")),
        )
        # 空文件没有 trailing_newline,所以新内容也没有
        assert result.final_bytes == b"new content"

    def test_empty_search_on_non_empty_file_rejected(self):
        _assert_apply_error(
            b"hello\n",
            EMPTY_SEARCH_NON_EMPTY_FILE,
            _blocks(("", "new")),
        )

    def test_empty_search_error_hint_mentions_replace_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _ = (tmp_path / "x.txt").write_text("hello\n")
        result = _invoke(
            _envelope(
                "*** Update File: x.txt",
                "<<<<<<< SEARCH",
                "=======",
                "new",
                ">>>>>>> REPLACE",
            )
        )
        _assert_error_result(result, EMPTY_SEARCH_NON_EMPTY_FILE)
        assert "Replace File" in result


class TestBlockOverlap:
    def test_overlapping_blocks_rejected(self):
        _assert_apply_error(
            b"a\nb\nc\n",
            BLOCK_OVERLAP,
            _blocks(("a\nb", "AB"), ("b\nc", "BC")),
        )

    def test_disjoint_blocks_ok(self):
        result = apply_update_blocks(
            _existing_content(b"a\nb\nc\nd\n"),
            _blocks(("a", "A"), ("c", "C")),
        )
        assert result.final_bytes == b"A\nb\nC\nd\n"

    def test_blocks_provided_out_of_order_apply_correctly(self):
        result = apply_update_blocks(
            _existing_content(b"A\nB\nC\nD\nE\n"),
            _blocks(("D", "DD"), ("B", "BB")),
        )
        assert result.final_bytes == b"A\nBB\nC\nDD\nE\n"


# ============================================================================
# 7. 换行处理:LF / CRLF / 混合 / 末尾换行
# ============================================================================


class TestNewlineHandling:
    def test_preserves_lf_newline(self):
        result = apply_update_blocks(
            _existing_content(b"alpha\nbeta\n"),
            _blocks(("beta", "BETA")),
        )
        assert result.final_bytes == b"alpha\nBETA\n"
        assert result.newline == "\n"

    def test_preserves_crlf_newline(self):
        result = apply_update_blocks(
            _existing_content(b"alpha\r\nbeta\r\n"),
            _blocks(("beta", "BETA")),
        )
        assert result.final_bytes == b"alpha\r\nBETA\r\n"
        assert result.newline == "\r\n"

    def test_rejects_mixed_lf_and_crlf(self):
        _assert_apply_error(
            b"one\ntwo\r\n",
            MIXED_NEWLINES,
            _blocks(("one", "ONE")),
        )

    def test_rejects_lone_cr(self):
        _assert_apply_error(
            b"one\rtwo\n",
            MIXED_NEWLINES,
            _blocks(("one", "ONE")),
        )

    def test_preserves_no_trailing_newline(self):
        result = apply_update_blocks(
            _existing_content(b"hello"),
            _blocks(("hello", "world")),
        )
        assert result.final_bytes == b"world"

    def test_preserves_trailing_newline(self):
        result = apply_update_blocks(
            _existing_content(b"hello\n"),
            _blocks(("hello", "world")),
        )
        assert result.final_bytes == b"world\n"

    def test_preserves_utf8_bom_through_update(self):
        result = apply_update_blocks(
            _existing_content(UTF8_BOM + b"one\n"),
            _blocks(("one", "uno")),
        )
        assert result.final_bytes == UTF8_BOM + b"uno\n"
        assert result.has_utf8_bom is True

    def test_e2e_preserves_crlf_through_invoke(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "f.txt"
        _ = target.write_bytes(b"one\r\ntwo\r\nthree\r\n")
        result = _invoke(
            _envelope(*_update_block("f.txt", ["two"], ["TWO"])),
            dry_run=False,
        )
        _assert_ok_result(result, dry_run=False)
        assert target.read_bytes() == b"one\r\nTWO\r\nthree\r\n"


# ============================================================================
# 8. NOOP 检测
# ============================================================================


class TestNoopDetection:
    def test_search_equals_replace_is_noop(self):
        _assert_apply_error(
            b"hello\n",
            NOOP_PATCH,
            _blocks(("hello", "hello")),
        )

    def test_replace_with_same_content_is_noop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _ = (tmp_path / "x.txt").write_text("identical\n")
        result = _invoke(_envelope(*_replace_block("x.txt", "identical")))
        _assert_error_result(result, NOOP_PATCH)


# ============================================================================
# 9. Multi-block ordering and patch text newline detection
# ============================================================================


class TestMultiBlockApplication:
    def test_multiple_blocks_apply_in_file_order_when_provided_out_of_order(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "x.txt"
        _ = target.write_text("A\nB\nC\nD\nE\n")
        # 故意把 D 的块写在前面
        text = _envelope(
            "*** Update File: x.txt",
            "<<<<<<< SEARCH",
            "D",
            "=======",
            "DD",
            ">>>>>>> REPLACE",
            "<<<<<<< SEARCH",
            "B",
            "=======",
            "BB",
            ">>>>>>> REPLACE",
        )
        result = _invoke(text, dry_run=False)
        _assert_ok_result(result, dry_run=False)
        assert target.read_text() == "A\nBB\nC\nDD\nE\n"

    def test_overlap_rejected_e2e(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _ = (tmp_path / "x.txt").write_text("a\nb\nc\n")
        text = _envelope(
            "*** Update File: x.txt",
            "<<<<<<< SEARCH",
            "a",
            "b",
            "=======",
            "AB",
            ">>>>>>> REPLACE",
            "<<<<<<< SEARCH",
            "b",
            "c",
            "=======",
            "BC",
            ">>>>>>> REPLACE",
        )
        result = _invoke(text)
        _assert_error_result(result, BLOCK_OVERLAP)


# ============================================================================
# 10. Dry-run 安全性
# ============================================================================


class TestDryRunSafety:
    def test_dry_run_validates_without_mutating_or_creating_dirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        existing = tmp_path / "existing.txt"
        old_file = tmp_path / "old.txt"
        _ = existing.write_text("old\n", encoding="utf-8")
        _ = old_file.write_text("remove\n", encoding="utf-8")
        text = _envelope(
            *_add_block("nested/new.txt", "created"),
            *_update_block("existing.txt", ["old"], ["new"]),
            "*** Delete File: old.txt",
        )
        result = _invoke(text, dry_run=True)
        _assert_ok_result(result, dry_run=True)
        assert "Add 1" in result
        assert "Update 1" in result
        assert "Delete 1" in result
        assert "nested/new.txt" in result
        assert existing.read_text(encoding="utf-8") == "old\n"
        assert old_file.read_text(encoding="utf-8") == "remove\n"
        assert not (tmp_path / "nested").exists()

    def test_dry_run_summary_format(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "f.txt"
        _ = target.write_bytes(b"one\r\ntwo\r\n")
        result = _invoke(_envelope(*_update_block("f.txt", ["two"], ["TWO"])))
        _assert_ok_result(result, dry_run=True)
        assert "验证:" in result
        assert "- Update: f.txt" in result
        assert target.read_bytes() == b"one\r\ntwo\r\n"

    def test_dry_run_failure_does_not_mutate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "f.txt"
        _ = target.write_text("original\n")
        text = _envelope(*_update_block("f.txt", ["wrong"], ["x"]))
        result = _invoke(text)
        _assert_error_result(result, SEARCH_NOT_FOUND)
        assert target.read_text() == "original\n"

    def test_apply_failure_does_not_mutate_or_create(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """SEARCH_NOT_FOUND 在真实 apply 路径下也不应留下任何痕迹。"""
        monkeypatch.chdir(tmp_path)
        existing = tmp_path / "a.txt"
        _ = existing.write_text("original\n")
        text = _envelope(
            *_update_block("a.txt", ["WRONG"], ["x"]),
            *_add_block("b.txt", "should not be created"),
        )
        result = _invoke(text, dry_run=False)
        _assert_error_result(result, SEARCH_NOT_FOUND)
        assert existing.read_text() == "original\n"
        assert not (tmp_path / "b.txt").exists()


# ============================================================================
# 11. 真实 apply: add / update / replace / delete / 多文件
# ============================================================================


class TestApplyAdd:
    def test_add_creates_simple_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(
            _envelope(*_add_block("hello.txt", "hello", "world")), dry_run=False
        )
        _assert_ok_result(result, dry_run=False)
        assert (tmp_path / "hello.txt").read_text() == "hello\nworld\n"

    def test_add_creates_empty_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(_envelope(*_add_block("empty.txt")), dry_run=False)
        _assert_ok_result(result, dry_run=False)
        assert (tmp_path / "empty.txt").read_bytes() == b""

    def test_add_creates_nested_dirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(
            _envelope(*_add_block("a/b/c/file.txt", "deep")), dry_run=False
        )
        _assert_ok_result(result, dry_run=False)
        assert (tmp_path / "a/b/c/file.txt").read_text() == "deep\n"

    def test_add_with_plus_prefix_kept_literal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(
            _envelope(*_add_block("p.txt", "+plus", "++double")), dry_run=False
        )
        _assert_ok_result(result, dry_run=False)
        assert (tmp_path / "p.txt").read_text() == "+plus\n++double\n"

    def test_add_with_unicode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(
            _envelope(*_add_block("zh.md", "# 标题", "", "中文内容,带「特殊」标点。")),
            dry_run=False,
        )
        _assert_ok_result(result, dry_run=False)
        assert "「特殊」" in (tmp_path / "zh.md").read_text(encoding="utf-8")


class TestApplyUpdate:
    def test_update_simple(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "f.py"
        _ = target.write_text("def greet(name):\n    return 'hi'\n")
        text = _envelope(
            "*** Update File: f.py",
            "<<<<<<< SEARCH",
            "def greet(name):",
            "    return 'hi'",
            "=======",
            "def greet(name: str) -> str:",
            "    return f'hi {name}'",
            ">>>>>>> REPLACE",
        )
        result = _invoke(text, dry_run=False)
        _assert_ok_result(result, dry_run=False)
        assert target.read_text() == (
            "def greet(name: str) -> str:\n    return f'hi {name}'\n"
        )

    def test_update_no_trailing_newline_preserved(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "f.txt"
        _ = target.write_bytes(b"hello")
        result = _invoke(
            _envelope(*_update_block("f.txt", ["hello"], ["world"])),
            dry_run=False,
        )
        _assert_ok_result(result, dry_run=False)
        assert target.read_bytes() == b"world"


class TestApplyReplace:
    def test_replace_overwrites_existing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "x.txt"
        _ = target.write_text("old\nstuff\n")
        result = _invoke(
            _envelope(*_replace_block("x.txt", "completely new")),
            dry_run=False,
        )
        _assert_ok_result(result, dry_run=False)
        assert target.read_text() == "completely new\n"

    def test_replace_writes_to_empty_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """v1 的 TARGET_EXISTS 在空文件上的痛点 → v2 用 Replace File 解决。"""
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "x.md"
        _ = target.write_text("")
        result = _invoke(
            _envelope(*_replace_block("x.md", "# 标题", "", "## 概述", "正文。")),
            dry_run=False,
        )
        _assert_ok_result(result, dry_run=False)
        # 原文件无尾换行,Replace 沿用其习惯
        assert target.read_text() == "# 标题\n\n## 概述\n正文。"

    def test_replace_preserves_crlf_style(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "x.txt"
        _ = target.write_bytes(b"old\r\nstuff\r\n")
        result = _invoke(
            _envelope(*_replace_block("x.txt", "new", "fresh")),
            dry_run=False,
        )
        _assert_ok_result(result, dry_run=False)
        assert target.read_bytes() == b"new\r\nfresh\r\n"


class TestApplyDelete:
    def test_delete_removes_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "victim.txt"
        _ = target.write_text("bye\n")
        result = _invoke(_envelope("*** Delete File: victim.txt"), dry_run=False)
        _assert_ok_result(result, dry_run=False)
        assert not target.exists()


class TestApplyMultiFile:
    def test_apply_add_update_replace_delete_in_one_patch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        update_target = tmp_path / "u.txt"
        replace_target = tmp_path / "r.txt"
        delete_target = tmp_path / "d.txt"
        _ = update_target.write_text("update me\n")
        _ = replace_target.write_text("replace me\n")
        _ = delete_target.write_text("delete me\n")
        text = _envelope(
            *_add_block("a.txt", "added"),
            *_update_block("u.txt", ["update me"], ["UPDATED"]),
            *_replace_block("r.txt", "replaced"),
            "*** Delete File: d.txt",
        )
        result = _invoke(text, dry_run=False)
        _assert_ok_result(result, dry_run=False)
        assert (tmp_path / "a.txt").read_text() == "added\n"
        assert update_target.read_text() == "UPDATED\n"
        assert replace_target.read_text() == "replaced\n"
        assert not delete_target.exists()
        # 没有残留临时文件
        assert not list(tmp_path.glob(".*.apply-patch-*"))


# ============================================================================
# 12. 权限保留
# ============================================================================


class TestPermissionPreservation:
    def test_update_preserves_executable_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        script = tmp_path / "run.sh"
        _ = script.write_text("old\n")
        os.chmod(script, 0o755)
        result = _invoke(
            _envelope(*_update_block("run.sh", ["old"], ["new"])),
            dry_run=False,
        )
        _assert_ok_result(result, dry_run=False)
        assert script.read_text() == "new\n"
        assert stat.S_IMODE(script.stat().st_mode) == 0o755

    def test_replace_preserves_executable_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        script = tmp_path / "run.sh"
        _ = script.write_text("#!/bin/sh\noriginal\n")
        os.chmod(script, 0o750)
        result = _invoke(
            _envelope(*_replace_block("run.sh", "#!/bin/sh", "fresh")),
            dry_run=False,
        )
        _assert_ok_result(result, dry_run=False)
        assert stat.S_IMODE(script.stat().st_mode) == 0o750


# ============================================================================
# 13. 回滚机制
# ============================================================================


class TestRollback:
    def test_rollback_after_partial_failure_removes_added_and_dirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "target.txt"
        _ = target.write_text("old\n")
        text = _envelope(
            *_add_block("new/dir/added.txt", "created"),
            *_update_block("target.txt", ["old"], ["new"]),
        )

        def fail_on_target(operation: PlannedOperation) -> None:
            if operation.path == "target.txt":
                raise PatchError(APPLY_FAILED, "测试注入失败")

        monkeypatch.setattr(
            apply_patch_module, "_before_apply_operation_hook", fail_on_target
        )
        result = _invoke(text, dry_run=False)
        _assert_error_result(result, APPLY_FAILED)
        assert target.read_text() == "old\n"
        assert not (tmp_path / "new").exists()

    def test_rollback_restores_replaced_file_via_backup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        _ = a.write_text("original A\n")
        _ = b.write_text("original B\n")
        text = _envelope(
            *_replace_block("a.txt", "modified A"),
            *_replace_block("b.txt", "modified B"),
        )

        def fail_on_b(operation: PlannedOperation) -> None:
            if operation.path == "b.txt":
                raise RuntimeError("synthetic")

        monkeypatch.setattr(
            apply_patch_module, "_before_apply_operation_hook", fail_on_b
        )
        result = _invoke(text, dry_run=False)
        _assert_error_result(result, APPLY_FAILED)
        # a 应被备份恢复
        assert a.read_text() == "original A\n"
        assert b.read_text() == "original B\n"

    def test_delete_rollback_restores_bytes_and_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        victim = tmp_path / "victim.txt"
        _ = victim.write_bytes(b"victim content\n")
        os.chmod(victim, 0o640)
        text = _envelope(
            "*** Delete File: victim.txt",
            *_add_block("after.txt", "after"),
        )

        def fail_on_after(operation: PlannedOperation) -> None:
            if operation.path == "after.txt":
                raise PatchError(APPLY_FAILED, "测试注入失败")

        monkeypatch.setattr(
            apply_patch_module, "_before_apply_operation_hook", fail_on_after
        )
        result = _invoke(text, dry_run=False)
        _assert_error_result(result, APPLY_FAILED)
        assert victim.read_bytes() == b"victim content\n"
        assert stat.S_IMODE(victim.stat().st_mode) == 0o640
        assert not (tmp_path / "after.txt").exists()

    def test_revalidate_target_changed_before_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "target.txt"
        _ = target.write_text("old\n")
        text = _envelope(*_update_block("target.txt", ["old"], ["new"]))

        def hijack(planned_patch: PlannedPatch) -> None:
            _ = planned_patch
            _ = target.write_text("changed\n")

        monkeypatch.setattr(apply_patch_module, "_pre_write_revalidation_hook", hijack)
        result = _invoke(text, dry_run=False)
        _assert_error_result(result, TARGET_CHANGED)
        # 我们没回滚劫持者的修改,但也没把 v2 写入劫持后的文件
        assert target.read_text() == "changed\n"

    def test_target_changed_during_target_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "target.txt"
        _ = target.write_text("old\n")
        text = _envelope(*_update_block("target.txt", ["old"], ["new"]))
        triggered = False

        def race(path: Path) -> None:
            nonlocal triggered
            if path.name == "target.txt" and not triggered:
                triggered = True
                _ = target.write_text("race\n")

        monkeypatch.setattr(apply_patch_module, "_after_existing_bytes_read_hook", race)
        result = _invoke(text, dry_run=False)
        _assert_error_result(result, TARGET_CHANGED)
        assert target.read_text() == "race\n"

    def test_target_changed_during_delete_target_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "delete-me.txt"
        _ = target.write_bytes(b"delete\n")
        text = _envelope("*** Delete File: delete-me.txt")
        triggered = False

        def race(path: Path) -> None:
            nonlocal triggered
            if path.name == "delete-me.txt" and not triggered:
                triggered = True
                _ = target.write_bytes(b"race\n")

        monkeypatch.setattr(apply_patch_module, "_after_existing_bytes_read_hook", race)
        result = _invoke(text, dry_run=False)
        _assert_error_result(result, TARGET_CHANGED)
        assert target.read_bytes() == b"race\n"


class TestSymlinkSwapAttacks:
    """父目录或目标在校验后被换成 symlink → 写入不能逃出工作区。"""

    def test_update_parent_dir_swapped_to_outside_symlink(
        self,
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
        _ = target.write_text("old\n")
        _ = outside_target.write_text("outside\n")
        text = _envelope(*_update_block("dir/target.txt", ["old"], ["new"]))
        real_dirs: list[Path] = []

        def swap(operation: PlannedOperation) -> None:
            if operation.path == "dir/target.txt" and not real_dirs:
                real_dir = workspace_dir.with_name("dir-real")
                _ = workspace_dir.rename(real_dir)
                workspace_dir.symlink_to(outside_dir, target_is_directory=True)
                real_dirs.append(real_dir)

        monkeypatch.setattr(apply_patch_module, "_before_apply_operation_hook", swap)
        result = _invoke(text, dry_run=False)
        # 必须被拒(以 TARGET_IS_SYMLINK 或 TARGET_CHANGED 任一形式)
        assert (
            f"[Error][APPLY_PATCH][{TARGET_IS_SYMLINK}]" in result
            or f"[Error][APPLY_PATCH][{TARGET_CHANGED}]" in result
        ), result
        assert real_dirs
        assert workspace_dir.is_symlink()
        assert outside_target.read_bytes() == b"outside\n"
        assert (real_dirs[0] / "target.txt").read_bytes() == b"old\n"

    def test_delete_parent_dir_swapped_to_outside_symlink(
        self,
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
        text = _envelope("*** Delete File: dir/target.txt")
        real_dirs: list[Path] = []

        def swap(operation: PlannedOperation) -> None:
            if operation.path == "dir/target.txt" and not real_dirs:
                real_dir = workspace_dir.with_name("dir-real")
                _ = workspace_dir.rename(real_dir)
                workspace_dir.symlink_to(outside_dir, target_is_directory=True)
                real_dirs.append(real_dir)

        monkeypatch.setattr(apply_patch_module, "_before_apply_operation_hook", swap)
        result = _invoke(text, dry_run=False)
        assert (
            f"[Error][APPLY_PATCH][{TARGET_IS_SYMLINK}]" in result
            or f"[Error][APPLY_PATCH][{TARGET_CHANGED}]" in result
        ), result
        assert outside_target.read_bytes() == b"outside\n"
        assert (real_dirs[0] / "target.txt").read_bytes() == b"delete\n"

    def test_apply_parent_dir_component_is_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        parent_file = tmp_path / "parent"
        _ = parent_file.write_text("still file")
        text = _envelope(*_add_block("parent/child.txt", "x"))
        result = _invoke(text, dry_run=False)
        _assert_error_result(result, INVALID_PATH)
        assert parent_file.read_text() == "still file"


# ============================================================================
# 14. 并发锁
# ============================================================================


class TestConcurrentLocking:
    def test_same_file_concurrent_apply_serialized(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """两个线程同时对同一文件 apply,锁保证一前一后,后到的看到已变化的内容。"""
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "target.txt"
        _ = target.write_bytes(b"old\n")
        patch_first = _envelope(*_update_block("target.txt", ["old"], ["first"]))
        patch_second = _envelope(*_update_block("target.txt", ["old"], ["second"]))

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

        def gate_first(planned: PlannedPatch) -> None:
            original_pre_write(planned)
            if threading.current_thread().name == "apply-first":
                first_holds_lock.set()
                if not first_can_finish.wait(5):
                    raise AssertionError("first apply was not released")

        def record_second_attempt(
            lock_keys: tuple[str, ...],
        ) -> list[threading.Lock]:
            if threading.current_thread().name == "apply-second":
                second_attempted_lock.set()
            return original_acquire(lock_keys)

        def run(name: str, patch_text: str) -> None:
            try:
                results[name] = _invoke(patch_text, dry_run=False)
            except BaseException as exc:
                errors[name] = exc

        monkeypatch.setattr(
            apply_patch_module, "_pre_write_revalidation_hook", gate_first
        )
        monkeypatch.setattr(
            apply_patch_module, "_acquire_file_locks", record_second_attempt
        )

        first_thread = threading.Thread(
            target=run, args=("first", patch_first), name="apply-first"
        )
        first_thread.start()
        assert first_holds_lock.wait(5)

        second_thread = threading.Thread(
            target=run, args=("second", patch_second), name="apply-second"
        )
        second_thread.start()
        assert second_attempted_lock.wait(5)
        assert second_thread.is_alive()  # 第二个线程在锁上阻塞

        first_can_finish.set()
        first_thread.join(5)
        second_thread.join(5)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert errors == {}
        assert results["first"].startswith("[OK]")
        # 第二个 SEARCH("old") 在第一个写完后已变成 "first" → SEARCH_NOT_FOUND
        _assert_error_result(results["second"], SEARCH_NOT_FOUND)
        assert target.read_bytes() == b"first\n"


# ============================================================================
# 15. CLEANUP_FAILED 警告
# ============================================================================


class TestCleanupWarning:
    def test_success_with_cleanup_failure_returns_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        text = _envelope(*_add_block("c.txt", "created"))

        original_remove = cast(
            Callable[[Path], None], apply_patch_module._remove_path_if_exists
        )

        def fail_cleanup(path: Path) -> None:
            if "apply-patch-temp" in path.name:
                raise OSError("synthetic cleanup failure")
            original_remove(path)

        monkeypatch.setattr(apply_patch_module, "_remove_path_if_exists", fail_cleanup)
        result = _invoke(text, dry_run=False)
        _assert_ok_result(result, dry_run=False)
        assert f"[Warning][APPLY_PATCH][{CLEANUP_FAILED}]" in result
        assert "残留路径" in result
        assert (tmp_path / "c.txt").read_bytes() == b"created\n"


# ============================================================================
# 16. 错误输出结构
# ============================================================================


class TestErrorOutputFormat:
    def test_error_has_phase_and_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(_envelope(*_add_block("api_key.py", "x")))
        _assert_error_result(result, SENSITIVE_PATH)
        assert "phase: dry-run" in result
        assert "hint:" in result

    def test_apply_phase_label(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = _invoke(
            "*** Begin Patch\n*** End Patch\n",
            dry_run=False,
        )
        _assert_error_result(result, INVALID_ENVELOPE)
        assert "phase: apply" in result

    def test_search_not_found_includes_file_block_expected_actual_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "x.txt"
        _ = target.write_text("alpha\nbeta\ngamma\n")
        text = _envelope(*_update_block("x.txt", ["delta"], ["DELTA"]))
        result = _invoke(text)
        _assert_error_result(result, SEARCH_NOT_FOUND)
        assert "file: x.txt" in result
        assert "block:" in result
        assert "expected:" in result
        assert "delta" in result  # expected 内容
        assert "actual:" in result
        assert "alpha" in result  # actual 文件预览
        assert "hint:" in result

    def test_ambiguous_match_lists_positions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _ = (tmp_path / "x.txt").write_text("foo\nfoo\nfoo\n")
        text = _envelope(*_update_block("x.txt", ["foo"], ["BAR"]))
        result = _invoke(text)
        _assert_error_result(result, AMBIGUOUS_MATCH)
        assert "line 1" in result
        assert "line 2" in result
        assert "line 3" in result

    def test_target_exists_hint_mentions_replace_or_update(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        _ = (tmp_path / "x.txt").write_text("anything\n")
        result = _invoke(_envelope(*_add_block("x.txt", "y")))
        _assert_error_result(result, TARGET_EXISTS)
        assert "Replace File" in result or "Update File" in result

    def test_error_does_not_leak_full_target_or_patch_content(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """SENSITIVE_PATH 错误信息中不要把补丁正文 echo 回来。"""
        monkeypatch.chdir(tmp_path)
        text = _envelope(
            *_add_block("api_key.py", "FULL_PATCH_CONTENT_SHOULD_NOT_APPEAR")
        )
        result = _invoke(text)
        _assert_error_result(result, SENSITIVE_PATH)
        assert "FULL_PATCH_CONTENT_SHOULD_NOT_APPEAR" not in result


class TestPatchErrorClass:
    def test_basic_construction(self):
        err = PatchError(
            SEARCH_NOT_FOUND,
            "msg",
            file="a.txt",
            block="block #1",
            expected="EXP",
            actual="ACT",
            hint="HINT",
        )
        assert err.code == SEARCH_NOT_FOUND
        assert err.message == "msg"
        assert err.file == "a.txt"
        assert err.expected == "EXP"

    def test_to_error_result_has_all_fields(self):
        err = PatchError(
            APPLY_FAILED,
            "details",
            file="x",
            line=42,
            expected="A",
            actual="B",
            hint="check stuff",
        )
        rendered = err.to_error_result(phase="apply")
        assert "[Error][APPLY_PATCH][APPLY_FAILED]" in rendered
        assert "message: details" in rendered
        assert "phase: apply" in rendered
        assert "line: 42" in rendered
        assert "file: x" in rendered
        assert "expected:" in rendered
        assert "  A" in rendered
        assert "actual:" in rendered
        assert "  B" in rendered
        assert "hint: check stuff" in rendered

    def test_default_hint_provided_per_code(self):
        err = PatchError(SEARCH_NOT_FOUND, "msg")
        rendered = err.to_error_result(phase="dry-run")
        # 即使没传 hint,也会有针对该错误码的默认提示
        assert "hint:" in rendered


# ============================================================================
# 17. Tool 元信息
# ============================================================================


class TestToolMetadata:
    def test_apply_patch_tool_name_is_stable(self):
        assert apply_patch.name == "apply_patch"

    def test_apply_patch_args_only_patch_text_and_dry_run(self):
        tool_args = cast(dict[str, object], apply_patch.args)
        assert set(tool_args) == {"patch_text", "dry_run"}

    def test_apply_patch_documentation_describes_v2_protocol(self):
        module_docs = apply_patch_module.__doc__ or ""
        tool_description = apply_patch.description or ""
        docs = f"{module_docs}\n{tool_description}"

        # v2 信封
        assert "*** Begin Patch" in docs
        assert "*** End Patch" in docs
        assert "*** Add File" in docs
        assert "*** Update File" in docs
        assert "*** Replace File" in docs
        assert "*** Delete File" in docs
        # SEARCH/REPLACE 块
        assert "<<<<<<< SEARCH" in docs
        assert ">>>>>>> REPLACE" in docs
        assert "<<<<<<< CONTENT" in docs
        assert ">>>>>>> END" in docs
        # 非目标(non-goals)
        assert "no git diff compatibility" in docs
        assert "diff --git" in docs or "git diff" in docs
        assert "no fuzzy matching" in docs
        assert "no terminal hardening" in docs
        # dry_run 行为
        assert "dry_run" in docs

    def test_apply_patch_tool_direct_function_call(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        direct = getattr(apply_patch, "func", None)
        assert direct is not None
        result = cast(Callable[..., str], direct)(
            patch_text=_envelope(*_add_block("d.txt", "x")), dry_run=True
        )
        _assert_ok_result(result, dry_run=True)
        assert "d.txt" in result
        assert not (tmp_path / "d.txt").exists()


class TestToolListIntegration:
    """与 src.tools 包的集成。这些测试需要真实的 src.tools 包结构。"""

    def test_tool_list_exposes_apply_patch_and_not_write_file(self):
        tools_module = importlib.import_module("src.tools")
        exported_tool_list = cast(list[_NamedTool], tools_module.tool_list)
        tool_names = [tool.name for tool in exported_tool_list]
        assert tool_names == ["read_file", "apply_patch", "terminal"]
        assert "write_file" not in tool_names

    def test_write_file_module_still_importable(self):
        write_file_module = importlib.import_module("src.tools.write_file")
        write_file_tool = cast(_NamedTool, write_file_module.write_file)
        assert write_file_tool.name == "write_file"

    def test_scope_boundaries_keep_terminal_logic_and_write_file_available(self):
        from src.llm import client as llm_client
        from src.tools import tool_list
        from src.tools.write_file import write_file

        tool_names = [tool.name for tool in tool_list]
        client_source = inspect.getsource(llm_client)

        assert tool_names == ["read_file", "apply_patch", "terminal"]
        assert "safe_prefixes" in client_source
        assert "terminal" in client_source
        assert write_file.name == "write_file"


class TestSummaryDoesNotLeakContent:
    """成功输出只摘要,不回显文件全文。"""

    def test_success_output_omits_full_content(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        hidden = "SUCCESS_CONTENT_SHOULD_NOT_APPEAR"
        text = _envelope(*_add_block("s.txt", hidden))
        dry = _invoke(text, dry_run=True)
        applied = _invoke(text, dry_run=False)
        for result in (dry, applied):
            assert "s.txt" in result
            assert hidden not in result
            assert text not in result
        assert (tmp_path / "s.txt").read_text() == f"{hidden}\n"


# ============================================================================
# 18. 端到端场景重放(对应 v1 中翻车的真实 LLM 对话)
# ============================================================================


class TestE2EScenarios:
    """复现 v1 中 LLM 实际翻车的场景,验证 v2 一次写对。"""

    def test_scene_write_into_empty_existing_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """v1 痛点:空文件已存在 → Add File 失败 (TARGET_EXISTS)。
        v2 解决:用 Replace File 一次成功。
        """
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "test.md"
        _ = target.write_text("")
        text = _envelope(
            *_replace_block(
                "test.md",
                "# apply_patch 工具使用方法",
                "",
                "## 概述",
                "apply_patch 用于精确修改文本文件。",
            )
        )
        result = _invoke(text, dry_run=False)
        _assert_ok_result(result, dry_run=False)
        expected = (
            "# apply_patch 工具使用方法\n\n## 概述\napply_patch 用于精确修改文本文件。"
        )
        # 原文件无尾换行 → Replace 沿用习惯
        assert target.read_text() == expected

    def test_scene_large_content_no_count_arithmetic(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """v1 痛点:LLM 算错 81 vs 97 行数。
        v2 解决:CONTENT 块,不需要任何计数。
        """
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "doc.md"
        _ = target.write_text("")
        big_lines = [f"line {i}" for i in range(1, 98)]  # 97 行
        text = _envelope(*_replace_block("doc.md", *big_lines))
        result = _invoke(text, dry_run=False)
        _assert_ok_result(result, dry_run=False)
        assert target.read_text() == "\n".join(big_lines)

    def test_scene_append_to_existing_doc(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """v1 痛点:在尾部追加,LLM 反复写错 hunk header。
        v2 解决:SEARCH 末尾 + REPLACE 末尾追加。
        """
        monkeypatch.chdir(tmp_path)
        initial = (
            "# 工具说明\n"
            "\n"
            "## 概述\n"
            "apply_patch 用于精确修改文本。\n"
            "\n"
            "## 使用流程\n"
            "1. 读取必要内容。\n"
            "2. 生成 patch。\n"
            "3. apply。\n"
        )
        target = tmp_path / "test.md"
        _ = target.write_text(initial)
        text = _envelope(
            "*** Update File: test.md",
            "<<<<<<< SEARCH",
            "3. apply。",
            "=======",
            "3. apply。",
            "",
            "## 常见错误",
            "",
            "### TARGET_EXISTS",
            "**原因**:目标文件已存在。",
            "**解决**:改用 Replace File 或 Update File。",
            ">>>>>>> REPLACE",
        )
        result = _invoke(text, dry_run=False)
        _assert_ok_result(result, dry_run=False)
        actual = target.read_text()
        assert actual.startswith(initial)
        assert "## 常见错误" in actual
        assert "### TARGET_EXISTS" in actual
        assert actual.endswith("Update File。\n")

    def test_scene_multiple_independent_edits_no_offset_arithmetic(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """v1 痛点:多 hunk 之间累计偏移算错。
        v2 解决:每个块独立匹配,不需算偏移。
        """
        monkeypatch.chdir(tmp_path)
        src = (
            "import os\n"
            "import sys\n"
            "\n"
            "def run():\n"
            "    print('hello')\n"
            "    return 0\n"
            "\n"
            "def main():\n"
            "    print('main')\n"
            "    return run()\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    sys.exit(main())\n"
        )
        target = tmp_path / "app.py"
        _ = target.write_text(src)
        text = _envelope(
            "*** Update File: app.py",
            "<<<<<<< SEARCH",
            "import os",
            "import sys",
            "=======",
            "import os",
            "import sys",
            "from typing import NoReturn",
            ">>>>>>> REPLACE",
            "<<<<<<< SEARCH",
            "def run():",
            "    print('hello')",
            "    return 0",
            "=======",
            "def run() -> int:",
            "    print('hello')",
            "    return 0",
            ">>>>>>> REPLACE",
            "<<<<<<< SEARCH",
            "def main():",
            "    print('main')",
            "    return run()",
            "=======",
            "def main() -> int:",
            "    print('main')",
            "    return run()",
            ">>>>>>> REPLACE",
        )
        result = _invoke(text, dry_run=False)
        _assert_ok_result(result, dry_run=False)
        actual = target.read_text()
        assert "from typing import NoReturn" in actual
        assert "def run() -> int:" in actual
        assert "def main() -> int:" in actual
        # 函数定义都只出现一次
        assert actual.count("def run") == 1
        assert actual.count("def main") == 1

    def test_scene_unicode_content(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """中文标点、Unicode 引号、emoji 等。"""
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "x.md"
        _ = target.write_text("## 标题\n\n这是一段中文内容。\n")
        text = _envelope(
            *_update_block(
                "x.md",
                ["这是一段中文内容。"],
                ["这是一段更新后的中文内容,带「特殊」标点。"],
            )
        )
        result = _invoke(text, dry_run=False)
        _assert_ok_result(result, dry_run=False)
        actual = target.read_text(encoding="utf-8")
        assert "「特殊」" in actual
