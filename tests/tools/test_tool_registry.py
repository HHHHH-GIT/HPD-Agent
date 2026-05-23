"""Tests for the public tool_list registry surface."""

from __future__ import annotations

from typing import Protocol, cast

import src.tools as tools_module


class _NamedTool(Protocol):
    name: str


def test_code_intel_tools_registered_without_removing_existing_tools() -> None:
    exported_tool_list = cast(list[_NamedTool], tools_module.tool_list)
    tool_names = [tool.name for tool in exported_tool_list]

    assert tool_names == [
        "read_file",
        "apply_patch",
        "terminal",
        "websearch",
        "browser_open",
        "browser_click",
        "browser_fill",
        "browser_extract",
        "browser_scroll",
        "browser_screenshot",
        "browser_wait",
        "crawl_site",
        "web_task",
        "code_search",
        "code_outline",
        "code_context",
        "code_semantic",
        "code_verify",
    ]
    assert "write_file" not in tool_names


def test_code_intel_tools_are_exported_from_src_tools() -> None:
    assert tools_module.code_search.name == "code_search"
    assert tools_module.code_outline.name == "code_outline"
    assert tools_module.code_context.name == "code_context"
    assert tools_module.code_semantic.name == "code_semantic"
    assert tools_module.code_verify.name == "code_verify"
    assert tools_module.websearch.name == "websearch"
    assert tools_module.browser_open.name == "browser_open"
    assert tools_module.crawl_site.name == "crawl_site"
    assert tools_module.web_task.name == "web_task"
