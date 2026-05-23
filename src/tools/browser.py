"""LangChain tools for browser-driven web automation."""

from __future__ import annotations

import json
from collections import deque
from urllib.parse import urljoin, urlparse

from langchain_core.tools import tool

from src.browser import get_browser_service


@tool
async def browser_open(url: str, session_id: str = "default", new_tab: bool = False) -> str:
    """Open a URL in the managed browser session."""
    return _format_response(
        await get_browser_service().request(
            "open",
            {"session_id": session_id, "url": url, "new_tab": new_tab},
        )
    )


@tool
async def browser_click(selector: str, session_id: str = "default", confirm: bool = False) -> str:
    """Click an element by CSS selector in the managed browser session."""
    payload = {"session_id": session_id, "selector": selector, "confirm": confirm}
    if _requires_confirmation("click", payload) and not confirm:
        return _confirmation_required("click", selector)
    return _format_response(await get_browser_service().request("click", payload))


@tool
async def browser_fill(
    selector: str,
    value: str,
    session_id: str = "default",
    confirm: bool = False,
) -> str:
    """Fill an input or textarea by CSS selector in the managed browser session."""
    payload = {
        "session_id": session_id,
        "selector": selector,
        "value": value,
        "confirm": confirm,
    }
    if _requires_confirmation("fill", payload) and not confirm:
        return _confirmation_required("fill", selector)
    return _format_response(await get_browser_service().request("fill", payload))


@tool
async def browser_extract(
    session_id: str = "default",
    max_chars: int = 6000,
    max_links: int = 50,
) -> str:
    """Extract visible page text, links, and tables from the current browser page."""
    return _format_response(
        await get_browser_service().request(
            "extract",
            {
                "session_id": session_id,
                "max_chars": max_chars,
                "max_links": max_links,
            },
        )
    )


@tool
async def browser_scroll(pixels: int = 1000, session_id: str = "default") -> str:
    """Scroll the current browser page vertically."""
    return _format_response(
        await get_browser_service().request(
            "scroll",
            {"session_id": session_id, "pixels": pixels},
        )
    )


@tool
async def browser_screenshot(
    path: str = "",
    session_id: str = "default",
    full_page: bool = True,
) -> str:
    """Take a screenshot of the current browser page and return the saved path."""
    payload: dict[str, object] = {"session_id": session_id, "full_page": full_page}
    if path:
        payload["path"] = path
    return _format_response(await get_browser_service().request("screenshot", payload))


@tool
async def browser_wait(
    selector: str = "",
    timeout_ms: int = 5000,
    session_id: str = "default",
) -> str:
    """Wait for a selector or fixed timeout in the current browser page."""
    return _format_response(
        await get_browser_service().request(
            "wait",
            {"session_id": session_id, "selector": selector, "timeout_ms": timeout_ms},
        )
    )


@tool
async def crawl_site(
    start_url: str,
    max_pages: int = 10,
    session_id: str = "default",
    same_domain_only: bool = True,
) -> str:
    """Crawl pages from a start URL and return concise extracted summaries."""
    return await _crawl_site(
        start_url=start_url,
        max_pages=max_pages,
        session_id=session_id,
        same_domain_only=same_domain_only,
    )


@tool
async def web_task(
    task: str,
    start_url: str = "",
    max_pages: int = 5,
    session_id: str = "default",
) -> str:
    """Run a high-level web research task against a start URL."""
    if not start_url.strip():
        return "[Error] web_task requires start_url in this version"
    result = await _crawl_site(
        start_url=start_url,
        max_pages=max_pages,
        session_id=session_id,
        same_domain_only=True,
    )
    return f"Web task: {task}\n\n{result}"


async def _crawl_site(
    *,
    start_url: str,
    max_pages: int,
    session_id: str,
    same_domain_only: bool,
) -> str:
    start_url = start_url.strip()
    if not start_url:
        return "[Error] start_url is required"
    limit = max(1, min(int(max_pages), 50))
    root_domain = urlparse(start_url).netloc
    queue: deque[str] = deque([start_url])
    seen: set[str] = set()
    pages: list[dict] = []

    while queue and len(pages) < limit:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)

        opened = await get_browser_service().request(
            "open",
            {"session_id": session_id, "url": current},
        )
        if not opened.get("ok"):
            pages.append({"url": current, "error": opened.get("error", "open failed")})
            continue

        extracted = await get_browser_service().request(
            "extract",
            {"session_id": session_id, "max_chars": 3000, "max_links": 50},
        )
        if not extracted.get("ok"):
            pages.append({"url": current, "error": extracted.get("error", "extract failed")})
            continue

        pages.append(extracted)
        for link in extracted.get("links", []):
            if not isinstance(link, dict):
                continue
            target = urljoin(current, str(link.get("url", "")))
            parsed = urlparse(target)
            if parsed.scheme not in {"http", "https"}:
                continue
            if same_domain_only and parsed.netloc != root_domain:
                continue
            if target not in seen:
                queue.append(target)

    return _format_crawl_result(start_url, pages)


def _format_crawl_result(start_url: str, pages: list[dict]) -> str:
    lines = [f"Crawl results for: {start_url}", f"Pages visited: {len(pages)}"]
    for index, page in enumerate(pages, start=1):
        lines.append("")
        lines.append(f"{index}. {page.get('title') or '(untitled)'}")
        lines.append(f"   URL: {page.get('url', '')}")
        if page.get("error"):
            lines.append(f"   Error: {page['error']}")
            continue
        text = str(page.get("text", "")).strip()
        if text:
            lines.append("   Text:")
            lines.append(_indent(_truncate(text, 800), "     "))
        links = page.get("links") or []
        if links:
            lines.append(f"   Links: {len(links)}")
        tables = page.get("tables") or []
        if tables:
            lines.append(f"   Tables: {len(tables)}")
    return "\n".join(lines)


def _format_response(response: dict) -> str:
    if not response.get("ok"):
        return f"[Error] {response.get('error', 'Browser action failed')}"
    return json.dumps(response, ensure_ascii=False, indent=2)


def _requires_confirmation(action: str, payload: dict) -> bool:
    haystack = " ".join(
        str(payload.get(key, ""))
        for key in ("selector", "value", "text", "url")
    ).lower()
    risky_terms = (
        "submit",
        "delete",
        "remove",
        "checkout",
        "purchase",
        "buy",
        "pay",
        "upload",
        "download",
        "send",
        "publish",
        "password",
        "credit",
        "card",
        "验证码",
        "支付",
        "购买",
        "删除",
        "发送",
        "发布",
        "上传",
        "下载",
    )
    return action in {"click", "fill"} and any(term in haystack for term in risky_terms)


def _confirmation_required(action: str, target: str) -> str:
    return (
        "[ConfirmationRequired] "
        f"browser_{action} target appears high-risk: {target}. "
        "Re-run with confirm=True only after explicit user approval."
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())
