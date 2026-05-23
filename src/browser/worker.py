"""Browser worker and Playwright-backed execution layer."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol


_ALLOWED_ACTIONS = {
    "open",
    "click",
    "fill",
    "extract",
    "scroll",
    "screenshot",
    "wait",
}


class BrowserBackend(Protocol):
    async def open(self, session_id: str, **params: Any) -> dict[str, Any]: ...
    async def click(self, session_id: str, **params: Any) -> dict[str, Any]: ...
    async def fill(self, session_id: str, **params: Any) -> dict[str, Any]: ...
    async def extract(self, session_id: str, **params: Any) -> dict[str, Any]: ...
    async def scroll(self, session_id: str, **params: Any) -> dict[str, Any]: ...
    async def screenshot(self, session_id: str, **params: Any) -> dict[str, Any]: ...
    async def wait(self, session_id: str, **params: Any) -> dict[str, Any]: ...


class BrowserWorker:
    """Routes JSON actions to a browser backend."""

    def __init__(self, backend: BrowserBackend | None = None, profile_dir: str | None = None) -> None:
        self.backend = backend or PlaywrightBrowserBackend(profile_dir=profile_dir)

    async def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        action = str(command.get("action", "")).strip()
        session_id = str(command.get("session_id") or "default")
        params = command.get("params") or {}
        if not isinstance(params, dict):
            return {"ok": False, "error": "Browser command params must be an object"}
        if action not in _ALLOWED_ACTIONS:
            return {"ok": False, "error": f"Unknown browser action: {action}"}
        handler = getattr(self.backend, action, None)
        if handler is None:
            return {"ok": False, "error": f"Unknown browser action: {action}"}
        try:
            result = await handler(session_id, **params)
        except Exception as exc:
            return {"ok": False, "error": f"{action} failed: {exc}"}
        return result if isinstance(result, dict) else {"ok": True, "result": result}


class PlaywrightBrowserBackend:
    """Playwright implementation used inside the worker process."""

    def __init__(self, profile_dir: str | None = None) -> None:
        self.profile_dir = Path(profile_dir).expanduser() if profile_dir else _default_profile_dir()
        self._playwright: Any = None
        self._context: Any = None
        self._pages: dict[str, Any] = {}

    async def open(self, session_id: str, **params: Any) -> dict[str, Any]:
        url = str(params.get("url") or "").strip()
        if not url:
            return {"ok": False, "error": "url is required"}
        page = await self._page(session_id, new_tab=bool(params.get("new_tab", False)))
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        return await self._page_summary(page)

    async def click(self, session_id: str, **params: Any) -> dict[str, Any]:
        selector = str(params.get("selector") or "").strip()
        if not selector:
            return {"ok": False, "error": "selector is required"}
        page = await self._page(session_id)
        await page.locator(selector).click(timeout=30_000)
        return await self._page_summary(page)

    async def fill(self, session_id: str, **params: Any) -> dict[str, Any]:
        selector = str(params.get("selector") or "").strip()
        value = str(params.get("value") or "")
        if not selector:
            return {"ok": False, "error": "selector is required"}
        page = await self._page(session_id)
        await page.locator(selector).fill(value, timeout=30_000)
        return await self._page_summary(page)

    async def extract(self, session_id: str, **params: Any) -> dict[str, Any]:
        page = await self._page(session_id)
        max_chars = int(params.get("max_chars") or 6000)
        max_links = int(params.get("max_links") or 50)
        text = ""
        try:
            text = await page.locator("body").inner_text(timeout=10_000)
        except Exception:
            text = await page.text_content("body") or ""
        links = await page.locator("a").evaluate_all(
            """(els, maxLinks) => els.slice(0, maxLinks).map((a) => ({
                text: (a.innerText || a.textContent || '').trim(),
                url: a.href || ''
            })).filter((item) => item.url)""",
            max_links,
        )
        tables = await page.locator("table").evaluate_all(
            """(tables) => tables.slice(0, 10).map((table) =>
                Array.from(table.rows).slice(0, 50).map((row) =>
                    Array.from(row.cells).slice(0, 20).map((cell) =>
                        (cell.innerText || cell.textContent || '').trim()
                    )
                )
            )"""
        )
        return {
            "ok": True,
            "url": page.url,
            "title": await page.title(),
            "text": _truncate(text, max_chars),
            "links": links,
            "tables": tables,
        }

    async def scroll(self, session_id: str, **params: Any) -> dict[str, Any]:
        pixels = int(params.get("pixels") or 1000)
        page = await self._page(session_id)
        await page.evaluate("(pixels) => window.scrollBy(0, pixels)", pixels)
        return await self._page_summary(page)

    async def screenshot(self, session_id: str, **params: Any) -> dict[str, Any]:
        page = await self._page(session_id)
        path = params.get("path")
        if not path:
            root = Path.home() / ".hpagent" / "browser_artifacts"
            root.mkdir(parents=True, exist_ok=True)
            safe_session = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in session_id)
            path = root / f"{safe_session}_{int(time.time() * 1000)}.png"
        path = Path(str(path)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(path), full_page=bool(params.get("full_page", True)))
        return {"ok": True, "path": str(path), "url": page.url, "title": await page.title()}

    async def wait(self, session_id: str, **params: Any) -> dict[str, Any]:
        page = await self._page(session_id)
        timeout_ms = int(params.get("timeout_ms") or 5000)
        selector = str(params.get("selector") or "").strip()
        if selector:
            await page.wait_for_selector(selector, timeout=timeout_ms)
        else:
            await page.wait_for_timeout(timeout_ms)
        return await self._page_summary(page)

    async def _page(self, session_id: str, new_tab: bool = False) -> Any:
        await self._ensure_context()
        page = self._pages.get(session_id)
        if new_tab or page is None or page.is_closed():
            page = await self._context.new_page()
            page.set_default_timeout(30_000)
            self._pages[session_id] = page
        return page

    async def _ensure_context(self) -> None:
        if self._context is not None:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Install dependencies and run "
                "`playwright install chromium`."
            ) from exc
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=True,
            accept_downloads=True,
        )

    async def _page_summary(self, page: Any) -> dict[str, Any]:
        return {"ok": True, "url": page.url, "title": await page.title()}


def _default_profile_dir() -> Path:
    explicit = os.environ.get("HPD_BROWSER_PROFILE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    candidates = [
        Path.home() / ".config" / "google-chrome",
        Path.home() / ".config" / "chromium",
        Path.home() / ".config" / "microsoft-edge",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.home() / ".hpagent" / "browser_profile" / "default"


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


async def _serve(profile_dir: str | None = None) -> None:
    worker = BrowserWorker(profile_dir=profile_dir)
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            break
        try:
            command = json.loads(line)
            result = await worker.handle(command)
        except Exception as exc:
            result = {"ok": False, "error": f"worker error: {exc}"}
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="HPD browser worker")
    parser.add_argument("--profile-dir", default=None)
    args = parser.parse_args()
    asyncio.run(_serve(args.profile_dir))


if __name__ == "__main__":
    main()
