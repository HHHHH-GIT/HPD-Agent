from __future__ import annotations

import pytest


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def open(self, session_id: str, **params) -> dict:
        self.calls.append(("open", session_id, params))
        return {"ok": True, "url": params["url"], "title": "opened"}

    async def extract(self, session_id: str, **params) -> dict:
        self.calls.append(("extract", session_id, params))
        return {"ok": True, "title": "page", "text": "body", "links": [], "tables": []}


@pytest.mark.asyncio
async def test_browser_worker_routes_actions_to_backend_and_preserves_session() -> None:
    from src.browser.worker import BrowserWorker

    backend = FakeBackend()
    worker = BrowserWorker(backend=backend)

    opened = await worker.handle(
        {"action": "open", "session_id": "session-a", "params": {"url": "https://example.com"}}
    )
    extracted = await worker.handle(
        {"action": "extract", "session_id": "session-a", "params": {}}
    )

    assert opened["ok"] is True
    assert extracted["ok"] is True
    assert backend.calls == [
        ("open", "session-a", {"url": "https://example.com"}),
        ("extract", "session-a", {}),
    ]


@pytest.mark.asyncio
async def test_browser_worker_returns_error_for_unknown_action() -> None:
    from src.browser.worker import BrowserWorker

    worker = BrowserWorker(backend=FakeBackend())

    result = await worker.handle(
        {"action": "unknown", "session_id": "session-a", "params": {}}
    )

    assert result["ok"] is False
    assert result["error"] == "Unknown browser action: unknown"
