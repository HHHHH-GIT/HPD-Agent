from __future__ import annotations

import importlib

import pytest


class FakeBrowserService:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.extracts: dict[str, dict] = {}

    async def request(self, action: str, payload: dict) -> dict:
        self.requests.append((action, payload))
        if action == "open":
            return {"ok": True, "url": payload["url"], "title": f"title:{payload['url']}"}
        if action == "extract":
            url = self.requests[-2][1]["url"] if len(self.requests) >= 2 else "about:blank"
            return self.extracts.get(
                url,
                {
                    "ok": True,
                    "url": url,
                    "title": f"title:{url}",
                    "text": f"text for {url}",
                    "links": [],
                    "tables": [],
                },
            )
        return {"ok": True, "action": action}


@pytest.fixture()
def browser_module(monkeypatch):
    module = importlib.import_module("src.tools.browser")
    fake = FakeBrowserService()
    monkeypatch.setattr(module, "get_browser_service", lambda: fake)
    return module, fake


@pytest.mark.asyncio
async def test_browser_open_sends_action_to_worker_service(browser_module) -> None:
    module, fake = browser_module

    result = await module.browser_open.ainvoke(
        {"url": "https://example.com", "session_id": "s1"}
    )

    assert "https://example.com" in result
    assert fake.requests == [
        ("open", {"session_id": "s1", "url": "https://example.com", "new_tab": False})
    ]


@pytest.mark.asyncio
async def test_browser_click_blocks_high_risk_selector_without_confirmation(browser_module) -> None:
    module, fake = browser_module

    result = await module.browser_click.ainvoke(
        {"selector": "button.delete-account", "session_id": "s1"}
    )

    assert result.startswith("[ConfirmationRequired]")
    assert fake.requests == []


@pytest.mark.asyncio
async def test_browser_click_allows_high_risk_selector_with_confirmation(browser_module) -> None:
    module, fake = browser_module

    result = await module.browser_click.ainvoke(
        {
            "selector": "button.delete-account",
            "session_id": "s1",
            "confirm": True,
        }
    )

    assert "ok" in result.lower()
    assert fake.requests == [
        (
            "click",
            {
                "session_id": "s1",
                "selector": "button.delete-account",
                "confirm": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_crawl_site_visits_same_domain_pages_and_respects_limit(browser_module) -> None:
    module, fake = browser_module
    fake.extracts = {
        "https://example.com/": {
            "ok": True,
            "url": "https://example.com/",
            "title": "Home",
            "text": "home text",
            "links": [
                {"url": "https://example.com/a", "text": "A"},
                {"url": "https://external.test/b", "text": "External"},
            ],
            "tables": [],
        },
        "https://example.com/a": {
            "ok": True,
            "url": "https://example.com/a",
            "title": "A",
            "text": "page a text",
            "links": [],
            "tables": [],
        },
    }

    result = await module.crawl_site.ainvoke(
        {"start_url": "https://example.com/", "max_pages": 2, "session_id": "s1"}
    )

    opened_urls = [payload["url"] for action, payload in fake.requests if action == "open"]
    assert opened_urls == ["https://example.com/", "https://example.com/a"]
    assert "Home" in result
    assert "A" in result
    assert "external.test" not in result
