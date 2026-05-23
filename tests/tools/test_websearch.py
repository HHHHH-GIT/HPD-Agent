from __future__ import annotations

import pytest
import importlib


_SAMPLE_HTML = """
<html>
  <body>
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://example.com/a">Alpha &amp; Beta</a>
      <a class="result__snippet">First <b>snippet</b> text.</a>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="//example.org/b">Second Result</a>
      <div class="result__snippet">Second snippet.</div>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://example.net/c">Third Result</a>
      <div class="result__snippet">Third snippet.</div>
    </div>
  </body>
</html>
"""


@pytest.mark.asyncio
async def test_websearch_parses_results_and_limits_count(monkeypatch) -> None:
    from src.tools import websearch as websearch_tool
    from src.tools import websearch as exported_tool
    websearch_module = importlib.import_module("src.tools.websearch")

    monkeypatch.setattr(websearch_module, "_fetch_websearch_html", lambda query: _SAMPLE_HTML)

    result = await websearch_tool.ainvoke({"query": "alpha", "max_results": 2})

    assert exported_tool.name == "websearch"
    assert "1. Alpha & Beta" in result
    assert "https://example.com/a" in result
    assert "First snippet text." in result
    assert "2. Second Result" in result
    assert "https://example.org/b" in result
    assert "Third Result" not in result


@pytest.mark.asyncio
async def test_websearch_returns_error_for_empty_results(monkeypatch) -> None:
    websearch_module = importlib.import_module("src.tools.websearch")
    from src.tools import websearch

    monkeypatch.setattr(websearch_module, "_fetch_websearch_html", lambda query: "<html></html>")

    result = await websearch.ainvoke({"query": "missing", "max_results": 5})

    assert result == "[Error] No web search results found for: missing"


@pytest.mark.asyncio
async def test_websearch_returns_error_when_backend_fails(monkeypatch) -> None:
    websearch_module = importlib.import_module("src.tools.websearch")
    from src.tools import websearch

    def fail(_query: str) -> str:
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(websearch_module, "_fetch_websearch_html", fail)

    result = await websearch.ainvoke({"query": "python", "max_results": 5})

    assert result == "[Error] Web search failed: network unavailable"
