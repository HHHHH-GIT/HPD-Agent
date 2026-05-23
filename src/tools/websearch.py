"""Web search tool backed by DuckDuckGo HTML results."""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str


@tool
def websearch(query: str, max_results: int = 5) -> str:
    """Search the web and return concise result summaries.

    Args:
        query: Search query.
        max_results: Maximum number of search results to return.
    """
    query = (query or "").strip()
    if not query:
        return "[Error] Web search query cannot be empty"

    try:
        limit = max(1, min(int(max_results), 10))
        html_text = _fetch_websearch_html(query)
        results = _parse_duckduckgo_results(html_text, limit)
    except Exception as exc:
        return f"[Error] Web search failed: {exc}"

    if not results:
        return f"[Error] No web search results found for: {query}"

    return _format_results(query, results)


def _fetch_websearch_html(query: str) -> str:
    """Fetch DuckDuckGo's no-JS HTML search page."""
    endpoint = "https://html.duckduckgo.com/html/"
    url = f"{endpoint}?{urlencode({'q': query})}"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; HPD-Agent/0.1; "
                "+https://github.com/hpd-agent)"
            )
        },
    )
    with urlopen(request, timeout=10) as response:
        data = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return data.decode(charset, errors="replace")


def _parse_duckduckgo_results(html_text: str, max_results: int) -> list[WebSearchResult]:
    """Parse result title, url, and snippet from DuckDuckGo HTML."""
    link_pattern = re.compile(
        r'<a\b[^>]*class=["\'][^"\']*\bresult__a\b[^"\']*["\'][^>]*'
        r'href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<(?:a|div|span)\b[^>]*class=["\'][^"\']*\bresult__snippet\b[^"\']*["\'][^>]*>'
        r"(.*?)</(?:a|div|span)>",
        re.IGNORECASE | re.DOTALL,
    )

    matches = list(link_pattern.finditer(html_text))
    results: list[WebSearchResult] = []
    for index, match in enumerate(matches):
        href = match.group(1)
        title = _clean_html_text(match.group(2))
        if not title:
            continue

        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(html_text)
        window = html_text[match.end() : next_start]
        snippet_match = snippet_pattern.search(window)
        snippet = _clean_html_text(snippet_match.group(1)) if snippet_match else ""
        url = _normalize_result_url(href)
        if not url:
            continue

        results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


def _format_results(query: str, results: list[WebSearchResult]) -> str:
    lines = [f"Web search results for: {query}"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}")
        lines.append(f"   URL: {result.url}")
        if result.snippet:
            lines.append(f"   Snippet: {result.snippet}")
    return "\n".join(lines)


def _clean_html_text(raw: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", raw)
    text = html_lib.unescape(without_tags)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_result_url(raw_url: str) -> str:
    url = html_lib.unescape(raw_url)
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "uddg" in params and params["uddg"]:
        return unquote(params["uddg"][0])
    if url.startswith("//"):
        return "https:" + url
    if parsed.scheme:
        return url
    return urljoin("https://duckduckgo.com", url)
