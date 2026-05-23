import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import importlib

from langchain_core.messages import AIMessageChunk


class TestInvokeWithToolsBudget(unittest.IsolatedAsyncioTestCase):
    @patch("src.llm.client.get_llm_with_tools")
    async def test_streams_final_answer_chunks(self, mock_get_llm_with_tools):
        llm = MagicMock()

        async def fake_stream(_messages):
            yield AIMessageChunk(content="hello ")
            yield AIMessageChunk(content="world")

        llm.astream = fake_stream
        mock_get_llm_with_tools.return_value = llm

        from src.llm.client import invoke_with_tools

        chunks: list[str] = []
        content, tool_log = await invoke_with_tools(
            "prompt",
            tools=[],
            stream=True,
            on_token=chunks.append,
        )

        self.assertEqual(content, "hello world")
        self.assertEqual(chunks, ["hello ", "world"])
        self.assertEqual(tool_log, "")

    @patch("src.llm.client.get_llm_with_tools")
    @patch("src.llm.client._token_callback._accumulate")
    async def test_does_not_manually_accumulate_usage_metadata(
        self, mock_accumulate, mock_get_llm_with_tools
    ):
        llm = AsyncMock()
        response = MagicMock()
        response.content = "ok"
        response.usage_metadata = {"input_tokens": 123, "output_tokens": 7}
        response.tool_calls = []
        llm.ainvoke.return_value = response
        mock_get_llm_with_tools.return_value = llm

        from src.llm.client import invoke_with_tools

        content, tool_log = await invoke_with_tools("prompt", tools=[])

        self.assertEqual(content, "ok")
        self.assertEqual(tool_log, "")
        mock_accumulate.assert_not_called()

    @patch("src.llm.client.get_llm_with_tools")
    async def test_estimates_llm_usage_when_api_usage_missing(self, mock_get_llm_with_tools):
        from src.core.observability import get_tracer

        llm = AsyncMock()
        response = MagicMock()
        response.content = "ok"
        response.tool_calls = []
        llm.ainvoke.return_value = response
        llm.model_name = "fake-model"
        mock_get_llm_with_tools.return_value = llm

        from src.llm.client import invoke_with_tools

        tracer = get_tracer()
        _ = tracer.end_trace()
        _ = tracer.start_trace(query="estimate", session_id="s")
        with tracer.span("direct") as span_id:
            await invoke_with_tools("prompt", tools=[])
            span = tracer.get_span(span_id)
        _ = tracer.end_trace()

        self.assertIsNotNone(span)
        self.assertGreater(span.tokens_in, 0)
        self.assertGreater(span.tokens_out, 0)
        self.assertEqual(span.token_source, "estimated")

    @patch("src.llm.client.get_llm_with_tools")
    async def test_raises_when_tool_call_budget_exceeded(self, mock_get_llm_with_tools):
        llm = AsyncMock()
        response = MagicMock()
        response.content = ""
        response.usage_metadata = {}
        response.tool_calls = [
            {"id": "1", "name": "read_file", "args": {"path": "a"}},
            {"id": "2", "name": "read_file", "args": {"path": "b"}},
        ]
        llm.ainvoke.return_value = response
        mock_get_llm_with_tools.return_value = llm

        tool = MagicMock()
        tool.name = "read_file"
        tool.ainvoke = AsyncMock(return_value="ok")

        from src.llm.client import invoke_with_tools

        with self.assertRaises(RuntimeError) as exc:
            await invoke_with_tools("prompt", tools=[tool], max_tool_calls=1, on_budget_exceeded="raise")

        self.assertIn("Tool budget exceeded", str(exc.exception))

    @patch("src.llm.client.get_llm_with_tools")
    async def test_raises_when_round_budget_exceeded(self, mock_get_llm_with_tools):
        llm = AsyncMock()
        response = MagicMock()
        response.content = ""
        response.usage_metadata = {}
        response.tool_calls = [
            {"id": "1", "name": "read_file", "args": {"path": "a"}},
        ]
        llm.ainvoke.return_value = response
        mock_get_llm_with_tools.return_value = llm

        tool = MagicMock()
        tool.name = "read_file"
        tool.ainvoke = AsyncMock(return_value="ok")

        from src.llm.client import invoke_with_tools

        with self.assertRaises(RuntimeError) as exc:
            await invoke_with_tools("prompt", tools=[tool], max_rounds=2, max_tool_calls=10, on_budget_exceeded="raise")

        self.assertIn("max rounds", str(exc.exception))

    @patch("src.llm.client.get_llm")
    @patch("src.llm.client.get_llm_with_tools")
    async def test_finalizes_when_tool_budget_exceeded_by_default(self, mock_get_llm_with_tools, mock_get_llm):
        tool_llm = AsyncMock()
        first_response = MagicMock()
        first_response.content = ""
        first_response.usage_metadata = {}
        first_response.tool_calls = [
            {"id": "1", "name": "read_file", "args": {"path": "a"}},
            {"id": "2", "name": "read_file", "args": {"path": "b"}},
        ]
        tool_llm.ainvoke.return_value = first_response
        mock_get_llm_with_tools.return_value = tool_llm

        final_llm = AsyncMock()
        final_response = MagicMock()
        final_response.content = "best effort answer"
        final_llm.ainvoke.return_value = final_response
        mock_get_llm.return_value = final_llm

        tool = MagicMock()
        tool.name = "read_file"
        tool.ainvoke = AsyncMock(return_value="ok")

        from src.llm.client import invoke_with_tools

        content, tool_log = await invoke_with_tools("prompt", tools=[tool], max_tool_calls=1)

        self.assertEqual(content, "best effort answer")
        self.assertEqual(tool_log, "")
        final_llm.ainvoke.assert_called_once()

    @patch("src.llm.client.get_llm_with_tools")
    async def test_invoke_with_tools_can_call_websearch_tool(self, mock_get_llm_with_tools):
        websearch_module = importlib.import_module("src.tools.websearch")
        from src.tools import websearch

        mock_get_llm_with_tools.return_value = AsyncMock()
        first_response = MagicMock()
        first_response.content = ""
        first_response.usage_metadata = {}
        first_response.tool_calls = [
            {
                "id": "web-1",
                "name": "websearch",
                "args": {"query": "alpha", "max_results": 1},
            }
        ]
        second_response = MagicMock()
        second_response.content = "searched"
        second_response.usage_metadata = {}
        second_response.tool_calls = []
        mock_get_llm_with_tools.return_value.ainvoke.side_effect = [
            first_response,
            second_response,
        ]
        monkeypatch_html = """
        <a rel="nofollow" class="result__a" href="https://example.com/a">Alpha</a>
        <a class="result__snippet">Alpha snippet.</a>
        """

        with patch.object(
            websearch_module,
            "_fetch_websearch_html",
            return_value=monkeypatch_html,
        ):
            from src.llm.client import invoke_with_tools

            content, tool_log = await invoke_with_tools("prompt", tools=[websearch])

        self.assertEqual(content, "searched")
        self.assertIn("Alpha", tool_log)
        self.assertIn("https://example.com/a", tool_log)

    @patch("src.llm.client.get_llm_with_tools")
    async def test_invoke_with_tools_can_call_browser_open_tool(self, mock_get_llm_with_tools):
        from src.tools import browser_open
        import src.tools.browser as browser_tools

        class FakeBrowserService:
            async def request(self, action, payload):
                return {"ok": True, "action": action, "url": payload["url"], "title": "Example"}

        tool_llm = AsyncMock()
        first_response = MagicMock()
        first_response.content = ""
        first_response.usage_metadata = {}
        first_response.tool_calls = [
            {
                "id": "browser-1",
                "name": "browser_open",
                "args": {"url": "https://example.com", "session_id": "s1"},
            }
        ]
        second_response = MagicMock()
        second_response.content = "opened"
        second_response.usage_metadata = {}
        second_response.tool_calls = []
        tool_llm.ainvoke.side_effect = [first_response, second_response]
        mock_get_llm_with_tools.return_value = tool_llm

        with patch.object(
            browser_tools,
            "get_browser_service",
            return_value=FakeBrowserService(),
        ):
            from src.llm.client import invoke_with_tools

            content, tool_log = await invoke_with_tools("prompt", tools=[browser_open])

        self.assertEqual(content, "opened")
        self.assertIn("browser_open", tool_log)
        self.assertIn("https://example.com", tool_log)

    @patch("src.llm.client.get_llm_with_tools")
    async def test_riskless_mode_skips_terminal_confirmation_for_normal_command(
        self, mock_get_llm_with_tools
    ):
        tool_llm = AsyncMock()
        first_response = MagicMock()
        first_response.content = ""
        first_response.usage_metadata = {}
        first_response.tool_calls = [
            {
                "id": "terminal-1",
                "name": "terminal",
                "args": {"cmd": "pytest tests/tools/test_terminal_policy.py"},
            }
        ]
        second_response = MagicMock()
        second_response.content = "done"
        second_response.usage_metadata = {}
        second_response.tool_calls = []
        tool_llm.ainvoke.side_effect = [first_response, second_response]
        mock_get_llm_with_tools.return_value = tool_llm

        renderer = MagicMock()
        renderer.confirm.return_value = False
        tool = MagicMock()
        tool.name = "terminal"
        tool.ainvoke = AsyncMock(return_value="passed")

        with patch("src.llm.client.get_renderer", return_value=renderer), patch(
            "src.llm.client.is_riskless_enabled",
            return_value=True,
        ):
            from src.llm.client import invoke_with_tools

            content, tool_log = await invoke_with_tools("prompt", tools=[tool])

        self.assertEqual(content, "done")
        self.assertIn("passed", tool_log)
        renderer.confirm.assert_not_called()
        tool.ainvoke.assert_awaited_once()

    @patch("src.llm.client.get_llm_with_tools")
    async def test_riskless_mode_still_confirms_extreme_terminal_command(
        self, mock_get_llm_with_tools
    ):
        tool_llm = AsyncMock()
        first_response = MagicMock()
        first_response.content = ""
        first_response.usage_metadata = {}
        first_response.tool_calls = [
            {
                "id": "terminal-1",
                "name": "terminal",
                "args": {"cmd": "git push --force origin main"},
            }
        ]
        second_response = MagicMock()
        second_response.content = "done"
        second_response.usage_metadata = {}
        second_response.tool_calls = []
        tool_llm.ainvoke.side_effect = [first_response, second_response]
        mock_get_llm_with_tools.return_value = tool_llm

        renderer = MagicMock()
        renderer.confirm.return_value = False
        tool = MagicMock()
        tool.name = "terminal"
        tool.ainvoke = AsyncMock(return_value="should not run")

        with patch("src.llm.client.get_renderer", return_value=renderer), patch(
            "src.llm.client.is_riskless_enabled",
            return_value=True,
        ):
            from src.llm.client import invoke_with_tools

            _content, tool_log = await invoke_with_tools("prompt", tools=[tool])

        self.assertIn("[Cancelled]", tool_log)
        renderer.confirm.assert_called_once()
        tool.ainvoke.assert_not_awaited()
