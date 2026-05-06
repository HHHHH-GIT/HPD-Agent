import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestInvokeWithToolsBudget(unittest.IsolatedAsyncioTestCase):
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
        tool.invoke.return_value = "ok"

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
        tool.invoke.return_value = "ok"

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
        tool.invoke.return_value = "ok"

        from src.llm.client import invoke_with_tools

        content, tool_log = await invoke_with_tools("prompt", tools=[tool], max_tool_calls=1)

        self.assertEqual(content, "best effort answer")
        self.assertEqual(tool_log, "")
        final_llm.ainvoke.assert_called_once()
