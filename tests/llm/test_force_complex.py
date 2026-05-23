import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, "/root/projects/evo_agent")

from src.core.enums import TaskDifficulty
from src.memory.context import ConversationContext


def _mock_tracer():
    tracer = MagicMock()
    span_ctx = MagicMock()
    span_ctx.__enter__ = MagicMock(return_value="span-123")
    span_ctx.__exit__ = MagicMock(return_value=False)
    tracer.span.return_value = span_ctx
    return tracer


class TestForceComplexHelpers(unittest.TestCase):
    def test_parse_query_control_flags_detects_trailing_ampersand(self) -> None:
        from src.agents.query_agent import parse_query_control_flags

        query, forced = parse_query_control_flags("分析一下这个项目 &")
        self.assertEqual(query, "分析一下这个项目")
        self.assertTrue(forced)

    def test_parse_query_control_flags_allows_escaped_ampersand(self) -> None:
        from src.agents.query_agent import parse_query_control_flags

        query, forced = parse_query_control_flags("A literal ampersand \\&")
        self.assertEqual(query, "A literal ampersand &")
        self.assertFalse(forced)


class TestForcedAssessment(unittest.IsolatedAsyncioTestCase):
    @patch("src.nodes.assessment.TokenTrackerCallback.snapshot", return_value=(0, 0, ""))
    @patch("src.nodes.assessment.get_tracer")
    @patch("src.nodes.assessment.get_structured_llm")
    async def test_first_level_assessment_bypasses_llm_when_forced(
        self,
        mock_get_llm,
        mock_tracer,
        mock_snapshot,
    ) -> None:
        mock_tracer.return_value = _mock_tracer()

        from src.nodes.assessment import first_level_assessment

        state = {
            "input": "分析一下这个项目",
            "force_complex": True,
            "conversation_history": ConversationContext(),
            "outputs": [],
        }

        result = await first_level_assessment(state)

        self.assertEqual(result["analysis"], TaskDifficulty.COMPLEX)
        self.assertTrue(result["outputs"][-1].result["forced"])
        mock_get_llm.assert_not_called()


class TestQueryAgentForceComplex(unittest.IsolatedAsyncioTestCase):
    @patch("src.agents.query_agent.build_boot_prompt", return_value="boot")
    async def test_ainvoke_passes_force_complex_into_initial_state(self, mock_boot_prompt) -> None:
        from src.agents.query_agent import QueryAgent

        agent = QueryAgent.__new__(QueryAgent)
        agent._contexts = {}
        agent._current_session = "default"
        agent._session_boot_done = set()
        agent._auto_save_enabled = False
        agent._project_hash = "test-project"

        captured: dict = {}

        async def fake_app_ainvoke(initial_state, config=None):
            captured["state"] = initial_state
            captured["config"] = config
            return {
                "outputs": [],
                "sub_task_outputs": [],
                "final_response": "",
                "synthesis_prompt": "",
            }

        agent._app = MagicMock()
        agent._app.ainvoke = fake_app_ainvoke

        await agent.ainvoke("分析一下这个项目 &", thread_id="default")

        self.assertEqual(captured["state"]["input"], "分析一下这个项目")
        self.assertTrue(captured["state"]["force_complex"])
        ctx = agent._get_context("default")
        self.assertEqual(ctx.messages[1].content, "分析一下这个项目")
