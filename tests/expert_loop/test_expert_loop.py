"""Test suite for the ExpertAgent TOT multi-path + evaluate→reflect loop.

Verifies:
  1. Evaluator — 1:1 scoring, score clamping
  2. Reflector — reflection result generation
  3. Rewriter — prompt angle generation
  4. Execution — easy vs hard branching, expert loop, multi-path
"""

import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, "/root/projects/evo_agent")

from src.core.models import (
    RewriteResult,
    CandidateResult,
    EvaluatorScore,
    ReflectionResult,
    SubTaskAssessmentResult,
    SubTaskOutput,
)
from src.core.enums import SubTaskDifficulty


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _mock_tracer():
    """Create a mock tracer that works as a context manager."""
    tracer = MagicMock()
    span_ctx = MagicMock()
    span_ctx.__enter__ = MagicMock(return_value="span-123")
    span_ctx.__exit__ = MagicMock(return_value=False)
    tracer.span.return_value = span_ctx
    return tracer


def _mock_token_snapshot():
    """Mock TokenTrackerCallback.snapshot to return zeros."""
    return (0, 0, "test-model")


# --------------------------------------------------------------------------
# 1. Evaluator
# --------------------------------------------------------------------------


class TestEvaluator(unittest.IsolatedAsyncioTestCase):
    """Tests for the evaluator node (1:1 scoring)."""

    @patch("src.nodes.evaluator.TokenTrackerCallback.snapshot", return_value=(0, 0, ""))
    @patch("src.nodes.evaluator.get_tracer")
    @patch("src.nodes.evaluator.get_structured_llm")
    async def test_evaluate_returns_score(self, mock_get_llm, mock_tracer, mock_token):
        """Evaluator returns a valid EvaluatorScore."""
        mock_tracer.return_value = _mock_tracer()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = EvaluatorScore(
            score=0.85, reasoning="Good quality", issues=[]
        )
        mock_get_llm.return_value = mock_llm

        from src.nodes.evaluator import evaluate_single
        result = await evaluate_single("some detail", "test task", "test context")

        self.assertAlmostEqual(result.score, 0.85)
        self.assertEqual(result.issues, [])

    @patch("src.nodes.evaluator.TokenTrackerCallback.snapshot", return_value=(0, 0, ""))
    @patch("src.nodes.evaluator.get_tracer")
    @patch("src.nodes.evaluator.get_structured_llm")
    async def test_evaluate_clamps_score(self, mock_get_llm, mock_tracer, mock_token):
        """Evaluator clamps score to [0.0, 1.0]."""
        mock_tracer.return_value = _mock_tracer()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = EvaluatorScore(
            score=1.5, reasoning="over", issues=[]
        )
        mock_get_llm.return_value = mock_llm

        from src.nodes.evaluator import evaluate_single
        result = await evaluate_single("detail", "task", "ctx")
        self.assertAlmostEqual(result.score, 1.0)

    @patch("src.nodes.evaluator.TokenTrackerCallback.snapshot", return_value=(0, 0, ""))
    @patch("src.nodes.evaluator.get_tracer")
    @patch("src.nodes.evaluator.get_structured_llm")
    async def test_evaluate_with_issues(self, mock_get_llm, mock_tracer, mock_token):
        """Evaluator returns issues list for low-quality results."""
        mock_tracer.return_value = _mock_tracer()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = EvaluatorScore(
            score=0.3,
            reasoning="Incomplete",
            issues=["Missing analysis", "No conclusion"],
        )
        mock_get_llm.return_value = mock_llm

        from src.nodes.evaluator import evaluate_single
        result = await evaluate_single("weak detail", "task", "ctx")

        self.assertAlmostEqual(result.score, 0.3)
        self.assertEqual(len(result.issues), 2)


# --------------------------------------------------------------------------
# 2. Reflector
# --------------------------------------------------------------------------


class TestReflector(unittest.IsolatedAsyncioTestCase):
    """Tests for the reflector node."""

    @patch("src.nodes.reflector.TokenTrackerCallback.snapshot", return_value=(0, 0, ""))
    @patch("src.nodes.reflector.get_tracer")
    @patch("src.nodes.reflector.get_structured_llm")
    async def test_reflect_returns_improvement(self, mock_get_llm, mock_tracer, mock_token):
        """Reflector returns a ReflectionResult with improved_prompt."""
        mock_tracer.return_value = _mock_tracer()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = ReflectionResult(
            improved_prompt="Focus on edge cases",
            strategy="Add edge case analysis",
            reasoning="Original missed boundary conditions",
        )
        mock_get_llm.return_value = mock_llm

        from src.nodes.reflector import reflect
        score = EvaluatorScore(score=0.4, reasoning="weak", issues=["Missing edge cases"])
        result = await reflect("task", "ctx", "detail", score)

        self.assertIn("edge case", result.strategy.lower())
        self.assertIn("edge case", result.improved_prompt.lower())


# --------------------------------------------------------------------------
# 3. Rewriter
# --------------------------------------------------------------------------


class TestRewriter(unittest.IsolatedAsyncioTestCase):
    """Tests for the rewriter node."""

    @patch("src.nodes.rewriter.TokenTrackerCallback.snapshot", return_value=(0, 0, ""))
    @patch("src.nodes.rewriter.get_tracer")
    @patch("src.nodes.rewriter.get_structured_llm")
    async def test_rewrite_returns_angles(self, mock_get_llm, mock_tracer, mock_token):
        """Rewriter returns N angles."""
        mock_tracer.return_value = _mock_tracer()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = RewriteResult(
            angles=["角度1", "角度2", "角度3"],
            reasoning="diverse perspectives",
        )
        mock_get_llm.return_value = mock_llm

        from src.nodes.rewriter import rewrite_prompt
        result = await rewrite_prompt(1, "test task", "ctx", n=3)

        self.assertEqual(len(result.angles), 3)


# --------------------------------------------------------------------------
# 4. Execution — easy vs hard branching
# --------------------------------------------------------------------------


class TestExecutionBranching(unittest.IsolatedAsyncioTestCase):
    """Tests for the execute() function's easy/hard branching."""

    @patch("src.nodes.execution._extract_key_findings_llm", return_value=[])
    @patch("src.nodes.execution.TokenTrackerCallback.snapshot", return_value=(0, 0, ""))
    @patch("src.nodes.execution.get_tracer")
    @patch("src.nodes.execution.get_structured_llm")
    @patch("src.nodes.execution.get_llm")
    async def test_easy_task_single_call_no_tools(
        self, mock_get_llm, mock_get_structured, mock_tracer, mock_token, mock_kf
    ):
        """Easy tasks use single LLM call without tools."""
        mock_tracer.return_value = _mock_tracer()

        # First call: assessment (returns easy)
        assessment_llm = AsyncMock()
        assessment_llm.ainvoke.return_value = SubTaskAssessmentResult(
            difficulty=SubTaskDifficulty.EASY, reasoning="simple"
        )

        # Second call: execution (returns content)
        exec_llm = AsyncMock()
        exec_response = MagicMock()
        exec_response.content = '{"detail": "answer", "summary": "done"}'
        exec_llm.ainvoke.return_value = exec_response

        # get_structured_llm is called for assessment
        mock_get_structured.return_value = assessment_llm
        # get_llm is called for execution
        mock_get_llm.return_value = exec_llm

        from src.nodes.execution import execute
        result = await execute(1, "simple task", "ctx")

        self.assertFalse(result.expert_mode)
        self.assertEqual(result.tools_used, [])
        self.assertEqual(result.tool_log, "")

    @patch("src.nodes.execution._expert_loop")
    @patch("src.nodes.execution.TokenTrackerCallback.snapshot", return_value=(0, 0, ""))
    @patch("src.nodes.execution.get_tracer")
    @patch("src.nodes.execution.get_structured_llm")
    async def test_hard_task_enters_expert_loop(
        self, mock_get_structured, mock_tracer, mock_token, mock_expert
    ):
        """Hard tasks enter the expert loop."""
        mock_tracer.return_value = _mock_tracer()
        mock_get_structured.return_value = AsyncMock(
            ainvoke=AsyncMock(return_value=SubTaskAssessmentResult(
                difficulty=SubTaskDifficulty.HARD, reasoning="complex"
            ))
        )
        mock_expert.return_value = "expert result"

        from src.nodes.execution import execute
        result = await execute(1, "hard task", "ctx")

        mock_expert.assert_called_once()


# --------------------------------------------------------------------------
# 5. Multi-path candidate execution
# --------------------------------------------------------------------------


class TestCandidateExecution(unittest.IsolatedAsyncioTestCase):
    """Tests for _execute_candidate."""

    @patch("src.nodes.execution.get_llm")
    async def test_candidate_parses_json(self, mock_get_llm):
        """Candidate parses JSON response correctly."""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = '{"detail": "analysis result", "summary": "found X"}'
        mock_llm.ainvoke.return_value = response
        mock_get_llm.return_value = mock_llm

        from src.nodes.execution import _execute_candidate
        result = await _execute_candidate(0, "angle1", 1, "task", "ctx")

        self.assertEqual(result.variation_index, 0)
        self.assertEqual(result.prompt_angle, "angle1")
        self.assertEqual(result.detail, "analysis result")
        self.assertEqual(result.summary, "found X")

    @patch("src.nodes.execution.get_llm")
    async def test_candidate_handles_non_json(self, mock_get_llm):
        """Candidate handles non-JSON response gracefully."""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = "This is a plain text answer."
        mock_llm.ainvoke.return_value = response
        mock_get_llm.return_value = mock_llm

        from src.nodes.execution import _execute_candidate
        result = await _execute_candidate(1, "angle2", 1, "task", "ctx")

        self.assertEqual(result.detail, "This is a plain text answer.")
        # summary should be extracted from the text
        self.assertTrue(len(result.summary) > 0)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
