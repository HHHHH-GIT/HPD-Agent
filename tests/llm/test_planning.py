import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, "/root/projects/evo_agent")

from src.core.models import PlannerResult, RewriteResult, SubTask


def _mock_tracer():
    tracer = MagicMock()
    span_ctx = MagicMock()
    span_ctx.__enter__ = MagicMock(return_value="span-123")
    span_ctx.__exit__ = MagicMock(return_value=False)
    tracer.span.return_value = span_ctx
    return tracer


class TestPlanningHelpers(unittest.TestCase):
    def test_extract_parallel_variant_count_from_chinese_query(self) -> None:
        from src.nodes.planning import _extract_parallel_variant_count

        count = _extract_parallel_variant_count("帮我分析一下接雨水这个题，至少给出三种不同解法")
        self.assertEqual(count, 3)

    def test_extract_parallel_variant_count_returns_none_for_non_variant_query(self) -> None:
        from src.nodes.planning import _extract_parallel_variant_count

        count = _extract_parallel_variant_count("帮我分析一下这个项目的目录结构")
        self.assertIsNone(count)


class TestDecomposeParallelExpansion(unittest.IsolatedAsyncioTestCase):
    @patch("src.nodes.planning.TokenTrackerCallback.snapshot", return_value=(0, 0, ""))
    @patch("src.nodes.planning.get_tracer")
    @patch("src.nodes.planning.check_circle", return_value=False)
    @patch("src.nodes.planning.get_structured_llm")
    async def test_decompose_expands_single_task_into_parallel_variants(
        self,
        mock_get_llm,
        mock_check_circle,
        mock_tracer,
        mock_snapshot,
    ) -> None:
        mock_tracer.return_value = _mock_tracer()

        planner_llm = AsyncMock()
        planner_llm.ainvoke.return_value = PlannerResult(
            total_sub_task_count=1,
            sub_tasks=[
                SubTask(
                    id=1,
                    name="分析接雨水问题并给出至少三种不同解法",
                    depends=[],
                )
            ],
            reasoning="先整体分析问题。",
        )

        rewrite_llm = AsyncMock()
        rewrite_llm.ainvoke.return_value = RewriteResult(
            angles=["前后缀最大值数组", "双指针收缩", "单调栈"],
            reasoning="三种经典路线。",
        )

        mock_get_llm.side_effect = [planner_llm, rewrite_llm]

        from src.nodes.planning import decompose

        tasks, result = await decompose("帮我分析一下接雨水这个题，至少给出三种不同解法")

        self.assertEqual(len(tasks), 3)
        self.assertEqual(result.total_sub_task_count, 3)
        self.assertTrue(all(task.depends == [] for task in tasks))
        self.assertIn("前后缀最大值数组", tasks[0].name)
        self.assertIn("双指针收缩", tasks[1].name)
        self.assertIn("单调栈", tasks[2].name)
        self.assertIn("展开为 3 个可并行执行的独立子任务", result.reasoning)
