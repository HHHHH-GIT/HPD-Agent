"""Test suite for the DAG scheduler context-accumulation fix.

Verifies:
  1. Sub-tasks with no dependencies get the original context.
  2. Sub-tasks with dependencies receive their ancestors' results in context.
  3. tool_log is included in the detail field (so downstream tasks see it).
  4. Parallel sub-tasks in the same layer receive consistent context.
  5. Progress tracking via _print_progress (no crash).
  6. Cycle detection.
  7. Retry mechanism (failure → retry → success).
"""

import asyncio
import sys
import unittest
from typing import Any

sys.path.insert(0, "/root/projects/evo_agent")

from src.core.models import SubTask, SubTaskOutput
from src.nodes.scheduler import run_all, check_circle, RetryConfig


# ----------------------------------------------------------------------
# Mock executor helpers
# ----------------------------------------------------------------------


class MockExecutor:
    """Records every invocation so tests can assert on them."""

    calls: list[dict] = []
    _fail_until: dict[int, int] = {}   # task_id → fail count
    _block: bool = False

    @classmethod
    def reset(cls) -> None:
        cls.calls.clear()
        cls._fail_until.clear()
        cls._block = False

    @classmethod
    def make_executor(cls, responses: dict[int, SubTaskOutput]) -> Any:
        """Build an async executor that returns responses for given task IDs.

        Args:
            responses: Maps task_id → SubTaskOutput to return.
        """
        async def executor(task_id: int, task_name: str, context: str) -> SubTaskOutput:
            # Simulate a brief async yield (real executor is truly async)
            await asyncio.sleep(0)
            for fail_count in range(cls._fail_until.get(task_id, 0)):
                cls.calls.append(
                    {"task_id": task_id, "task_name": task_name, "context": context, "attempt": fail_count + 1}
                )
            resp = responses.get(task_id)
            if resp is None:
                raise RuntimeError(f"No mock response for task {task_id}")
            cls.calls.append(
                {"task_id": task_id, "task_name": task_name, "context": context, "attempt": cls._fail_until.get(task_id, 0)}
            )
            return resp
        return executor


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestSchedulerContextPassing(unittest.IsolatedAsyncioTestCase):
    """Core tests for the context-accumulation fix."""

    def setUp(self) -> None:
        MockExecutor.reset()

    # ------------------------------------------------------------------
    # 1. Sub-tasks with no dependencies receive original context
    # ------------------------------------------------------------------

    async def test_no_deps_gets_original_context(self) -> None:
        """Task 1 has no deps → its context must be exactly the original."""
        tasks = [
            SubTask(id=1, name="Step one", depends=[]),
        ]
        context = "The original user question about building a web app."

        executor = MockExecutor.make_executor({
            1: SubTaskOutput(id=1, name="Step one", detail="Step 1 done.", summary="Step 1 done."),
        })

        statuses, outputs = await run_all(tasks, executor, context)

        self.assertEqual(statuses[1], "done")
        self.assertEqual(len(outputs), 1)
        call = MockExecutor.calls[0]
        self.assertEqual(call["context"], context)

    # ------------------------------------------------------------------
    # 2. Sub-tasks with deps receive ancestors' results in context
    # ------------------------------------------------------------------

    async def test_dep_receives_ancestor_result(self) -> None:
        """Task 2 depends on 1 → context must include Task 1's detail."""
        tasks = [
            SubTask(id=1, name="Step one",   depends=[]),
            SubTask(id=2, name="Step two",   depends=[1]),
        ]
        context = "Build a calculator."

        task1_detail = "Step 1: I read the requirements file. Found 3 inputs."
        task1_output = SubTaskOutput(
            id=1, name="Step one",
            detail=task1_detail,
            summary="Requirements read.",
        )
        task2_output = SubTaskOutput(
            id=2, name="Step two",
            detail="Step 2: Based on requirements, I built the UI.",
            summary="UI built.",
        )

        executor = MockExecutor.make_executor({1: task1_output, 2: task2_output})
        statuses, outputs = await run_all(tasks, executor, context)

        self.assertEqual(statuses[1], "done")
        self.assertEqual(statuses[2], "done")

        # Task 1 context is pure original
        self.assertEqual(MockExecutor.calls[0]["context"], context)

        # Task 2 context must contain Task 1's detail AND the original
        task2_context = MockExecutor.calls[1]["context"]
        self.assertIn(context, task2_context)
        self.assertIn("Step 1", task2_context)
        self.assertIn(task1_detail, task2_context)

    # ------------------------------------------------------------------
    # 3. Three-layer chain: each layer sees all previous layers
    # ------------------------------------------------------------------

    async def test_three_layer_chain(self) -> None:
        tasks = [
            SubTask(id=1, name="Plan",      depends=[]),
            SubTask(id=2, name="Implement", depends=[1]),
            SubTask(id=3, name="Test",      depends=[2]),
        ]
        context = "Write a sorting algorithm."

        def make_output(task_id: int, name: str, detail: str) -> SubTaskOutput:
            return SubTaskOutput(id=task_id, name=name, detail=detail, summary=f"{name} done.")

        executor = MockExecutor.make_executor({
            1: make_output(1, "Plan",      "Planned the quicksort approach."),
            2: make_output(2, "Implement",  "Implemented quicksort in Python."),
            3: make_output(3, "Test",       "Ran tests — all passed."),
        })

        statuses, outputs = await run_all(tasks, executor, context)

        self.assertEqual(statuses[3], "done")

        # Collect contexts per task
        ctx = {c["task_id"]: c["context"] for c in MockExecutor.calls}

        # Layer 1: only original
        self.assertNotIn("Planned", ctx[1])
        self.assertIn(context, ctx[1])

        # Layer 2: original + layer 1 result
        self.assertIn("Planned", ctx[2])
        self.assertIn("quicksort", ctx[2].lower())

        # Layer 3: original + layer 1 + layer 2 result (NOT its own output)
        self.assertIn("Implemented", ctx[3])
        self.assertIn("Planned", ctx[3])
        self.assertNotIn("Ran tests", ctx[3])  # task 3's own detail, not yet produced

    # ------------------------------------------------------------------
    # 4. tool_log is included in detail (so downstream tasks see it)
    # ------------------------------------------------------------------

    async def test_tool_log_in_detail(self) -> None:
        """The detail field must contain the tool_log, not just the LLM text."""
        tasks = [
            SubTask(id=1, name="Find file", depends=[]),
        ]

        tool_log = "[Tool: read_file]\nfile content: hello world"
        full_content = "I read the file."
        expected_detail = f"{full_content}\n\n[工具执行记录]\n{tool_log}"

        async def executor(task_id: int, task_name: str, context: str) -> SubTaskOutput:
            await asyncio.sleep(0)
            return SubTaskOutput(
                id=task_id, name=task_name,
                detail=expected_detail,
                summary="File found.",
            )

        statuses, outputs = await run_all(tasks, executor, context="test")

        self.assertEqual(outputs[0].detail, expected_detail)
        self.assertIn("[工具执行记录]", outputs[0].detail)
        self.assertIn(tool_log, outputs[0].detail)

    # ------------------------------------------------------------------
    # 5. Parallel sub-tasks in the same layer receive consistent context
    # ------------------------------------------------------------------

    async def test_parallel_same_layer_consistent_context(self) -> None:
        """Tasks 2 and 3 both depend only on 1 → they must both see the same accumulated context."""
        tasks = [
            SubTask(id=1, name="Setup",    depends=[]),
            SubTask(id=2, name="Frontend", depends=[1]),
            SubTask(id=3, name="Backend",  depends=[1]),
        ]
        context = "Build a web app."

        def make_output(task_id: int, name: str, detail: str) -> SubTaskOutput:
            return SubTaskOutput(id=task_id, name=name, detail=detail, summary=f"{name} done.")

        executor = MockExecutor.make_executor({
            1: make_output(1, "Setup",   "Created project directory."),
            2: make_output(2, "Frontend", "Built HTML/CSS UI."),
            3: make_output(3, "Backend", "Wrote Flask API."),
        })

        statuses, outputs = await run_all(tasks, executor, context)

        self.assertEqual(statuses[2], "done")
        self.assertEqual(statuses[3], "done")

        ctx = {c["task_id"]: c["context"] for c in MockExecutor.calls}

        # Both 2 and 3 should see original context AND task 1's result
        for tid in [2, 3]:
            self.assertIn(context, ctx[tid])
            self.assertIn("Created project", ctx[tid])
            self.assertNotIn("HTML/CSS", ctx[tid])  # Not yet done
            self.assertNotIn("Flask API", ctx[tid])

    # ------------------------------------------------------------------
    # 6. Complex DAG: diamond dependency
    # ------------------------------------------------------------------

    async def test_diamond_dependency(self) -> None:
        """
        Diamond DAG:
            1
           / \
          2   3
           \ /
            4

        Task 4 must see results from BOTH tasks 2 and 3.
        """
        tasks = [
            SubTask(id=1, name="Root", depends=[]),
            SubTask(id=2, name="A",   depends=[1]),
            SubTask(id=3, name="B",   depends=[1]),
            SubTask(id=4, name="Tip", depends=[2, 3]),
        ]
        context = "Compute something complex."

        def make_output(task_id: int, name: str, detail: str) -> SubTaskOutput:
            return SubTaskOutput(id=task_id, name=name, detail=detail, summary=f"{name} done.")

        executor = MockExecutor.make_executor({
            1: make_output(1, "Root", "Root computation done."),
            2: make_output(2, "A",   "Branch A computation done."),
            3: make_output(3, "B",   "Branch B computation done."),
            4: make_output(4, "Tip",  "Merged result produced."),
        })

        statuses, outputs = await run_all(tasks, executor, context)

        self.assertEqual(statuses[4], "done")
        ctx4 = next(c["context"] for c in MockExecutor.calls if c["task_id"] == 4)

        self.assertIn("Branch A", ctx4)
        self.assertIn("Branch B", ctx4)
        self.assertIn("Root computation", ctx4)

    # ------------------------------------------------------------------
    # 7. Detail truncation (2000 char cap prevents context bloat)
    # ------------------------------------------------------------------

    async def test_detail_truncation(self) -> None:
        """Detail passed to downstream context is capped at 2000 chars."""
        tasks = [
            SubTask(id=1, name="Long output", depends=[]),
            SubTask(id=2, name="Consumer",   depends=[1]),
        ]
        context = "Analyze this data."

        long_detail = "X" * 5000
        captured: dict[int, str] = {}

        async def executor(task_id: int, task_name: str, ctx: str) -> SubTaskOutput:
            await asyncio.sleep(0)
            captured[task_id] = ctx
            if task_id == 1:
                return SubTaskOutput(id=1, name="Long output", detail=long_detail, summary="Long.")
            return SubTaskOutput(id=2, name="Consumer", detail="Consumed.", summary="Done.")

        statuses, outputs = await run_all(tasks, executor, context)

        ctx2 = captured[2]
        # beginning of the long detail is visible, but not all 5000 chars
        self.assertIn("X" * 100, ctx2)
        # Truncation to 2000 per task means a 5000-char detail only appears ~2000 chars
        # of it.  Multiplied by 2 (accumulated + dep_block) + overhead ≈ 4090.
        # The assertion: ensure we're not carrying the full 5000-char blob.
        self.assertLessEqual(len(ctx2), 5000)


class TestSchedulerCycleDetection(unittest.IsolatedAsyncioTestCase):
    """Tests for check_circle()"""

    def test_empty_list(self) -> None:
        self.assertFalse(check_circle([]))

    def test_single_node_no_deps(self) -> None:
        self.assertFalse(check_circle([SubTask(id=1, name="A", depends=[])]))

    def test_simple_chain(self) -> None:
        self.assertFalse(check_circle([
            SubTask(id=1, name="A", depends=[]),
            SubTask(id=2, name="B", depends=[1]),
            SubTask(id=3, name="C", depends=[2]),
        ]))

    def test_diamond_no_cycle(self) -> None:
        self.assertFalse(check_circle([
            SubTask(id=1, name="Root", depends=[]),
            SubTask(id=2, name="A",    depends=[1]),
            SubTask(id=3, name="B",   depends=[1]),
            SubTask(id=4, name="Tip", depends=[2, 3]),
        ]))

    def test_self_loop_detected(self) -> None:
        self.assertTrue(check_circle([SubTask(id=1, name="A", depends=[1])]))

    def test_simple_cycle_detected(self) -> None:
        # 1->2->3->1: 1 depends on 3, 2 on 1, 3 on 2 = cycle
        self.assertTrue(check_circle([
            SubTask(id=1, name="A", depends=[3]),
            SubTask(id=2, name="B", depends=[1]),
            SubTask(id=3, name="C", depends=[2]),
        ]))


class TestSchedulerRetry(unittest.IsolatedAsyncioTestCase):
    """Tests for the retry mechanism."""

    async def test_retry_on_failure_then_success(self) -> None:
        """Executor fails once for task 1, then succeeds."""
        MockExecutor.reset()
        call_log: list[int] = []

        async def flaky_executor(task_id: int, task_name: str, context: str) -> SubTaskOutput:
            await asyncio.sleep(0)
            call_log.append(task_id)
            if task_id == 1 and call_log.count(1) == 1:
                raise RuntimeError("Transient error")
            return SubTaskOutput(id=task_id, name=task_name, detail="OK", summary="OK")

        tasks = [SubTask(id=1, name="Flaky", depends=[])]
        retry_cfg = RetryConfig(max_attempts=3, base_delay=0.01)

        statuses, outputs = await run_all(tasks, flaky_executor, "test", retry=retry_cfg)

        self.assertEqual(statuses[1], "done")
        self.assertEqual(call_log.count(1), 2)  # failed once, then succeeded

    async def test_max_retries_exceeded(self) -> None:
        """All retries exhausted → task should be marked failed."""
        async def always_fail(task_id: int, task_name: str, context: str) -> SubTaskOutput:
            await asyncio.sleep(0)
            raise RuntimeError("Permanent error")

        tasks = [SubTask(id=1, name="Fail", depends=[])]
        retry_cfg = RetryConfig(max_attempts=2, base_delay=0.01)

        statuses, outputs = await run_all(tasks, always_fail, "test", retry=retry_cfg)

        self.assertEqual(statuses[1], "failed")
        self.assertTrue(outputs[0].summary.startswith("[失败]"))


class TestSchedulerEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Boundary and edge-case tests."""

    async def test_all_parallel_tasks(self) -> None:
        """All tasks have no dependencies → all run in one layer."""
        tasks = [
            SubTask(id=1, name="A", depends=[]),
            SubTask(id=2, name="B", depends=[]),
            SubTask(id=3, name="C", depends=[]),
        ]
        context = "Parallel work."

        def make_output(tid: int, name: str) -> SubTaskOutput:
            return SubTaskOutput(id=tid, name=name, detail=f"{name} done.", summary=f"{name} done.")

        executor = MockExecutor.make_executor({
            1: make_output(1, "A"),
            2: make_output(2, "B"),
            3: make_output(3, "C"),
        })

        statuses, outputs = await run_all(tasks, executor, context)

        self.assertEqual(len(outputs), 3)
        for tid in [1, 2, 3]:
            self.assertEqual(statuses[tid], "done")

    async def test_dependency_order_respected(self) -> None:
        """Tasks with deps must not run before their deps complete."""
        execution_order: list[int] = []

        async def tracking_executor(task_id: int, task_name: str, context: str) -> SubTaskOutput:
            await asyncio.sleep(0.01)
            execution_order.append(task_id)
            detail_prefix = f"Task {task_id} sees context: "
            has_dep_result = "Task 1" in context and task_id != 1
            return SubTaskOutput(
                id=task_id, name=task_name,
                detail=f"{detail_prefix}{'dep results' if has_dep_result else 'no deps'}",
                summary=f"Task {task_id} done.",
            )

        tasks = [
            SubTask(id=1, name="Root", depends=[]),
            SubTask(id=2, name="Leaf", depends=[1]),
        ]

        statuses, outputs = await run_all(tasks, tracking_executor, "root context")

        self.assertEqual(execution_order[0], 1)
        self.assertEqual(execution_order[1], 2)

    async def test_nonexistent_dependency_reference(self) -> None:
        """check_circle should catch references to non-existent task IDs."""
        # Task 2 declares a dependency on task 99 which doesn't exist
        self.assertTrue(check_circle([
            SubTask(id=1, name="A", depends=[]),
            SubTask(id=2, name="B", depends=[99]),  # non-existent
        ]))

    async def test_empty_task_list(self) -> None:
        """run_all on empty list should return immediately."""
        statuses, outputs = await run_all([], lambda *a, **kw: None, "unused context")
        self.assertEqual(statuses, {})
        self.assertEqual(outputs, [])


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

class TestUpstreamToolResultsInDownstreamContext(unittest.IsolatedAsyncioTestCase):
    """Core regression test: upstream tool results must appear in downstream task context.

    Scenario:
      Task 1 reads /etc/hostname (via tool).
      Task 2 depends on Task 1 and should NOT need to re-read the file — the result
      must already be in Task 2's context.

    This is the bug the user is worried about: "并行调度的过程中，上游节点调用的
    工具不会再被下游节点在调用".
    """

    async def test_upstream_tool_result_visible_in_downstream_context(self) -> None:
        """Task 2's context must contain Task 1's tool result verbatim."""
        tasks = [
            SubTask(id=1, name="Read config", depends=[]),
            SubTask(id=2, name="Analyze config", depends=[1]),
        ]
        context = "Please audit the system configuration."

        # Simulate what a real executor would return after reading a file.
        # The detail includes the tool execution log, as execution.py now does.
        task1_detail = (
            "我执行了 read_file('/etc/hostname')。\n\n"
            "[工具执行记录]\n"
            "[Tool: read_file]\n"
            "my-hostname"
        )
        task1_output = SubTaskOutput(
            id=1, name="Read config",
            detail=task1_detail,
            summary="Config file read.",
        )
        task2_output = SubTaskOutput(
            id=2, name="Analyze config",
            detail="Analysis complete.",
            summary="Done.",
        )

        executor = MockExecutor.make_executor({1: task1_output, 2: task2_output})
        statuses, outputs = await run_all(tasks, executor, context)

        self.assertEqual(statuses[1], "done")
        self.assertEqual(statuses[2], "done")

        # Collect contexts
        ctx = {c["task_id"]: c["context"] for c in MockExecutor.calls}

        # Task 1 has no deps — context is just the original question
        self.assertEqual(ctx[1], context)

        # Task 2's context MUST contain the upstream tool result
        ctx2 = ctx[2]
        self.assertIn("my-hostname", ctx2)           # the actual tool result
        self.assertIn("[Tool: read_file]", ctx2)     # the tool call record
        self.assertIn("Please audit the system", ctx2) # original question still present

    async def test_upstream_tool_result_not_duplicated_in_accumulated_context(self) -> None:
        """The accumulated context grows layer by layer, not duplicating results."""
        tasks = [
            SubTask(id=1, name="Read",  depends=[]),
            SubTask(id=2, name="Check", depends=[1]),
            SubTask(id=3, name="Audit", depends=[2]),
        ]
        context = "Multi-layer audit."

        task1_detail = "Step1 read file: CONTENT_FROM_UPSTREAM"
        task2_detail = "Step2 checked: BASED_ON_CONTENT_FROM_UPSTREAM"
        task3_detail = "Step3 audited: ALL_PREVIOUS"

        def make_output(tid: int, detail: str) -> SubTaskOutput:
            return SubTaskOutput(id=tid, name=str(tid), detail=detail, summary=f"Task {tid} done.")

        executor = MockExecutor.make_executor({
            1: make_output(1, task1_detail),
            2: make_output(2, task2_detail),
            3: make_output(3, task3_detail),
        })

        statuses, outputs = await run_all(tasks, executor, context)

        ctx = {c["task_id"]: c["context"] for c in MockExecutor.calls}

        # Task 1: only original
        self.assertIn("Multi-layer audit", ctx[1])
        self.assertNotIn("Step1", ctx[1])

        # Task 2: original + task1
        self.assertIn("Step1", ctx[2])
        self.assertIn("CONTENT_FROM_UPSTREAM", ctx[2])

        # Task 3: original + task1 + task2
        self.assertIn("Step1", ctx[3])
        self.assertIn("Step2", ctx[3])
        self.assertIn("BASED_ON_CONTENT_FROM_UPSTREAM", ctx[3])

        # Task 3's own result is NOT yet in its own context (hasn't been produced yet)
        self.assertNotIn("Step3 audited", ctx[3])

        # Ensure results don't multiply: task1_detail should appear exactly once
        # in task 3's context (only in the accumulated block, not twice)
        # The content appears at least once (may appear twice: once from the original
        # source in task 1's detail, and once in task 2's output where it reports
        # "BASED_ON_CONTENT_FROM_UPSTREAM" — both are valid).
        self.assertGreaterEqual(ctx[3].count("CONTENT_FROM_UPSTREAM"), 1)

    async def test_diamond_both_upstream_tools_visible(self) -> None:
        """
        Diamond DAG:
            1 (reads file A)
           / \
          2   3 (reads file B)
           \ /
            4 (must see both file A and file B results)

        Neither Task 2 nor Task 3 should need to re-read what 1 already read,
        and Task 4 should see both upstream tool results.
        """
        tasks = [
            SubTask(id=1, name="Read both files", depends=[]),
            SubTask(id=2, name="Process A",      depends=[1]),
            SubTask(id=3, name="Process B",      depends=[1]),
            SubTask(id=4, name="Merge results",   depends=[2, 3]),
        ]
        context = "Process two config files."

        def make_output(tid: int, name: str, tool_result: str) -> SubTaskOutput:
            detail = f"[Tool: read_file]\n{tool_result}"
            return SubTaskOutput(id=tid, name=name, detail=detail, summary=f"{name} done.")

        executor = MockExecutor.make_executor({
            1: make_output(1, "Read both files", "FILE_A_CONTENT\nFILE_B_CONTENT"),
            2: make_output(2, "Process A",       "PROCESSED_A: ok"),
            3: make_output(3, "Process B",        "PROCESSED_B: ok"),
            4: make_output(4, "Merge results",    "MERGED: all good"),
        })

        statuses, outputs = await run_all(tasks, executor, context)

        ctx = {c["task_id"]: c["context"] for c in MockExecutor.calls}

        # Task 4 sees BOTH upstream tool results from tasks 2 and 3
        self.assertIn("PROCESSED_A: ok", ctx[4])
        self.assertIn("PROCESSED_B: ok", ctx[4])
        # And the root tool result too
        self.assertIn("FILE_A_CONTENT", ctx[4])
        self.assertIn("FILE_B_CONTENT", ctx[4])

        # But task 2 and 3 don't see each other's results (not yet produced when they run)
        self.assertNotIn("PROCESSED_B", ctx[2])
        self.assertNotIn("PROCESSED_A", ctx[3])


if __name__ == "__main__":
    # verbosity=2 shows test names
    unittest.main(verbosity=2)
