import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.observability import TokenTrackerCallback, get_tracer


def _span(record, name: str):
    for span in record.spans:
        if span.name == name:
            return span
    raise AssertionError(f"missing span {name}")


def test_token_usage_is_recorded_on_current_span_without_snapshot() -> None:
    tracer = get_tracer()
    _ = tracer.end_trace()
    _ = tracer.start_trace(query="token test", session_id="s")
    TokenTrackerCallback.reset()

    with tracer.span("first"):
        TokenTrackerCallback._accumulate(10, 3, "model-a")
    with tracer.span("second"):
        TokenTrackerCallback._accumulate(20, 4, "model-b", source="estimated")

    record = tracer.end_trace()
    assert record is not None

    first = _span(record, "first")
    second = _span(record, "second")
    assert (first.tokens_in, first.tokens_out, first.model, first.token_source) == (
        10,
        3,
        "model-a",
        "api",
    )
    assert (second.tokens_in, second.tokens_out, second.model, second.token_source) == (
        20,
        4,
        "model-b",
        "estimated",
    )


def test_concurrent_spans_keep_token_usage_isolated() -> None:
    async def scenario():
        tracer = get_tracer()
        _ = tracer.end_trace()
        _ = tracer.start_trace(query="parallel", session_id="s")
        TokenTrackerCallback.reset()

        async def worker(name: str, tokens: int) -> None:
            with tracer.span(name):
                await asyncio.sleep(0)
                TokenTrackerCallback._accumulate(tokens, 1, name)

        await asyncio.gather(worker("a", 11), worker("b", 22))
        return tracer.end_trace()

    record = asyncio.run(scenario())
    assert record is not None
    assert (_span(record, "a").tokens_in, _span(record, "a").tokens_out) == (11, 1)
    assert (_span(record, "b").tokens_in, _span(record, "b").tokens_out) == (22, 1)


def test_zero_snapshot_does_not_erase_span_token_usage() -> None:
    tracer = get_tracer()
    _ = tracer.end_trace()
    _ = tracer.start_trace(query="no overwrite", session_id="s")
    TokenTrackerCallback.reset()

    with tracer.span("execution") as span_id:
        TokenTrackerCallback._accumulate(100, 9, "model-a")
        tin, tout, model = TokenTrackerCallback.snapshot()
        assert (tin, tout, model) == (0, 0, "")
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

    record = tracer.end_trace()
    assert record is not None
    span = _span(record, "execution")
    assert (span.tokens_in, span.tokens_out, span.model, span.token_source) == (
        100,
        9,
        "model-a",
        "api",
    )
