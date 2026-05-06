"""Observability: tracing, token accounting, and metrics for the agent.

Architecture:
  - Tracer                : manages a tree of spans (one per node/agent invocation).
  - TokenTrackerCallback   : captures token usage from every LLM call via monkey-patch.
  - TraceRecord           : the serialisable payload written to disk and printed to console.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


# ----------------------------------------------------------------------
# Pricing — used for cost estimation (USD per 1M tokens).
# Extend as needed. Key is matched as a substring of the model name.
# ----------------------------------------------------------------------
_PRICING: dict[str, tuple[float, float]] = {
    # (input_price_per_m, output_price_per_m)
    "deepseek":  (0.27,   1.10),
    "gpt":    (2.50,  10.00),
    "gemini": (0.15, 0.60),
    "claude":  (3.00,  15.00),
}

_DATADIR = Path(__file__).resolve().parents[2] / ".analysis" / "metrics"


# ----------------------------------------------------------------------
# TraceSpan — a single timed interval
# ----------------------------------------------------------------------
@dataclass
class TraceSpan:
    name:      str
    parent_id: str | None = None
    span_id:   str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    start_ms:  float = 0.0
    end_ms:    float = 0.0
    status:    str = "ok"          # "ok" | "error"
    error_msg: str = ""
    tokens_in:     int = 0
    tokens_out:    int = 0
    model:     str = ""
    metadata:  dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict:
        return {
            "span_id":  self.span_id,
            "parent_id": self.parent_id,
            "name":     self.name,
            "start_ms": self.start_ms,
            "end_ms":   self.end_ms,
            "duration_ms": self.duration_ms,
            "status":   self.status,
            "error_msg": self.error_msg,
            "tokens":   {"in": self.tokens_in, "out": self.tokens_out},
            "model":    self.model,
            "metadata": self.metadata,
        }


# ----------------------------------------------------------------------
# TraceRecord — the complete record for one query
# ----------------------------------------------------------------------
@dataclass
class TraceRecord:
    trace_id:  str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    session_id: str = ""
    query:     str = ""
    start_ms:  float = 0.0
    end_ms:    float = 0.0
    spans:     list[TraceSpan] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def total_tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.spans)

    @property
    def total_tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.spans)

    def estimate_cost(self) -> float:
        total = 0.0
        for s in self.spans:
            if not s.model:
                continue
            price_in, price_out = 0.0, 0.0
            for key, (pi, po) in _PRICING.items():
                if key.lower() in s.model.lower():
                    price_in, price_out = pi, po
                    break
            if price_in == 0.0:
                continue
            total += (s.tokens_in / 1_000_000) * price_in
            total += (s.tokens_out / 1_000_000) * price_out
        return round(total, 6)

    def to_dict(self) -> dict:
        return {
            "trace_id":    self.trace_id,
            "session_id":  self.session_id,
            "query":       self.query,
            "start_ms":    self.start_ms,
            "end_ms":      self.end_ms,
            "duration_ms": self.duration_ms,
            "total_tokens": {"in": self.total_tokens_in, "out": self.total_tokens_out},
            "estimated_cost_usd": self.estimate_cost(),
            "spans":       [s.to_dict() for s in self.spans],
        }

    def _tree(self) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {}
        for s in self.spans:
            children.setdefault(s.parent_id or "", []).append(s.span_id)
        return children

    def print_console(self) -> None:
        """Print a hierarchical span tree to stdout."""
        from src.cli import get_renderer

        id_map: dict[str, TraceSpan] = {s.span_id: s for s in self.spans}
        children: dict[str, list[str]] = self._tree()

        # Find root spans (no parent)
        roots = children.get("", [])
        root_spans = sorted((id_map[sid] for sid in roots if sid in id_map), key=lambda span: span.start_ms)
        get_renderer().render_trace_record(self, root_spans, children, id_map)

    def save(self) -> Path:
        _DATADIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.trace_id}_{ts}.json"
        path = _DATADIR / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path


# ----------------------------------------------------------------------
# Tracer — thread-safe span context manager
# ----------------------------------------------------------------------
class Tracer:
    """Manages the active trace and its spans."""

    _local = threading.local()

    def __init__(self) -> None:
        self._record: TraceRecord | None = None
        self._lock   = threading.Lock()

    # ── Trace lifecycle ───────────────────────────────────────────────

    def start_trace(self, query: str = "", session_id: str = "") -> str:
        """Begin a new trace. Returns the trace_id."""
        with self._lock:
            record = TraceRecord(
                query=query,
                session_id=session_id,
                start_ms=time.time() * 1000,
            )
            self._record = record
            return record.trace_id

    def end_trace(self) -> TraceRecord | None:
        """End the current trace and return the record."""
        with self._lock:
            if self._record is None:
                return None
            self._record.end_ms = time.time() * 1000
            record = self._record
            self._record = None
            return record

    @property
    def active_record(self) -> TraceRecord | None:
        return self._record

    # ── Span management ─────────────────────────────────────────────────

    def start_span(
        self,
        name: str,
        parent_id: str | None = None,
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Push a new span onto the active trace. Returns the span_id."""
        if self._record is None:
            return ""
        span = TraceSpan(
            name=name,
            parent_id=parent_id,
            start_ms=time.time() * 1000,
            model=model,
            metadata=metadata or {},
        )
        with self._lock:
            self._record.spans.append(span)
        return span.span_id

    def end_span(
        self,
        span_id: str,
        status: str = "ok",
        tokens_in: int = 0,
        tokens_out: int = 0,
        error_msg: str = "",
    ) -> None:
        """Close a span, recording duration and any metrics."""
        if self._record is None:
            return
        with self._lock:
            for s in self._record.spans:
                if s.span_id == span_id:
                    s.end_ms    = time.time() * 1000
                    s.status    = status
                    if tokens_in or tokens_out:
                        s.tokens_in  = tokens_in
                        s.tokens_out = tokens_out
                    if error_msg:
                        s.error_msg  = error_msg
                    break

    # ── Convenience context-manager ─────────────────────────────────────

    def span(
        self,
        name: str,
        parent_id: str | None = None,
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> _SpanContext:
        """Enter a span as a context manager.

        Usage:
            with tracer.span("my_node") as span_id:
                ...do work...
        """
        return _SpanContext(self, name, parent_id, model, metadata)

    def record_tokens(
        self,
        span_id: str,
        tokens_in: int,
        tokens_out: int,
        model: str = "",
    ) -> None:
        """Add or update token counts on a span."""
        if self._record is None or not span_id:
            return
        with self._lock:
            for s in self._record.spans:
                if s.span_id == span_id:
                    s.tokens_in  = tokens_in
                    s.tokens_out = tokens_out
                    if model:
                        s.model = model
                    break


class _SpanContext:
    """Returned by Tracer.span() so callers can use it with `with`."""

    __slots__ = ("_tracer", "_name", "_parent_id", "_model", "_metadata", "_span_id")

    def __init__(
        self,
        tracer: Tracer,
        name: str,
        parent_id: str | None,
        model: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        self._tracer   = tracer
        self._name     = name
        self._parent_id = parent_id
        self._model    = model
        self._metadata = metadata
        self._span_id  = ""

    def __enter__(self) -> str:
        self._span_id = self._tracer.start_span(
            self._name, self._parent_id, self._model, self._metadata,
        )
        return self._span_id

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status: str
        error_msg: str = ""
        if exc_type is not None:
            status    = "error"
            error_msg = f"{exc_type.__name__}: {exc_val}"
        else:
            status    = "ok"
        self._tracer.end_span(
            self._span_id,
            status=status,
            error_msg=error_msg,
        )


# ----------------------------------------------------------------------
# Token-tracking callback — populated by the monkey-patch in client.py
# ----------------------------------------------------------------------
class TokenTrackerCallback(BaseCallbackHandler):
    """Records token usage across all LLM calls.

    The global singleton is populated by the monkey-patch in client.py.
    Each span calls snapshot() to pop its own token delta.
    """

    _accumulated: dict = {}
    _lock = threading.Lock()

    @classmethod
    def _accumulate(cls, tokens_in: int, tokens_out: int, model: str) -> None:
        """Called by the monkey-patch to record tokens (thread-safe)."""
        with cls._lock:
            cls._accumulated.setdefault("", {"in": 0, "out": 0, "model": ""})
            cls._accumulated[""]["in"]  += tokens_in
            cls._accumulated[""]["out"] += tokens_out
            if model and not cls._accumulated[""]["model"]:
                cls._accumulated[""]["model"] = model

    @classmethod
    def snapshot(cls) -> tuple[int, int, str]:
        """Pop and return accumulated tokens, resetting state."""
        with cls._lock:
            entry = cls._accumulated.pop("", {"in": 0, "out": 0, "model": ""})
        return entry["in"], entry["out"], entry["model"]

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._accumulated.clear()


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Return the global Tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
