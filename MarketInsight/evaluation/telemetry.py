"""Latency / token / cost telemetry structures and aggregation.

Every field is optional: when a provider (or a recorded fixture) does not
expose a value it stays ``None`` — nothing is ever synthesised. Cost is only
computed when explicit pricing is supplied by the caller; no prices are
hardcoded anywhere.

Aggregation guards against misleading statistics: ``p95`` is only reported
when the sample size reaches ``MIN_SAMPLES_FOR_P95``; below that the report
carries ``None`` plus an explanatory note.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

MIN_SAMPLES_FOR_P95 = 20

_LATENCY_FIELDS = ("total_ms", "retrieval_ms", "tool_ms", "model_ms")
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


@dataclass(frozen=True)
class RunTelemetry:
    """Timing/token metadata for one evaluated run."""

    total_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    tool_ms: Optional[float] = None
    model_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    # "fixture" (recorded run), "live" (real LLM call) or "retrieval_only".
    source: str = "fixture"

    @classmethod
    def from_dict(cls, raw: Any, *, context: str = "telemetry") -> "RunTelemetry":
        """Tolerant parser: missing/null fields stay ``None``; present fields
        must be non-negative numbers."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"{context}: telemetry must be an object or null")

        def _num(key: str) -> Optional[float]:
            value = raw.get(key)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{context}: '{key}' must be a number or null")
            if value < 0:
                raise ValueError(f"{context}: '{key}' must be non-negative")
            return value

        source = raw.get("source", "fixture")
        if source not in ("fixture", "live", "retrieval_only"):
            raise ValueError(f"{context}: unknown telemetry source {source!r}")
        return cls(
            total_ms=_num("total_ms"),
            retrieval_ms=_num("retrieval_ms"),
            tool_ms=_num("tool_ms"),
            model_ms=_num("model_ms"),
            input_tokens=_num("input_tokens"),
            output_tokens=_num("output_tokens"),
            total_tokens=_num("total_tokens"),
            source=source,
        )


def summarize_latency(
    values: Sequence[float],
    *,
    min_samples_for_p95: int = MIN_SAMPLES_FOR_P95,
) -> Dict[str, Any]:
    """mean/p50 always (when n>=1); p95 only with enough samples."""
    samples = sorted(values)
    n = len(samples)
    if n == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None,
                "note": "no samples"}
    p95: Optional[float] = None
    note = None
    if n >= min_samples_for_p95:
        rank = max(1, math.ceil(0.95 * n))
        p95 = samples[rank - 1]
    else:
        note = f"p95 suppressed: n={n} < {min_samples_for_p95}"
    return {
        "n": n,
        "mean": round(sum(samples) / n, 2),
        "p50": samples[max(1, math.ceil(0.5 * n)) - 1],
        "p95": p95,
        "max": samples[-1],
        "note": note,
    }


def estimate_cost(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    pricing: Optional[Dict[str, float]],
) -> Optional[float]:
    """Cost in the pricing's currency unit, or ``None``.

    ``pricing`` must look like ``{"input_per_1k": 0.15, "output_per_1k": 0.6}``.
    Returns ``None`` when pricing is not configured or token data is missing —
    cost is never estimated from invented numbers.
    """
    if not pricing or input_tokens is None or output_tokens is None:
        return None
    return (
        (input_tokens / 1000.0) * pricing["input_per_1k"]
        + (output_tokens / 1000.0) * pricing["output_per_1k"]
    )


def aggregate_telemetry(
    records: Sequence[RunTelemetry],
    *,
    pricing: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Aggregate per-run telemetry into report-ready performance metrics."""
    latency = {
        field: summarize_latency([getattr(r, field) for r in records
                                  if getattr(r, field) is not None])
        for field in _LATENCY_FIELDS
    }

    tokens: Dict[str, Any] = {}
    for field in _TOKEN_FIELDS:
        present = [getattr(r, field) for r in records if getattr(r, field) is not None]
        tokens[field] = {
            "n_with_data": len(present),
            "total": sum(present) if present else None,
        }
    tokens["runs_without_data"] = sum(
        1 for r in records if r.input_tokens is None and r.output_tokens is None
    )

    costs = [
        estimate_cost(r.input_tokens, r.output_tokens, pricing) for r in records
    ]
    available_costs = [c for c in costs if c is not None]
    if pricing is None:
        cost: Dict[str, Any] = {
            "total": None,
            "note": "pricing not configured; cost unavailable",
        }
    else:
        cost = {
            "total": round(sum(available_costs), 6) if available_costs else None,
            "n_with_data": len(available_costs),
            "pricing": dict(pricing),
        }

    sources: Dict[str, int] = {}
    for r in records:
        sources[r.source] = sources.get(r.source, 0) + 1

    return {
        "runs": len(records),
        "sources": sources,
        "latency_ms": latency,
        "tokens": tokens,
        "cost": cost,
    }
