"""Loading and validation for recorded agent-run fixtures (JSONL).

Fixtures are **untrusted data**: they are parsed into immutable dataclasses
and only ever read by the deterministic scorers. Nothing in a fixture is
executed, eval'd, or interpolated into code — tool "results" and retrieved
chunk texts are opaque strings/JSON payloads used solely for containment
checks.

Schema (one JSON object per line)::

    {
      "id": "rag-helium-risks",
      "question": "What are the risks from helium supply disruption?",
      "category": "rag" | "financial_tool" | "mixed" | "out_of_scope",
      "expected": {
        "tools": ["search_documents"],
        "answer_terms": ["helium", "liquefaction"],
        "cited_sources": ["zephyr-systems-10k.md"],
        "min_citations": 1
      },
      "run": {
        "tools_called": [{"name": "search_documents", "ok": true,
                          "latency_ms": 32.0, "result": {...}}],
        "retrieved": [{"source_file": "zephyr-systems-10k.md",
                       "chunk_id": "e57aac...", "text": "...", "score": 0.83}],
        "answer": "...",
        "answer_citations": ["zephyr-systems-10k.md"],
        "telemetry": {"total_ms": 1480.0, "retrieval_ms": 32.0,
                      "tool_ms": 35.0, "model_ms": 1445.0,
                      "input_tokens": 2400, "output_tokens": 190,
                      "total_tokens": 2590}
      },
      "synthetic_negative": false,   // true only for detector self-checks
      "expect_flags": ["fabricated_citation"],  // required for negatives
      "notes": "provenance for this fixture"
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from MarketInsight.evaluation.telemetry import RunTelemetry

CATEGORIES = ("rag", "financial_tool", "mixed", "out_of_scope")


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    ok: bool
    latency_ms: Optional[float] = None
    result: Any = None


@dataclass(frozen=True)
class RetrievedRecord:
    source_file: str
    text: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None


@dataclass(frozen=True)
class ExpectedSpec:
    tools: Tuple[str, ...] = ()
    answer_terms: Tuple[str, ...] = ()
    cited_sources: Tuple[str, ...] = ()
    min_citations: int = 0


@dataclass(frozen=True)
class AgentFixture:
    id: str
    question: str
    category: str
    expected: ExpectedSpec
    tools_called: Tuple[ToolCallRecord, ...]
    retrieved: Tuple[RetrievedRecord, ...]
    answer: str
    answer_citations: Tuple[str, ...]
    telemetry: RunTelemetry
    synthetic_negative: bool = False
    expect_flags: Tuple[str, ...] = ()
    notes: str = ""


def evidence_text(fixture: AgentFixture) -> str:
    """All text the answer could legitimately be grounded in: retrieved chunk
    texts plus serialized tool results."""
    parts = [r.text for r in fixture.retrieved]
    for call in fixture.tools_called:
        if call.result is not None:
            parts.append(json.dumps(call.result, ensure_ascii=False))
    return "\n".join(parts)


def evidence_sources(fixture: AgentFixture) -> List[str]:
    """Document sources that were actually available to the answer."""
    return [r.source_file for r in fixture.retrieved]


def _err(path: Path, lineno: int, message: str) -> ValueError:
    return ValueError(f"{path}:{lineno}: {message}")


def _require_str(raw: dict, key: str, path: Path, lineno: int, *, allow_empty: bool = False) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise _err(path, lineno, f"'{key}' must be a non-empty string")
    return value


def _str_tuple(value: Any, key: str, path: Path, lineno: int) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise _err(path, lineno, f"'{key}' must be a list of non-empty strings")
    return tuple(value)


def _parse_fixture(raw: Any, path: Path, lineno: int) -> AgentFixture:
    if not isinstance(raw, dict):
        raise _err(path, lineno, "each fixture must be a JSON object")

    fixture_id = _require_str(raw, "id", path, lineno)
    question = _require_str(raw, "question", path, lineno)
    category = raw.get("category")
    if category not in CATEGORIES:
        raise _err(path, lineno, f"{fixture_id}: 'category' must be one of {CATEGORIES}")

    expected_raw = raw.get("expected")
    if not isinstance(expected_raw, dict):
        raise _err(path, lineno, f"{fixture_id}: 'expected' must be an object")
    min_citations = expected_raw.get("min_citations", 0)
    if isinstance(min_citations, bool) or not isinstance(min_citations, int) or min_citations < 0:
        raise _err(path, lineno, f"{fixture_id}: 'min_citations' must be a non-negative integer")
    expected = ExpectedSpec(
        tools=_str_tuple(expected_raw.get("tools"), "expected.tools", path, lineno),
        answer_terms=_str_tuple(expected_raw.get("answer_terms"), "expected.answer_terms", path, lineno),
        cited_sources=_str_tuple(expected_raw.get("cited_sources"), "expected.cited_sources", path, lineno),
        min_citations=min_citations,
    )

    run_raw = raw.get("run")
    if not isinstance(run_raw, dict):
        raise _err(path, lineno, f"{fixture_id}: 'run' must be an object")

    tools_called = []
    for i, call in enumerate(run_raw.get("tools_called") or []):
        if not isinstance(call, dict):
            raise _err(path, lineno, f"{fixture_id}: tools_called[{i}] must be an object")
        name = call.get("name")
        ok = call.get("ok")
        if not isinstance(name, str) or not name:
            raise _err(path, lineno, f"{fixture_id}: tools_called[{i}].name must be a non-empty string")
        if not isinstance(ok, bool):
            raise _err(path, lineno, f"{fixture_id}: tools_called[{i}].ok must be a boolean")
        latency = call.get("latency_ms")
        if latency is not None and (isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0):
            raise _err(path, lineno, f"{fixture_id}: tools_called[{i}].latency_ms must be a non-negative number or null")
        tools_called.append(ToolCallRecord(name=name, ok=ok, latency_ms=latency, result=call.get("result")))

    retrieved = []
    for i, chunk in enumerate(run_raw.get("retrieved") or []):
        if not isinstance(chunk, dict):
            raise _err(path, lineno, f"{fixture_id}: retrieved[{i}] must be an object")
        source_file = chunk.get("source_file")
        text = chunk.get("text")
        if not isinstance(source_file, str) or not source_file:
            raise _err(path, lineno, f"{fixture_id}: retrieved[{i}].source_file must be a non-empty string")
        if not isinstance(text, str) or not text:
            raise _err(path, lineno, f"{fixture_id}: retrieved[{i}].text must be a non-empty string")
        retrieved.append(RetrievedRecord(
            source_file=source_file, text=text,
            chunk_id=chunk.get("chunk_id"), score=chunk.get("score"),
        ))

    answer = run_raw.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise _err(path, lineno, f"{fixture_id}: 'run.answer' must be a non-empty string")

    try:
        telemetry = RunTelemetry.from_dict(run_raw.get("telemetry"), context=f"{fixture_id}")
    except ValueError as exc:
        raise _err(path, lineno, str(exc)) from None

    synthetic_negative = bool(raw.get("synthetic_negative", False))
    expect_flags = _str_tuple(raw.get("expect_flags"), "expect_flags", path, lineno)
    if synthetic_negative and not expect_flags:
        raise _err(path, lineno, f"{fixture_id}: synthetic negatives must declare 'expect_flags'")

    return AgentFixture(
        id=fixture_id,
        question=question,
        category=category,
        expected=expected,
        tools_called=tuple(tools_called),
        retrieved=tuple(retrieved),
        answer=answer,
        answer_citations=_str_tuple(run_raw.get("answer_citations"), "run.answer_citations", path, lineno),
        telemetry=telemetry,
        synthetic_negative=synthetic_negative,
        expect_flags=expect_flags,
        notes=raw.get("notes", "") if isinstance(raw.get("notes", ""), str) else "",
    )


def load_fixtures(path: Union[str, Path]) -> List[AgentFixture]:
    """Parse and validate a JSONL fixture file. Raises ``ValueError`` with
    file/line context on the first invalid entry."""
    path = Path(path)
    fixtures: List[AgentFixture] = []
    seen_ids = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _err(path, lineno, f"invalid JSON: {exc.msg}") from None
        fixture = _parse_fixture(raw, path, lineno)
        if fixture.id in seen_ids:
            raise _err(path, lineno, f"duplicate fixture id {fixture.id!r}")
        seen_ids.add(fixture.id)
        fixtures.append(fixture)
    if not fixtures:
        raise ValueError(f"{path}: no fixtures found")
    return fixtures
