"""Report builder and CLI for the Phase 3 evaluation harness.

Pipeline: fixtures -> deterministic/heuristic scorers -> aggregation ->
JSON + human-readable report. The retrieval section reuses the Phase 2
``MarketInsight.rag.eval`` harness against the live local index (no API key
needed); answer/agent/performance sections always run offline from recorded
fixtures. Nothing is ever fabricated: skipped sections say so, missing
telemetry stays null, and p95 is suppressed for small samples.

CLI::

    python -m MarketInsight.evaluation [--fixtures tests/eval/agent_fixtures.jsonl]
        [--golden tests/eval/golden.jsonl] [--samples samples/docs]
        [--data-dir data/rag] [--skip-retrieval] [--format text|json]
        [--price-input-per-1k X --price-output-per-1k Y] [--json-out report.json]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from MarketInsight.evaluation.datasets import (
    AgentFixture,
    evidence_sources,
    evidence_text,
    load_fixtures,
)
from MarketInsight.evaluation.scoring import (
    citation_checks,
    detector_flags,
    is_grounded,
    numeric_claim_support,
    score_tools,
    term_coverage,
)
from MarketInsight.evaluation.telemetry import aggregate_telemetry, estimate_cost
from MarketInsight.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1


def evaluate_fixture(fixture: AgentFixture) -> Dict[str, Any]:
    """Score one fixture; returns the per-case report row."""
    terms = term_coverage(fixture.answer, fixture.expected.answer_terms)
    citations = citation_checks(
        fixture.answer_citations,
        expected_sources=fixture.expected.cited_sources,
        evidence_sources=evidence_sources(fixture),
        min_citations=fixture.expected.min_citations,
    )
    numeric = numeric_claim_support(fixture.answer, evidence_text(fixture))
    tools = score_tools(fixture.expected.tools, fixture.tools_called)
    flags = detector_flags(terms, citations, numeric, tools)
    return {
        "id": fixture.id,
        "category": fixture.category,
        "synthetic_negative": fixture.synthetic_negative,
        "fact_coverage": terms["coverage"],
        "missing_terms": terms["missing"],
        "citation_presence": citations["presence"],
        "citation_correctness": citations["citation_correctness"],
        "fabricated_citations": citations["fabricated"],
        "missing_expected_sources": citations["expected_missing"],
        "numeric_claims": numeric["claims"],
        "unsupported_claims": numeric["unsupported"],
        "grounded": is_grounded(terms, citations, numeric),
        "tool_selection": {
            "expected": tools["expected"],
            "called": tools["called"],
            "correct": tools["selection_correct"],
            "missing": tools["missing"],
            "unexpected": tools["unexpected"],
            "unnecessary": tools["unnecessary"],
            "execution_success_rate": tools["execution_success_rate"],
        },
        "flags": flags,
        "expect_flags": list(fixture.expect_flags),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_report(
    fixtures: Sequence[AgentFixture],
    *,
    retrieval_section: Optional[Dict[str, Any]] = None,
    pricing: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Aggregate per-case scores into the full evaluation report."""
    rows = [evaluate_fixture(fx) for fx in fixtures]
    pos_pairs = [(fx, r) for fx, r in zip(fixtures, rows) if not fx.synthetic_negative]
    positives = [r for _, r in pos_pairs]
    negatives = [r for r in rows if r["synthetic_negative"]]
    positive_fixtures = [fx for fx, _ in pos_pairs]

    citation_required = [r for fx, r in pos_pairs if fx.expected.min_citations > 0]
    cited_cases = [r for fx, r in pos_pairs if fx.answer_citations]
    numeric_cases = [r for r in positives if r["numeric_claims"] > 0]

    answer_metrics = {
        "cases": len(positives),
        "fact_coverage": round(_mean([r["fact_coverage"] for r in positives]), 4),
        "grounded_answer_rate": round(_mean([r["grounded"] for r in positives]), 4),
        "citation_coverage": (
            round(_mean([r["citation_presence"] for r in citation_required]), 4)
            if citation_required else None
        ),
        "citation_correctness": (
            round(_mean([r["citation_correctness"] for r in cited_cases]), 4)
            if cited_cases else None
        ),
        "unsupported_claim_rate": (
            round(_mean([
                len(r["unsupported_claims"]) / r["numeric_claims"]
                for r in numeric_cases
            ]), 4)
            if numeric_cases else None
        ),
        "cases_with_numeric_claims": len(numeric_cases),
        "unsupported_claim_note": "heuristic check (digit-form quantities only); "
                                  "see module docs for known false positive/negative modes",
    }

    total_calls = sum(len(r["tool_selection"]["called"]) for r in positives)
    total_unexpected = sum(len(r["tool_selection"]["unexpected"]) for r in positives)
    total_unnecessary = sum(len(r["tool_selection"]["unnecessary"]) for r in positives)
    total_succeeded = sum(
        r["tool_selection"]["execution_success_rate"] * len(r["tool_selection"]["called"])
        for r in positives
    )
    agent_metrics = {
        "cases": len(positives),
        "tool_selection_accuracy": round(_mean([r["tool_selection"]["correct"] for r in positives]), 4),
        "total_tool_calls": total_calls,
        "unexpected_tool_call_rate": round(total_unexpected / total_calls, 4) if total_calls else None,
        "unnecessary_tool_call_rate": round(total_unnecessary / total_calls, 4) if total_calls else None,
        "tool_execution_success_rate": round(total_succeeded / total_calls, 4) if total_calls else None,
    }

    detector_rows = []
    for row in negatives:
        detected = set(row["expect_flags"]).issubset(set(row["flags"]))
        detector_rows.append({
            "id": row["id"],
            "expect_flags": row["expect_flags"],
            "flags": row["flags"],
            "fully_detected": detected,
        })
    detector = {
        "negative_cases": len(detector_rows),
        "fully_detected": sum(r["fully_detected"] for r in detector_rows),
        "detection_rate": (
            round(_mean([r["fully_detected"] for r in detector_rows]), 4)
            if detector_rows else None
        ),
        "per_case": detector_rows,
    }

    telemetry_records = [fx.telemetry for fx in positive_fixtures]
    performance = aggregate_telemetry(telemetry_records, pricing=pricing)
    if pricing is not None:
        performance["cost"]["per_case"] = {
            fx.id: estimate_cost(fx.telemetry.input_tokens, fx.telemetry.output_tokens, pricing)
            for fx in positive_fixtures
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "offline-fixtures",
        "retrieval": retrieval_section or {"status": "skipped", "reason": "not requested"},
        "answer": answer_metrics,
        "agent": agent_metrics,
        "detector_self_check": detector,
        "performance": performance,
        "per_case": rows,
    }


# --------------------------------------------------------------------------------
# Human-readable rendering
# --------------------------------------------------------------------------------

def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _latency_line(label: str, summary: Dict[str, Any]) -> str:
    if summary["n"] == 0:
        return f"  {label:<14} n/a (no samples)"
    p95 = f"{summary['p95']:.1f}" if summary["p95"] is not None else "suppressed (n<20)"
    return (
        f"  {label:<14} mean {summary['mean']:.1f} ms  p50 {summary['p50']:.1f} ms  "
        f"p95 {p95}  max {summary['max']:.1f} ms  (n={summary['n']})"
    )


def render_text(report: Dict[str, Any]) -> str:
    answer, agent = report["answer"], report["agent"]
    detector, perf = report["detector_self_check"], report["performance"]
    lines: List[str] = [
        "AI Financial Research Copilot — Phase 3 Evaluation Report",
        f"Mode: {report['mode']} (no live LLM/API calls for answer & agent sections)",
        f"Positive cases: {answer['cases']}   Synthetic negative checks: {detector['negative_cases']}",
        "",
        "Retrieval (live local index, no LLM):",
    ]
    retrieval = report["retrieval"]
    if retrieval.get("status") == "ok":
        metrics = retrieval["metrics"]
        lines.append(
            f"  recall@5 {_pct(metrics.get('recall@5'))}   mrr {_pct(metrics.get('mrr'))}   "
            f"citation coverage {_pct(metrics.get('citation_coverage'))}   "
            f"questions {metrics.get('questions')}"
        )
    else:
        lines.append(f"  skipped: {retrieval.get('reason', 'unknown')}")
    lines += [
        "",
        "Answer quality (deterministic + labelled-heuristic checks):",
        f"  fact coverage              {_pct(answer['fact_coverage'])}",
        f"  grounded answer rate       {_pct(answer['grounded_answer_rate'])}",
        f"  citation coverage          {_pct(answer['citation_coverage'])}",
        f"  citation correctness       {_pct(answer['citation_correctness'])}",
        f"  unsupported claim rate     {_pct(answer['unsupported_claim_rate'])}"
        f"  (heuristic, n={answer['cases_with_numeric_claims']})",
        "",
        "Agent / tool selection:",
        f"  tool-selection accuracy    {_pct(agent['tool_selection_accuracy'])}",
        f"  unexpected tool-call rate  {_pct(agent['unexpected_tool_call_rate'])}",
        f"  unnecessary tool-call rate {_pct(agent['unnecessary_tool_call_rate'])}",
        f"  tool execution success     {_pct(agent['tool_execution_success_rate'])}"
        f"  ({agent['total_tool_calls']} calls)",
        "",
        "Detector self-check (synthetic negatives must be flagged):",
        f"  detection rate             {_pct(detector['detection_rate'])}"
        f"  ({detector['fully_detected']}/{detector['negative_cases']})",
        "",
        f"Performance (source: {', '.join(f'{k}={v}' for k, v in perf['sources'].items())}; "
        "descriptive, small sample — p95 guarded):",
        _latency_line("total", perf["latency_ms"]["total_ms"]),
        _latency_line("retrieval", perf["latency_ms"]["retrieval_ms"]),
        _latency_line("tool", perf["latency_ms"]["tool_ms"]),
        _latency_line("model", perf["latency_ms"]["model_ms"]),
    ]
    tokens = perf["tokens"]
    total_tokens = tokens["total_tokens"]["total"]
    lines.append(
        f"  tokens         total {total_tokens if total_tokens is not None else 'n/a'}"
        f"  (input n={tokens['input_tokens']['n_with_data']},"
        f" output n={tokens['output_tokens']['n_with_data']},"
        f" runs without token data: {tokens['runs_without_data']})"
    )
    cost = perf["cost"]
    if cost["total"] is None:
        lines.append(f"  cost           n/a — {cost['note'] if 'note' in cost else 'no token data'}")
    else:
        lines.append(f"  cost           total {cost['total']} (n={cost['n_with_data']}, configured pricing)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def _run_retrieval_section(args) -> Dict[str, Any]:
    """Live local retrieval eval via the Phase 2 harness (local model, no API
    key). Any failure becomes an honest 'skipped' section, never fake data."""
    try:
        from config.settings import get_settings
        from MarketInsight.rag.embed import FastEmbedder
        from MarketInsight.rag.eval import evaluate as run_retrieval_eval
        from MarketInsight.rag.eval import load_golden
        from MarketInsight.rag.ingest import build_default_retriever, ingest_path
        from MarketInsight.rag.store import RagStore

        settings = get_settings()
        data_dir = Path(args.data_dir or settings.rag_data_dir)
        embedder = FastEmbedder(settings.rag_embedding_model, cache_dir=data_dir / "models")
        store = RagStore(data_dir / "index.db", dim=embedder.dim)
        try:
            ingest_path(args.samples, store=store, embedder=embedder)
        finally:
            store.close()
        retriever = build_default_retriever(settings, data_dir, embedder=embedder)
        report = run_retrieval_eval(retriever, load_golden(args.golden), top_k=5)
        return {"status": "ok", "dataset": str(args.golden), "metrics": report["aggregate"]}
    except Exception as exc:  # noqa: BLE001 - degraded section, not a crash
        logger.warning("Retrieval section skipped (%s)", type(exc).__name__)
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 agent/answer evaluation harness (offline).")
    parser.add_argument("--fixtures", default="tests/eval/agent_fixtures.jsonl")
    parser.add_argument("--golden", default="tests/eval/golden.jsonl")
    parser.add_argument("--samples", default="samples/docs")
    parser.add_argument("--data-dir")
    parser.add_argument("--skip-retrieval", action="store_true",
                        help="Skip the live local retrieval section (answer/agent sections are always offline)")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--json-out", help="Also write the machine-readable report to this path")
    parser.add_argument("--price-input-per-1k", type=float)
    parser.add_argument("--price-output-per-1k", type=float)
    args = parser.parse_args(argv)

    if (args.price_input_per_1k is None) != (args.price_output_per_1k is None):
        parser.error("--price-input-per-1k and --price-output-per-1k must be given together")
    pricing = None
    if args.price_input_per_1k is not None:
        pricing = {"input_per_1k": args.price_input_per_1k, "output_per_1k": args.price_output_per_1k}

    fixtures = load_fixtures(args.fixtures)
    retrieval_section = None if args.skip_retrieval else _run_retrieval_section(args)
    report = build_report(fixtures, retrieval_section=retrieval_section, pricing=pricing)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0
