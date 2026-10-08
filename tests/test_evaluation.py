"""Tests for the Phase 3 evaluation harness (fixtures, scoring, telemetry,
report, CLI). Everything here is offline: no API keys, no network, no live
Yahoo Finance, no Langfuse, no embedding model."""
import json
from pathlib import Path

import pytest

from MarketInsight.evaluation import report as report_mod
from MarketInsight.evaluation.datasets import (
    AgentFixture,
    evidence_sources,
    evidence_text,
    load_fixtures,
)
from MarketInsight.evaluation.report import build_report, evaluate_fixture, render_text
from MarketInsight.evaluation.scoring import (
    citation_checks,
    detector_flags,
    extract_number_tokens,
    is_grounded,
    normalize_text,
    numeric_claim_support,
    score_tools,
    term_coverage,
)
from MarketInsight.evaluation.telemetry import (
    RunTelemetry,
    aggregate_telemetry,
    estimate_cost,
    summarize_latency,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = ROOT / "tests" / "eval" / "agent_fixtures.jsonl"


def _write_jsonl(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


def _minimal_entry(**overrides):
    entry = {
        "id": "fx-1",
        "question": "q?",
        "category": "rag",
        "expected": {"tools": ["search_documents"], "answer_terms": ["helium"]},
        "run": {"answer": "helium answer", "tools_called": [], "retrieved": []},
    }
    entry.update(overrides)
    return entry


class TestFixtureLoading:
    def test_real_fixture_file_loads(self):
        fixtures = load_fixtures(FIXTURES_PATH)
        assert len(fixtures) == 12
        positives = [f for f in fixtures if not f.synthetic_negative]
        negatives = [f for f in fixtures if f.synthetic_negative]
        assert len(positives) == 10 and len(negatives) == 2
        assert all(isinstance(f, AgentFixture) for f in fixtures)
        for f in negatives:
            assert f.expect_flags

    def test_invalid_json_rejected(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_fixtures(path)

    def test_missing_required_fields_rejected(self, tmp_path):
        for mutant in (
            {k: v for k, v in _minimal_entry().items() if k != "question"},
            _minimal_entry(category="unknown_category"),
            _minimal_entry(expected={"tools": "search_documents"}),
            _minimal_entry(expected={"min_citations": -1}),
            _minimal_entry(run={"answer": "   "}),
            _minimal_entry(synthetic_negative=True),  # no expect_flags
        ):
            path = _write_jsonl(tmp_path / "case.jsonl", [mutant])
            with pytest.raises(ValueError):
                load_fixtures(path)

    def test_duplicate_ids_rejected(self, tmp_path):
        path = _write_jsonl(tmp_path / "dup.jsonl", [_minimal_entry(), _minimal_entry()])
        with pytest.raises(ValueError, match="duplicate"):
            load_fixtures(path)

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no fixtures"):
            load_fixtures(path)

    def test_telemetry_validation(self, tmp_path):
        bad = _minimal_entry(run={"answer": "ok", "telemetry": {"total_ms": -5}})
        path = _write_jsonl(tmp_path / "tel.jsonl", [bad])
        with pytest.raises(ValueError, match="non-negative"):
            load_fixtures(path)
        bad2 = _minimal_entry(run={"answer": "ok", "telemetry": {"total_ms": "fast"}})
        path = _write_jsonl(tmp_path / "tel2.jsonl", [bad2])
        with pytest.raises(ValueError, match="number or null"):
            load_fixtures(path)

    def test_tool_call_validation(self, tmp_path):
        bad = _minimal_entry(run={"answer": "ok", "tools_called": [{"name": "x"}]})
        path = _write_jsonl(tmp_path / "tc.jsonl", [bad])
        with pytest.raises(ValueError, match="ok must be a boolean"):
            load_fixtures(path)


class TestScoring:
    def test_normalize_text_collapses_whitespace(self):
        assert normalize_text("Held  at\n6.5   percent") == "held at 6.5 percent"

    def test_term_coverage(self):
        result = term_coverage("Revenue was 412 MILLION dollars", ["revenue", "412 million", "missing"])
        assert result["covered"] == ["revenue", "412 million"]
        assert result["missing"] == ["missing"]
        assert result["coverage"] == pytest.approx(2 / 3)
        assert term_coverage("anything", [])["coverage"] == 1.0

    def test_number_token_extraction(self):
        tokens = extract_number_tokens("Revenue of 412 million dollars, margin 61 percent, $1.85 per share")
        assert "412million" in tokens
        assert "61" in tokens  # units are intentionally not compared
        assert "1.85" in tokens

    def test_number_tokens_ignore_comma_and_trailing_zero_formats(self):
        assert extract_number_tokens("25,000 tonnes") == extract_number_tokens("25000 tonnes")
        assert extract_number_tokens("238.10") == extract_number_tokens("238.1")

    def test_numeric_support_flags_invented_quantities(self):
        evidence = "The company paid a dividend of 1.85 euros per share."
        supported = numeric_claim_support("The dividend was 1.85 euros per share.", evidence)
        assert supported["unsupported_count"] == 0
        invented = numeric_claim_support("The dividend was 9.99 euros and revenue hit 999 million.", evidence)
        assert set(invented["unsupported"]) == {"9.99", "999million"}
        assert invented["unsupported_rate"] == 1.0

    def test_numeric_support_does_not_confuse_magnitudes(self):
        support = numeric_claim_support("Revenue was 412 billion.", "Revenue was 412 million.")
        assert support["unsupported"] == ["412billion"]

    def test_citation_checks(self):
        ok = citation_checks(["a.md"], expected_sources=["a.md"],
                             evidence_sources={"a.md", "b.md"}, min_citations=1)
        assert ok["presence"] and ok["citation_correctness"] == 1.0 and not ok["fabricated"]

        fabricated = citation_checks(["ghost.pdf"], expected_sources=[],
                                     evidence_sources={"a.md"}, min_citations=0)
        assert fabricated["fabricated"] == ["ghost.pdf"]
        assert fabricated["citation_correctness"] == 0.0

        missing = citation_checks([], expected_sources=["a.md"],
                                  evidence_sources={"a.md"}, min_citations=1)
        assert not missing["presence"] and missing["citation_correctness"] == 0.0

    def test_tool_scoring_exact_match(self):
        result = score_tools(["search_documents"], [{"name": "search_documents", "ok": True}])
        assert result["selection_correct"] and result["execution_success_rate"] == 1.0

    def test_tool_scoring_missing_unexpected_unnecessary(self):
        result = score_tools(["get_stock_price"], [
            {"name": "get_stock_price", "ok": True},
            {"name": "get_stock_price", "ok": True},   # unnecessary duplicate
            {"name": "get_stock_news", "ok": False},   # unexpected + failed
        ])
        assert result["unexpected"] == ["get_stock_news"]
        assert result["unnecessary"] == ["get_stock_price"]
        assert not result["selection_correct"]
        assert result["execution_success_rate"] == pytest.approx(2 / 3)

    def test_tool_scoring_expected_duplicates_are_not_unnecessary(self):
        calls = [{"name": "get_income_statement", "ok": True} for _ in range(2)]
        result = score_tools(["get_income_statement", "get_income_statement"], calls)
        assert result["selection_correct"] and result["unnecessary"] == []

    def test_detector_flags_and_grounding(self):
        terms = {"missing": [], "coverage": 1.0}
        citations = {"presence": True, "fabricated": [], "expected_missing": [],
                     "citation_correctness": 1.0}
        numeric = {"unsupported_count": 0}
        tools = {"missing": [], "unexpected_count": 0, "unnecessary_count": 0,
                 "execution_success_rate": 1.0}
        assert detector_flags(terms, citations, numeric, tools) == []
        assert is_grounded(terms, citations, numeric) is True

        citations_bad = {**citations, "fabricated": ["x.pdf"], "citation_correctness": 0.5}
        assert "fabricated_citation" in detector_flags(terms, citations_bad, numeric, tools)
        assert is_grounded(terms, citations_bad, numeric) is False


class TestTelemetry:
    def test_missing_values_stay_null(self):
        telemetry = RunTelemetry.from_dict(None)
        assert telemetry.total_ms is None and telemetry.input_tokens is None
        summary = summarize_latency([])
        assert summary["n"] == 0 and summary["mean"] is None

    def test_p95_suppressed_for_small_samples(self):
        small = summarize_latency([1.0, 2.0, 3.0])
        assert small["p95"] is None and "suppressed" in small["note"]
        assert small["mean"] == 2.0 and small["p50"] == 2.0 and small["max"] == 3.0

    def test_p95_reported_with_enough_samples(self):
        big = summarize_latency([float(i) for i in range(1, 101)])
        assert big["p95"] == 95.0 and big["note"] is None

    def test_cost_never_invented(self):
        assert estimate_cost(1000, 500, None) is None
        assert estimate_cost(None, 500, {"input_per_1k": 1.0, "output_per_1k": 2.0}) is None

    def test_cost_with_configured_pricing(self):
        cost = estimate_cost(2000, 1000, {"input_per_1k": 0.15, "output_per_1k": 0.60})
        assert cost == pytest.approx(0.15 * 2 + 0.60 * 1)

    def test_aggregate_separates_missing_data(self):
        records = [
            RunTelemetry(total_ms=100.0, input_tokens=10, output_tokens=5, source="fixture"),
            RunTelemetry(total_ms=200.0, source="live"),  # no token data
        ]
        result = aggregate_telemetry(records)
        assert result["runs"] == 2
        assert result["sources"] == {"fixture": 1, "live": 1}
        assert result["latency_ms"]["total_ms"]["mean"] == 150.0
        assert result["tokens"]["input_tokens"]["total"] == 10
        assert result["tokens"]["runs_without_data"] == 1
        assert result["cost"]["total"] is None


class TestReport:
    def test_build_report_metrics_on_real_fixtures(self):
        fixtures = load_fixtures(FIXTURES_PATH)
        report = build_report(fixtures)
        answer, agent, detector = report["answer"], report["agent"], report["detector_self_check"]

        assert answer["cases"] == 10
        assert answer["fact_coverage"] == 1.0
        assert answer["grounded_answer_rate"] == 1.0
        assert answer["citation_coverage"] == 1.0
        assert answer["citation_correctness"] == 1.0
        assert answer["unsupported_claim_rate"] == 0.0

        assert agent["tool_selection_accuracy"] == 1.0
        assert agent["unexpected_tool_call_rate"] == 0.0
        assert agent["unnecessary_tool_call_rate"] == 0.0
        assert agent["tool_execution_success_rate"] == 1.0
        assert agent["total_tool_calls"] == 11

        assert detector["detection_rate"] == 1.0
        assert detector["fully_detected"] == 2

        # Machine-readable: the whole report must be JSON-serialisable.
        json.dumps(report)

    def test_retrieval_section_is_honestly_skipped(self):
        report = build_report(load_fixtures(FIXTURES_PATH), retrieval_section=None)
        assert report["retrieval"]["status"] == "skipped"

    def test_render_text_has_all_sections(self):
        report = build_report(load_fixtures(FIXTURES_PATH))
        text = render_text(report)
        for section in ("Retrieval", "Answer quality", "tool selection",
                        "Detector self-check", "Performance", "cost"):
            assert section in text

    def test_pricing_flows_into_cost(self):
        report = build_report(
            load_fixtures(FIXTURES_PATH),
            pricing={"input_per_1k": 0.15, "output_per_1k": 0.60},
        )
        cost = report["performance"]["cost"]
        assert cost["total"] is not None and cost["total"] > 0
        # Only fixtures with token data contribute.
        assert cost["n_with_data"] == 8

    def test_evaluation_is_deterministic(self):
        fixtures = load_fixtures(FIXTURES_PATH)
        first = build_report(fixtures)
        second = build_report(load_fixtures(FIXTURES_PATH))
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


class TestCli:
    def test_json_output_is_machine_readable(self, capsys):
        rc = report_mod._main(["--fixtures", str(FIXTURES_PATH), "--skip-retrieval",
                               "--format", "json"])
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["schema_version"] == 1
        assert report["retrieval"]["status"] == "skipped"
        assert report["answer"]["grounded_answer_rate"] == 1.0

    def test_json_out_writes_file(self, tmp_path, capsys):
        out = tmp_path / "report.json"
        rc = report_mod._main(["--fixtures", str(FIXTURES_PATH), "--skip-retrieval",
                               "--json-out", str(out)])
        assert rc == 0
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["agent"]["tool_selection_accuracy"] == 1.0
        assert "Phase 3 Evaluation Report" in capsys.readouterr().out

    def test_pricing_requires_both_sides(self):
        with pytest.raises(SystemExit):
            report_mod._main(["--fixtures", str(FIXTURES_PATH), "--skip-retrieval",
                              "--price-input-per-1k", "0.15"])

    def test_per_case_rows_cover_every_fixture(self):
        report = build_report(load_fixtures(FIXTURES_PATH))
        ids = [row["id"] for row in report["per_case"]]
        assert len(ids) == 12 and len(set(ids)) == 12
        neg = {row["id"]: row for row in report["per_case"] if row["synthetic_negative"]}
        assert set(neg) == {"neg-fabricated-citation", "neg-unsupported-numbers"}
        assert "fabricated_citation" in neg["neg-fabricated-citation"]["flags"]
        assert "unsupported_numbers" in neg["neg-unsupported-numbers"]["flags"]


class TestEvidenceHandling:
    def test_evidence_text_combines_chunks_and_tool_results(self):
        fixtures = {f.id: f for f in load_fixtures(FIXTURES_PATH)}
        rag = fixtures["rag-baltic-dividend"]
        assert "payout ratio" in evidence_text(rag)
        assert evidence_sources(rag) == ["baltic-freightways-profile.md"]

        tool = fixtures["tool-stock-price"]
        assert "182.45" in evidence_text(tool)  # from the tool result JSON
        assert evidence_sources(tool) == []
