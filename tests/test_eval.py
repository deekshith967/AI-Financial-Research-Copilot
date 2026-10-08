"""Tests for the retrieval evaluation harness and the golden dataset baseline.

The baseline assertions run against the real fastembed model over the sample
corpus (skipped when the model is unavailable). Harness mechanics are also
exercised with the deterministic HashEmbedder so they run everywhere.
"""
from pathlib import Path

import pytest

from MarketInsight.rag.eval import evaluate, load_golden
from MarketInsight.rag.ingest import ingest_path
from MarketInsight.rag.retrieve import Retriever
from MarketInsight.rag.store import RagStore

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "eval" / "golden.jsonl"


def _retriever(store, embedder):
    return Retriever(store, embedder, top_k=8, max_top_k=16, candidate_k=24,
                     rrf_k=60, max_context_chars=6000)


class TestGoldenDataset:
    def test_loads_and_validates(self):
        questions = load_golden(GOLDEN_PATH)
        assert len(questions) >= 10
        for q in questions:
            assert q.question and q.source_file

    def test_rejects_empty_dataset(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no questions"):
            load_golden(empty)

    def test_rejects_entries_without_source(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"question": "no source here"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="source_file"):
            load_golden(bad)


class TestHarnessMechanics:
    """Harness plumbing with the deterministic embedder (no quality claims)."""

    def test_evaluate_returns_bounded_metrics(self, make_store, hash_embedder, samples_dir):
        store = make_store()
        ingest_path(samples_dir, store=store, embedder=hash_embedder,
                    count_tokens=lambda t: len(t.split()))
        report = evaluate(_retriever(store, hash_embedder), load_golden(GOLDEN_PATH), top_k=5)

        aggregate = report["aggregate"]
        assert aggregate["questions"] == len(report["per_question"])
        for key in ("recall@5", "precision@5", "mrr", "answer_term_coverage",
                    "citation_coverage", "duplicate_rate"):
            assert 0.0 <= aggregate[key] <= 1.0
        assert aggregate["avg_retrieved_chunks"] >= 0
        latency = aggregate["latency_ms"]
        assert 0 <= latency["p50"] <= latency["p95"] <= latency["max"]
        # The harness never invents citations: coverage counts real fields only.
        for row in report["per_question"]:
            if row["retrieved"]:
                assert row["citation_coverage"] == 1.0


@pytest.fixture(scope="module")
def real_eval_report(real_embedder, tmp_path_factory):
    samples_dir = ROOT / "samples" / "docs"
    store = RagStore(tmp_path_factory.mktemp("eval") / "index.db", dim=real_embedder.dim)
    ingest_path(samples_dir, store=store, embedder=real_embedder)
    report = evaluate(_retriever(store, real_embedder), load_golden(GOLDEN_PATH), top_k=5)
    store.close()
    return report


class TestGoldenBaseline:
    """Recorded baseline for the sample corpus with the real model."""

    def test_every_question_is_answerable(self, real_eval_report):
        misses = [r["question"] for r in real_eval_report["per_question"] if not r["hit"]]
        assert misses == []
        assert real_eval_report["aggregate"]["recall@5"] == 1.0

    def test_snippets_carry_the_answer_terms(self, real_eval_report):
        gaps = [r["question"] for r in real_eval_report["per_question"] if r["missing_terms"]]
        assert gaps == []
        assert real_eval_report["aggregate"]["answer_term_coverage"] == 1.0

    def test_every_chunk_is_fully_cited(self, real_eval_report):
        assert real_eval_report["aggregate"]["citation_coverage"] == 1.0

    def test_ranking_is_strong(self, real_eval_report):
        assert real_eval_report["aggregate"]["mrr"] >= 0.75
