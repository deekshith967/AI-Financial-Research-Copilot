"""Tests for hybrid retrieval (vector + FTS5 + RRF), dedup, limits, citations."""
import pytest

from MarketInsight.rag.models import Chunk, DocumentMeta, make_chunk_id
from MarketInsight.rag.retrieve import Retriever


def add_doc(store, embedder, *, title, ticker, texts, sha, path=None, doc_type="10-K"):
    meta = DocumentMeta(
        source_path=path or f"/docs/{title}.md", title=title, ticker=ticker,
        doc_type=doc_type, sha256=sha, ingested_at="2026-10-08T00:00:00Z",
    )
    chunks = [
        Chunk(chunk_id=make_chunk_id(meta.doc_id, i, t), doc_id=meta.doc_id,
              chunk_index=i, text=t, token_count=len(t.split()), section=f"S{i}")
        for i, t in enumerate(texts)
    ]
    store.ingest_document(meta, chunks, embedder.embed_documents([c.text for c in chunks]))
    return meta, chunks


def make_retriever(store, embedder, **overrides):
    params = dict(top_k=8, max_top_k=4, candidate_k=24, rrf_k=60,
                  max_context_chars=6000)
    params.update(overrides)
    return Retriever(store, embedder, **params)


CORPUS = [
    ("helium supply constraints affect production output", "AAA", "a1" * 32),
    ("helium balloon festival celebration party fun times", "AAA", "b2" * 32),
    # Topically unrelated doc under a different ticker. It still contains the
    # token "helium" so the ticker-filter test exercises the FTS signal (weak
    # single-token vector matches are below the similarity floor by design).
    ("unrelated zebra content with helium vocabulary and different terminology", "BBB", "c3" * 32),
]


@pytest.fixture
def corpus_store(make_store, hash_embedder):
    store = make_store()
    for i, (text, ticker, sha) in enumerate(CORPUS):
        add_doc(store, hash_embedder, title=f"Doc{i}", ticker=ticker,
                texts=[text], sha=sha)
    return store


class TestHybridRanking:
    def test_best_match_ranks_first(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        result = retriever.search("helium supply constraints")
        assert result.chunks
        assert "constraints affect production" in result.chunks[0].chunk.text

    def test_chunk_in_both_signals_reports_both_sources(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        result = retriever.search("helium supply")
        top = result.chunks[0]
        assert set(top.match_sources) == {"vector", "fts"}

    def test_fts_recovers_rare_exact_terms(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        result = retriever.search("zebra vocabulary")
        assert any("zebra" in r.chunk.text for r in result.chunks)

    def test_stats_are_populated(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        stats = retriever.search("helium supply").stats
        assert stats.vector_hits > 0 and stats.fts_hits > 0
        assert stats.fused_candidates >= stats.returned > 0
        assert stats.latency_ms >= 0 and stats.reranked is False


class TestEdgeCases:
    def test_empty_query_returns_nothing(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        assert retriever.search("").chunks == ()
        assert retriever.search("   ").chunks == ()

    def test_empty_index_returns_nothing(self, make_store, hash_embedder):
        retriever = make_retriever(make_store(), hash_embedder)
        result = retriever.search("anything at all")
        assert result.chunks == ()
        assert result.stats.returned == 0

    def test_query_with_fts_operators_does_not_crash(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        result = retriever.search('helium AND (supply OR "weird") NEAR* :boom')
        assert result.chunks  # sanitised query still matches helium chunks

    def test_vector_floor_drops_unrelated_query(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        # KNN always returns the k nearest chunks, even for queries sharing no
        # vocabulary with the corpus; the similarity floor must drop them so an
        # off-topic query surfaces nothing instead of garbage.
        result = retriever.search("photosynthesis chlorophyll mitochondria")
        assert result.chunks == ()
        assert result.stats.vector_hits == 0
        assert result.stats.fts_hits == 0

    def test_fts_matches_survive_below_vector_floor(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        # Single shared token: cosine is below the floor for every doc, but the
        # exact-term FTS matches are exempt and must still be returned.
        result = retriever.search("helium")
        assert result.chunks
        assert result.stats.vector_hits == 0
        assert all("fts" in r.match_sources for r in result.chunks)


class TestFiltersAndLimits:
    def test_ticker_filter_scopes_results(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        result = retriever.search("helium", ticker="BBB")
        assert result.chunks
        assert all(r.citation.ticker == "BBB" for r in result.chunks)

    def test_top_k_is_clamped_to_max(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        result = retriever.search("helium supply unrelated", top_k=100)
        assert len(result.chunks) <= 4

    def test_context_budget_trims_results(self, make_store, hash_embedder):
        store = make_store()
        texts = ["helium " + "word " * 30 + f"variant{i}" for i in range(4)]
        add_doc(store, hash_embedder, title="Big", ticker="AAA", texts=texts, sha="d4" * 32)
        retriever = make_retriever(store, hash_embedder, max_context_chars=200, top_k=8)
        result = retriever.search("helium")
        total = sum(len(r.chunk.text) for r in result.chunks)
        assert len(result.chunks) < 4
        assert total <= 200 + max(len(t) for t in texts)

    def test_duplicate_text_across_documents_removed(self, make_store, hash_embedder):
        store = make_store()
        duplicate_text = "identical disclosure language appears in both filings"
        add_doc(store, hash_embedder, title="FilingA", ticker="AAA",
                texts=[duplicate_text], sha="e5" * 32)
        add_doc(store, hash_embedder, title="FilingB", ticker="BBB",
                texts=[duplicate_text], sha="f6" * 32, path="/docs/other.md")
        retriever = make_retriever(store, hash_embedder)
        result = retriever.search("identical disclosure language")
        assert len(result.chunks) == 1
        assert result.stats.duplicates_removed == 1


class TestCitations:
    def test_every_result_carries_complete_citation(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        result = retriever.search("helium supply constraints")
        assert result.chunks
        for r in result.chunks:
            cit = r.citation
            assert cit.doc_id and cit.chunk_id
            assert cit.title and cit.source_path and cit.doc_type
            assert cit.chunk_id == r.chunk.chunk_id
            assert cit.section is not None
            payload = cit.to_dict()
            assert payload["label"] and payload["title"] == cit.title

    def test_ranks_are_sequential(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        result = retriever.search("helium")
        assert [r.rank for r in result.chunks] == list(range(1, len(result.chunks) + 1))


class TestDeterminism:
    def test_same_query_same_results(self, corpus_store, hash_embedder):
        retriever = make_retriever(corpus_store, hash_embedder)
        first = retriever.search("helium supply constraints")
        second = retriever.search("helium supply constraints")
        assert [r.chunk.chunk_id for r in first.chunks] == [r.chunk.chunk_id for r in second.chunks]
        assert [r.score for r in first.chunks] == [r.score for r in second.chunks]


class FakeReranker:
    """Scripted reranker: reverses the candidate order."""

    def rerank(self, query, documents):
        return [float(i) for i in range(len(documents))]  # later docs score higher


class FailingReranker:
    def rerank(self, query, documents):
        raise RuntimeError("model unavailable")


class TestReranking:
    def test_reranker_reorders_results(self, corpus_store, hash_embedder):
        baseline = make_retriever(corpus_store, hash_embedder).search("helium")
        reranked = make_retriever(
            corpus_store, hash_embedder, reranker=FakeReranker()
        ).search("helium")
        assert reranked.stats.reranked is True
        assert len(reranked.chunks) == len(baseline.chunks)
        assert [r.chunk.chunk_id for r in reranked.chunks] == [
            r.chunk.chunk_id for r in reversed(baseline.chunks)
        ]

    def test_failing_reranker_falls_back_to_fused_order(self, corpus_store, hash_embedder):
        baseline = make_retriever(corpus_store, hash_embedder).search("helium")
        result = make_retriever(
            corpus_store, hash_embedder, reranker=FailingReranker()
        ).search("helium")
        assert result.stats.reranked is False
        assert [r.chunk.chunk_id for r in result.chunks] == [
            r.chunk.chunk_id for r in baseline.chunks
        ]
