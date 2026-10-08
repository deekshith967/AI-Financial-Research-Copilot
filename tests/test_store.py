"""Tests for the SQLite + sqlite-vec + FTS5 store."""
import math

import pytest

from MarketInsight.rag.models import Chunk, DocumentMeta, make_chunk_id
from MarketInsight.rag.store import (
    DuplicateDocument,
    RagStore,
    StoreError,
    fts_query_from_text,
)


def make_meta(title="Test Doc", ticker="ACME", sha="a" * 64, path="/docs/a.md", doc_type="10-K"):
    return DocumentMeta(
        source_path=path, title=title, ticker=ticker, doc_type=doc_type,
        sha256=sha, ingested_at="2026-10-08T00:00:00Z",
    )


def make_chunks(doc_id, texts):
    return [
        Chunk(chunk_id=make_chunk_id(doc_id, i, t), doc_id=doc_id, chunk_index=i,
              text=t, token_count=len(t.split()), section=f"Section {i}")
        for i, t in enumerate(texts)
    ]


def unit_vec(dim, index):
    vec = [0.0] * dim
    vec[index % dim] = 1.0
    return vec


class TestSchemaAndDim:
    def test_fresh_store_has_zero_counts(self, make_store):
        store = make_store(dim=64)
        assert store.document_count() == 0
        assert store.chunk_count() == 0
        assert store.dim == 64

    def test_reopen_with_wrong_dim_rejected(self, tmp_path):
        store = RagStore(tmp_path / "i.db", dim=64)
        store.close()
        with pytest.raises(StoreError, match="dim"):
            RagStore(tmp_path / "i.db", dim=128)

    def test_embedding_dim_mismatch_on_insert_rejected(self, make_store):
        store = make_store(dim=64)
        meta = make_meta()
        chunks = make_chunks(meta.doc_id, ["hello world"])
        with pytest.raises(StoreError, match="dim"):
            store.ingest_document(meta, chunks, [unit_vec(32, 0)])


class TestIngestAndDedup:
    def test_ingest_document_and_counts(self, make_store):
        store = make_store()
        meta = make_meta()
        texts = ["revenue grew strongly", "risk factors include helium supply"]
        store.ingest_document(meta, make_chunks(meta.doc_id, texts),
                              [unit_vec(64, 0), unit_vec(64, 1)])
        assert store.document_count() == 1
        assert store.chunk_count() == 2

    def test_duplicate_content_rejected(self, make_store):
        store = make_store()
        meta = make_meta()
        chunks = make_chunks(meta.doc_id, ["some text here"])
        store.ingest_document(meta, chunks, [unit_vec(64, 0)])
        with pytest.raises(DuplicateDocument) as excinfo:
            store.ingest_document(meta, chunks, [unit_vec(64, 0)])
        assert excinfo.value.doc_id == meta.doc_id
        assert store.document_count() == 1
        assert store.has_content(meta.sha256) == meta.doc_id

    def test_empty_chunk_list_rejected(self, make_store):
        store = make_store()
        with pytest.raises(ValueError):
            store.ingest_document(make_meta(), [], [])


class TestVectorSearch:
    def test_knn_orders_by_similarity(self, make_store):
        store = make_store()
        meta = make_meta()
        texts = ["chunk zero", "chunk one", "chunk two"]
        chunks = make_chunks(meta.doc_id, texts)
        store.ingest_document(meta, chunks, [unit_vec(64, 0), unit_vec(64, 1), unit_vec(64, 2)])
        hits = store.vector_search(unit_vec(64, 1), k=3)
        assert hits[0][0] == chunks[1].chunk_id
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
        assert {h[0] for h in hits} == {c.chunk_id for c in chunks}

    def test_k_limits_results(self, make_store):
        store = make_store()
        meta = make_meta()
        chunks = make_chunks(meta.doc_id, [f"text {i}" for i in range(5)])
        store.ingest_document(meta, chunks, [unit_vec(64, i) for i in range(5)])
        assert len(store.vector_search(unit_vec(64, 0), k=2)) == 2
        assert store.vector_search(unit_vec(64, 0), k=0) == []


class TestFtsSearch:
    def test_keyword_match_returns_right_chunk(self, make_store):
        store = make_store()
        meta = make_meta()
        texts = ["revenue grew twelve percent", "helium supply is constrained"]
        chunks = make_chunks(meta.doc_id, texts)
        store.ingest_document(meta, chunks, [unit_vec(64, 0), unit_vec(64, 1)])
        hits = store.fts_search(fts_query_from_text("helium supply"), k=5)
        assert hits and hits[0][0] == chunks[1].chunk_id

    def test_no_match_returns_empty(self, make_store):
        store = make_store()
        meta = make_meta()
        store.ingest_document(meta, make_chunks(meta.doc_id, ["ordinary words here"]),
                              [unit_vec(64, 0)])
        assert store.fts_search(fts_query_from_text("zzzqxx"), k=5) == []

    def test_fts_query_sanitizes_operators(self):
        query = fts_query_from_text('revenue AND (growth) OR "quote" NEAR* :weird')
        assert '"' in query and " OR " in query
        for operator in ("AND", "NEAR", "*", "(", ")"):
            assert operator not in query.replace(" OR ", "")
        # must not raise on the sqlite side
        assert fts_query_from_text("*** (((") == ""

    def test_fts_query_caps_terms(self):
        query = fts_query_from_text(" ".join(f"t{i}" for i in range(50)))
        assert query.count(" OR ") + 1 == 12


class TestFetchAndCitations:
    def test_get_chunks_with_citations(self, make_store):
        store = make_store()
        meta = make_meta(title="Zephyr 10-K", ticker="ZPHY", path="/docs/zeph.md")
        chunks = make_chunks(meta.doc_id, ["alpha text", "beta text"])
        store.ingest_document(meta, chunks, [unit_vec(64, 0), unit_vec(64, 1)])
        fetched = store.get_chunks_with_citations([chunks[0].chunk_id, chunks[1].chunk_id, "missing"])
        assert len(fetched) == 2
        chunk, citation = fetched[chunks[0].chunk_id]
        assert chunk.text == "alpha text"
        assert citation.title == "Zephyr 10-K"
        assert citation.ticker == "ZPHY"
        assert citation.source_path == "/docs/zeph.md"
        assert citation.doc_type == "10-K"
        assert citation.section == "Section 0"
        assert citation.chunk_index == 0
        assert "Zephyr 10-K" in citation.label()

    def test_list_documents_and_ticker_filter(self, make_store):
        store = make_store()
        meta_a = make_meta(title="A", ticker="AAA", sha="a" * 64)
        meta_b = make_meta(title="B", ticker="BBB", sha="b" * 64, path="/docs/b.md")
        store.ingest_document(meta_a, make_chunks(meta_a.doc_id, ["text a"]), [unit_vec(64, 0)])
        store.ingest_document(meta_b, make_chunks(meta_b.doc_id, ["text b1", "text b2"]),
                              [unit_vec(64, 1), unit_vec(64, 2)])
        assert len(store.list_documents()) == 2
        only_b = store.list_documents(ticker="bbb")
        assert len(only_b) == 1 and only_b[0]["title"] == "B"
        assert only_b[0]["chunk_count"] == 2


class TestDeleteAndPersistence:
    def test_delete_document_cascades_everywhere(self, make_store):
        store = make_store()
        meta = make_meta()
        chunks = make_chunks(meta.doc_id, ["helium supply risk", "revenue growth"])
        store.ingest_document(meta, chunks, [unit_vec(64, 0), unit_vec(64, 1)])
        assert store.delete_document(meta.doc_id) is True
        assert store.document_count() == 0 and store.chunk_count() == 0
        assert store.vector_search(unit_vec(64, 0), k=5) == []
        assert store.fts_search(fts_query_from_text("helium"), k=5) == []
        assert store.delete_document(meta.doc_id) is False  # idempotent

    def test_index_survives_close_and_reopen(self, tmp_path):
        path = tmp_path / "persist.db"
        store = RagStore(path, dim=64)
        meta = make_meta()
        chunks = make_chunks(meta.doc_id, ["persistent helium text"])
        store.ingest_document(meta, chunks, [unit_vec(64, 0)])
        store.close()

        reopened = RagStore(path, dim=64)
        assert reopened.document_count() == 1
        assert reopened.chunk_count() == 1
        hits = reopened.vector_search(unit_vec(64, 0), k=1)
        assert hits and hits[0][0] == chunks[0].chunk_id
        fts = reopened.fts_search(fts_query_from_text("helium"), k=1)
        assert fts and fts[0][0] == chunks[0].chunk_id
        reopened.close()
