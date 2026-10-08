"""Tests for the agent-facing RAG tools and their registration on the agent.

Every failure mode must return the shared error envelope (never raise) so the
agent loop and the 15 market-data tools keep working when RAG is unavailable.
"""
import dataclasses

import pytest

import MarketInsight.rag.ingest as rag_ingest
import MarketInsight.rag.tools as rag_tools
from config.settings import get_settings
from MarketInsight.rag.models import Chunk, DocumentMeta, make_chunk_id
from MarketInsight.rag.retrieve import Retriever


@pytest.fixture(autouse=True)
def _clean_retriever_singleton():
    rag_tools.reset_retriever()
    yield
    rag_tools.reset_retriever()


def _patch_settings(monkeypatch, **overrides):
    settings = dataclasses.replace(get_settings(), **overrides)
    monkeypatch.setattr(rag_tools, "get_settings", lambda: settings)
    return settings


def _add_doc(store, embedder, *, title, ticker, text, sha, doc_type="10-K"):
    meta = DocumentMeta(
        source_path=f"/docs/{title}.md", title=title, ticker=ticker,
        doc_type=doc_type, sha256=sha, ingested_at="2026-10-08T00:00:00Z",
    )
    chunk = Chunk(
        chunk_id=make_chunk_id(meta.doc_id, 0, text), doc_id=meta.doc_id,
        chunk_index=0, text=text, token_count=len(text.split()), section="Overview",
    )
    store.ingest_document(meta, [chunk], embedder.embed_documents([text]))
    return meta, chunk


def _make_retriever(store, embedder):
    return Retriever(store, embedder, top_k=8, max_top_k=4, candidate_k=24,
                     rrf_k=60, max_context_chars=6000)


class TestSearchDocuments:
    def test_ok_envelope_with_traceable_citations(self, make_store, hash_embedder):
        store = make_store()
        _add_doc(store, hash_embedder, title="Filing", ticker="AAA",
                 text="helium supply constraints affect production output", sha="aa" * 32)
        rag_tools.set_retriever(_make_retriever(store, hash_embedder))

        out = rag_tools.search_documents.invoke({"query": "helium supply constraints"})

        assert out["status"] == "ok"
        assert out["tool"] == "search_documents"
        assert out["query"] == "helium supply constraints"
        assert out["source"]["provider"] == "Local document index"
        assert out["as_of"] and out["cached"] is False
        assert "cite" in out["note"].lower()

        data = out["data"]
        assert data["result_count"] >= 1
        assert data["retrieval"]["latency_ms"] >= 0
        first = data["results"][0]
        assert first["rank"] == 1 and first["matched_by"]
        assert [r["rank"] for r in data["results"]] == list(range(1, len(data["results"]) + 1))

        # The citation must trace back to a real indexed chunk.
        citation = first["citation"]
        assert citation["doc_id"] and citation["chunk_id"]
        assert citation["title"] == "Filing" and citation["ticker"] == "AAA"
        assert citation["label"]
        fetched = store.get_chunks_with_citations([citation["chunk_id"]])
        assert citation["chunk_id"] in fetched
        chunk, _ = fetched[citation["chunk_id"]]
        assert first["snippet"] == chunk.text  # short text: no truncation

    def test_snippet_is_capped(self, make_store, hash_embedder, monkeypatch):
        store = make_store()
        long_text = "helium " + "filler " * 300  # ~2KB, one chunk
        _add_doc(store, hash_embedder, title="Long", ticker="AAA",
                 text=long_text, sha="bb" * 32)
        _patch_settings(monkeypatch, rag_snippet_max_chars=120)
        rag_tools.set_retriever(_make_retriever(store, hash_embedder))

        out = rag_tools.search_documents.invoke({"query": "helium"})

        assert out["status"] == "ok"
        snippet = out["data"]["results"][0]["snippet"]
        assert len(snippet) <= 123
        assert snippet.endswith("...")

    def test_no_data_for_off_topic_query(self, make_store, hash_embedder):
        store = make_store()
        _add_doc(store, hash_embedder, title="Filing", ticker="AAA",
                 text="helium supply constraints affect production output", sha="cc" * 32)
        rag_tools.set_retriever(_make_retriever(store, hash_embedder))

        out = rag_tools.search_documents.invoke({"query": "photosynthesis chlorophyll"})

        assert out["status"] == "no_data"
        assert out["error_type"] == "no_data"
        assert "fabricate" in out["guidance"].lower()

    def test_empty_query_is_invalid_input(self):
        for query in ("", "   "):
            out = rag_tools.search_documents.invoke({"query": query})
            assert out["status"] == "error"
            assert out["error_type"] == "invalid_input"

    def test_disabled_returns_envelope_without_touching_retriever(self, monkeypatch):
        _patch_settings(monkeypatch, rag_enabled=False)
        # No retriever injected on purpose: the disabled path must not build one.
        out = rag_tools.search_documents.invoke({"query": "anything"})
        assert out["status"] == "error"
        assert out["error_type"] == "disabled"
        assert "fabricate" in out["guidance"].lower()

    def test_rag_unavailable_is_latched_when_build_fails(self, monkeypatch):
        calls = []

        def boom(settings):
            calls.append(1)
            raise RuntimeError("model download failed")

        monkeypatch.setattr(rag_ingest, "build_default_retriever", boom)

        first = rag_tools.search_documents.invoke({"query": "helium"})
        second = rag_tools.search_documents.invoke({"query": "helium"})

        for out in (first, second):
            assert out["status"] == "error"
            assert out["error_type"] == "rag_unavailable"
            assert "fabricate" in out["guidance"].lower()
        # The failure is latched: the expensive build is not retried per call.
        assert len(calls) == 1

    def test_internal_error_when_retriever_raises(self):
        class BrokenRetriever:
            def search(self, query, *, ticker=None):
                raise RuntimeError("disk corruption")

        rag_tools.set_retriever(BrokenRetriever())
        out = rag_tools.search_documents.invoke({"query": "helium"})
        assert out["status"] == "error"
        assert out["error_type"] == "internal_error"
        assert "fabricate" in out["guidance"].lower()

    def test_ticker_is_normalised_and_filters_results(self, make_store, hash_embedder):
        store = make_store()
        _add_doc(store, hash_embedder, title="A", ticker="AAA",
                 text="helium supply constraints affect production output", sha="dd" * 32)
        _add_doc(store, hash_embedder, title="B", ticker="BBB",
                 text="unrelated zebra content with helium vocabulary", sha="ee" * 32)
        rag_tools.set_retriever(_make_retriever(store, hash_embedder))

        out = rag_tools.search_documents.invoke({"query": "helium", "ticker": "bbb"})

        assert out["status"] == "ok"
        assert out["ticker"] == "BBB"
        assert out["data"]["result_count"] >= 1
        assert all(r["citation"]["ticker"] == "BBB" for r in out["data"]["results"])


class TestListIngestedDocuments:
    def test_ok_lists_ingested_documents(self, make_store, hash_embedder, tmp_path, monkeypatch):
        store = make_store()  # tmp_path / "index.db"
        _add_doc(store, hash_embedder, title="TenK", ticker="AAA",
                 text="helium supply constraints affect production output", sha="ff" * 32)
        _add_doc(store, hash_embedder, title="Note", ticker="BBB",
                 text="zebra migration across the savannah", sha="11" * 32,
                 doc_type="research-note")
        _patch_settings(monkeypatch, rag_data_dir=str(tmp_path))

        out = rag_tools.list_ingested_documents.invoke({})

        assert out["status"] == "ok"
        assert out["tool"] == "list_ingested_documents"
        data = out["data"]
        assert data["document_count"] == 2
        by_ticker = {d["ticker"]: d for d in data["documents"]}
        assert set(by_ticker) == {"AAA", "BBB"}
        ten_k = by_ticker["AAA"]
        assert ten_k["title"] == "TenK" and ten_k["doc_type"] == "10-K"
        assert ten_k["chunk_count"] == 1
        assert ten_k["doc_id"] and ten_k["ingested_at"] and ten_k["source_path"]

    def test_ticker_filter(self, make_store, hash_embedder, tmp_path, monkeypatch):
        store = make_store()
        _add_doc(store, hash_embedder, title="TenK", ticker="AAA",
                 text="helium supply constraints affect production output", sha="22" * 32)
        _patch_settings(monkeypatch, rag_data_dir=str(tmp_path))

        out = rag_tools.list_ingested_documents.invoke({"ticker": "aaa"})
        assert out["status"] == "ok"
        assert out["ticker"] == "AAA"
        assert [d["ticker"] for d in out["data"]["documents"]] == ["AAA"]

        missing = rag_tools.list_ingested_documents.invoke({"ticker": "ZZZ"})
        assert missing["status"] == "no_data"
        assert missing["error_type"] == "no_data"

    def test_no_data_on_empty_index(self, make_store, tmp_path, monkeypatch):
        make_store()
        _patch_settings(monkeypatch, rag_data_dir=str(tmp_path))
        out = rag_tools.list_ingested_documents.invoke({})
        assert out["status"] == "no_data"
        assert out["error_type"] == "no_data"

    def test_rag_unavailable_when_index_cannot_be_opened(self, tmp_path, monkeypatch):
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("a file blocks index.db creation", encoding="utf-8")
        _patch_settings(monkeypatch, rag_data_dir=str(blocker))
        out = rag_tools.list_ingested_documents.invoke({})
        assert out["status"] == "error"
        assert out["error_type"] == "rag_unavailable"

    def test_disabled(self, monkeypatch):
        _patch_settings(monkeypatch, rag_enabled=False)
        out = rag_tools.list_ingested_documents.invoke({})
        assert out["status"] == "error"
        assert out["error_type"] == "disabled"


class TestAgentRegistration:
    def test_all_seventeen_tools_registered(self):
        from MarketInsight.components.agent import TOOLS

        names = [t.name for t in TOOLS]
        assert len(TOOLS) == 17
        # The original 15 market-data tools are untouched...
        assert names[:15] == [
            "get_stock_price", "get_historical_data", "get_stock_news",
            "get_balance_sheet", "get_income_statement", "get_cash_flow",
            "get_company_info", "get_dividends", "get_splits",
            "get_institutional_holders", "get_major_shareholders",
            "get_mutual_fund_holders", "get_insider_transactions",
            "get_analyst_recommendations", "get_ticker",
        ]
        # ...and the two RAG tools are appended.
        assert names[15:] == ["search_documents", "list_ingested_documents"]
