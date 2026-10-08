"""Tests for the ingestion pipeline."""
from pathlib import Path

import pytest

from MarketInsight.rag.ingest import ingest_file, ingest_path
from MarketInsight.rag.retrieve import Retriever
from tests.conftest import word_count

CHUNK_KW = dict(max_tokens=512, overlap_tokens=77, min_tokens=40, count_tokens=word_count)


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestIngestFile:
    def test_ingest_txt_success(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        path = write(tmp_path, "note.txt", "Helium markets tightened sharply in Q3.\n\nPrices rose.")
        result = ingest_file(path, store=store, embedder=hash_embedder, ticker=" zphy ", **CHUNK_KW)
        assert result.status == "ingested"
        assert result.chunks >= 1 and result.doc_id
        assert store.document_count() == 1
        doc = store.list_documents()[0]
        assert doc["ticker"] == "ZPHY"  # normalized

    def test_markdown_title_and_sections_preserved(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        path = write(tmp_path, "report.md",
                     "# Aurora Initiation\n\nSummary text here.\n\n## Key Risks\n\nBrine grade risk.")
        result = ingest_file(path, store=store, embedder=hash_embedder,
                             max_tokens=8, overlap_tokens=0, min_tokens=1,
                             count_tokens=word_count)
        assert result.status == "ingested" and result.title == "Aurora Initiation"
        chunk_rows = store._db.execute("SELECT chunk_id FROM chunks").fetchall()
        fetched = store.get_chunks_with_citations([r["chunk_id"] for r in chunk_rows])
        assert all(cit.title == "Aurora Initiation" for _, cit in fetched.values())
        assert any(c.section == "Key Risks" for c, _ in fetched.values())

    def test_doc_type_guessed_from_filename(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        path = write(tmp_path, "acme-10k-2025.md", "# Acme\n\nBusiness overview text.")
        result = ingest_file(path, store=store, embedder=hash_embedder, **CHUNK_KW)
        assert store.list_documents()[0]["doc_type"] == "10-K"

    def test_duplicate_ingestion_skipped(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        path = write(tmp_path, "note.txt", "Identical content body.")
        first = ingest_file(path, store=store, embedder=hash_embedder, **CHUNK_KW)
        second = ingest_file(path, store=store, embedder=hash_embedder, **CHUNK_KW)
        assert first.status == "ingested"
        assert second.status == "duplicate"
        assert second.doc_id == first.doc_id
        assert store.document_count() == 1

    def test_same_content_different_file_is_duplicate(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        one = write(tmp_path, "a.txt", "Same body of text.")
        two = write(tmp_path, "b.txt", "Same body of text.")
        assert ingest_file(one, store=store, embedder=hash_embedder, **CHUNK_KW).status == "ingested"
        assert ingest_file(two, store=store, embedder=hash_embedder, **CHUNK_KW).status == "duplicate"

    def test_empty_file_is_error_not_crash(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        result = ingest_file(write(tmp_path, "empty.txt", ""), store=store,
                             embedder=hash_embedder, **CHUNK_KW)
        assert result.status == "error" and result.message
        assert store.document_count() == 0

    def test_binary_file_is_error(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        path = tmp_path / "fake.txt"
        path.write_bytes(b"\x00\x01\x02\x03binary")
        result = ingest_file(path, store=store, embedder=hash_embedder, **CHUNK_KW)
        assert result.status == "error"

    def test_unsupported_extension_is_error(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        result = ingest_file(write(tmp_path, "doc.pdf", "fake pdf"), store=store,
                             embedder=hash_embedder, **CHUNK_KW)
        assert result.status == "error" and "Unsupported" in result.message

    def test_missing_path_is_error(self, make_store, hash_embedder):
        store = make_store()
        results = ingest_path("/nonexistent/file.txt", store=store, embedder=hash_embedder)
        assert results[0].status == "error"


class TestIngestDirectory:
    def test_directory_ingest_processes_supported_files(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        write(tmp_path, "one.txt", "First body of text about helium.")
        write(tmp_path, "two.md", "# Two\n\nSecond body about lithium.")
        (tmp_path / "skip.pdf").write_text("nope", encoding="utf-8")
        results = ingest_path(tmp_path, store=store, embedder=hash_embedder, **CHUNK_KW)
        assert len(results) == 2  # pdf skipped by extension filter
        assert all(r.status == "ingested" for r in results)
        assert store.document_count() == 2

    def test_empty_directory_reports_error(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        results = ingest_path(tmp_path, store=store, embedder=hash_embedder)
        assert results[0].status == "error"


class TestIngestThenRetrieve:
    def test_ingested_document_is_searchable_with_citation(self, tmp_path, make_store, hash_embedder):
        store = make_store()
        path = write(tmp_path, "zephyr.md",
                     "# Zephyr 10-K\n\n## Risk Factors\n\nHelium supply disruption risk is material.")
        ingest_file(path, store=store, embedder=hash_embedder, ticker="ZPHY",
                    max_tokens=8, overlap_tokens=0, min_tokens=1, count_tokens=word_count)
        retriever = Retriever(store, hash_embedder, top_k=4, max_top_k=8,
                              candidate_k=10, rrf_k=60, max_context_chars=6000)
        result = retriever.search("helium supply disruption")
        assert result.chunks
        top = result.chunks[0]
        assert "Helium supply disruption" in top.chunk.text
        assert top.citation.title == "Zephyr 10-K"
        assert top.citation.ticker == "ZPHY"
        assert top.citation.section == "Risk Factors"
