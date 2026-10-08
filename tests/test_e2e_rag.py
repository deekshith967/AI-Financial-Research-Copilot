"""End-to-end offline RAG test: a real document file goes through extraction,
chunking, embedding, indexing, retrieval and citation — no LLM involved.

Uses the real fastembed model (skipped when unavailable) and the persisted
index is reopened from disk before searching. The same flow with a
deterministic test double is covered in tests/test_ingest.py.
"""
from MarketInsight.rag.ingest import ingest_file
from MarketInsight.rag.retrieve import Retriever
from MarketInsight.rag.store import RagStore


class TestEndToEndRag:
    def test_file_to_cited_retrieval(self, real_embedder, samples_dir, tmp_path):
        db_path = tmp_path / "index.db"
        source = samples_dir / "zephyr-systems-10k.md"

        store = RagStore(db_path, dim=real_embedder.dim)
        result = ingest_file(source, store=store, embedder=real_embedder,
                             ticker="ZPHY", doc_type="10-K")
        assert result.status == "ingested"
        assert result.chunks >= 2  # the sample 10-K spans several sections
        store.close()

        # Reopen from disk: retrieval runs from the persisted index alone.
        store = RagStore(db_path, dim=real_embedder.dim)
        try:
            retriever = Retriever(store, real_embedder, top_k=5, max_top_k=8,
                                  candidate_k=24, rrf_k=60, max_context_chars=6000)
            out = retriever.search("What are the risks from helium supply disruption?",
                                   ticker="ZPHY")

            assert out.chunks
            assert out.stats.vector_hits > 0
            top = out.chunks[0]
            assert "helium" in top.chunk.text.lower()

            citation = top.citation
            assert citation.ticker == "ZPHY"
            assert citation.doc_type == "10-K"
            assert citation.title and citation.label()
            assert citation.section  # heading metadata survived the whole pipeline
            assert citation.source_path.endswith("zephyr-systems-10k.md")

            # Every returned chunk carries a citation traceable to the index.
            for r in out.chunks:
                fetched = store.get_chunks_with_citations([r.citation.chunk_id])
                assert r.citation.chunk_id in fetched
        finally:
            store.close()
