"""Document ingestion pipeline and CLI.

Pipeline per file: extract → normalize → dedup check (sha256 of the normalized
text) → chunk → embed → index in a single transaction. Ingestion is a local,
offline operation — it runs from the CLI or tests, never on the request path.

CLI:  python -m MarketInsight.rag.ingest <path...> [--ticker ZPHY] [--doc-type 10-K]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from config.settings import Settings, get_settings
from MarketInsight.rag.chunk import TokenCounter, chunk_text, tiktoken_counter
from MarketInsight.rag.embed import Embedder, FastEmbedder
from MarketInsight.rag.extract import SUPPORTED_EXTENSIONS, ExtractionError, extract_file
from MarketInsight.rag.models import Chunk, DocumentMeta, content_sha256, make_chunk_id, utc_now_iso
from MarketInsight.rag.retrieve import Retriever
from MarketInsight.rag.rerank import FastReranker
from MarketInsight.rag.store import DuplicateDocument, RagStore
from MarketInsight.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    path: str
    status: str  # "ingested" | "duplicate" | "error"
    doc_id: Optional[str] = None
    title: Optional[str] = None
    chunks: int = 0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path, "status": self.status, "doc_id": self.doc_id,
            "title": self.title, "chunks": self.chunks, "message": self.message,
        }


def _guess_doc_type(path: Path) -> str:
    name = path.stem.lower().replace("_", "-")
    if "10-k" in name or "10k" in name:
        return "10-K"
    if "10-q" in name or "10q" in name:
        return "10-Q"
    if "research" in name or "note" in name:
        return "research-note"
    if "outlook" in name or "market" in name:
        return "market-report"
    return "document"


def ingest_file(
    path: Union[str, Path],
    *,
    store: RagStore,
    embedder: Embedder,
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
    max_tokens: int = 512,
    overlap_tokens: int = 77,
    min_tokens: int = 40,
    count_tokens: Optional[TokenCounter] = None,
) -> IngestResult:
    path = Path(path)
    try:
        extracted = extract_file(path)
    except (ExtractionError, OSError) as exc:
        logger.warning("Ingestion failed for %s: %s", path, exc)
        return IngestResult(path=str(path), status="error", message=str(exc))

    sha256 = content_sha256(extracted.text)
    existing = store.has_content(sha256)
    if existing:
        return IngestResult(
            path=str(path), status="duplicate", doc_id=existing,
            title=extracted.title, message="Identical content is already indexed.",
        )

    ticker_norm = ticker.strip().upper() if ticker else None
    meta = DocumentMeta(
        source_path=str(path),
        title=extracted.title,
        ticker=ticker_norm,
        doc_type=doc_type or _guess_doc_type(path),
        sha256=sha256,
        ingested_at=utc_now_iso(),
    )
    drafts = chunk_text(
        extracted.text,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        count_tokens=count_tokens or tiktoken_counter(),
        headings=extracted.headings,
    )
    if not drafts:
        return IngestResult(
            path=str(path), status="error", title=extracted.title,
            message="Extraction produced no chunkable content.",
        )

    chunks = [
        Chunk(
            chunk_id=make_chunk_id(meta.doc_id, i, draft.text),
            doc_id=meta.doc_id,
            chunk_index=i,
            text=draft.text,
            token_count=draft.token_count,
            section=draft.section,
        )
        for i, draft in enumerate(drafts)
    ]
    embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
    try:
        store.ingest_document(meta, chunks, embeddings)
    except DuplicateDocument as dup:
        return IngestResult(
            path=str(path), status="duplicate", doc_id=dup.doc_id,
            title=extracted.title, message="Identical content is already indexed.",
        )
    return IngestResult(
        path=str(path), status="ingested", doc_id=meta.doc_id,
        title=extracted.title, chunks=len(chunks),
    )


def ingest_path(
    path: Union[str, Path],
    *,
    store: RagStore,
    embedder: Embedder,
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
    **chunk_kwargs,
) -> List[IngestResult]:
    """Ingest a single file or every supported file under a directory."""
    path = Path(path)
    if not path.exists():
        return [IngestResult(path=str(path), status="error", message="Path does not exist.")]
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            return [IngestResult(path=str(path), status="error",
                                 message="No supported documents found in directory.")]
    else:
        files = [path]
    return [
        ingest_file(f, store=store, embedder=embedder, ticker=ticker,
                    doc_type=doc_type, **chunk_kwargs)
        for f in files
    ]


# --------------------------------------------------------------------------------
# Default pipeline factory (used by the CLI, the agent tools and the eval runner)
# --------------------------------------------------------------------------------

def build_default_embedder(settings: Settings, data_dir: Optional[Union[str, Path]] = None) -> FastEmbedder:
    root = Path(data_dir or settings.rag_data_dir)
    return FastEmbedder(settings.rag_embedding_model, cache_dir=root / "models")


def build_default_retriever(
    settings: Settings,
    data_dir: Optional[Union[str, Path]] = None,
    *,
    embedder: Optional[Embedder] = None,
) -> Retriever:
    root = Path(data_dir or settings.rag_data_dir)
    emb = embedder or build_default_embedder(settings, root)
    store = RagStore(root / "index.db", dim=emb.dim)
    reranker = (
        FastReranker(settings.rag_rerank_model, cache_dir=root / "models")
        if settings.rag_rerank_enabled
        else None
    )
    return Retriever(
        store,
        emb,
        top_k=settings.rag_top_k,
        max_top_k=settings.rag_max_top_k,
        candidate_k=settings.rag_candidate_k,
        rrf_k=settings.rag_rrf_k,
        max_context_chars=settings.rag_max_context_chars,
        vector_min_score=settings.rag_vector_min_score,
        reranker=reranker,
        rerank_candidates=settings.rag_rerank_candidates,
    )


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest local documents (.txt/.md/.html) into the RAG index."
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
    parser.add_argument("--ticker", help="Ticker to attach to every ingested document")
    parser.add_argument("--doc-type", help="Document type label (default: guessed from filename)")
    parser.add_argument("--data-dir", help="RAG data directory (default: settings)")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.rag_enabled:
        print(json.dumps({"error": "RAG is disabled (RAG_ENABLED=false)."}))
        return 2

    root = Path(args.data_dir or settings.rag_data_dir)
    embedder = build_default_embedder(settings, root)
    store = RagStore(root / "index.db", dim=embedder.dim)

    results: List[IngestResult] = []
    try:
        for raw_path in args.paths:
            results.extend(
                ingest_path(
                    raw_path,
                    store=store,
                    embedder=embedder,
                    ticker=args.ticker,
                    doc_type=args.doc_type,
                    max_tokens=settings.rag_chunk_tokens,
                    overlap_tokens=settings.rag_chunk_overlap_tokens,
                    min_tokens=settings.rag_min_chunk_tokens,
                )
            )
    finally:
        stats = store.index_stats()
        store.close()

    summary = {
        "results": [r.to_dict() for r in results],
        "ingested": sum(1 for r in results if r.status == "ingested"),
        "duplicates": sum(1 for r in results if r.status == "duplicate"),
        "errors": sum(1 for r in results if r.status == "error"),
        "index": stats,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["ingested"] or summary["duplicates"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
