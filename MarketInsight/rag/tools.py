"""Agent-facing LangChain tools for the local RAG subsystem.

Both tools follow the same response-envelope conventions as the market-data
tools and degrade gracefully: a missing index, an unavailable embedding
model, or RAG being disabled returns an error envelope instead of raising,
so the agent and the 15 financial tools keep working unaffected.

The retriever is built lazily on first use (the embedding model is large),
and tests can inject a pre-built retriever through ``set_retriever``.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from langchain.tools import tool

from config.settings import get_settings
from MarketInsight.rag.models import utc_now_iso
from MarketInsight.utils.logger import get_logger

logger = get_logger("RAG-Tools")

_SOURCE = {"provider": "Local document index", "via": "fastembed + SQLite (sqlite-vec + FTS5)"}
_NO_FABRICATION_HINT = "Do not estimate or fabricate this data; tell the user it is unavailable."

_lock = threading.Lock()
_retriever: Any = None
_retriever_failed = False


def _get_retriever(settings):
    """Build the shared retriever once; remember failures so we do not rebuild
    on every turn (the agent would otherwise pay the model-load cost repeatedly)."""
    global _retriever, _retriever_failed
    if _retriever is not None or _retriever_failed:
        return _retriever
    with _lock:
        if _retriever is None and not _retriever_failed:
            try:
                from MarketInsight.rag.ingest import build_default_retriever

                _retriever = build_default_retriever(settings)
                logger.info("RAG retriever initialised")
            except Exception as exc:  # noqa: BLE001 - any failure must not break the agent
                _retriever_failed = True
                logger.error("RAG retriever unavailable (%s)", type(exc).__name__)
    return _retriever


def set_retriever(retriever) -> None:
    """Test hook: inject a pre-built retriever."""
    global _retriever, _retriever_failed
    with _lock:
        _retriever = retriever
        _retriever_failed = False


def reset_retriever() -> None:
    """Test hook: clear the singleton so the next call rebuilds it."""
    global _retriever, _retriever_failed
    with _lock:
        _retriever = None
        _retriever_failed = False


def _fail(tool_name: str, error_type: str, message: str, *, subject: dict) -> dict:
    status = "no_data" if error_type == "no_data" else "error"
    response: dict[str, Any] = {
        "status": status,
        "tool": tool_name,
        **subject,
        "error_type": error_type,
        "message": message,
        "source": dict(_SOURCE),
    }
    if error_type in ("no_data", "rag_unavailable", "disabled", "internal_error"):
        response["guidance"] = _NO_FABRICATION_HINT
    return response


@tool(
    "search_documents",
    description=(
        "Search the local research document index (ingested filings, research notes and "
        "market reports) for long-form qualitative context that market-data tools do not "
        "provide, such as risk factors, management discussion, strategy or dividend policy. "
        "Returns ranked excerpts, each with a citation (title, doc_type, ticker, section, "
        "chunk). Only covers documents that were explicitly ingested; use "
        "list_ingested_documents to check coverage. If results are empty or off-topic, say "
        "so and answer from other tools or general knowledge instead; never invent sources."
    ),
)
def search_documents(query: str, ticker: Optional[str] = None) -> dict:
    tool_name = "search_documents"
    settings = get_settings()
    subject = {"query": (query or "").strip()}
    if ticker and ticker.strip():
        subject["ticker"] = ticker.strip().upper()

    if not settings.rag_enabled:
        return _fail(tool_name, "disabled",
                     "Document search is disabled on this server (RAG_ENABLED=false).",
                     subject=subject)
    if not subject["query"]:
        return _fail(tool_name, "invalid_input", "A non-empty search query is required.",
                     subject=subject)

    retriever = _get_retriever(settings)
    if retriever is None:
        return _fail(tool_name, "rag_unavailable",
                     "The local document index is not available (index or embedding model "
                     "missing). Answer using other tools or general knowledge.",
                     subject=subject)

    try:
        result = retriever.search(subject["query"], ticker=subject.get("ticker"))
    except Exception as exc:  # noqa: BLE001 - never crash the agent loop
        logger.error("search_documents failed (%s)", type(exc).__name__)
        return _fail(tool_name, "internal_error",
                     "Document search failed unexpectedly. Try again later or use other tools.",
                     subject=subject)

    if not result.chunks:
        return _fail(tool_name, "no_data",
                     "No relevant passages found in the ingested documents. The index may "
                     "not cover this company or topic; use list_ingested_documents to see "
                     "what is available.",
                     subject=subject)

    cap = settings.rag_snippet_max_chars
    results = [
        {
            "rank": r.rank,
            "score": round(r.score, 4),
            "matched_by": list(r.match_sources),
            "snippet": r.chunk.text[:cap] + ("..." if len(r.chunk.text) > cap else ""),
            "citation": r.citation.to_dict(),
        }
        for r in result.chunks
    ]
    return {
        "status": "ok",
        "tool": tool_name,
        **subject,
        "data": {
            "result_count": len(results),
            "results": results,
            "retrieval": {
                "reranked": result.stats.reranked,
                "latency_ms": round(result.stats.latency_ms, 1),
            },
        },
        "source": dict(_SOURCE),
        "as_of": utc_now_iso(),
        "cached": False,
        "note": "Ground the answer in these excerpts and cite them using the provided "
                "citation fields. Do not invent additional sources.",
    }


@tool(
    "list_ingested_documents",
    description=(
        "List the documents available in the local research index (title, ticker, document "
        "type, chunk count, ingestion time). Use this to check whether a company or topic "
        "is covered before searching, and tell the user plainly when something is not covered."
    ),
)
def list_ingested_documents(ticker: Optional[str] = None) -> dict:
    tool_name = "list_ingested_documents"
    settings = get_settings()
    subject: dict[str, str] = {}
    if ticker and ticker.strip():
        subject["ticker"] = ticker.strip().upper()

    if not settings.rag_enabled:
        return _fail(tool_name, "disabled",
                     "Document search is disabled on this server (RAG_ENABLED=false).",
                     subject=subject)

    try:
        from MarketInsight.rag.store import RagStore

        # dim=None: metadata-only access, no embedding model is loaded.
        store = RagStore(Path(settings.rag_data_dir) / "index.db", dim=None)
        try:
            docs = store.list_documents(ticker=subject.get("ticker"))
            total = store.document_count()
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("list_ingested_documents failed (%s)", type(exc).__name__)
        return _fail(tool_name, "rag_unavailable",
                     "The local document index is not available yet. Ingest documents first.",
                     subject=subject)

    if not docs:
        scope = f" for ticker {subject['ticker']}" if subject else ""
        return _fail(tool_name, "no_data",
                     f"No documents have been ingested{scope} yet.",
                     subject=subject)

    return {
        "status": "ok",
        "tool": tool_name,
        **subject,
        "data": {
            "document_count": total,
            "documents": [
                {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "ticker": d["ticker"],
                    "doc_type": d["doc_type"],
                    "chunk_count": d["chunk_count"],
                    "ingested_at": d["ingested_at"],
                    "source_path": d["source_path"],
                }
                for d in docs[:100]
            ],
        },
        "source": dict(_SOURCE),
        "as_of": utc_now_iso(),
        "cached": False,
    }
