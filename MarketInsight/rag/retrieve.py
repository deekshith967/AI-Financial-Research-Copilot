"""Hybrid retrieval: vector KNN + FTS5 BM25, fused with reciprocal rank fusion.

Both signals return ranked ``(chunk_id, score)`` lists; RRF merges them by
rank only, so no score-scale calibration between cosine similarity and BM25 is
needed. Results are deduplicated (by chunk id and by normalized text), can be
scoped to a ticker, optionally re-scored by a cross-encoder reranker, and are
trimmed to a top-k plus a total character budget so downstream LLM context
stays bounded. Retrieval is fully deterministic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from MarketInsight.rag.embed import Embedder
from MarketInsight.rag.models import RetrievedChunk
from MarketInsight.rag.rerank import Reranker
from MarketInsight.rag.store import RagStore, fts_query_from_text
from MarketInsight.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalStats:
    vector_hits: int
    fts_hits: int
    fused_candidates: int
    duplicates_removed: int
    returned: int
    latency_ms: float
    reranked: bool

    def to_dict(self) -> dict:
        return vars(self)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: Tuple[RetrievedChunk, ...]
    stats: RetrievalStats


def _normalize_key(text: str) -> str:
    return " ".join(text.lower().split())


class Retriever:
    def __init__(
        self,
        store: RagStore,
        embedder: Embedder,
        *,
        top_k: int = 8,
        max_top_k: int = 16,
        candidate_k: int = 24,
        rrf_k: int = 60,
        max_context_chars: int = 6000,
        vector_min_score: float = 0.45,
        reranker: Optional[Reranker] = None,
        rerank_candidates: int = 20,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.max_top_k = max_top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.max_context_chars = max_context_chars
        # Cosine-similarity floor for the vector signal. KNN always returns the
        # k nearest chunks, even for unrelated queries (bi-encoder baselines are
        # far above 0), so without a floor every garbage query would look
        # "successful". Exact-term FTS matches are never filtered.
        self.vector_min_score = vector_min_score
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    def search(
        self,
        query: str,
        *,
        ticker: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        query = (query or "").strip()
        if not query:
            return self._empty(started)

        k = max(1, min(top_k or self.top_k, self.max_top_k))

        # Signal 1: vector KNN over the embedded query, below-floor hits dropped.
        vec_hits = [
            (chunk_id, score)
            for chunk_id, score in self.store.vector_search(
                self.embedder.embed_query(query), self.candidate_k
            )
            if score >= self.vector_min_score
        ]

        # Signal 2: BM25 over a sanitised FTS5 query.
        fts_hits = self.store.fts_search(fts_query_from_text(query), self.candidate_k)

        # Reciprocal rank fusion.
        rrf: Dict[str, float] = {}
        sources: Dict[str, set] = {}
        for rank, (chunk_id, _score) in enumerate(vec_hits, start=1):
            rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            sources.setdefault(chunk_id, set()).add("vector")
        for rank, (chunk_id, _score) in enumerate(fts_hits, start=1):
            rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            sources.setdefault(chunk_id, set()).add("fts")

        # Deterministic order: score desc, chunk_id asc for ties.
        ordered = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))

        fetched = self.store.get_chunks_with_citations([cid for cid, _ in ordered])

        if ticker:
            wanted = ticker.strip().upper()
            ordered = [
                (cid, s) for cid, s in ordered
                if cid in fetched and (fetched[cid][1].ticker or "").upper() == wanted
            ]

        # Dedup by normalized text: identical text indexed twice (same content
        # in different files) would otherwise waste context budget.
        seen_text: set = set()
        candidates: List[Tuple[str, float]] = []
        duplicates = 0
        for chunk_id, score in ordered:
            entry = fetched.get(chunk_id)
            if entry is None:
                continue
            key = _normalize_key(entry[0].text)
            if key in seen_text:
                duplicates += 1
                continue
            seen_text.add(key)
            candidates.append((chunk_id, score))

        reranked = False
        if self.reranker is not None and len(candidates) > 1:
            pool = candidates[: self.rerank_candidates]
            try:
                scores = self.reranker.rerank(query, [fetched[cid][0].text for cid, _ in pool])
                order = sorted(
                    range(len(pool)),
                    key=lambda i: (-scores[i], i),
                )
                pool = [(pool[i][0], float(scores[i])) for i in order]
                candidates = pool + candidates[self.rerank_candidates:]
                reranked = True
            except Exception as exc:  # noqa: BLE001 - rerank is an enhancement, never a hard failure
                logger.warning("Reranker failed (%s); keeping fused ranking", type(exc).__name__)

        results: List[RetrievedChunk] = []
        total_chars = 0
        for chunk_id, score in candidates:
            if len(results) >= k:
                break
            chunk, citation = fetched[chunk_id]
            if results and total_chars + len(chunk.text) > self.max_context_chars:
                break
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    citation=citation,
                    score=score,
                    match_sources=tuple(sorted(sources.get(chunk_id, ()))),
                    rank=len(results) + 1,
                )
            )
            total_chars += len(chunk.text)

        stats = RetrievalStats(
            vector_hits=len(vec_hits),
            fts_hits=len(fts_hits),
            fused_candidates=len(rrf),
            duplicates_removed=duplicates,
            returned=len(results),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            reranked=reranked,
        )
        return RetrievalResult(chunks=tuple(results), stats=stats)

    def _empty(self, started: float) -> RetrievalResult:
        return RetrievalResult(
            chunks=(),
            stats=RetrievalStats(
                vector_hits=0, fts_hits=0, fused_candidates=0, duplicates_removed=0,
                returned=0, latency_ms=(time.perf_counter() - started) * 1000.0,
                reranked=False,
            ),
        )


# --------------------------------------------------------------------------------
# CLI: python -m MarketInsight.rag.retrieve "query" [--ticker ZPHY] [--top-k 5]
# --------------------------------------------------------------------------------

def _main() -> int:
    import argparse
    import json

    from config.settings import get_settings
    from MarketInsight.rag.ingest import build_default_retriever

    parser = argparse.ArgumentParser(description="Search the local document index.")
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--ticker", help="Restrict results to one ticker")
    parser.add_argument("--top-k", type=int, help="Number of chunks to return")
    parser.add_argument("--data-dir", help="RAG data directory (default: settings)")
    args = parser.parse_args()

    settings = get_settings()
    retriever = build_default_retriever(settings, data_dir=args.data_dir)
    result = retriever.search(args.query, ticker=args.ticker, top_k=args.top_k)
    payload = [
        {
            "rank": r.rank,
            "score": round(r.score, 4),
            "match": list(r.match_sources),
            "section": r.chunk.section,
            "citation": r.citation.label(),
            "snippet": r.chunk.text[:300],
        }
        for r in result.chunks
    ]
    print(json.dumps({"results": payload, "stats": result.stats.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
