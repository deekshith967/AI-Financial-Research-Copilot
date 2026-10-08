"""Retrieval evaluation harness — runs a golden question set against the
local index and reports standard IR metrics. No LLM required.

Golden dataset format (JSONL, one object per line)::

    {"question": "...", "source_file": "zephyr-systems-10k.md",
     "must_contain": ["helium"]}

A retrieved chunk counts as *relevant* for a question when its citation's
source path ends with the question's ``source_file``. ``must_contain`` terms
measure answer-term coverage: whether the retrieved snippets actually carry
the facts an answer would need.

CLI::

    python -m MarketInsight.rag.eval [--golden tests/eval/golden.jsonl]
        [--samples samples/docs] [--data-dir data/rag] [--top-k 5]

The sample directory is ingested first (already-indexed content is skipped),
so the command is self-contained and repeatable.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from MarketInsight.rag.retrieve import Retriever
from MarketInsight.utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_CITATION_FIELDS = ("doc_id", "chunk_id", "title", "source_path")


@dataclass(frozen=True)
class GoldenQuestion:
    question: str
    source_file: str
    must_contain: Tuple[str, ...] = ()


def load_golden(path: Union[str, Path]) -> List[GoldenQuestion]:
    path = Path(path)
    questions: List[GoldenQuestion] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if not raw.get("question") or not raw.get("source_file"):
            raise ValueError(f"{path}:{lineno}: every entry needs 'question' and 'source_file'")
        questions.append(
            GoldenQuestion(
                question=raw["question"],
                source_file=raw["source_file"],
                must_contain=tuple(raw.get("must_contain", ())),
            )
        )
    if not questions:
        raise ValueError(f"{path}: no questions found")
    return questions


def _is_relevant(source_path: str, source_file: str) -> bool:
    return source_path.replace("\\", "/").endswith(source_file)


def evaluate(
    retriever: Retriever,
    questions: Sequence[GoldenQuestion],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run every golden question and return per-question rows + aggregates."""
    rows: List[Dict[str, Any]] = []
    for q in questions:
        result = retriever.search(q.question, top_k=top_k)
        chunks = result.chunks

        relevant_ranks = [
            r.rank for r in chunks if _is_relevant(r.citation.source_path, q.source_file)
        ]
        first_relevant = min(relevant_ranks) if relevant_ranks else None

        # Whitespace-insensitive: answer terms may span line breaks in the
        # source document ("6.5\npercent"); an answer could still be written.
        snippet_text = " ".join("\n".join(r.chunk.text for r in chunks).split()).lower()
        missing_terms = [t for t in q.must_contain if t.lower() not in snippet_text]

        complete_citations = sum(
            1 for r in chunks
            if all(getattr(r.citation, field) for field in _REQUIRED_CITATION_FIELDS)
        )

        rows.append(
            {
                "question": q.question,
                "source_file": q.source_file,
                "hit": bool(relevant_ranks),
                "first_relevant_rank": first_relevant,
                "reciprocal_rank": (1.0 / first_relevant) if first_relevant else 0.0,
                "relevant_in_topk": len(relevant_ranks),
                "retrieved": len(chunks),
                "precision": (len(relevant_ranks) / len(chunks)) if chunks else 0.0,
                "missing_terms": missing_terms,
                "citation_coverage": (complete_citations / len(chunks)) if chunks else 1.0,
                "duplicates_removed": result.stats.duplicates_removed,
                "fused_candidates": result.stats.fused_candidates,
                "latency_ms": round(result.stats.latency_ms, 2),
            }
        )
    return {"aggregate": _aggregate(rows, top_k), "per_question": rows}


def _aggregate(rows: Sequence[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    n = len(rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    total_candidates = sum(r["fused_candidates"] for r in rows)
    total_duplicates = sum(r["duplicates_removed"] for r in rows)
    return {
        "questions": n,
        f"recall@{top_k}": _mean(r["hit"] for r in rows),
        f"precision@{top_k}": _mean(r["precision"] for r in rows),
        "mrr": _mean(r["reciprocal_rank"] for r in rows),
        "answer_term_coverage": _mean(not r["missing_terms"] for r in rows),
        "citation_coverage": _mean(r["citation_coverage"] for r in rows),
        "duplicate_rate": (total_duplicates / total_candidates) if total_candidates else 0.0,
        "avg_retrieved_chunks": _mean(r["retrieved"] for r in rows),
        "latency_ms": {
            "mean": _mean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; ``sorted_values`` must be non-empty and sorted."""
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def _main() -> int:
    import argparse

    from config.settings import get_settings
    from MarketInsight.rag.embed import FastEmbedder
    from MarketInsight.rag.ingest import build_default_retriever, ingest_path
    from MarketInsight.rag.store import RagStore

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on a golden dataset.")
    parser.add_argument("--golden", default="tests/eval/golden.jsonl", help="Golden JSONL dataset")
    parser.add_argument("--samples", default="samples/docs", help="Directory of documents to index")
    parser.add_argument("--data-dir", help="RAG data directory (default: settings)")
    parser.add_argument("--top-k", type=int, default=5, help="Cut-off for recall/precision (default: 5)")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path(args.data_dir or settings.rag_data_dir)

    embedder = FastEmbedder(settings.rag_embedding_model, cache_dir=data_dir / "models")
    store = RagStore(data_dir / "index.db", dim=embedder.dim)
    results = ingest_path(args.samples, store=store, embedder=embedder)
    for r in results:
        logger.info("ingest %s: %s (%s chunks)", r.path, r.status, r.chunks)
    store.close()

    retriever = build_default_retriever(settings, data_dir, embedder=embedder)
    questions = load_golden(args.golden)
    report = evaluate(retriever, questions, top_k=args.top_k)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
