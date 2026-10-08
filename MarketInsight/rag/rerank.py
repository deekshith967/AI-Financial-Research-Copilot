"""Optional cross-encoder reranking (fastembed, ONNX, CPU).

Disabled by default; when enabled, the fused candidate list is re-scored by a
cross-encoder that sees (query, document) pairs, which is more accurate than
bi-encoder similarity but slower. The model is downloaded once on first use.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Union

from MarketInsight.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        """Return one relevance score per document, in input order."""
        ...


class FastReranker:
    """Lazy wrapper around ``fastembed.rerank.cross_encoder.TextCrossEncoder``."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        cache_dir: Optional[Union[str, Path]] = None,
        threads: Optional[int] = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._threads = threads
        self._lock = threading.Lock()
        self._model = None

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                kwargs = {"model_name": self._model_name}
                if self._cache_dir:
                    kwargs["cache_dir"] = self._cache_dir
                if self._threads:
                    kwargs["threads"] = self._threads
                logger.info("Loading reranker model %s", self._model_name)
                self._model = TextCrossEncoder(**kwargs)
        return self._model

    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        documents = list(documents)
        if not documents:
            return []
        scores = self._load().rerank(query, documents)
        return [float(s) for s in scores]
