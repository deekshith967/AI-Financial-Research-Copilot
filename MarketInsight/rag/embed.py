"""Local embeddings via fastembed (ONNX Runtime, CPU).

The model is downloaded once into the RAG data directory and then runs fully
offline. The wrapper is lazy and thread-safe: importing this module never
loads the model, and the agent tools only construct it on first use.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Union

from MarketInsight.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d, strong quality/size trade-off


class Embedder(Protocol):
    """What the pipeline needs from any embedding backend."""

    dim: int

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


def _l2_normalize(vector) -> List[float]:
    values = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in values))
    return values if norm == 0.0 else [v / norm for v in values]


class FastEmbedder:
    """Lazy wrapper around ``fastembed.TextEmbedding``.

    Documents are embedded with ``embed``; queries use ``query_embed`` so
    models with a query instruction prefix (such as BGE) are used correctly.
    Vectors are L2-normalized so cosine distance in sqlite-vec behaves well.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[Union[str, Path]] = None,
        threads: Optional[int] = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._threads = threads
        self._lock = threading.Lock()
        self._model = None
        self._dim: Optional[int] = None

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                kwargs = {"model_name": self._model_name}
                if self._cache_dir:
                    kwargs["cache_dir"] = self._cache_dir
                if self._threads:
                    kwargs["threads"] = self._threads
                logger.info(
                    "Loading embedding model %s (cache: %s)",
                    self._model_name,
                    self._cache_dir or "fastembed default",
                )
                self._model = TextEmbedding(**kwargs)
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        vectors = self._load().embed(texts)
        return [_l2_normalize(v) for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = next(iter(self._load().query_embed(text)))
        return _l2_normalize(vector)
