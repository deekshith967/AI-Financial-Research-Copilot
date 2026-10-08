"""Shared pytest fixtures for the RAG test suite.

Unit tests use a deterministic HashEmbedder (no model download, no network).
Tests that exercise the real fastembed model use the ``real_embedder``
session fixture and skip automatically when the model cannot be loaded.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLES_DIR = ROOT / "samples" / "docs"


def word_count(text: str) -> int:
    """Token counter used by chunker unit tests (one whitespace word = one token)."""
    return len(text.split())


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedder for pipeline unit tests.

    Texts sharing tokens get positive cosine similarity, which is enough to
    exercise ranking mechanics. This is a test double only; the shipped
    pipeline always uses fastembed.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            digest = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            vec[digest % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES_DIR


@pytest.fixture
def make_store(tmp_path):
    """Factory for RagStore instances backed by a temp directory."""
    from MarketInsight.rag.store import RagStore

    created = []

    def _make(dim: int = 64, name: str = "index.db") -> RagStore:
        store = RagStore(tmp_path / name, dim=dim)
        created.append(store)
        return store

    yield _make
    for store in created:
        store.close()


@pytest.fixture(scope="session")
def real_embedder():
    """The real fastembed model. Skips when the model cannot be loaded
    (for example on a machine with no network for the first download).

    Reuses the project's model cache (data/rag/models) when present so the
    suite stays offline after the first ever download.
    """
    try:
        from MarketInsight.rag.embed import FastEmbedder

        cache_dir = ROOT / "data" / "rag" / "models"
        kwargs = {"cache_dir": cache_dir} if cache_dir.is_dir() else {}
        embedder = FastEmbedder(**kwargs)
        _ = embedder.dim  # force the model to actually load
    except Exception as exc:  # noqa: BLE001 - any load failure means "unavailable"
        pytest.skip(f"fastembed model unavailable: {type(exc).__name__}: {exc}")
    return embedder
