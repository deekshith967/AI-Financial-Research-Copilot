"""Tests against the real fastembed model.

Skipped automatically when the model cannot be loaded (e.g. no network for a
first download). These pin the default model's dimension, normalisation,
determinism and basic semantic sanity; ranking mechanics are covered
deterministically by the HashEmbedder tests.
"""
import math

import pytest


def _cosine(a, b):
    # fastembed embeddings are L2-normalised, so the dot product is the cosine.
    return sum(x * y for x, y in zip(a, b))


class TestFastEmbedderReal:
    def test_dimension_matches_default_model(self, real_embedder):
        # bge-small-en-v1.5 is a 384-dimensional model.
        assert real_embedder.dim == 384
        assert len(real_embedder.embed_query("dimension check")) == 384

    def test_embeddings_are_l2_normalised(self, real_embedder):
        vec = real_embedder.embed_query("normalisation check")
        assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0, abs=1e-6)

    def test_query_embedding_is_deterministic(self, real_embedder):
        first = real_embedder.embed_query("helium supply disruption")
        second = real_embedder.embed_query("helium supply disruption")
        assert first == second

    def test_document_embedding_is_deterministic(self, real_embedder):
        docs = ["revenue grew year over year", "the dividend was maintained"]
        assert real_embedder.embed_documents(docs) == real_embedder.embed_documents(docs)

    def test_semantic_relatedness_ordering(self, real_embedder):
        anchor = real_embedder.embed_query("helium supply constraints disrupt production")
        related = real_embedder.embed_query("a shortage of helium interrupts manufacturing output")
        unrelated = real_embedder.embed_query("chocolate chip cookie baking recipe")
        assert _cosine(anchor, related) > _cosine(anchor, unrelated)

    def test_related_paraphrase_clears_retrieval_floor(self, real_embedder):
        # The default vector_min_score is 0.45: a genuine paraphrase must score
        # comfortably above it or retrieval would silently drop relevant chunks.
        anchor = real_embedder.embed_query("helium supply constraints disrupt production")
        related = real_embedder.embed_query("a shortage of helium interrupts manufacturing output")
        assert _cosine(anchor, related) > 0.45
