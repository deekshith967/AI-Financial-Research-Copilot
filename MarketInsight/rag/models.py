"""Core data structures for the local RAG pipeline.

All structures are immutable. Citations are always derived from stored
document/chunk rows, never constructed by the language model, so every
citation the agent emits can be traced back to an ingested file on disk.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Stable chunk id: identical content at the same position reuses the id."""
    digest = hashlib.sha256(f"{doc_id}:{chunk_index}\n{text}".encode("utf-8")).hexdigest()
    return digest[:32]


@dataclass(frozen=True)
class DocumentMeta:
    """One ingested document. ``sha256`` of the normalized text is the dedup key."""

    source_path: str
    title: str
    ticker: Optional[str]
    doc_type: str
    sha256: str
    page_count: Optional[int] = None
    ingested_at: str = ""

    @property
    def doc_id(self) -> str:
        return self.sha256[:16]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    token_count: int
    section: Optional[str] = None
    page: Optional[int] = None


@dataclass(frozen=True)
class Citation:
    """Source attribution attached to every retrieved chunk."""

    doc_id: str
    chunk_id: str
    title: str
    source_path: str
    doc_type: str
    chunk_index: int
    ticker: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None

    def label(self) -> str:
        parts = [self.title]
        if self.doc_type:
            parts.append(self.doc_type)
        if self.ticker:
            parts.append(self.ticker)
        if self.section and self.section != self.title:
            parts.append(self.section)
        if self.page is not None:
            parts.append(f"page {self.page}")
        parts.append(f"chunk {self.chunk_index}")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "source_path": self.source_path,
            "doc_type": self.doc_type,
            "ticker": self.ticker,
            "section": self.section,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "label": self.label(),
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    citation: Citation
    score: float
    match_sources: Tuple[str, ...]  # subset of ("vector", "fts")
    rank: int
