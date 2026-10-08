"""SQLite persistence for the RAG index.

One database file holds everything: the document registry, chunk text and
metadata, a sqlite-vec KNN index over the embeddings, and an FTS5 lexical
index kept in sync by triggers. Both search primitives return
``(chunk_id, score)`` lists ordered best-first; the retriever fuses them.

The store is thread-safe (single connection guarded by an RLock) and the
schema is versioned through a ``meta`` table, including the embedding
dimension so an index built with one model cannot be silently queried with
another.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from MarketInsight.rag.models import Chunk, Citation, DocumentMeta
from MarketInsight.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "1"


class StoreError(Exception):
    """Store-level failure (missing extensions, dimension mismatch, ...)."""


class DuplicateDocument(Exception):
    """Raised when ingesting content whose sha256 is already indexed."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(f"Document already indexed (doc_id={doc_id}).")
        self.doc_id = doc_id


def fts_query_from_text(text: str, *, max_terms: int = 12) -> str:
    """Build a safe FTS5 MATCH query from arbitrary user text.

    Each alphanumeric term is double-quoted (neutralising FTS5 operators) and
    terms are OR-ed, which favours recall; BM25 still ranks documents matching
    more/rarer terms higher. Returns "" when nothing usable remains.
    """
    terms = re.findall(r"[A-Za-z0-9]+", text.lower())
    seen: set[str] = set()
    unique: List[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return " OR ".join(f'"{term}"' for term in unique[:max_terms])


class RagStore:
    def __init__(self, db_path: Union[str, Path], *, dim: Optional[int] = None) -> None:
        """Open (or create) an index.

        ``dim`` is required when creating a new index and, when supplied for an
        existing index, is validated against the stored dimension. Opening an
        existing index with ``dim=None`` is allowed for metadata-only access
        (for example listing documents without loading an embedding model).
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            try:
                import sqlite_vec

                self._db.enable_load_extension(True)
                sqlite_vec.load(self._db)
                self._db.enable_load_extension(False)
            except Exception as exc:  # noqa: BLE001 - surface any extension failure uniformly
                raise StoreError(f"sqlite-vec could not be loaded: {exc}") from exc
            if not self._fts5_available():
                raise StoreError("SQLite FTS5 is not available in this Python build.")
            self._dim = self._init_schema(dim)
        except Exception:
            self._db.close()
            raise

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _fts5_available(self) -> bool:
        try:
            self._db.execute("CREATE VIRTUAL TABLE temp._fts5_probe USING fts5(x)")
            self._db.execute("DROP TABLE temp._fts5_probe")
            return True
        except sqlite3.Error:
            return False

    def _get_meta(self, key: str) -> Optional[str]:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _init_schema(self, dim: int) -> int:
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    ticker TEXT,
                    doc_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    page_count INTEGER,
                    ingested_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    section TEXT,
                    page INTEGER,
                    UNIQUE(doc_id, chunk_index)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text, content='chunks', content_rowid='rowid'
                );
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES ('delete', old.rowid, old.text);
                END;
                """
            )
            vec_exists = self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'vec_chunks'"
            ).fetchone()
            if vec_exists is None:
                if dim is None:
                    raise StoreError(
                        "dim is required to create a new index; open an existing index "
                        "or pass the embedding model's dimension."
                    )
                self._db.execute(
                    "CREATE VIRTUAL TABLE vec_chunks USING vec0("
                    "chunk_id TEXT PRIMARY KEY, "
                    f"embedding float[{dim}] distance_metric=cosine)"
                )
                self._set_meta("embedding_dim", str(dim))
                self._set_meta("schema_version", SCHEMA_VERSION)
                return dim

            stored_dim = int(self._get_meta("embedding_dim"))
            if dim is not None and stored_dim != dim:
                raise StoreError(
                    f"Index was built with embedding dim {stored_dim} but the current "
                    f"model produces {dim}. Use the matching model or rebuild the index."
                )
            return stored_dim

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        meta: DocumentMeta,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> str:
        """Insert one document, its chunks and vectors in a single transaction."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")
        if not chunks:
            raise ValueError("Cannot index a document with no chunks.")
        for emb in embeddings:
            if len(emb) != self._dim:
                raise StoreError(
                    f"Embedding dim {len(emb)} does not match index dim {self._dim}."
                )

        from sqlite_vec import serialize_float32

        with self._lock:
            existing = self._db.execute(
                "SELECT doc_id FROM documents WHERE sha256 = ?", (meta.sha256,)
            ).fetchone()
            if existing:
                raise DuplicateDocument(existing["doc_id"])
            with self._db:
                self._db.execute(
                    "INSERT INTO documents (doc_id, source_path, title, ticker, doc_type,"
                    " sha256, page_count, ingested_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        meta.doc_id, meta.source_path, meta.title, meta.ticker,
                        meta.doc_type, meta.sha256, meta.page_count, meta.ingested_at,
                    ),
                )
                self._db.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, chunk_index, text,"
                    " token_count, section, page) VALUES (?,?,?,?,?,?,?)",
                    [
                        (c.chunk_id, c.doc_id, c.chunk_index, c.text, c.token_count,
                         c.section, c.page)
                        for c in chunks
                    ],
                )
                self._db.executemany(
                    "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                    [
                        (c.chunk_id, serialize_float32([float(v) for v in emb]))
                        for c, emb in zip(chunks, embeddings)
                    ],
                )
        logger.info("Indexed document %s (%d chunks)", meta.doc_id, len(chunks))
        return meta.doc_id

    def has_content(self, sha256: str) -> Optional[str]:
        with self._lock:
            row = self._db.execute(
                "SELECT doc_id FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return row["doc_id"] if row else None

    # ------------------------------------------------------------------
    # Search primitives
    # ------------------------------------------------------------------

    def vector_search(self, query_embedding: Sequence[float], k: int) -> List[Tuple[str, float]]:
        """KNN search. Returns (chunk_id, similarity) best-first, similarity = 1 - cosine distance."""
        if k <= 0:
            return []
        from sqlite_vec import serialize_float32

        with self._lock:
            rows = self._db.execute(
                "SELECT chunk_id, distance FROM vec_chunks"
                " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (serialize_float32([float(v) for v in query_embedding]), k),
            ).fetchall()
        return [(row["chunk_id"], 1.0 - float(row["distance"])) for row in rows]

    def fts_search(self, fts_query: str, k: int) -> List[Tuple[str, float]]:
        """BM25 lexical search. Returns (chunk_id, score) best-first, higher is better."""
        if not fts_query or k <= 0:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS score"
                " FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid"
                " WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (fts_query, k),
            ).fetchall()
        return [(row["chunk_id"], -float(row["score"])) for row in rows]

    # ------------------------------------------------------------------
    # Chunk / document access
    # ------------------------------------------------------------------

    def get_chunks_with_citations(
        self, chunk_ids: Sequence[str]
    ) -> Dict[str, Tuple[Chunk, Citation]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._lock:
            rows = self._db.execute(
                "SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.token_count,"
                "       c.section, c.page, d.title, d.source_path, d.doc_type, d.ticker"
                " FROM chunks c JOIN documents d ON d.doc_id = c.doc_id"
                f" WHERE c.chunk_id IN ({placeholders})",
                list(chunk_ids),
            ).fetchall()
        out: Dict[str, Tuple[Chunk, Citation]] = {}
        for row in rows:
            chunk = Chunk(
                chunk_id=row["chunk_id"], doc_id=row["doc_id"],
                chunk_index=row["chunk_index"], text=row["text"],
                token_count=row["token_count"], section=row["section"], page=row["page"],
            )
            citation = Citation(
                doc_id=row["doc_id"], chunk_id=row["chunk_id"], title=row["title"],
                source_path=row["source_path"], doc_type=row["doc_type"],
                chunk_index=row["chunk_index"], ticker=row["ticker"],
                section=row["section"], page=row["page"],
            )
            out[row["chunk_id"]] = (chunk, citation)
        return out

    def list_documents(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        query = (
            "SELECT d.doc_id, d.source_path, d.title, d.ticker, d.doc_type,"
            "       d.sha256, d.page_count, d.ingested_at, COUNT(c.chunk_id) AS chunk_count"
            " FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id"
        )
        params: List[Any] = []
        if ticker:
            query += " WHERE UPPER(d.ticker) = UPPER(?)"
            params.append(ticker)
        query += " GROUP BY d.doc_id ORDER BY d.ingested_at DESC, d.doc_id"
        with self._lock:
            rows = self._db.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def document_count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"])

    def chunk_count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                return False
            chunk_ids = [
                r["chunk_id"]
                for r in self._db.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
                ).fetchall()
            ]
            with self._db:
                for chunk_id in chunk_ids:
                    self._db.execute(
                        "DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
                    )
                # Deleting chunks fires the FTS delete trigger; FK cascade removes chunks.
                self._db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        logger.info("Deleted document %s (%d chunks)", doc_id, len(chunk_ids))
        return True

    def index_stats(self) -> Dict[str, Any]:
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "documents": self.document_count(),
            "chunks": self.chunk_count(),
            "embedding_dim": self._dim,
            "db_bytes": size,
        }

    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "RagStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
