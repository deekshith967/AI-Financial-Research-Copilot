"""Local retrieval-augmented generation (RAG) subsystem.

Added 2026-10-08 (Phase 2). Everything here runs locally: document ingestion,
extraction, chunking, embeddings (fastembed/ONNX), and a SQLite index
(sqlite-vec for vectors, FTS5 for lexical search). No external service or API
key is required, and nothing in this package is imported at application
startup — the agent tools construct the pipeline lazily on first use.
"""
