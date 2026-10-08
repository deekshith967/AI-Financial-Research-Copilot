"""Agent-level evaluation harness (Phase 3).

Extends the retrieval-only evaluation in ``MarketInsight.rag.eval`` with
answer-level fact/citation/grounding checks, tool-selection scoring and
latency/token/cost telemetry — all driven by recorded fixtures so the suite
runs fully offline. Package is import-light: submodules are imported
directly where used.
"""
