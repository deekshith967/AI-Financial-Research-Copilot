"""Optional Langfuse tracing.

Tracing is enabled only when both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are
set (and LANGFUSE_ENABLED is not false). Every function here is defensive: any
tracing problem is logged (without credentials) and never propagates to the app.
"""
from typing import Any, List, Optional

from config.settings import Settings, get_settings
from MarketInsight.utils.logger import get_logger

logger = get_logger(__name__)

_initialised = False
_active = False


def init_langfuse(settings: Optional[Settings] = None) -> bool:
    """Initialise the global Langfuse client once. Returns True if tracing is active."""
    global _initialised, _active
    if _initialised:
        return _active
    _initialised = True

    settings = settings or get_settings()
    if not settings.langfuse_configured:
        logger.info("Langfuse tracing disabled (keys not configured or LANGFUSE_ENABLED=false)")
        return False

    try:
        from langfuse import Langfuse

        kwargs = {
            "public_key": settings.langfuse_public_key,
            "secret_key": settings.langfuse_secret_key,
        }
        if settings.langfuse_host:
            kwargs["host"] = settings.langfuse_host
        Langfuse(**kwargs)  # registers the global client used by the callback handler
        _active = True
        logger.info("Langfuse tracing enabled")
    except Exception as e:
        # Only the exception type is logged to avoid echoing anything credential-related.
        logger.error(f"Langfuse initialisation failed; tracing disabled ({type(e).__name__})")
        _active = False
    return _active


def get_callbacks() -> List[Any]:
    """Return LangChain callbacks for one request (empty list when tracing is off)."""
    if not _active:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        # update_trace=True also records the graph's overall input/output on the trace.
        # LLM calls (real model name, usage when exposed by the provider), tool calls and
        # their latencies are captured from the LangChain callback events.
        return [CallbackHandler(update_trace=True)]
    except Exception as e:
        logger.error(f"Could not create Langfuse callback handler ({type(e).__name__})")
        return []


def trace_metadata(thread_id: str) -> dict:
    """Metadata that Langfuse's LangChain integration maps onto trace attributes."""
    return {"langfuse_session_id": thread_id, "langfuse_tags": ["market-insight"]}


def flush_langfuse() -> None:
    """Flush pending events (call on shutdown). Never raises."""
    if not _active:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        logger.error(f"Langfuse flush failed ({type(e).__name__})")
