"""Central runtime settings.

Every tunable value is read from environment variables (optionally loaded from a
local ``.env`` file) and exposed through a single immutable :class:`Settings`
object. No secrets have defaults. See ``.env.example`` for the full list.
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Defaults preserve the behaviour of the original application.
DEFAULT_LLM_MODEL = "c1/openai/gpt-5/v-20250930"
DEFAULT_LLM_BASE_URL = "https://api.thesys.dev/v1/embed/"
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"


def _env(name: str) -> Optional[str]:
    """Return a stripped env var, treating empty strings as unset."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}")


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}")


def _env_list(name: str, default: str) -> Tuple[str, ...]:
    raw = _env(name) or default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # --- LLM -------------------------------------------------------------
    llm_model: str
    llm_base_url: str
    # Thesys key for the default base URL. Never logged or serialised.
    llm_api_key: Optional[str]
    llm_timeout_seconds: int
    llm_max_retries: int
    # Ask the provider to include token usage in streamed responses
    # (stream_options.include_usage). Off by default because langchain-openai only
    # enables it automatically for the default OpenAI URL and it is unverified for Thesys.
    llm_stream_usage: bool

    # --- Agent limits ----------------------------------------------------
    agent_recursion_limit: int   # max LangGraph steps per request
    agent_max_tool_calls: int    # max tool calls per request

    # --- HTTP API --------------------------------------------------------
    api_host: str
    api_port: int
    cors_origins: Tuple[str, ...]
    max_prompt_chars: int
    max_thread_id_chars: int
    rate_limit_per_minute: int   # per client; 0 disables rate limiting
    trust_forwarded_for: bool    # honour X-Forwarded-For (only behind a trusted proxy)

    # --- Langfuse (optional) --------------------------------------------
    langfuse_enabled: bool
    langfuse_public_key: Optional[str]
    langfuse_secret_key: Optional[str]
    langfuse_host: Optional[str]

    @property
    def langfuse_configured(self) -> bool:
        """Tracing is active only when enabled AND both keys are present."""
        return bool(
            self.langfuse_enabled
            and self.langfuse_public_key
            and self.langfuse_secret_key
        )

    # --- RAG (local document retrieval; added 2026-10-08) ----------------
    # All local: embeddings, vector store and lexical index live on disk under
    # rag_data_dir. No external service or API key is required.
    rag_enabled: bool
    rag_data_dir: str            # index DB, model cache and eval output root
    rag_embedding_model: str     # fastembed model id
    rag_chunk_tokens: int        # target chunk size in tokens
    rag_chunk_overlap_tokens: int
    rag_min_chunk_tokens: int    # smaller trailing chunks are merged/dropped
    rag_top_k: int               # default chunks returned per search
    rag_max_top_k: int           # hard cap regardless of caller input
    rag_candidate_k: int         # per-signal candidates fed into fusion
    rag_rrf_k: int               # reciprocal-rank-fusion constant
    rag_max_context_chars: int   # total text budget across returned chunks
    rag_snippet_max_chars: int   # per-chunk cap in tool output
    rag_vector_min_score: float  # cosine floor for the vector signal (FTS unaffected)
    rag_rerank_enabled: bool     # cross-encoder rerank of fused candidates
    rag_rerank_model: str
    rag_rerank_candidates: int


def load_settings() -> Settings:
    return Settings(
        llm_model=_env("LLM_MODEL") or DEFAULT_LLM_MODEL,
        llm_base_url=_env("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL,
        # LLM_API_KEY wins; OPENAI_API_KEY is kept for backward compatibility
        # (with the default base URL it must hold a Thesys key).
        llm_api_key=_env("LLM_API_KEY") or _env("OPENAI_API_KEY"),
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 120),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        llm_stream_usage=_env_bool("LLM_STREAM_USAGE", False),
        agent_recursion_limit=_env_int("AGENT_RECURSION_LIMIT", 25),
        agent_max_tool_calls=_env_int("AGENT_MAX_TOOL_CALLS", 12),
        api_host=_env("API_HOST") or "0.0.0.0",
        api_port=_env_int("PORT", _env_int("API_PORT", 8000)),
        cors_origins=_env_list("CORS_ORIGINS", DEFAULT_CORS_ORIGINS),
        max_prompt_chars=_env_int("MAX_PROMPT_CHARS", 4000),
        max_thread_id_chars=_env_int("MAX_THREAD_ID_CHARS", 128),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 20),
        trust_forwarded_for=_env_bool("TRUST_FORWARDED_FOR", False),
        langfuse_enabled=_env_bool("LANGFUSE_ENABLED", True),
        langfuse_public_key=_env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_env("LANGFUSE_SECRET_KEY"),
        langfuse_host=_env("LANGFUSE_HOST"),
        rag_enabled=_env_bool("RAG_ENABLED", True),
        rag_data_dir=_env("RAG_DATA_DIR") or "data/rag",
        rag_embedding_model=_env("RAG_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5",
        rag_chunk_tokens=_env_int("RAG_CHUNK_TOKENS", 512),
        rag_chunk_overlap_tokens=_env_int("RAG_CHUNK_OVERLAP_TOKENS", 77),
        rag_min_chunk_tokens=_env_int("RAG_MIN_CHUNK_TOKENS", 40),
        rag_top_k=_env_int("RAG_TOP_K", 8),
        rag_max_top_k=_env_int("RAG_MAX_TOP_K", 16),
        rag_candidate_k=_env_int("RAG_CANDIDATE_K", 24),
        rag_rrf_k=_env_int("RAG_RRF_K", 60),
        rag_max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 6000),
        rag_snippet_max_chars=_env_int("RAG_SNIPPET_MAX_CHARS", 700),
        rag_vector_min_score=_env_float("RAG_VECTOR_MIN_SCORE", 0.45),
        rag_rerank_enabled=_env_bool("RAG_RERANK_ENABLED", False),
        rag_rerank_model=_env("RAG_RERANK_MODEL") or "Xenova/ms-marco-MiniLM-L-6-v2",
        rag_rerank_candidates=_env_int("RAG_RERANK_CANDIDATES", 20),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
