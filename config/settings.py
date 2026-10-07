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
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
