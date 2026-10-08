from functools import lru_cache
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from config.settings import Settings, get_settings
from MarketInsight.prompts import SYSTEM_PROMPT
from MarketInsight.rag.tools import list_ingested_documents, search_documents
from MarketInsight.utils.logger import get_logger
from MarketInsight.utils.tools import (
    get_analyst_recommendations,
    get_balance_sheet,
    get_cash_flow,
    get_company_info,
    get_dividends,
    get_historical_data,
    get_income_statement,
    get_insider_transactions,
    get_institutional_holders,
    get_major_shareholders,
    get_mutual_fund_holders,
    get_splits,
    get_stock_news,
    get_stock_price,
    get_ticker,
)

logger = get_logger(__name__)

# NOTE: `get_analyst_recommendations_summary` was removed from this list: in
# yfinance 0.2.66 `recommendations_summary` simply returns `recommendations`,
# so it duplicated `get_analyst_recommendations`.
#
# `search_documents` / `list_ingested_documents` (added 2026-10-08) expose the
# local RAG index (ingested filings/research notes). They are independent of
# the market-data tools and return an error envelope when the index is
# unavailable, so the agent keeps working either way.
TOOLS = [
    get_stock_price, get_historical_data, get_stock_news, get_balance_sheet,
    get_income_statement, get_cash_flow, get_company_info, get_dividends,
    get_splits, get_institutional_holders, get_major_shareholders,
    get_mutual_fund_holders, get_insider_transactions,
    get_analyst_recommendations, get_ticker,
    search_documents, list_ingested_documents,
]


def build_model(settings: Settings) -> ChatOpenAI:
    kwargs = {}
    if settings.llm_api_key:
        kwargs["api_key"] = settings.llm_api_key
    if settings.llm_stream_usage:
        kwargs["stream_usage"] = True
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        **kwargs,
    )


def build_agent(settings: Optional[Settings] = None, model=None, checkpointer=None):
    """Build the LangGraph agent.

    ``model`` and ``checkpointer`` can be injected (used by tests); by default the
    configured ChatOpenAI client (Thesys C1) and an in-memory checkpointer are used.
    """
    settings = settings or get_settings()
    return create_agent(
        model or build_model(settings),
        tools=TOOLS,
        # Supplied once here, not appended to the conversation on every request.
        system_prompt=SYSTEM_PROMPT,
        middleware=[ToolCallLimitMiddleware(run_limit=settings.agent_max_tool_calls)],
        checkpointer=checkpointer or MemorySaver(),
    )


@lru_cache(maxsize=1)
def get_agent():
    """Lazily build the shared agent so a missing API key surfaces per request
    (as a user-visible error) instead of crashing the process at import time."""
    agent = build_agent()
    logger.info("Agent Initiated Successfully")
    return agent