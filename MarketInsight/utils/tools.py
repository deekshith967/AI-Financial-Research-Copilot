"""LangChain tools that retrieve market data from Yahoo Finance.

All 15 tools share one reliability layer:

* strict input validation (tickers, company names, dates);
* bounded execution time for every upstream call;
* bounded retries (exponential backoff + jitter) for *transient* network
  failures only; rate-limit responses are never retried and trip a short
  cool-down so we stop hammering Yahoo;
* a small in-process TTL cache for read-only datasets;
* JSON-safe, consistently shaped responses that carry provider metadata;
* error messages that never expose raw exceptions to the LLM/user, and logs
  that redact URL query strings / credentials.

Response shape (every tool)::

    {
        "status": "ok" | "no_data" | "error",
        "tool": "<tool name>",
        "ticker": "AAPL",               # or "query" for get_ticker
        "data": ...,                    # only when status == "ok"
        "error_type": "...",            # only when status != "ok"
        "message": "...",               # only when status != "ok"
        "source": {"provider": "Yahoo Finance", "via": "yfinance"},
        "as_of": "2026-01-01T00:00:00Z",  # when the data was fetched
        "cached": false,
    }
"""

from __future__ import annotations

import copy
import math
import random
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional, TypeVar

import pandas as pd
import requests
import yfinance as yf
from langchain.tools import tool

from MarketInsight.utils.logger import get_logger

logger = get_logger("Tools")

# yfinance / curl_cffi exception types (both are already yfinance dependencies).
# Imported defensively so a yfinance upgrade that moves them cannot break import.
try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:  # pragma: no cover
    class YFRateLimitError(Exception):  # type: ignore[no-redef]
        """Fallback placeholder when yfinance does not expose this exception."""

try:
    from yfinance.exceptions import YFPricesMissingError, YFTickerMissingError
    _YF_NO_DATA_ERRORS: tuple[type[BaseException], ...] = (YFTickerMissingError, YFPricesMissingError)
except ImportError:  # pragma: no cover
    _YF_NO_DATA_ERRORS = ()

try:
    from curl_cffi.requests import exceptions as _curl_exceptions
    _CURL_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
        _curl_exceptions.ConnectionError,
        _curl_exceptions.Timeout,
    )
except Exception:  # pragma: no cover
    _CURL_TRANSIENT_ERRORS = ()


# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
# Yahoo's search endpoint tends to reject the default python-requests User-Agent.
_YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
_HTTP_CONNECT_TIMEOUT_SECONDS = 5.0
_HTTP_READ_TIMEOUT_SECONDS = 10.0
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
_MAX_TICKER_CANDIDATES = 8
_PREFERRED_QUOTE_TYPES = {"EQUITY", "ETF", "INDEX", "MUTUALFUND"}

# Retry / timeout budget (per tool call).
_MAX_ATTEMPTS = 3                  # 1 initial try + up to 2 retries
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 4.0
_YF_ATTEMPT_TIMEOUT_SECONDS = 20.0  # wall-clock cap for one yfinance attempt
_TOTAL_BUDGET_SECONDS = 40.0        # wall-clock cap across all attempts
_RATE_LIMIT_COOLDOWN_SECONDS = 60.0

# Bounded worker pool for yfinance calls: caps concurrency against Yahoo and
# lets us enforce a wall-clock timeout (yfinance's own timeouts are not
# configurable for property access such as ``Ticker.info``).
_YF_MAX_WORKERS = 4
_yf_executor = ThreadPoolExecutor(max_workers=_YF_MAX_WORKERS, thread_name_prefix="yf-tool")

# Cache TTLs (seconds).
_TTL_QUOTE = 60                 # Ticker.info (price + profile)
_TTL_NEWS = 10 * 60
_TTL_HISTORY_RECENT = 5 * 60    # ranges that include today / the future
_TTL_HISTORY_PAST = 60 * 60     # ranges fully in the past
_TTL_FUNDAMENTALS = 6 * 60 * 60  # statements, holders, dividends, splits
_TTL_RECOMMENDATIONS = 60 * 60
_TTL_TICKER_SEARCH = 24 * 60 * 60
_CACHE_MAX_ENTRIES = 256

# Output size limits (keep tool output within a reasonable LLM context size).
_MAX_HISTORY_ROWS = 500
_MAX_NEWS_ITEMS = 10
_MAX_NEWS_SUMMARY_CHARS = 500

# Input validation.
# Yahoo symbols: optional leading '^' (indices), then letters/digits plus
# '.', '-', '=', '&' (e.g. RELIANCE.NS, 500325.BO, BRK-B, EURUSD=X, GC=F, M&M.NS).
_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=&]{0,19}$")
_COMPANY_NAME_MAX_LEN = 100
_COMPANY_NAME_RE = re.compile(r"^[\w\s&.,'’()\-/+:]+$", re.UNICODE)
_DATE_FORMAT = "%Y-%m-%d"

_YF_SOURCE = {"provider": "Yahoo Finance", "via": "yfinance"}
_SEARCH_SOURCE = {"provider": "Yahoo Finance", "via": "Yahoo Finance search API"}
_NO_FABRICATION_HINT = "Do not estimate or fabricate this data; tell the user it is unavailable."


# --------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------
class _ToolError(Exception):
    """Controlled failure whose ``message`` is safe to show to the LLM/user."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class _HTTPStatusError(Exception):
    """Non-200 response from an endpoint we call directly."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    ConnectionError,
    TimeoutError,
    *_CURL_TRANSIENT_ERRORS,
)

_URL_QUERY_RE = re.compile(r"(https?://[^\s?'\"]+)\?[^\s'\"]*")
_SECRET_RE = re.compile(
    r"(?i)\b(crumb|api[_-]?key|apikey|token|access[_-]?token|cookie|authorization|password|secret)"
    r"(\s*[=:]\s*)([^\s&,'\"]+)"
)


def _safe_exc(exc: BaseException) -> str:
    """Exception summary for logs with URL query strings and secrets redacted."""
    message = _URL_QUERY_RE.sub(r"\1?<redacted>", str(exc))
    message = _SECRET_RE.sub(r"\1\2<redacted>", message)
    return f"{type(exc).__name__}: {message[:300]}"


def _http_status_of(exc: BaseException) -> Optional[int]:
    if isinstance(exc, _HTTPStatusError):
        return exc.status_code
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _classify_exception(exc: BaseException) -> str:
    """Return one of: 'rate_limited', 'no_data', 'transient', 'fatal'."""
    if isinstance(exc, YFRateLimitError):
        return "rate_limited"
    status = _http_status_of(exc)
    if status is not None:
        if status == 429:
            return "rate_limited"
        if status == 404:
            return "no_data"
        if status >= 500:
            return "transient"
        return "fatal"
    if _YF_NO_DATA_ERRORS and isinstance(exc, _YF_NO_DATA_ERRORS):
        return "no_data"
    if isinstance(exc, _TRANSIENT_ERRORS):
        return "transient"
    return "fatal"


# --------------------------------------------------------------------------------
# Rate-limit circuit breaker
# --------------------------------------------------------------------------------
_breaker_lock = threading.Lock()
_rate_limited_until = 0.0


def _trip_rate_limit_breaker() -> None:
    global _rate_limited_until
    with _breaker_lock:
        _rate_limited_until = time.monotonic() + _RATE_LIMIT_COOLDOWN_SECONDS


def _rate_limit_remaining() -> float:
    with _breaker_lock:
        return max(0.0, _rate_limited_until - time.monotonic())


def _rate_limited_error(wait_seconds: float) -> _ToolError:
    return _ToolError(
        "rate_limited",
        "Yahoo Finance is currently rate-limiting requests. Do not retry immediately; "
        f"try again in about {max(1, math.ceil(wait_seconds))} seconds.",
    )


# --------------------------------------------------------------------------------
# Bounded execution + retry
# --------------------------------------------------------------------------------
T = TypeVar("T")


def _run_bounded(fn: Callable[[], T], timeout_seconds: float) -> T:
    """Run ``fn`` in the shared worker pool with a wall-clock timeout.

    On timeout the worker thread cannot be killed, but it is bounded by
    yfinance's own internal HTTP timeouts and we do not retry, so timed-out
    calls never pile up.
    """
    future = _yf_executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        if future.done():
            # ``fn`` itself raised a TimeoutError (e.g. socket timeout): let the
            # retry layer classify it as a transient network error.
            raise
        future.cancel()
        raise _ToolError(
            "timeout",
            f"Yahoo Finance did not respond within {timeout_seconds:.0f} seconds. Please try again later.",
        ) from None


def _call_with_retry(attempt_fn: Callable[[float], T], *, operation: str) -> T:
    """Call ``attempt_fn(time_budget_seconds)`` with bounded retries.

    Only transient network failures are retried. Rate limits, timeouts of our
    own wall-clock budget, and all other errors fail fast.
    """
    deadline = time.monotonic() + _TOTAL_BUDGET_SECONDS

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        wait = _rate_limit_remaining()
        if wait > 0:
            raise _rate_limited_error(wait)

        budget = deadline - time.monotonic()
        if budget <= 1.0:
            raise _ToolError("timeout", "Yahoo Finance request took too long. Please try again later.")

        try:
            return attempt_fn(budget)
        except _ToolError:
            raise
        except Exception as exc:
            kind = _classify_exception(exc)

            if kind == "rate_limited":
                _trip_rate_limit_breaker()
                logger.warning("%s: rate-limited by Yahoo Finance (%s)", operation, type(exc).__name__)
                raise _rate_limited_error(_RATE_LIMIT_COOLDOWN_SECONDS) from None

            if kind == "no_data":
                logger.info("%s: provider reported no data (%s)", operation, type(exc).__name__)
                raise _ToolError("no_data", "") from None

            if kind != "transient":
                logger.error("%s: upstream error: %s", operation, _safe_exc(exc))
                raise _ToolError(
                    "upstream_error",
                    "Yahoo Finance returned an unexpected error for this request. Please try again later.",
                ) from None

            delay = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            delay *= random.uniform(0.8, 1.2)  # jitter
            if attempt >= _MAX_ATTEMPTS or time.monotonic() + delay >= deadline:
                logger.error(
                    "%s: transient failure, giving up after %d attempt(s): %s",
                    operation, attempt, _safe_exc(exc),
                )
                raise _ToolError(
                    "network_error",
                    f"Could not reach Yahoo Finance after {attempt} attempt(s). Please try again later.",
                ) from None

            logger.warning(
                "%s: transient failure (attempt %d/%d), retrying in %.2fs: %s",
                operation, attempt, _MAX_ATTEMPTS, delay, _safe_exc(exc),
            )
            time.sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise _ToolError("network_error", "Could not reach Yahoo Finance. Please try again later.")


# --------------------------------------------------------------------------------
# TTL cache
# --------------------------------------------------------------------------------
_MISS = object()


class _TTLCache:
    """Small thread-safe LRU cache with per-entry expiry.

    Values are deep-copied on read so callers can never mutate cached data.
    """

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._data: OrderedDict[tuple, tuple[float, str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Any:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return _MISS
            expires_at, fetched_at, value = entry
            if expires_at <= time.monotonic():
                del self._data[key]
                return _MISS
            self._data.move_to_end(key)
        return fetched_at, copy.deepcopy(value)

    def set(self, key: tuple, value: Any, ttl_seconds: float, fetched_at: str) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + ttl_seconds, fetched_at, copy.deepcopy(value))
            self._data.move_to_end(key)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_cache = _TTLCache(_CACHE_MAX_ENTRIES)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fetch_cached(
    key: tuple,
    ttl_seconds: float,
    operation: str,
    attempt_fn: Callable[[float], Any],
) -> tuple[Any, bool, str]:
    """Return ``(data, cached, fetched_at)``; only non-empty results are cached."""
    hit = _cache.get(key)
    if hit is not _MISS:
        fetched_at, value = hit
        return value, True, fetched_at

    value = _call_with_retry(attempt_fn, operation=operation)
    fetched_at = _utc_now_iso()
    if not _is_empty(value):
        _cache.set(key, value, ttl_seconds, fetched_at)
    return value, False, fetched_at


# --------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------
def _validate_ticker(ticker: Any) -> str:
    """Return a normalized Yahoo symbol or raise ``_ToolError('invalid_input')``.

    Upper-casing is safe for Yahoo symbols, including Indian ones (RELIANCE.NS,
    500325.BO) and indices (^NSEI). A single leading '$' (e.g. "$AAPL") is
    tolerated.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise _ToolError("invalid_input", "A ticker symbol is required (e.g. AAPL, RELIANCE.NS, ^NSEI).")
    symbol = ticker.strip().upper()
    if symbol.startswith("$"):
        symbol = symbol[1:]
    if not _TICKER_RE.fullmatch(symbol):
        raise _ToolError(
            "invalid_input",
            "Invalid ticker symbol format. Use a Yahoo Finance symbol such as AAPL, BRK-B, "
            "RELIANCE.NS, 500325.BO or ^NSEI. Use get_ticker to look up a company's symbol.",
        )
    return symbol


def _validate_company_name(company_name: Any) -> str:
    if not isinstance(company_name, str) or not company_name.strip():
        raise _ToolError("invalid_input", "A company name is required.")
    name = " ".join(company_name.split())  # collapse whitespace / newlines
    if len(name) > _COMPANY_NAME_MAX_LEN:
        raise _ToolError("invalid_input", f"Company name is too long (max {_COMPANY_NAME_MAX_LEN} characters).")
    if not any(ch.isalnum() for ch in name) or not _COMPANY_NAME_RE.fullmatch(name):
        raise _ToolError(
            "invalid_input",
            "Company name contains unsupported characters. Use letters, digits, spaces and basic punctuation.",
        )
    return name


def _validate_date(value: Any, field: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise _ToolError("invalid_input", f"{field} is required and must be formatted as YYYY-MM-DD.")
    try:
        parsed = datetime.strptime(value.strip(), _DATE_FORMAT).date()
    except ValueError:
        raise _ToolError("invalid_input", f"{field} must be a valid date formatted as YYYY-MM-DD.") from None
    if parsed.year < 1900:
        raise _ToolError("invalid_input", f"{field} must be on or after 1900-01-01.")
    return parsed


# --------------------------------------------------------------------------------
# Data conversion (pandas / numpy -> JSON-safe Python)
# --------------------------------------------------------------------------------
def _is_empty(data: Any) -> bool:
    """True for None and for empty DataFrame / Series / dict / list results."""
    if data is None:
        return True
    empty = getattr(data, "empty", None)  # pandas DataFrame / Series
    if empty is not None:
        return bool(empty)
    try:
        return len(data) == 0
    except TypeError:
        return False


def _format_timestamp(value: datetime) -> str:
    """ISO string; date-only when there is no time-of-day component."""
    if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
        return value.date().isoformat()
    return value.isoformat()


def _json_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, tuple):
        return " | ".join(_json_key(k) for k in key)
    converted = _jsonable(key)
    return converted if isinstance(converted, str) else str(converted)


def _jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy/datetime values to JSON-safe Python.

    NaN / NaT / +-inf become ``None`` (missing), never a made-up number.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if value is pd.NaT:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):  # includes pandas.Timestamp
        return _format_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict())
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {_json_key(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)  # numpy scalars
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _frame_to_data(obj: Any, orient: str = "dict") -> Any:
    """Convert a yfinance DataFrame/Series result; ``None`` when empty."""
    if _is_empty(obj):
        return None
    if isinstance(obj, pd.DataFrame):
        if orient == "records":
            frame = obj if isinstance(obj.index, pd.RangeIndex) else obj.reset_index()
            return _jsonable(frame.to_dict(orient="records"))
        return _jsonable(obj.to_dict(orient=orient))
    return _jsonable(obj)


def _epoch_to_iso(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return value


# --------------------------------------------------------------------------------
# Response envelopes
# --------------------------------------------------------------------------------
def _ok(
    tool_name: str,
    data: Any,
    *,
    subject: dict[str, str],
    source: dict[str, str],
    cached: bool,
    as_of: str,
    note: Optional[str] = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {"status": "ok", "tool": tool_name, **subject, "data": data,
                                "source": dict(source), "as_of": as_of, "cached": cached}
    if note:
        response["note"] = note
    return response


def _fail(
    tool_name: str,
    error_type: str,
    message: str,
    *,
    subject: dict[str, str],
    source: dict[str, str],
) -> dict[str, Any]:
    status = "no_data" if error_type == "no_data" else "error"
    response: dict[str, Any] = {"status": status, "tool": tool_name, **subject,
                                "error_type": error_type, "message": message, "source": dict(source)}
    if error_type in ("no_data", "rate_limited", "timeout", "network_error", "upstream_error", "internal_error"):
        response["guidance"] = _NO_FABRICATION_HINT
    return response


def _no_data_message(what: str, symbol: str) -> str:
    return (
        f"No {what} available for {symbol} from Yahoo Finance. The symbol may be wrong or delisted, "
        "or Yahoo may not cover this dataset for this market. Use get_ticker to verify the symbol."
    )


# --------------------------------------------------------------------------------
# Shared tool runner for yfinance-backed tools
# --------------------------------------------------------------------------------
def _run_yf_tool(
    tool_name: str,
    ticker: Any,
    *,
    what: str,
    dataset: str,
    ttl_seconds: float,
    fetch: Callable[[yf.Ticker], Any],
    transform: Optional[Callable[[Any], Any]] = None,
    key_extra: tuple = (),
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Validate -> cache lookup -> bounded/retried yfinance fetch -> envelope.

    ``fetch`` receives a ``yf.Ticker`` and must return JSON-safe data (it runs
    in the worker pool). ``dataset`` + ``key_extra`` form the cache key, so
    tools reading the same yfinance dataset (e.g. ``Ticker.info``) share it.
    ``transform`` post-processes (possibly cached) data for this tool.
    """
    started = time.monotonic()
    try:
        symbol = _validate_ticker(ticker)
    except _ToolError as err:
        logger.info("%s: rejected invalid ticker input", tool_name)
        return _fail(tool_name, err.error_type, err.message, subject={}, source=_YF_SOURCE)

    subject = {"ticker": symbol}
    operation = f"{tool_name}({symbol})"
    logger.info("%s: retrieving %s", operation, what)

    try:
        data, cached, as_of = _fetch_cached(
            (dataset, symbol, *key_extra),
            ttl_seconds,
            operation,
            lambda budget: _run_bounded(
                lambda: fetch(yf.Ticker(symbol)),
                min(_YF_ATTEMPT_TIMEOUT_SECONDS, budget),
            ),
        )
        if transform is not None and not _is_empty(data):
            data = transform(data)
    except _ToolError as err:
        message = err.message or _no_data_message(what, symbol)
        return _fail(tool_name, err.error_type, message, subject=subject, source=_YF_SOURCE)
    except Exception as exc:  # defensive: never leak raw exceptions to the LLM
        logger.error("%s: unexpected failure: %s", operation, _safe_exc(exc))
        return _fail(tool_name, "internal_error",
                     f"Failed to retrieve {what} for {symbol}. Please try again later.",
                     subject=subject, source=_YF_SOURCE)

    if _is_empty(data):
        return _fail(tool_name, "no_data", _no_data_message(what, symbol), subject=subject, source=_YF_SOURCE)

    logger.info("%s: retrieved %s in %.3fs (cached=%s)", operation, what, time.monotonic() - started, cached)
    return _ok(tool_name, data, subject=subject, source=_YF_SOURCE, cached=cached, as_of=as_of, note=note)


# --------------------------------------------------------------------------------
# Dataset fetchers / transforms
# --------------------------------------------------------------------------------
def _fetch_info(stock: yf.Ticker) -> Optional[dict[str, Any]]:
    info = _jsonable(stock.info or {})
    # Unknown symbols typically yield {} or {"trailingPegRatio": None}.
    if not isinstance(info, dict) or not any(
        v is not None for k, v in info.items() if k != "trailingPegRatio"
    ):
        return None
    return info


def _price_payload(info: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract the quote from ``Ticker.info`` without inventing missing fields."""
    price, price_field = None, None
    for field in ("regularMarketPrice", "currentPrice"):
        if info.get(field) is not None:
            price, price_field = info[field], field
            break
    if price is None:
        return None

    payload: dict[str, Any] = {"price": price, "price_field": price_field}
    optional_fields = {
        "currency": info.get("currency"),
        "name": info.get("shortName") or info.get("longName"),
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        "market_state": info.get("marketState"),
        "previous_close": info.get("regularMarketPreviousClose") or info.get("previousClose"),
        "change": info.get("regularMarketChange"),
        "change_percent": info.get("regularMarketChangePercent"),
        "market_time": _epoch_to_iso(info.get("regularMarketTime")),
    }
    payload.update({k: v for k, v in optional_fields.items() if v is not None})
    return payload


def _compact_news(items: Any) -> Optional[list[dict[str, Any]]]:
    """Reduce yfinance news items to the fields an agent needs."""
    if not isinstance(items, list):
        return None
    compact: list[dict[str, Any]] = []
    for item in items:
        if len(compact) >= _MAX_NEWS_ITEMS:
            break
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider")
        summary = content.get("summary")
        entry = {
            "title": title,
            "publisher": provider.get("displayName") if isinstance(provider, dict) else content.get("publisher"),
            "published_at": content.get("pubDate") or content.get("displayTime")
            or _epoch_to_iso(content.get("providerPublishTime")),
            "url": (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or content.get("link"),
            "summary": summary[:_MAX_NEWS_SUMMARY_CHARS] if isinstance(summary, str) and summary else None,
        }
        compact.append({k: v for k, v in entry.items() if v is not None})
    return compact or None


def _major_holders_data(frame: Any) -> Any:
    """major_holders is a single 'Value' column indexed by breakdown label."""
    if isinstance(frame, pd.DataFrame) and not frame.empty and frame.shape[1] == 1:
        return _jsonable(frame.iloc[:, 0].to_dict())
    return _frame_to_data(frame)


# --------------------------------------------------------------------------------
# Tool 1: Retrieve Company Stock Price
# --------------------------------------------------------------------------------
@tool('get_stock_price', description="A function that returns the current stock price of a given ticker. Indian tickers use Yahoo suffixes: .NS for NSE (e.g. RELIANCE.NS) and .BO for BSE.")
def get_stock_price(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_stock_price", ticker,
        what="price data", dataset="info", ttl_seconds=_TTL_QUOTE,
        fetch=_fetch_info, transform=_price_payload,
        note="Yahoo Finance quotes may be delayed; check market_time/market_state.",
    )


# --------------------------------------------------------------------------------
# Tool 2: Retrieve Company Stock Historical Data
# --------------------------------------------------------------------------------
@tool('get_historical_data', description="A function that returns the historical data of a given ticker in the given start and end date. Dates must be strings formatted as YYYY-MM-DD; end_date is exclusive.")
def get_historical_data(ticker: str, start_date: str, end_date: str) -> dict[str, Any]:
    tool_name = "get_historical_data"

    try:
        symbol = _validate_ticker(ticker)
    except _ToolError as err:
        logger.info("%s: rejected invalid ticker input", tool_name)
        return _fail(tool_name, err.error_type, err.message, subject={}, source=_YF_SOURCE)

    subject = {"ticker": symbol}

    try:
        start = _validate_date(start_date, "start_date")
        end = _validate_date(end_date, "end_date")
        if start >= end:
            raise _ToolError("invalid_input", "end_date must be after start_date (end_date is exclusive).")
        if start > date.today():
            raise _ToolError("invalid_input", "start_date is in the future; no historical data exists yet.")
    except _ToolError as err:
        return _fail(tool_name, err.error_type, err.message, subject=subject, source=_YF_SOURCE)

    start_s, end_s = start.isoformat(), end.isoformat()

    def fetch(stock: yf.Ticker) -> Optional[dict[str, Any]]:
        history = stock.history(start=start_s, end=end_s)
        if _is_empty(history):
            return None
        total_rows = len(history)
        truncated = total_rows > _MAX_HISTORY_ROWS
        if truncated:
            history = history.tail(_MAX_HISTORY_ROWS)  # keep the most recent rows
        result: dict[str, Any] = {
            "start_date": start_s,
            "end_date": end_s,
            "interval": "1d",
            "total_rows": total_rows,
            "returned_rows": len(history),
            "truncated": truncated,
            "rows": _frame_to_data(history, orient="index"),
        }
        if truncated:
            result["truncation_note"] = (
                f"Only the most recent {_MAX_HISTORY_ROWS} rows are included; "
                "request a narrower date range for earlier data."
            )
        return result

    return _run_yf_tool(
        tool_name, ticker,
        what=f"historical data between {start_s} and {end_s}", dataset="history",
        ttl_seconds=_TTL_HISTORY_PAST if end <= date.today() else _TTL_HISTORY_RECENT,
        fetch=fetch, key_extra=(start_s, end_s),
    )


# --------------------------------------------------------------------------------
# Tool 3: Retrieve Company Stock News
# --------------------------------------------------------------------------------
@tool('get_stock_news', description="A function that returns the news of a given ticker")
def get_stock_news(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_stock_news", ticker,
        what="news", dataset="news", ttl_seconds=_TTL_NEWS,
        fetch=lambda stock: _compact_news(_jsonable(stock.news)),
    )


# --------------------------------------------------------------------------------
# Tool 4: Retrieve Company's Balance Sheet
# --------------------------------------------------------------------------------
@tool('get_balance_sheet', description="A function that returns the balance sheet of a given ticker")
def get_balance_sheet(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_balance_sheet", ticker,
        what="balance sheet", dataset="balance_sheet", ttl_seconds=_TTL_FUNDAMENTALS,
        fetch=lambda stock: _frame_to_data(stock.balance_sheet),
        note="Annual statements keyed by period end date, then line item.",
    )


# --------------------------------------------------------------------------------
# Tool 5: Retrieve Company's Income Statement
# --------------------------------------------------------------------------------
@tool('get_income_statement', description="A function that returns the income statement of a given ticker")
def get_income_statement(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_income_statement", ticker,
        what="income statement", dataset="income_statement", ttl_seconds=_TTL_FUNDAMENTALS,
        fetch=lambda stock: _frame_to_data(stock.financials),
        note="Annual statements keyed by period end date, then line item.",
    )


# --------------------------------------------------------------------------------
# Tool 6: Retrieve Company's Cash Flow Statement
# --------------------------------------------------------------------------------
@tool('get_cash_flow', description="A function that returns the cash flow statement of a given ticker")
def get_cash_flow(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_cash_flow", ticker,
        what="cash flow statement", dataset="cash_flow", ttl_seconds=_TTL_FUNDAMENTALS,
        fetch=lambda stock: _frame_to_data(stock.cashflow),
        note="Annual statements keyed by period end date, then line item.",
    )


# --------------------------------------------------------------------------------
# Tool 7: Retrieve Company Info & Ratios
# --------------------------------------------------------------------------------
@tool('get_company_info', description="A function that returns company profile and key financial ratios")
def get_company_info(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_company_info", ticker,
        what="company info", dataset="info", ttl_seconds=_TTL_QUOTE,
        fetch=_fetch_info,
    )


# --------------------------------------------------------------------------------
# Tool 8: Retrieve Dividend History
# --------------------------------------------------------------------------------
@tool('get_dividends', description="A function that returns the dividend payment history of a given ticker")
def get_dividends(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_dividends", ticker,
        what="dividends", dataset="dividends", ttl_seconds=_TTL_FUNDAMENTALS,
        fetch=lambda stock: _frame_to_data(stock.dividends),
        note="Mapping of ex-dividend date to dividend amount per share (in the listing currency).",
    )


# --------------------------------------------------------------------------------
# Tool 9: Retrieve Stock Split History
# --------------------------------------------------------------------------------
@tool('get_splits', description="A function that returns the stock split history of a given ticker")
def get_splits(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_splits", ticker,
        what="stock splits", dataset="splits", ttl_seconds=_TTL_FUNDAMENTALS,
        fetch=lambda stock: _frame_to_data(stock.splits),
        note="Mapping of split date to split ratio (e.g. 4.0 means 4-for-1).",
    )


# --------------------------------------------------------------------------------
# Tool 10: Retrieve Institutional Holders
# --------------------------------------------------------------------------------
@tool('get_institutional_holders', description="A function that returns the institutional ownership data of a given ticker")
def get_institutional_holders(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_institutional_holders", ticker,
        what="institutional holders", dataset="institutional_holders", ttl_seconds=_TTL_FUNDAMENTALS,
        fetch=lambda stock: _frame_to_data(stock.institutional_holders, orient="records"),
    )


# --------------------------------------------------------------------------------
# Tool 11: Retrieve Major Share Holders
# --------------------------------------------------------------------------------
@tool('get_major_shareholders', description="A function that returns the major share holder data of a given ticker")
def get_major_shareholders(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_major_shareholders", ticker,
        what="major share holders", dataset="major_holders", ttl_seconds=_TTL_FUNDAMENTALS,
        fetch=lambda stock: _major_holders_data(stock.major_holders),
    )


# --------------------------------------------------------------------------------
# Tool 12: Retrieve Mutual Fund Holders
# --------------------------------------------------------------------------------
@tool('get_mutual_fund_holders', description="A function that returns the mutual fund ownership data of a given ticker")
def get_mutual_fund_holders(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_mutual_fund_holders", ticker,
        what="mutual fund holders", dataset="mutualfund_holders", ttl_seconds=_TTL_FUNDAMENTALS,
        fetch=lambda stock: _frame_to_data(stock.mutualfund_holders, orient="records"),
    )


# --------------------------------------------------------------------------------
# Tool 13: Retrieve Insider Transactions
# --------------------------------------------------------------------------------
@tool('get_insider_transactions', description="A function that returns the insider buy/sell transactions of a given ticker")
def get_insider_transactions(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_insider_transactions", ticker,
        what="insider transactions", dataset="insider_transactions", ttl_seconds=_TTL_FUNDAMENTALS,
        fetch=lambda stock: _frame_to_data(stock.insider_transactions, orient="records"),
    )


# --------------------------------------------------------------------------------
# Tool 14: Retrieve Analyst Recommendations
#
# This is the single analyst tool. The former `get_analyst_recommendations_summary`
# was removed because, in yfinance 0.2.66, `Ticker.get_recommendations_summary()`
# just returns `get_recommendations()` (identical data), so the two tools were
# exact duplicates.
# --------------------------------------------------------------------------------
@tool('get_analyst_recommendations', description="A function that returns the analyst recommendations of a given ticker (counts of strong buy / buy / hold / sell / strong sell ratings by period)")
def get_analyst_recommendations(ticker: str) -> dict[str, Any]:
    return _run_yf_tool(
        "get_analyst_recommendations", ticker,
        what="analyst recommendations", dataset="recommendations", ttl_seconds=_TTL_RECOMMENDATIONS,
        fetch=lambda stock: _frame_to_data(stock.recommendations, orient="records"),
        note="period '0m' is the current month, '-1m' the previous month, and so on.",
    )


# --------------------------------------------------------------------------------
# Tool 15: Retrieve Company's Ticker/Symbol candidates
# --------------------------------------------------------------------------------
def _exchange_hint(symbol: str, exchange_code: str) -> str:
    """Human-readable market hint, mainly to help pick Indian listings."""
    symbol = (symbol or "").upper()
    if symbol.endswith(".NS") or exchange_code == "NSI":
        return "India - NSE"
    if symbol.endswith(".BO") or exchange_code in ("BSE", "BOM"):
        return "India - BSE"
    return ""


def _parse_ticker_candidates(data: Any) -> list[dict[str, str]]:
    quotes = (data or {}).get("quotes") if isinstance(data, dict) else None
    candidates: list[dict[str, str]] = []
    for quote in quotes or []:
        if not isinstance(quote, dict):
            continue
        symbol = quote.get("symbol")
        if not symbol or not isinstance(symbol, str):
            continue
        candidates.append({
            "symbol": symbol,
            "name": quote.get("longname") or quote.get("shortname") or "",
            "exchange": quote.get("exchDisp") or quote.get("exchange") or "",
            "type": quote.get("quoteType") or quote.get("typeDisp") or "",
            "market": _exchange_hint(symbol, quote.get("exchange") or ""),
        })

    preferred = [c for c in candidates if str(c["type"]).upper() in _PREFERRED_QUOTE_TYPES]
    return (preferred or candidates)[:_MAX_TICKER_CANDIDATES]


def _search_yahoo(query: str, budget_seconds: float) -> list[dict[str, str]]:
    """One search attempt against Yahoo's symbol search endpoint."""
    response = requests.get(
        _YAHOO_SEARCH_URL,
        params={"q": query, "quotesCount": 10, "newsCount": 0},
        headers=_BROWSER_HEADERS,
        timeout=(
            min(_HTTP_CONNECT_TIMEOUT_SECONDS, budget_seconds),
            min(_HTTP_READ_TIMEOUT_SECONDS, budget_seconds),
        ),
    )
    if response.status_code != 200:
        raise _HTTPStatusError(response.status_code)
    try:
        payload = response.json()
    except ValueError:
        raise _ToolError("upstream_error", "Yahoo Finance search returned an unreadable response.") from None
    return _parse_ticker_candidates(payload)


@tool('get_ticker', description=(
    "A function that searches for the ticker/symbol of a given company name and returns a list of "
    "candidate matches (symbol, name, exchange, type, market) in relevance order. Choose the best "
    "match for the user's intent, and ask the user if it is ambiguous. Indian listings end in .NS "
    "(NSE) or .BO (BSE); prefer those when the user asks about an Indian company."
))
def get_ticker(company_name: str) -> dict[str, Any]:
    tool_name = "get_ticker"
    started = time.monotonic()
    try:
        query = _validate_company_name(company_name)
    except _ToolError as err:
        logger.info("%s: rejected invalid company name input", tool_name)
        return _fail(tool_name, err.error_type, err.message, subject={}, source=_SEARCH_SOURCE)

    subject = {"query": query}
    operation = f"{tool_name}({query!r})"
    logger.info("%s: searching ticker candidates", operation)

    try:
        candidates, cached, as_of = _fetch_cached(
            ("ticker_search", query.casefold()),
            _TTL_TICKER_SEARCH,
            operation,
            lambda budget: _search_yahoo(query, budget),
        )
    except _ToolError as err:
        message = err.message or f"No ticker candidates found for '{query}'."
        return _fail(tool_name, err.error_type, message, subject=subject, source=_SEARCH_SOURCE)
    except Exception as exc:  # defensive: never leak raw exceptions to the LLM
        logger.error("%s: unexpected failure: %s", operation, _safe_exc(exc))
        return _fail(tool_name, "internal_error",
                     f"Failed to search tickers for '{query}'. Please try again later.",
                     subject=subject, source=_SEARCH_SOURCE)

    if not candidates:
        return _fail(tool_name, "no_data",
                     f"No ticker candidates found for '{query}'. Try the company's full or official name.",
                     subject=subject, source=_SEARCH_SOURCE)

    logger.info("%s: %d candidate(s) in %.3fs (cached=%s)",
                operation, len(candidates), time.monotonic() - started, cached)
    return _ok(tool_name, candidates, subject=subject, source=_SEARCH_SOURCE, cached=cached, as_of=as_of)
