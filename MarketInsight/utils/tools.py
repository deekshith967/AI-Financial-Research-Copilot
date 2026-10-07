import time
import requests
import yfinance as yf
from langchain.tools import tool
from MarketInsight.utils.logger import get_logger

logger = get_logger("Tools")

# Yahoo's search endpoint tends to reject the default python-requests User-Agent.
_YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
_HTTP_TIMEOUT_SECONDS = 10
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
_MAX_TICKER_CANDIDATES = 8
_PREFERRED_QUOTE_TYPES = {"EQUITY", "ETF", "INDEX", "MUTUALFUND"}


# --------------------------------------------------------------------------------
# Small shared helpers (kept deliberately minimal; the larger tool-layer refactor
# is a later phase)
# --------------------------------------------------------------------------------
_INVALID_TICKER_MSG = "Error: Invalid ticker provided. Please provide a valid ticker symbol."


def _clean_ticker(ticker):
    """Return a stripped, upper-cased ticker, or None if it is unusable.

    Upper-casing is safe for Yahoo symbols, including Indian ones (RELIANCE.NS,
    500325.BO) and indices (^NSEI).
    """
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        return None
    return ticker.strip().upper()


def _is_empty(data) -> bool:
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


def _result_or_message(data, what: str, ticker: str):
    """Return a plain-Python result, or an explanatory message when there is no data.

    yfinance can return None (e.g. holders for non-US tickers) or an empty
    DataFrame/Series; ``.to_dict()`` on those would either raise or produce ``{}``,
    which tells the LLM nothing.
    """
    if _is_empty(data):
        return f"No {what} available for {ticker}."
    return data.to_dict() if hasattr(data, "to_dict") else data


# --------------------------------------------------------------------------------
# Tool 1: Retrieve Company Stock Price
# --------------------------------------------------------------------------------
@tool('get_stock_price', description="A function that returns the current stock price of a given ticker. Indian tickers use Yahoo suffixes: .NS for NSE (e.g. RELIANCE.NS) and .BO for BSE.")
def get_stock_price(ticker: str):
    logger.info(f"Retrieving Stock Price of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    start_time = time.time()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        stock_price = info.get('regularMarketPrice')
        if stock_price is None:
            stock_price = info.get('currentPrice')
        end_time = time.time()

        if stock_price is None:
            return f"No price data available for {ticker}."

        logger.info(f"Retrieved Stock Price of {ticker} in {end_time - start_time:.3f} seconds")
        return stock_price

    except Exception as e:
        logger.error(f"Failed to retrieve stock price of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve stock price for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 2: Retrieve Company Stock Historical Data
# --------------------------------------------------------------------------------
@tool('get_historical_data', description="A function that returns the historical data of a given ticker in the given start and end date. Dates must be strings formatted as YYYY-MM-DD.")
def get_historical_data(ticker: str, start_date: str, end_date: str):
    logger.info(f"Retrieving Historical Data of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)

        if _is_empty(history):
            return f"No historical data available for {ticker} between {start_date} and {end_date}."

        end_time = time.time()
        logger.info(f"Retrieved Historical Data of {ticker} in {end_time - start_time:.3f} seconds")
        return history.to_dict()

    except Exception as e:
        logger.error(f"Failed to retrieve historical data of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve historical data for {ticker}. Check that the dates are YYYY-MM-DD, then try again later."


# --------------------------------------------------------------------------------
# Tool 3: Retrieve Company Stock News
# --------------------------------------------------------------------------------
@tool('get_stock_news', description="A function that returns the news of a given ticker")
def get_stock_news(ticker: str):
    logger.info(f"Retrieving News of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        news = stock.news

        if _is_empty(news):
            return f"No news available for {ticker}."

        end_time = time.time()
        logger.info(f"Retrieved News of {ticker} in {end_time - start_time:.3f} seconds")
        return news

    except Exception as e:
        logger.error(f"Failed to retrieve news of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve news for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 4: Retrieve Company's Balance Sheet
# --------------------------------------------------------------------------------
@tool('get_balance_sheet', description="A function that returns the balance sheet of a given ticker")
def get_balance_sheet(ticker: str):
    logger.info(f"Retrieving Balance Sheet of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        balance_sheet = _result_or_message(stock.balance_sheet, "balance sheet", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Balance Sheet of {ticker} in {end_time - start_time:.3f} seconds")
        return balance_sheet

    except Exception as e:
        logger.error(f"Failed to retrieve balance sheet of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve balance sheet for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 5: Retrieve Company's Income Statement
# --------------------------------------------------------------------------------
@tool('get_income_statement', description="A function that returns the income statement of a given ticker")
def get_income_statement(ticker: str):
    logger.info(f"Retrieving Income Statement of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        income_statement = _result_or_message(stock.financials, "income statement", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Income Statement of {ticker} in {end_time - start_time:.3f} seconds")
        return income_statement

    except Exception as e:
        logger.error(f"Failed to retrieve income statement of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve income statement for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 6: Retrieve Company's Cash Flow Statement
# --------------------------------------------------------------------------------
@tool('get_cash_flow', description="A function that returns the cash flow statement of a given ticker")
def get_cash_flow(ticker: str):
    logger.info(f"Retrieving Cash Flow of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        cash_flow = _result_or_message(stock.cashflow, "cash flow statement", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Cash Flow of {ticker} in {end_time - start_time:.3f} seconds")
        return cash_flow

    except Exception as e:
        logger.error(f"Failed to retrieve cash flow of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve cash flow for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 7: Retrieve Company Info & Ratios
# --------------------------------------------------------------------------------
@tool('get_company_info', description="A function that returns company profile and key financial ratios")
def get_company_info(ticker: str):
    logger.info(f"Retrieving Company Info of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        info = _result_or_message(stock.info, "company info", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Company Info of {ticker} in {end_time - start_time:.3f} seconds")
        return info

    except Exception as e:
        logger.error(f"Failed to retrieve company info of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve company info for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 8: Retrieve Dividend History
# --------------------------------------------------------------------------------
@tool('get_dividends', description="A function that returns the dividend payment history of a given ticker")
def get_dividends(ticker: str):
    logger.info(f"Retrieving Dividends of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        dividends = _result_or_message(stock.dividends, "dividends", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Dividends of {ticker} in {end_time - start_time:.3f} seconds")
        return dividends

    except Exception as e:
        logger.error(f"Failed to retrieve dividends of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve dividends for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 9: Retrieve Stock Split History
# --------------------------------------------------------------------------------
@tool('get_splits', description="A function that returns the stock split history of a given ticker")
def get_splits(ticker: str):
    logger.info(f"Retrieving Stock Splits of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        splits = _result_or_message(stock.splits, "stock splits", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Stock Splits of {ticker} in {end_time - start_time:.3f} seconds")
        return splits

    except Exception as e:
        logger.error(f"Failed to retrieve stock splits of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve stock splits for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 10: Retrieve Institutional Holders
# --------------------------------------------------------------------------------
@tool('get_institutional_holders', description="A function that returns the institutional ownership data of a given ticker")
def get_institutional_holders(ticker: str):
    logger.info(f"Retrieving Institutional Holders of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        holders = _result_or_message(stock.institutional_holders, "institutional holders", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Institutional Holders of {ticker} in {end_time - start_time:.3f} seconds")
        return holders

    except Exception as e:
        logger.error(f"Failed to retrieve institutional holders of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve institutional holders for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 11: Retrieve Major Share Holders
# --------------------------------------------------------------------------------
@tool('get_major_shareholders', description="A function that returns the major share holder data of a given ticker")
def get_major_shareholders(ticker: str):
    logger.info(f"Retrieving Major Share Holders of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        holders = _result_or_message(stock.major_holders, "major share holders", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Major Share Holders of {ticker} in {end_time - start_time:.3f} seconds")
        return holders

    except Exception as e:
        logger.error(f"Failed to retrieve major share holders of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve major share holders for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 12: Retrieve Mutual Fund Holders
# --------------------------------------------------------------------------------
@tool('get_mutual_fund_holders', description="A function that returns the mutual fund ownership data of a given ticker")
def get_mutual_fund_holders(ticker: str):
    logger.info(f"Retrieving Mutual Fund Holders of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        holders = _result_or_message(stock.mutualfund_holders, "mutual fund holders", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Mutual Fund Holders of {ticker} in {end_time - start_time:.3f} seconds")
        return holders

    except Exception as e:
        logger.error(f"Failed to retrieve mutual fund holders of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve mutual fund holders for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 13: Retrieve Insider Transactions
# --------------------------------------------------------------------------------
@tool('get_insider_transactions', description="A function that returns the insider buy/sell transactions of a given ticker")
def get_insider_transactions(ticker: str):
    logger.info(f"Retrieving Insider Transactions of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        insider_txn = _result_or_message(stock.insider_transactions, "insider transactions", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Insider Transactions of {ticker} in {end_time - start_time:.3f} seconds")
        return insider_txn

    except Exception as e:
        logger.error(f"Failed to retrieve insider transactions of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve insider transactions for {ticker}. Please try again later."


# --------------------------------------------------------------------------------
# Tool 14: Retrieve Analyst Recommendations
#
# This is the single analyst tool. The former `get_analyst_recommendations_summary`
# was removed because, in yfinance 0.2.66, `Ticker.get_recommendations_summary()`
# just returns `get_recommendations()` (identical data), so the two tools were
# exact duplicates.
# --------------------------------------------------------------------------------
@tool('get_analyst_recommendations', description="A function that returns the analyst recommendations of a given ticker (counts of strong buy / buy / hold / sell / strong sell ratings by period)")
def get_analyst_recommendations(ticker: str):
    logger.info(f"Retrieving Analyst Recommendations of {ticker}")

    ticker = _clean_ticker(ticker)
    if ticker is None:
        return _INVALID_TICKER_MSG

    try:
        start_time = time.time()
        stock = yf.Ticker(ticker)
        recommendations = _result_or_message(stock.recommendations, "analyst recommendations", ticker)

        end_time = time.time()
        logger.info(f"Retrieved Analyst Recommendations of {ticker} in {end_time - start_time:.3f} seconds")
        return recommendations

    except Exception as e:
        logger.error(f"Failed to retrieve analyst recommendations of {ticker}: {str(e)}")
        return f"Error: Failed to retrieve analyst recommendations for {ticker}. Please try again later."


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


def _parse_ticker_candidates(data: dict) -> list:
    quotes = (data or {}).get("quotes") or []
    candidates = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if not symbol:
            continue
        candidates.append({
            "symbol": symbol,
            "name": quote.get("longname") or quote.get("shortname") or "",
            "exchange": quote.get("exchDisp") or quote.get("exchange") or "",
            "type": quote.get("quoteType") or quote.get("typeDisp") or "",
            "market": _exchange_hint(symbol, quote.get("exchange") or ""),
        })

    preferred = [c for c in candidates if c["type"].upper() in _PREFERRED_QUOTE_TYPES]
    return (preferred or candidates)[:_MAX_TICKER_CANDIDATES]


@tool('get_ticker', description=(
    "A function that searches for the ticker/symbol of a given company name and returns a list of "
    "candidate matches (symbol, name, exchange, type, market) in relevance order. Choose the best "
    "match for the user's intent, and ask the user if it is ambiguous. Indian listings end in .NS "
    "(NSE) or .BO (BSE); prefer those when the user asks about an Indian company."
))
def get_ticker(company_name: str):
    logger.info(f"Retrieving Ticker of {company_name}")

    if not company_name or not isinstance(company_name, str) or not company_name.strip():
        return "Error: Invalid company name provided. Please provide a valid company name."

    company_name = company_name.strip()

    try:
        start_time = time.time()
        response = requests.get(
            _YAHOO_SEARCH_URL,
            params={"q": company_name, "quotesCount": 10, "newsCount": 0},
            headers=_BROWSER_HEADERS,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            logger.error(f"Ticker search for {company_name} returned HTTP {response.status_code}")
            return f"Error: Ticker search for '{company_name}' failed (HTTP {response.status_code}). Please try again later."

        candidates = _parse_ticker_candidates(response.json())
        end_time = time.time()

        if not candidates:
            return f"No ticker candidates found for '{company_name}'."

        logger.info(f"Retrieved Ticker candidates of {company_name} in {end_time - start_time:.3f} seconds")
        return candidates

    except requests.Timeout:
        logger.error(f"Ticker search for {company_name} timed out")
        return f"Error: Ticker search for '{company_name}' timed out. Please try again later."
    except Exception as e:
        logger.error(f"Failed to retrieve ticker of {company_name}: {str(e)}")
        return f"Error: Failed to retrieve ticker for '{company_name}'. Please try again later."