# Market Insight

An AI-powered stock market analysis platform that provides comprehensive financial data and intelligent insights through a conversational interface.

## Overview

Market Insight leverages advanced AI agents to deliver real-time stock market information, financial analysis, and investment insights. The platform combines the power of LangChain and OpenAI's language models with Yahoo Finance data to create an intelligent assistant for stock market research.

## Technology Stack

**Backend:**
- FastAPI for high-performance API endpoints
- LangChain & LangGraph for AI agent orchestration
- OpenAI GPT models for intelligent responses
- YFinance for financial data retrieval
- Langfuse for observability and tracing

**Frontend:**
- Modern React-based interface
- Real-time streaming responses
- Responsive design for all devices

## Getting Started

### Prerequisites
- Python 3.x
- Node.js (for frontend)
- OpenAI API key

### Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file
4. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```
5. Run the backend server:
   ```bash
   python main.py
   ```
6. Run the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```
7. Access the API at `http://localhost:8000` and frontend at `http://localhost:5173`

## Evaluation (Phase 3)

The platform ships a fully offline evaluation harness in `MarketInsight/evaluation/`.
Retrieval metrics alone cannot tell you whether the *final answer* is faithful,
whether the agent picked the right tools, or what a run costs — so the harness
evaluates three layers independently:

1. **Retrieval quality** — recall@k / MRR / citation coverage against a labeled
   golden dataset (`tests/eval/golden.jsonl`), run against the real local index.
2. **Answer faithfulness** — recorded agent answers (fixtures in
   `tests/eval/agent_fixtures.jsonl`) are scored for:
   - fact coverage and citation presence/correctness (deterministic checks);
   - numeric-claim support against the exact retrieved evidence
     (*heuristic* — it canonicalizes `number+magnitude`, so it can false-positive
     on rounded/re-scaled values and false-negative on spelled-out numbers or
     date-like tokens; heuristic metrics are labeled in the report);
   - fabricated-citation detection (cited source not present in the evidence).
3. **Agent / tool use** — tool-selection accuracy, unexpected-call rate,
   unnecessary-call rate, and execution success from recorded tool calls.

Fixtures are recordings, not live calls: evaluation runs with **no LLM key, no
network, and no live data source**. Telemetry (latency/token counts) is recorded
per run and aggregated honestly: p95 is only reported for n ≥ 20, missing token
counts stay null, and cost is only computed when explicit per-1K prices are
passed on the CLI — nothing is invented. Synthetic negative fixtures
(`synthetic_negative: true`) are scored separately as a detector self-check and
never mixed into the headline metrics. Live-LLM evaluation is intentionally out
of scope for this offline mode; the same scorers can be applied to live traces.

### Latest measured results (offline, this machine)

| Area | Metric | Value |
|------|--------|-------|
| Retrieval (14 questions, live index) | recall@5 / MRR / citation coverage | 1.000 / 0.964 / 1.000 |
| Answer (10 positive fixtures) | fact coverage / grounded answer rate | 1.000 / 1.000 |
| | citation coverage / correctness | 1.000 / 1.000 |
| | unsupported claim rate (heuristic, n=8) | 0.000 |
| Agent | tool-selection accuracy | 1.000 |
| | unexpected / unnecessary tool-call rate | 0.000 / 0.000 |
| | tool execution success (11 calls) | 1.000 |
| Detector self-check | synthetic negatives fully flagged | 2/2 |
| Performance | total latency mean / p50 / max | 1263.9 / 1180.2 / 1890.0 ms |
| | p95 | suppressed (n=10 < 20) |
| | tokens (input/output) | 12080 / 955 (2 runs without token data) |
| | cost | n/a without explicit pricing |

### Run it

```bash
python -m MarketInsight.evaluation                 # text report (fixtures + live retrieval eval)
python -m MarketInsight.evaluation --format json   # machine-readable
python -m MarketInsight.evaluation --skip-retrieval \
    --price-input-per-1k 0.15 --price-output-per-1k 0.60   # explicit cost estimate
```

## Project Structure

```
MarketInsight/
├── components/     # AI agent configuration
├── utils/          # Tools and utilities
├── config/         # Configuration files
├── frontend/       # React frontend application
└── main.py         # FastAPI server entry point
```

## API Capabilities

The platform provides 17 specialized tools for comprehensive stock analysis and document research:
- Stock price tracking
- Historical data analysis
- Financial statements (Balance Sheet, Income Statement, Cash Flow)
- Company information and ratios
- Dividend and split history
- Ownership and holder data
- Insider transactions
- Analyst recommendations
- Company ticker lookup
- Local document/filings search with citations (RAG)
- Document ingestion status