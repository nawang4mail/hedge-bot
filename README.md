# Hedge Bot — Multi-Agent Trading System

A four-agent quantitative trading pipeline powered by **LangGraph** + **Ollama** (local LLM, zero API cost), **TimescaleDB** for persistent market data and trade history, an **XGBoost ML model** trained on price + news + Reddit + earnings data, a **FastAPI** backend, and a suite of real-time HTML dashboards.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         LangGraph Pipeline                           │
│                                                                      │
│  [Observation] ──► [Research] ──► [Decision] ──► [Implementation]   │
│   yfinance /         pandas-ta    XGBoost ML       Alpaca API        │
│   NewsAPI            + Ollama     (falls back       (paper/live)     │
│   GDELT/Reddit       + DB enrich  to Ollama LLM)                    │
│   TimescaleDB cache                                                  │
└──────────────────────────────────────────────────────────────────────┘
              │                                    │
           FastAPI + WebSocket              TimescaleDB
              │                           (tickers, ohlcv,
     ┌────────┴──────────┐               signals, executions)
     │   dashboard.html  │
     │  performance.html │
     └───────────────────┘
```

### Agents & LLM usage

| Agent | LLM | Reason |
|---|---|---|
| Observation | ❌ | Deterministic data fetching — no hallucination risk |
| Research | ✅ small payload | Synthesises pre-computed indicator values only |
| Decision | ✅ small payload | Direction + confidence from indicator summary |
| Implementation | ❌ | Order execution must be 100% deterministic |

### Hallucination firewall

Trade execution is deliberately isolated from LLM reasoning. The LLM sees only pre-computed indicator numbers, never raw prices or order book data. Risk sizing, position limits, and the kill-switch are enforced by deterministic Python code before any order reaches the broker.

---

## Cost

| Component | Cost |
|---|---|
| Ollama + local model | Free |
| TimescaleDB | Free (open source) |
| yfinance market data | Free |
| NewsAPI | Free tier (100 req/day) |
| Alpaca paper trading | Free |
| Alpaca live trading | Free (no commissions) |

**Total running cost: $0/month** on your existing machine.

---

## Quick Start

### Option A — Local (no Docker)

**1. Install Ollama and pull a model**

Download from [ollama.com](https://ollama.com) (macOS: single app install), then:

```bash
ollama pull llama3.1:8b    # ~4.7 GB — or try phi3:mini (~2.3 GB) to save space
ollama serve               # starts on http://localhost:11434
```

**2. Start TimescaleDB**

```bash
docker run -d \
  --name hedge_tsdb \
  -e POSTGRES_USER=hedge \
  -e POSTGRES_PASSWORD=hedge \
  -e POSTGRES_DB=hedgebot \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg16
```

**3. Configure environment**

```bash
cp .env.example .env
# Required: add ALPACA_API_KEY + ALPACA_SECRET_KEY (free at alpaca.markets)
# Optional: add NEWS_API_KEY (free at newsapi.org)
```

**4. Install Python dependencies**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**5. Start the API**

```bash
uvicorn api.main:app --reload
# DB schema and TimescaleDB hypertable are created automatically on first startup
```

**6. Open the dashboard**

Open `ui/dashboard.html` in any browser — no build step needed.

---

### Option B — Docker (everything in containers)

```bash
docker-compose up --build
# TimescaleDB, Ollama, and the API all start together

# Pull a model into the Ollama container (one-time):
docker exec hedge_ollama ollama pull llama3.1:8b
```

Open `ui/dashboard.html` in your browser.

---

## UI Overview

### dashboard.html — Command Center

| Panel | Description |
|---|---|
| **Portfolio Telemetry** | Live equity, cash, buying power, open position count (refreshes every 15s) |
| **Agent Pipeline** | Visual status of each agent (Idle → Processing → Completed) with streaming JSON output |
| **Agent Thought Log** | Real-time scrolling log of every agent handoff and intermediate output |
| **Latest Signal** | Current BUY/SELL/HOLD badge with quantity, order type, confidence, and rationale |
| **Open Positions** | Live positions table from Alpaca with unrealised P&L |
| **Watchlist** | Add/remove tickers; click any chip to pre-fill the run input |
| **Signal History** | Scrollable log of all past decisions stored in TimescaleDB |
| **Execution Log** | All order attempts with fill status, average price, and slippage |
| **Risk Override** | Hot-update max position size, max portfolio risk, and order type without restarting |
| **Kill-Switch** | Instantly halts all automated trading; resume with one click |

### performance.html — Performance Analytics

Opened via the **📊 PERFORMANCE** button. Three tabs: **Daily**, **Weekly**, **Monthly**.

| Section | Metrics |
|---|---|
| **Summary cards** | Total P&L, win rate, W/L count, total signals, buy/sell/hold counts, avg confidence, avg slippage, current win/loss streak |
| **Equity curve** | Cumulative P&L line chart — green if net positive, red if negative |
| **P&L per period** | Bar chart of profit/loss per day, week, or month |
| **Trade breakdown** | Donut chart showing BUY / SELL / HOLD signal split |
| **Best & worst trade** | Side-by-side detail cards for the single best and worst individual trade |
| **P&L by ticker** | Per-symbol table: trade count, buys, sells, total P&L, win rate, avg slippage |
| **Trade count by period** | Raw signal counts per time bucket |

### connections.html — External Integrations

Opened via the **🔌 CONNECTIONS** button. Connect each service once — credentials saved locally to `connections.json`.

| Service | What it provides | Cost |
|---|---|---|
| **Alpaca Markets** | Paper & live trade execution | Free |
| **Google BigQuery (GDELT)** | Years of historical news sentiment | Free tier |
| **Reddit (PRAW)** | r/wallstreetbets, r/investing, r/stocks mentions | Free |
| **NewsAPI** | Recent news headlines | Free (30 days) |
| **Discord** | Watchlist trade alerts + LLM chat bot | Free |

Each card has a **Test Connection** button that makes a live API call and shows the result before you commit to using it.

### training.html — ML Training Centre

Opened via the **🧠 TRAINING** button. Full ML pipeline in the browser.

| Section | What it does |
|---|---|
| **Backfill** | Downloads years of price, news, Reddit, and earnings data per ticker into TimescaleDB |
| **Train Model** | Trains an XGBoost classifier per ticker (daily / hourly / both) using all backfilled data |
| **Backtest** | Walk-forward simulation of the trained model — shows equity curve, win rate, Sharpe, max drawdown, alpha vs buy-and-hold |
| **Models panel** | Lists all trained models with test accuracy, live accuracy, and last-trained date. Flags stale models (>30 days old). |
| **Job log** | Live terminal output of running backfill/training jobs via WebSocket |

---

## ML Training Workflow

### 1. Connect services (one-time)

Open `connections.html` and save your Google BigQuery project ID and Reddit API credentials.

### 2. Backfill historical data

```bash
# Via UI: open training.html → add symbols → check sources → Start Backfill
# Or via CLI:
python -m training.backfill_ohlcv   --symbols AAPL TSLA NVDA --years 5 --hourly
python -m training.backfill_news    --symbols AAPL TSLA NVDA --years 3
python -m training.backfill_reddit  --symbols AAPL TSLA NVDA --days 365
python -m training.backfill_earnings --symbols AAPL TSLA NVDA
```

### 3. Train the model

```bash
# Via UI: training.html → Train Model tab
# Or via CLI:
python -m training.train --symbols AAPL --timeframe daily --years 5
python -m training.train --symbols AAPL --timeframe hourly --years 2
python -m training.train --symbols AAPL --timeframe both
```

Models are saved to `models/AAPL_daily_v1.pkl` + `models/AAPL_daily_scaler_v1.pkl`.

### 4. Backtest

```bash
python -m training.backtest --symbol AAPL --timeframe daily --years 3
```

Results are stored in DB and shown in the Training dashboard.

### 5. Run the pipeline

Once a model exists for a ticker, the Decision Agent automatically uses it instead of the LLM. No configuration needed.

If no model exists → falls back to Ollama LLM as before.

---

## Discord Integration

### Watchlist Alerts (webhook — no bot account needed)

1. In Discord: right-click a channel → **Edit Channel → Integrations → Webhooks → New Webhook** → copy the URL.
2. Open `ui/insider_profile.html` → paste the URL into the **Discord Alerts** panel in the sidebar → **Connect Discord**.
3. A test embed is sent immediately. From then on, the API polls your watchlist every 60 minutes and sends a colour-coded embed to that channel whenever a watched insider files a new open-market Form 4.

You can also hit **Scan now** in the sidebar to trigger an immediate check, or call `POST /discord/alert/watchlist` directly.

### LLM Chat Bot (`discord_bot.py`)

Lets you query your Ollama LLM about insider trades from inside Discord.

**Extra dependency:**

```bash
pip install discord.py
```

**Setup:**

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications) → New Application → Bot → copy the **Token**.
2. Enable **Message Content Intent** under Bot → Privileged Gateway Intents.
3. Invite the bot: `https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=2048&scope=bot`
4. Paste the token into the **Bot Token** field in the Discord Alerts sidebar (or set `DISCORD_BOT_TOKEN` env var).
5. Run alongside the API:

```bash
python discord_bot.py
```

**Commands:**

| Command | Description |
|---|---|
| `!insider <question>` | Ask the LLM about your most recently watched insider (uses their Form 4 history as context) |
| `!watchlist` | List your currently watched insiders |
| `!alert` | Manually trigger a watchlist scan right now |
| `!help` | Show available commands |

### Discord API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/discord/status` | Webhook connection status |
| `POST` | `/discord/settings` | Save webhook URL and optional bot token |
| `DELETE` | `/discord/settings` | Clear Discord settings |
| `POST` | `/discord/test` | Send a test embed to the configured webhook |
| `POST` | `/discord/alert/watchlist` | Trigger a watchlist scan immediately |

---

## API Reference

### Pipeline

| Method | Path | Description |
|---|---|---|
| `POST` | `/run` | Trigger a pipeline run for a symbol → returns `run_id` immediately |
| `GET` | `/status/{run_id}` | Poll run status and final state |
| `GET` | `/portfolio` | Live Alpaca positions and account metrics |

### Controls

| Method | Path | Description |
|---|---|---|
| `POST` | `/halt` | Engage kill-switch — all subsequent Decision nodes output HOLD |
| `POST` | `/resume` | Clear kill-switch |
| `POST` | `/risk` | Hot-update risk parameters (`max_position_pct`, `max_portfolio_risk_pct`, `default_order_type`) |

### Watchlist

| Method | Path | Description |
|---|---|---|
| `GET` | `/tickers` | List all tickers (watchlist) |
| `POST` | `/tickers` | Add a ticker `{"symbol": "AAPL", "name": "", "notes": ""}` |
| `DELETE` | `/tickers/{symbol}` | Soft-delete a ticker (history preserved) |

### History

| Method | Path | Description |
|---|---|---|
| `GET` | `/history/signals` | Signal history (`?symbol=AAPL&limit=100`) |
| `GET` | `/history/executions` | Execution history (`?symbol=AAPL&limit=100`) |
| `GET` | `/history/ohlcv/{symbol}` | Cached daily candles (`?days=60`) |

### Analytics

| Method | Path | Description |
|---|---|---|
| `GET` | `/analytics` | Full analytics bundle (`?period=daily\|weekly\|monthly`) |
| `GET` | `/analytics/summary` | Summary card only — lightweight polling |

### System

| Method | Path | Description |
|---|---|---|
| `WS` | `/ws/{run_id}` | Stream agent log events for a specific run |
| `WS` | `/ws/system` | Stream global events (kill-switch, risk changes) |
| `GET` | `/health` | Server health, model name, kill-switch state |

---

## Database Schema

Stored in **TimescaleDB** (PostgreSQL with time-series extensions).

| Table | Type | Description |
|---|---|---|
| `tickers` | Regular | Watchlist of symbols — symbol, name, active flag, notes |
| `ohlcv` | **Hypertable** | Daily OHLCV candles partitioned by timestamp — auto-deduplicated on upsert |
| `ohlcv_hourly` | **Hypertable** | Hourly OHLCV candles — up to 730 days per ticker |
| `news_sentiment` | Regular | Daily GDELT news sentiment score, article count, tone per ticker |
| `reddit_activity` | Regular | Daily Reddit mention count, avg score, avg sentiment per subreddit per ticker |
| `earnings_events` | Regular | Earnings dates, EPS estimate vs actual, surprise %, beat flag |
| `sec_filings` | Regular | 8-K, 10-Q, 10-K filing dates and URLs from SEC EDGAR |
| `model_registry` | Regular | Trained model metadata — accuracy, features, version, file path |
| `backtest_results` | Regular | Walk-forward backtest output — equity curve, trades, Sharpe, drawdown |
| `signals` | Regular | Every BUY/SELL/HOLD decision with confidence, rationale, risk flags |
| `executions` | Regular | Every order attempt — fill status, avg price, slippage, order ID |

OHLCV data is cached locally: only candles missing since the last stored date are fetched from yfinance on each run.

---

## Swapping the LLM

`agents/llm_router.py` is the single swap point. Change one function:

```python
# LM Studio (OpenAI-compatible local endpoint):
from langchain_openai import ChatOpenAI
return ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", model="local")

# Groq (fast remote inference, generous free tier):
from langchain_groq import ChatGroq
return ChatGroq(model="llama3-8b-8192", api_key="your_groq_key")

# Anthropic Claude (via API):
from langchain_anthropic import ChatAnthropic
return ChatAnthropic(model="claude-haiku-4-5-20251001")
```

Recommended local models (Ollama):

| Model | Size | Notes |
|---|---|---|
| `llama3.1:8b` | 4.7 GB | Best balance of quality and speed |
| `phi3:mini` | 2.3 GB | Good quality, lighter on RAM |
| `gemma2:2b` | 1.6 GB | Fastest, lowest quality |
| `mistral:7b` | 4.1 GB | Strong at structured JSON output |

---

## File Layout

```
hedge_bot/
├── agents/
│   ├── state.py                # Shared AgentState + Pydantic models
│   ├── llm_router.py           # Single LLM config/swap point
│   ├── observation_agent.py    # Data gathering — yfinance, OHLCV cache
│   ├── research_agent.py       # pandas-ta indicators + DB enrichment + LLM synthesis
│   ├── decision_agent.py       # ML model → LLM fallback + risk sizing
│   ├── implementation_agent.py # Alpaca order execution (no LLM)
│   └── graph.py                # LangGraph pipeline wiring + error routing
├── api/
│   ├── main.py                 # All FastAPI routes + WebSocket + analytics + training
│   ├── connections.py          # Connections API — credential storage + live testing
│   ├── discord_alerts.py       # Discord webhook sender + watchlist poller + /discord/* routes
│   └── ws_manager.py           # WebSocket fan-out connection manager
├── config/
│   ├── settings.py             # Pydantic settings — all config from .env
│   └── __init__.py
├── db/
│   ├── connection.py           # Async SQLAlchemy engine + init_db() (creates hypertables)
│   ├── models.py               # ORM: Ticker, OHLCV, Signal, Execution
│   ├── training_models.py      # ORM: OHLCVHourly, NewsSentiment, RedditActivity,
│   │                           #      EarningsEvent, SECFiling, ModelRegistry, BacktestResult
│   ├── queries.py              # CRUD operations
│   ├── analytics.py            # Performance analytics queries (P&L, win rate, Sharpe…)
│   └── __init__.py
├── training/
│   ├── backfill_ohlcv.py       # Download daily + hourly OHLCV into TimescaleDB
│   ├── backfill_news.py        # GDELT news sentiment via Google BigQuery
│   ├── backfill_reddit.py      # Reddit mentions via PRAW
│   ├── backfill_earnings.py    # Earnings history + SEC EDGAR filings
│   ├── features.py             # Feature engineering — joins all sources, generates labels
│   ├── train.py                # XGBoost training per ticker/timeframe + model registry
│   └── backtest.py             # Walk-forward backtest engine
├── models/                     # Saved model files (auto-created)
│   ├── AAPL_daily_v1.pkl
│   ├── AAPL_daily_scaler_v1.pkl
│   └── AAPL_daily_v1_meta.json
├── ui/
│   ├── dashboard.html          # Main command center — real-time pipeline view
│   ├── performance.html        # Performance analytics — daily/weekly/monthly
│   ├── connections.html        # External API connections management
│   ├── training.html           # ML training — backfill, train, backtest, model health
│   ├── insider_profile.html    # SEC Form 4 insider tracker + Discord alert setup
│   └── data_browser.html       # Raw data browser across all TimescaleDB tables
├── discord_bot.py              # Standalone Discord LLM chat bot (optional, run separately)
├── connections.json            # Saved API credentials (auto-created, gitignored)
├── .env.example                # Environment variable template
├── docker-compose.yml          # TimescaleDB + Ollama + API
├── Dockerfile
└── requirements.txt
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model name to use |
| `DATABASE_URL` | `postgresql://hedge:hedge@localhost:5432/hedgebot` | TimescaleDB connection string |
| `ALPACA_API_KEY` | — | Alpaca API key (required for execution) |
| `ALPACA_SECRET_KEY` | — | Alpaca secret key |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | Use paper endpoint until ready for live |
| `NEWS_API_KEY` | — | NewsAPI key (optional — sentiment will be 0 if missing) |
| `MAX_POSITION_PCT` | `0.05` | Max 5% of portfolio per trade |
| `MAX_PORTFOLIO_RISK_PCT` | `0.20` | Halt if drawdown exceeds 20% |
| `DEFAULT_ORDER_TYPE` | `limit` | `limit` or `market` |
