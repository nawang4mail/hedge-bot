"""
Observation Agent — Data Gatherer
==================================
Ingests real-time market data, order book, and news sentiment.
Produces a clean MarketSnapshot JSON for the Research Agent.

DB integration: OHLCV candles are cached in TimescaleDB.
On each run we fetch only the candles missing since the last stored date,
then read the full history from the DB — avoiding redundant downloads.

IMPORTANT: This agent does NOT call the LLM.
"""
from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any

import yfinance as yf
import pandas as pd

from agents.state import AgentState, MarketSnapshot
from config import settings

# Optional: NewsAPI for sentiment
try:
    from newsapi import NewsApiClient
    _newsapi = NewsApiClient(api_key=settings.news_api_key) if settings.news_api_key else None
except ImportError:
    _newsapi = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_ohlcv(symbol: str, period: str = "60d", interval: str = "1d") -> list[dict]:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return []
    df = df.reset_index()
    df["Date"] = df["Date"].astype(str)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].round(4).to_dict("records")


def _fetch_ohlcv_since(symbol: str, since: datetime) -> list[dict]:
    """Fetch only candles newer than `since` date — minimises yfinance calls."""
    ticker = yf.Ticker(symbol)
    start  = (since + timedelta(days=1)).strftime("%Y-%m-%d")
    df     = ticker.history(start=start, interval="1d")
    if df.empty:
        return []
    df = df.reset_index()
    df["Date"] = df["Date"].astype(str)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].round(4).to_dict("records")


def _fetch_quote(symbol: str) -> dict[str, float]:
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    return {
        "price": round(float(info.last_price or 0), 4),
        "volume": round(float(info.three_month_average_volume or 0), 0),
        "bid": round(float(getattr(info, "bid", 0) or 0), 4),
        "ask": round(float(getattr(info, "ask", 0) or 0), 4),
    }


def _fetch_news_sentiment(symbol: str) -> tuple[float, list[str]]:
    """
    Returns (sentiment_score, headlines).
    Sentiment is a naive average: positive title keywords → +1, negative → −1.
    Replace with a proper NLP model (e.g., FinBERT) for production.
    """
    if _newsapi is None:
        return 0.0, []

    try:
        resp = _newsapi.get_everything(
            q=symbol, language="en", sort_by="publishedAt", page_size=10
        )
        headlines = [a["title"] for a in resp.get("articles", [])]
    except Exception:
        return 0.0, []

    POSITIVE = {"surge", "rally", "beat", "gain", "rise", "soar", "upgrade", "bull"}
    NEGATIVE = {"fall", "drop", "miss", "loss", "crash", "downgrade", "bear", "cut"}

    scores = []
    for h in headlines:
        words = set(h.lower().split())
        score = len(words & POSITIVE) - len(words & NEGATIVE)
        scores.append(max(-1.0, min(1.0, float(score))))

    sentiment = round(sum(scores) / len(scores), 4) if scores else 0.0
    return sentiment, headlines


# ── Agent node ────────────────────────────────────────────────────────────────

def observation_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node.  Receives state dict, returns partial update dict.
    """
    logs = list(state.get("agent_logs") or [])
    s = AgentState(**state)
    symbol = s.symbol
    logs = list(s.agent_logs)

    logs.append({
        "agent": "observation",
        "status": "processing",
        "ts": datetime.now(timezone.utc).isoformat(),
        "msg": f"Fetching market data for {symbol}",
    })

    try:
        quote = _fetch_quote(symbol)
        bid, ask = quote["bid"], quote["ask"]
        spread = round(ask - bid, 4) if ask and bid else 0.0

        # ── DB-cached OHLCV: fetch only what's missing ────────────────────
        from db.connection import AsyncSessionLocal
        from db import queries

        async def _sync_candles():
            async with AsyncSessionLocal() as db:
                latest = await queries.latest_candle_date(db, symbol)
                if latest:
                    new_candles = _fetch_ohlcv_since(symbol, latest)
                else:
                    new_candles = _fetch_ohlcv(symbol, period="200d")
                if new_candles:
                    await queries.upsert_candles(db, symbol, new_candles)
                    await db.commit()
                return await queries.get_candles(db, symbol, days=200)

        ohlcv = asyncio.run(_sync_candles())

        sentiment, headlines = _fetch_news_sentiment(symbol)

        snapshot = MarketSnapshot(
            symbol=symbol,
            price=quote["price"],
            volume=quote["volume"],
            bid=bid,
            ask=ask,
            spread=spread,
            news_sentiment=sentiment,
            news_headlines=headlines[:5],   # cap for token efficiency
            ohlcv_1d=ohlcv[-60:],           # last 60 candles
        )

        logs.append({
            "agent": "observation",
            "status": "completed",
            "ts": datetime.now(timezone.utc).isoformat(),
            "output": json.loads(snapshot.model_dump_json()),
        })

        return {
            **state,
            "market_snapshot": snapshot,
            "agent_logs": logs,
        }

    except Exception as exc:
        logs.append({"agent": "observation", "status": "error", "msg": str(exc)})
        return {**state, "error": str(exc), "agent_logs": logs}
