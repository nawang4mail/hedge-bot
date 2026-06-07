"""
Research Agent — Quantitative Analyst
=======================================
Receives the MarketSnapshot, runs pandas-ta indicators, then asks the local
LLM to synthesize a token-efficient analyst summary.

The LLM sees ONLY the computed indicator values, never raw OHLCV data, to
minimise context size and hallucination surface area.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pandas_ta as ta
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState, ResearchReport, MarketSnapshot
from agents.llm_router import get_llm

# ── System prompt (stored here, close to the agent that owns it) ──────────────

RESEARCH_SYSTEM_PROMPT = """You are a senior quantitative analyst at a hedge fund.
You receive pre-computed technical indicator values for a single equity symbol.
Your job is to synthesise these indicators into a concise trading-relevant summary.

RULES:
- Output ONLY valid JSON — no prose outside the JSON object.
- The JSON must have exactly two keys:
    "trend"   : one of "uptrend" | "downtrend" | "sideways"
    "summary" : a single sentence (max 40 words) explaining the dominant signal.
- Do NOT invent numbers, prices, or data not present in the input.
- Do NOT issue trading recommendations — that is the Decision Agent's role.
- If the data is insufficient, set trend to "sideways" and summarise accordingly."""


# ── Indicator computation (pure Python / pandas-ta, no LLM) ──────────────────

def _compute_indicators(snapshot: MarketSnapshot) -> dict[str, Any]:
    if not snapshot.ohlcv_1d:
        return {}

    df = pd.DataFrame(snapshot.ohlcv_1d)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "datetime"})

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    result: dict[str, Any] = {}

    # RSI
    rsi = ta.rsi(close, length=14)
    result["rsi_14"] = round(float(rsi.iloc[-1]), 2) if rsi is not None and not rsi.empty else None

    # MACD
    macd_df = ta.macd(close)
    if macd_df is not None and not macd_df.empty:
        result["macd_signal"] = round(float(macd_df["MACDs_12_26_9"].iloc[-1]), 4)
        result["macd_hist"]   = round(float(macd_df["MACDh_12_26_9"].iloc[-1]), 4)

    # SMAs
    sma50  = ta.sma(close, length=50)
    sma200 = ta.sma(close, length=200)
    result["sma_50"]  = round(float(sma50.iloc[-1]),  2) if sma50  is not None else None
    result["sma_200"] = round(float(sma200.iloc[-1]), 2) if sma200 is not None else None

    # ATR (volatility for position sizing)
    atr = ta.atr(high, low, close, length=14)
    result["atr_14"] = round(float(atr.iloc[-1]), 4) if atr is not None else None

    # Bollinger Bands
    bb = ta.bbands(close, length=20)
    if bb is not None and not bb.empty:
        result["bollinger_upper"] = round(float(bb["BBU_20_2.0"].iloc[-1]), 4)
        result["bollinger_lower"] = round(float(bb["BBL_20_2.0"].iloc[-1]), 4)

    # Volume spike: today's volume > 2× 20-day average
    avg_vol = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    result["volume_spike"] = bool(float(vol.iloc[-1]) > 2 * avg_vol)

    return result


def _detect_anomalies(indicators: dict, snapshot: MarketSnapshot) -> list[str]:
    anomalies = []
    rsi = indicators.get("rsi_14")
    if rsi is not None:
        if rsi > 70:
            anomalies.append(f"RSI overbought ({rsi})")
        elif rsi < 30:
            anomalies.append(f"RSI oversold ({rsi})")
    if indicators.get("volume_spike"):
        anomalies.append("Volume spike detected (>2× 20-day avg)")
    if snapshot.news_sentiment < -0.4:
        anomalies.append(f"Strong negative news sentiment ({snapshot.news_sentiment})")
    elif snapshot.news_sentiment > 0.4:
        anomalies.append(f"Strong positive news sentiment ({snapshot.news_sentiment})")
    return anomalies


# ── Agent node ────────────────────────────────────────────────────────────────

def research_node(state: dict[str, Any]) -> dict[str, Any]:
    logs = list(state.get("agent_logs") or [])
    s = AgentState(**state)
    logs = list(s.agent_logs)

    logs.append({
        "agent": "research",
        "status": "processing",
        "ts": datetime.now(timezone.utc).isoformat(),
        "msg": "Computing technical indicators",
    })

    try:
        snapshot: MarketSnapshot = s.market_snapshot
        indicators = _compute_indicators(snapshot)
        anomalies  = _detect_anomalies(indicators, snapshot)

        # ── Pull enriched sentiment features from DB ──────────────────────
        from db.connection import AsyncSessionLocal
        from db.training_models import NewsSentiment, RedditActivity, EarningsEvent
        from sqlalchemy import select as sa_select
        from datetime import timedelta

        enrichment = {}
        try:
            async def _load_enrichment():
                since = datetime.now(timezone.utc) - timedelta(days=7)
                async with AsyncSessionLocal() as db:
                    # Latest news sentiment
                    ns = await db.execute(
                        sa_select(NewsSentiment)
                        .where(NewsSentiment.symbol == snapshot.symbol,
                               NewsSentiment.date >= since)
                        .order_by(NewsSentiment.date.desc()).limit(1)
                    )
                    ns_row = ns.scalar_one_or_none()

                    # Latest Reddit activity
                    ra = await db.execute(
                        sa_select(RedditActivity)
                        .where(RedditActivity.symbol == snapshot.symbol,
                               RedditActivity.date >= since)
                        .order_by(RedditActivity.date.desc()).limit(1)
                    )
                    ra_row = ra.scalar_one_or_none()

                    # Upcoming earnings (within 14 days)
                    earn_soon = await db.execute(
                        sa_select(EarningsEvent)
                        .where(EarningsEvent.symbol == snapshot.symbol,
                               EarningsEvent.report_date >= datetime.now(timezone.utc),
                               EarningsEvent.report_date <= datetime.now(timezone.utc) + timedelta(days=14))
                        .order_by(EarningsEvent.report_date).limit(1)
                    )
                    earn_row = earn_soon.scalar_one_or_none()

                    return {
                        "db_news_sentiment":    ns_row.sentiment if ns_row else None,
                        "db_news_article_count": ns_row.article_count if ns_row else 0,
                        "db_reddit_mentions":   ra_row.mention_count if ra_row else 0,
                        "db_reddit_sentiment":  ra_row.avg_sentiment if ra_row else None,
                        "earnings_within_14d":  earn_row is not None,
                        "next_earnings_date":   earn_row.report_date.isoformat() if earn_row else None,
                    }

            enrichment = asyncio.run(_load_enrichment())
        except Exception:
            pass   # enrichment is optional — don't block the pipeline

        # ── LLM call: synthesis only, on indicator values ─────────────────
        llm = get_llm(temperature=0.1)
        user_payload = {
            "symbol": snapshot.symbol,
            "current_price": snapshot.price,
            "news_sentiment": snapshot.news_sentiment,
            **indicators,
            "anomalies": anomalies,
            **{k: v for k, v in enrichment.items() if v is not None},
        }
        response = llm.invoke([
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(user_payload, default=str)),
        ])

        llm_json = json.loads(response.content)

        report = ResearchReport(
            symbol=snapshot.symbol,
            anomalies=anomalies,
            trend=llm_json.get("trend", "sideways"),
            analyst_summary=llm_json.get("summary", ""),
            **{k: v for k, v in indicators.items() if k in ResearchReport.model_fields},
        )

        logs.append({
            "agent": "research",
            "status": "completed",
            "ts": datetime.now(timezone.utc).isoformat(),
            "output": json.loads(report.model_dump_json()),
        })

        return {**state, "research_report": report, "agent_logs": logs}

    except Exception as exc:
        logs.append({"agent": "research", "status": "error", "msg": str(exc)})
        return {**state, "error": str(exc), "agent_logs": logs}
