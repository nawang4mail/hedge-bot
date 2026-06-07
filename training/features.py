"""
Feature Engineering — joins all data sources into a flat training dataset.

For each ticker and timeframe (daily / hourly), produces a DataFrame where:
  - Each row = one candle
  - Columns = technical indicators + sentiment + Reddit + earnings flags
  - Target column = forward return classification (BUY / SELL / HOLD)

Label logic:
  Look forward N candles. If return > threshold → BUY, < -threshold → SELL, else HOLD.
  Default: 5 candles forward, ±2% threshold (configurable).
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import pandas_ta as ta
from sqlalchemy import select, text

from db.connection import AsyncSessionLocal
from db.models import OHLCV
from db.training_models import OHLCVHourly, NewsSentiment, RedditActivity, EarningsEvent

Timeframe = Literal["daily", "hourly"]


# ── OHLCV loader ──────────────────────────────────────────────────────────────

async def load_ohlcv(symbol: str, timeframe: Timeframe, years: int = 5) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=365 * years)
    model = OHLCV if timeframe == "daily" else OHLCVHourly

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(model)
            .where(model.symbol == symbol.upper(), model.ts >= since)
            .order_by(model.ts)
        )
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([r.to_dict() for r in rows])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


# ── Sentiment loaders ─────────────────────────────────────────────────────────

async def load_news_sentiment(symbol: str, years: int = 5) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=365 * years)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(NewsSentiment)
            .where(NewsSentiment.symbol == symbol.upper(), NewsSentiment.date >= since)
            .order_by(NewsSentiment.date)
        )
        rows = result.scalars().all()
    if not rows:
        return pd.DataFrame(columns=["date", "news_sentiment", "news_article_count", "news_tone"])
    df = pd.DataFrame([{
        "date":               r.date.date(),
        "news_sentiment":     r.sentiment or 0.0,
        "news_article_count": r.article_count or 0,
        "news_tone":          r.avg_tone or 0.0,
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")


async def load_reddit(symbol: str, years: int = 5) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=365 * years)
    async with AsyncSessionLocal() as db:
        # Aggregate across subreddits
        result = await db.execute(text("""
            SELECT
                date_trunc('day', date) as day,
                SUM(mention_count) as reddit_mentions,
                AVG(avg_sentiment) as reddit_sentiment,
                AVG(avg_score) as reddit_avg_score
            FROM reddit_activity
            WHERE symbol = :sym AND date >= :since
            GROUP BY day ORDER BY day
        """), {"sym": symbol.upper(), "since": since})
        rows = result.fetchall()
    if not rows:
        return pd.DataFrame(columns=["reddit_mentions", "reddit_sentiment", "reddit_avg_score"])
    df = pd.DataFrame([{
        "date":              r.day,
        "reddit_mentions":   int(r.reddit_mentions or 0),
        "reddit_sentiment":  float(r.reddit_sentiment or 0),
        "reddit_avg_score":  float(r.reddit_avg_score or 0),
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")


async def load_insider_features(symbol: str, years: int = 5) -> pd.DataFrame:
    """
    Compute daily insider trading signals:
      insider_buy_30d      — open-market buy count in rolling 30-day window
      insider_sell_30d     — open-market sell count in rolling 30-day window
      insider_net_shares_30d — net shares (buys - sells) rolling 30 days
      insider_buy_value_30d  — total $ value of open-market buys, 30-day window
    """
    since = datetime.now(timezone.utc) - timedelta(days=365 * years)
    async with AsyncSessionLocal() as db:
        result = await db.execute(text("""
            SELECT
                date_trunc('day', transaction_date) as day,
                SUM(CASE WHEN transaction_code = 'P' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN transaction_code = 'S' THEN 1 ELSE 0 END) as sell_count,
                SUM(CASE WHEN transaction_code = 'P' THEN COALESCE(shares, 0) ELSE 0 END) as buy_shares,
                SUM(CASE WHEN transaction_code = 'S' THEN COALESCE(shares, 0) ELSE 0 END) as sell_shares,
                SUM(CASE WHEN transaction_code = 'P' THEN COALESCE(total_value, 0) ELSE 0 END) as buy_value
            FROM sec_insider_trades
            WHERE symbol = :sym
              AND transaction_date >= :since
              AND is_open_market = TRUE
            GROUP BY day
            ORDER BY day
        """), {"sym": symbol.upper(), "since": since})
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame(columns=["insider_buy_30d","insider_sell_30d",
                                      "insider_net_shares_30d","insider_buy_value_30d"])

    df = pd.DataFrame([{
        "date":       r.day,
        "_buy_count": int(r.buy_count or 0),
        "_sell_count":int(r.sell_count or 0),
        "_buy_shares": float(r.buy_shares or 0),
        "_sell_shares":float(r.sell_shares or 0),
        "_buy_value":  float(r.buy_value or 0),
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")

    # Rolling 30-day sums
    df["insider_buy_30d"]       = df["_buy_count"].rolling("30D").sum()
    df["insider_sell_30d"]      = df["_sell_count"].rolling("30D").sum()
    df["insider_net_shares_30d"]= (df["_buy_shares"] - df["_sell_shares"]).rolling("30D").sum()
    df["insider_buy_value_30d"] = df["_buy_value"].rolling("30D").sum()

    return df[["insider_buy_30d","insider_sell_30d",
               "insider_net_shares_30d","insider_buy_value_30d"]]


async def load_earnings_flags(symbol: str, years: int = 5) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=365 * years)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(EarningsEvent)
            .where(EarningsEvent.symbol == symbol.upper(), EarningsEvent.report_date >= since)
        )
        rows = result.scalars().all()
    if not rows:
        return pd.DataFrame(columns=["earnings_beat", "eps_surprise_pct"])
    df = pd.DataFrame([{
        "date":             r.report_date.date(),
        "earnings_beat":    1 if r.beat_estimate else 0,
        "eps_surprise_pct": float(r.eps_surprise_pct or 0),
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")


# ── Feature computation ───────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    df["rsi_14"]         = ta.rsi(close, length=14)
    df["rsi_7"]          = ta.rsi(close, length=7)

    macd = ta.macd(close)
    if macd is not None:
        df["macd"]       = macd["MACD_12_26_9"]
        df["macd_signal"]= macd["MACDs_12_26_9"]
        df["macd_hist"]  = macd["MACDh_12_26_9"]

    df["sma_20"]  = ta.sma(close, length=20)
    df["sma_50"]  = ta.sma(close, length=50)
    df["sma_200"] = ta.sma(close, length=200)
    df["ema_12"]  = ta.ema(close, length=12)
    df["ema_26"]  = ta.ema(close, length=26)

    df["atr_14"]  = ta.atr(high, low, close, length=14)
    df["adx_14"]  = ta.adx(high, low, close, length=14)["ADX_14"]

    bb = ta.bbands(close, length=20)
    if bb is not None:
        bbu = next((c for c in bb.columns if c.startswith("BBU_")), None)
        bbl = next((c for c in bb.columns if c.startswith("BBL_")), None)
        bbm = next((c for c in bb.columns if c.startswith("BBM_")), None)
        if bbu and bbl and bbm:
            df["bb_upper"] = bb[bbu]
            df["bb_lower"] = bb[bbl]
            df["bb_mid"]   = bb[bbm]
            df["bb_width"] = (bb[bbu] - bb[bbl]) / bb[bbm]
            df["bb_pct"]   = (close - bb[bbl]) / (bb[bbu] - bb[bbl])

    stoch = ta.stoch(high, low, close)
    if stoch is not None:
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]

    # Price-derived features
    df["returns_1d"]  = close.pct_change(1)
    df["returns_5d"]  = close.pct_change(5)
    df["returns_20d"] = close.pct_change(20)
    df["volatility_20"] = df["returns_1d"].rolling(20).std()

    # Volume features
    df["volume_sma_20"] = volume.rolling(20).mean()
    df["volume_ratio"]  = volume / df["volume_sma_20"]

    # Price position features
    df["close_vs_sma50"]  = (close - df["sma_50"])  / df["sma_50"]
    df["close_vs_sma200"] = (close - df["sma_200"]) / df["sma_200"]
    df["close_vs_bb_mid"] = (close - df["bb_mid"])  / df["bb_mid"]

    # Earnings proximity flag (within 5 candles of earnings)
    if "earnings_beat" not in df.columns:
        df["earnings_week"] = 0
        df["earnings_beat"] = 0
        df["eps_surprise_pct"] = 0.0

    return df


def generate_labels(df: pd.DataFrame, forward: int = 5, threshold: float = 0.02) -> pd.Series:
    """
    Forward return over `forward` candles.
    BUY=2, SELL=0, HOLD=1
    """
    fwd_return = df["close"].shift(-forward) / df["close"] - 1
    labels = pd.Series(1, index=df.index, name="label")   # default HOLD
    labels[fwd_return >  threshold] = 2   # BUY
    labels[fwd_return < -threshold] = 0   # SELL
    return labels


# ── Main feature builder ──────────────────────────────────────────────────────

async def build_features(
    symbol: str,
    timeframe: Timeframe = "daily",
    years: int = 5,
    forward_candles: int = 5,
    label_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Returns a fully-featured DataFrame ready for model training.
    Drops rows with NaN values (caused by indicator lookback periods).
    """
    print(f"  Loading OHLCV ({timeframe})...")
    df = await load_ohlcv(symbol, timeframe, years)
    if df.empty:
        raise ValueError(f"No {timeframe} OHLCV data for {symbol}. Run backfill first.")

    print(f"  Computing indicators...")
    df = compute_indicators(df)

    # Merge daily sentiment data (both daily and hourly models use daily sentiment)
    print("  Loading sentiment data...")
    news_df    = await load_news_sentiment(symbol, years)
    reddit_df  = await load_reddit(symbol, years)
    earn_df    = await load_earnings_flags(symbol, years)
    insider_df = await load_insider_features(symbol, years)

    if timeframe == "daily":
        merge_idx = df.index.normalize()
    else:
        merge_idx = df.index.normalize()   # hourly → merge on date

    df_date = df.copy()
    df_date.index = merge_idx

    for sentiment_df, cols in [
        (news_df,    ["news_sentiment", "news_article_count", "news_tone"]),
        (reddit_df,  ["reddit_mentions", "reddit_sentiment", "reddit_avg_score"]),
        (earn_df,    ["earnings_beat", "eps_surprise_pct"]),
        (insider_df, ["insider_buy_30d", "insider_sell_30d",
                       "insider_net_shares_30d", "insider_buy_value_30d"]),
    ]:
        if not sentiment_df.empty:
            df_date = df_date.join(sentiment_df[cols], how="left")

    # Forward-fill sentiment (weekend news affects Monday trading)
    sentiment_cols = ["news_sentiment", "news_article_count", "news_tone",
                      "reddit_mentions", "reddit_sentiment", "reddit_avg_score",
                      "earnings_beat", "eps_surprise_pct"]
    for col in sentiment_cols:
        if col not in df_date.columns:
            df_date[col] = 0.0
    df_date[sentiment_cols] = df_date[sentiment_cols].fillna(method="ffill").fillna(0)

    # Earnings proximity flag
    df_date["earnings_week"] = (
        df_date["earnings_beat"].rolling(5, center=True).max().fillna(0).astype(int)
    )

    # Restore original index
    df_date.index = df.index

    # Generate labels
    print("  Generating labels...")
    df_date["label"] = generate_labels(df_date, forward_candles, label_threshold)

    # Drop NaN rows (from indicator lookback + forward label)
    df_date = df_date.dropna(subset=["rsi_14", "macd", "sma_200", "label"])

    print(f"  Final dataset: {len(df_date)} rows, {df_date.columns.tolist().count} features")
    print(f"  Label distribution: {df_date['label'].value_counts().to_dict()}")
    return df_date
