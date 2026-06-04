"""
OHLCV Backfill — daily + hourly candles via yfinance.

Usage (CLI):
  python -m training.backfill_ohlcv --symbols AAPL TSLA NVDA --years 5
  python -m training.backfill_ohlcv --symbols AAPL --hourly --days 730
"""
from __future__ import annotations
import argparse
import asyncio
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import text

from db.connection import AsyncSessionLocal, init_db
from db.models import OHLCV
from db.training_models import OHLCVHourly
from training.progress import ProgressEmitter


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_daily(symbol: str, start: str, end: str) -> list[dict]:
    df = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
    if df.empty:
        return []
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return [
        {
            "symbol": symbol.upper(), "ts": row["Date"].to_pydatetime(),
            "open":   round(float(row["Open"]),   4),
            "high":   round(float(row["High"]),   4),
            "low":    round(float(row["Low"]),    4),
            "close":  round(float(row["Close"]),  4),
            "volume": round(float(row["Volume"]), 0),
        }
        for _, row in df.iterrows()
    ]


def fetch_hourly_chunked(symbol: str, start: str, end: str,
                          on_chunk: callable | None = None) -> list[dict]:
    """Fetch hourly data in 59-day chunks, calling on_chunk(fetched_so_far, total_days) each chunk."""
    all_rows = []
    s        = datetime.fromisoformat(start)
    e        = datetime.fromisoformat(end)
    chunk    = timedelta(days=59)
    total_days = (e - s).days
    days_done  = 0

    while s < e:
        chunk_end = min(s + chunk, e)
        df = yf.Ticker(symbol).history(
            start=s.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval="1h",
        )
        if not df.empty:
            df = df.reset_index()
            df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
            for _, row in df.iterrows():
                all_rows.append({
                    "symbol": symbol.upper(), "ts": row["Datetime"].to_pydatetime(),
                    "open":   round(float(row["Open"]),   4),
                    "high":   round(float(row["High"]),   4),
                    "low":    round(float(row["Low"]),    4),
                    "close":  round(float(row["Close"]),  4),
                    "volume": round(float(row["Volume"]), 0),
                })
        days_done += (chunk_end - s).days
        if on_chunk:
            on_chunk(days_done, total_days)
        s = chunk_end
    return all_rows


# ── DB upsert ─────────────────────────────────────────────────────────────────

async def upsert_rows(rows: list[dict], model, constraint: str) -> int:
    if not rows:
        return 0
    async with AsyncSessionLocal() as db:
        stmt   = pg_insert(model).values(rows).on_conflict_do_nothing(constraint=constraint)
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount


# ── Main ──────────────────────────────────────────────────────────────────────

async def backfill(
    symbols: list[str],
    years: int = 5,
    hourly: bool = False,
    hourly_days: int = 730,
    emitter: ProgressEmitter | None = None,
):
    await init_db()

    end          = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_daily  = (datetime.now(timezone.utc) - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    start_hourly = (datetime.now(timezone.utc) - timedelta(days=hourly_days)).strftime("%Y-%m-%d")
    total_daily_days = 365 * years

    sources = ["daily"] + (["hourly"] if hourly else [])
    if emitter:
        await emitter.phase_start("OHLCV Backfill",
                                   total_tickers=len(symbols), sources=sources)

    for symbol in symbols:
        # Ensure ticker exists in watchlist
        async with AsyncSessionLocal() as db:
            await db.execute(text("""
                INSERT INTO tickers (symbol, active, created_at, updated_at)
                VALUES (:sym, true, now(), now())
                ON CONFLICT (symbol) DO NOTHING
            """), {"sym": symbol.upper()})
            await db.commit()

        # ── Daily ─────────────────────────────────────────────────────────
        if emitter:
            await emitter.ticker_start(symbol, total=total_daily_days,
                                        unit="days", source="yfinance/daily")
        t0   = time.monotonic()
        rows = fetch_daily(symbol, start_daily, end)
        n    = await upsert_rows(rows, OHLCV, "uq_ohlcv_symbol_ts")
        if emitter:
            await emitter.ticker_progress(symbol, current=total_daily_days,
                                           total=total_daily_days, unit="days",
                                           detail=f"{len(rows)} candles fetched")
            await emitter.ticker_done(symbol, rows_inserted=n,
                                       rows_total=len(rows), source="daily")
            await emitter.emit("row_insert", symbol=symbol, table="ohlcv",
                               count=n, cumulative=n)
        else:
            print(f"  {symbol} daily: {len(rows)} fetched, {n} new")

        # ── Hourly ────────────────────────────────────────────────────────
        if hourly:
            if emitter:
                await emitter.ticker_start(symbol, total=hourly_days,
                                            unit="days", source="yfinance/hourly")

            chunk_progress = {"days": 0}

            def _on_chunk(done, total):
                chunk_progress["days"] = done
                # Schedule emit on event loop (we're in a sync callback)
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(
                            emitter.ticker_progress(
                                symbol, current=done, total=total, unit="days",
                                detail=f"chunk {done}/{total} days fetched"
                            )
                        )
                    )
                except Exception:
                    pass

            hourly_rows = fetch_hourly_chunked(
                symbol, start_hourly, end,
                on_chunk=_on_chunk if emitter else None
            )
            nh = await upsert_rows(hourly_rows, OHLCVHourly, "uq_ohlcv_hourly_symbol_ts")
            if emitter:
                await emitter.ticker_done(symbol, rows_inserted=nh,
                                           rows_total=len(hourly_rows), source="hourly")
                await emitter.emit("row_insert", symbol=symbol, table="ohlcv_hourly",
                                   count=nh, cumulative=nh)
            else:
                print(f"  {symbol} hourly: {len(hourly_rows)} fetched, {nh} new")

    if emitter:
        await emitter.phase_done("OHLCV Backfill")
    else:
        print("\n✅ OHLCV backfill complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",     nargs="+", required=True)
    parser.add_argument("--years",       type=int,  default=5)
    parser.add_argument("--hourly",      action="store_true")
    parser.add_argument("--hourly-days", type=int,  default=730)
    args = parser.parse_args()
    asyncio.run(backfill(args.symbols, args.years, args.hourly, args.hourly_days))
