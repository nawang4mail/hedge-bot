"""
All database read/write operations in one place.
Agents and API routes import from here — never write raw SQL elsewhere.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, delete, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Ticker, OHLCV, Signal, Execution


# ── Tickers (watchlist) ───────────────────────────────────────────────────────

async def get_watchlist(db: AsyncSession, active_only: bool = True) -> list[Ticker]:
    stmt = select(Ticker)
    if active_only:
        stmt = stmt.where(Ticker.active == True)
    result = await db.execute(stmt.order_by(Ticker.symbol))
    return result.scalars().all()


async def add_ticker(db: AsyncSession, symbol: str, name: str = "", notes: str = "") -> Ticker:
    ticker = Ticker(symbol=symbol.upper(), name=name, notes=notes, active=True)
    db.add(ticker)
    await db.flush()
    return ticker


async def remove_ticker(db: AsyncSession, symbol: str) -> bool:
    result = await db.execute(select(Ticker).where(Ticker.symbol == symbol.upper()))
    ticker = result.scalar_one_or_none()
    if ticker:
        ticker.active = False   # soft delete — keep history
        return True
    return False


async def ticker_exists(db: AsyncSession, symbol: str) -> bool:
    result = await db.execute(
        select(Ticker.id).where(Ticker.symbol == symbol.upper())
    )
    return result.scalar_one_or_none() is not None


# ── OHLCV ─────────────────────────────────────────────────────────────────────

async def upsert_candles(db: AsyncSession, symbol: str, candles: list[dict]) -> int:
    """
    Insert candles, silently skipping duplicates (upsert on symbol+ts).
    Returns number of rows inserted.
    """
    if not candles:
        return 0

    rows = []
    for c in candles:
        ts = c.get("Date") or c.get("ts") or c.get("datetime")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        rows.append({
            "symbol": symbol.upper(),
            "ts":     ts,
            "open":   float(c.get("Open",  c.get("open",  0))),
            "high":   float(c.get("High",  c.get("high",  0))),
            "low":    float(c.get("Low",   c.get("low",   0))),
            "close":  float(c.get("Close", c.get("close", 0))),
            "volume": float(c.get("Volume",c.get("volume",0))),
        })

    stmt = pg_insert(OHLCV).values(rows).on_conflict_do_nothing(
        index_elements=["symbol", "ts"]
    )
    result = await db.execute(stmt)
    return result.rowcount


async def get_candles(
    db: AsyncSession,
    symbol: str,
    days: int = 200,
) -> list[dict]:
    """Return up to `days` recent daily candles for a symbol."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        select(OHLCV)
        .where(OHLCV.symbol == symbol.upper(), OHLCV.ts >= since)
        .order_by(OHLCV.ts)
    )
    return [r.to_dict() for r in result.scalars().all()]


async def latest_candle_date(db: AsyncSession, symbol: str) -> Optional[datetime]:
    """Returns the timestamp of the most recent stored candle, or None."""
    result = await db.execute(
        select(OHLCV.ts)
        .where(OHLCV.symbol == symbol.upper())
        .order_by(OHLCV.ts.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ── Signals ───────────────────────────────────────────────────────────────────

async def save_signal(db: AsyncSession, run_id: str, signal) -> Signal:
    """Persist a TradingSignal pydantic model to the DB."""
    row = Signal(
        run_id=run_id,
        symbol=signal.symbol,
        action=signal.action,
        quantity=signal.quantity,
        order_type=signal.order_type,
        limit_price=signal.limit_price,
        confidence=signal.confidence,
        rationale=signal.rationale,
        risk_checks_passed=signal.risk_checks_passed,
    )
    db.add(row)
    await db.flush()
    return row


async def get_signal_history(
    db: AsyncSession,
    symbol: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    stmt = select(Signal).order_by(Signal.created_at.desc()).limit(limit)
    if symbol:
        stmt = stmt.where(Signal.symbol == symbol.upper())
    result = await db.execute(stmt)
    return [r.to_dict() for r in result.scalars().all()]


# ── Executions ────────────────────────────────────────────────────────────────

async def save_execution(db: AsyncSession, run_id: str, symbol: str, execution) -> Execution:
    """Persist an ExecutionReport pydantic model to the DB."""
    row = Execution(
        run_id=run_id,
        symbol=symbol,
        order_id=execution.order_id,
        status=execution.status,
        filled_qty=execution.filled_qty,
        avg_fill_price=execution.avg_fill_price,
        slippage_pct=execution.slippage_pct,
        message=execution.message,
    )
    db.add(row)
    await db.flush()
    return row


async def get_execution_history(
    db: AsyncSession,
    symbol: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    stmt = select(Execution).order_by(Execution.created_at.desc()).limit(limit)
    if symbol:
        stmt = stmt.where(Execution.symbol == symbol.upper())
    result = await db.execute(stmt)
    return [r.to_dict() for r in result.scalars().all()]
