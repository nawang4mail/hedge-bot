"""
Performance analytics queries.

All queries use raw SQL via SQLAlchemy text() for clarity and TimescaleDB
time_bucket() support.  Results are returned as plain dicts ready for JSON.

Metrics computed:
  - Trade counts (total, BUY, SELL, HOLD) per period per symbol
  - Realised P&L per trade (fill_price vs prior close from OHLCV)
  - Total P&L and cumulative equity curve per period
  - Win rate, average confidence, average slippage
  - Best / worst single trade
  - Most active symbols
  - Streak: current consecutive winning / losing trades
"""
from __future__ import annotations
from typing import Literal
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

Period = Literal["daily", "weekly", "monthly"]

# Map period → PostgreSQL interval for time_bucket
_BUCKET = {
    "daily":   "1 day",
    "weekly":  "1 week",
    "monthly": "1 month",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _since(period: Period) -> str:
    """Return a SQL interval string that covers enough history for the period."""
    return {"daily": "30 days", "weekly": "12 weeks", "monthly": "12 months"}[period]


# ── Trade summary per bucket ──────────────────────────────────────────────────

async def trade_counts_by_period(db: AsyncSession, period: Period) -> list[dict]:
    """
    Number of BUY / SELL / HOLD signals per time bucket.
    Returns rows ordered by bucket descending.
    """
    bucket = _BUCKET[period]
    since  = _since(period)
    rows = await db.execute(text(f"""
        SELECT
            time_bucket('{bucket}', created_at)  AS bucket,
            COUNT(*)                              AS total,
            COUNT(*) FILTER (WHERE action='BUY')  AS buys,
            COUNT(*) FILTER (WHERE action='SELL') AS sells,
            COUNT(*) FILTER (WHERE action='HOLD') AS holds
        FROM signals
        WHERE created_at >= NOW() - INTERVAL '{since}'
        GROUP BY bucket
        ORDER BY bucket DESC
    """))
    return [
        {
            "bucket":  str(r.bucket.date()) if hasattr(r.bucket, "date") else str(r.bucket),
            "total":   r.total,
            "buys":    r.buys,
            "sells":   r.sells,
            "holds":   r.holds,
        }
        for r in rows
    ]


# ── P&L per execution ─────────────────────────────────────────────────────────

async def pnl_per_execution(db: AsyncSession, period: Period) -> list[dict]:
    """
    Realised P&L for each filled execution.

    P&L approximation:
      BUY  → cost  = filled_qty × avg_fill_price  (negative cash flow)
      SELL → revenue = filled_qty × avg_fill_price (positive cash flow)

    We join executions → signals to get the action direction.
    A complete round-trip BUY→SELL gives true realised P&L.
    For open positions we show unrealised based on latest close from OHLCV.
    """
    since = _since(period)
    rows = await db.execute(text(f"""
        SELECT
            e.id,
            e.symbol,
            e.created_at,
            e.filled_qty,
            e.avg_fill_price,
            e.slippage_pct,
            e.status,
            s.action,
            s.confidence,
            -- latest close price for unrealised P&L estimation
            (
                SELECT close FROM ohlcv
                WHERE ohlcv.symbol = e.symbol
                ORDER BY ts DESC LIMIT 1
            ) AS latest_close
        FROM executions e
        JOIN signals s ON e.run_id = s.run_id AND e.symbol = s.symbol
        WHERE e.created_at >= NOW() - INTERVAL '{since}'
          AND e.status IN ('filled', 'partial')
        ORDER BY e.created_at DESC
    """))

    result = []
    for r in rows:
        qty        = float(r.filled_qty or 0)
        fill       = float(r.avg_fill_price or 0)
        latest     = float(r.latest_close or fill)
        action     = r.action

        # Cash flow: SELL = +revenue, BUY = unrealised at latest price
        if action == "SELL":
            pnl = qty * fill          # positive = proceeds received
        else:
            pnl = qty * (latest - fill)   # unrealised gain/loss vs fill

        result.append({
            "id":           r.id,
            "symbol":       r.symbol,
            "created_at":   r.created_at.isoformat(),
            "action":       action,
            "filled_qty":   qty,
            "avg_fill_price": fill,
            "latest_close": latest,
            "pnl":          round(pnl, 4),
            "slippage_pct": round(float(r.slippage_pct or 0), 4),
            "confidence":   round(float(r.confidence or 0), 4),
            "status":       r.status,
        })
    return result


# ── P&L per ticker ────────────────────────────────────────────────────────────

async def pnl_per_ticker(db: AsyncSession, period: Period) -> list[dict]:
    """Aggregate P&L, trade counts, and win rate per symbol."""
    executions = await pnl_per_execution(db, period)

    agg: dict[str, dict] = {}
    for e in executions:
        sym = e["symbol"]
        if sym not in agg:
            agg[sym] = {
                "symbol":     sym,
                "total_trades": 0,
                "buys":       0,
                "sells":      0,
                "total_pnl":  0.0,
                "wins":       0,
                "losses":     0,
                "avg_slippage": 0.0,
                "slippage_sum": 0.0,
                "best_trade": None,
                "worst_trade": None,
            }
        a = agg[sym]
        a["total_trades"] += 1
        a["buys"]  += 1 if e["action"] == "BUY"  else 0
        a["sells"] += 1 if e["action"] == "SELL" else 0
        a["total_pnl"]    = round(a["total_pnl"] + e["pnl"], 4)
        a["slippage_sum"] += e["slippage_pct"]

        if e["pnl"] > 0:
            a["wins"] += 1
            if a["best_trade"] is None or e["pnl"] > a["best_trade"]["pnl"]:
                a["best_trade"] = e
        elif e["pnl"] < 0:
            a["losses"] += 1
            if a["worst_trade"] is None or e["pnl"] < a["worst_trade"]["pnl"]:
                a["worst_trade"] = e

    for sym, a in agg.items():
        n = a["total_trades"]
        a["win_rate"]     = round(a["wins"] / n * 100, 1) if n else 0.0
        a["avg_slippage"] = round(a["slippage_sum"] / n, 4) if n else 0.0
        del a["slippage_sum"]

    return sorted(agg.values(), key=lambda x: x["total_pnl"], reverse=True)


# ── P&L bucketed by period ────────────────────────────────────────────────────

async def pnl_by_period(db: AsyncSession, period: Period) -> list[dict]:
    """
    Total P&L per time bucket — used to draw the equity curve.
    P&L here is the net of all SELL proceeds + unrealised BUY gains per bucket.
    """
    executions = await pnl_per_execution(db, period)
    bucket_fmt = {"daily": "%Y-%m-%d", "weekly": "%Y-W%W", "monthly": "%Y-%m"}[period]

    buckets: dict[str, dict] = {}
    for e in executions:
        dt  = datetime.fromisoformat(e["created_at"])
        key = dt.strftime(bucket_fmt)
        if key not in buckets:
            buckets[key] = {"bucket": key, "pnl": 0.0, "trades": 0}
        buckets[key]["pnl"]    = round(buckets[key]["pnl"] + e["pnl"], 4)
        buckets[key]["trades"] += 1

    # Sort chronologically
    rows = sorted(buckets.values(), key=lambda x: x["bucket"])

    # Compute cumulative P&L
    cum = 0.0
    for r in rows:
        cum += r["pnl"]
        r["cumulative_pnl"] = round(cum, 4)

    return rows


# ── Overall summary stats ─────────────────────────────────────────────────────

async def summary_stats(db: AsyncSession, period: Period) -> dict:
    """Single-object summary card for the top of the analytics panel."""
    executions  = await pnl_per_execution(db, period)
    trade_rows  = await trade_counts_by_period(db, period)

    total_signals  = sum(r["total"] for r in trade_rows)
    total_buys     = sum(r["buys"]  for r in trade_rows)
    total_sells    = sum(r["sells"] for r in trade_rows)
    total_holds    = sum(r["holds"] for r in trade_rows)

    filled = [e for e in executions]
    total_pnl   = round(sum(e["pnl"] for e in filled), 4)
    wins        = [e for e in filled if e["pnl"] > 0]
    losses      = [e for e in filled if e["pnl"] < 0]
    win_rate    = round(len(wins) / len(filled) * 100, 1) if filled else 0.0
    avg_conf    = round(sum(e["confidence"] for e in filled) / len(filled), 4) if filled else 0.0
    avg_slip    = round(sum(e["slippage_pct"] for e in filled) / len(filled), 4) if filled else 0.0
    best_trade  = max(filled, key=lambda e: e["pnl"], default=None)
    worst_trade = min(filled, key=lambda e: e["pnl"], default=None)

    # Current win/loss streak
    streak_val, streak_type = 0, "none"
    for e in executions:   # already ordered desc
        if streak_val == 0:
            streak_type = "win" if e["pnl"] > 0 else "loss"
        if (streak_type == "win" and e["pnl"] > 0) or \
           (streak_type == "loss" and e["pnl"] < 0):
            streak_val += 1
        else:
            break

    return {
        "period":          period,
        "total_signals":   total_signals,
        "total_buys":      total_buys,
        "total_sells":     total_sells,
        "total_holds":     total_holds,
        "filled_executions": len(filled),
        "total_pnl":       total_pnl,
        "win_rate":        win_rate,
        "wins":            len(wins),
        "losses":          len(losses),
        "avg_confidence":  avg_conf,
        "avg_slippage_pct": avg_slip,
        "best_trade":      best_trade,
        "worst_trade":     worst_trade,
        "current_streak":  {"type": streak_type, "count": streak_val},
        "as_of":           datetime.now(timezone.utc).isoformat(),
    }


# ── Full analytics bundle ─────────────────────────────────────────────────────

async def full_analytics(db: AsyncSession, period: Period) -> dict:
    """All analytics in one call — used by the /analytics endpoint."""
    return {
        "summary":      await summary_stats(db, period),
        "by_period":    await pnl_by_period(db, period),
        "by_ticker":    await pnl_per_ticker(db, period),
        "trade_counts": await trade_counts_by_period(db, period),
    }
