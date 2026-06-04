"""
Insider Trading API

Endpoints:
  GET  /insider/trades              — trades filtered by symbol/person/type/date
  GET  /insider/person/{cik}        — full profile for one insider
  GET  /insider/search              — search insiders by name
  GET  /insider/watchlist           — list watched insiders
  POST /insider/watch               — add insider to watchlist
  DELETE /insider/watch/{cik}       — remove from watchlist
  POST /insider/backfill/symbol     — trigger Form 4 backfill for tickers
  POST /insider/backfill/person     — trigger Form 4 backfill for a person
  GET  /insider/summary/{symbol}    — insider activity summary for a ticker
  POST /insider/chat                — LLM chat over insider data
"""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import select, and_, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_db, AsyncSessionLocal
from db.insider_models import Insider, InsiderWatchlist, InsiderTrade

router = APIRouter(prefix="/insider", tags=["insider"])

INSIDER_CHAT_SYSTEM = """You are a financial analyst specialising in SEC insider trading data.
You receive Form 4 transaction records and answer questions about insider activity.

Rules:
- Focus on open-market transactions (transaction_code P=Buy, S=Sell) — these are most meaningful.
- Distinguish between open-market purchases (real conviction) and option exercises (compensation).
- Note patterns: cluster of buys before earnings, sustained selling, large single transactions.
- Be specific: cite insider names, dates, share counts, dollar values.
- Flag anything unusual or noteworthy.
- End with: "Based on: N transactions, [date range]"
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(s, fallback):
    if not s:
        return fallback
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return fallback


# ── Trades ─────────────────────────────────────────────────────────────────────

@router.get("/trades")
async def get_trades(
    symbol:          Optional[str]  = None,
    insider_name:    Optional[str]  = None,
    insider_cik:     Optional[str]  = None,
    start:           Optional[str]  = None,
    end:             Optional[str]  = None,
    transaction_type: Optional[str] = None,   # Buy | Sell | Exercise | Award
    open_market_only: bool          = False,
    min_value:       Optional[float] = None,
    is_officer:      Optional[bool]  = None,
    is_director:     Optional[bool]  = None,
    limit:           int             = 200,
    db: AsyncSession = Depends(get_db),
):
    since = _parse_dt(start, datetime.now(timezone.utc) - timedelta(days=365))
    until = _parse_dt(end,   datetime.now(timezone.utc))

    filters = [
        InsiderTrade.transaction_date >= since,
        InsiderTrade.transaction_date <= until,
    ]
    if symbol:
        filters.append(InsiderTrade.symbol == symbol.upper())
    if insider_name:
        filters.append(InsiderTrade.insider_name.ilike(f"%{insider_name}%"))
    if insider_cik:
        filters.append(InsiderTrade.insider_cik == insider_cik)
    if transaction_type:
        filters.append(InsiderTrade.transaction_type == transaction_type)
    if open_market_only:
        filters.append(InsiderTrade.is_open_market == True)
    if min_value is not None:
        filters.append(InsiderTrade.total_value >= min_value)
    if is_officer is not None:
        filters.append(InsiderTrade.is_officer == is_officer)
    if is_director is not None:
        filters.append(InsiderTrade.is_director == is_director)

    result = await db.execute(
        select(InsiderTrade).where(and_(*filters))
        .order_by(InsiderTrade.transaction_date.desc()).limit(limit)
    )
    rows = [r.to_dict() for r in result.scalars().all()]
    return {"count": len(rows), "rows": rows}


# ── Insider person profile ─────────────────────────────────────────────────────

@router.get("/person/{cik}")
async def get_person_profile(cik: str, db: AsyncSession = Depends(get_db)):
    """
    Full profile for one insider:
    - identity + known companies
    - all transactions (last 3 years)
    - aggregated stats per company
    - portfolio (shares owned per company, latest filing)
    """
    # Identity
    result = await db.execute(
        select(Insider).where(Insider.cik == cik)
    )
    insider = result.scalar_one_or_none()

    # All transactions
    trades_result = await db.execute(
        select(InsiderTrade)
        .where(InsiderTrade.insider_cik == cik)
        .order_by(InsiderTrade.transaction_date.desc())
        .limit(500)
    )
    trades = [t.to_dict() for t in trades_result.scalars().all()]

    # Portfolio — latest shares_owned_after per company
    portfolio_result = await db.execute(text("""
        SELECT DISTINCT ON (symbol)
            symbol, company_name, shares_owned_after,
            price_per_share, transaction_date, insider_title
        FROM sec_insider_trades
        WHERE insider_cik = :cik
          AND shares_owned_after IS NOT NULL
        ORDER BY symbol, transaction_date DESC
    """), {"cik": cik})
    portfolio = [
        {
            "symbol":        r.symbol,
            "company_name":  r.company_name,
            "shares_owned":  r.shares_owned_after,
            "last_price":    r.price_per_share,
            "est_value":     round(float(r.shares_owned_after or 0) * float(r.price_per_share or 0), 2),
            "title":         r.insider_title,
            "as_of":         r.transaction_date.isoformat() if r.transaction_date else None,
        }
        for r in portfolio_result.fetchall()
    ]

    # Stats per company
    stats_result = await db.execute(text("""
        SELECT
            symbol,
            COUNT(*) FILTER (WHERE transaction_code = 'P') as open_buys,
            COUNT(*) FILTER (WHERE transaction_code = 'S') as open_sells,
            SUM(total_value) FILTER (WHERE transaction_code = 'P') as buy_value,
            SUM(total_value) FILTER (WHERE transaction_code = 'S') as sell_value,
            MAX(transaction_date) as last_tx_date
        FROM sec_insider_trades
        WHERE insider_cik = :cik
        GROUP BY symbol
        ORDER BY last_tx_date DESC
    """), {"cik": cik})
    stats = [
        {
            "symbol":     r.symbol,
            "open_buys":  r.open_buys or 0,
            "open_sells": r.open_sells or 0,
            "buy_value":  round(float(r.buy_value or 0), 2),
            "sell_value": round(float(r.sell_value or 0), 2),
            "net_value":  round(float(r.buy_value or 0) - float(r.sell_value or 0), 2),
            "last_tx_date": r.last_tx_date.isoformat() if r.last_tx_date else None,
        }
        for r in stats_result.fetchall()
    ]

    return {
        "insider":   insider.to_dict() if insider else {"cik": cik},
        "trades":    trades,
        "portfolio": portfolio,
        "stats":     stats,
    }


# ── Search insiders ────────────────────────────────────────────────────────────

@router.get("/search")
async def search_insiders(
    name:  Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Search insiders by name or by the company they trade in."""
    if name:
        result = await db.execute(
            select(Insider)
            .where(Insider.name.ilike(f"%{name}%"))
            .limit(limit)
        )
        return [r.to_dict() for r in result.scalars().all()]

    if symbol:
        result = await db.execute(
            select(InsiderTrade.insider_cik, InsiderTrade.insider_name,
                   InsiderTrade.insider_title)
            .where(InsiderTrade.symbol == symbol.upper())
            .distinct()
            .limit(limit)
        )
        return [
            {"cik": r.insider_cik, "name": r.insider_name, "title": r.insider_title}
            for r in result.fetchall()
        ]
    return []


# ── Watchlist ──────────────────────────────────────────────────────────────────

@router.get("/watchlist")
async def get_watchlist(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(InsiderWatchlist).order_by(InsiderWatchlist.added_at.desc())
    )
    return [r.to_dict() for r in result.scalars().all()]


class WatchRequest(BaseModel):
    cik:        str
    name_clean: Optional[str] = None
    notes:      Optional[str] = None


@router.post("/watch", status_code=201)
async def watch_insider(req: WatchRequest, db: AsyncSession = Depends(get_db)):
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    stmt = pg_insert(InsiderWatchlist).values({
        "cik":        req.cik,
        "name_clean": req.name_clean,
        "notes":      req.notes,
    }).on_conflict_do_nothing(constraint="uq_watchlist_cik")
    await db.execute(stmt)
    await db.commit()
    return {"watched": True, "cik": req.cik}


@router.delete("/watch/{cik}")
async def unwatch_insider(cik: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete
    await db.execute(delete(InsiderWatchlist).where(InsiderWatchlist.cik == cik))
    await db.commit()
    return {"unwatched": True, "cik": cik}


# ── Activity summary for a ticker ─────────────────────────────────────────────

@router.get("/summary/{symbol}")
async def insider_summary(symbol: str, days: int = 90,
                           db: AsyncSession = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(text("""
        SELECT
            insider_name, insider_title, insider_cik,
            COUNT(*) FILTER (WHERE transaction_code = 'P') as open_buys,
            COUNT(*) FILTER (WHERE transaction_code = 'S') as open_sells,
            SUM(shares)      FILTER (WHERE transaction_code = 'P') as buy_shares,
            SUM(shares)      FILTER (WHERE transaction_code = 'S') as sell_shares,
            SUM(total_value) FILTER (WHERE transaction_code = 'P') as buy_value,
            SUM(total_value) FILTER (WHERE transaction_code = 'S') as sell_value,
            MAX(transaction_date) as last_activity
        FROM sec_insider_trades
        WHERE symbol = :sym
          AND transaction_date >= :since
          AND is_open_market = TRUE
        GROUP BY insider_name, insider_title, insider_cik
        ORDER BY last_activity DESC
    """), {"sym": symbol.upper(), "since": since})

    insiders = [
        {
            "name":         r.insider_name,
            "title":        r.insider_title,
            "cik":          r.insider_cik,
            "open_buys":    r.open_buys or 0,
            "open_sells":   r.open_sells or 0,
            "buy_shares":   float(r.buy_shares  or 0),
            "sell_shares":  float(r.sell_shares or 0),
            "buy_value":    round(float(r.buy_value  or 0), 2),
            "sell_value":   round(float(r.sell_value or 0), 2),
            "last_activity": r.last_activity.isoformat() if r.last_activity else None,
        }
        for r in result.fetchall()
    ]

    total_buys  = sum(i["open_buys"]  for i in insiders)
    total_sells = sum(i["open_sells"] for i in insiders)
    net_value   = sum(i["buy_value"] - i["sell_value"] for i in insiders)

    return {
        "symbol": symbol.upper(), "period_days": days,
        "total_buys": total_buys, "total_sells": total_sells,
        "net_value": round(net_value, 2),
        "signal": "bullish" if net_value > 0 and total_buys > total_sells else
                  "bearish" if net_value < 0 and total_sells > total_buys else "neutral",
        "insiders": insiders,
    }


# ── Backfill triggers ──────────────────────────────────────────────────────────

class SymbolBackfillReq(BaseModel):
    symbols: list[str]

class PersonBackfillReq(BaseModel):
    name_or_cik: str


@router.post("/backfill/symbol", status_code=202)
async def backfill_symbol(req: SymbolBackfillReq, background_tasks: BackgroundTasks):
    import uuid
    from api.main import _training_jobs, ws_manager
    from training.progress import ProgressEmitter

    job_id = str(uuid.uuid4())
    _training_jobs[job_id] = {"status": "running", "job_id": job_id, "events": []}

    async def _run():
        async def _bcast(ch, data):
            _training_jobs[job_id]["events"].append(data)
            await ws_manager.broadcast(ch, data)
        emitter = ProgressEmitter(job_id, _bcast)
        try:
            from training.backfill_insider import backfill_by_symbol
            await backfill_by_symbol(req.symbols, emitter=emitter)
            _training_jobs[job_id]["status"] = "completed"
            await emitter.done(f"Insider backfill complete for {req.symbols}")
        except Exception as e:
            _training_jobs[job_id]["status"] = "error"
            await emitter.fatal(str(e))

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued"}


@router.post("/backfill/person", status_code=202)
async def backfill_person(req: PersonBackfillReq, background_tasks: BackgroundTasks):
    import uuid
    from api.main import _training_jobs, ws_manager
    from training.progress import ProgressEmitter

    job_id = str(uuid.uuid4())
    _training_jobs[job_id] = {"status": "running", "job_id": job_id, "events": []}

    async def _run():
        async def _bcast(ch, data):
            _training_jobs[job_id]["events"].append(data)
            await ws_manager.broadcast(ch, data)
        emitter = ProgressEmitter(job_id, _bcast)
        try:
            from training.backfill_insider import backfill_by_person
            await backfill_by_person(req.name_or_cik, emitter=emitter)
            _training_jobs[job_id]["status"] = "completed"
            await emitter.done(f"Insider backfill complete: {req.name_or_cik}")
        except Exception as e:
            _training_jobs[job_id]["status"] = "error"
            await emitter.fatal(str(e))

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued"}


# ── LLM chat over insider data ─────────────────────────────────────────────────

class InsiderChatReq(BaseModel):
    question: str
    rows:     list[dict]
    context:  str = ""   # e.g. "AAPL insider trades 2023"


@router.post("/chat")
async def insider_chat(req: InsiderChatReq):
    if not req.rows:
        return {"answer": "No insider data loaded — run a backfill first."}

    from langchain_core.messages import SystemMessage, HumanMessage
    from agents.llm_router import get_llm
    from langchain_ollama import ChatOllama
    from config import settings

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.2,
    )

    payload = (
        f"Context: {req.context}\n"
        f"Rows ({len(req.rows)}):\n"
        f"{json.dumps(req.rows[:150], default=str, separators=(',',':'))}"
    )

    response = llm.invoke([
        SystemMessage(content=INSIDER_CHAT_SYSTEM),
        HumanMessage(content=f"{payload}\n\nQUESTION: {req.question}"),
    ])
    return {"answer": str(response.content), "rows_used": min(len(req.rows), 150)}
