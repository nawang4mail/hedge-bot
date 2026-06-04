"""
Data Browser API — filtered reads + LLM chat over visible data.

Endpoints:
  GET  /data/tickers                    — all symbols in DB
  GET  /data/earnings                   — earnings rows with filters
  GET  /data/news                       — news sentiment rows with filters
  GET  /data/reddit                     — reddit activity rows with filters
  GET  /data/filings                    — SEC filings with filters
  GET  /data/ohlcv                      — OHLCV candles with filters
  GET  /data/summary/{symbol}           — quick data coverage report per symbol
  POST /data/chat                       — ask LLM about currently visible rows
  WS   /ws/data-chat                    — streaming version of /data/chat
"""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_db, AsyncSessionLocal
from db.models import OHLCV, Ticker
from db.training_models import (
    NewsSentiment, RedditActivity, EarningsEvent, SECFiling
)

router = APIRouter(prefix="/data", tags=["data"])

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _parse_dt(s: str | None, fallback: datetime) -> datetime:
    if not s:
        return fallback
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return fallback


def _compact_json(rows: list[dict], max_rows: int = 200) -> str:
    """Serialise rows to compact JSON for LLM context. Caps at max_rows."""
    return json.dumps(rows[:max_rows], default=str, separators=(",", ":"))


# ── Ticker list ────────────────────────────────────────────────────────────────

@router.get("/tickers")
async def list_data_tickers(db: AsyncSession = Depends(get_db)):
    """All symbols that have any data in the DB."""
    result = await db.execute(select(Ticker.symbol).order_by(Ticker.symbol))
    return [r[0] for r in result.fetchall()]


# ── Earnings ───────────────────────────────────────────────────────────────────

@router.get("/earnings")
async def get_earnings(
    symbol: str,
    start:  Optional[str] = None,
    end:    Optional[str] = None,
    beat:   Optional[bool] = None,      # True=beat, False=miss, None=all
    min_surprise_pct: Optional[float] = None,
    max_surprise_pct: Optional[float] = None,
    limit:  int = 200,
    db: AsyncSession = Depends(get_db),
):
    since = _parse_dt(start, datetime(2010, 1, 1, tzinfo=timezone.utc))
    until = _parse_dt(end,   datetime.now(timezone.utc))

    filters = [
        EarningsEvent.symbol      == symbol.upper(),
        EarningsEvent.report_date >= since,
        EarningsEvent.report_date <= until,
    ]
    if beat is not None:
        filters.append(EarningsEvent.beat_estimate == beat)
    if min_surprise_pct is not None:
        filters.append(EarningsEvent.eps_surprise_pct >= min_surprise_pct)
    if max_surprise_pct is not None:
        filters.append(EarningsEvent.eps_surprise_pct <= max_surprise_pct)

    result = await db.execute(
        select(EarningsEvent).where(and_(*filters))
        .order_by(EarningsEvent.report_date.desc()).limit(limit)
    )
    rows = [r.to_dict() for r in result.scalars().all()]
    return {"symbol": symbol.upper(), "count": len(rows), "rows": rows}


# ── News sentiment ─────────────────────────────────────────────────────────────

@router.get("/news")
async def get_news(
    symbol: str,
    start:  Optional[str] = None,
    end:    Optional[str] = None,
    source: Optional[str] = None,            # gdelt | newsapi
    min_sentiment: Optional[float] = None,   # e.g. -1.0
    max_sentiment: Optional[float] = None,   # e.g. +1.0
    min_articles:  Optional[int]   = None,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    since = _parse_dt(start, datetime(2010, 1, 1, tzinfo=timezone.utc))
    until = _parse_dt(end,   datetime.now(timezone.utc))

    filters = [
        NewsSentiment.symbol == symbol.upper(),
        NewsSentiment.date   >= since,
        NewsSentiment.date   <= until,
    ]
    if source:
        filters.append(NewsSentiment.source == source)
    if min_sentiment is not None:
        filters.append(NewsSentiment.sentiment >= min_sentiment)
    if max_sentiment is not None:
        filters.append(NewsSentiment.sentiment <= max_sentiment)
    if min_articles is not None:
        filters.append(NewsSentiment.article_count >= min_articles)

    result = await db.execute(
        select(NewsSentiment).where(and_(*filters))
        .order_by(NewsSentiment.date.desc()).limit(limit)
    )
    rows = [r.to_dict() for r in result.scalars().all()]
    return {"symbol": symbol.upper(), "count": len(rows), "rows": rows}


# ── Reddit ─────────────────────────────────────────────────────────────────────

@router.get("/reddit")
async def get_reddit(
    symbol:    str,
    start:     Optional[str] = None,
    end:       Optional[str] = None,
    subreddit: Optional[str] = None,
    min_mentions: Optional[int]   = None,
    min_sentiment: Optional[float] = None,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    since = _parse_dt(start, datetime(2010, 1, 1, tzinfo=timezone.utc))
    until = _parse_dt(end,   datetime.now(timezone.utc))

    filters = [
        RedditActivity.symbol == symbol.upper(),
        RedditActivity.date   >= since,
        RedditActivity.date   <= until,
    ]
    if subreddit:
        filters.append(RedditActivity.subreddit.ilike(f"%{subreddit}%"))
    if min_mentions is not None:
        filters.append(RedditActivity.mention_count >= min_mentions)
    if min_sentiment is not None:
        filters.append(RedditActivity.avg_sentiment >= min_sentiment)

    result = await db.execute(
        select(RedditActivity).where(and_(*filters))
        .order_by(RedditActivity.date.desc()).limit(limit)
    )
    rows = [r.to_dict() for r in result.scalars().all()]
    return {"symbol": symbol.upper(), "count": len(rows), "rows": rows}


# ── SEC Filings ────────────────────────────────────────────────────────────────

@router.get("/filings")
async def get_filings(
    symbol:    str,
    start:     Optional[str] = None,
    end:       Optional[str] = None,
    form_type: Optional[str] = None,   # 8-K | 10-Q | 10-K
    keyword:   Optional[str] = None,   # search in description
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    since = _parse_dt(start, datetime(2010, 1, 1, tzinfo=timezone.utc))
    until = _parse_dt(end,   datetime.now(timezone.utc))

    filters = [
        SECFiling.symbol     == symbol.upper(),
        SECFiling.filed_date >= since,
        SECFiling.filed_date <= until,
    ]
    if form_type:
        filters.append(SECFiling.form_type == form_type.upper())
    if keyword:
        filters.append(SECFiling.description.ilike(f"%{keyword}%"))

    result = await db.execute(
        select(SECFiling).where(and_(*filters))
        .order_by(SECFiling.filed_date.desc()).limit(limit)
    )
    rows = [r.to_dict() for r in result.scalars().all()]
    return {"symbol": symbol.upper(), "count": len(rows), "rows": rows}


# ── OHLCV ──────────────────────────────────────────────────────────────────────

@router.get("/ohlcv")
async def get_ohlcv(
    symbol:     str,
    start:      Optional[str] = None,
    end:        Optional[str] = None,
    timeframe:  str = "daily",          # daily | hourly
    min_volume: Optional[float] = None,
    limit: int  = 500,
    db: AsyncSession = Depends(get_db),
):
    since = _parse_dt(start, datetime.now(timezone.utc) - timedelta(days=365))
    until = _parse_dt(end,   datetime.now(timezone.utc))

    from db.training_models import OHLCVHourly
    model = OHLCV if timeframe == "daily" else OHLCVHourly

    filters = [
        model.symbol == symbol.upper(),
        model.ts     >= since,
        model.ts     <= until,
    ]
    if min_volume is not None:
        filters.append(model.volume >= min_volume)

    result = await db.execute(
        select(model).where(and_(*filters))
        .order_by(model.ts.desc()).limit(limit)
    )
    rows = [r.to_dict() for r in result.scalars().all()]
    return {"symbol": symbol.upper(), "timeframe": timeframe,
            "count": len(rows), "rows": rows}


# ── Data coverage summary ──────────────────────────────────────────────────────

@router.get("/summary/{symbol}")
async def data_summary(symbol: str, db: AsyncSession = Depends(get_db)):
    """
    Quick coverage report: how many rows exist per table,
    earliest and latest date, and gaps > 7 days in OHLCV.
    """
    from sqlalchemy import func, text

    sym = symbol.upper()

    async def count_range(model, date_col):
        r = await db.execute(
            select(func.count(), func.min(date_col), func.max(date_col))
            .where(model.symbol == sym)
        )
        row = r.one()
        return {
            "count": row[0],
            "earliest": row[1].isoformat() if row[1] else None,
            "latest":   row[2].isoformat() if row[2] else None,
        }

    ohlcv_cov   = await count_range(OHLCV,         OHLCV.ts)
    news_cov    = await count_range(NewsSentiment,  NewsSentiment.date)
    reddit_cov  = await count_range(RedditActivity, RedditActivity.date)
    earn_cov    = await count_range(EarningsEvent,  EarningsEvent.report_date)
    filing_cov  = await count_range(SECFiling,      SECFiling.filed_date)

    # Detect OHLCV gaps > 7 calendar days
    gaps = []
    ohlcv_rows = await db.execute(
        select(OHLCV.ts).where(OHLCV.symbol == sym).order_by(OHLCV.ts)
    )
    dates = [r[0] for r in ohlcv_rows.fetchall()]
    for i in range(1, len(dates)):
        diff = (dates[i] - dates[i-1]).days
        if diff > 7:
            gaps.append({
                "from": dates[i-1].date().isoformat(),
                "to":   dates[i].date().isoformat(),
                "days": diff,
            })

    return {
        "symbol":  sym,
        "ohlcv":   ohlcv_cov,
        "news":    news_cov,
        "reddit":  reddit_cov,
        "earnings": earn_cov,
        "filings": filing_cov,
        "ohlcv_gaps": gaps[:20],   # cap at 20
    }


# ── LLM Chat ───────────────────────────────────────────────────────────────────

DATA_CHAT_SYSTEM = """You are a financial data analyst assistant.
The user has filtered a dataset from a trading bot's database and is asking questions about it.
You will receive the visible rows as compact JSON, then the user's question.

Rules:
- Answer ONLY from the data provided. Do not invent numbers not in the data.
- Be specific: cite dates, values, percentages from the rows.
- If the data is insufficient to answer, say so clearly.
- Keep answers concise — 3–6 sentences unless detail is explicitly requested.
- If you spot data quality issues (gaps, outliers, suspicious values), mention them.
- Format numbers clearly: currency as $X.XX, percentages as X.X%.
- At the end, add one line: "Based on: N rows, [earliest date] → [latest date]"
"""


class ChatRequest(BaseModel):
    question: str
    rows:     list[dict]
    source:   str = ""   # earnings | news | reddit | filings | ohlcv
    symbol:   str = ""


@router.post("/chat")
async def data_chat(req: ChatRequest):
    """Non-streaming LLM chat over visible rows."""
    if not req.rows:
        return {"answer": "No data visible — apply filters and load data first."}
    if not req.question.strip():
        return {"answer": "Please enter a question."}

    from langchain_core.messages import SystemMessage, HumanMessage
    from agents.llm_router import get_llm

    llm = get_llm(temperature=0.2)

    context = (
        f"Data source: {req.source or 'mixed'}\n"
        f"Symbol: {req.symbol or 'unknown'}\n"
        f"Row count: {len(req.rows)}\n\n"
        f"DATA (JSON):\n{_compact_json(req.rows, max_rows=150)}"
    )

    response = llm.invoke([
        SystemMessage(content=DATA_CHAT_SYSTEM),
        HumanMessage(content=f"{context}\n\nQUESTION: {req.question}"),
    ])

    # LLM is set to JSON format globally — parse if needed
    try:
        parsed = json.loads(response.content)
        answer = parsed.get("answer") or parsed.get("response") or str(parsed)
    except Exception:
        answer = str(response.content)

    return {"answer": answer, "rows_used": min(len(req.rows), 150)}


# ── WebSocket streaming chat ───────────────────────────────────────────────────

@router.websocket("/ws/data-chat")
async def data_chat_ws(websocket: WebSocket):
    """
    Streaming LLM chat over visible rows via WebSocket.

    Client sends: {"question": "...", "rows": [...], "source": "...", "symbol": "..."}
    Server streams: {"type": "token", "text": "..."} then {"type": "done"}
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            req = json.loads(raw)

            question = req.get("question", "").strip()
            rows     = req.get("rows", [])
            source   = req.get("source", "")
            symbol   = req.get("symbol", "")

            if not question or not rows:
                await websocket.send_json({"type": "error",
                    "text": "No question or data provided."})
                continue

            context = (
                f"Data source: {source or 'mixed'}\n"
                f"Symbol: {symbol or 'unknown'}\n"
                f"Row count: {len(rows)}\n\n"
                f"DATA (JSON):\n{_compact_json(rows, max_rows=150)}"
            )

            from langchain_core.messages import SystemMessage, HumanMessage
            from langchain_ollama import ChatOllama
            from config import settings

            # Use non-JSON format for chat (plain text response)
            llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.2,
                # No format="json" here — we want natural prose
            )

            await websocket.send_json({"type": "start"})

            full_text = ""
            async for chunk in llm.astream([
                SystemMessage(content=DATA_CHAT_SYSTEM),
                HumanMessage(content=f"{context}\n\nQUESTION: {question}"),
            ]):
                token = chunk.content
                if token:
                    full_text += token
                    await websocket.send_json({"type": "token", "text": token})

            await websocket.send_json({
                "type": "done",
                "rows_used": min(len(rows), 150),
                "full_text": full_text,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "text": str(e)})
        except Exception:
            pass
