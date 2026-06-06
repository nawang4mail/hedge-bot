"""
FastAPI entry point.

Endpoints:
  POST /run               — trigger a full pipeline run (async, returns run_id)
  GET  /status/{id}       — poll run status + latest state
  GET  /portfolio         — live portfolio snapshot from Alpaca
  POST /halt              — engage kill-switch
  POST /resume            — disengage kill-switch
  POST /risk              — update risk parameters at runtime
  GET  /tickers           — list watchlist
  POST /tickers           — add ticker to watchlist
  DELETE /tickers/{sym}   — remove ticker from watchlist
  GET  /history/signals   — signal history (optionally filtered by symbol)
  GET  /history/executions— execution history
  GET  /history/ohlcv/{sym}— cached candles for a symbol
  WS   /ws/{run_id}       — stream agent log events as they are emitted
"""
from __future__ import annotations
import asyncio
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from agents import run_pipeline, AgentState
from config import settings
from api.ws_manager import ConnectionManager
from db import init_db, get_db, queries, analytics as db_analytics
from api.connections import router as connections_router, load_all_connections_to_settings
from api.data import router as data_router
from api.insider import router as insider_router
from api.discord_alerts import router as discord_router, start_watchlist_poller

# ── WebSocket connection manager (shared across routes) ───────────────────────
ws_manager = ConnectionManager()

# ── In-memory run registry (replace with Redis for multi-process) ─────────────
_runs: dict[str, dict[str, Any]] = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load saved API credentials from connections.json into settings
    load_all_connections_to_settings()

    # Init TimescaleDB schema
    try:
        await init_db()
        print("✅ TimescaleDB schema ready")
    except Exception as e:
        print(f"⚠️  DB init failed: {e}")

    # Warm-up: verify Ollama is reachable
    try:
        from agents.llm_router import get_llm
        get_llm()
        print("✅ Ollama LLM connection verified")
    except Exception as e:
        print(f"⚠️  Ollama not reachable: {e} — agents will fail until resolved")

    # Start Discord watchlist poller (runs every POLL_INTERVAL_MINUTES)
    poller_task = asyncio.create_task(start_watchlist_poller())
    yield
    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Hedge Bot API", version="1.0.0", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Restrict to localhost origins. Expand in .env via ALLOWED_ORIGINS if needed.
_ALLOWED_ORIGINS = ["http://localhost", "http://127.0.0.1", "null"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API-key authentication middleware ─────────────────────────────────────────
# Paths that do NOT require authentication (read-only, low-risk).
_AUTH_EXEMPT = {"/health"}
_AUTH_EXEMPT_PREFIXES = ("/status/", "/ws/")

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path
    if path in _AUTH_EXEMPT or any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES):
        return await call_next(request)

    expected = settings.api_key
    if expected:  # auth is disabled when API_KEY is not set (development mode)
        provided = request.headers.get("X-API-Key", "")
        if provided != expected:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    return await call_next(request)
app.include_router(connections_router)
app.include_router(data_router)
app.include_router(insider_router)
app.include_router(discord_router)


# ── Request/Response models ───────────────────────────────────────────────────

_TICKER_RE = re.compile(r'^[A-Z0-9.\-]{1,12}$')


class RunRequest(BaseModel):
    symbol: str

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not _TICKER_RE.match(v):
            raise ValueError("symbol must be 1-12 uppercase letters, digits, dots, or hyphens")
        return v

class RiskOverride(BaseModel):
    max_position_pct: float | None = None
    max_portfolio_risk_pct: float | None = None
    default_order_type: str | None = None


# ── Background pipeline runner ────────────────────────────────────────────────

async def _run_and_broadcast(run_id: str, symbol: str):
    """
    Runs the pipeline in a thread (blocking), broadcasting each agent log
    event to connected WebSocket clients as they appear.
    Persists signal and execution results to TimescaleDB.
    """
    loop = asyncio.get_event_loop()
    _runs[run_id] = {"status": "running", "symbol": symbol, "logs": [], "result": None}

    try:
        # Run blocking pipeline in thread pool
        state: AgentState = await loop.run_in_executor(
            None, run_pipeline, symbol, run_id
        )

        # Broadcast each log entry to subscribers of this run
        for entry in state.agent_logs:
            await ws_manager.broadcast(run_id, entry)

        # ── Persist signal & execution to DB ──────────────────────────────
        from db.connection import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            # Ensure ticker exists in watchlist (auto-add if missing)
            if not await queries.ticker_exists(db, symbol):
                await queries.add_ticker(db, symbol)

            if state.trading_signal:
                await queries.save_signal(db, run_id, state.trading_signal)

            if state.execution_report and state.trading_signal:
                await queries.save_execution(
                    db, run_id, symbol, state.execution_report
                )
            await db.commit()

        _runs[run_id]["status"]  = "completed"
        _runs[run_id]["logs"]    = state.agent_logs
        _runs[run_id]["result"]  = state.model_dump(mode="json")

        await ws_manager.broadcast(run_id, {
            "agent": "system", "status": "completed",
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
        })

    except Exception as exc:
        _runs[run_id]["status"] = "error"
        _runs[run_id]["error"]  = str(exc)
        await ws_manager.broadcast(run_id, {
            "agent": "system", "status": "error", "msg": str(exc),
            "ts": datetime.now(timezone.utc).isoformat(),
        })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/run", status_code=202)
async def trigger_run(req: RunRequest, background_tasks: BackgroundTasks):
    """Kick off an asynchronous pipeline run. Returns run_id immediately."""
    run_id = str(uuid.uuid4())
    background_tasks.add_task(_run_and_broadcast, run_id, req.symbol.upper())
    return {"run_id": run_id, "symbol": req.symbol.upper(), "status": "queued"}


@app.get("/status/{run_id}")
async def get_status(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    return _runs[run_id]


@app.get("/portfolio")
async def get_portfolio():
    """Live portfolio positions from Alpaca (paper account)."""
    if not settings.alpaca_api_key:
        return {"error": "Alpaca keys not configured", "positions": [], "equity": 0}
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,
        )
        account   = client.get_account()
        positions = client.get_all_positions()
        return {
            "equity": float(account.equity),
            "cash":   float(account.cash),
            "buying_power": float(account.buying_power),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty":    float(p.qty),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
                for p in positions
            ],
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/halt")
async def halt_trading():
    """Engage the kill-switch — all subsequent Decision nodes will HOLD."""
    settings.trading_halted = True
    await ws_manager.broadcast("system", {
        "agent": "system", "status": "HALTED",
        "ts": datetime.now(timezone.utc).isoformat(),
        "msg": "Kill-switch engaged by operator.",
    })
    return {"trading_halted": True}


@app.post("/resume")
async def resume_trading():
    """Disengage the kill-switch."""
    settings.trading_halted = False
    await ws_manager.broadcast("system", {
        "agent": "system", "status": "RESUMED",
        "ts": datetime.now(timezone.utc).isoformat(),
        "msg": "Kill-switch cleared by operator.",
    })
    return {"trading_halted": False}


@app.post("/risk")
async def update_risk(override: RiskOverride):
    """Hot-update risk parameters without restarting the server."""
    if override.max_position_pct is not None:
        settings.max_position_pct = override.max_position_pct
    if override.max_portfolio_risk_pct is not None:
        settings.max_portfolio_risk_pct = override.max_portfolio_risk_pct
    if override.default_order_type is not None:
        settings.default_order_type = override.default_order_type
    return {
        "max_position_pct":      settings.max_position_pct,
        "max_portfolio_risk_pct": settings.max_portfolio_risk_pct,
        "default_order_type":    settings.default_order_type,
    }


# ── Watchlist (tickers) ───────────────────────────────────────────────────────

class TickerRequest(BaseModel):
    symbol: str
    name: str = ""
    notes: str = ""


@app.get("/tickers")
async def list_tickers(db: AsyncSession = Depends(get_db)):
    """Return the full watchlist."""
    tickers = await queries.get_watchlist(db, active_only=False)
    return [t.to_dict() for t in tickers]


@app.post("/tickers", status_code=201)
async def add_ticker(req: TickerRequest, db: AsyncSession = Depends(get_db)):
    """Add a symbol to the watchlist."""
    if await queries.ticker_exists(db, req.symbol):
        raise HTTPException(409, f"{req.symbol.upper()} already in watchlist")
    ticker = await queries.add_ticker(db, req.symbol, req.name, req.notes)
    return ticker.to_dict()


@app.delete("/tickers/{symbol}")
async def remove_ticker(symbol: str, db: AsyncSession = Depends(get_db)):
    """Soft-delete a ticker (history is preserved)."""
    removed = await queries.remove_ticker(db, symbol)
    if not removed:
        raise HTTPException(404, f"{symbol.upper()} not found")
    return {"removed": symbol.upper()}


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/history/signals")
async def signal_history(
    symbol: str | None = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    return await queries.get_signal_history(db, symbol=symbol, limit=limit)


@app.get("/history/executions")
async def execution_history(
    symbol: str | None = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    return await queries.get_execution_history(db, symbol=symbol, limit=limit)


@app.get("/history/ohlcv/{symbol}")
async def ohlcv_history(
    symbol: str,
    days: int = 60,
    db: AsyncSession = Depends(get_db),
):
    """Return cached daily candles for a symbol."""
    return await queries.get_candles(db, symbol=symbol, days=days)


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.get("/analytics")
async def get_analytics(
    period: str = "daily",
    db: AsyncSession = Depends(get_db),
):
    """
    Full performance analytics bundle for the given period.
    period: daily | weekly | monthly
    """
    if period not in ("daily", "weekly", "monthly"):
        raise HTTPException(400, "period must be daily, weekly, or monthly")
    return await db_analytics.full_analytics(db, period)  # type: ignore[arg-type]


# ── Training / backfill routes ────────────────────────────────────────────────

class BackfillRequest(BaseModel):
    symbols: list[str]
    sources: list[str]   # ohlcv | news | reddit | earnings
    years: int = 5
    hourly: bool = False

class TrainRequest(BaseModel):
    symbols: list[str]
    timeframe: str = "daily"   # daily | hourly | both
    years: int = 5

_training_jobs: dict[str, dict] = {}


async def _run_backfill(job_id: str, req: BackfillRequest):
    _training_jobs[job_id] = {"status": "running", "job_id": job_id, "events": []}

    from training.progress import ProgressEmitter

    async def _broadcast_and_store(channel: str, data: dict):
        _training_jobs[job_id]["events"].append(data)
        await ws_manager.broadcast(channel, data)

    emitter = ProgressEmitter(job_id, _broadcast_and_store)

    try:
        if "ohlcv" in req.sources:
            from training.backfill_ohlcv import backfill as bf_ohlcv
            await bf_ohlcv(req.symbols, req.years, req.hourly, emitter=emitter)

        if "news" in req.sources:
            from training.backfill_news import backfill as bf_news
            await bf_news(req.symbols, req.years, emitter=emitter)

        if "reddit" in req.sources:
            from training.backfill_reddit import backfill as bf_reddit
            await bf_reddit(req.symbols, req.years * 365, emitter=emitter)

        if "earnings" in req.sources:
            from training.backfill_earnings import backfill as bf_earn
            await bf_earn(req.symbols, emitter=emitter)

        _training_jobs[job_id]["status"] = "completed"
        await emitter.done(f"Backfill complete for {req.symbols}")
    except Exception as e:
        _training_jobs[job_id]["status"] = "error"
        _training_jobs[job_id]["error"]  = str(e)
        await emitter.fatal(str(e))


async def _run_training(job_id: str, req: TrainRequest):
    _training_jobs[job_id] = {"status": "running", "job_id": job_id, "events": []}

    from training.progress import ProgressEmitter

    async def _broadcast_and_store(channel: str, data: dict):
        _training_jobs[job_id]["events"].append(data)
        await ws_manager.broadcast(channel, data)

    emitter = ProgressEmitter(job_id, _broadcast_and_store)

    try:
        from training.train import train_model
        timeframes = ["daily", "hourly"] if req.timeframe == "both" else [req.timeframe]
        for sym in req.symbols:
            for tf in timeframes:
                meta = await train_model(sym, tf, req.years, emitter=emitter)

        _training_jobs[job_id]["status"] = "completed"
        await emitter.done(f"Training complete for {req.symbols}")
    except Exception as e:
        _training_jobs[job_id]["status"] = "error"
        _training_jobs[job_id]["error"]  = str(e)
        await emitter.fatal(str(e))


@app.post("/training/backfill", status_code=202)
async def start_backfill(req: BackfillRequest, background_tasks: BackgroundTasks):
    import uuid
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_backfill, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.post("/training/train", status_code=202)
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    import uuid
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_training, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.post("/training/backtest")
async def start_backtest(
    symbol: str, timeframe: str = "daily", years: int = 3,
    background_tasks: BackgroundTasks = None,
):
    import uuid
    job_id = str(uuid.uuid4())
    async def _run():
        from training.backtest import backtest
        await backtest(symbol, timeframe, years)
        await ws_manager.broadcast("training", {"type": "done", "job_id": job_id, "msg": f"Backtest {symbol}/{timeframe} complete"})
    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued"}


@app.get("/training/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in _training_jobs:
        raise HTTPException(404, "Job not found")
    return _training_jobs[job_id]


@app.get("/training/models")
async def list_models(db: AsyncSession = Depends(get_db)):
    from db.training_models import ModelRegistry
    from sqlalchemy import select
    result = await db.execute(
        select(ModelRegistry).order_by(ModelRegistry.symbol, ModelRegistry.timeframe)
    )
    return [r.to_dict() for r in result.scalars().all()]


@app.get("/training/backtests")
async def list_backtests(
    symbol: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    from db.training_models import BacktestResult
    from sqlalchemy import select
    stmt = select(BacktestResult).order_by(BacktestResult.created_at.desc()).limit(50)
    if symbol:
        stmt = stmt.where(BacktestResult.symbol == symbol.upper())
    result = await db.execute(stmt)
    # Exclude heavy equity_curve and trades arrays from list view
    rows = []
    for r in result.scalars().all():
        d = r.to_dict()
        d.pop("equity_curve", None)
        d.pop("trades", None)
        rows.append(d)
    return rows


@app.get("/training/backtests/{backtest_id}")
async def get_backtest(backtest_id: int, db: AsyncSession = Depends(get_db)):
    from db.training_models import BacktestResult
    from sqlalchemy import select
    result = await db.execute(
        select(BacktestResult).where(BacktestResult.id == backtest_id)
    )
    r = result.scalar_one_or_none()
    if not r:
        raise HTTPException(404, "Backtest not found")
    return r.to_dict()


@app.websocket("/ws/training")
async def training_ws(websocket: WebSocket):
    await ws_manager.connect("training", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect("training", websocket)


@app.get("/analytics/summary")
async def get_analytics_summary(
    period: str = "daily",
    db: AsyncSession = Depends(get_db),
):
    if period not in ("daily", "weekly", "monthly"):
        raise HTTPException(400, "period must be daily, weekly, or monthly")
    return await db_analytics.summary_stats(db, period)  # type: ignore[arg-type]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "trading_halted": settings.trading_halted,
        "model": settings.ollama_model,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    Connect to receive real-time agent log events for a specific run.
    Use run_id="system" to receive kill-switch and global notifications.
    """
    await ws_manager.connect(run_id, websocket)
    try:
        # Replay existing logs for late-connecting clients
        if run_id in _runs:
            for entry in _runs[run_id].get("logs", []):
                await websocket.send_json(entry)

        while True:
            # Keep connection alive; client messages are ignored
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, websocket)
