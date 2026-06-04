"""
Progress Emitter — structured job event system.

All backfill and training scripts receive a ProgressEmitter instance.
They call emit() at key checkpoints; the emitter pushes structured JSON
events to the WebSocket /ws/training channel so the UI can render
phase trackers, progress bars, ETA, and error cards in real time.

Event types
-----------
phase_start     — a named phase is beginning (e.g. "OHLCV Backfill")
phase_done      — a named phase completed successfully
ticker_start    — starting work on a specific ticker
ticker_progress — progress within a ticker (current, total, unit, eta_s)
ticker_done     — ticker finished (rows_inserted, rows_total, elapsed_s)
ticker_error    — ticker failed (message, retryable)
row_insert      — rows written to DB (symbol, table, count, cumulative)
metric          — training metric (epoch, loss, accuracy, etc.)
feature_imp     — top feature importances after training
log             — plain text log line (fallback)
done            — entire job completed
error           — job-level fatal error
"""
from __future__ import annotations
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Literal

EventType = Literal[
    "phase_start", "phase_done",
    "ticker_start", "ticker_progress", "ticker_done", "ticker_error",
    "row_insert", "metric", "feature_imp",
    "log", "done", "error",
]


class ProgressEmitter:
    """
    Thread-safe progress emitter.

    Usage inside an async background task:
        emitter = ProgressEmitter(job_id, broadcast_fn)
        await emitter.phase_start("OHLCV Backfill", total_tickers=3)
        await emitter.ticker_start("AAPL", total=1825, unit="candles")
        await emitter.ticker_progress("AAPL", current=400, total=1825)
        await emitter.ticker_done("AAPL", rows_inserted=400, elapsed_s=3.2)
        await emitter.phase_done("OHLCV Backfill")

    The broadcast_fn signature: async (channel: str, data: dict) -> None
    """

    def __init__(self, job_id: str, broadcast: Callable[..., Coroutine]):
        self.job_id    = job_id
        self._broadcast = broadcast
        self._start_time = time.monotonic()
        self._ticker_starts: dict[str, float] = {}
        self._phases: list[dict] = []
        self._errors: list[dict] = []

    # ── Core emit ─────────────────────────────────────────────────────────────

    async def emit(self, event_type: EventType, **data):
        payload = {
            "type":    event_type,
            "job_id":  self.job_id,
            "ts":      datetime.now(timezone.utc).isoformat(),
            "elapsed": round(time.monotonic() - self._start_time, 1),
            **data,
        }
        await self._broadcast("training", payload)

    # ── Phase helpers ─────────────────────────────────────────────────────────

    async def phase_start(self, name: str, total_tickers: int = 0, sources: list[str] | None = None):
        self._phases.append({"name": name, "started_at": time.monotonic()})
        await self.emit("phase_start", phase=name,
                        total_tickers=total_tickers, sources=sources or [])

    async def phase_done(self, name: str):
        elapsed = 0.0
        for p in self._phases:
            if p["name"] == name:
                elapsed = round(time.monotonic() - p["started_at"], 1)
        await self.emit("phase_done", phase=name, elapsed_s=elapsed)

    # ── Ticker helpers ────────────────────────────────────────────────────────

    async def ticker_start(self, symbol: str, total: int, unit: str = "rows",
                           source: str = ""):
        self._ticker_starts[symbol] = time.monotonic()
        await self.emit("ticker_start", symbol=symbol,
                        total=total, unit=unit, source=source)

    async def ticker_progress(self, symbol: str, current: int, total: int,
                              unit: str = "rows", detail: str = ""):
        elapsed = time.monotonic() - self._ticker_starts.get(symbol, time.monotonic())
        eta_s   = None
        pct     = 0
        if total > 0:
            pct = round(current / total * 100, 1)
            if current > 0:
                rate  = current / elapsed if elapsed > 0 else 1
                eta_s = round((total - current) / rate, 0)

        await self.emit("ticker_progress",
                        symbol=symbol, current=current, total=total,
                        pct=pct, unit=unit, eta_s=eta_s, detail=detail)

    async def ticker_done(self, symbol: str, rows_inserted: int = 0,
                          rows_total: int = 0, source: str = ""):
        elapsed = round(time.monotonic() - self._ticker_starts.get(symbol, time.monotonic()), 1)
        await self.emit("ticker_done", symbol=symbol,
                        rows_inserted=rows_inserted,
                        rows_total=rows_total,
                        elapsed_s=elapsed, source=source)

    async def ticker_error(self, symbol: str, message: str,
                           source: str = "", retryable: bool = True):
        self._errors.append({"symbol": symbol, "source": source, "message": message})
        await self.emit("ticker_error", symbol=symbol,
                        source=source, message=message, retryable=retryable)

    # ── Training-specific helpers ─────────────────────────────────────────────

    async def metric(self, symbol: str, timeframe: str, epoch: int,
                     total_epochs: int, train_loss: float | None = None,
                     val_loss: float | None = None, accuracy: float | None = None):
        pct   = round(epoch / total_epochs * 100, 1) if total_epochs else 0
        await self.emit("metric", symbol=symbol, timeframe=timeframe,
                        epoch=epoch, total_epochs=total_epochs, pct=pct,
                        train_loss=train_loss, val_loss=val_loss, accuracy=accuracy)

    async def feature_importance(self, symbol: str, timeframe: str,
                                  importances: list[tuple[str, float]]):
        """importances: list of (feature_name, score) sorted descending."""
        await self.emit("feature_imp", symbol=symbol, timeframe=timeframe,
                        top_features=[{"name": n, "score": round(s, 4)}
                                       for n, s in importances[:10]])

    # ── Log / done / error ────────────────────────────────────────────────────

    async def log(self, message: str, level: str = "info"):
        await self.emit("log", message=message, level=level)

    async def done(self, summary: str = ""):
        await self.emit("done", summary=summary,
                        total_elapsed=round(time.monotonic() - self._start_time, 1),
                        error_count=len(self._errors),
                        errors=self._errors)

    async def fatal(self, message: str):
        await self.emit("error", message=message,
                        total_elapsed=round(time.monotonic() - self._start_time, 1))
