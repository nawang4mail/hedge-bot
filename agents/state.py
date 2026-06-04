"""
Shared state schema that flows through the LangGraph pipeline.

Each agent reads the fields it needs and writes only its own output field.
This strict read/write discipline prevents cross-agent contamination.
"""
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class MarketSnapshot(BaseModel):
    """Raw data bundle produced by the Observation Agent."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    news_sentiment: float          # −1.0 (bearish) … +1.0 (bullish)
    news_headlines: list[str] = []
    ohlcv_1d: list[dict] = []      # last N daily candles


class ResearchReport(BaseModel):
    """Quantitative analysis produced by the Research Agent."""
    symbol: str
    rsi_14: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    atr_14: Optional[float] = None        # volatility proxy for sizing
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    volume_spike: bool = False
    trend: Literal["uptrend", "downtrend", "sideways"] = "sideways"
    anomalies: list[str] = []
    analyst_summary: str = ""             # LLM narrative (token-capped)


class TradingSignal(BaseModel):
    """Decision produced by the Decision Agent — the only input to execution."""
    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    quantity: float                       # shares / contracts
    order_type: Literal["limit", "market"] = "limit"
    limit_price: Optional[float] = None
    rationale: str                        # brief LLM justification
    confidence: float = Field(ge=0.0, le=1.0)
    risk_checks_passed: bool = True


class ExecutionReport(BaseModel):
    """Result produced by the Implementation Agent."""
    order_id: Optional[str] = None
    status: Literal["submitted", "filled", "partial", "rejected", "skipped"]
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    slippage_pct: Optional[float] = None
    message: str = ""


# ── Top-level graph state ─────────────────────────────────────────────────────

class AgentState(BaseModel):
    """
    Immutable-style state passed between nodes in the LangGraph.
    Each agent returns a dict with only its own key updated.
    """
    run_id: str
    symbol: str                                     # target asset for this run
    market_snapshot: Optional[MarketSnapshot] = None
    research_report: Optional[ResearchReport] = None
    trading_signal: Optional[TradingSignal] = None
    execution_report: Optional[ExecutionReport] = None

    # telemetry — broadcast to UI via WebSocket
    agent_logs: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
