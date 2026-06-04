"""
SQLAlchemy ORM models for all four tables.

tickers   — watchlist of symbols to monitor
ohlcv     — daily candles per ticker (TimescaleDB hypertable on 'ts')
signals   — every BUY/SELL/HOLD decision the Decision Agent produced
executions — every order attempt by the Implementation Agent
"""
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Float, Boolean, Integer,
    DateTime, Text, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from db.connection import Base


def _now():
    return datetime.now(timezone.utc)


# ── Tickers (watchlist) ───────────────────────────────────────────────────────

class Ticker(Base):
    __tablename__ = "tickers"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(16), unique=True, nullable=False, index=True)
    name        = Column(String(128), nullable=True)
    active      = Column(Boolean, default=True, nullable=False)
    notes       = Column(Text, nullable=True)
    created_at  = Column(DateTime(timezone=True), default=_now)
    updated_at  = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    candles    = relationship("OHLCV",     back_populates="ticker_ref", lazy="dynamic")
    signals    = relationship("Signal",    back_populates="ticker_ref", lazy="dynamic")
    executions = relationship("Execution", back_populates="ticker_ref", lazy="dynamic")

    def to_dict(self):
        return {
            "id": self.id, "symbol": self.symbol, "name": self.name,
            "active": self.active, "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── OHLCV (TimescaleDB hypertable) ────────────────────────────────────────────

class OHLCV(Base):
    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint("symbol", "ts", name="uq_ohlcv_symbol_ts"),
        Index("ix_ohlcv_symbol_ts", "symbol", "ts"),
    )

    id     = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), ForeignKey("tickers.symbol", ondelete="CASCADE"),
                    nullable=False)
    ts     = Column(DateTime(timezone=True), nullable=False)   # hypertable dimension
    open   = Column(Float, nullable=False)
    high   = Column(Float, nullable=False)
    low    = Column(Float, nullable=False)
    close  = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    ticker_ref = relationship("Ticker", back_populates="candles")

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "ts": self.ts.isoformat(),
            "open": self.open, "high": self.high,
            "low": self.low, "close": self.close, "volume": self.volume,
        }


# ── Signals ───────────────────────────────────────────────────────────────────

class Signal(Base):
    __tablename__ = "signals"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(String(36), nullable=False, index=True)
    symbol      = Column(String(16), ForeignKey("tickers.symbol", ondelete="CASCADE"),
                         nullable=False)
    action      = Column(String(4),  nullable=False)    # BUY | SELL | HOLD
    quantity    = Column(Float,      nullable=False, default=0.0)
    order_type  = Column(String(16), nullable=True)
    limit_price = Column(Float,      nullable=True)
    confidence  = Column(Float,      nullable=True)
    rationale   = Column(Text,       nullable=True)
    risk_checks_passed = Column(Boolean, default=True)
    created_at  = Column(DateTime(timezone=True), default=_now, index=True)

    ticker_ref = relationship("Ticker", back_populates="signals")

    def to_dict(self):
        return {
            "id": self.id, "run_id": self.run_id, "symbol": self.symbol,
            "action": self.action, "quantity": self.quantity,
            "order_type": self.order_type, "limit_price": self.limit_price,
            "confidence": self.confidence, "rationale": self.rationale,
            "risk_checks_passed": self.risk_checks_passed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── Executions ────────────────────────────────────────────────────────────────

class Execution(Base):
    __tablename__ = "executions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_id          = Column(String(36), nullable=False, index=True)
    symbol          = Column(String(16), ForeignKey("tickers.symbol", ondelete="CASCADE"),
                             nullable=False)
    order_id        = Column(String(64), nullable=True)
    status          = Column(String(16), nullable=False)   # filled|partial|rejected|skipped
    filled_qty      = Column(Float,      default=0.0)
    avg_fill_price  = Column(Float,      nullable=True)
    slippage_pct    = Column(Float,      nullable=True)
    message         = Column(Text,       nullable=True)
    created_at      = Column(DateTime(timezone=True), default=_now, index=True)

    ticker_ref = relationship("Ticker", back_populates="executions")

    def to_dict(self):
        return {
            "id": self.id, "run_id": self.run_id, "symbol": self.symbol,
            "order_id": self.order_id, "status": self.status,
            "filled_qty": self.filled_qty, "avg_fill_price": self.avg_fill_price,
            "slippage_pct": self.slippage_pct, "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
