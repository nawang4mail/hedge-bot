"""
Additional ORM models for enrichment data and training results.

Tables:
  ohlcv_hourly     — hourly OHLCV candles (TimescaleDB hypertable)
  news_sentiment   — daily GDELT news sentiment per ticker
  reddit_activity  — daily Reddit mention stats per ticker
  earnings_events  — earnings dates + EPS surprise per ticker
  sec_filings      — 8-K / 10-K / 10-Q filing dates per ticker
  backtest_results — walk-forward backtest output per ticker/model
  model_registry   — metadata for each trained model file
"""
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Float, Boolean, Integer,
    DateTime, Text, JSON, UniqueConstraint, Index, ForeignKey
)
from db.connection import Base


def _now():
    return datetime.now(timezone.utc)


# ── Hourly OHLCV (TimescaleDB hypertable) ─────────────────────────────────────

class OHLCVHourly(Base):
    __tablename__ = "ohlcv_hourly"
    __table_args__ = (
        UniqueConstraint("symbol", "ts", name="uq_ohlcv_hourly_symbol_ts"),
        Index("ix_ohlcv_hourly_symbol_ts", "symbol", "ts"),
    )

    id     = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), ForeignKey("tickers.symbol", ondelete="CASCADE"), nullable=False)
    ts     = Column(DateTime(timezone=True), nullable=False)
    open   = Column(Float, nullable=False)
    high   = Column(Float, nullable=False)
    low    = Column(Float, nullable=False)
    close  = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    def to_dict(self):
        return {
            "symbol": self.symbol, "ts": self.ts.isoformat(),
            "open": self.open, "high": self.high,
            "low": self.low, "close": self.close, "volume": self.volume,
        }


# ── News sentiment (GDELT) ─────────────────────────────────────────────────────

class NewsSentiment(Base):
    __tablename__ = "news_sentiment"
    __table_args__ = (
        UniqueConstraint("symbol", "date", "source", name="uq_news_symbol_date_source"),
        Index("ix_news_symbol_date", "symbol", "date"),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    symbol        = Column(String(16), ForeignKey("tickers.symbol", ondelete="CASCADE"), nullable=False)
    date          = Column(DateTime(timezone=True), nullable=False)
    source        = Column(String(32), nullable=False, default="gdelt")  # gdelt | newsapi
    sentiment     = Column(Float, nullable=True)     # -1.0 to +1.0
    article_count = Column(Integer, nullable=True)
    avg_tone      = Column(Float, nullable=True)     # GDELT GoldsteinScale
    positive_pct  = Column(Float, nullable=True)
    negative_pct  = Column(Float, nullable=True)
    top_themes    = Column(JSON, nullable=True)       # list of top GDELT themes
    created_at    = Column(DateTime(timezone=True), default=_now)

    def to_dict(self):
        return {
            "symbol": self.symbol, "date": self.date.isoformat(),
            "source": self.source, "sentiment": self.sentiment,
            "article_count": self.article_count, "avg_tone": self.avg_tone,
            "positive_pct": self.positive_pct, "negative_pct": self.negative_pct,
            "top_themes": self.top_themes,
        }


# ── Reddit activity ────────────────────────────────────────────────────────────

class RedditActivity(Base):
    __tablename__ = "reddit_activity"
    __table_args__ = (
        UniqueConstraint("symbol", "date", "subreddit", name="uq_reddit_symbol_date_sub"),
        Index("ix_reddit_symbol_date", "symbol", "date"),
    )

    id             = Column(Integer, primary_key=True, autoincrement=True)
    symbol         = Column(String(16), ForeignKey("tickers.symbol", ondelete="CASCADE"), nullable=False)
    date           = Column(DateTime(timezone=True), nullable=False)
    subreddit      = Column(String(64), nullable=False)
    mention_count  = Column(Integer, default=0)
    avg_score      = Column(Float, nullable=True)      # avg Reddit upvote score
    avg_sentiment  = Column(Float, nullable=True)      # -1.0 to +1.0 title sentiment
    top_post_title = Column(Text, nullable=True)
    top_post_score = Column(Integer, nullable=True)
    created_at     = Column(DateTime(timezone=True), default=_now)

    def to_dict(self):
        return {
            "symbol": self.symbol, "date": self.date.isoformat(),
            "subreddit": self.subreddit, "mention_count": self.mention_count,
            "avg_score": self.avg_score, "avg_sentiment": self.avg_sentiment,
            "top_post_title": self.top_post_title,
        }


# ── Earnings events ────────────────────────────────────────────────────────────

class EarningsEvent(Base):
    __tablename__ = "earnings_events"
    __table_args__ = (
        UniqueConstraint("symbol", "report_date", name="uq_earnings_symbol_date"),
        Index("ix_earnings_symbol_date", "symbol", "report_date"),
    )

    id              = Column(Integer, primary_key=True, autoincrement=True)
    symbol          = Column(String(16), ForeignKey("tickers.symbol", ondelete="CASCADE"), nullable=False)
    report_date     = Column(DateTime(timezone=True), nullable=False)
    fiscal_quarter  = Column(String(8), nullable=True)    # e.g. "Q1 2024"
    eps_estimate    = Column(Float, nullable=True)
    eps_actual      = Column(Float, nullable=True)
    eps_surprise    = Column(Float, nullable=True)        # actual - estimate
    eps_surprise_pct= Column(Float, nullable=True)        # % surprise
    revenue_estimate= Column(Float, nullable=True)
    revenue_actual  = Column(Float, nullable=True)
    beat_estimate   = Column(Boolean, nullable=True)
    created_at      = Column(DateTime(timezone=True), default=_now)

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "report_date": self.report_date.isoformat(),
            "fiscal_quarter": self.fiscal_quarter,
            "eps_estimate": self.eps_estimate, "eps_actual": self.eps_actual,
            "eps_surprise": self.eps_surprise, "eps_surprise_pct": self.eps_surprise_pct,
            "beat_estimate": self.beat_estimate,
        }


# ── SEC filings ────────────────────────────────────────────────────────────────

class SECFiling(Base):
    __tablename__ = "sec_filings"
    __table_args__ = (
        UniqueConstraint("symbol", "accession_number", name="uq_sec_accession"),
        Index("ix_sec_symbol_date", "symbol", "filed_date"),
    )

    id               = Column(Integer, primary_key=True, autoincrement=True)
    symbol           = Column(String(16), ForeignKey("tickers.symbol", ondelete="CASCADE"), nullable=False)
    accession_number = Column(String(32), nullable=True)
    form_type        = Column(String(16), nullable=False)   # 8-K, 10-Q, 10-K
    filed_date       = Column(DateTime(timezone=True), nullable=False)
    period_of_report = Column(DateTime(timezone=True), nullable=True)
    description      = Column(Text, nullable=True)
    filing_url       = Column(Text, nullable=True)
    created_at       = Column(DateTime(timezone=True), default=_now)

    def to_dict(self):
        return {
            "symbol": self.symbol, "form_type": self.form_type,
            "filed_date": self.filed_date.isoformat(),
            "description": self.description, "filing_url": self.filing_url,
        }


# ── Model registry ─────────────────────────────────────────────────────────────

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "version", name="uq_model_sym_tf_ver"),
    )

    id              = Column(Integer, primary_key=True, autoincrement=True)
    symbol          = Column(String(16), nullable=False)
    timeframe       = Column(String(8), nullable=False)    # daily | hourly
    version         = Column(Integer, nullable=False, default=1)
    file_path       = Column(Text, nullable=False)
    scaler_path     = Column(Text, nullable=True)
    features        = Column(JSON, nullable=True)          # list of feature names
    train_start     = Column(DateTime(timezone=True), nullable=True)
    train_end       = Column(DateTime(timezone=True), nullable=True)
    train_accuracy  = Column(Float, nullable=True)
    test_accuracy   = Column(Float, nullable=True)
    live_accuracy   = Column(Float, nullable=True)         # updated as live trades come in
    is_active       = Column(Boolean, default=True)
    trained_at      = Column(DateTime(timezone=True), default=_now)
    last_used_at    = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self):
        return {
            "id": self.id, "symbol": self.symbol,
            "timeframe": self.timeframe, "version": self.version,
            "file_path": self.file_path, "features": self.features,
            "train_accuracy": self.train_accuracy, "test_accuracy": self.test_accuracy,
            "live_accuracy": self.live_accuracy, "is_active": self.is_active,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }


# ── Backtest results ───────────────────────────────────────────────────────────

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    symbol          = Column(String(16), nullable=False)
    timeframe       = Column(String(8), nullable=False)
    model_id        = Column(Integer, ForeignKey("model_registry.id"), nullable=True)
    start_date      = Column(DateTime(timezone=True), nullable=False)
    end_date        = Column(DateTime(timezone=True), nullable=False)
    total_trades    = Column(Integer, default=0)
    win_rate        = Column(Float, nullable=True)
    sharpe_ratio    = Column(Float, nullable=True)
    max_drawdown    = Column(Float, nullable=True)
    total_return    = Column(Float, nullable=True)
    benchmark_return= Column(Float, nullable=True)   # buy-and-hold return
    alpha           = Column(Float, nullable=True)   # total_return - benchmark_return
    equity_curve    = Column(JSON, nullable=True)    # list of {date, value}
    trades          = Column(JSON, nullable=True)    # list of individual trade results
    created_at      = Column(DateTime(timezone=True), default=_now)

    def to_dict(self):
        return {
            "id": self.id, "symbol": self.symbol, "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_trades": self.total_trades, "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio, "max_drawdown": self.max_drawdown,
            "total_return": self.total_return, "benchmark_return": self.benchmark_return,
            "alpha": self.alpha, "equity_curve": self.equity_curve,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
