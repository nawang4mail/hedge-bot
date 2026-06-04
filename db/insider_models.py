"""
Insider trading ORM models.

Tables:
  insiders          — person registry (name, personal CIK, known titles)
  insider_watchlist — people the user is actively tracking
  sec_insider_trades — every Form 4 transaction row
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


# ── Insider person registry ────────────────────────────────────────────────────

class Insider(Base):
    __tablename__ = "insiders"
    __table_args__ = (
        UniqueConstraint("cik", name="uq_insider_cik"),
        Index("ix_insider_name", "name"),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    cik        = Column(String(12), nullable=False)        # personal SEC CIK
    name       = Column(String(128), nullable=False)       # "COOK TIMOTHY D"
    name_clean = Column(String(128), nullable=True)        # "Tim Cook" (user-friendly)
    companies  = Column(JSON, nullable=True)               # [{symbol, title, is_current}]
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    def to_dict(self):
        return {
            "id": self.id, "cik": self.cik,
            "name": self.name, "name_clean": self.name_clean,
            "companies": self.companies,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ── Insider watchlist ──────────────────────────────────────────────────────────

class InsiderWatchlist(Base):
    __tablename__ = "insider_watchlist"
    __table_args__ = (
        UniqueConstraint("cik", name="uq_watchlist_cik"),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    cik        = Column(String(12), ForeignKey("insiders.cik", ondelete="CASCADE"),
                        nullable=False)
    name_clean = Column(String(128), nullable=True)
    notes      = Column(Text, nullable=True)
    added_at   = Column(DateTime(timezone=True), default=_now)

    def to_dict(self):
        return {
            "id": self.id, "cik": self.cik,
            "name_clean": self.name_clean, "notes": self.notes,
            "added_at": self.added_at.isoformat() if self.added_at else None,
        }


# ── Form 4 transactions ────────────────────────────────────────────────────────

class InsiderTrade(Base):
    __tablename__ = "sec_insider_trades"
    __table_args__ = (
        UniqueConstraint("accession_number", "line_number",
                         name="uq_insider_trade_acc_line"),
        Index("ix_insider_trade_symbol_date", "symbol", "transaction_date"),
        Index("ix_insider_trade_cik_date",    "insider_cik", "transaction_date"),
    )

    id               = Column(Integer, primary_key=True, autoincrement=True)
    symbol           = Column(String(16), nullable=False)
    company_name     = Column(String(128), nullable=True)

    # Insider identity
    insider_cik      = Column(String(12), nullable=True)
    insider_name     = Column(String(128), nullable=False)
    insider_title    = Column(String(256), nullable=True)  # "CEO", "Director", etc.
    is_director      = Column(Boolean, default=False)
    is_officer       = Column(Boolean, default=False)
    is_ten_pct_owner = Column(Boolean, default=False)

    # Filing metadata
    accession_number = Column(String(32), nullable=True)
    filed_date       = Column(DateTime(timezone=True), nullable=False)
    line_number      = Column(Integer, default=0)          # row within the filing

    # Transaction details
    transaction_date = Column(DateTime(timezone=True), nullable=True)
    transaction_code = Column(String(4), nullable=True)
    # Transaction codes:
    #   P = Open market purchase (strongest bullish signal)
    #   S = Open market sale
    #   M = Option exercise (compensation — weaker signal)
    #   A = Grant/award
    #   D = Disposition to company
    #   G = Gift
    #   F = Tax withholding
    transaction_type = Column(String(32), nullable=True)   # Buy / Sell / Exercise / Award / Other
    is_open_market   = Column(Boolean, default=False)      # P or S only

    shares           = Column(Float, nullable=True)        # shares transacted
    price_per_share  = Column(Float, nullable=True)
    total_value      = Column(Float, nullable=True)        # shares * price
    shares_owned_after = Column(Float, nullable=True)
    ownership_type   = Column(String(1), nullable=True)    # D=Direct, I=Indirect

    filing_url       = Column(Text, nullable=True)
    created_at       = Column(DateTime(timezone=True), default=_now)

    def to_dict(self):
        return {
            "id":               self.id,
            "symbol":           self.symbol,
            "company_name":     self.company_name,
            "insider_cik":      self.insider_cik,
            "insider_name":     self.insider_name,
            "insider_title":    self.insider_title,
            "is_director":      self.is_director,
            "is_officer":       self.is_officer,
            "filed_date":       self.filed_date.isoformat() if self.filed_date else None,
            "transaction_date": self.transaction_date.isoformat() if self.transaction_date else None,
            "transaction_code": self.transaction_code,
            "transaction_type": self.transaction_type,
            "is_open_market":   self.is_open_market,
            "shares":           self.shares,
            "price_per_share":  self.price_per_share,
            "total_value":      self.total_value,
            "shares_owned_after": self.shares_owned_after,
            "ownership_type":   self.ownership_type,
            "filing_url":       self.filing_url,
        }
