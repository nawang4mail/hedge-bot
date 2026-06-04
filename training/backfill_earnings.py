"""
Earnings & SEC Filings Backfill.
"""
from __future__ import annotations
import argparse
import asyncio
import httpx
from datetime import datetime, timezone

import yfinance as yf
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.connection import AsyncSessionLocal, init_db
from db.training_models import EarningsEvent, SECFiling
from training.progress import ProgressEmitter

EDGAR_HEADERS = {"User-Agent": "hedge_bot/1.0 nawang4mail@gmail.com"}


def fetch_earnings(symbol: str) -> list[dict]:
    ticker  = yf.Ticker(symbol)
    history = ticker.earnings_history
    if history is None or history.empty:
        return []
    rows = []
    for idx, row in history.iterrows():
        try:
            report_dt = idx if hasattr(idx, 'tzinfo') else \
                        datetime.fromisoformat(str(idx)).replace(tzinfo=timezone.utc)
            if report_dt.tzinfo is None:
                report_dt = report_dt.replace(tzinfo=timezone.utc)
            eps_est    = float(row.get("EPS Estimate", 0) or 0)
            eps_actual = float(row.get("Reported EPS", 0) or 0)
            surprise   = float(row.get("Surprise(%)", 0) or 0)
            rows.append({
                "symbol": symbol.upper(), "report_date": report_dt,
                "fiscal_quarter": None,
                "eps_estimate": eps_est, "eps_actual": eps_actual,
                "eps_surprise": round(eps_actual - eps_est, 4),
                "eps_surprise_pct": round(surprise, 4),
                "revenue_estimate": None, "revenue_actual": None,
                "beat_estimate": eps_actual > eps_est,
            })
        except Exception:
            continue
    return rows


async def fetch_sec_filings(symbol: str) -> list[dict]:
    async with httpx.AsyncClient(headers=EDGAR_HEADERS, timeout=30) as client:
        ticker_resp = await client.get("https://www.sec.gov/files/company_tickers.json")
        if ticker_resp.status_code != 200:
            return []
        tickers_data = ticker_resp.json()
        cik = None
        for entry in tickers_data.values():
            if entry.get("ticker", "").upper() == symbol.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break
        if not cik:
            return []
        sub_resp = await client.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json")
        if sub_resp.status_code != 200:
            return []
        data    = sub_resp.json()
        filings = data.get("filings", {}).get("recent", {})
        rows    = []
        for form, date, acc, period, desc in zip(
            filings.get("form", []), filings.get("filingDate", []),
            filings.get("accessionNumber", []), filings.get("reportDate", []),
            filings.get("primaryDocument", []),
        ):
            if form not in ("10-K", "10-Q", "8-K"):
                continue
            try:
                rows.append({
                    "symbol": symbol.upper(), "accession_number": acc,
                    "form_type": form,
                    "filed_date": datetime.fromisoformat(date).replace(tzinfo=timezone.utc),
                    "period_of_report": datetime.fromisoformat(period).replace(tzinfo=timezone.utc) if period else None,
                    "description": desc,
                    "filing_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-','')}/{desc}",
                })
            except Exception:
                continue
    return rows


async def upsert_earnings(rows):
    if not rows: return 0
    async with AsyncSessionLocal() as db:
        r = await db.execute(pg_insert(EarningsEvent).values(rows).on_conflict_do_nothing(constraint="uq_earnings_symbol_date"))
        await db.commit(); return r.rowcount

async def upsert_filings(rows):
    if not rows: return 0
    async with AsyncSessionLocal() as db:
        r = await db.execute(pg_insert(SECFiling).values(rows).on_conflict_do_nothing(constraint="uq_sec_accession"))
        await db.commit(); return r.rowcount


async def backfill(symbols: list[str], emitter: ProgressEmitter | None = None):
    await init_db()
    total = len(symbols) * 2   # earnings + filings per symbol

    if emitter:
        await emitter.phase_start("Earnings + SEC Filings",
                                   total_tickers=len(symbols),
                                   sources=["yfinance/earnings", "sec_edgar"])

    cumulative = 0
    for i, symbol in enumerate(symbols):
        if emitter:
            await emitter.ticker_start(symbol, total=2, unit="sources")

        # Earnings
        if emitter:
            await emitter.ticker_progress(symbol, current=0, total=2, unit="sources",
                                           detail="Fetching earnings history…")
        try:
            rows = fetch_earnings(symbol)
            n    = await upsert_earnings(rows)
            cumulative += n
            if emitter:
                await emitter.ticker_progress(symbol, current=1, total=2, unit="sources",
                                               detail=f"Earnings: {len(rows)} quarters, {n} new")
                await emitter.emit("row_insert", symbol=symbol,
                                   table="earnings_events", count=n, cumulative=cumulative)
            else:
                print(f"  {symbol} earnings: {len(rows)} quarters, {n} new")
        except Exception as e:
            if emitter: await emitter.ticker_error(symbol, str(e), source="earnings")
            else: print(f"  FAILED {symbol} earnings: {e}")

        # SEC filings
        if emitter:
            await emitter.ticker_progress(symbol, current=1, total=2, unit="sources",
                                           detail="Fetching SEC EDGAR filings…")
        try:
            rows = await fetch_sec_filings(symbol)
            n    = await upsert_filings(rows)
            cumulative += n
            if emitter:
                await emitter.ticker_progress(symbol, current=2, total=2, unit="sources",
                                               detail=f"SEC filings: {len(rows)} filings, {n} new")
                await emitter.emit("row_insert", symbol=symbol,
                                   table="sec_filings", count=n, cumulative=cumulative)
            else:
                print(f"  {symbol} SEC: {len(rows)} filings, {n} new")
        except Exception as e:
            if emitter: await emitter.ticker_error(symbol, str(e), source="sec_edgar")
            else: print(f"  FAILED {symbol} SEC: {e}")

        if emitter:
            await emitter.ticker_done(symbol, rows_inserted=cumulative,
                                       source="earnings+sec")

    if emitter:
        await emitter.phase_done("Earnings + SEC Filings")
    else:
        print("✅ Earnings + SEC backfill complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    args = parser.parse_args()
    asyncio.run(backfill(args.symbols))
