"""
Insider Trading Backfill — SEC EDGAR Form 4 filings.

Two modes:
  1. By ticker  — fetch all insider trades for a company symbol
  2. By person  — fetch all trades filed by a specific insider (by name or CIK)
     This lets you track Tim Cook across Apple, any boards he sits on, etc.

Source: SEC EDGAR (free, no API key)
  - Company submissions: https://data.sec.gov/submissions/CIK{cik}.json
  - Individual filer:    https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4
  - Form 4 XML parser:   https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/

Transaction code mapping:
  P = Open market purchase   → Buy  (strongest signal)
  S = Open market sale       → Sell
  M = Option exercise        → Exercise
  A = Grant / award          → Award
  D = Disposition to issuer  → Other
  G = Gift                   → Other
  F = Tax withholding sale   → Other
  C = Conversion             → Other

Usage:
  python -m training.backfill_insider --symbols AAPL TSLA
  python -m training.backfill_insider --person "Tim Cook"
  python -m training.backfill_insider --cik 0001513362
"""
from __future__ import annotations
import argparse
import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.connection import AsyncSessionLocal, init_db
from db.insider_models import Insider, InsiderWatchlist, InsiderTrade
from training.progress import ProgressEmitter

EDGAR_HEADERS = {"User-Agent": "hedge_bot/1.0 nawang4mail@gmail.com"}
EDGAR_BASE    = "https://data.sec.gov"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

# Transaction code → human-readable type + is_open_market flag
TX_MAP = {
    "P": ("Buy",      True),
    "S": ("Sell",     True),
    "M": ("Exercise", False),
    "A": ("Award",    False),
    "D": ("Other",    False),
    "G": ("Other",    False),
    "F": ("Other",    False),
    "C": ("Other",    False),
    "X": ("Exercise", False),
    "J": ("Other",    False),
}


# ── EDGAR helpers ──────────────────────────────────────────────────────────────

async def get_company_cik(symbol: str, client: httpx.AsyncClient) -> Optional[str]:
    r = await client.get("https://www.sec.gov/files/company_tickers.json")
    if r.status_code != 200:
        return None
    for entry in r.json().values():
        if entry.get("ticker", "").upper() == symbol.upper():
            return str(entry["cik_str"]).zfill(10)
    return None


async def search_person_cik(name: str, client: httpx.AsyncClient) -> list[dict]:
    """Search EDGAR for a person's CIK by name. Returns list of matches."""
    r = await client.get(
        "https://efts.sec.gov/LATEST/search-index",
        params={"q": f'"{name}"', "dateRange": "custom",
                "startdt": "2000-01-01", "forms": "4"},
        timeout=20,
    )
    # Fallback: use company search endpoint
    r2 = await client.get(
        "https://www.sec.gov/cgi-bin/browse-edgar",
        params={"company": "", "CIK": name, "type": "4",
                "dateb": "", "owner": "include", "count": "10",
                "search_text": "", "action": "getcompany", "output": "atom"},
        timeout=20,
    )
    matches = []
    # Parse basic name→CIK from the response
    content = r2.text
    for m in re.finditer(r'CIK=(\d+)[^>]*>([^<]+)<', content):
        cik, found_name = m.group(1).zfill(10), m.group(2).strip()
        if name.lower() in found_name.lower():
            matches.append({"cik": cik, "name": found_name})
    return matches[:5]


async def fetch_form4_list(cik: str, client: httpx.AsyncClient,
                            is_company: bool = True) -> list[dict]:
    """
    Return list of Form 4 filing metadata for a given CIK.
    Works for both company CIKs (to get all insider filings for that company)
    and personal CIKs (to get all filings by that individual).
    """
    r = await client.get(f"{EDGAR_BASE}/submissions/CIK{cik}.json", timeout=30)
    if r.status_code != 200:
        return []
    data     = r.json()
    filings  = data.get("filings", {}).get("recent", {})
    forms    = filings.get("form", [])
    dates    = filings.get("filingDate", [])
    accnums  = filings.get("accessionNumber", [])
    primary  = filings.get("primaryDocument", [])

    results = []
    for form, date, acc, doc in zip(forms, dates, accnums, primary):
        if form == "4":
            results.append({
                "filed_date":      date,
                "accession_number": acc,
                "primary_doc":     doc,
                "cik":             cik,
            })
    return results


async def parse_form4_xml(cik: str, accession: str, filed_date: str,
                           symbol: str, client: httpx.AsyncClient) -> list[dict]:
    """
    Download and parse the Form 4 XML for one filing.
    Returns list of transaction dicts.
    """
    acc_clean = accession.replace("-", "")
    # Try to find the XML document
    index_url = f"{EDGAR_ARCHIVE}/{int(cik)}/{acc_clean}/{accession}-index.htm"
    try:
        r = await client.get(index_url, timeout=20)
        # Find the .xml link in the index
        xml_match = re.search(r'href="([^"]+\.xml)"', r.text, re.IGNORECASE)
        if not xml_match:
            return []
        xml_path = xml_match.group(1)
        xml_url  = f"https://www.sec.gov{xml_path}" if xml_path.startswith('/') else \
                   f"{EDGAR_ARCHIVE}/{int(cik)}/{acc_clean}/{xml_path}"
        xml_r = await client.get(xml_url, timeout=20)
        if xml_r.status_code != 200:
            return []
        return _parse_xml(xml_r.text, symbol, accession, filed_date)
    except Exception:
        return []


def _parse_xml(xml_text: str, symbol: str, accession: str,
               filed_date: str) -> list[dict]:
    """Parse Form 4 XML into a list of transaction dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    def _find(node, *tags):
        for tag in tags:
            node = node.find(tag)
            if node is None:
                return None
        return node.text if node is not None else None

    # Issuer (company)
    issuer    = root.find("issuer")
    co_symbol = (_find(issuer, "issuerTradingSymbol") or symbol).upper()
    co_name   = _find(issuer, "issuerName") or ""

    # Reporting owner
    owner        = root.find("reportingOwner")
    insider_name = ""
    insider_cik  = ""
    insider_title = ""
    is_director = is_officer = is_ten_pct = False

    if owner is not None:
        insider_name  = _find(owner, "reportingOwnerId", "rptOwnerName") or ""
        insider_cik   = (_find(owner, "reportingOwnerId", "rptOwnerCik") or "").zfill(10)
        rel           = owner.find("reportingOwnerRelationship")
        if rel is not None:
            is_director = (rel.findtext("isDirector") or "0") == "1"
            is_officer  = (rel.findtext("isOfficer")  or "0") == "1"
            is_ten_pct  = (rel.findtext("isTenPercentOwner") or "0") == "1"
            insider_title = rel.findtext("officerTitle") or (
                "Director" if is_director else
                "10% Owner" if is_ten_pct else ""
            )

    filed_dt = datetime.fromisoformat(filed_date).replace(tzinfo=timezone.utc)

    rows = []
    line = 0

    # Non-derivative transactions (open market buys/sells)
    for tx_table in root.findall(".//nonDerivativeTransaction"):
        line += 1
        code     = tx_table.findtext(".//transactionCode") or ""
        tx_type, is_open = TX_MAP.get(code, ("Other", False))
        shares_el = tx_table.find(".//transactionShares/value")
        price_el  = tx_table.find(".//transactionPricePerShare/value")
        owned_el  = tx_table.find(".//sharesOwnedFollowingTransaction/value")
        date_el   = tx_table.findtext(".//transactionDate/value") or filed_date
        own_el    = tx_table.findtext(".//directOrIndirectOwnership/value") or "D"

        shares = float(shares_el.text) if shares_el is not None and shares_el.text else None
        price  = float(price_el.text)  if price_el  is not None and price_el.text  else None
        owned  = float(owned_el.text)  if owned_el  is not None and owned_el.text  else None
        value  = round(shares * price, 2) if shares and price else None

        try:
            tx_date = datetime.fromisoformat(date_el).replace(tzinfo=timezone.utc)
        except Exception:
            tx_date = filed_dt

        rows.append({
            "symbol":             co_symbol,
            "company_name":       co_name,
            "insider_cik":        insider_cik,
            "insider_name":       insider_name,
            "insider_title":      insider_title,
            "is_director":        is_director,
            "is_officer":         is_officer,
            "is_ten_pct_owner":   is_ten_pct,
            "accession_number":   accession,
            "filed_date":         filed_dt,
            "line_number":        line,
            "transaction_date":   tx_date,
            "transaction_code":   code,
            "transaction_type":   tx_type,
            "is_open_market":     is_open,
            "shares":             shares,
            "price_per_share":    price,
            "total_value":        value,
            "shares_owned_after": owned,
            "ownership_type":     own_el,
            "filing_url":         f"{EDGAR_ARCHIVE}/{int(insider_cik or 0)}/{accession.replace('-','')}/",
        })

    return rows


# ── DB upsert ─────────────────────────────────────────────────────────────────

async def upsert_trades(rows: list[dict]) -> int:
    if not rows:
        return 0
    async with AsyncSessionLocal() as db:
        stmt   = pg_insert(InsiderTrade).values(rows).on_conflict_do_nothing(
            constraint="uq_insider_trade_acc_line")
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount


async def upsert_insider(cik: str, name: str, companies: list) -> None:
    async with AsyncSessionLocal() as db:
        stmt = pg_insert(Insider).values({
            "cik": cik, "name": name, "name_clean": _clean_name(name),
            "companies": companies,
        }).on_conflict_do_update(
            constraint="uq_insider_cik",
            set_={"name": name, "companies": companies,
                  "updated_at": datetime.now(timezone.utc)}
        )
        await db.execute(stmt)
        await db.commit()


def _clean_name(raw: str) -> str:
    """'COOK TIMOTHY D' → 'Timothy D Cook'"""
    parts = raw.strip().split()
    if not parts:
        return raw
    # If ALL CAPS, title-case and reverse last/first
    if raw == raw.upper() and len(parts) >= 2:
        last  = parts[0].title()
        first = " ".join(p.title() for p in parts[1:])
        return f"{first} {last}"
    return raw.title()


# ── Backfill by ticker ────────────────────────────────────────────────────────

async def backfill_by_symbol(
    symbols: list[str],
    emitter: Optional[ProgressEmitter] = None,
):
    await init_db()
    if emitter:
        await emitter.phase_start("Insider Trades (by company)",
                                   total_tickers=len(symbols),
                                   sources=["sec_edgar_form4"])

    async with httpx.AsyncClient(headers=EDGAR_HEADERS, timeout=30) as client:
        for symbol in symbols:
            if emitter:
                await emitter.ticker_start(symbol, total=100,
                                            unit="filings", source="form4")
            try:
                cik = await get_company_cik(symbol, client)
                if not cik:
                    raise ValueError(f"CIK not found for {symbol}")

                filings = await fetch_form4_list(cik, client, is_company=True)
                if emitter:
                    await emitter.log(f"{symbol}: found {len(filings)} Form 4 filings")

                all_rows  = []
                insiders_seen: dict[str, dict] = {}
                for i, f in enumerate(filings):
                    rows = await parse_form4_xml(
                        cik, f["accession_number"], f["filed_date"], symbol, client
                    )
                    all_rows.extend(rows)
                    # Collect insider metadata
                    for r in rows:
                        if r["insider_cik"]:
                            key = r["insider_cik"]
                            if key not in insiders_seen:
                                insiders_seen[key] = {
                                    "cik": key, "name": r["insider_name"],
                                    "companies": []
                                }
                            insiders_seen[key]["companies"].append({
                                "symbol": symbol, "title": r["insider_title"],
                                "is_director": r["is_director"],
                                "is_officer": r["is_officer"],
                            })

                    if emitter and (i % 10 == 0 or i == len(filings) - 1):
                        await emitter.ticker_progress(
                            symbol, current=i+1, total=len(filings),
                            unit="filings",
                            detail=f"{len(all_rows)} transactions parsed"
                        )

                n = await upsert_trades(all_rows)
                # Register insiders in registry
                for insider in insiders_seen.values():
                    await upsert_insider(
                        insider["cik"], insider["name"], insider["companies"]
                    )

                if emitter:
                    await emitter.ticker_done(symbol, rows_inserted=n,
                                               rows_total=len(all_rows),
                                               source="form4")
                else:
                    print(f"  {symbol}: {len(all_rows)} transactions, {n} new")

            except Exception as e:
                if emitter:
                    await emitter.ticker_error(symbol, str(e), source="form4")
                else:
                    print(f"  FAILED {symbol}: {e}")

    if emitter:
        await emitter.phase_done("Insider Trades (by company)")


# ── Backfill by person ────────────────────────────────────────────────────────

async def backfill_by_person(
    name_or_cik: str,
    emitter: Optional[ProgressEmitter] = None,
) -> list[dict]:
    """
    Fetch all Form 4 filings by a specific insider (by name or CIK).
    Returns the insider's trade rows across ALL companies.
    """
    await init_db()

    async with httpx.AsyncClient(headers=EDGAR_HEADERS, timeout=30) as client:
        # Resolve CIK
        if name_or_cik.isdigit():
            cik = name_or_cik.zfill(10)
        else:
            matches = await search_person_cik(name_or_cik, client)
            if not matches:
                raise ValueError(f"No SEC filer found for: {name_or_cik}")
            cik = matches[0]["cik"]
            if emitter:
                await emitter.log(f"Resolved '{name_or_cik}' → CIK {cik} ({matches[0]['name']})")

        if emitter:
            await emitter.phase_start(f"Insider: {name_or_cik}",
                                       total_tickers=1, sources=["form4_personal"])
            await emitter.ticker_start(name_or_cik, total=100,
                                        unit="filings", source="form4_personal")

        filings = await fetch_form4_list(cik, client, is_company=False)
        if emitter:
            await emitter.log(f"Found {len(filings)} Form 4 filings for CIK {cik}")

        all_rows = []
        companies_seen = {}
        for i, f in enumerate(filings):
            # For personal CIK, the issuer symbol comes from parsing the XML
            rows = await parse_form4_xml(cik, f["accession_number"],
                                          f["filed_date"], "", client)
            all_rows.extend(rows)
            for r in rows:
                if r["symbol"] and r["symbol"] not in companies_seen:
                    companies_seen[r["symbol"]] = r["insider_title"]

            if emitter and (i % 5 == 0 or i == len(filings) - 1):
                await emitter.ticker_progress(
                    name_or_cik, current=i+1, total=len(filings),
                    unit="filings",
                    detail=f"{len(all_rows)} transactions · {len(companies_seen)} companies"
                )

        # Build company list for insider registry
        companies = [{"symbol": s, "title": t} for s, t in companies_seen.items()]
        insider_name = all_rows[0]["insider_name"] if all_rows else name_or_cik
        await upsert_insider(cik, insider_name, companies)

        n = await upsert_trades(all_rows)
        if emitter:
            await emitter.ticker_done(name_or_cik, rows_inserted=n,
                                       rows_total=len(all_rows), source="form4_personal")
            await emitter.phase_done(f"Insider: {name_or_cik}")
        else:
            print(f"  {name_or_cik} (CIK {cik}): {len(all_rows)} transactions "
                  f"across {len(companies_seen)} companies, {n} new")

        return all_rows


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main(symbols: list[str], person: str | None, cik: str | None):
    if person or cik:
        await backfill_by_person(cik or person)
    elif symbols:
        await backfill_by_symbol(symbols)
    else:
        print("Provide --symbols or --person / --cik")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--person",  type=str, help="Insider name e.g. 'Tim Cook'")
    parser.add_argument("--cik",     type=str, help="Personal SEC CIK number")
    args = parser.parse_args()
    asyncio.run(main(args.symbols, args.person, args.cik))
