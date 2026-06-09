"""
GDELT News Sentiment Backfill via Google BigQuery.
"""
from __future__ import annotations
import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.connection import AsyncSessionLocal, init_db
from db.training_models import NewsSentiment
from training.progress import ProgressEmitter

_CONN_FILE = Path(__file__).parent.parent / "connections.json"


def _get_bq_client():
    creds_data = json.loads(_CONN_FILE.read_text()) if _CONN_FILE.exists() else {}
    bq = creds_data.get("bigquery", {})
    project  = bq.get("project_id", "")
    cred_path = bq.get("credentials_json", "")
    from google.cloud import bigquery
    if cred_path and Path(cred_path).exists():
        from google.oauth2 import service_account
        sa = service_account.Credentials.from_service_account_file(cred_path)
        return bigquery.Client(project=project, credentials=sa)
    return bigquery.Client(project=project)


_STOP_WORDS = {"inc", "corp", "corporation", "ltd", "llc", "co", "company",
               "group", "holdings", "technologies", "technology", "the"}

def _company_search_term(symbol: str) -> str:
    """Return the best single search word for GDELT: first meaningful word of
    the company name, falling back to the ticker itself."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        name = info.get("longName") or info.get("shortName") or ""
        if name:
            import re
            words = re.split(r"[\s,\.&/]+", name.lower())
            for word in words:
                clean = re.sub(r"[^a-z]", "", word)
                if len(clean) > 2 and clean not in _STOP_WORDS:
                    return clean
    except Exception:
        pass
    return symbol.lower()


def fetch_gdelt_sentiment(symbol: str, start: str, end: str,
                           on_progress: callable | None = None) -> list[dict]:
    from google.cloud import bigquery
    client = _get_bq_client()
    query = """
    WITH articles AS (
      SELECT
        PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) as article_date,
        SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) as tone,
        CASE WHEN SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) > 0 THEN 1 ELSE 0 END as is_positive,
        CASE WHEN SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) < 0 THEN 1 ELSE 0 END as is_negative
      FROM `gdelt-bq.gdeltv2.gkg`
      WHERE DATE >= @start_date AND DATE <= @end_date
        AND (
          LOWER(SourceCommonName) LIKE '%finance%'
          OR LOWER(SourceCommonName) LIKE '%reuters%'
          OR LOWER(SourceCommonName) LIKE '%bloomberg%'
          OR LOWER(SourceCommonName) LIKE '%cnbc%'
          OR LOWER(SourceCommonName) LIKE '%marketwatch%'
        )
        AND (
          LOWER(Persons)       LIKE @symbol_pattern
          OR LOWER(Organizations) LIKE @symbol_pattern
        )
    )
    SELECT
      article_date,
      COUNT(*)                          as article_count,
      AVG(tone)                         as avg_tone,
      SUM(is_positive) / COUNT(*)       as positive_pct,
      SUM(is_negative) / COUNT(*)       as negative_pct
    FROM articles
    GROUP BY article_date
    ORDER BY article_date
    """
    # GDELT DATE column is INT64 in YYYYMMDDHHMMSS format (e.g. 20240101120000)
    start_i      = int(start.replace("-", "") + "000000")
    end_i        = int(end.replace("-", "")   + "235959")
    search_term  = _company_search_term(symbol)
    print(f"    GDELT search term for {symbol}: %{search_term}%")
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date",     "INT64",  start_i),
            bigquery.ScalarQueryParameter("end_date",       "INT64",  end_i),
            bigquery.ScalarQueryParameter("symbol_pattern", "STRING", f"%{search_term}%"),
        ]
    )
    raw_rows = list(client.query(query, job_config=job_config).result())
    total    = len(raw_rows)
    rows     = []
    for i, row in enumerate(raw_rows):
        avg_tone  = float(row.avg_tone or 0)
        sentiment = max(-1.0, min(1.0, avg_tone / 10.0))
        rows.append({
            "symbol":        symbol.upper(),
            "date":          datetime.combine(row.article_date,
                                               datetime.min.time()).replace(tzinfo=timezone.utc),
            "source":        "gdelt",
            "sentiment":     round(sentiment, 4),
            "article_count": int(row.article_count),
            "avg_tone":      round(avg_tone, 4),
            "positive_pct":  round(float(row.positive_pct or 0), 4),
            "negative_pct":  round(float(row.negative_pct or 0), 4),
            "top_themes":    None,
        })
        if on_progress and (i % 10 == 0 or i == total - 1):
            on_progress(i + 1, total)
    return rows


async def upsert_sentiment(rows: list[dict]) -> int:
    if not rows:
        return 0
    async with AsyncSessionLocal() as db:
        stmt   = pg_insert(NewsSentiment).values(rows).on_conflict_do_nothing(
            constraint="uq_news_symbol_date_source")
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount


async def backfill(symbols: list[str], years: int = 3,
                   emitter: ProgressEmitter | None = None):
    await init_db()
    end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    total_days = 365 * years

    if emitter:
        await emitter.phase_start("News Sentiment (GDELT)",
                                   total_tickers=len(symbols), sources=["gdelt"])

    for symbol in symbols:
        if emitter:
            await emitter.ticker_start(symbol, total=total_days,
                                        unit="days", source="gdelt")
        try:
            loop = asyncio.get_event_loop()

            def _on_prog(current, total):
                asyncio.run_coroutine_threadsafe(
                    emitter.ticker_progress(symbol, current, total, unit="days",
                        detail=f"Processing GDELT articles…"),
                    loop
                ) if emitter else None

            rows = fetch_gdelt_sentiment(symbol, start, end,
                                          on_progress=_on_prog if emitter else None)
            n    = await upsert_sentiment(rows)
            if emitter:
                await emitter.ticker_done(symbol, rows_inserted=n,
                                           rows_total=len(rows), source="gdelt")
                await emitter.emit("row_insert", symbol=symbol,
                                   table="news_sentiment", count=n, cumulative=n)
            else:
                print(f"  {symbol}: {len(rows)} days, {n} new rows")
        except Exception as e:
            if emitter:
                await emitter.ticker_error(symbol, str(e), source="gdelt")
            else:
                print(f"  FAILED {symbol}: {e}")

    if emitter:
        await emitter.phase_done("News Sentiment (GDELT)")
    else:
        print("✅ News backfill complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--years",   type=int, default=3)
    args = parser.parse_args()
    asyncio.run(backfill(args.symbols, args.years))
