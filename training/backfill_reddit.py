"""
Reddit Mentions Backfill via PRAW.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.connection import AsyncSessionLocal, init_db
from db.training_models import RedditActivity
from training.progress import ProgressEmitter

_CONN_FILE = Path(__file__).parent.parent / "connections.json"
SUBREDDITS = ["wallstreetbets", "investing", "stocks"]


def _get_reddit():
    creds = json.loads(_CONN_FILE.read_text()).get("reddit", {}) if _CONN_FILE.exists() else {}
    import praw
    return praw.Reddit(
        client_id=creds.get("client_id", ""),
        client_secret=creds.get("client_secret", ""),
        user_agent=creds.get("user_agent", "hedge_bot/1.0"),
    )


def _sentiment_score(text: str) -> float:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    except ImportError:
        pos = {"bull","buy","long","moon","rocket","calls","gain"}
        neg = {"bear","sell","short","crash","puts","loss","rekt"}
        words = set(text.lower().split())
        return max(-1.0, min(1.0, (len(words & pos) - len(words & neg)) / 5.0))


def fetch_subreddit_mentions(
    symbol: str, subreddit_name: str, days: int,
    on_progress: callable | None = None,
) -> list[dict]:
    reddit = _get_reddit()
    sub    = reddit.subreddit(subreddit_name)
    since  = datetime.now(timezone.utc) - timedelta(days=days)
    by_day: dict[str, dict] = defaultdict(lambda: {
        "mentions": 0, "scores": [], "sentiments": [],
        "top_post": None, "top_score": -1,
    })

    queries    = [f"${symbol}", symbol]
    total_posts = 500 * len(queries)
    processed  = 0

    for q in queries:
        for post in sub.search(q, sort="new", time_filter="year", limit=500):
            processed += 1
            if on_progress and processed % 25 == 0:
                on_progress(processed, total_posts,
                            f"r/{subreddit_name}: scanning post {processed}")

            ts = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            if ts < since:
                continue
            text = f"{post.title} {post.selftext}"
            if not re.search(rf'\b\$?{re.escape(symbol)}\b', text, re.IGNORECASE):
                continue

            day = ts.strftime("%Y-%m-%d")
            d   = by_day[day]
            d["mentions"] += 1
            d["scores"].append(post.score)
            d["sentiments"].append(_sentiment_score(post.title))
            if post.score > d["top_score"]:
                d["top_score"] = post.score
                d["top_post"]  = post.title[:200]

    rows = []
    for day_str, d in by_day.items():
        rows.append({
            "symbol":         symbol.upper(),
            "date":           datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc),
            "subreddit":      subreddit_name,
            "mention_count":  d["mentions"],
            "avg_score":      round(sum(d["scores"]) / len(d["scores"]), 2) if d["scores"] else 0.0,
            "avg_sentiment":  round(sum(d["sentiments"]) / len(d["sentiments"]), 4) if d["sentiments"] else 0.0,
            "top_post_title": d["top_post"],
            "top_post_score": d["top_score"] if d["top_score"] > -1 else None,
        })
    return rows


async def upsert_reddit(rows: list[dict]) -> int:
    if not rows:
        return 0
    async with AsyncSessionLocal() as db:
        stmt   = pg_insert(RedditActivity).values(rows).on_conflict_do_nothing(
            constraint="uq_reddit_symbol_date_sub")
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount


async def backfill(symbols: list[str], days: int = 365,
                   emitter: ProgressEmitter | None = None):
    await init_db()
    total_steps = len(symbols) * len(SUBREDDITS)

    if emitter:
        await emitter.phase_start("Reddit Mentions",
                                   total_tickers=len(symbols),
                                   sources=SUBREDDITS)

    for symbol in symbols:
        cumulative = 0
        if emitter:
            await emitter.ticker_start(symbol,
                                        total=len(SUBREDDITS) * 500,
                                        unit="posts scanned", source="reddit")

        for sub_name in SUBREDDITS:
            if emitter:
                await emitter.log(f"{symbol} — scanning r/{sub_name}…")
            try:
                loop = asyncio.get_event_loop()

                def _on_prog(current, total, detail=""):
                    asyncio.run_coroutine_threadsafe(
                        emitter.ticker_progress(
                            symbol, current=current, total=total,
                            unit="posts", detail=detail
                        ), loop
                    ) if emitter else None

                rows = fetch_subreddit_mentions(
                    symbol, sub_name, days,
                    on_progress=_on_prog if emitter else None,
                )
                n         = await upsert_reddit(rows)
                cumulative += n
                if emitter:
                    await emitter.emit("row_insert", symbol=symbol,
                                       table="reddit_activity",
                                       count=n, cumulative=cumulative,
                                       detail=f"r/{sub_name}: {len(rows)} days, {n} new rows")
                else:
                    print(f"  {symbol} r/{sub_name}: {len(rows)} days, {n} new")
            except Exception as e:
                if emitter:
                    await emitter.ticker_error(symbol, str(e),
                                                source=f"reddit/{sub_name}")
                else:
                    print(f"  FAILED {symbol} r/{sub_name}: {e}")

        if emitter:
            await emitter.ticker_done(symbol, rows_inserted=cumulative,
                                       source="reddit")

    if emitter:
        await emitter.phase_done("Reddit Mentions")
    else:
        print("✅ Reddit backfill complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--days",    type=int, default=365)
    args = parser.parse_args()
    asyncio.run(backfill(args.symbols, args.days))
