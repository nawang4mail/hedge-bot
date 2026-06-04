"""
Discord integration for Hedge Bot.

Outbound alerts  — rich embeds via webhook (no bot account needed)
Watchlist poller — background task that checks for new insider trades
                   every POLL_INTERVAL_MINUTES and fires alerts

FastAPI router:
  GET    /discord/status          — webhook connection status
  POST   /discord/settings        — save / update webhook URL (+ optional bot token)
  DELETE /discord/settings        — clear Discord settings
  POST   /discord/test            — send a test embed to the webhook
  POST   /discord/alert/watchlist — manually fire watchlist digest now
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────

_CONN_FILE = Path(__file__).parent.parent / "connections.json"
POLL_INTERVAL_MINUTES = 60          # how often to scan for new watchlist trades
_DISCORD_KEY = "discord"

# In-memory: last trade date seen per CIK so we don't re-alert
_last_seen: dict[str, str] = {}     # cik → ISO date string


# ── Storage helpers ────────────────────────────────────────────────────────────

def _load_creds() -> dict:
    if _CONN_FILE.exists():
        data = json.loads(_CONN_FILE.read_text())
        return data.get(_DISCORD_KEY, {})
    return {}


def _save_creds(creds: dict):
    data: dict = {}
    if _CONN_FILE.exists():
        data = json.loads(_CONN_FILE.read_text())
    data[_DISCORD_KEY] = creds
    _CONN_FILE.write_text(json.dumps(data, indent=2))
    import os; _CONN_FILE.chmod(0o600)


def get_webhook_url() -> str | None:
    return _load_creds().get("webhook_url")


def get_bot_token() -> str | None:
    return _load_creds().get("bot_token")


# ── Embed builders ─────────────────────────────────────────────────────────────

_COLOR_BUY      = 0x22C55E   # green
_COLOR_SELL     = 0xEF4444   # red
_COLOR_EXERCISE = 0xA78BFA   # purple
_COLOR_INFO     = 0x3B82F6   # blue
_COLOR_TEST     = 0xF59E0B   # yellow


def _fmt_usd(n) -> str:
    if n is None:
        return "—"
    return f"${n:,.0f}"


def _fmt_n(n) -> str:
    if n is None:
        return "—"
    return f"{n:,.0f}"


def _trade_embed(trade: dict) -> dict:
    """Build a Discord embed dict for a single insider trade."""
    tx_type  = trade.get("transaction_type", "Unknown")
    symbol   = trade.get("symbol", "—")
    name     = trade.get("insider_name") or trade.get("name", "Unknown Insider")
    title    = trade.get("insider_title", "")
    date_str = (trade.get("transaction_date") or "")[:10]
    shares   = trade.get("shares")
    price    = trade.get("price_per_share")
    value    = trade.get("total_value")
    owned    = trade.get("shares_owned_after")
    filing   = trade.get("filing_url", "")
    is_open  = trade.get("is_open_market", False)

    color = (
        _COLOR_BUY      if tx_type == "Buy"      else
        _COLOR_SELL     if tx_type == "Sell"      else
        _COLOR_EXERCISE if tx_type == "Exercise"  else
        _COLOR_INFO
    )

    icon  = "🟢" if tx_type == "Buy" else "🔴" if tx_type == "Sell" else "🟣"
    badge = " `OPEN MARKET`" if is_open else ""

    fields = [
        {"name": "Symbol",       "value": f"**{symbol}**",          "inline": True},
        {"name": "Type",         "value": f"{icon} {tx_type}{badge}", "inline": True},
        {"name": "Date",         "value": date_str or "—",            "inline": True},
        {"name": "Shares",       "value": _fmt_n(shares),             "inline": True},
        {"name": "Price",        "value": f"${price:.2f}" if price else "—", "inline": True},
        {"name": "Total Value",  "value": _fmt_usd(value),            "inline": True},
    ]
    if owned is not None:
        fields.append({"name": "Owned After", "value": _fmt_n(owned), "inline": True})
    if title:
        fields.append({"name": "Title", "value": title, "inline": True})

    embed: dict = {
        "title":       f"📋 Insider Trade — {name}",
        "description": f"New Form 4 filing detected for a watched insider.",
        "color":       color,
        "fields":      fields,
        "footer":      {"text": "Hedge Bot · SEC EDGAR Form 4"},
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    if filing:
        embed["url"] = filing

    return embed


def _watchlist_digest_embed(insider_name: str, new_trades: list[dict]) -> dict:
    """Compact digest embed when multiple trades arrive at once."""
    buys  = [t for t in new_trades if t.get("transaction_type") == "Buy"]
    sells = [t for t in new_trades if t.get("transaction_type") == "Sell"]
    net   = sum(t.get("total_value") or 0 for t in buys) - \
            sum(t.get("total_value") or 0 for t in sells)

    color   = _COLOR_BUY if net > 0 else _COLOR_SELL if net < 0 else _COLOR_INFO
    symbols = list({t.get("symbol", "?") for t in new_trades})

    lines = []
    for t in new_trades[:8]:   # cap at 8 rows
        icon = "🟢" if t.get("transaction_type") == "Buy" else \
               "🔴" if t.get("transaction_type") == "Sell" else "🟣"
        lines.append(
            f"{icon} **{t.get('symbol','?')}** "
            f"{t.get('transaction_type','')} "
            f"{_fmt_n(t.get('shares'))} shares "
            f"@ {_fmt_usd(t.get('total_value'))} "
            f"({(t.get('transaction_date') or '')[:10]})"
        )
    if len(new_trades) > 8:
        lines.append(f"…and {len(new_trades) - 8} more")

    return {
        "title":       f"🚨 Watchlist Alert — {insider_name}",
        "description": "\n".join(lines),
        "color":       color,
        "fields": [
            {"name": "New Trades",   "value": str(len(new_trades)), "inline": True},
            {"name": "Symbols",      "value": ", ".join(symbols),   "inline": True},
            {"name": "Net Activity", "value": _fmt_usd(abs(net)) + (" net buy" if net > 0 else " net sell" if net < 0 else ""), "inline": True},
        ],
        "footer":    {"text": "Hedge Bot · Watchlist Monitor"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Sender ─────────────────────────────────────────────────────────────────────

async def send_embed(embed: dict, webhook_url: str | None = None,
                     content: str | None = None) -> bool:
    """
    POST a single embed to the Discord webhook.
    Returns True on success, False on any error (caller decides whether to log).
    """
    url = webhook_url or get_webhook_url()
    if not url:
        return False
    payload: dict = {"embeds": [embed]}
    if content:
        payload["content"] = content
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            return r.status_code in (200, 204)
    except Exception:
        return False


async def send_trade_alert(trade: dict) -> bool:
    """Send a single-trade embed to the configured webhook."""
    return await send_embed(_trade_embed(trade))


async def send_watchlist_digest(insider_name: str, trades: list[dict]) -> bool:
    if not trades:
        return False
    if len(trades) == 1:
        return await send_trade_alert(trades[0])
    return await send_embed(_watchlist_digest_embed(insider_name, trades))


# ── Watchlist poller ───────────────────────────────────────────────────────────

async def _poll_watchlist_once():
    """
    Check every watched insider for trades newer than _last_seen[cik].
    Fires a Discord alert for any new activity found.
    """
    if not get_webhook_url():
        return   # silently skip — user hasn't configured Discord yet

    from db.connection import AsyncSessionLocal
    from db.insider_models import InsiderWatchlist, InsiderTrade
    from sqlalchemy import select, and_

    try:
        async with AsyncSessionLocal() as db:
            wl_result = await db.execute(select(InsiderWatchlist))
            watchlist = wl_result.scalars().all()

            for entry in watchlist:
                cik  = entry.cik
                name = entry.name_clean or cik

                # Determine cutoff: last seen trade date or 7 days ago as first-run window
                cutoff_str = _last_seen.get(cik)
                if cutoff_str:
                    cutoff = datetime.fromisoformat(cutoff_str).replace(tzinfo=timezone.utc)
                else:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

                new_trades_result = await db.execute(
                    select(InsiderTrade)
                    .where(and_(
                        InsiderTrade.insider_cik == cik,
                        InsiderTrade.transaction_date > cutoff,
                        InsiderTrade.is_open_market == True,   # open-market only
                    ))
                    .order_by(InsiderTrade.transaction_date.desc())
                    .limit(20)
                )
                trades = [t.to_dict() for t in new_trades_result.scalars().all()]

                if trades:
                    await send_watchlist_digest(name, trades)
                    # Update cursor to most recent trade date
                    _last_seen[cik] = trades[0].get("transaction_date", "")[:19]

    except Exception as exc:
        print(f"[discord] Watchlist poll error: {exc}")


async def start_watchlist_poller():
    """Infinite loop — runs as a background asyncio task from lifespan."""
    print(f"[discord] Watchlist poller started (interval: {POLL_INTERVAL_MINUTES}m)")
    while True:
        await asyncio.sleep(POLL_INTERVAL_MINUTES * 60)
        await _poll_watchlist_once()


# ── FastAPI router ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/discord", tags=["discord"])


class DiscordSettings(BaseModel):
    webhook_url: str
    bot_token:   Optional[str] = None


@router.get("/status")
async def discord_status():
    creds = _load_creds()
    wh    = creds.get("webhook_url", "")
    return {
        "configured":    bool(wh),
        "webhook_url":   (wh[:30] + "…" + wh[-6:]) if len(wh) > 36 else wh,
        "bot_connected": bool(creds.get("bot_token")),
    }


@router.post("/settings")
async def save_discord_settings(body: DiscordSettings):
    creds: dict = _load_creds()
    creds["webhook_url"] = body.webhook_url
    if body.bot_token:
        creds["bot_token"] = body.bot_token
    _save_creds(creds)
    return {"saved": True}


@router.delete("/settings")
async def clear_discord_settings():
    _save_creds({})
    return {"cleared": True}


@router.post("/test")
async def test_webhook():
    url = get_webhook_url()
    if not url:
        raise HTTPException(400, "No webhook URL configured. POST /discord/settings first.")
    embed = {
        "title":       "✅ Hedge Bot Connected",
        "description": "Discord alerts are working. You'll receive watchlist notifications here.",
        "color":       _COLOR_TEST,
        "fields": [
            {"name": "Alert types",  "value": "Watchlist insider trades (open-market only)", "inline": False},
            {"name": "Poll interval","value": f"Every {POLL_INTERVAL_MINUTES} minutes",       "inline": True},
        ],
        "footer":    {"text": "Hedge Bot · Discord Integration"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    ok = await send_embed(embed, url)
    if not ok:
        raise HTTPException(502, "Failed to reach Discord webhook. Check the URL.")
    return {"sent": True}


@router.post("/alert/watchlist")
async def trigger_watchlist_alert():
    """Manually run the watchlist scan right now (don't wait for the poller)."""
    await _poll_watchlist_once()
    return {"triggered": True}
