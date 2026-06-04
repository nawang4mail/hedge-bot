"""
Connections API — manage external service credentials.

Credentials are stored in connections.json (gitignored).
The file lives next to .env in the project root.
Never stored in the DB or sent back to the client in plaintext.

Endpoints:
  GET    /connections              — list all services + connection status
  POST   /connections/{service}   — save credentials for a service
  DELETE /connections/{service}   — clear credentials for a service
  POST   /connections/{service}/test — live test the connection
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ── Storage ───────────────────────────────────────────────────────────────────

_CONN_FILE = Path(__file__).parent.parent / "connections.json"

SERVICES = {
    "alpaca": {
        "label": "Alpaca Markets",
        "description": "Paper and live trading broker. Required for order execution.",
        "docs_url": "https://alpaca.markets/docs/api-references/",
        "fields": [
            {"key": "api_key",    "label": "API Key",    "secret": False},
            {"key": "secret_key", "label": "Secret Key", "secret": True},
            {"key": "base_url",   "label": "Base URL",   "secret": False,
             "default": "https://paper-api.alpaca.markets"},
        ],
    },
    "bigquery": {
        "label": "Google BigQuery (GDELT)",
        "description": "Free historical news sentiment via GDELT dataset. Requires a Google Cloud project.",
        "docs_url": "https://console.cloud.google.com/bigquery",
        "fields": [
            {"key": "project_id",        "label": "GCP Project ID",           "secret": False},
            {"key": "credentials_json",  "label": "Service Account JSON path","secret": False,
             "placeholder": "/path/to/service-account.json"},
        ],
    },
    "reddit": {
        "label": "Reddit (PRAW)",
        "description": "Historical Reddit mention data from r/wallstreetbets, r/investing, r/stocks.",
        "docs_url": "https://www.reddit.com/prefs/apps",
        "fields": [
            {"key": "client_id",     "label": "Client ID",     "secret": False},
            {"key": "client_secret", "label": "Client Secret", "secret": True},
            {"key": "user_agent",    "label": "User Agent",    "secret": False,
             "default": "hedge_bot/1.0"},
        ],
    },
    "newsapi": {
        "label": "NewsAPI",
        "description": "Recent news headlines. Free tier covers last 30 days.",
        "docs_url": "https://newsapi.org/account",
        "fields": [
            {"key": "api_key", "label": "API Key", "secret": True},
        ],
    },
    "discord": {
        "label": "Discord Alerts",
        "description": "Send watchlist trade alerts and LLM chat replies to a Discord channel.",
        "docs_url": "https://support.discord.com/hc/en-us/articles/228383668",
        "fields": [
            {"key": "webhook_url", "label": "Webhook URL", "secret": True,
             "placeholder": "https://discord.com/api/webhooks/…"},
            {"key": "bot_token",   "label": "Bot Token (optional — for !insider chat)",
             "secret": True, "placeholder": ""},
        ],
    },
}


def _load() -> dict:
    if _CONN_FILE.exists():
        return json.loads(_CONN_FILE.read_text())
    return {}


def _save(data: dict):
    _CONN_FILE.write_text(json.dumps(data, indent=2))
    os.chmod(_CONN_FILE, 0o600)   # owner read/write only


def _mask(value: str) -> str:
    if not value:
        return ""
    return value[:4] + "••••••••" + value[-2:] if len(value) > 6 else "••••••••"


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/connections", tags=["connections"])


class CredentialsPayload(BaseModel):
    credentials: dict[str, str]


@router.get("")
async def list_connections():
    """
    Return all services with connection status.
    Secret field values are masked — never returned in plaintext.
    """
    stored = _load()
    result = []
    for svc_id, meta in SERVICES.items():
        saved = stored.get(svc_id, {})
        fields_out = []
        for f in meta["fields"]:
            val = saved.get(f["key"], f.get("default", ""))
            fields_out.append({
                **f,
                "value": _mask(val) if f.get("secret") else val,
                "filled": bool(saved.get(f["key"])),
            })

        # connected = all non-optional fields are filled
        required_keys = [f["key"] for f in meta["fields"]]
        connected = all(saved.get(k) for k in required_keys)

        result.append({
            "id":          svc_id,
            "label":       meta["label"],
            "description": meta["description"],
            "docs_url":    meta["docs_url"],
            "fields":      fields_out,
            "connected":   connected,
        })
    return result


@router.post("/{service_id}")
async def save_credentials(service_id: str, payload: CredentialsPayload):
    """Save credentials for a service. Merges with existing values."""
    if service_id not in SERVICES:
        raise HTTPException(404, f"Unknown service: {service_id}")
    stored = _load()
    if service_id not in stored:
        stored[service_id] = {}
    for k, v in payload.credentials.items():
        if v:   # don't overwrite with empty string
            stored[service_id][k] = v
    _save(stored)

    # Sync to live settings object so running server picks up new keys immediately
    _sync_to_settings(service_id, stored[service_id])
    return {"saved": True, "service": service_id}


@router.delete("/{service_id}")
async def clear_credentials(service_id: str):
    """Remove all stored credentials for a service."""
    if service_id not in SERVICES:
        raise HTTPException(404, f"Unknown service: {service_id}")
    stored = _load()
    stored.pop(service_id, None)
    _save(stored)
    return {"cleared": True, "service": service_id}


@router.post("/{service_id}/test")
async def test_connection(service_id: str):
    """Live-test the stored credentials for a service."""
    if service_id not in SERVICES:
        raise HTTPException(404, f"Unknown service: {service_id}")
    stored = _load().get(service_id, {})
    try:
        result = await _run_test(service_id, stored)
        return {"success": True, "service": service_id, "detail": result}
    except Exception as e:
        return {"success": False, "service": service_id, "detail": str(e)}


# ── Settings sync ─────────────────────────────────────────────────────────────

def _sync_to_settings(service_id: str, creds: dict):
    """Push saved credentials into the live settings object."""
    from config import settings
    if service_id == "alpaca":
        if creds.get("api_key"):    settings.alpaca_api_key    = creds["api_key"]
        if creds.get("secret_key"): settings.alpaca_secret_key = creds["secret_key"]
        if creds.get("base_url"):   settings.alpaca_base_url   = creds["base_url"]
    elif service_id == "newsapi":
        if creds.get("api_key"):    settings.news_api_key = creds["api_key"]
    elif service_id == "discord":
        pass   # Discord creds are read directly from connections.json by discord_alerts.py


def load_all_connections_to_settings():
    """Called at startup to populate settings from connections.json."""
    stored = _load()
    for svc_id, creds in stored.items():
        _sync_to_settings(svc_id, creds)


# ── Connection testers ────────────────────────────────────────────────────────

async def _run_test(service_id: str, creds: dict) -> str:
    if service_id == "alpaca":
        return await _test_alpaca(creds)
    elif service_id == "bigquery":
        return await _test_bigquery(creds)
    elif service_id == "reddit":
        return await _test_reddit(creds)
    elif service_id == "newsapi":
        return await _test_newsapi(creds)
    elif service_id == "discord":
        return await _test_discord(creds)
    return "No test available"


async def _test_alpaca(creds: dict) -> str:
    from alpaca.trading.client import TradingClient
    client = TradingClient(
        api_key=creds.get("api_key", ""),
        secret_key=creds.get("secret_key", ""),
        paper="paper" in creds.get("base_url", "paper"),
    )
    account = client.get_account()
    return f"Connected ✓  Equity: ${float(account.equity):,.2f}  Status: {account.status}"


async def _test_bigquery(creds: dict) -> str:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    cred_path = creds.get("credentials_json", "")
    project   = creds.get("project_id", "")
    if cred_path and Path(cred_path).exists():
        sa_creds = service_account.Credentials.from_service_account_file(cred_path)
        client = bigquery.Client(project=project, credentials=sa_creds)
    else:
        client = bigquery.Client(project=project)
    # Quick test: count rows in GDELT
    q = "SELECT COUNT(*) as n FROM `gdelt-bq.gdeltv2.gkg` LIMIT 1"
    result = list(client.query(q).result())
    return f"Connected ✓  BigQuery project: {project}"


async def _test_reddit(creds: dict) -> str:
    import praw
    reddit = praw.Reddit(
        client_id=creds.get("client_id", ""),
        client_secret=creds.get("client_secret", ""),
        user_agent=creds.get("user_agent", "hedge_bot/1.0"),
    )
    sub = reddit.subreddit("wallstreetbets")
    return f"Connected ✓  r/wallstreetbets subscribers: {sub.subscribers:,}"


async def _test_newsapi(creds: dict) -> str:
    from newsapi import NewsApiClient
    client = NewsApiClient(api_key=creds.get("api_key", ""))
    resp = client.get_top_headlines(category="business", language="en", page_size=1)
    return f"Connected ✓  Status: {resp['status']}  Total results: {resp['totalResults']:,}"


async def _test_discord(creds: dict) -> str:
    from api.discord_alerts import send_embed, _COLOR_TEST
    from datetime import datetime, timezone
    url = creds.get("webhook_url", "")
    if not url:
        raise ValueError("No webhook_url configured")
    embed = {
        "title":       "✅ Hedge Bot — Connection Test",
        "description": "Webhook is working correctly.",
        "color":       _COLOR_TEST,
        "footer":      {"text": "Hedge Bot · Connections"},
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    ok = await send_embed(embed, url)
    if not ok:
        raise ValueError("Discord returned an error — check the webhook URL")
    return "Connected ✓  Test message delivered to Discord channel"
