"""
Ticker API smoke tests — no DB, no external services required.

Uses httpx.AsyncClient + anyio (both already in requirements) instead of the
deprecated starlette TestClient so it works with httpx >= 0.23.
"""
import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

# ── Path ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# conftest.py stubs fastapi/httpx/starlette; clear them so we get the real ones
for _key in list(sys.modules.keys()):
    if _key in ("fastapi", "httpx") or _key.startswith(("fastapi.", "starlette.", "httpx.")):
        del sys.modules[_key]

import fastapi    # real fastapi
import httpx      # real httpx

# ── Stub api sub-modules before importing api.main ────────────────────────────
for _name in ("api.ws_manager", "api.connections", "api.data", "api.insider"):
    _m = MagicMock()
    _m.router = fastapi.APIRouter()
    _m.load_all_connections_to_settings = lambda: None
    sys.modules[_name] = _m


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ticker(symbol: str, name: str = "", notes: str = ""):
    t = MagicMock()
    t.to_dict.return_value = {
        "symbol": symbol.upper(),
        "name": name,
        "notes": notes,
        "active": True,
    }
    return t


async def _fake_get_db():
    yield MagicMock()


# ── App fixture ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture(scope="module")
def app():
    with patch("api.discord_alerts.start_watchlist_poller", new=AsyncMock()):
        with patch("api.main.lifespan", _noop_lifespan):
            import importlib
            import api.main as _mod
            importlib.reload(_mod)
            from db import get_db
            _mod.app.dependency_overrides[get_db] = _fake_get_db
            yield _mod.app
            _mod.app.dependency_overrides.clear()


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


pytestmark = pytest.mark.anyio


# ── GET /tickers ──────────────────────────────────────────────────────────────

class TestListTickers:
    async def test_returns_empty_list(self, client, app):
        with patch("api.main.queries") as q:
            q.get_watchlist = AsyncMock(return_value=[])
            r = await client.get("/tickers")
        assert r.status_code == 200
        assert r.json() == []

    async def test_returns_watchlist(self, client, app):
        tickers = [_make_ticker("AAPL", "Apple"), _make_ticker("TSLA", "Tesla")]
        with patch("api.main.queries") as q:
            q.get_watchlist = AsyncMock(return_value=tickers)
            r = await client.get("/tickers")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        assert body[0]["symbol"] == "AAPL"
        assert body[1]["symbol"] == "TSLA"


# ── POST /tickers ─────────────────────────────────────────────────────────────

class TestAddTicker:
    async def test_happy_path_returns_201(self, client, app):
        with patch("api.main.queries") as q:
            q.ticker_exists = AsyncMock(return_value=False)
            q.add_ticker    = AsyncMock(return_value=_make_ticker("AAPL", "Apple Inc."))
            r = await client.post("/tickers", json={"symbol": "AAPL", "name": "Apple Inc."})
        assert r.status_code == 201
        assert r.json()["symbol"] == "AAPL"

    async def test_duplicate_returns_409(self, client, app):
        with patch("api.main.queries") as q:
            q.ticker_exists = AsyncMock(return_value=True)
            r = await client.post("/tickers", json={"symbol": "AAPL"})
        assert r.status_code == 409

    async def test_invalid_symbol_rejected(self, client, app):
        r = await client.post("/tickers", json={"symbol": "../../etc/passwd"})
        assert r.status_code == 422

    async def test_lowercase_symbol_accepted(self, client, app):
        with patch("api.main.queries") as q:
            q.ticker_exists = AsyncMock(return_value=False)
            q.add_ticker    = AsyncMock(return_value=_make_ticker("TSLA"))
            r = await client.post("/tickers", json={"symbol": "tsla"})
        assert r.status_code == 201

    async def test_notes_field_optional(self, client, app):
        with patch("api.main.queries") as q:
            q.ticker_exists = AsyncMock(return_value=False)
            q.add_ticker    = AsyncMock(return_value=_make_ticker("NVDA", notes="AI play"))
            r = await client.post("/tickers", json={"symbol": "NVDA", "notes": "AI play"})
        assert r.status_code == 201
        assert r.json()["notes"] == "AI play"


# ── DELETE /tickers/{symbol} ──────────────────────────────────────────────────

class TestRemoveTicker:
    async def test_happy_path_returns_200(self, client, app):
        with patch("api.main.queries") as q:
            q.remove_ticker = AsyncMock(return_value=True)
            r = await client.delete("/tickers/AAPL")
        assert r.status_code == 200
        assert r.json()["removed"] == "AAPL"

    async def test_not_found_returns_404(self, client, app):
        with patch("api.main.queries") as q:
            q.remove_ticker = AsyncMock(return_value=False)
            r = await client.delete("/tickers/FAKE")
        assert r.status_code == 404
