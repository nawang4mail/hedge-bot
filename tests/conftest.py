"""
conftest.py — shared test setup.

Stubs out production dependencies (langgraph, yfinance, alpaca, db, etc.)
so the tests run without the full production environment installed.
Stubs are installed into sys.modules before any agent module is imported.

pandas-ta is NOT stubbed — it is installed so tests get real RSI/MACD values.
"""
import sys
import os
from unittest.mock import MagicMock

# ── Make the hedge-bot package root importable ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stub(*names):
    """Register each name as a MagicMock module in sys.modules if not present."""
    for name in names:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()


# ── langgraph — provide a functional StateGraph replacement ───────────────────

class _StateGraph:
    """Minimal StateGraph that runs nodes sequentially, honouring error short-circuit."""
    def __init__(self, schema):
        self._nodes: dict = {}

    def add_node(self, name, fn):
        self._nodes[name] = fn

    def set_entry_point(self, name):
        pass

    def add_edge(self, src, dst):
        pass

    def add_conditional_edges(self, src, cond_fn, mapping):
        pass

    def compile(self):
        nodes = self._nodes
        _ORDER = ["observation", "research", "decision", "implementation"]

        class _Compiled:
            def invoke(self, state: dict) -> dict:
                current = dict(state)
                for name in _ORDER:
                    fn = nodes.get(name)
                    if fn is None:
                        continue
                    current.update(fn(current))
                    if current.get("error"):
                        break
                return current

        return _Compiled()


_langgraph_graph = MagicMock()
_langgraph_graph.StateGraph = _StateGraph
_langgraph_graph.END = "END"
sys.modules.setdefault("langgraph", MagicMock())
sys.modules["langgraph.graph"] = _langgraph_graph

# ── langchain / LLM stack ─────────────────────────────────────────────────────
_stub(
    "langchain_core",
    "langchain_ollama",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_groq",
    "langchain_community",
    "ollama",
)

_lc_messages = MagicMock()
_lc_messages.SystemMessage = lambda content: {"role": "system", "content": content}
_lc_messages.HumanMessage  = lambda content: {"role": "user",   "content": content}
sys.modules["langchain_core.messages"] = _lc_messages

# ── Market data / brokerage ───────────────────────────────────────────────────
# yfinance must have __spec__ set; importlib.util.find_spec() (called by pandas_ta
# at import time) raises ValueError if sys.modules contains a module with no __spec__.
from importlib.machinery import ModuleSpec
_yfinance = MagicMock()
_yfinance.__spec__ = ModuleSpec("yfinance", None)
sys.modules["yfinance"] = _yfinance

_stub("newsapi", "newsapi.newsapi_client")
_stub(
    "alpaca",
    "alpaca.trading",
    "alpaca.trading.client",
    "alpaca.trading.requests",
    "alpaca.trading.enums",
    "alpaca.trading.models",
)

# ── Database (asyncpg / SQLAlchemy / TimescaleDB) ─────────────────────────────
_stub("asyncpg", "psycopg2", "psycopg2.extras")
_stub(
    "sqlalchemy",
    "sqlalchemy.ext",
    "sqlalchemy.ext.asyncio",
    "sqlalchemy.orm",
    "sqlalchemy.orm.declarative",
    "sqlalchemy.dialects",
    "sqlalchemy.dialects.postgresql",
    "alembic",
)

# Provide a usable db package stub so `from db.connection import AsyncSessionLocal`
# and `from db import queries` don't raise ImportError.
_db_pkg      = MagicMock()
_db_conn     = MagicMock()
_db_queries  = MagicMock()
_db_models   = MagicMock()
_db_training = MagicMock()
_db_analytics = MagicMock()

sys.modules["db"]                = _db_pkg
sys.modules["db.connection"]     = _db_conn
sys.modules["db.queries"]        = _db_queries
sys.modules["db.models"]         = _db_models
sys.modules["db.training_models"] = _db_training
sys.modules["db.analytics"]      = _db_analytics
sys.modules["db.insider_models"] = MagicMock()

# ── ML / Training ─────────────────────────────────────────────────────────────
# Stub training.train so `patch("training.train.predict", ...)` resolves.
# The default predict raises so decision_agent falls back to LLM automatically.
_training_pkg   = MagicMock()
_training_train = MagicMock()
_training_train.predict = MagicMock(side_effect=Exception("no model installed"))

sys.modules["training"]           = _training_pkg
sys.modules["training.train"]     = _training_train
sys.modules["training.features"]  = MagicMock()
sys.modules["training.backfill_ohlcv"]   = MagicMock()
sys.modules["training.backfill_news"]    = MagicMock()
sys.modules["training.backfill_reddit"]  = MagicMock()
sys.modules["training.backfill_earnings"] = MagicMock()
sys.modules["training.backtest"]  = MagicMock()
sys.modules["training.progress"]  = MagicMock()
sys.modules["training.backfill_insider"] = MagicMock()

_training_pkg.train = _training_train

# ── Optional / peripheral deps ────────────────────────────────────────────────
_stub(
    "praw",
    "google", "google.cloud", "google.cloud.bigquery", "google.auth",
    "sse_starlette", "sse_starlette.sse",
    "structlog",
    "vaderSentiment", "vaderSentiment.vaderSentiment",
    "xgboost",
    "sklearn", "sklearn.preprocessing", "sklearn.model_selection",
    "joblib",
    "httpx",
    "uvicorn",
    "fastapi",
    "websockets",
    "discord",
)
