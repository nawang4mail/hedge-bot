"""
Central settings — loaded once at import time.
All values are overridable via environment variables or .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Local LLM ────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # ── Brokerage ─────────────────────────────────────────────────────────────
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # ── Data feeds ────────────────────────────────────────────────────────────
    news_api_key: str = ""

    # ── Risk parameters (runtime-mutable via UI kill-switch / override) ───────
    max_position_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    max_portfolio_risk_pct: float = Field(default=0.20, ge=0.0, le=1.0)
    default_order_type: str = "limit"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql://hedge:hedge@localhost:5432/hedgebot"

    # ── Server ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── API authentication ────────────────────────────────────────────────────
    # Set a strong random value in .env.  When empty, auth is disabled (dev only).
    api_key: str = ""

    # ── Kill-switch (toggled at runtime, NOT persisted to .env) ──────────────
    trading_halted: bool = False


settings = Settings()
