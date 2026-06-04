"""
Implementation Agent — Execution Trader
=========================================
Translates a TradingSignal into a brokerage order via Alpaca.
This agent is INTENTIONALLY isolated from research/decision logic.

It receives ONE input: a validated TradingSignal.
It performs ONE action: place the order and monitor fill status.

The LLM is NOT used here — execution must be 100 % deterministic.
"""
from __future__ import annotations
import time
import json
from datetime import datetime, timezone
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order

from agents.state import AgentState, TradingSignal, ExecutionReport
from config import settings


# ── Alpaca client (lazy init so tests can mock without env vars) ───────────────

_alpaca_client: TradingClient | None = None

def _get_client() -> TradingClient:
    global _alpaca_client
    if _alpaca_client is None:
        _alpaca_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,   # always paper until production keys are set
        )
    return _alpaca_client


# ── Order helpers ─────────────────────────────────────────────────────────────

def _place_order(signal: TradingSignal) -> Order:
    client = _get_client()
    side   = OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL

    if signal.order_type == "limit" and signal.limit_price:
        req = LimitOrderRequest(
            symbol=signal.symbol,
            qty=signal.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=signal.limit_price,
        )
    else:
        req = MarketOrderRequest(
            symbol=signal.symbol,
            qty=signal.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
        )

    return client.submit_order(req)


def _poll_fill(order_id: str, timeout_s: int = 15) -> Order:
    """Poll order status for up to timeout_s seconds."""
    client = _get_client()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        order = client.get_order_by_id(order_id)
        if order.status in ("filled", "partially_filled", "canceled", "rejected"):
            return order
        time.sleep(1)
    return client.get_order_by_id(order_id)   # return whatever state it's in


def _calc_slippage(signal: TradingSignal, avg_price: float | None) -> float | None:
    if avg_price is None or signal.limit_price is None:
        return None
    return round(abs(avg_price - signal.limit_price) / signal.limit_price * 100, 4)


# ── Agent node ────────────────────────────────────────────────────────────────

def implementation_node(state: dict[str, Any]) -> dict[str, Any]:
    s    = AgentState(**state)
    logs = list(s.agent_logs)
    signal: TradingSignal = s.trading_signal

    logs.append({
        "agent": "implementation",
        "status": "processing",
        "ts": datetime.now(timezone.utc).isoformat(),
        "msg": f"Preparing to execute {signal.action} {signal.quantity} {signal.symbol}",
    })

    # ── Skip execution for HOLD or failed risk checks ─────────────────────
    if signal.action == "HOLD" or not signal.risk_checks_passed:
        report = ExecutionReport(
            status="skipped",
            message=signal.rationale or "Signal is HOLD — no order placed.",
        )
        logs.append({
            "agent": "implementation", "status": "completed",
            "ts": datetime.now(timezone.utc).isoformat(),
            "output": json.loads(report.model_dump_json()),
        })
        return {"execution_report": report, "agent_logs": logs}

    # ── Guard: keys must be configured ────────────────────────────────────
    if not settings.alpaca_api_key:
        report = ExecutionReport(
            status="rejected",
            message="Alpaca API keys not configured — skipping live execution.",
        )
        logs.append({"agent": "implementation", "status": "completed",
                     "output": json.loads(report.model_dump_json())})
        return {"execution_report": report, "agent_logs": logs}

    try:
        order = _place_order(signal)
        order = _poll_fill(str(order.id))

        avg_price = float(order.filled_avg_price) if order.filled_avg_price else None
        slippage  = _calc_slippage(signal, avg_price)

        status_map = {
            "filled": "filled",
            "partially_filled": "partial",
            "canceled": "rejected",
            "rejected": "rejected",
        }

        report = ExecutionReport(
            order_id=str(order.id),
            status=status_map.get(str(order.status), "submitted"),
            filled_qty=float(order.filled_qty or 0),
            avg_fill_price=avg_price,
            slippage_pct=slippage,
            message=f"Order {order.id} — {order.status}",
        )

        logs.append({
            "agent": "implementation",
            "status": "completed",
            "ts": datetime.now(timezone.utc).isoformat(),
            "output": json.loads(report.model_dump_json()),
        })

        return {"execution_report": report, "agent_logs": logs}

    except Exception as exc:
        report = ExecutionReport(status="rejected", message=str(exc))
        logs.append({"agent": "implementation", "status": "error", "msg": str(exc)})
        return {"execution_report": report, "error": str(exc), "agent_logs": logs}
