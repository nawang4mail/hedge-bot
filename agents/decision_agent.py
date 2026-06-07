"""
Decision Agent — Portfolio Manager
====================================
Synthesises the ResearchReport into a strictly formatted TradingSignal.

Signal source priority:
  1. Trained ML model (models/{symbol}_daily_vN.pkl) — if available
  2. Ollama local LLM — fallback when no model trained yet

Risk gates (position sizing, kill-switch, confidence threshold) are always
enforced regardless of signal source — deterministic code, not LLM logic.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState, TradingSignal, ResearchReport, MarketSnapshot
from agents.llm_router import get_llm
from config import settings

# ── System prompt ─────────────────────────────────────────────────────────────

DECISION_SYSTEM_PROMPT = """You are the Portfolio Manager at a quantitative hedge fund.
You receive a structured research report with technical indicators and market context.
Your sole responsibility is to decide: BUY, SELL, or HOLD.

STRICT OUTPUT FORMAT — respond with only this JSON, no other text:
{
  "action":     "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0–1.0>,
  "rationale":  "<single sentence, max 30 words>"
}

DECISION RULES:
- HOLD if confidence < 0.55 (uncertainty → inaction).
- HOLD if trend is "sideways" and no strong anomalies exist.
- BUY  only if trend is "uptrend" AND RSI < 65 AND MACD hist > 0.
- SELL only if trend is "downtrend" AND RSI > 35 AND MACD hist < 0.
- News sentiment below −0.5 overrides a BUY signal → HOLD.
- Never let rationale exceed 30 words.
- Do NOT include position size or price — those are calculated separately."""


# ── Risk sizing (deterministic, never LLM) ────────────────────────────────────

def _calculate_position_size(
    portfolio_value: float,
    price: float,
    atr: float | None,
) -> float:
    """
    Kelly-lite sizing: risk 1% of portfolio per trade, sized by ATR.
    Falls back to max_position_pct cap if ATR is unavailable.
    """
    risk_per_trade = portfolio_value * 0.01          # 1 % risk budget
    stop_distance  = (atr * 2) if atr else (price * 0.02)   # 2×ATR or 2 %
    raw_qty        = risk_per_trade / stop_distance
    max_qty        = (portfolio_value * settings.max_position_pct) / price
    return round(min(raw_qty, max_qty), 4)


# ── Agent node ────────────────────────────────────────────────────────────────

def decision_node(state: dict[str, Any]) -> dict[str, Any]:
    logs = list(state.get("agent_logs") or [])
    s = AgentState(**state)
    logs = list(s.agent_logs)

    logs.append({
        "agent": "decision",
        "status": "processing",
        "ts": datetime.now(timezone.utc).isoformat(),
        "msg": "Evaluating risk parameters and generating signal",
    })

    try:
        report: ResearchReport = s.research_report
        snapshot: MarketSnapshot = s.market_snapshot

        # ── Hard stop: trading halted via kill-switch ──────────────────────
        if settings.trading_halted:
            signal = TradingSignal(
                action="HOLD", symbol=snapshot.symbol, quantity=0.0,
                rationale="Kill-switch active — trading halted by operator.",
                confidence=1.0, risk_checks_passed=False,
            )
            logs.append({"agent": "decision", "status": "completed",
                         "output": json.loads(signal.model_dump_json())})
            return {**state, "trading_signal": signal, "agent_logs": logs}

        # ── Build flat feature dict for ML model ──────────────────────────
        feature_dict = {
            "rsi_14":          report.rsi_14 or 0.0,
            "rsi_7":           report.rsi_14 or 0.0,   # approximation
            "macd":            report.macd_signal or 0.0,
            "macd_signal":     report.macd_signal or 0.0,
            "macd_hist":       report.macd_hist or 0.0,
            "sma_50":          report.sma_50 or 0.0,
            "sma_200":         report.sma_200 or 0.0,
            "atr_14":          report.atr_14 or 0.0,
            "bb_upper":        report.bollinger_upper or 0.0,
            "bb_lower":        report.bollinger_lower or 0.0,
            "news_sentiment":  snapshot.news_sentiment,
            "news_article_count": 0.0,
            "reddit_mentions": 0.0,
            "reddit_sentiment": 0.0,
            "earnings_week":   0,
            "earnings_beat":   0,
            "eps_surprise_pct": 0.0,
            "volume_ratio":    1.0,
            "returns_1d":      0.0,
            "returns_5d":      0.0,
            "volatility_20":   0.0,
        }

        # ── Try ML model first ─────────────────────────────────────────────
        ml_result = None
        try:
            from training.train import predict as ml_predict
            ml_result = ml_predict(snapshot.symbol, feature_dict, timeframe="daily")
        except Exception:
            pass

        if ml_result:
            action     = ml_result["action"]
            confidence = ml_result["confidence"]
            rationale  = f"ML model ({ml_result['source']}) — confidence {confidence:.0%}"
            logs.append({
                "agent": "decision", "status": "processing",
                "ts": datetime.now(timezone.utc).isoformat(),
                "msg": f"Using ML model: {ml_result['source']}",
                "probabilities": ml_result.get("probabilities"),
            })
        else:
            # ── Fallback: LLM ─────────────────────────────────────────────
            logs.append({
                "agent": "decision", "status": "processing",
                "ts": datetime.now(timezone.utc).isoformat(),
                "msg": "No ML model found — falling back to LLM",
            })
            llm = get_llm(temperature=0.05)
            payload = json.loads(report.model_dump_json())
            payload["news_sentiment"] = snapshot.news_sentiment

            response = llm.invoke([
                SystemMessage(content=DECISION_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, default=str)),
            ])

            llm_json   = json.loads(response.content)
            action     = llm_json.get("action", "HOLD").upper()
            confidence = float(llm_json.get("confidence", 0.0))
            rationale  = str(llm_json.get("rationale", ""))[:200]

        # ── Enforce confidence gate ────────────────────────────────────────
        if confidence < 0.55:
            action = "HOLD"

        # ── Calculate quantity (only needed for BUY/SELL) ─────────────────
        # Fetch portfolio value — in paper mode, approximate 100 000 USD
        portfolio_value = 100_000.0   # TODO: replace with live Alpaca query
        qty = 0.0
        if action in ("BUY", "SELL"):
            qty = _calculate_position_size(
                portfolio_value=portfolio_value,
                price=snapshot.price,
                atr=report.atr_14,
            )
            if qty <= 0:
                action = "HOLD"

        limit_price = round(snapshot.ask * 1.001, 4) if action == "BUY" else \
                      round(snapshot.bid * 0.999, 4) if action == "SELL" else None

        signal = TradingSignal(
            action=action,
            symbol=snapshot.symbol,
            quantity=qty,
            order_type=settings.default_order_type,
            limit_price=limit_price,
            rationale=rationale,
            confidence=confidence,
            risk_checks_passed=True,
        )

        logs.append({
            "agent": "decision",
            "status": "completed",
            "ts": datetime.now(timezone.utc).isoformat(),
            "output": json.loads(signal.model_dump_json()),
        })

        return {**state, "trading_signal": signal, "agent_logs": logs}

    except Exception as exc:
        logs.append({"agent": "decision", "status": "error", "msg": str(exc)})
        return {**state, "error": str(exc), "agent_logs": logs}
