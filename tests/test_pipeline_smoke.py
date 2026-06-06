"""
Smoke tests for the four-agent LangGraph pipeline.

All external I/O (yfinance, NewsAPI, TimescaleDB, Ollama, Alpaca) is mocked.
Tests verify state flow, error propagation, and key decision gates.

Run from the hedge-bot directory:
    pytest tests/ -v
"""
from __future__ import annotations
from unittest.mock import patch, MagicMock, call
import agents.implementation_agent as impl_module

import pytest

from agents.state import (
    AgentState,
    MarketSnapshot,
    ResearchReport,
    TradingSignal,
)

# ── Shared test data ──────────────────────────────────────────────────────────

SYMBOL = "AAPL"

# Pre-computed indicators returned by _compute_indicators mock in research tests.
_FAKE_INDICATORS = {
    "rsi_14": 55.0,
    "macd_signal": 0.45,
    "macd_hist": 0.18,
    "sma_50": 170.0,
    "sma_200": 160.0,
    "atr_14": 3.2,
    "bollinger_upper": 180.0,
    "bollinger_lower": 165.0,
    "volume_spike": False,
}


def _make_ohlcv(n: int = 60) -> list[dict]:
    """Generate n fake daily OHLCV rows with a gentle uptrend."""
    base = 150.0
    return [
        {
            "Date": f"2024-01-{i + 1:02d}" if i < 28 else f"2024-02-{i - 27:02d}",
            "Open":   round(base + i * 0.5, 4),
            "High":   round(base + i * 0.5 + 1.5, 4),
            "Low":    round(base + i * 0.5 - 0.5, 4),
            "Close":  round(base + i * 0.5 + 0.8, 4),
            "Volume": 1_000_000 + i * 5_000,
        }
        for i in range(n)
    ]


def _base_state(**overrides) -> dict:
    s = AgentState(run_id="test-run-001", symbol=SYMBOL)
    d = s.model_dump()
    d.update(overrides)
    return d


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol=SYMBOL,
        price=175.0,
        volume=1_200_000,
        bid=174.9,
        ask=175.1,
        spread=0.2,
        news_sentiment=0.1,
        news_headlines=["AAPL posts strong quarterly results"],
        ohlcv_1d=_make_ohlcv(60),
    )


def _report() -> ResearchReport:
    return ResearchReport(
        symbol=SYMBOL,
        **{k: v for k, v in _FAKE_INDICATORS.items()},
        trend="uptrend",
        analyst_summary="Moderate uptrend with healthy RSI and positive MACD.",
    )


# ── LLM mock helpers ──────────────────────────────────────────────────────────

def _research_llm(trend: str = "uptrend", summary: str = "Steady uptrend confirmed."):
    """Mock LLM that always returns a research JSON response."""
    m = MagicMock()
    m.invoke.return_value = MagicMock(
        content=f'{{"trend": "{trend}", "summary": "{summary}"}}'
    )
    return m


def _decision_llm(action: str = "BUY", confidence: float = 0.78,
                  rationale: str = "Uptrend confirmed."):
    """Mock LLM that always returns a decision JSON response."""
    m = MagicMock()
    m.invoke.return_value = MagicMock(
        content=f'{{"action": "{action}", "confidence": {confidence}, "rationale": "{rationale}"}}'
    )
    return m




def _fake_asyncio_run(coro):
    """asyncio.run replacement: closes the unused coroutine and returns fake OHLCV."""
    coro.close()  # suppress "coroutine was never awaited" warnings
    return _make_ohlcv(60)


# ── Observation Agent ─────────────────────────────────────────────────────────

class TestObservationNode:

    def test_happy_path_produces_snapshot(self):
        from agents.observation_agent import observation_node

        with (
            patch("agents.observation_agent._fetch_quote",
                  return_value={"price": 175.0, "volume": 1_200_000.0,
                                "bid": 174.9, "ask": 175.1}),
            patch("agents.observation_agent._fetch_news_sentiment",
                  return_value=(0.1, ["AAPL rises on earnings"])),
            patch("agents.observation_agent.asyncio.run", side_effect=_fake_asyncio_run),
        ):
            result = observation_node(_base_state())

        assert "error" not in result
        snap = result["market_snapshot"]
        assert snap.symbol == SYMBOL
        assert snap.price == 175.0
        assert snap.spread == pytest.approx(0.2, abs=1e-3)
        assert len(snap.ohlcv_1d) == 60

    def test_logs_contain_completed_status(self):
        from agents.observation_agent import observation_node

        with (
            patch("agents.observation_agent._fetch_quote",
                  return_value={"price": 100.0, "volume": 500_000.0,
                                "bid": 99.9, "ask": 100.1}),
            patch("agents.observation_agent._fetch_news_sentiment",
                  return_value=(0.0, [])),
            patch("agents.observation_agent.asyncio.run", side_effect=_fake_asyncio_run),
        ):
            result = observation_node(_base_state())

        statuses = {log["status"] for log in result["agent_logs"]}
        assert "completed" in statuses

    def test_error_sets_error_key_without_crashing(self):
        from agents.observation_agent import observation_node

        with patch("agents.observation_agent._fetch_quote",
                   side_effect=RuntimeError("yfinance timeout")):
            result = observation_node(_base_state())

        assert "error" in result
        assert "yfinance timeout" in result["error"]
        assert "market_snapshot" not in result

    def test_spread_calculated_from_bid_ask(self):
        from agents.observation_agent import observation_node

        with (
            patch("agents.observation_agent._fetch_quote",
                  return_value={"price": 200.0, "volume": 1_000_000.0,
                                "bid": 199.5, "ask": 200.5}),
            patch("agents.observation_agent._fetch_news_sentiment",
                  return_value=(0.0, [])),
            patch("agents.observation_agent.asyncio.run", side_effect=_fake_asyncio_run),
        ):
            result = observation_node(_base_state())

        assert result["market_snapshot"].spread == pytest.approx(1.0, abs=1e-3)


# ── Research Agent ────────────────────────────────────────────────────────────

class TestResearchNode:
    """
    _compute_indicators is mocked to avoid pandas_ta version sensitivity.
    These tests verify the LLM wiring, anomaly detection, and report structure.
    """

    def test_happy_path_produces_report(self):
        from agents.research_agent import research_node

        state = _base_state(market_snapshot=_snapshot())

        with (
            patch("agents.research_agent._compute_indicators",
                  return_value=_FAKE_INDICATORS.copy()),
            patch("agents.research_agent.get_llm", return_value=_research_llm()),
        ):
            result = research_node(state)

        assert "error" not in result
        report = result["research_report"]
        assert report.symbol == SYMBOL
        assert report.trend == "uptrend"
        assert report.rsi_14 == 55.0

    def test_indicators_populate_report_fields(self):
        from agents.research_agent import research_node

        state = _base_state(market_snapshot=_snapshot())

        with (
            patch("agents.research_agent._compute_indicators",
                  return_value=_FAKE_INDICATORS.copy()),
            patch("agents.research_agent.get_llm", return_value=_research_llm()),
        ):
            result = research_node(state)

        report = result["research_report"]
        assert report.rsi_14 == pytest.approx(55.0)
        assert report.macd_signal == pytest.approx(0.45)
        assert report.atr_14 == pytest.approx(3.2)

    def test_overbought_rsi_flagged_as_anomaly(self):
        from agents.research_agent import research_node

        indicators_overbought = {**_FAKE_INDICATORS, "rsi_14": 78.0}
        snap = MarketSnapshot(
            symbol=SYMBOL, price=220.0, volume=1_000_000,
            bid=219.9, ask=220.1, spread=0.2,
            news_sentiment=0.0, ohlcv_1d=_make_ohlcv(40),
        )

        with (
            patch("agents.research_agent._compute_indicators",
                  return_value=indicators_overbought),
            patch("agents.research_agent.get_llm", return_value=_research_llm()),
        ):
            result = research_node(_base_state(market_snapshot=snap))

        report = result["research_report"]
        assert any("overbought" in a.lower() for a in report.anomalies)

    def test_negative_sentiment_flagged_as_anomaly(self):
        from agents.research_agent import research_node

        bearish_snap = MarketSnapshot(
            symbol=SYMBOL, price=160.0, volume=800_000,
            bid=159.9, ask=160.1, spread=0.2,
            news_sentiment=-0.7,  # strongly negative
            ohlcv_1d=_make_ohlcv(40),
        )

        with (
            patch("agents.research_agent._compute_indicators",
                  return_value=_FAKE_INDICATORS.copy()),
            patch("agents.research_agent.get_llm", return_value=_research_llm()),
        ):
            result = research_node(_base_state(market_snapshot=bearish_snap))

        report = result["research_report"]
        assert any("sentiment" in a.lower() for a in report.anomalies)

    def test_error_when_snapshot_is_missing(self):
        from agents.research_agent import research_node

        with patch("agents.research_agent.get_llm", return_value=_research_llm()):
            result = research_node(_base_state())  # no market_snapshot

        assert "error" in result


# ── Decision Agent ────────────────────────────────────────────────────────────

class TestDecisionNode:

    def _state(self, **overrides):
        return _base_state(
            market_snapshot=_snapshot(),
            research_report=_report(),
            **overrides,
        )

    def test_llm_fallback_produces_valid_signal(self):
        from agents.decision_agent import decision_node

        with (
            patch("agents.decision_agent.get_llm", return_value=_decision_llm("BUY", 0.80)),
            patch("training.train.predict", side_effect=Exception("no model")),
        ):
            result = decision_node(self._state())

        assert "error" not in result
        signal = result["trading_signal"]
        assert signal.action in ("BUY", "SELL", "HOLD")
        assert 0.0 <= signal.confidence <= 1.0

    def test_ml_model_takes_priority_over_llm(self):
        from agents.decision_agent import decision_node

        ml_result = {
            "action": "SELL", "confidence": 0.88,
            "source": "xgb_daily_v1",
            "probabilities": {"BUY": 0.05, "SELL": 0.88, "HOLD": 0.07},
        }

        with patch("training.train.predict", return_value=ml_result):
            result = decision_node(self._state())

        signal = result["trading_signal"]
        assert signal.action == "SELL"
        assert signal.confidence == 0.88

    def test_confidence_below_threshold_forces_hold(self):
        from agents.decision_agent import decision_node

        with (
            patch("agents.decision_agent.get_llm", return_value=_decision_llm("BUY", 0.40)),
            patch("training.train.predict", side_effect=Exception("no model")),
        ):
            result = decision_node(self._state())

        assert result["trading_signal"].action == "HOLD"

    def test_kill_switch_always_returns_hold(self):
        from agents.decision_agent import decision_node
        from config import settings

        original = settings.trading_halted
        settings.trading_halted = True
        try:
            result = decision_node(self._state())
        finally:
            settings.trading_halted = original

        signal = result["trading_signal"]
        assert signal.action == "HOLD"
        assert not signal.risk_checks_passed

    def test_buy_signal_has_positive_quantity(self):
        from agents.decision_agent import decision_node

        with (
            patch("agents.decision_agent.get_llm", return_value=_decision_llm("BUY", 0.80)),
            patch("training.train.predict", side_effect=Exception("no model")),
        ):
            result = decision_node(self._state())

        signal = result["trading_signal"]
        if signal.action == "BUY":
            assert signal.quantity > 0

    def test_buy_signal_sets_limit_price_above_ask(self):
        from agents.decision_agent import decision_node

        with (
            patch("agents.decision_agent.get_llm", return_value=_decision_llm("BUY", 0.80)),
            patch("training.train.predict", side_effect=Exception("no model")),
        ):
            result = decision_node(self._state())

        signal = result["trading_signal"]
        if signal.action == "BUY":
            assert signal.limit_price is not None
            assert signal.limit_price > 175.0  # ask is 175.1


# ── Implementation Agent ──────────────────────────────────────────────────────

class TestImplementationNode:

    def _state_with_signal(
        self,
        action: str = "BUY",
        confidence: float = 0.8,
        risk_ok: bool = True,
    ) -> dict:
        signal = TradingSignal(
            action=action,
            symbol=SYMBOL,
            quantity=5.0,
            order_type="limit",
            limit_price=175.18 if action == "BUY" else 174.82,
            rationale="Smoke test signal.",
            confidence=confidence,
            risk_checks_passed=risk_ok,
        )
        return _base_state(
            market_snapshot=_snapshot(),
            research_report=_report(),
            trading_signal=signal,
        )

    def test_hold_signal_is_skipped_without_order(self):
        from agents.implementation_agent import implementation_node

        result = implementation_node(self._state_with_signal(action="HOLD"))
        assert result["execution_report"].status == "skipped"

    def test_failed_risk_check_is_skipped(self):
        from agents.implementation_agent import implementation_node

        result = implementation_node(self._state_with_signal(action="BUY", risk_ok=False))
        assert result["execution_report"].status == "skipped"

    def test_missing_alpaca_keys_returns_rejected(self):
        from agents.implementation_agent import implementation_node
        from config import settings

        original = settings.alpaca_api_key
        settings.alpaca_api_key = ""
        try:
            result = implementation_node(self._state_with_signal(action="BUY"))
        finally:
            settings.alpaca_api_key = original

        assert result["execution_report"].status == "rejected"

    def test_successful_order_fill(self):
        from agents.implementation_agent import implementation_node
        from config import settings

        mock_order = MagicMock()
        mock_order.id = "order-abc-123"
        mock_order.status = "filled"
        mock_order.filled_qty = 5.0
        mock_order.filled_avg_price = 175.20

        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        mock_client.get_order_by_id.return_value = mock_order

        original_key = settings.alpaca_api_key
        settings.alpaca_api_key = "fake-key"
        impl_module._alpaca_client = mock_client
        try:
            result = implementation_node(self._state_with_signal(action="BUY"))
        finally:
            settings.alpaca_api_key = original_key
            impl_module._alpaca_client = None

        report = result["execution_report"]
        assert report.status == "filled"
        assert report.filled_qty == 5.0
        assert report.order_id == "order-abc-123"
        assert report.avg_fill_price == pytest.approx(175.20)

    def test_slippage_calculated_on_filled_limit_order(self):
        from agents.implementation_agent import implementation_node
        from config import settings

        mock_order = MagicMock()
        mock_order.id = "order-slip-456"
        mock_order.status = "filled"
        mock_order.filled_qty = 5.0
        mock_order.filled_avg_price = 175.50  # filled above the 175.18 limit

        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        mock_client.get_order_by_id.return_value = mock_order

        original_key = settings.alpaca_api_key
        settings.alpaca_api_key = "fake-key"
        impl_module._alpaca_client = mock_client
        try:
            result = implementation_node(self._state_with_signal(action="BUY"))
        finally:
            settings.alpaca_api_key = original_key
            impl_module._alpaca_client = None

        report = result["execution_report"]
        assert report.slippage_pct is not None
        assert report.slippage_pct > 0


# ── Full pipeline (end-to-end) ────────────────────────────────────────────────

class TestPipelineEndToEnd:
    """
    Runs the full LangGraph graph with all external I/O mocked.
    Verifies that state flows through all four agents correctly.
    """

    def _run(self, decision_action: str = "BUY", decision_confidence: float = 0.78,
             decision_rationale: str = "Uptrend.", ml_result: dict | None = None):
        from agents.graph import run_pipeline

        ml_side_effect = None if ml_result else Exception("no model")

        with (
            patch("agents.observation_agent._fetch_quote",
                  return_value={"price": 175.0, "volume": 1_200_000.0,
                                "bid": 174.9, "ask": 175.1}),
            patch("agents.observation_agent._fetch_news_sentiment",
                  return_value=(0.1, ["AAPL rises"])),
            patch("agents.observation_agent.asyncio.run", side_effect=_fake_asyncio_run),
            patch("agents.research_agent._compute_indicators",
                  return_value=_FAKE_INDICATORS.copy()),
            patch("agents.research_agent.get_llm", return_value=_research_llm()),
            patch("agents.decision_agent.get_llm",
                  return_value=_decision_llm(decision_action, decision_confidence,
                                             decision_rationale)),
            patch("training.train.predict",
                  return_value=ml_result, side_effect=ml_side_effect),
        ):
            return run_pipeline(SYMBOL)

    def test_all_four_agents_produce_output(self):
        final = self._run()

        assert final.market_snapshot is not None
        assert final.research_report is not None
        assert final.trading_signal is not None
        assert final.execution_report is not None
        assert final.error is None

    def test_all_four_agents_logged(self):
        final = self._run()

        agent_names = {log["agent"] for log in final.agent_logs}
        assert agent_names == {"observation", "research", "decision", "implementation"}

    def test_hold_decision_skips_execution(self):
        final = self._run(decision_action="HOLD", decision_confidence=0.90,
                         decision_rationale="No clear signal.")

        assert final.trading_signal.action == "HOLD"
        assert final.execution_report.status == "skipped"

    def test_observation_error_aborts_pipeline(self):
        """Error in Observation sets state.error; downstream agents are never called."""
        from agents.graph import run_pipeline

        with patch("agents.observation_agent._fetch_quote",
                   side_effect=RuntimeError("yfinance down")):
            final = run_pipeline(SYMBOL)

        assert final.error is not None
        assert final.research_report is None
        assert final.trading_signal is None
        assert final.execution_report is None

    def test_ml_model_used_when_available(self):
        ml_result = {
            "action": "SELL", "confidence": 0.88,
            "source": "xgb_daily_v1",
            "probabilities": {"BUY": 0.04, "SELL": 0.88, "HOLD": 0.08},
        }
        final = self._run(ml_result=ml_result)

        assert final.trading_signal.action == "SELL"
        assert final.trading_signal.confidence == 0.88

    def test_run_id_propagates_through_state(self):
        from agents.graph import run_pipeline

        with (
            patch("agents.observation_agent._fetch_quote",
                  return_value={"price": 175.0, "volume": 1_200_000.0,
                                "bid": 174.9, "ask": 175.1}),
            patch("agents.observation_agent._fetch_news_sentiment",
                  return_value=(0.0, [])),
            patch("agents.observation_agent.asyncio.run", side_effect=_fake_asyncio_run),
            patch("agents.research_agent._compute_indicators",
                  return_value=_FAKE_INDICATORS.copy()),
            patch("agents.research_agent.get_llm", return_value=_research_llm()),
            patch("agents.decision_agent.get_llm", return_value=_decision_llm()),
            patch("training.train.predict", side_effect=Exception("no model")),
        ):
            final = run_pipeline(SYMBOL, run_id="fixed-run-id")

        assert final.run_id == "fixed-run-id"
        assert final.symbol == SYMBOL
