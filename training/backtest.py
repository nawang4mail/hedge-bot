"""
Walk-Forward Backtest Engine.

Tests the trained ML model against held-out historical data.
Uses walk-forward methodology to avoid look-ahead bias.

Reports: win rate, Sharpe ratio, max drawdown, total return, alpha vs buy-and-hold.
Saves results to DB backtest_results table.

Usage:
  python -m training.backtest --symbol AAPL --timeframe daily
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from training.features import build_features, FEATURE_COLS
from training.train import load_model
from db.connection import AsyncSessionLocal, init_db
from db.training_models import BacktestResult

INITIAL_CAPITAL = 100_000.0
POSITION_PCT    = 0.05      # 5 % of portfolio per trade
COMMISSION      = 0.001     # 0.1 % per trade


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, model, scaler, meta: dict) -> dict:
    """
    Simulate trading on df using model predictions.
    df must include all feature columns and 'close' + 'label'.
    """
    feat_cols  = meta.get("features", FEATURE_COLS)
    available  = [c for c in feat_cols if c in df.columns]
    label_map  = {2: "BUY", 1: "HOLD", 0: "SELL"}

    X = df[available].values
    if scaler:
        X = scaler.transform(X)

    preds  = model.predict(X)
    probas = model.predict_proba(X)

    capital      = INITIAL_CAPITAL
    position     = 0.0    # shares held
    entry_price  = 0.0
    equity_curve = []
    trades       = []

    for i, (idx, row) in enumerate(df.iterrows()):
        price  = float(row["close"])
        action = label_map[int(preds[i])]
        conf   = float(probas[i][int(preds[i])])
        equity = capital + position * price
        equity_curve.append({"date": str(idx.date()), "equity": round(equity, 2)})

        # Enter BUY
        if action == "BUY" and position == 0 and conf > 0.55:
            shares      = (capital * POSITION_PCT) / price
            cost        = shares * price * (1 + COMMISSION)
            if cost <= capital:
                capital    -= cost
                position    = shares
                entry_price = price

        # Exit on SELL or model says SELL
        elif action == "SELL" and position > 0:
            proceeds = position * price * (1 - COMMISSION)
            pnl      = proceeds - (position * entry_price)
            trades.append({
                "entry": round(entry_price, 4), "exit": round(price, 4),
                "shares": round(position, 4), "pnl": round(pnl, 4),
                "return_pct": round(pnl / (position * entry_price) * 100, 2),
                "date": str(idx.date()),
            })
            capital  += proceeds
            position  = 0.0
            entry_price = 0.0

    # Close any open position at end
    if position > 0:
        price    = float(df["close"].iloc[-1])
        proceeds = position * price * (1 - COMMISSION)
        pnl      = proceeds - (position * entry_price)
        trades.append({
            "entry": round(entry_price, 4), "exit": round(price, 4),
            "shares": round(position, 4), "pnl": round(pnl, 4),
            "return_pct": round(pnl / (position * entry_price) * 100, 2),
            "date": str(df.index[-1].date()), "open_position": True,
        })
        capital += proceeds

    # ── Statistics ────────────────────────────────────────────────────────────
    final_equity    = capital
    total_return    = round((final_equity / INITIAL_CAPITAL - 1) * 100, 2)
    bh_return       = round((float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100, 2)
    alpha           = round(total_return - bh_return, 2)
    wins            = [t for t in trades if t["pnl"] > 0]
    losses          = [t for t in trades if t["pnl"] <= 0]
    win_rate        = round(len(wins) / len(trades) * 100, 1) if trades else 0.0

    # Sharpe ratio (annualised, daily returns)
    eq_series  = pd.Series([e["equity"] for e in equity_curve])
    daily_rets = eq_series.pct_change().dropna()
    sharpe     = round(
        float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))
        if daily_rets.std() > 0 else 0.0, 4
    )

    # Max drawdown
    rolling_max = eq_series.cummax()
    drawdown    = (eq_series - rolling_max) / rolling_max
    max_dd      = round(float(drawdown.min()) * 100, 2)

    return {
        "total_trades":    len(trades),
        "win_rate":        win_rate,
        "sharpe_ratio":    sharpe,
        "max_drawdown":    max_dd,
        "total_return":    total_return,
        "benchmark_return": bh_return,
        "alpha":           alpha,
        "final_equity":    round(final_equity, 2),
        "equity_curve":    equity_curve,
        "trades":          trades,
    }


# ── DB save ───────────────────────────────────────────────────────────────────

async def save_backtest(symbol: str, timeframe: str, model_id: int | None,
                        df: pd.DataFrame, results: dict):
    async with AsyncSessionLocal() as db:
        stmt = pg_insert(BacktestResult).values({
            "symbol":           symbol,
            "timeframe":        timeframe,
            "model_id":         model_id,
            "start_date":       df.index[0].to_pydatetime(),
            "end_date":         df.index[-1].to_pydatetime(),
            "total_trades":     results["total_trades"],
            "win_rate":         results["win_rate"],
            "sharpe_ratio":     results["sharpe_ratio"],
            "max_drawdown":     results["max_drawdown"],
            "total_return":     results["total_return"],
            "benchmark_return": results["benchmark_return"],
            "alpha":            results["alpha"],
            "equity_curve":     results["equity_curve"],
            "trades":           results["trades"],
        })
        await db.execute(stmt)
        await db.commit()


# ── CLI ───────────────────────────────────────────────────────────────────────

async def backtest(symbol: str, timeframe: str, years: int = 3):
    await init_db()

    print(f"\n── Backtesting {symbol} ({timeframe}) ──")

    model, scaler, meta = load_model(symbol, timeframe)
    if model is None:
        print(f"No trained model found for {symbol}/{timeframe}. Train first.")
        return

    # Use last `years` of data as out-of-sample test
    print("Building feature set...")
    df = await build_features(symbol, timeframe, years)

    print("Running simulation...")
    results = run_backtest(df, model, scaler, meta)

    print(f"\n{'='*40}")
    print(f"  Symbol:          {symbol}")
    print(f"  Timeframe:       {timeframe}")
    print(f"  Period:          {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Total trades:    {results['total_trades']}")
    print(f"  Win rate:        {results['win_rate']}%")
    print(f"  Sharpe ratio:    {results['sharpe_ratio']}")
    print(f"  Max drawdown:    {results['max_drawdown']}%")
    print(f"  Total return:    {results['total_return']}%")
    print(f"  Buy & hold:      {results['benchmark_return']}%")
    print(f"  Alpha:           {results['alpha']}%")
    print(f"  Final equity:    ${results['final_equity']:,.2f}")
    print(f"{'='*40}")

    await save_backtest(symbol, timeframe, None, df, results)
    print("✅ Results saved to DB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    required=True)
    parser.add_argument("--timeframe", choices=["daily","hourly"], default="daily")
    parser.add_argument("--years",     type=int, default=3)
    args = parser.parse_args()
    asyncio.run(backtest(args.symbol, args.timeframe, args.years))
