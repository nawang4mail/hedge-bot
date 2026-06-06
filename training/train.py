"""
Model Training — XGBoost classifier per ticker per timeframe.

Trains on feature matrix from features.py.
Saves model + scaler to models/ and registers metadata in DB.

Usage:
  python -m training.train --symbols AAPL TSLA --timeframe daily
  python -m training.train --symbols AAPL --timeframe hourly
  python -m training.train --symbols AAPL TSLA --timeframe both
"""
from __future__ import annotations
import argparse
import asyncio
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sqlalchemy.dialects.postgresql import insert as pg_insert

from training.features import build_features
from training.progress import ProgressEmitter
from db.connection import AsyncSessionLocal, init_db
from db.training_models import ModelRegistry

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")


def _validate_symbol(symbol: str) -> str:
    if not _TICKER_RE.match(symbol):
        raise ValueError(f"Invalid ticker symbol: {symbol!r}")
    return symbol

# Features used for training (must match features.py output)
FEATURE_COLS = [
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "atr_14", "adx_14",
    "bb_width", "bb_pct",
    "stoch_k", "stoch_d",
    "returns_1d", "returns_5d", "returns_20d", "volatility_20",
    "volume_ratio",
    "close_vs_sma50", "close_vs_sma200", "close_vs_bb_mid",
    "news_sentiment", "news_article_count",
    "reddit_mentions", "reddit_sentiment",
    "earnings_week", "earnings_beat", "eps_surprise_pct",
    "insider_buy_30d", "insider_sell_30d",
    "insider_net_shares_30d", "insider_buy_value_30d",
]

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


# ── Train ─────────────────────────────────────────────────────────────────────

async def train_model(
    symbol: str,
    timeframe: Literal["daily", "hourly"],
    years: int = 5,
    forward_candles: int = 5,
    label_threshold: float = 0.02,
    emitter: ProgressEmitter | None = None,
) -> dict:
    _validate_symbol(symbol)
    N_ESTIMATORS = 300

    if emitter:
        await emitter.phase_start(f"Training {symbol} ({timeframe})",
                                   total_tickers=1, sources=[timeframe])
        await emitter.ticker_start(symbol, total=N_ESTIMATORS,
                                    unit="trees", source=timeframe)
        await emitter.log(f"Building feature matrix for {symbol} ({timeframe})…")
    else:
        print(f"\n{'='*50}\nTraining {symbol} — {timeframe.upper()}\n{'='*50}")

    # Build features
    df = await build_features(symbol, timeframe, years, forward_candles, label_threshold)

    if emitter:
        await emitter.log(f"{symbol}: {len(df)} rows, {df.shape[1]} features after engineering")

    # Select features — only keep columns that exist
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df["label"].astype(int).values

    # Time-series split (no data leakage)
    tss    = TimeSeriesSplit(n_splits=5)
    splits = list(tss.split(X))
    train_idx, test_idx = splits[-1]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if emitter:
        await emitter.log(
            f"{symbol}: train rows={len(train_idx)}, test rows={len(test_idx)} "
            f"| labels — SELL:{int((y_train==0).sum())} "
            f"HOLD:{int((y_train==1).sum())} BUY:{int((y_train==2).sum())}"
        )

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Class weights
    class_counts = np.bincount(y_train, minlength=3)
    total_c      = class_counts.sum()
    weights      = {i: total_c / (3 * c) if c > 0 else 1.0
                    for i, c in enumerate(class_counts)}

    # ── XGBoost with per-checkpoint progress emission ──────────────────────
    # XGBoost doesn't expose per-tree callbacks in sklearn API,
    # so we train in increments to emit realistic progress.
    CHECKPOINTS  = 10   # emit metrics every N_ESTIMATORS / CHECKPOINTS trees
    step         = max(1, N_ESTIMATORS // CHECKPOINTS)
    model        = None
    eval_results : dict = {}

    for checkpoint in range(step, N_ESTIMATORS + 1, step):
        model = XGBClassifier(
            n_estimators=checkpoint,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=weights.get(2, 1.0),
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        evals = model.evals_result()
        val_loss = evals.get("validation_0", {}).get("mlogloss", [None])[-1]
        acc      = round(accuracy_score(y_test, model.predict(X_test)), 4)

        if emitter:
            await emitter.ticker_progress(symbol, current=checkpoint,
                                           total=N_ESTIMATORS, unit="trees",
                                           detail=f"val_loss={val_loss:.4f}  acc={acc:.1%}")
            await emitter.metric(symbol, timeframe, epoch=checkpoint,
                                  total_epochs=N_ESTIMATORS,
                                  val_loss=val_loss, accuracy=acc)
        else:
            print(f"  [{checkpoint:3d}/{N_ESTIMATORS}] val_loss={val_loss:.4f}  acc={acc:.1%}")

    # Final evaluation
    train_acc = round(accuracy_score(y_train, model.predict(X_train)), 4)
    test_acc  = round(accuracy_score(y_test,  model.predict(X_test)),  4)
    report    = classification_report(y_test, model.predict(X_test),
                                      target_names=["SELL","HOLD","BUY"])

    if emitter:
        await emitter.log(
            f"✅ {symbol}/{timeframe} — train acc: {train_acc:.1%}  test acc: {test_acc:.1%}"
        )
    else:
        print(f"\nTrain accuracy : {train_acc:.1%}")
        print(f"Test accuracy  : {test_acc:.1%}")
        print(f"\n{report}")

    # Feature importance
    importances  = dict(zip(available, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    if emitter:
        await emitter.feature_importance(symbol, timeframe, top_features)
    else:
        print("Top features:", [(f, round(v,3)) for f, v in top_features])

    # Save files
    version    = _next_version(symbol, timeframe)
    model_path  = MODELS_DIR / f"{symbol}_{timeframe}_v{version}.pkl"
    scaler_path = MODELS_DIR / f"{symbol}_{timeframe}_scaler_v{version}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata JSON
    meta = {
        "symbol": symbol, "timeframe": timeframe, "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_accuracy": train_acc, "test_accuracy": test_acc,
        "features": available, "top_features": top_features,
        "label_map": LABEL_MAP,
        "forward_candles": forward_candles, "label_threshold": label_threshold,
        "model_path": str(model_path), "scaler_path": str(scaler_path),
        "train_rows": len(train_idx), "test_rows": len(test_idx),
    }
    meta_path = MODELS_DIR / f"{symbol}_{timeframe}_v{version}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # Register in DB
    await _register_model(meta, str(model_path), str(scaler_path), df.index[0], df.index[-1])

    if emitter:
        await emitter.ticker_done(symbol,
                                   rows_inserted=len(df),
                                   rows_total=len(df),
                                   source=timeframe)
        await emitter.phase_done(f"Training {symbol} ({timeframe})")
    else:
        print(f"\n✅ Model saved: {model_path}")
    return meta


def _next_version(symbol: str, timeframe: str) -> int:
    _validate_symbol(symbol)
    existing = list(MODELS_DIR.glob(f"{symbol}_{timeframe}_v*.pkl"))
    existing = [f for f in existing if "scaler" not in f.name]
    if not existing:
        return 1
    versions = [int(f.stem.split("_v")[-1]) for f in existing]
    return max(versions) + 1


async def _register_model(meta: dict, model_path: str, scaler_path: str,
                           train_start, train_end):
    async with AsyncSessionLocal() as db:
        stmt = pg_insert(ModelRegistry).values({
            "symbol":        meta["symbol"],
            "timeframe":     meta["timeframe"],
            "version":       meta["version"],
            "file_path":     model_path,
            "scaler_path":   scaler_path,
            "features":      meta["features"],
            "train_start":   train_start,
            "train_end":     train_end,
            "train_accuracy": meta["train_accuracy"],
            "test_accuracy":  meta["test_accuracy"],
            "is_active":     True,
        }).on_conflict_do_update(
            constraint="uq_model_sym_tf_ver",
            set_={"train_accuracy": meta["train_accuracy"],
                  "test_accuracy": meta["test_accuracy"],
                  "file_path": model_path}
        )
        await db.execute(stmt)
        await db.commit()


# ── Predict helper (used by Decision Agent) ───────────────────────────────────

def load_model(symbol: str, timeframe: str = "daily"):
    """Load the latest active model for a symbol."""
    _validate_symbol(symbol)
    _models_root = MODELS_DIR.resolve()
    pattern = list(MODELS_DIR.glob(f"{symbol}_{timeframe}_v*.pkl"))
    pattern = [
        f for f in pattern
        if "scaler" not in f.name and f.resolve().is_relative_to(_models_root)
    ]
    if not pattern:
        return None, None, None
    latest = max(pattern, key=lambda f: int(f.stem.split("_v")[-1]))
    version = int(latest.stem.split("_v")[-1])
    scaler_path = MODELS_DIR / f"{symbol}_{timeframe}_scaler_v{version}.pkl"

    with open(latest, "rb") as f:
        model = pickle.load(f)
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    meta_path = MODELS_DIR / f"{symbol}_{timeframe}_v{version}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, scaler, meta


def predict(symbol: str, features: dict, timeframe: str = "daily") -> dict | None:
    """
    Given a dict of feature values, return {action, confidence, source}.
    Returns None if no model is available for this ticker.
    """
    model, scaler, meta = load_model(symbol, timeframe)
    if model is None:
        return None

    feat_cols = meta.get("features", FEATURE_COLS)
    row = np.array([[features.get(c, 0.0) for c in feat_cols]])

    if scaler:
        row = scaler.transform(row)

    proba     = model.predict_proba(row)[0]   # [P(SELL), P(HOLD), P(BUY)]
    pred_idx  = int(np.argmax(proba))
    action    = LABEL_MAP[pred_idx]
    confidence = round(float(proba[pred_idx]), 4)

    return {
        "action":     action,
        "confidence": confidence,
        "probabilities": {
            "SELL": round(float(proba[0]), 4),
            "HOLD": round(float(proba[1]), 4),
            "BUY":  round(float(proba[2]), 4),
        },
        "source": f"ml_model_{timeframe}_v{meta.get('version', 1)}",
        "model_test_accuracy": meta.get("test_accuracy"),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main(symbols: list[str], timeframe: str, years: int):
    await init_db()
    timeframes = ["daily", "hourly"] if timeframe == "both" else [timeframe]
    for sym in symbols:
        for tf in timeframes:
            try:
                await train_model(sym, tf, years)
            except Exception as e:
                print(f"✖ Failed {sym}/{tf}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",   nargs="+", required=True)
    parser.add_argument("--timeframe", choices=["daily","hourly","both"], default="daily")
    parser.add_argument("--years",     type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.symbols, args.timeframe, args.years))
