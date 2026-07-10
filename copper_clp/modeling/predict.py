"""
Signal generation for live/paper trading.

Loads the latest feature matrix, trains models on all available history,
and outputs a signal for the next trading day.

Usage:
    uv run python -m copper_clp.modeling.predict
"""
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from copper_clp.config import FEATURE_MATRIX, MODELS_DIR
from copper_clp.features import get_feature_cols


def generate_signal(feat: pd.DataFrame | None = None) -> dict:
    """
    Train on all available history and predict the signal for the NEXT trading day.

    Returns
    -------
    dict with keys:
        date           — last date in the feature matrix (the training anchor)
        signal         — ensemble vote: +1 (long USD/CLP) or -1 (short)
        signal_text    — human-readable
        model_votes    — per-model vote
        confidence     — fraction of models agreeing on the signal
        lag_policy     — reminder string
    """
    if feat is None:
        feat = pd.read_csv(FEATURE_MATRIX, index_col=0, parse_dates=True)

    FCOLS = get_feature_cols(feat)
    X_all = feat[FCOLS].values
    y_dir = (feat["y_ret"].values > 0).astype(int)

    # Remove NaN rows for training
    ok = ~np.isnan(X_all).any(axis=1)
    X_train = X_all[ok]
    y_train = y_dir[ok]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Latest row is our prediction input (copper features are already lagged)
    last_row = X_all[-1].reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)
    last_date = feat.index[-1]

    votes = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=0.1)
    lr.fit(X_scaled, y_train)
    votes["lr"] = int(lr.predict(last_row_scaled)[0])

    # XGBoost
    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                eval_metric="logloss", random_state=42, verbosity=0)
    xgb_m.fit(X_train, y_train)
    votes["xgb"] = int(xgb_m.predict(last_row)[0])

    # LightGBM
    lgb_m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 random_state=42, verbosity=-1)
    lgb_m.fit(X_train, y_train)
    votes["lgb"] = int(lgb_m.predict(last_row)[0])

    # Ensemble vote (majority)
    raw_votes = list(votes.values())
    long_votes = sum(raw_votes)
    total = len(raw_votes)
    ensemble = 1 if long_votes > total / 2 else 0
    signal = 1 if ensemble == 1 else -1
    confidence = max(long_votes, total - long_votes) / total

    signal_text = (
        "LONG USD/CLP (expect CLP to WEAKEN vs USD)"
        if signal == 1 else
        "SHORT USD/CLP (expect CLP to STRENGTHEN vs USD)"
    )

    result = {
        "date": str(last_date.date()),
        "signal": signal,
        "signal_text": signal_text,
        "model_votes": {m: ("LONG" if v == 1 else "SHORT") for m, v in votes.items()},
        "confidence": round(confidence, 2),
        "lag_policy": "Copper features are lagged ≥1 day — signal is tradeable on the NEXT business day",
    }

    out = MODELS_DIR / "latest_signal.json"
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    signal = generate_signal()
    print(f"\n{'='*60}")
    print(f"  Signal as of {signal['date']}")
    print(f"  {signal['signal_text']}")
    print(f"  Confidence: {signal['confidence']:.0%}")
    print(f"  Model votes: {signal['model_votes']}")
    print(f"\n  {signal['lag_policy']}")
    print(f"{'='*60}\n")
