"""
Experiment 3: Feature engineering
Build a rich feature matrix from copper price for predicting CLP/USD direction.

Features:
  - Copper log-returns at lags 1,2,3,5,10,21
  - Copper momentum: rolling mean returns over 5,10,21,63 days
  - Copper volatility: rolling std of returns over 10,21,63 days
  - Copper RSI (14-day)
  - Copper price vs rolling SMA (5,21,63 day z-score)
  - Copper trend: slope of linear regression over 21 days
  - CLP auto-regressive features: lagged CLP returns 1,2,3,5
  - Day-of-week dummies

Target:
  - y_dir: sign of next-day CLP return (classification)
  - y_ret: next-day CLP log-return (regression)
  - y_5d: 5-day ahead CLP cumulative return
  - y_21d: 21-day ahead CLP cumulative return

Output: data/feature_matrix.csv, data/feature_matrix_meta.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress

df = pd.read_csv("data/daily_panel.csv", index_col=0, parse_dates=True)
df["log_copper"] = np.log(df["copper"])
df["log_usd_clp"] = np.log(df["usd_clp"])
df["r_copper"] = df["log_copper"].diff()
df["r_usd_clp"] = df["log_usd_clp"].diff()

feat = pd.DataFrame(index=df.index)

# ── Copper lagged returns ──────────────────────────────────────────────────
for lag in [1, 2, 3, 5, 10, 21]:
    feat[f"cu_ret_lag{lag}"] = df["r_copper"].shift(lag)

# ── Copper momentum (rolling mean of returns) ──────────────────────────────
for w in [5, 10, 21, 63]:
    feat[f"cu_mom{w}"] = df["r_copper"].rolling(w).mean().shift(1)

# ── Copper volatility ──────────────────────────────────────────────────────
for w in [10, 21, 63]:
    feat[f"cu_vol{w}"] = df["r_copper"].rolling(w).std().shift(1)

# ── Copper RSI (14-day) ────────────────────────────────────────────────────
def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

feat["cu_rsi14"] = rsi(df["copper"]).shift(1)

# ── Copper price z-score vs SMA ────────────────────────────────────────────
for w in [5, 21, 63]:
    sma = df["copper"].rolling(w).mean()
    std = df["copper"].rolling(w).std()
    feat[f"cu_zscore{w}"] = ((df["copper"] - sma) / std.replace(0, np.nan)).shift(1)

# ── Copper trend slope (21-day linear regression) ─────────────────────────
def rolling_slope(series, w=21):
    slopes = series.copy() * np.nan
    for i in range(w - 1, len(series)):
        y = series.iloc[i - w + 1 : i + 1].values
        if np.isnan(y).any():
            continue
        slope, *_ = linregress(np.arange(w), y)
        slopes.iloc[i] = slope
    return slopes

feat["cu_slope21"] = rolling_slope(df["log_copper"], 21).shift(1)

# ── Copper 5-day return ────────────────────────────────────────────────────
feat["cu_ret5d"] = df["log_copper"].diff(5).shift(1)
feat["cu_ret21d"] = df["log_copper"].diff(21).shift(1)

# ── CLP auto-regressive features ───────────────────────────────────────────
for lag in [1, 2, 3, 5]:
    feat[f"clp_ret_lag{lag}"] = df["r_usd_clp"].shift(lag)

# ── Day-of-week ────────────────────────────────────────────────────────────
for d in range(5):
    feat[f"dow_{d}"] = (df.index.dayofweek == d).astype(int)

# ── Targets ───────────────────────────────────────────────────────────────
feat["y_dir"] = np.sign(df["r_usd_clp"])           # -1, 0, +1
feat["y_ret"] = df["r_usd_clp"]                     # next-day return
feat["y_5d"]  = df["log_usd_clp"].diff(5).shift(-4) # 5-day ahead
feat["y_21d"] = df["log_usd_clp"].diff(21).shift(-20) # 21-day ahead

# Drop rows with NaN in features or y_ret
feat = feat.dropna(subset=["cu_ret_lag1", "cu_slope21", "y_ret"])
feat.to_csv("data/feature_matrix.csv")

meta = {
    "n_rows": len(feat),
    "n_features": len([c for c in feat.columns if not c.startswith("y_")]),
    "features": [c for c in feat.columns if not c.startswith("y_")],
    "targets": [c for c in feat.columns if c.startswith("y_")],
    "date_range": [str(feat.index[0].date()), str(feat.index[-1].date())],
}
Path("data/feature_matrix_meta.json").write_text(json.dumps(meta, indent=2))
print(f"Feature matrix: {feat.shape[0]} rows × {feat.shape[1]} cols")
print("Features:", meta["features"][:8], "...")
print("Saved data/feature_matrix.csv")
