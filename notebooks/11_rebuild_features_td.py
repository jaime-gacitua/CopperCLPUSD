"""
Rebuild the feature matrix on CORRECTLY-DATED data (TwelveData MCF closes).

Background (2026-07-10): yfinance CLPUSD=X daily closes are dated one day off
(return stamped date T = market move of T-1), which injected same-day copper
co-movement into the old feature_matrix.csv as fake predictability. See the
warning block in reports/conclusions.md.

This script rebuilds a leaner matrix where every feature is strictly
observable at the decision time (day-D MCF close, ~13:45 Santiago):

  - USD/CLP series: TwelveData hourly, daily close = last bar with hour<=13
    (bar close ~14:00 Santiago, right after the MCF close).
  - Copper (HG=F settlements): TWO conventions saved side by side —
      cu*_same : settlement of day D   (available at decision only part of
                 the year, DST-dependent; ~live-quote proxy)
      cu*_lag  : settlement of day D-1 (always available — STRICT)
  - Aux series (DXY, VIX, oil, gold, US10Y, SPX, IPSA): all close AFTER the
    13:45 decision -> lagged one full day. yfinance FX crosses (BRL/PEN/MXN)
    are DROPPED (same dating defect as CLPUSD=X).
  - Target: y_ret = log(fx[D+1]/fx[D]) winsorized +-3%, y_dir.

Output: data/processed/feature_matrix_td.csv

Run:  uv run python notebooks/11_rebuild_features_td.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"

# ── USD/CLP: TwelveData hourly -> daily MCF close ──────────────────────────
h = pd.read_csv(RAW / "td_usdclp_hourly.csv")
tcol = [c for c in h.columns if "time" in c.lower() or "date" in c.lower()][0]
h[tcol] = pd.to_datetime(h[tcol])
h = h.set_index(tcol).sort_index()
mcf = h[h.index.hour <= 13].groupby(h[h.index.hour <= 13].index.date)["close"].last()
mcf.index = pd.to_datetime(mcf.index)
mcf = mcf[mcf.index.dayofweek < 5]                      # weekdays only
fx_ret = np.log(mcf / mcf.shift(1)).clip(-0.03, 0.03)

df = pd.DataFrame(index=mcf.index)
df["clp_ret"]      = fx_ret                             # day-D return, known at decision
for k in (1, 2, 3, 5):
    df[f"clp_ret_lag{k}"] = fx_ret.shift(k)
df["clp_vol5"]     = fx_ret.rolling(5).std()
df["clp_vol21"]    = fx_ret.rolling(21).std()
df["clp_mom5"]     = np.log(mcf / mcf.shift(5))
df["clp_mom21"]    = np.log(mcf / mcf.shift(21))
df["clp_zscore21"] = (mcf - mcf.rolling(21).mean()) / mcf.rolling(21).std()

# ── copper: HG=F daily settlements, both conventions ───────────────────────
cu = pd.read_csv(RAW / "copper_hgf_daily.csv", skiprows=[1, 2], index_col=0,
                 parse_dates=True)["Close"].astype(float).dropna()
cu_on_d = cu.reindex(df.index, method="ffill")          # settle of day D (or last before)
def cu_feats(s, tag):
    r1  = np.log(s / s.shift(1))
    out = pd.DataFrame({
        f"cu_ret1_{tag}":  r1,
        f"cu_ret5_{tag}":  np.log(s / s.shift(5)),
        f"cu_ret21_{tag}": np.log(s / s.shift(21)),
        f"cu_vol21_{tag}": r1.rolling(21).std(),
        f"cu_z21_{tag}":   (s - s.rolling(21).mean()) / s.rolling(21).std(),
    })
    return out
df = df.join(cu_feats(cu_on_d, "same"))                 # settle of day D
df = df.join(cu_feats(cu_on_d, "lag").shift(1))         # settle of day D-1 (STRICT)

# ── aux daily series (indices/futures, correctly dated), lagged 1 day ──────
AUX = {"dxy": "dxy_daily.csv", "vix": "vix_daily.csv", "oil": "oil_daily.csv",
       "gold": "gold_daily.csv", "us10y": "us10y_daily.csv",
       "spx": "spx_daily.csv", "ipsa": "ipsa_daily.csv"}
for tag, f in AUX.items():
    p = RAW / f
    if not p.exists():
        print(f"  (skip {tag}: {f} not found)"); continue
    s = pd.read_csv(p, skiprows=[1, 2], index_col=0, parse_dates=True)["Close"].astype(float).dropna()
    s = s.reindex(df.index, method="ffill")
    df[f"{tag}_ret1_lag"] = np.log(s / s.shift(1)).shift(1)
    df[f"{tag}_mom5_lag"] = np.log(s / s.shift(5)).shift(1)

# ── calendar ────────────────────────────────────────────────────────────────
for d in range(5):
    df[f"dow_{d}"] = (df.index.dayofweek == d).astype(int)

# ── target ──────────────────────────────────────────────────────────────────
df["y_ret"] = fx_ret.shift(-1)
df["y_dir"] = (df["y_ret"] > 0).astype(int)

df = df.dropna(subset=["y_ret", "clp_ret", "cu_ret1_lag"])
out = PROC / "feature_matrix_td.csv"
df.to_csv(out)
print(f"Saved {out}: {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()}), "
      f"{df.shape[1]} cols, {df['y_dir'].mean():.1%} up-days")
