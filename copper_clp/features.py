"""
Feature engineering for the copper → CLP/USD signal.

LAG POLICY (critical for live trading):
  On day T we observe:
    - Copper close at T-1 (yesterday's settlement, known before CLP market opens)
    - CLP/USD close at T-1

  We predict: direction of CLP/USD on day T (open-to-close or close-to-close).

  Implementation: every copper feature is computed on the copper series and
  then .shift(MIN_COPPER_LAG) before being used as a predictor. This ensures
  we only use information that was available BEFORE day T's trading session.

  MIN_COPPER_LAG = 1 → use yesterday's copper data to predict today's CLP.
  You can increase this (e.g. 2) to simulate a 2-day decision lag.
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress

from copper_clp.config import DAILY_PANEL, FEATURE_MATRIX, MIN_COPPER_LAG


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _rolling_slope(series: pd.Series, w: int = 21) -> pd.Series:
    slopes = series.copy() * np.nan
    arr = series.values
    for i in range(w - 1, len(arr)):
        y = arr[i - w + 1 : i + 1]
        if np.isnan(y).any():
            continue
        slope, *_ = linregress(np.arange(w), y)
        slopes.iloc[i] = slope
    return slopes


def _lag(series: pd.Series, lag: int = MIN_COPPER_LAG) -> pd.Series:
    """Shift a copper-derived series forward by `lag` days so it's tradeable."""
    return series.shift(lag)


def build_features(df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    """
    Build the feature matrix from the daily panel.

    All copper features are lagged by MIN_COPPER_LAG days (default 1) relative
    to the target date, so they represent information available before you need
    to place the trade.

    Parameters
    ----------
    df : optional pre-loaded daily panel; loaded from disk if None.
    save : write result to data/processed/feature_matrix.csv

    Returns
    -------
    DataFrame with features (X_*) and targets (y_*) aligned on the same date index.
    The date index represents the day on which you act (place the trade).
    """
    if df is None:
        df = pd.read_csv(DAILY_PANEL, index_col=0, parse_dates=True)

    log_copper  = np.log(df["copper"])
    log_usd_clp = np.log(df["usd_clp"])
    r_cu  = df["r_copper"]
    r_clp = df["r_usd_clp"]

    feat = pd.DataFrame(index=df.index)

    # ── Copper lagged returns (lag enforces no look-ahead) ─────────────────
    # cu_ret_lag1 = copper return on day T-1, known before trading on day T
    for extra_lag in [0, 1, 2, 4, 9, 20]:
        total_lag = MIN_COPPER_LAG + extra_lag
        feat[f"cu_ret_lag{total_lag}"] = _lag(r_cu, total_lag)

    # ── Copper momentum: rolling mean of returns (lagged) ──────────────────
    for w in [5, 10, 21, 63]:
        feat[f"cu_mom{w}"] = _lag(r_cu.rolling(w).mean())

    # ── Copper volatility ──────────────────────────────────────────────────
    for w in [10, 21, 63]:
        feat[f"cu_vol{w}"] = _lag(r_cu.rolling(w).std())

    # ── Copper RSI-14 ──────────────────────────────────────────────────────
    feat["cu_rsi14"] = _lag(_rsi(df["copper"]))

    # ── Copper price z-score vs SMA (mean-reversion signal) ────────────────
    for w in [5, 21, 63]:
        sma = df["copper"].rolling(w).mean()
        std = df["copper"].rolling(w).std().replace(0, np.nan)
        feat[f"cu_zscore{w}"] = _lag((df["copper"] - sma) / std)

    # ── Copper trend (21-day log-price slope) ──────────────────────────────
    feat["cu_slope21"] = _lag(_rolling_slope(log_copper, 21))

    # ── Multi-day copper returns (lagged) ──────────────────────────────────
    feat["cu_ret5d"]  = _lag(log_copper.diff(5))
    feat["cu_ret21d"] = _lag(log_copper.diff(21))

    # ── CLP auto-regressive features (own lags, also clean) ───────────────
    for lag in [1, 2, 3, 5]:
        feat[f"clp_ret_lag{lag}"] = r_clp.shift(lag)

    # ── DXY (USD index) — lag 1 ────────────────────────────────────────────
    # Strong dollar → EM currencies weaken; adds a second channel beyond copper
    if "r_dxy" in df.columns:
        r_dxy = df["r_dxy"]
        feat["dxy_ret_lag1"]  = r_dxy.shift(1)
        for w in [5, 21]:
            feat[f"dxy_mom{w}"]  = r_dxy.rolling(w).mean().shift(1)
            feat[f"dxy_vol{w}"]  = r_dxy.rolling(w).std().shift(1)

    # ── VIX (risk-off proxy) — lag 1 ───────────────────────────────────────
    # VIX spike → EM sell-off → CLP weakens
    if "vix" in df.columns:
        d_vix = df["d_vix"]
        feat["vix_chg_lag1"]  = d_vix.shift(1)
        feat["vix_level_lag1"] = df["vix"].shift(1)
        feat["vix_mom5"]      = d_vix.rolling(5).mean().shift(1)
        # VIX z-score vs 63-day mean (elevated vs normal)
        vix_sma = df["vix"].rolling(63).mean()
        vix_std = df["vix"].rolling(63).std().replace(0, np.nan)
        feat["vix_zscore63"]  = ((df["vix"] - vix_sma) / vix_std).shift(1)

    # ── EM peer currencies — lag 1 ─────────────────────────────────────────
    # BRL, PEN, MXN co-move with CLP; deviations from EM trend are mean-reverting
    em_pairs = [("r_brl", "brl"), ("r_pen", "pen"), ("r_mxn", "mxn")]
    em_rets  = []
    for col, name in em_pairs:
        if col in df.columns:
            feat[f"{name}_ret_lag1"] = df[col].shift(1)
            feat[f"{name}_mom5"]     = df[col].rolling(5).mean().shift(1)
            em_rets.append(df[col])

    # EM composite return (equal-weight average of available peers)
    if em_rets:
        em_avg = pd.concat(em_rets, axis=1).mean(axis=1)
        feat["em_composite_lag1"] = em_avg.shift(1)
        # CLP deviation from EM peers (mean-reversion signal)
        feat["clp_vs_em_lag1"]    = (r_clp - em_avg).shift(1)

    # ── Oil — lag 1 ────────────────────────────────────────────────────────
    # Chile imports all oil; high oil → wider current-account deficit → CLP weakens
    if "r_oil" in df.columns:
        feat["oil_ret_lag1"] = df["r_oil"].shift(1)
        feat["oil_mom21"]    = df["r_oil"].rolling(21).mean().shift(1)

    # ── Gold — lag 1 ───────────────────────────────────────────────────────
    # Gold/copper ratio as a risk-sentiment indicator
    if "r_gold" in df.columns:
        feat["gold_ret_lag1"] = df["r_gold"].shift(1)
        if "r_dxy" in df.columns:
            # Gold and copper often move together vs DXY; residual is informative
            feat["gold_vs_copper_lag1"] = (df["r_gold"] - r_cu).shift(1)

    # ── US 10Y yield — lag 1 ───────────────────────────────────────────────
    # Higher US rates → stronger USD → EM pressure
    if "d_us10y" in df.columns:
        feat["us10y_chg_lag1"]   = df["d_us10y"].shift(1)
        feat["us10y_mom21"]      = df["d_us10y"].rolling(21).mean().shift(1)
        if "us10y" in df.columns:
            feat["us10y_level_lag1"] = df["us10y"].shift(1)

    # ── IPSA (Chilean equities) — lag 1 ────────────────────────────────────
    # Domestic equity market leads / coincides with CLP moves
    if "r_ipsa" in df.columns:
        feat["ipsa_ret_lag1"] = df["r_ipsa"].shift(1)
        feat["ipsa_mom5"]     = df["r_ipsa"].rolling(5).mean().shift(1)

    # ── Cross-asset interaction features ───────────────────────────────────
    # Copper-DXY divergence: copper up but DXY also up is unusual (bullish copper
    # but dollar-strength headwind for CLP)
    if "r_dxy" in df.columns:
        feat["cu_dxy_spread_lag1"]  = (r_cu - df["r_dxy"]).shift(1)
    # VIX × copper: high-VIX copper drop is a bigger CLP risk than low-VIX
    if "vix" in df.columns:
        feat["cu_vix_interact_lag1"] = (r_cu * df["vix"] / 20).shift(1)

    # ── v3: Same-day intraday signals (NO shift — observable before T close) ──
    #
    # Decision workflow: you see ALL of these before placing order at ~5pm Santiago:
    #   - copper_T_ret : COMEX closes 1pm NY (2pm Santiago) — full day T return
    #   - dxy_T_ret    : DXY 5pm NY close — same time as CLP
    #   - vix_T_chg    : VIX 4pm NY close
    #   - brl/pen/mxn  : EM FX 5pm NY
    #   - spx_T_ret    : SPY closes 4pm NY (5pm Santiago) — risk-on/off confirmation
    #   - clp_gap      : overnight gap (T-1 close → T open) — known since morning
    #   - clp_otc      : CLP open→close SO FAR (partial, but full day by 5pm)

    # Copper same-day return (strongest same-day signal — COMEX settles 2h before CLP)
    if "copper_T_ret" in df.columns:
        feat["id_copper_T_ret"]    = df["copper_T_ret"]
        feat["id_copper_T_gap"]    = df["copper_T_gap"]
        # Same-day copper/DXY spread (net USD impact on copper)
        if "dxy_T_ret" in df.columns:
            feat["id_cu_dxy_T_spread"] = df["copper_T_ret"] - df["dxy_T_ret"]

    # DXY same-day
    if "dxy_T_ret" in df.columns:
        feat["id_dxy_T_ret"] = df["dxy_T_ret"]
        feat["id_dxy_T_gap"] = df["dxy_T_gap"]

    # VIX same-day
    if "vix_T_chg" in df.columns:
        feat["id_vix_T_chg"] = df["vix_T_chg"]
        feat["id_vix_T_gap"] = df["vix_T_gap"]

    # EM peer FX same-day returns
    em_T_rets = []
    for col in ["brl", "pen", "mxn"]:
        ret_col = f"{col}_T_ret"
        if ret_col in df.columns:
            feat[f"id_{col}_T_ret"] = df[ret_col]
            em_T_rets.append(df[ret_col])
    if em_T_rets:
        em_T_avg = pd.concat(em_T_rets, axis=1).mean(axis=1)
        feat["id_em_T_composite"] = em_T_avg
        # CLP gap relative to EM peers today
        if "clp_gap" in df.columns:
            feat["id_clp_vs_em_T"] = df["clp_gap"] - em_T_avg

    # S&P 500 same-day (closes 5pm Santiago — risk-on/off right before CLP close)
    if "spx_T_ret" in df.columns:
        feat["id_spx_T_ret"] = df["spx_T_ret"]
        feat["id_spx_T_gap"] = df["spx_T_gap"]

    # CLP overnight gap (T-1 close → T open) — observable since morning
    if "clp_gap" in df.columns:
        feat["id_clp_gap"] = df["clp_gap"]

    # NOTE: id_clp_otc (CLP open→close on day T) is intentionally excluded —
    # it equals y_ret - clp_gap, making it a component of the target (look-ahead leak).

    # ── Calendar features ─────────────────────────────────────────────────
    for d in range(5):
        feat[f"dow_{d}"] = (df.index.dayofweek == d).astype(int)

    # ── TARGETS ───────────────────────────────────────────────────────────
    # y_* represents what HAPPENS on day T (the day you trade).
    # Features represent what you KNEW before day T.
    feat["y_ret"]  = r_clp                                 # next-day CLP log-return
    feat["y_dir"]  = (r_clp > 0).astype(int)              # 1 = CLP weakens (USD/CLP rises)
    feat["y_5d"]   = log_usd_clp.diff(5).shift(-4)        # 5-day forward return
    feat["y_21d"]  = log_usd_clp.diff(21).shift(-20)      # 21-day forward return

    # Drop rows where key features or target are NaN
    feat = feat.dropna(subset=["cu_ret_lag1", "cu_slope21", "y_ret"])

    if save:
        feat.to_csv(FEATURE_MATRIX)
        print(f"Feature matrix: {feat.shape[0]} rows × {feat.shape[1]} cols")
        print(f"  Date range: {feat.index[0].date()} → {feat.index[-1].date()}")
        print(f"  Copper lag: MIN={MIN_COPPER_LAG} day(s) — NO look-ahead")

    return feat


def get_feature_cols(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns if not c.startswith("y_")]


if __name__ == "__main__":
    build_features()
