"""
Download and align raw daily data.

Sources:
  - HG=F       COMEX copper futures
  - CLPUSD=X   CLP/USD FX rate
  - DX-Y.NYB   DXY US Dollar index
  - ^VIX        CBOE VIX (risk-off proxy)
  - BRLUSD=X   BRL/USD  (EM peer)
  - PENUSD=X   PEN/USD  (EM peer, also copper exporter)
  - MXNUSD=X   MXN/USD  (EM peer)
  - CL=F        WTI crude oil
  - GC=F        Gold
  - ^TNX        US 10-year Treasury yield
  - ^IPSA       Santiago Stock Exchange (Chilean equities)

Run directly to refresh:
    uv run python -m copper_clp.dataset
"""
import os
import json
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from dotenv import load_dotenv

from copper_clp.config import (
    ROOT, DATA_RAW, DATA_PROCESSED, COPPER_RAW, CLPUSD_RAW,
    DAILY_PANEL, START_DATE,
)

load_dotenv(ROOT / ".env")
AV_KEY  = os.getenv("ALPHA_VANTAGE_API_KEY")
AV_BASE = "https://www.alphavantage.co/query"


# ── Download helpers ───────────────────────────────────────────────────────

def download_yfinance(ticker: str, path: Path, start: str = "2000-01-01") -> pd.Series:
    """Download daily close from yfinance, cache to CSV, return close series."""
    if not path.exists():
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        df.to_csv(path)
    raw = pd.read_csv(path, index_col=0, header=[0, 1], parse_dates=True)
    return raw["Close"][ticker].dropna()


def fetch_av(params: dict, cache_path: Path, force: bool = False) -> dict:
    """GET from Alpha Vantage with file caching."""
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())
    params["apikey"] = AV_KEY
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()
    if "Note" in d or "Information" in d:
        raise RuntimeError(f"AV rate limit: {d.get('Note') or d.get('Information')}")
    cache_path.write_text(json.dumps(d))
    return d


def _to_month_period(s: pd.Series) -> pd.Series:
    s.index = s.index.to_period("M").to_timestamp()
    return s


def fetch_copper_monthly_av(force: bool = False) -> pd.Series:
    """Global copper price (USD/metric ton) monthly from Alpha Vantage — back to 1992."""
    cache = DATA_RAW / "commodity_copper_monthly.json"
    d = fetch_av({"function": "COPPER"}, cache, force)
    records = d.get("data", [])
    s = pd.Series(
        {r["date"]: float(r["value"]) for r in records if r["value"] != "."},
        name="copper_mt",
    )
    s.index = pd.to_datetime(s.index)
    return _to_month_period(s).sort_index()


# Tickers with full OHLC needed for same-day intraday features (v3+)
OHLC_TICKERS = [
    ("HG=F",     "copper", "copper_hgf_daily.csv"),
    ("DX-Y.NYB", "dxy",    "dxy_daily.csv"),
    ("^VIX",     "vix",    "vix_daily.csv"),
    ("BRLUSD=X", "brl",    "brl_daily.csv"),
    ("PENUSD=X", "pen",    "pen_daily.csv"),
    ("MXNUSD=X", "mxn",    "mxn_daily.csv"),
    ("SPY",      "spx",    "spx_daily.csv"),   # S&P 500 proxy — closes 5pm Santiago
    ("CLPUSD=X", "clp",    "clpusd_daily_full.csv"),
]

# Tickers for additional signals: (yfinance ticker, column name, csv filename)
EXTRA_TICKERS = [
    ("DX-Y.NYB", "dxy",   "dxy_daily.csv"),
    ("^VIX",     "vix",   "vix_daily.csv"),
    ("BRLUSD=X", "brl",   "brl_daily.csv"),
    ("PENUSD=X", "pen",   "pen_daily.csv"),
    ("MXNUSD=X", "mxn",   "mxn_daily.csv"),
    ("CL=F",     "oil",   "oil_daily.csv"),
    ("GC=F",     "gold",  "gold_daily.csv"),
    ("^TNX",     "us10y", "us10y_daily.csv"),
    ("^IPSA",    "ipsa",  "ipsa_daily.csv"),
]


# ── Public API ─────────────────────────────────────────────────────────────

def load_raw() -> tuple[pd.Series, pd.Series]:
    """Return (copper_close, clpusd_close) daily series, downloading if needed."""
    copper = download_yfinance("HG=F",     COPPER_RAW)
    clp    = download_yfinance("CLPUSD=X", CLPUSD_RAW)
    return copper, clp


def load_ohlc(ticker: str, col: str, fname: str, start: str = "2003-01-01") -> pd.DataFrame:
    """Load full OHLC for a ticker, downloading if needed. Returns open/close columns."""
    path = DATA_RAW / fname
    if not path.exists():
        print(f"Downloading {ticker} (OHLC)...")
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        df.to_csv(path)
    raw = pd.read_csv(path, index_col=0, header=[0, 1], parse_dates=True)
    out = pd.DataFrame({
        f"{col}_open":  raw["Open"][ticker],
        f"{col}_close": raw["Close"][ticker],
    }).dropna()
    return out


def load_same_day_ohlc() -> pd.DataFrame:
    """Load open and close for all tickers needed for same-day intraday features."""
    frames = []
    for ticker, col, fname in OHLC_TICKERS:
        try:
            frames.append(load_ohlc(ticker, col, fname))
        except Exception as e:
            print(f"  Warning: could not load OHLC for {col} ({ticker}): {e}")
    return pd.concat(frames, axis=1)


def load_extra_signals() -> pd.DataFrame:
    """Download and return all extra signal series as a DataFrame."""
    series = {}
    for ticker, col, fname in EXTRA_TICKERS:
        path = DATA_RAW / fname
        try:
            s = download_yfinance(ticker, path, start="2003-01-01")
            series[col] = s
        except Exception as e:
            print(f"  Warning: could not load {col} ({ticker}): {e}")
    return pd.DataFrame(series)


def build_daily_panel(force: bool = False) -> pd.DataFrame:
    """
    Build aligned daily panel and save to data/processed/daily_panel.csv.

    Core columns:
        copper    — HG=F close (USD/lb)
        clp_usd   — CLPUSD=X close (USD per CLP)
        usd_clp   — pesos per dollar (= 1/clp_usd)
        r_copper  — log return of copper
        r_usd_clp — log return of usd_clp (positive = CLP weakens)

    Extra signal columns (raw levels, log-returns computed in features.py):
        dxy, vix, brl, pen, mxn, oil, gold, us10y, ipsa

    Same-day OHLC columns (for v3 intraday features, NO lag needed — these
    are all observable before CLP closes at 5pm Santiago):
        copper_open, copper_close  — COMEX copper (closes 2pm Santiago)
        dxy_open, dxy_close        — DXY (continuous, 5pm NY close)
        vix_open, vix_close        — VIX (4pm NY)
        brl/pen/mxn _open/_close   — EM FX
        spx_open, spx_close        — SPY as S&P proxy (closes 5pm Santiago)
        clp_open, clp_close        — CLP/USD open and close (in CLP per USD)
        clp_gap                    — log(clp_open / clp_prev_close): overnight gap

    LAG NOTE: T-1 signals use .shift(1); same-day signals (v3 group) use NO shift.
    """
    if DAILY_PANEL.exists() and not force:
        return pd.read_csv(DAILY_PANEL, index_col=0, parse_dates=True)

    copper, clp = load_raw()
    usd_clp = 1.0 / clp

    df = pd.DataFrame({"copper": copper, "clp_usd": clp, "usd_clp": usd_clp})

    # Merge extra signals (T-1 features)
    extras = load_extra_signals()
    df = df.join(extras, how="left")

    # Merge same-day OHLC (v3 intraday features)
    ohlc = load_same_day_ohlc()
    df = df.join(ohlc, how="left")

    df = df.ffill().dropna(subset=["copper", "clp_usd"])
    df = df[df.index >= START_DATE].copy()

    df["r_copper"]  = np.log(df["copper"]).diff()
    df["r_usd_clp"] = np.log(df["usd_clp"]).diff()
    # Log returns for price-based extras (not VIX, us10y which are already rates)
    for col in ["dxy", "brl", "pen", "mxn", "oil", "gold", "ipsa"]:
        if col in df.columns:
            df[f"r_{col}"] = np.log(df[col]).diff()
    # VIX and us10y: simple differences (already in % / bps)
    for col in ["vix", "us10y"]:
        if col in df.columns:
            df[f"d_{col}"] = df[col].diff()

    # Same-day log-returns from OHLC (open→close on day T, no lag)
    # CLP columns are in CLP/USD (pesos per dollar) — higher = CLP weaker
    if "clp_open" in df.columns and "clp_close" in df.columns:
        clp_o = 1.0 / df["clp_open"]    # convert to USD/CLP then invert → CLP/USD
        clp_c = 1.0 / df["clp_close"]
        # Actually clp_open/close from OHLC are already CLPUSD=X (USD per CLP)
        # so 1/x gives pesos per dollar
        df["clp_open_usd"]  = 1.0 / df["clp_open"]
        df["clp_close_usd"] = 1.0 / df["clp_close"]
        df["clp_gap"]       = np.log(df["clp_open_usd"] / df["clp_close_usd"].shift(1))
        df["clp_otc"]       = np.log(df["clp_close_usd"] / df["clp_open_usd"])

    for col in ["copper", "dxy", "brl", "pen", "mxn", "spx"]:
        o_col, c_col = f"{col}_open", f"{col}_close"
        if o_col in df.columns and c_col in df.columns:
            df[f"{col}_T_ret"]  = np.log(df[c_col] / df[o_col])   # intraday T return
            df[f"{col}_T_gap"]  = np.log(df[o_col] / df[c_col].shift(1))  # overnight gap

    # VIX same-day change (close - open, levels not returns)
    if "vix_open" in df.columns and "vix_close" in df.columns:
        df["vix_T_chg"] = df["vix_close"] - df["vix_open"]
        df["vix_T_gap"] = df["vix_open"]  - df["vix_close"].shift(1)

    df = df.dropna(subset=["r_copper", "r_usd_clp"])

    df.to_csv(DAILY_PANEL)
    print(f"Panel: {df.index[0].date()} → {df.index[-1].date()}, {len(df)} rows, "
          f"{len(df.columns)} cols")
    return df


if __name__ == "__main__":
    df = build_daily_panel(force=True)
    print(df.tail(3))
