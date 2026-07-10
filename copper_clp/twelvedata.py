"""
TwelveData downloader — USD/CLP hourly bars.

Fetches the full history available on Basic plan (2019-09-22 onwards) in
paginated batches of 5000 bars, respecting the 8-credit/min rate limit.

Saved to: data/raw/td_usdclp_hourly.csv
Columns : datetime (Santiago time, America/Santiago), open, high, low, close

Run:
    uv run python -m copper_clp.twelvedata
    uv run python -m copper_clp.twelvedata --force   # re-download everything
"""
import os
import time
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

from copper_clp.config import ROOT, DATA_RAW

load_dotenv(ROOT / ".env")
API_KEY  = os.getenv("TWELVE_DATA_API_KEY")
BASE_URL = "https://api.twelvedata.com/time_series"

SYMBOL    = "USD/CLP"
INTERVAL  = "1h"
TIMEZONE  = "America/Santiago"
OUTPUTSIZE = 5000           # max per request on Basic plan
SLEEP_SEC  = 62             # seconds between batches (8 credits/min limit)
EARLIEST   = "2019-09-22"  # confirmed earliest for USD/CLP hourly on TwelveData

OUT_PATH   = DATA_RAW / "td_usdclp_hourly.csv"


def fetch_batch(end_date: str) -> list[dict]:
    """Fetch up to OUTPUTSIZE hourly bars ending at end_date (YYYY-MM-DD or datetime str)."""
    params = {
        "symbol":     SYMBOL,
        "interval":   INTERVAL,
        "outputsize": OUTPUTSIZE,
        "timezone":   TIMEZONE,
        "end_date":   end_date,
        "apikey":     API_KEY,
        "order":      "DESC",    # newest first — easier to paginate backwards
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()

    if d.get("status") == "error":
        raise RuntimeError(f"TwelveData error: {d.get('message')}")

    return d.get("values", [])


def load_existing() -> pd.DataFrame | None:
    """Load existing CSV if present, return sorted DataFrame or None."""
    if not OUT_PATH.exists():
        return None
    df = pd.read_csv(OUT_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime")
    return df


def bars_to_df(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col])
    return df.sort_values("datetime")


def download(force: bool = False) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY not set in .env")

    existing = load_existing()

    # Determine cutoff: skip dates we already have
    if existing is not None and not force:
        latest_dt = existing["datetime"].max()
        # Re-fetch last 7 days to catch any late fills
        fetch_from = latest_dt - timedelta(days=7)
        print(f"Existing data through {latest_dt.date()}. Fetching updates from {fetch_from.date()}...")
        new_bars = _fetch_range(start=fetch_from.strftime("%Y-%m-%d %H:%M:%S"),
                                end=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        if new_bars.empty:
            print("No new bars.")
            return existing
        combined = pd.concat([existing, new_bars]).drop_duplicates("datetime").sort_values("datetime")
        combined.to_csv(OUT_PATH, index=False)
        print(f"Updated: {len(combined)} total hourly bars → {OUT_PATH}")
        return combined
    else:
        print(f"Full download: {EARLIEST} → now")
        all_bars = _fetch_range(start=EARLIEST, end=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        if all_bars.empty:
            print("No data returned.")
            return all_bars
        all_bars.to_csv(OUT_PATH, index=False)
        print(f"Saved {len(all_bars)} hourly bars → {OUT_PATH}")
        return all_bars


def _fetch_range(start: str, end: str) -> pd.DataFrame:
    """Paginate backwards from end until we reach start, respecting rate limits."""
    start_dt = pd.Timestamp(start)
    cursor   = end          # end_date for next request

    all_dfs = []
    batch_num = 0

    while True:
        batch_num += 1
        print(f"  Batch {batch_num}: end_date={cursor[:16]} ...", end=" ", flush=True)

        if batch_num > 1:
            time.sleep(SLEEP_SEC)   # respect 8 credits/min

        bars = fetch_batch(end_date=cursor)
        if not bars:
            print("empty — done")
            break

        df = bars_to_df(bars)
        oldest_in_batch = df["datetime"].min()
        newest_in_batch = df["datetime"].max()

        # Trim to requested range
        df = df[df["datetime"] >= start_dt]
        all_dfs.append(df)
        print(f"got {len(df)} bars ({oldest_in_batch.date()} → {newest_in_batch.date()})")

        # Stop if we've reached or passed the start date
        if oldest_in_batch <= start_dt:
            break

        # Move cursor one second before oldest bar in this batch
        cursor = (oldest_in_batch - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs).drop_duplicates("datetime").sort_values("datetime")
    return result


def load() -> pd.DataFrame:
    """Load cached hourly data. Download if not present."""
    if not OUT_PATH.exists():
        print(f"{OUT_PATH} not found — downloading...")
        return download()
    df = pd.read_csv(OUT_PATH, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def mcf_daily(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Extract MCF session open and close from hourly bars.

    MCF session: 08:30–13:45 Santiago time.
    - 'open'  = open of the 08:00 Santiago bar (first bar of MCF session)
    - 'close' = close of the 13:00 Santiago bar (last full bar before 13:45 close)

    Returns a daily DataFrame with columns: date, mcf_open, mcf_close, mcf_ret
    aligned to Santiago calendar dates.
    """
    if df is None:
        df = load()

    df = df.copy()
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    # MCF open: 08:00 bar open price
    opens = (df[df["hour"] == 8]
             .groupby("date")["open"]
             .first()
             .rename("mcf_open"))

    # MCF close: 13:00 bar close price
    closes = (df[df["hour"] == 13]
              .groupby("date")["close"]
              .first()
              .rename("mcf_close"))

    daily = pd.DataFrame({"mcf_open": opens, "mcf_close": closes})
    daily.index = pd.to_datetime(daily.index)
    daily = daily.dropna()

    import numpy as np
    daily["mcf_ret"] = np.log(daily["mcf_close"] / daily["mcf_open"])

    return daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download USD/CLP hourly data from TwelveData")
    parser.add_argument("--force", action="store_true", help="Re-download full history")
    args = parser.parse_args()

    df = download(force=args.force)

    print()
    print(f"Date range : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"Total bars : {len(df)}")
    print()

    # Show MCF daily summary
    daily = mcf_daily(df)
    print(f"MCF daily rows (08:00 open + 13:00 close): {len(daily)}")
    print()
    print("Last 10 MCF sessions:")
    print(daily.tail(10)[["mcf_open", "mcf_close", "mcf_ret"]].round(4).to_string())

    # Quick data quality check: is mcf_open != prev_close meaningfully?
    import numpy as np
    df2 = df.copy()
    df2["hour"] = df2["datetime"].dt.hour
    prev_close_daily = (df2[df2["hour"] == 16]
                        .groupby(df2[df2["hour"] == 16]["datetime"].dt.date)["close"]
                        .last()
                        .rename("prev_session_close"))
    prev_close_daily.index = pd.to_datetime(prev_close_daily.index)

    check = daily.join(prev_close_daily.shift(1).rename("prev_close"))
    check = check.dropna()
    check["gap"] = np.log(check["mcf_open"] / check["prev_close"])
    check["otc"]  = check["mcf_ret"]

    print()
    print(f"Data quality (on {len(check)} days with prev session close):")
    print(f"  avg |gap|            : {check['gap'].abs().mean()*10000:.1f} bps")
    print(f"  corr(gap, mcf_ret)   : {check['gap'].corr(check['otc']):.3f}")
    print(f"  % days |gap| > 20bps : {(check['gap'].abs() > 0.002).mean():.0%}")
