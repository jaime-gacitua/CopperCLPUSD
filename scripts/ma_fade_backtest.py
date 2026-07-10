"""MA-deviation fade backtest for USDCLP (Capitaria daily closes).

Strategy: z = (close - SMA_n) / rolling_std_n
  short USDCLP when z > +thr, long when z < -thr
  exit when z crosses 0; 8% catastrophe stop; one position
Execution: signal on close of day t, fill at close of day t+1 (no lookahead).
Cost: 40 bps round trip (20 bps per side). No swap modeled.
"""
import pandas as pd
import numpy as np

CSV = "/sessions/adoring-loving-carson/mnt/CopperCLPUSD/mt5/usdclp_daily.csv"
COST_RT = 0.0040
STOP = 0.08

df = pd.read_csv(CSV, header=None, names=["date", "close"])
df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
df = df.set_index("date").sort_index()
px = df["close"]


def run(n, thr, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sma = px.rolling(n).mean()
    sd = px.rolling(n).std()
    z = (px - sma) / sd

    idx = px.index
    pos = 0          # +1 long USDCLP, -1 short
    entry_px = None
    trades = []
    daily_ret = pd.Series(0.0, index=idx)

    # iterate; decisions use info up to close[t], applied to return t+1
    mask = (idx >= start) & (idx <= end)
    ilocs = np.where(mask)[0]
    for i in ilocs:
        t = idx[i]
        if i + 1 >= len(idx):
            break
        # accrue return for existing position over next bar
        r = px.iloc[i + 1] / px.iloc[i] - 1
        zi = z.iloc[i]
        if np.isnan(zi):
            continue
        if pos != 0:
            daily_ret.iloc[i + 1] = pos * r
            # check exits at close t (using z at t) -> fill t+1
            cur = px.iloc[i]
            pl = pos * (cur / entry_px - 1)
            exit_reason = None
            if (pos == 1 and zi >= 0) or (pos == -1 and zi <= 0):
                exit_reason = "z0"
            elif pl <= -STOP:
                exit_reason = "stop"
            elif idx[i + 1] > end:
                exit_reason = "eop"
            if exit_reason:
                fill = px.iloc[i + 1]
                net = pos * (fill / entry_px - 1) - COST_RT
                trades.append(dict(entry=entry_t, exit=idx[i + 1],
                                   side=pos, entry_px=entry_px, exit_px=fill,
                                   net=net, reason=exit_reason))
                daily_ret.iloc[i + 1] -= COST_RT
                pos = 0
                entry_px = None
        else:
            if zi > thr:
                pos, entry_px, entry_t = -1, px.iloc[i + 1], idx[i + 1]
            elif zi < -thr:
                pos, entry_px, entry_t = 1, px.iloc[i + 1], idx[i + 1]

    # force-close open position at end
    if pos != 0:
        j = ilocs[-1] + 1 if ilocs[-1] + 1 < len(idx) else ilocs[-1]
        fill = px.iloc[j]
        net = pos * (fill / entry_px - 1) - COST_RT
        trades.append(dict(entry=entry_t, exit=idx[j], side=pos,
                           entry_px=entry_px, exit_px=fill, net=net,
                           reason="open"))
        daily_ret.iloc[j] -= COST_RT

    dr = daily_ret[(daily_ret.index >= start) & (daily_ret.index <= end)]
    total = (1 + dr).prod() - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    eq = (1 + dr).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(n=n, thr=thr, total=total, sharpe=sharpe, mdd=mdd,
                ntrades=len(trades), trades=trades)


if __name__ == "__main__":
    IS = ("2021-01-01", "2024-12-31")
    OOS = ("2025-01-01", "2026-06-19")

    print("=== IN-SAMPLE 2021-2024 (net 40bps RT) ===")
    rows = []
    for n in [50, 100, 150, 200]:
        for thr in [1.5, 2.0, 2.5]:
            r = run(n, thr, *IS)
            rows.append(r)
            print(f"SMA{n:>3} thr{thr:.1f}: ret {r['total']*100:+7.2f}%  "
                  f"sharpe {r['sharpe']:+5.2f}  mdd {r['mdd']*100:6.2f}%  "
                  f"trades {r['ntrades']}")

    best = max(rows, key=lambda r: r["sharpe"])
    print(f"\nBest IS by Sharpe: SMA{best['n']} thr{best['thr']}")

    print("\n=== HOLDOUT 2025-01-01..2026-06-19, best IS config ===")
    r = run(best["n"], best["thr"], *OOS)
    print(f"ret {r['total']*100:+.2f}%  sharpe {r['sharpe']:+.2f}  "
          f"mdd {r['mdd']*100:.2f}%  trades {r['ntrades']}")
    for t in r["trades"]:
        print(f"  {'L' if t['side']==1 else 'S'} {t['entry'].date()} "
              f"@{t['entry_px']:.1f} -> {t['exit'].date()} @{t['exit_px']:.1f} "
              f"net {t['net']*100:+.2f}% ({t['reason']})")

    print("\n=== HOLDOUT for ALL grid configs (robustness view) ===")
    for n in [50, 100, 150, 200]:
        for thr in [1.5, 2.0, 2.5]:
            r = run(n, thr, *OOS)
            print(f"SMA{n:>3} thr{thr:.1f}: ret {r['total']*100:+7.2f}%  "
                  f"trades {r['ntrades']}")
