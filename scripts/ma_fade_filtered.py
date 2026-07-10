"""Trend-filtered MA-deviation fade for USDCLP.

Base: z = (close - SMA_n)/std_n; enter |z|>thr, exit z=0, 8% stop,
next-day-close fills, 40 bps RT.

Filter (entry-only): slope = SMA200(t)/SMA200(t-20) - 1.
  Long allowed  only if slope >= -eps  (don't catch falling knives vs trend)
  Short allowed only if slope <= +eps
eps grid: 0.0 (strict trend agreement) and 0.005 (mild tolerance).

DISCIPLINE: grid selected on IS 2021-2024 only. Holdout 2025-2026 run ONCE
on the best IS config (only if IS Sharpe > 0).
"""
import pandas as pd
import numpy as np

CSV = "/sessions/adoring-loving-carson/mnt/CopperCLPUSD/mt5/usdclp_daily.csv"
COST_RT = 0.0040
STOP = 0.08

df = pd.read_csv(CSV, header=None, names=["date", "close"])
df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
px = df.set_index("date")["close"].sort_index()

sma200 = px.rolling(200).mean()
slope = sma200 / sma200.shift(20) - 1


def run(n, thr, eps, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sma = px.rolling(n).mean()
    sd = px.rolling(n).std()
    z = (px - sma) / sd

    idx = px.index
    pos, entry_px, entry_t = 0, None, None
    trades = []
    daily_ret = pd.Series(0.0, index=idx)

    mask = (idx >= start) & (idx <= end)
    ilocs = np.where(mask)[0]
    for i in ilocs:
        if i + 1 >= len(idx):
            break
        r = px.iloc[i + 1] / px.iloc[i] - 1
        zi, sl = z.iloc[i], slope.iloc[i]
        if np.isnan(zi) or np.isnan(sl):
            continue
        if pos != 0:
            daily_ret.iloc[i + 1] = pos * r
            pl = pos * (px.iloc[i] / entry_px - 1)
            reason = None
            if (pos == 1 and zi >= 0) or (pos == -1 and zi <= 0):
                reason = "z0"
            elif pl <= -STOP:
                reason = "stop"
            if reason:
                fill = px.iloc[i + 1]
                net = pos * (fill / entry_px - 1) - COST_RT
                trades.append(dict(entry=entry_t, exit=idx[i + 1], side=pos,
                                   entry_px=entry_px, exit_px=fill,
                                   net=net, reason=reason))
                daily_ret.iloc[i + 1] -= COST_RT
                pos = 0
        else:
            if zi > thr and sl <= eps:
                pos, entry_px, entry_t = -1, px.iloc[i + 1], idx[i + 1]
            elif zi < -thr and sl >= -eps:
                pos, entry_px, entry_t = 1, px.iloc[i + 1], idx[i + 1]

    if pos != 0:
        j = min(ilocs[-1] + 1, len(idx) - 1)
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
    return dict(n=n, thr=thr, eps=eps, total=total, sharpe=sharpe, mdd=mdd,
                ntrades=len(trades), trades=trades)


if __name__ == "__main__":
    IS = ("2021-01-01", "2024-12-31")
    OOS = ("2025-01-01", "2026-06-19")

    print("=== IN-SAMPLE 2021-2024, trend-filtered (net 40bps RT) ===")
    rows = []
    for n in [50, 100, 150, 200]:
        for thr in [1.5, 2.0, 2.5]:
            for eps in [0.0, 0.005]:
                r = run(n, thr, eps, *IS)
                rows.append(r)
                print(f"SMA{n:>3} thr{thr:.1f} eps{eps:.3f}: "
                      f"ret {r['total']*100:+7.2f}%  sharpe {r['sharpe']:+5.2f}  "
                      f"mdd {r['mdd']*100:6.2f}%  trades {r['ntrades']}")

    best = max(rows, key=lambda r: r["sharpe"])
    print(f"\nBest IS: SMA{best['n']} thr{best['thr']} eps{best['eps']} "
          f"(sharpe {best['sharpe']:+.2f}, ret {best['total']*100:+.2f}%)")

    if best["sharpe"] <= 0:
        print("IS Sharpe <= 0 -> strategy NOT validated; skipping holdout.")
    else:
        print("\n=== HOLDOUT 2025-01-01..2026-06-19 (single run) ===")
        r = run(best["n"], best["thr"], best["eps"], *OOS)
        print(f"ret {r['total']*100:+.2f}%  sharpe {r['sharpe']:+.2f}  "
              f"mdd {r['mdd']*100:.2f}%  trades {r['ntrades']}")
        for t in r["trades"]:
            print(f"  {'L' if t['side']==1 else 'S'} {t['entry'].date()} "
                  f"@{t['entry_px']:.1f} -> {t['exit'].date()} "
                  f"@{t['exit_px']:.1f} net {t['net']*100:+.2f}% ({t['reason']})")
