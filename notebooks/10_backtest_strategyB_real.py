"""
Notebook 10: Strategy B backtest with real intraday data (TwelveData USD/CLP hourly).

This is the definitive Strategy B backtest, replacing the invalid estimates
from notebook 09 (which were built on contaminated yfinance open prices).

Data: TwelveData USD/CLP hourly bars, America/Santiago timezone.
      32,922 bars from 2019-09-22 to 2026-06-21 → 1,719 MCF daily sessions.

Key findings:
  - corr(gap, otc) = -0.091 (exists but weak, not -0.59 as previously estimated)
  - Strategy B gross Sharpe (|gap|>43bps, no cost) ≈ +2.4
  - At XTB spreads (6.5 bps RT): net Sharpe ≈ +0.9 (|gap|>43bps)
  - At 30 bps RT: net Sharpe ≈ -3.3 — NOT viable
  - Signal is time-varying: strongly negative 2019–2021, near-zero 2022–2023,
    slightly positive (wrong sign) 2024–2026 (regime shift or data artifact?)
  - The prior ~6.6 Sharpe estimate was purely an artifact of look-ahead bias.

Run: uv run python notebooks/10_backtest_strategyB_real.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from pathlib import Path

from copper_clp.config import MODELS_DIR, FIGURES_DIR
from copper_clp.twelvedata import load, mcf_daily

# ─── 1. Load data ────────────────────────────────────────────────────────────

print("Loading TwelveData USD/CLP hourly...")
td_hourly = load()
print(f"  {len(td_hourly):,} bars: {td_hourly['datetime'].iloc[0].date()} → {td_hourly['datetime'].iloc[-1].date()}")

daily = mcf_daily(td_hourly)
daily['prev_close'] = daily['mcf_close'].shift(1)
daily = daily.dropna()
daily['gap'] = np.log(daily['mcf_open'] / daily['prev_close'])
daily['otc'] = daily['mcf_ret']
daily.index = pd.to_datetime(daily.index)

print(f"  {len(daily)} MCF sessions: {daily.index[0].date()} → {daily.index[-1].date()}")

# Load yfinance close-to-close series for vol filter
clp_raw = pd.read_csv("data/raw/clpusd_daily.csv", header=[0,1], index_col=0, parse_dates=True)
closes = clp_raw["Close"].iloc[:, 0].astype(float).sort_index()
ret = np.log(closes / closes.shift(1))
rv21 = ret.rolling(21).std() * np.sqrt(252)
daily = daily.join(rv21.rename("rv21"), how="left")
daily["rv21"] = daily["rv21"].ffill()

# ─── 2. Signal analysis ───────────────────────────────────────────────────────

print("\n=== SIGNAL QUALITY ===")

corr_all   = daily["gap"].corr(daily["otc"])
avg_abs_gap = daily["gap"].abs().mean() * 10000
avg_abs_otc = daily["otc"].abs().mean() * 10000

print(f"  corr(gap, otc) all     : {corr_all:.4f}")
print(f"  avg |gap|              : {avg_abs_gap:.1f} bps")
print(f"  avg |otc|              : {avg_abs_otc:.1f} bps")
print(f"  % days |gap| > 20 bps : {(daily['gap'].abs() > 0.002).mean():.0%}")
print(f"  % days |gap| > 43 bps : {(daily['gap'].abs() > 0.0043).mean():.0%}")
print(f"  % days |gap| > 60 bps : {(daily['gap'].abs() > 0.006).mean():.0%}")

# Year-by-year
print("\n  Year-by-year corr(gap,otc) and gross Sharpe (|gap|>43bps):")
daily["year"] = daily.index.year
year_stats = []
for yr, g in daily.groupby("year"):
    n   = len(g)
    c   = g["gap"].corr(g["otc"])
    sub = g[g["gap"].abs() > 0.0043]
    if len(sub) > 10:
        gr  = -np.sign(sub["gap"]) * sub["otc"]
        sh  = gr.mean() / gr.std() * np.sqrt(252) if gr.std() > 0 else np.nan
    else:
        sh = np.nan
    year_stats.append({"year": yr, "n": n, "corr": c, "sharpe_43bps": sh})
    print(f"    {yr}: n={n:3d}  corr={c:+.3f}  gross_Sharpe(>43bps)={sh:+.2f}" if not np.isnan(sh) else
          f"    {yr}: n={n:3d}  corr={c:+.3f}  gross_Sharpe(>43bps)=N/A (too few)")

# ─── 3. Backtest results ──────────────────────────────────────────────────────

print("\n=== BACKTEST RESULTS ===")
COST_XTB  = 0.00065   # 6.5 bps RT (XTB Chile confirmed)
COST_30BP = 0.0030    # 30 bps RT

results = []
for gap_th in [0.000, 0.002, 0.0043, 0.006, 0.010]:
    for vol_th, vol_label in [(0.0, "any"), (0.10, "rv21>10%")]:
        for cost_name, cost in [("XTB 6.5bps", COST_XTB), ("30bps", COST_30BP)]:
            mask = (daily["gap"].abs() > gap_th) & (daily["rv21"].fillna(0) > vol_th)
            sub  = daily[mask].copy()
            if len(sub) < 30:
                continue
            sub["signal"] = -np.sign(sub["gap"])
            sub["gross"]  = sub["signal"] * sub["otc"]
            sub["net"]    = sub["gross"] - cost
            sharpe  = sub["net"].mean() / sub["net"].std() * np.sqrt(252)
            ann_ret = sub["net"].mean() * 252
            win     = (sub["gross"] > 0).mean()
            avg_g   = sub["gross"].mean() * 10000
            results.append({
                "gap_th_bps": gap_th * 10000,
                "vol_filter": vol_label,
                "cost": cost_name,
                "n_trades": len(sub),
                "net_sharpe": round(sharpe, 3),
                "ann_ret_pct": round(ann_ret * 100, 2),
                "avg_gross_bps": round(avg_g, 1),
                "win_rate": round(win, 3),
            })

for r in results:
    if r["cost"] == "XTB 6.5bps":
        print(f"  |gap|>{r['gap_th_bps']:.0f}bps {r['vol_filter']:10s}  "
              f"n={r['n_trades']:4d}  Sharpe={r['net_sharpe']:+.2f}  "
              f"avg_gross={r['avg_gross_bps']:.1f}bps  win={r['win_rate']:.1%}")

# ─── 4. Optimal configuration ─────────────────────────────────────────────────

# Best XTB config
xtb_results = [r for r in results if r["cost"] == "XTB 6.5bps"]
best = max(xtb_results, key=lambda x: x["net_sharpe"])
print(f"\n  Best XTB config: |gap|>{best['gap_th_bps']:.0f}bps {best['vol_filter']}  "
      f"→ Sharpe={best['net_sharpe']:+.2f}  n={best['n_trades']}")

# Equity curve for best config
mask   = (daily["gap"].abs() > best["gap_th_bps"] / 10000) & (daily["rv21"].fillna(0) > 0.0)
sub    = daily[mask].copy()
sub["signal"] = -np.sign(sub["gap"])
sub["net"]    = sub["signal"] * sub["otc"] - COST_XTB
sub["equity"] = sub["net"].cumsum()

# ─── 5. Plot ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

# (a) corr(gap,otc) scatter
ax0 = fig.add_subplot(gs[0, 0])
sub_all = daily[daily["gap"].abs() > 0.002].copy()
ax0.scatter(sub_all["gap"] * 10000, sub_all["otc"] * 10000,
            alpha=0.3, s=10, c="#2196F3")
m, b = np.polyfit(sub_all["gap"], sub_all["otc"], 1)
xs = np.linspace(sub_all["gap"].min(), sub_all["gap"].max(), 100)
ax0.plot(xs * 10000, (m * xs + b) * 10000, "r-", linewidth=2)
ax0.set_xlabel("Gap (bps)")
ax0.set_ylabel("OTC / MCF session return (bps)")
ax0.set_title(f"gap vs otc  (|gap|>20bps, n={len(sub_all)}, corr={sub_all['gap'].corr(sub_all['otc']):.3f})")
ax0.axhline(0, color="k", linewidth=0.5)
ax0.axvline(0, color="k", linewidth=0.5)

# (b) Year-by-year corr
ax1 = fig.add_subplot(gs[0, 1])
yrs  = [s["year"] for s in year_stats]
cors = [s["corr"] for s in year_stats]
cols = ["#F44336" if c > 0 else "#4CAF50" for c in cors]
ax1.bar(yrs, cors, color=cols, edgecolor="white")
ax1.axhline(0, color="k", linewidth=1)
ax1.set_title("corr(gap, otc) by year")
ax1.set_ylabel("Pearson correlation")
ax1.set_xlabel("Year")
ax1.set_ylim(-0.5, 0.35)
for y, c in zip(yrs, cors):
    ax1.text(y, c + (0.01 if c >= 0 else -0.03), f"{c:+.2f}", ha="center", fontsize=8)

# (c) Gross Sharpe by gap threshold
ax2 = fig.add_subplot(gs[1, 0])
gap_ths   = [0, 10, 20, 30, 43, 60, 80, 100]
sharpes_g = []
ns_g      = []
for gt in gap_ths:
    sub_g = daily[daily["gap"].abs() > gt / 10000]
    if len(sub_g) < 20:
        sharpes_g.append(np.nan)
        ns_g.append(0)
        continue
    gr = -np.sign(sub_g["gap"]) * sub_g["otc"]
    sh = gr.mean() / gr.std() * np.sqrt(252)
    sharpes_g.append(sh)
    ns_g.append(len(sub_g))

ax2.plot(gap_ths, sharpes_g, "o-", color="#2196F3", linewidth=2, markersize=6)
ax2.axhline(0, color="k", linewidth=0.5)
ax2.set_xlabel("|gap| threshold (bps)")
ax2.set_ylabel("Gross Sharpe (annualised)")
ax2.set_title("Gross Sharpe by gap filter (all 2019–2026)")
ax2b = ax2.twinx()
ax2b.bar(gap_ths, ns_g, alpha=0.2, color="grey", width=6)
ax2b.set_ylabel("N trades", color="grey")

# (d) Equity curve (best config)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(sub.index, sub["equity"] * 10000, color="#4CAF50", linewidth=1.5)
ax3.fill_between(sub.index, 0, sub["equity"] * 10000, alpha=0.2, color="#4CAF50")
ax3.set_title(f"Equity curve (Strategy B, |gap|>{best['gap_th_bps']:.0f}bps, XTB 6.5bps RT)\n"
              f"Net Sharpe={best['net_sharpe']:+.2f}, n={best['n_trades']}")
ax3.set_ylabel("Cumulative net return (bps)")
ax3.axhline(0, color="k", linewidth=0.5)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

# (e) Net Sharpe sensitivity table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis("off")

xtb_tbl = [[r["gap_th_bps"], r["vol_filter"], r["n_trades"],
            r["net_sharpe"], r["avg_gross_bps"], f"{r['win_rate']:.1%}"]
           for r in results if r["cost"] == "XTB 6.5bps" and r["vol_filter"] == "any"]

col_labels = ["|gap| threshold", "vol filter", "N trades", "Net Sharpe", "Avg gross (bps)", "Win rate"]
tbl = ax4.table(
    cellText=[[f"{row[0]:.0f} bps", row[1], str(row[2]),
               f"{row[3]:+.2f}", f"{row[4]:.1f}", row[5]] for row in xtb_tbl],
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
ax4.set_title("Strategy B — XTB 6.5 bps RT: net Sharpe by filter", fontsize=11, pad=10)

fig.suptitle("Strategy B (Fade-the-Gap) — Real Intraday Data (TwelveData, 2019–2026)\n"
             "Hourly USD/CLP — America/Santiago timezone",
             fontsize=13, fontweight="bold", y=0.98)

FIGURES_DIR.mkdir(exist_ok=True)
fig_path = FIGURES_DIR / "10_backtest_strategyB_real.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {fig_path}")
plt.close()

# ─── 6. Save summary ──────────────────────────────────────────────────────────

summary = {
    "notebook": "10_backtest_strategyB_real",
    "data_source": "TwelveData USD/CLP hourly (America/Santiago)",
    "data_range": f"{daily.index[0].date()} to {daily.index[-1].date()}",
    "n_mcf_sessions": len(daily),
    "corr_gap_otc_all": round(corr_all, 4),
    "avg_abs_gap_bps": round(avg_abs_gap, 1),
    "avg_abs_otc_bps": round(avg_abs_otc, 1),
    "best_xtb_config": best,
    "backtest_results": results,
    "year_stats": year_stats,
    "conclusion": (
        "Strategy B signal exists but is WEAK (corr=-0.091 vs. prior invalid estimate of -0.59). "
        "Best achievable net Sharpe at XTB spreads (6.5bps RT): +1.5 (|gap|>60bps, 191 trades/6yr). "
        "The gap-fade signal was strongest 2019-2021 and has deteriorated since 2022. "
        "At 30bps RT the strategy is NOT viable under any filter. "
        "XTB is the ONLY broker where Strategy B could be net-positive."
    ),
}

out_path = MODELS_DIR / "10_backtest_strategyB_real.json"
out_path.write_text(json.dumps(summary, indent=2, default=str))
print(f"Summary saved → {out_path}")

print("\n=== CONCLUSION ===")
print(summary["conclusion"])
