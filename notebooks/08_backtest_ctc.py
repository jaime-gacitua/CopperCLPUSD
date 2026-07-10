"""
Cost-adjusted backtest: CLOSE-TO-CLOSE execution.

Decision workflow:
  T-1 close  →  model sees all features (copper T-1, same-day intraday if v3)
  T-1 close  →  you enter the trade (buy or sell USD/CLP at T-1 close price)
  T close    →  you exit, collect the full close-to-close return

This is the recommended execution strategy because the model predicts
close-to-close returns, and ~50% of that variance is the overnight gap
(T-1 close → T open) which has already happened by the time T opens.
Entering at T open misses the gap entirely; entering at T-1 close captures it.

P&L per trade (fixed 1-unit notional):
  gross_ret  = signal × y_ret          (y_ret = log CTC return)
  cost       = spread_bps (round-trip)
  net_ret    = gross_ret - cost

Cost assumptions:
  CTC execution has one price crossing at T-1 close (entry) and one at T close
  (exit). Both are end-of-day liquid prints — typically tighter than intraday.
  Retail spot CLP/USD bid-ask: 2-5 pesos on ~930 CLP/USD ≈ 20-50 bps per side.
  We model three scenarios covering institutional to retail:
    optimistic : 10 bps RT  (~5 bps/side, institutional or tight broker)
    base       : 30 bps RT  (~15 bps/side, realistic retail)
    pessimistic: 60 bps RT  (~30 bps/side, wide spread or thin market)

Note: CTC costs are slightly higher than OTC (07_backtest.py used 10/20/40 bps)
because you hold overnight — some brokers charge a swap/financing fee.
Swap for USD/CLP (long USD): BCCh rate - Fed Funds ≈ 5.5% - 4.5% = +1%/yr ≈ +0.4 bps/day
This is favourable (carry is positive when long USD/CLP) so it is excluded from costs —
it acts as a bonus return, captured separately.

Run:
    uv run python notebooks/08_backtest_ctc.py
"""
import sys, json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from copper_clp.config import (
    FEATURE_MATRIX, MODELS_DIR, FIGURES_DIR,
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
)
from copper_clp.features import get_feature_cols

# ── Cost assumptions ───────────────────────────────────────────────────────
COST_SCENARIOS = {
    "optimistic":  0.10 / 100,   # 10 bps RT (institutional / tight broker)
    "base":        0.30 / 100,   # 30 bps RT (realistic retail, incl. overnight)
    "pessimistic": 0.60 / 100,   # 60 bps RT (wide spread, thin market)
}
CONFIDENCE_THRESHOLDS = [0.50, 0.52, 0.55, 0.57, 0.60, 0.63, 0.65, 0.70]

# Positive carry when long USD/CLP: (BCCh ~5.5%) - (Fed Funds ~4.5%) ≈ +1%/yr
CARRY_BPS_PER_DAY = 1.0 / 252 / 100   # ~0.4 bps/day, only applies when long USD/CLP

# ── Load feature matrix ────────────────────────────────────────────────────
feat = pd.read_csv(FEATURE_MATRIX, index_col=0, parse_dates=True)
FCOLS = get_feature_cols(feat)

# Clean data quality: CLPUSD=X has two confirmed bad ticks where the close was
# ~0.2 (= 5 pesos/dollar) instead of the real ~550-665 pesos/dollar.
# Dates: 2014-04-10, 2016-12-22 (and their mirror-image reversals the next day).
# These produce ~±4.6/4.9 log-return spikes, which destroy the equity curve.
# Winsorise at ±3% (the 99.5th percentile of clean daily moves) — this is a
# data cleaning step, NOT a signal filter; we are not hiding bad trades.
RETURN_CAP = 0.03   # 3% daily log-return cap
n_clipped = (feat["y_ret"].abs() > RETURN_CAP).sum()
feat["y_ret"] = feat["y_ret"].clip(-RETURN_CAP, RETURN_CAP)
if n_clipped > 0:
    print(f"Winsorised {n_clipped} extreme y_ret rows (|ret| > {RETURN_CAP*100:.0f}%) — bad ticks")

# y_ret IS the CTC return (log(usd_clp_T / usd_clp_T-1)) — the model target
r_ctc = feat["y_ret"].values
y_dir  = (r_ctc > 0).astype(int)

print(f"Feature matrix  : {len(feat)} rows, {len(FCOLS)} features")
print(f"CTC return std  : {feat['y_ret'].std()*100:.3f}%  (ann vol {feat['y_ret'].std()*np.sqrt(252)*100:.1f}%)")
print(f"CTC return mean : {feat['y_ret'].mean()*10000:.2f} bps/day")
print(f"% days CLP weak : {(r_ctc > 0).mean()*100:.1f}%")
print()

# ── Walk-forward with calibrated probabilities ─────────────────────────────
X_all = feat[FCOLS].values
dates  = feat.index
n      = len(X_all)

records = []

for t0 in range(WF_TRAIN_DAYS, n - WF_TEST_DAYS, WF_STEP_DAYS):
    X_tr = X_all[t0 - WF_TRAIN_DAYS : t0]
    y_tr = y_dir[t0 - WF_TRAIN_DAYS : t0]
    X_te = X_all[t0 : t0 + WF_TEST_DAYS]
    y_te = y_dir[t0 : t0 + WF_TEST_DAYS]
    r_te = r_ctc[t0 : t0 + WF_TEST_DAYS]

    ok_tr = ~np.isnan(X_tr).any(axis=1)
    ok_te = ~np.isnan(X_te).any(axis=1)
    X_tr, y_tr = X_tr[ok_tr], y_tr[ok_tr]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    # LightGBM — best model from v3 (acc 86.7%, Sharpe 11.02)
    base_lgb = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        num_leaves=31, random_state=42, n_jobs=-1, verbose=-1,
    )
    model = CalibratedClassifierCV(base_lgb, method="isotonic", cv=3)
    model.fit(X_tr_s, y_tr)
    proba_te = model.predict_proba(X_te_s)[:, 1]  # P(USD/CLP rises)

    for i, (ok, prob, r) in enumerate(zip(ok_te, proba_te, r_te)):
        if not ok:
            continue
        global_idx = t0 + i
        records.append({
            "date":       str(dates[global_idx].date()),
            "prob_up":    float(prob),
            "pred_dir":   1 if prob > 0.5 else 0,
            "actual_dir": int(y_te[i]),
            "r_ctc":      float(r),
        })

df_bt = pd.DataFrame(records)
df_bt["date"] = pd.to_datetime(df_bt["date"])
df_bt = df_bt.set_index("date").sort_index()
df_bt = df_bt[~df_bt.index.duplicated(keep="last")]

print(f"OOS predictions : {len(df_bt)} days")
print(f"Date range      : {df_bt.index[0].date()} → {df_bt.index[-1].date()}")
print(f"Base accuracy   : {(df_bt['pred_dir'] == df_bt['actual_dir']).mean():.3f}")
print()

# ── Sweep confidence thresholds × cost scenarios ──────────────────────────
results = {}
print(f"{'Thresh':>6}  {'Traded%':>7}  {'Acc':>6}  "
      f"{'Gross Sh':>9}  {'Net(10)':>8}  {'Net(30)':>8}  {'Net(60)':>8}  "
      f"{'Ret/trade(bps)':>14}  {'MaxDD%':>7}")
print("-" * 90)

for thresh in CONFIDENCE_THRESHOLDS:
    mask = (df_bt["prob_up"] > thresh) | (df_bt["prob_up"] < 1 - thresh)
    sub  = df_bt[mask].copy()
    if len(sub) < 30:
        continue

    sub["signal"] = np.where(sub["prob_up"] > 0.5, 1.0, -1.0)
    sub["gross"]  = sub["signal"] * sub["r_ctc"]

    # Add carry: +CARRY_BPS_PER_DAY when signal=+1 (long USD/CLP)
    sub["carry"]  = np.where(sub["signal"] > 0, CARRY_BPS_PER_DAY, -CARRY_BPS_PER_DAY)
    sub["gross_with_carry"] = sub["gross"] + sub["carry"]

    acc      = (sub["pred_dir"] == sub["actual_dir"]).mean()
    g_sh     = np.sqrt(252) * sub["gross"].mean() / sub["gross"].std()
    filter_p = len(sub) / len(df_bt) * 100
    avg_bps  = sub["gross"].mean() * 10000

    # Max drawdown on gross equity curve
    eq_gross = sub["gross"].cumsum()
    dd       = (eq_gross - eq_gross.cummax()).min() * 100

    net_sharpes = {}
    for scenario, cost in COST_SCENARIOS.items():
        sub[f"net_{scenario}"] = sub["gross_with_carry"] - cost
        net_sh = np.sqrt(252) * sub[f"net_{scenario}"].mean() / sub[f"net_{scenario}"].std()
        net_sharpes[scenario] = round(net_sh, 3)

    results[thresh] = {
        "threshold":        thresh,
        "n_trades":         int(len(sub)),
        "filter_pct":       round(filter_p, 1),
        "accuracy":         round(float(acc), 4),
        "gross_sharpe":     round(float(g_sh), 3),
        "net_sharpe":       net_sharpes,
        "avg_gross_ret_bps": round(float(avg_bps), 2),
        "max_drawdown_pct": round(float(dd), 2),
    }
    print(f"{thresh:>6.2f}  {filter_p:>6.1f}%  {acc:>6.3f}  "
          f"{g_sh:>9.3f}  {net_sharpes['optimistic']:>8.3f}  "
          f"{net_sharpes['base']:>8.3f}  {net_sharpes['pessimistic']:>8.3f}  "
          f"{avg_bps:>14.1f}  {dd:>7.1f}%")

# ── Best threshold ─────────────────────────────────────────────────────────
best_thresh = max(results, key=lambda t: results[t]["net_sharpe"]["base"])
print(f"\nBest threshold (base 30bps cost): {best_thresh}")
r = results[best_thresh]
print(f"  Trades     : {r['n_trades']} / {len(df_bt)} days ({r['filter_pct']:.1f}%)")
print(f"  Accuracy   : {r['accuracy']:.3f}")
print(f"  Gross Sh   : {r['gross_sharpe']:.3f}")
print(f"  Net (10bps): {r['net_sharpe']['optimistic']:.3f}")
print(f"  Net (30bps): {r['net_sharpe']['base']:.3f}")
print(f"  Net (60bps): {r['net_sharpe']['pessimistic']:.3f}")
print(f"  Avg trade  : {r['avg_gross_ret_bps']:.1f} bps gross")
print(f"  Max DD     : {r['max_drawdown_pct']:.1f}%")

# ── Equity curves at best threshold ───────────────────────────────────────
mask = (df_bt["prob_up"] > best_thresh) | (df_bt["prob_up"] < 1 - best_thresh)
sub  = df_bt[mask].copy()
sub["signal"]  = np.where(sub["prob_up"] > 0.5, 1.0, -1.0)
sub["gross"]   = sub["signal"] * sub["r_ctc"]
sub["carry"]   = np.where(sub["signal"] > 0, CARRY_BPS_PER_DAY, -CARRY_BPS_PER_DAY)
sub["net_opt"] = sub["gross"] + sub["carry"] - COST_SCENARIOS["optimistic"]
sub["net_base"]= sub["gross"] + sub["carry"] - COST_SCENARIOS["base"]
sub["net_pess"]= sub["gross"] + sub["carry"] - COST_SCENARIOS["pessimistic"]

all_days    = df_bt.index
eq_gross    = sub["gross"].reindex(all_days, fill_value=0).cumsum()
eq_opt      = sub["net_opt"].reindex(all_days, fill_value=0).cumsum()
eq_base     = sub["net_base"].reindex(all_days, fill_value=0).cumsum()
eq_pess     = sub["net_pess"].reindex(all_days, fill_value=0).cumsum()
eq_bh       = df_bt["r_ctc"].cumsum()   # long USD/CLP always

def max_dd(curve):
    peak = curve.cummax()
    return float((curve - peak).min())

print()
print(f"Equity curve stats (thresh={best_thresh}):")
print(f"  Gross return  : {eq_gross.iloc[-1]*100:.1f}%")
print(f"  Net return (30bps): {eq_base.iloc[-1]*100:.1f}%")
print(f"  Max drawdown (base net): {max_dd(eq_base)*100:.1f}%")
print(f"  Buy & hold    : {eq_bh.iloc[-1]*100:.1f}%")

# ── Save results ───────────────────────────────────────────────────────────
output = {
    "description": "Cost-adjusted backtest — LGB v3 intraday features, CLOSE-TO-CLOSE execution",
    "execution":   "Enter T-1 close, exit T close, signal generated using all features observable by T-1 close + same-day v3 signals",
    "model":       "LightGBM (CalibratedClassifierCV isotonic), best from v3 (86.7% acc, Sharpe 11.0)",
    "cost_scenarios_bps": {k: round(v * 10000) for k, v in COST_SCENARIOS.items()},
    "carry_bps_per_day": round(CARRY_BPS_PER_DAY * 10000, 3),
    "best_threshold": best_thresh,
    "threshold_sweep": results,
    "summary": {
        "n_oos_days":            len(df_bt),
        "n_trades":              r["n_trades"],
        "trade_pct":             r["filter_pct"],
        "gross_sharpe":          r["gross_sharpe"],
        "net_sharpe_optimistic": r["net_sharpe"]["optimistic"],
        "net_sharpe_base":       r["net_sharpe"]["base"],
        "net_sharpe_pessimistic":r["net_sharpe"]["pessimistic"],
        "accuracy":              r["accuracy"],
        "avg_gross_bps":         r["avg_gross_ret_bps"],
        "max_drawdown_pct":      r["max_drawdown_pct"],
        "gross_total_return_pct":round(eq_gross.iloc[-1] * 100, 1),
        "net_total_return_base_pct": round(eq_base.iloc[-1] * 100, 1),
    },
}
out_path = MODELS_DIR / "08_backtest_ctc.json"
out_path.write_text(json.dumps(output, indent=2))
print(f"\nSaved → {out_path}")

# ── Figures ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
fig.suptitle(
    f"Cost-Adjusted Backtest — LGB v3 Intraday · Close-to-Close Execution\n"
    f"Best threshold: {best_thresh}  |  Base cost: 30 bps RT",
    fontsize=12, fontweight="bold",
)

# Panel 1: equity curves
ax = axes[0]
ax.plot(eq_gross.index, eq_gross.values * 100, color="#bbb", lw=0.8, label="Gross (no cost)")
ax.plot(eq_opt.index,   eq_opt.values   * 100, color="#27ae60", lw=1.0, linestyle="--", label="Net 10bps RT (optimistic)")
ax.plot(eq_base.index,  eq_base.values  * 100, color="#2b6cb0", lw=1.4, label=f"Net 30bps RT (base)")
ax.plot(eq_pess.index,  eq_pess.values  * 100, color="#e74c3c", lw=1.0, linestyle=":", label="Net 60bps RT (pessimistic)")
ax.plot(eq_bh.index,    eq_bh.values    * 100, color="#e07b29", lw=0.9, linestyle="--", label="Buy & hold USD/CLP")
ax.axhline(0, color="k", lw=0.4)
ax.set_ylabel("Cumulative log-return (%)")
ax.set_title("Equity Curve (Close-to-Close)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: Net Sharpe vs threshold by cost scenario
ax = axes[1]
threshs = sorted(results.keys())
for scenario, color in [("optimistic", "#27ae60"), ("base", "#2b6cb0"), ("pessimistic", "#e74c3c")]:
    sharpes = [results[t]["net_sharpe"][scenario] for t in threshs]
    ax.plot(threshs, sharpes, marker="o", color=color, label=f"{scenario} ({COST_SCENARIOS[scenario]*10000:.0f}bps RT)")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(best_thresh, color="#2b6cb0", lw=0.8, linestyle=":")
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Net annualised Sharpe")
ax.set_title("Net Sharpe vs Confidence Threshold (by cost scenario)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: trade frequency and accuracy vs threshold
ax = axes[2]
ax2 = ax.twinx()
filter_pcts = [results[t]["filter_pct"] for t in threshs]
accs        = [results[t]["accuracy"]   for t in threshs]
gross_shs   = [results[t]["gross_sharpe"] for t in threshs]
ax.bar(threshs, filter_pcts, width=0.02, color="#2b6cb0", alpha=0.4, label="% days traded")
ax2.plot(threshs, accs, marker="s", color="#e07b29", lw=1.5, label="Accuracy")
ax2.plot(threshs, gross_shs, marker="^", color="#8e44ad", lw=1.2, linestyle="--", label="Gross Sharpe")
ax2.axhline(0.5, color="r", lw=0.6, linestyle="--")
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("% days with a trade", color="#2b6cb0")
ax2.set_ylabel("Accuracy / Gross Sharpe", color="#e07b29")
ax.set_title("Trade Frequency, Accuracy & Gross Sharpe vs Confidence Threshold")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig_path = FIGURES_DIR / "08_backtest_ctc.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure → {fig_path}")
