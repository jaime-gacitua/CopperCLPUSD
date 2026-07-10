"""
Cost-adjusted backtest with confidence filtering and fixed-size positions.

Decision workflow:
  T-1 close  →  model sees features (copper, DXY, VIX, EM FX all at T-1 close)
  T open     →  you enter the trade (buy or sell USD/CLP)
  T close    →  you exit, collect open-to-close return

P&L per trade (fixed 1-unit notional):
  gross_ret  = signal × log(usd_clp_close_T / usd_clp_open_T)
  cost       = spread_bps × 2   (entry + exit, in log-return terms)
  net_ret    = gross_ret - cost

Confidence filter:
  Only trade when model predict_proba() > CONFIDENCE_THRESHOLD.
  We sweep thresholds to find the Sharpe-optimal cutoff.

Run:
    uv run python notebooks/07_backtest.py
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
    FEATURE_MATRIX, DATA_RAW, MODELS_DIR, FIGURES_DIR, START_DATE,
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
)
from copper_clp.features import get_feature_cols

# ── Transaction cost assumptions ───────────────────────────────────────────
# Retail spot CLP/USD typical bid-ask: 20-50 pips on ~900 CLP/USD
# 30 pips / 900 = 0.033% per side → 0.067% round-trip
# We model three scenarios:
COST_SCENARIOS = {
    "optimistic":  0.10 / 100,   # 10bps round-trip (tight spread, good execution)
    "base":        0.20 / 100,   # 20bps round-trip (realistic retail)
    "pessimistic": 0.40 / 100,   # 40bps round-trip (wide spread, thin market)
}
CONFIDENCE_THRESHOLDS = [0.50, 0.52, 0.55, 0.57, 0.60, 0.63, 0.65]

# ── Load data ──────────────────────────────────────────────────────────────
feat = pd.read_csv(FEATURE_MATRIX, index_col=0, parse_dates=True)
FCOLS = get_feature_cols(feat)

# Load OHLC for open-to-close return calculation
ohlc_raw = pd.read_csv(DATA_RAW / "clpusd_daily_full.csv",
                        index_col=0, header=[0, 1], parse_dates=True)
clp_open  = (1 / ohlc_raw["Open"]["CLPUSD=X"]).rename("usd_clp_open")
clp_close = (1 / ohlc_raw["Close"]["CLPUSD=X"]).rename("usd_clp_close")

# Open-to-close log return (what we actually capture trading T_open → T_close)
otc_ret = np.log(clp_close / clp_open).rename("r_otc")

# Align with feature matrix index
panel = pd.DataFrame({
    "usd_clp_open":  clp_open,
    "usd_clp_close": clp_close,
    "r_otc":         otc_ret,
}).reindex(feat.index).ffill()

# Close-to-close return (what the model was trained on)
panel["r_ctc"] = feat["y_ret"]

print(f"Feature matrix : {len(feat)} rows, {len(FCOLS)} features")
print(f"OTC return std : {panel['r_otc'].std()*100:.3f}%  "
      f"(CTC: {panel['r_ctc'].std()*100:.3f}%)")
print(f"OTC/CTC corr   : {panel['r_otc'].corr(panel['r_ctc']):.3f}")
print()

# ── Walk-forward with calibrated probabilities ─────────────────────────────
X_all  = feat[FCOLS].values
y_ctc  = feat["y_ret"].values           # close-to-close (used for training)
y_dir  = (y_ctc > 0).astype(int)
r_otc  = panel["r_otc"].values          # open-to-close (actual P&L)
dates  = feat.index
n      = len(X_all)

records = []   # one row per OOS day

for t0 in range(WF_TRAIN_DAYS, n - WF_TEST_DAYS, WF_STEP_DAYS):
    X_tr = X_all[t0 - WF_TRAIN_DAYS : t0]
    y_tr = y_dir[t0 - WF_TRAIN_DAYS : t0]
    X_te = X_all[t0 : t0 + WF_TEST_DAYS]
    y_te = y_dir[t0 : t0 + WF_TEST_DAYS]
    r_te = r_otc[t0 : t0 + WF_TEST_DAYS]

    ok_tr = ~np.isnan(X_tr).any(axis=1)
    ok_te = ~np.isnan(X_te).any(axis=1)
    X_tr, y_tr = X_tr[ok_tr], y_tr[ok_tr]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # Best model from v2: Random Forest (acc 63.3%, Sharpe 4.38)
    # Use isotonic calibration so predict_proba is well-calibrated
    base_rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                      random_state=42, n_jobs=-1)
    rf = CalibratedClassifierCV(base_rf, method="isotonic", cv=3)
    rf.fit(X_tr, y_tr)
    proba_te = rf.predict_proba(X_te)[:, 1]  # P(USD/CLP rises = CLP weakens)

    for i, (ok, prob, r) in enumerate(zip(ok_te, proba_te, r_te)):
        if not ok:
            continue
        global_idx = t0 + i
        records.append({
            "date":      str(dates[global_idx].date()),
            "prob_up":   float(prob),
            "pred_dir":  1 if prob > 0.5 else 0,
            "actual_dir": int(y_te[i]),
            "r_otc":     float(r),
        })

df_bt = pd.DataFrame(records)
df_bt["date"] = pd.to_datetime(df_bt["date"])
df_bt = df_bt.set_index("date").sort_index()
# Drop duplicate dates (overlapping folds — keep last fold's prediction)
df_bt = df_bt[~df_bt.index.duplicated(keep="last")]

print(f"OOS days with predictions: {len(df_bt)}")
print(f"Date range: {df_bt.index[0].date()} → {df_bt.index[-1].date()}")
print(f"Base accuracy (>0.5 threshold): "
      f"{(df_bt['pred_dir'] == df_bt['actual_dir']).mean():.3f}")
print()

# ── Sweep confidence thresholds × cost scenarios ──────────────────────────
results = {}
print(f"{'Threshold':>10}  {'Filter%':>7}  {'Acc':>6}  "
      f"{'Gross Sh':>9}  {'Net(opt)':>9}  {'Net(base)':>10}  {'Net(pess)':>10}")
print("-" * 75)

for thresh in CONFIDENCE_THRESHOLDS:
    # Trade only when conviction is high: prob > thresh (long) or < 1-thresh (short)
    mask = (df_bt["prob_up"] > thresh) | (df_bt["prob_up"] < 1 - thresh)
    sub  = df_bt[mask].copy()
    if len(sub) < 30:
        continue

    sub["signal"] = np.where(sub["prob_up"] > 0.5, 1.0, -1.0)
    sub["gross"]  = sub["signal"] * sub["r_otc"]

    acc    = (sub["pred_dir"] == sub["actual_dir"]).mean()
    g_sh   = np.sqrt(252) * sub["gross"].mean() / sub["gross"].std()
    filter_pct = len(sub) / len(df_bt) * 100

    net_sharpes = {}
    for scenario, cost in COST_SCENARIOS.items():
        sub[f"net_{scenario}"] = sub["gross"] - cost
        net_sh = np.sqrt(252) * sub[f"net_{scenario}"].mean() / sub[f"net_{scenario}"].std()
        net_sharpes[scenario] = round(net_sh, 3)

    results[thresh] = {
        "threshold":    thresh,
        "n_trades":     int(len(sub)),
        "filter_pct":   round(filter_pct, 1),
        "accuracy":     round(float(acc), 4),
        "gross_sharpe": round(float(g_sh), 3),
        "net_sharpe":   net_sharpes,
        "avg_gross_ret_bps": round(float(sub["gross"].mean() * 10000), 2),
    }
    print(f"{thresh:>10.2f}  {filter_pct:>6.1f}%  {acc:>6.3f}  "
          f"{g_sh:>9.3f}  {net_sharpes['optimistic']:>9.3f}  "
          f"{net_sharpes['base']:>10.3f}  {net_sharpes['pessimistic']:>10.3f}")

# ── Full equity curve at base case, best threshold ─────────────────────────
# Pick threshold that maximises net Sharpe under base cost
best_thresh = max(results, key=lambda t: results[t]["net_sharpe"]["base"])
print(f"\nBest threshold (base cost): {best_thresh}")

mask  = (df_bt["prob_up"] > best_thresh) | (df_bt["prob_up"] < 1 - best_thresh)
sub   = df_bt[mask].copy()
sub["signal"] = np.where(sub["prob_up"] > 0.5, 1.0, -1.0)
sub["gross"]  = sub["signal"] * sub["r_otc"]
sub["net"]    = sub["gross"] - COST_SCENARIOS["base"]

# Reindex to all OOS days (flat on non-trade days)
all_days = df_bt.index
equity_gross = sub["gross"].reindex(all_days, fill_value=0).cumsum()
equity_net   = sub["net"].reindex(all_days, fill_value=0).cumsum()
equity_bh    = df_bt["r_otc"].cumsum()   # buy-and-hold USD/CLP

# Max drawdown
def max_dd(curve):
    peak = curve.cummax()
    return float((curve - peak).min())

print(f"\nEquity curve stats (base cost, thresh={best_thresh}):")
print(f"  Trades      : {len(sub)} / {len(df_bt)} days ({len(sub)/len(df_bt)*100:.1f}%)")
print(f"  Gross return: {equity_gross.iloc[-1]*100:.1f}%")
print(f"  Net return  : {equity_net.iloc[-1]*100:.1f}%")
print(f"  Max drawdown: {max_dd(equity_net)*100:.1f}%")
print(f"  Avg trade   : {sub['net'].mean()*10000:.1f} bps net")

# ── Save ───────────────────────────────────────────────────────────────────
output = {
    "description": "Cost-adjusted backtest — RF v2 macro signals, open-to-close, fixed size",
    "execution":   "Enter T-open, exit T-close, signal from T-1 close prices",
    "cost_scenarios_bps": {k: round(v*10000) for k, v in COST_SCENARIOS.items()},
    "best_threshold": best_thresh,
    "threshold_sweep": results,
    "summary": {
        "n_oos_days":   len(df_bt),
        "n_trades":     len(sub),
        "trade_pct":    round(len(sub)/len(df_bt)*100, 1),
        "gross_sharpe": results[best_thresh]["gross_sharpe"],
        "net_sharpe_optimistic":  results[best_thresh]["net_sharpe"]["optimistic"],
        "net_sharpe_base":        results[best_thresh]["net_sharpe"]["base"],
        "net_sharpe_pessimistic": results[best_thresh]["net_sharpe"]["pessimistic"],
        "accuracy":     results[best_thresh]["accuracy"],
        "max_drawdown_pct": round(max_dd(equity_net)*100, 2),
    },
}
out_path = MODELS_DIR / "07_backtest.json"
out_path.write_text(json.dumps(output, indent=2))
print(f"\nSaved → {out_path}")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
fig.suptitle(
    f"Cost-Adjusted Backtest — RF v2 Macro Signals\n"
    f"Open-to-close, fixed size, base cost 20bps RT",
    fontsize=12, fontweight="bold"
)

# Panel 1: equity curves
ax = axes[0]
ax.plot(equity_gross.index, equity_gross.values * 100,
        color="#aaa", lw=0.9, label="Gross (no cost)")
ax.plot(equity_net.index, equity_net.values * 100,
        color="#2b6cb0", lw=1.3, label=f"Net (20bps RT, thresh={best_thresh})")
ax.plot(equity_bh.index, equity_bh.values * 100,
        color="#e07b29", lw=0.9, linestyle="--", label="Buy & hold USD/CLP")
ax.axhline(0, color="k", lw=0.4)
ax.set_ylabel("Cumulative log-return (%)")
ax.set_title("Equity Curve")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 2: net Sharpe vs threshold, by cost scenario
ax = axes[1]
threshs = sorted(results.keys())
for scenario, color in [("optimistic","#27ae60"),("base","#2b6cb0"),("pessimistic","#e74c3c")]:
    sharpes = [results[t]["net_sharpe"][scenario] for t in threshs]
    n_trades = [results[t]["filter_pct"] for t in threshs]
    ax.plot(threshs, sharpes, marker="o", color=color,
            label=f"{scenario} ({COST_SCENARIOS[scenario]*10000:.0f}bps RT)")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(best_thresh, color="#2b6cb0", lw=0.8, linestyle=":")
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Net annualised Sharpe")
ax.set_title("Net Sharpe vs Confidence Threshold (by cost scenario)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 3: trade frequency and accuracy vs threshold
ax = axes[2]
ax2 = ax.twinx()
filter_pcts = [results[t]["filter_pct"] for t in threshs]
accs = [results[t]["accuracy"] for t in threshs]
ax.bar(threshs, filter_pcts, width=0.02, color="#2b6cb0", alpha=0.4, label="% days traded")
ax2.plot(threshs, accs, marker="s", color="#e07b29", lw=1.5, label="Accuracy")
ax2.axhline(0.5, color="r", lw=0.6, linestyle="--")
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("% days with a trade", color="#2b6cb0")
ax2.set_ylabel("Directional accuracy", color="#e07b29")
ax.set_title("Trade Frequency & Accuracy vs Confidence Threshold")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig_path = FIGURES_DIR / "07_backtest.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure → {fig_path}")
