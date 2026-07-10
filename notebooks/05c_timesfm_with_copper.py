"""
TimesFM 2.5-200M + copper as exogenous covariate.

Uses forecast_with_covariates() which runs XReg on residuals + TimesFM.
Copper is passed as a dynamic numerical covariate (already lagged by MIN_COPPER_LAG=1d).

This is the meaningful experiment: the 200M model alone ≈ naive baseline,
but adding copper as a known signal should improve directional accuracy.

Run:
    uv run python notebooks/05c_timesfm_with_copper.py
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from copper_clp.config import (
    FEATURE_MATRIX, DAILY_PANEL, MODELS_DIR, FIGURES_DIR,
    TIMESFM_CONTEXT, TIMESFM_HORIZON, TIMESFM_REPO, MIN_COPPER_LAG,
)

import timesfm
from timesfm import ForecastConfig

CONTEXT     = TIMESFM_CONTEXT   # 512 days
HORIZON     = TIMESFM_HORIZON   # 21 days
N_WINDOWS   = 9
STEP_DAYS   = 126

print(f"Loading TimesFM 2.5-200M from {TIMESFM_REPO}…")
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO)
forecast_config = ForecastConfig(
    max_context=CONTEXT,
    max_horizon=HORIZON,
    return_backcast=True,   # required by forecast_with_covariates
)
tfm.compile(forecast_config)
print("  Model ready.\n")

# ── Load data ──────────────────────────────────────────────────────────────
panel = pd.read_csv(DAILY_PANEL, index_col=0, parse_dates=True)
usd_clp  = panel["usd_clp"].values
r_copper = panel["r_copper"].values
dates    = panel.index
n        = len(usd_clp)

# ── Walk-forward evaluation ────────────────────────────────────────────────
results = []
start_idx = n - N_WINDOWS * STEP_DAYS

for i in range(N_WINDOWS):
    t0 = start_idx + i * STEP_DAYS
    if t0 + HORIZON >= n:
        break

    ctx_start = max(0, t0 - CONTEXT)

    # Target series: log USD/CLP price (what we forecast)
    context_price = usd_clp[ctx_start : t0].tolist()

    # Copper covariate: must span context + horizon, lagged by MIN_COPPER_LAG
    # At position t in the covariate array, we use copper[t - MIN_COPPER_LAG]
    # so when predicting CLP at time T we only see copper from T-1 onwards.
    copper_ctx    = r_copper[ctx_start : t0].tolist()
    # For the forecast horizon, we don't know future copper — use last known value
    copper_future = [r_copper[t0 - MIN_COPPER_LAG]] * HORIZON
    copper_full   = copper_ctx + copper_future  # context + horizon

    # Ground truth
    actual = usd_clp[t0 : t0 + HORIZON]
    actual_ret = np.diff(np.log(actual), prepend=np.log(actual[0]))
    actual_dir = (actual_ret > 0).astype(int)

    # ── 1. Univariate baseline (no copper) ────────────────────────────────
    pt_uni, _ = tfm.forecast(horizon=HORIZON, inputs=[np.array(context_price)])
    fc_uni    = pt_uni[0][:HORIZON]
    last_p    = context_price[-1]
    ret_uni   = np.diff(np.log(np.concatenate([[last_p], fc_uni])))
    dir_uni   = (ret_uni > 0).astype(int)
    acc_uni   = float(np.mean(actual_dir == dir_uni))

    # ── 2. With copper covariate (xreg + timesfm) ─────────────────────────
    try:
        new_pts, _ = tfm.forecast_with_covariates(
            inputs=[context_price],
            dynamic_numerical_covariates={"copper_ret": [copper_full]},
            xreg_mode="xreg + timesfm",
            ridge=0.1,
            normalize_xreg_target_per_input=True,
        )
        fc_cov  = np.array(new_pts[0])[:HORIZON]
        ret_cov = np.diff(np.log(np.concatenate([[last_p], fc_cov])))
        dir_cov = (ret_cov > 0).astype(int)
        acc_cov = float(np.mean(actual_dir == dir_cov))
        signal  = np.where(dir_cov == 1, 1, -1)
        trade_r = signal * actual_ret
        sharpe_cov = float(np.sqrt(252) * trade_r.mean() / (trade_r.std() + 1e-9))
    except Exception as e:
        print(f"    xreg failed: {e}")
        acc_cov = float("nan")
        sharpe_cov = float("nan")

    # ── Naive baseline ─────────────────────────────────────────────────────
    naive_dir = np.ones(HORIZON, dtype=int)
    acc_naive = float(np.mean(actual_dir == naive_dir))

    r = {
        "window": i,
        "date":   str(dates[t0].date()),
        "acc_uni":      round(acc_uni, 4),
        "acc_cov":      round(acc_cov, 4) if not np.isnan(acc_cov) else None,
        "acc_naive":    round(acc_naive, 4),
        "sharpe_cov":   round(sharpe_cov, 4) if not np.isnan(sharpe_cov) else None,
    }
    results.append(r)
    print(f"  [{i}] {r['date']}  uni={acc_uni:.3f}  +copper={acc_cov:.3f}  "
          f"naive={acc_naive:.3f}  Sharpe={sharpe_cov:.2f}")

# ── Summary ────────────────────────────────────────────────────────────────
valid = [r for r in results if r["acc_cov"] is not None]
avg_uni   = np.mean([r["acc_uni"]   for r in results])
avg_cov   = np.mean([r["acc_cov"]   for r in valid])
avg_naive = np.mean([r["acc_naive"] for r in results])
avg_sh    = np.mean([r["sharpe_cov"] for r in valid])

summary = {
    "model": "TimesFM 2.5-200M",
    "repo": TIMESFM_REPO,
    "context_days": CONTEXT,
    "horizon_days": HORIZON,
    "n_windows": len(results),
    "avg_acc_univariate":     round(float(avg_uni),   4),
    "avg_acc_with_copper":    round(float(avg_cov),   4),
    "avg_acc_naive":          round(float(avg_naive), 4),
    "avg_trade_sharpe_cov":   round(float(avg_sh),    4),
    "copper_lag_days": MIN_COPPER_LAG,
    "xreg_mode": "xreg + timesfm",
    "windows": results,
}

out = MODELS_DIR / "05c_timesfm_copper.json"
out.write_text(json.dumps(summary, indent=2))

print(f"\n{'='*60}")
print(f"  Univariate TimesFM      avg dir-acc : {avg_uni:.3f}")
print(f"  TimesFM + copper xreg  avg dir-acc : {avg_cov:.3f}  (naive: {avg_naive:.3f})")
print(f"  Avg trade Sharpe (copper):            {avg_sh:.2f}")
print(f"  Saved → {out}")
print(f"{'='*60}\n")

# ── Figure ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("TimesFM 200M — Univariate vs +Copper Covariate", fontsize=12)

wdates = [r["date"] for r in results]
ax = axes[0]
ax.plot(wdates, [r["acc_uni"]   for r in results], "o-", label="Univariate", color="#aaa")
ax.plot(wdates, [r["acc_cov"] or 0 for r in results], "s-", label="+Copper", color="#e07b29")
ax.plot(wdates, [r["acc_naive"] for r in results], "--", label="Naive", color="#ccc")
ax.axhline(0.5, color="r", lw=0.7, linestyle=":")
ax.set_ylabel("Directional Accuracy (21d horizon)")
ax.set_title("Direction Accuracy per Window")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=45)

ax = axes[1]
sharpes = [r["sharpe_cov"] or 0 for r in results]
colors  = ["#27ae60" if s > 0 else "#e74c3c" for s in sharpes]
ax.bar(wdates, sharpes, color=colors, edgecolor="none")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Annualised Sharpe")
ax.set_title("Trade Sharpe (+Copper) per Window")
ax.grid(axis="y", alpha=0.3)
ax.tick_params(axis="x", rotation=45)

fig_path = FIGURES_DIR / "05c_timesfm_copper.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure → {fig_path}")
