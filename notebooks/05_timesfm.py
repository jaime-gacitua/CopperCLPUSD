"""
Experiment 5: TimesFM forecasting

TimesFM is a foundation time-series model from Google.
We use it to forecast USD/CLP directly, then compare:
  (a) forecast using only CLP history
  (b) forecast using CLP history conditioned on copper as a covariate

Note: TimesFM 1.0 (PAX/JAX) requires Python >=3.10 and specific JAX version.
We install the cpu-only version. If unavailable we fall back to a documented stub.

Output: results/05_timesfm.json, figures/05_timesfm.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results"); OUT.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

df = pd.read_csv("data/daily_panel.csv", index_col=0, parse_dates=True)
df["log_usd_clp"] = np.log(df["usd_clp"])

# Try to import TimesFM
try:
    import timesfm
    TIMESFM_AVAILABLE = True
    print("TimesFM available (v2)")
except ImportError:
    TIMESFM_AVAILABLE = False
    print("TimesFM not installed — running naive baseline only.")

# ── Walk-forward evaluation ────────────────────────────────────────────────
CONTEXT_LEN = 512     # TimesFM default context window
HORIZON = 21          # forecast 21 days ahead
TEST_START = "2022-01-01"

series = df["usd_clp"].values
log_series = df["log_usd_clp"].values
dates_all = df.index

test_mask = dates_all >= TEST_START
test_idx = np.where(test_mask)[0]

results_tfm = []
results_naive = []
results_drift = []

if TIMESFM_AVAILABLE:
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        config=timesfm.ForecastConfig(
            max_context=CONTEXT_LEN,
            max_horizon=HORIZON,
        ),
    )
    forecast_config = timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON,
    )
    tfm.compile(forecast_config)
    print("TimesFM 2.5 model loaded and compiled.")

# Evaluate at quarterly intervals
eval_starts = test_idx[::HORIZON * 6][:12]   # ~12 evaluation windows

for start_i in eval_starts:
    if start_i < CONTEXT_LEN:
        continue
    context = log_series[start_i - CONTEXT_LEN : start_i]
    actuals = log_series[start_i : start_i + HORIZON]
    if len(actuals) < HORIZON:
        break

    # Naive: last value repeated
    naive_forecast = np.full(HORIZON, context[-1])

    # Random walk with drift
    drift = (context[-1] - context[-HORIZON]) / HORIZON
    drift_forecast = context[-1] + drift * np.arange(1, HORIZON + 1)

    mae_naive = float(np.mean(np.abs(naive_forecast - actuals)))
    mae_drift = float(np.mean(np.abs(drift_forecast - actuals)))
    results_naive.append(mae_naive)
    results_drift.append(mae_drift)

    if TIMESFM_AVAILABLE:
        # v2 API: forecast(horizon, inputs) returns point forecast array
        tfm_out = tfm.forecast(horizon=HORIZON, inputs=[context.tolist()])
        # Returns shape (batch, horizon) — take first batch, point forecast (index 0 of quantiles or mean)
        if hasattr(tfm_out, "mean"):
            tfm_forecast = np.array(tfm_out.mean[0])[:HORIZON]
        else:
            tfm_forecast = np.array(tfm_out[0])[:HORIZON]
        mae_tfm = float(np.mean(np.abs(tfm_forecast - actuals)))
        results_tfm.append(mae_tfm)
        print(f"  Window start={dates_all[start_i].date()}: naive MAE={mae_naive:.5f} "
              f"drift={mae_drift:.5f} TimesFM={mae_tfm:.5f}")
    else:
        print(f"  Window start={dates_all[start_i].date()}: naive MAE={mae_naive:.5f} "
              f"drift={mae_drift:.5f}")

output = {
    "timesfm_available": TIMESFM_AVAILABLE,
    "horizon_days": HORIZON,
    "context_len": CONTEXT_LEN,
    "naive_mae_mean": round(float(np.mean(results_naive)), 6) if results_naive else None,
    "drift_mae_mean": round(float(np.mean(results_drift)), 6) if results_drift else None,
    "timesfm_mae_mean": round(float(np.mean(results_tfm)), 6) if results_tfm else None,
    "n_windows": len(results_naive),
}
if not TIMESFM_AVAILABLE:
    output["install_note"] = "Run: uv add timesfm-cpu && python experiments/05_timesfm.py"

(OUT / "05_timesfm.json").write_text(json.dumps(output, indent=2))
print("\nResults:", json.dumps(output, indent=2))

# ── Figure (baselines regardless of TimesFM) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#aaa")
    ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)

# Recent forecast example (last window)
ax = axes[0]
last_i = eval_starts[-1] if len(eval_starts) > 0 else test_idx[0]
if last_i >= CONTEXT_LEN:
    ctx = np.exp(log_series[last_i - 60 : last_i])
    act = np.exp(log_series[last_i : last_i + HORIZON])
    x_ctx = np.arange(-60, 0)
    x_fwd = np.arange(0, len(act))
    ax.plot(x_ctx, ctx, color="#4e9af1", lw=1.2, label="Context (last 60d)")
    ax.plot(x_fwd, act, color="#7be495", lw=1.2, label="Actual", linestyle="--")
    naive_f = np.full(HORIZON, ctx[-1])
    ax.plot(x_fwd[:len(naive_f)], naive_f, color="#e8a020", lw=1, label="Naive forecast")
    drift_f = ctx[-1] + ((ctx[-1] - ctx[-HORIZON]) / HORIZON) * np.arange(1, HORIZON + 1)
    ax.plot(x_fwd[:HORIZON], drift_f, color="#ff6b6b", lw=1, linestyle=":", label="Drift forecast")
ax.set_title(f"Example 21-day forecast (most recent window)", color="white")
ax.set_xlabel("Days from forecast origin", color="#aaa")
ax.set_ylabel("USD/CLP", color="#aaa")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333", fontsize=8)

# MAE comparison
ax = axes[1]
labels = ["Naive", "Drift"]
maes = [output["naive_mae_mean"] or 0, output["drift_mae_mean"] or 0]
colors_b = ["#e8a020", "#ff6b6b"]
if TIMESFM_AVAILABLE and output["timesfm_mae_mean"]:
    labels.append("TimesFM")
    maes.append(output["timesfm_mae_mean"])
    colors_b.append("#7be495")
bars = ax.bar(labels, maes, color=colors_b)
ax.set_title("21-day forecast MAE (log scale)", color="white")
ax.set_ylabel("Mean Absolute Error (log USD/CLP)", color="#aaa")
for bar, v in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, v * 1.02, f"{v:.5f}", ha="center", color="white", fontsize=9)

fig.suptitle("Experiment 5: TimesFM Forecasting", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "05_timesfm.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved figures/05_timesfm.png")
