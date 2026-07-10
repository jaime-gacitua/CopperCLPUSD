"""
Experiment 1: Lag correlation analysis
- Cross-correlation of copper returns vs CLP returns at many lags
- Rolling Pearson and Spearman correlations
- Level vs return correlation
- Output: results/01_lag_correlation.json
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUT = Path("results")
OUT.mkdir(exist_ok=True)
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

df = pd.read_csv("data/daily_panel.csv", index_col=0, parse_dates=True)

# Log returns
df["r_copper"] = np.log(df["copper"]).diff()
df["r_usd_clp"] = np.log(df["usd_clp"]).diff()
df = df.dropna()

# ── 1. Cross-correlation at lags -30..+30 (copper leads CLP) ──────────────
lags = range(-30, 31)
pearson_xcorr = []
spearman_xcorr = []

for lag in lags:
    shifted_copper = df["r_copper"].shift(lag)
    aligned = pd.concat([shifted_copper, df["r_usd_clp"]], axis=1).dropna()
    p, _ = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    s, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    pearson_xcorr.append(round(p, 4))
    spearman_xcorr.append(round(s, 4))

# ── 2. Level cross-correlation ─────────────────────────────────────────────
level_xcorr = []
for lag in lags:
    shifted_cu = df["copper"].shift(lag)
    aligned = pd.concat([shifted_cu, df["usd_clp"]], axis=1).dropna()
    p, _ = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    level_xcorr.append(round(p, 4))

# ── 3. Rolling 90-day Pearson on returns ──────────────────────────────────
roll90 = df["r_copper"].rolling(90).corr(df["r_usd_clp"])
roll252 = df["r_copper"].rolling(252).corr(df["r_usd_clp"])

# ── 4. Regime stats: strong vs weak copper ─────────────────────────────────
median_cu = df["copper"].median()
bull = df[df["copper"] > median_cu]
bear = df[df["copper"] <= median_cu]
bull_corr, _ = stats.pearsonr(bull["r_copper"].dropna(), bull["r_usd_clp"].dropna())
bear_corr, _ = stats.pearsonr(bear["r_copper"].dropna(), bear["r_usd_clp"].dropna())

# ── 5. Best lag by absolute Pearson ───────────────────────────────────────
lag_list = list(lags)
best_lag_idx = int(np.argmax(np.abs(pearson_xcorr)))
best_lag = lag_list[best_lag_idx]
best_corr = pearson_xcorr[best_lag_idx]

results = {
    "lags": lag_list,
    "pearson_xcorr": pearson_xcorr,
    "spearman_xcorr": spearman_xcorr,
    "level_xcorr": level_xcorr,
    "best_lag_returns": best_lag,
    "best_corr_returns": best_corr,
    "bull_regime_corr": round(bull_corr, 4),
    "bear_regime_corr": round(bear_corr, 4),
    "overall_pearson": round(df["r_copper"].corr(df["r_usd_clp"]), 4),
    "overall_spearman": round(df["r_copper"].corr(df["r_usd_clp"], method="spearman"), 4),
}
(OUT / "01_lag_correlation.json").write_text(json.dumps(results, indent=2))
print("Results:", json.dumps(results, indent=2))

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.patch.set_facecolor("#0f1117")
for ax in axes.flat:
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#aaa")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax = axes[0, 0]
ax.bar(lag_list, pearson_xcorr, color=["#e8a020" if x >= 0 else "#4e9af1" for x in pearson_xcorr], width=0.8)
ax.axvline(0, color="#555", lw=0.8, ls="--")
ax.axhline(0, color="#555", lw=0.5)
ax.set_title("Cross-correlation: copper return vs CLP return (Pearson)", color="white")
ax.set_xlabel("Lag (days, positive = copper leads)", color="#aaa")
ax.set_ylabel("Pearson r", color="#aaa")

ax = axes[0, 1]
ax.bar(lag_list, spearman_xcorr, color=["#7be495" if x >= 0 else "#ff6b6b" for x in spearman_xcorr], width=0.8)
ax.axvline(0, color="#555", lw=0.8, ls="--")
ax.axhline(0, color="#555", lw=0.5)
ax.set_title("Cross-correlation: copper return vs CLP return (Spearman)", color="white")
ax.set_xlabel("Lag (days, positive = copper leads)", color="#aaa")
ax.set_ylabel("Spearman r", color="#aaa")

ax = axes[1, 0]
ax.plot(df.index, roll90, color="#e8a020", lw=0.8, label="90-day rolling")
ax.plot(df.index, roll252, color="#4e9af1", lw=0.8, label="252-day rolling")
ax.axhline(0, color="#555", lw=0.5, ls="--")
ax.set_title("Rolling correlation: copper returns vs CLP returns", color="white")
ax.set_ylabel("Pearson r", color="#aaa")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333")

ax = axes[1, 1]
ax.bar(["Bull copper\n(above median)", "Bear copper\n(below median)"],
       [bull_corr, bear_corr], color=["#e8a020", "#4e9af1"])
ax.axhline(0, color="#555", lw=0.5)
ax.set_title("Correlation by copper price regime", color="white")
ax.set_ylabel("Pearson r", color="#aaa")

fig.suptitle("Experiment 1: Lag Correlation Analysis", color="white", fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "01_lag_correlation.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved figures/01_lag_correlation.png")
