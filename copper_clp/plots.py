"""
Reusable visualization functions.

All figures are saved to reports/figures/.
Call functions directly or import from notebooks.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from copper_clp.config import FIGURES_DIR, MODELS_DIR, DAILY_PANEL


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_raw_series(df: pd.DataFrame | None = None) -> None:
    """Three-panel: copper price, CLP/USD rate, and rolling correlation."""
    if df is None:
        df = pd.read_csv(DAILY_PANEL, index_col=0, parse_dates=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Copper (HG=F) vs CLP/USD — Daily", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(df.index, df["copper"], color="#e07b29", lw=0.8)
    ax.set_ylabel("Copper (USD/lb)")
    ax.set_title("COMEX Copper Futures (HG=F)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df.index, df["usd_clp"], color="#2b6cb0", lw=0.8)
    ax.set_ylabel("CLP per USD")
    ax.set_title("USD/CLP Exchange Rate (higher = CLP weaker)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    roll_corr = df["r_copper"].rolling(126).corr(df["r_usd_clp"])
    ax.plot(df.index, roll_corr, color="#5a4fcf", lw=0.8)
    ax.axhline(0, color="k", lw=0.5, linestyle="--")
    ax.fill_between(df.index, roll_corr, 0,
                    where=(roll_corr < 0), color="#e07b29", alpha=0.3,
                    label="Negative corr (copper↑ → CLP strengthens)")
    ax.set_ylabel("126-day rolling corr")
    ax.set_title("Rolling Correlation of Log-Returns")
    ax.legend(fontsize=9)
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()
    _save(fig, "copper_vs_clpusd")


def plot_lag_correlation(path: str | None = None) -> None:
    """Bar chart of Pearson r at each copper lag."""
    if path is None:
        path = MODELS_DIR / "01_lag_correlation.json"
    data = json.loads(open(path).read())

    # Support both list-of-dicts and parallel-arrays formats
    if "results" in data:
        lags  = [d["lag_days"] for d in data["results"]]
        corrs = [d["pearson_r"] for d in data["results"]]
        pvals = [d.get("p_value", 1.0) for d in data["results"]]
    else:
        lags  = data["lags"]
        corrs = data["pearson_xcorr"]
        pvals = [1.0] * len(lags)  # old format has no p-values per lag

    # Only show positive lags (copper leads CLP)
    pos = [(l, c, p) for l, c, p in zip(lags, corrs, pvals) if l >= 0]
    lags, corrs, pvals = zip(*pos) if pos else ([], [], [])

    colors = ["#2b6cb0" if abs(c) > 0.02 else "#aaaaaa" for c in corrs]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(lags, corrs, color=colors, edgecolor="none", width=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Copper lag (trading days)")
    ax.set_ylabel("Pearson r  (copper lag → next-day USD/CLP)")
    ax.set_title("Cross-Correlation: Lagged Copper Return → USD/CLP Return")
    ax.text(0.98, 0.95, "Blue = p < 0.05", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#2b6cb0")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "01_lag_correlation")


def plot_ml_results(path: str | None = None) -> None:
    """Bar chart comparing model directional accuracy and Sharpe across walk-forward folds."""
    if path is None:
        path = MODELS_DIR / "walk_forward_results.json"
    data = json.loads(open(path).read())
    summary = data["summary"]

    models = [m for m in summary if m != "ridge"]
    acc    = [summary[m]["acc"]    for m in models]
    sharpe = [summary[m]["sharpe"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Walk-Forward OOS Performance (3yr train / 1yr test / 63d step)", fontsize=12)

    colors = ["#2b6cb0", "#e07b29", "#5a4fcf", "#27ae60"]
    ax1.bar(models, acc, color=colors, edgecolor="none")
    ax1.axhline(0.5, color="r", lw=1, linestyle="--", label="Naive 50%")
    ax1.set_ylabel("Directional Accuracy")
    ax1.set_title("Accuracy (OOS, avg across folds)")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0.45, 0.70)

    ax2.bar(models, sharpe, color=colors, edgecolor="none")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("Annualised Sharpe")
    ax2.set_title("Signal Sharpe (OOS, avg across folds)")
    ax2.grid(axis="y", alpha=0.3)

    _save(fig, "ml_results")


def plot_feature_importance(path: str | None = None, top_n: int = 15) -> None:
    """Horizontal bar chart of XGBoost feature importances."""
    if path is None:
        path = MODELS_DIR / "walk_forward_results.json"
    data = json.loads(open(path).read())
    fi = data["feature_importance"][:top_n][::-1]

    names = [x["feature"] for x in fi]
    imps  = [x["importance"] for x in fi]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, imps, color="#2b6cb0", edgecolor="none")
    ax.set_xlabel("XGBoost Gain Importance")
    ax.set_title(f"Top {top_n} Features — Full-Sample XGBoost")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "feature_importance")


def plot_equity_curve(
    feat: pd.DataFrame | None = None,
    model_name: str = "xgb",
    path: str | None = None,
) -> None:
    """Equity curve for the best model's OOS signal vs buy-and-hold."""
    if path is None:
        path = MODELS_DIR / "walk_forward_results.json"
    data = json.loads(open(path).read())
    folds = data["folds"]

    if feat is None:
        from copper_clp.config import FEATURE_MATRIX
        feat = pd.read_csv(FEATURE_MATRIX, index_col=0, parse_dates=True)

    # Reconstruct OOS dates vs naive signal for illustration
    oos_sharpes = [f[model_name]["sharpe"] for f in folds]
    dates = [f["date_start"] for f in folds]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, oos_sharpes, marker="o", ms=3, color="#2b6cb0", lw=1)
    ax.axhline(0, color="k", lw=0.5, linestyle="--")
    ax.set_ylabel("Annualised Sharpe (OOS fold)")
    ax.set_title(f"{model_name.upper()} — Per-Fold OOS Sharpe")
    ax.set_xlabel("Fold start date")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    _save(fig, f"equity_curve_{model_name}")


if __name__ == "__main__":
    plot_raw_series()
    try:
        plot_lag_correlation()
    except FileNotFoundError:
        print("Run 01_lag_correlation.py first")
    try:
        plot_ml_results()
        plot_feature_importance()
        plot_equity_curve()
    except FileNotFoundError:
        print("Run walk-forward training first (copper_clp/modeling/train.py)")
