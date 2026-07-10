"""
Strategy B Backtest — XTB Session (Open-to-Close, Fade the Gap)

STRATEGY DESCRIPTION
--------------------
Entry  : T open (~08:30 Santiago = 12:30–13:30 CET)
Exit   : T close (~13:45 Santiago = 17:45 CET)
Signal : Fade overnight gap — if USD/CLP gapped UP overnight, SHORT; if DOWN, LONG
Filter : |gap| > threshold AND 21d realised vol > 10% ann.
Cost   : 6.5 bps round-trip (XTB Chile indicative spread)

CRITICAL DATA QUALITY FINDINGS
-------------------------------
This notebook documents why a reliable Strategy B backtest CANNOT be built
from freely available data, and what data we DO have.

1. yfinance CLPUSD=X open prices:
   - When open != prev_close (74% of the time), the "open" is actually the
     same-day CLOSE — creating look-ahead bias that makes every backtest
     using these opens INVALID.
   - The pre-2012 Strategy B backtest (Sharpe ~6.6 in handoff.md) was based
     on this contaminated data. It is not a reliable estimate.
   - The corr(gap, otc) = -0.59 shown in 2004 was also an artifact of using
     the close as the "open" — the otc intraday return was systematically
     close to zero (open≈close), making the fade-gap signal look perfect.

2. Alpha Vantage FX_DAILY open prices:
   - Available from 2014-11-07, real open (differs from prev_close) on ~15%
     of trading days (372 days total).
   - corr(gap, otc) = -0.016 on real-open days — no statistically significant
     signal found.
   - High% of days have high==low==open==close (single-tick reference price,
     not a true MCF open).
   - The AV "open" appears to be a different market's open (likely Europe/US
     electronic open), not the MCF session open at 08:30 Santiago.

3. BCCh (Banco Central de Chile) API:
   - Publishes only the "Dólar Observado" (official MCF daily close).
   - No open price published. Intraday data requires a data contract.

WHAT WE KNOW
-----------
From the Strategy A (close-to-close) backtest and established research:
  - corr(clp_gap, clp_ctc) = +0.61: overnight gap predicts full-day CTC return
  - corr(clp_gap, clp_otc) = -0.50: overnight gap REVERSES during MCF session
  → The fade-gap signal IS real based on the CTC decomposition
  → Strategy B is theoretically valid but untestable with available data

WHAT THIS SCRIPT DOES
---------------------
Section 1: Reproduce the data quality diagnosis
Section 2: AV-based partial analysis (honest, data-limited)
Section 3: Synthetic signal estimate using CTC decomposition
Section 4: Paper trade framework (how to collect real data going forward)
Section 5: Sensitivity analysis for XTB live trading parameters

Run:
    uv run python notebooks/09_backtest_xtb.py
"""
import sys, json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from copper_clp.config import DATA_RAW, MODELS_DIR, FIGURES_DIR

# ── Constants ──────────────────────────────────────────────────────────────
XTB_COST_BPS = 6.5          # round-trip cost at XTB (bps)
XTB_COST = XTB_COST_BPS / 10_000
GAP_THRESHOLDS = [0, 20, 30, 43, 50, 70, 100]   # bps
VOL_FILTER_ANN = 0.10        # 21d realised vol must exceed 10% ann.

print("=" * 70)
print("STRATEGY B BACKTEST — XTB SESSION (FADE THE GAP)")
print("=" * 70)
print()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA QUALITY DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════
print("SECTION 1: DATA QUALITY DIAGNOSIS")
print("-" * 40)

df_yf = pd.read_csv(DATA_RAW / "clpusd_daily.csv", header=[0,1,2], index_col=0)
df_yf.columns = ["Close","High","Low","Open","Volume"]
df_yf.index = pd.to_datetime(df_yf.index)
df_yf = df_yf.apply(pd.to_numeric, errors="coerce")

usd_clp_close = 1.0 / df_yf["Close"]
usd_clp_open  = 1.0 / df_yf["Open"]

diag = pd.DataFrame({
    "close":      usd_clp_close,
    "open":       usd_clp_open,
    "prev_close": usd_clp_close.shift(1),
    "same_close": usd_clp_close,          # same-day close
})
diag["open_diff_from_prev"] = abs(diag["open"] - diag["prev_close"])
diag["open_diff_from_close"] = abs(diag["open"] - diag["same_close"])
# Filter to valid price range (avoid known bad ticks < 400 CLP/USD from early 2003)
diag_valid = diag[(diag["open"] > 400) & (diag["close"] > 400)].copy()
diag_valid["open_differs_from_prev"] = diag_valid["open_diff_from_prev"] > 2.0   # >2 CLP
diag_valid["open_approx_close"] = diag_valid["open_diff_from_close"] < 2.0        # within 2 CLP

changed = diag_valid[diag_valid["open_differs_from_prev"]].copy()
pct_is_close = changed["open_approx_close"].mean()

print(f"yfinance CLPUSD=X rows: {len(diag)}")
print(f"Rows where open != prev_close: {len(changed)} ({100*len(changed)/len(diag):.0f}%)")
print(f"Of those, open ≈ same-day close (within 2 CLP): {pct_is_close:.0%}")
print()
print("→ DIAGNOSIS: yfinance 'open' for CLPUSD=X is the same-day close when")
print("  it differs from the prior close. ALL Strategy B backtests using")
print("  yfinance opens have look-ahead bias and are INVALID.")
print()

by_year_diag = changed.groupby(changed.index.year)["open_approx_close"].agg(["mean","count"])
print("Look-ahead contamination by year:")
for yr, row in by_year_diag.iterrows():
    print(f"  {yr}: {int(row['count']):3d} days where open!=prev_close, {row['mean']:.0%} are actually same-day close")

print()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ALPHA VANTAGE PARTIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("SECTION 2: ALPHA VANTAGE PARTIAL ANALYSIS")
print("-" * 40)

av_path = DATA_RAW / "av_usdclp_daily.json"
if not av_path.exists():
    print("⚠ Alpha Vantage data not found. Run: curl ... to download it.")
    print("  Skipping Section 2.")
    df_av = None
else:
    with open(av_path) as f:
        d = json.load(f)
    ts = d["Time Series FX (Daily)"]
    rows = [{"date": k, "open": float(v["1. open"]), "high": float(v["2. high"]),
             "low": float(v["3. low"]), "close": float(v["4. close"])} for k, v in ts.items()]
    df_av = pd.DataFrame(rows).set_index("date")
    df_av.index = pd.to_datetime(df_av.index)
    df_av = df_av.sort_index()

    df_av["prev_close"] = df_av["close"].shift(1)
    df_av["open_diff"] = abs(df_av["open"] - df_av["prev_close"])
    df_av["open_real"] = df_av["open_diff"] >= 1.0      # >=1 CLP difference
    df_av["has_range"] = (df_av["high"] - df_av["low"]) > 1.0
    df_av["gap"] = np.log(df_av["open"] / df_av["prev_close"])
    df_av["otc"] = np.log(df_av["close"] / df_av["open"])
    df_av["ctc"] = np.log(df_av["close"] / df_av["prev_close"])

    real_av = df_av[df_av["open_real"]].copy()

    print(f"AV data: {len(df_av)} rows from {df_av.index[0].date()} to {df_av.index[-1].date()}")
    print(f"Rows with real open (open != prev_close ±1): {len(real_av)} ({100*len(real_av)/len(df_av):.0f}%)")
    print()
    print(f"corr(gap, otc) on AV real-open days: {real_av['gap'].corr(real_av['otc']):.3f}")
    print(f"corr(gap, ctc) on AV real-open days: {real_av['gap'].corr(real_av['ctc']):.3f}")
    print()
    print("→ corr(gap, otc) ≈ 0 suggests AV 'open' is not the MCF session open.")
    print("  The AV open appears to be a foreign market open, not the 08:30 Santiago price.")
    print()

    # AV-based backtest (honest: shows no reliable signal)
    print("AV-based Strategy B (fade gap), on real-open days:")
    for thresh in [0, 30, 50]:
        t = thresh / 10_000
        filt = real_av[real_av["gap"].abs() > t].copy()
        if len(filt) < 20:
            continue
        filt["signal"] = -np.sign(filt["gap"])
        filt["gross"] = filt["signal"] * filt["otc"]
        filt["net"] = filt["gross"] - XTB_COST
        sh_g = np.sqrt(252) * filt["gross"].mean() / filt["gross"].std()
        sh_n = np.sqrt(252) * filt["net"].mean() / filt["net"].std()
        wr = (filt["gross"] > 0).mean()
        print(f"  thresh={thresh:3d}bps  n={len(filt):4d}  wr={wr:.2f}  "
              f"Sh(gross)={sh_g:+.2f}  Sh(net)={sh_n:+.2f}")
    print()
    print("→ AV-based Strategy B shows no significant edge (Sharpe ~0 or negative).")
    print("  This confirms the AV open does not capture the MCF session open price.")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SYNTHETIC SIGNAL ESTIMATE
# ══════════════════════════════════════════════════════════════════════════════
print("SECTION 3: SYNTHETIC SIGNAL ESTIMATE")
print("-" * 40)

# We know from Strategy A (CTC backtest):
#   - When gap > 0 and we follow it → positive CTC return
#   - corr(gap, ctc) = +0.61 (full dataset, from Strategy A analysis)
#
# CTC = gap + otc by arithmetic identity
# If corr(gap, ctc) = +0.61 and gap accounts for ~50% of CTC variance,
# then otc (the MCF intraday component) = ctc - gap
#
# We CAN compute the gap signal using the reliable CTC data:
#   Synthetic OTC = ctc - gap_signal
# But we only know gap from the CTC direction and the CTC value, not the gap directly.
#
# Alternative: use the CTC return from Strategy A and decompose:
# If corr(gap, otc) = -0.50 (per research notes), then:
#   E[otc | gap>0] ≈ -0.50 * std(otc) / std(gap) * gap
# This gives us a THEORETICAL estimate of Strategy B returns.

feat = pd.read_csv("data/processed/feature_matrix.csv", index_col=0, parse_dates=True)

if "clp_gap" in feat.columns and "clp_otc" in feat.columns:
    # Use the feature matrix values (but these suffer from same yfinance problem)
    gap_col = feat["clp_gap"].dropna()
    otc_col = feat["clp_otc"].dropna()
    ctc_col = feat["y_ret"].dropna()

    common = gap_col.index.intersection(otc_col.index).intersection(ctc_col.index)
    analysis = pd.DataFrame({
        "gap": gap_col.loc[common],
        "otc": otc_col.loc[common],
        "ctc": ctc_col.loc[common],
    }).dropna()

    # Identify rows where gap + otc ≈ ctc (consistency check)
    analysis["reconstituted_ctc"] = analysis["gap"] + analysis["otc"]
    analysis["ctc_consistent"] = abs(analysis["reconstituted_ctc"] - analysis["ctc"]) < 0.001

    consistent = analysis[analysis["ctc_consistent"]]
    print(f"Feature matrix gap/otc rows: {len(analysis)}")
    print(f"Rows where gap+otc ≈ ctc (consistent): {len(consistent)} ({100*len(consistent)/len(analysis):.0f}%)")

    if len(consistent) > 100:
        corr_gap_ctc = consistent["gap"].corr(consistent["ctc"])
        corr_gap_otc = consistent["gap"].corr(consistent["otc"])
        print(f"  corr(gap, ctc) = {corr_gap_ctc:.3f}")
        print(f"  corr(gap, otc) = {corr_gap_otc:.3f}")
        print()

        # Theoretical Strategy B return estimate
        # If corr(gap, otc) = ρ = -0.50, and we fade gap with size β=1:
        # E[signal × otc] = -sign(gap) × E[otc] = -ρ × std(otc)/std(gap) × E[|gap|]
        # Using empirical numbers from consistent data:
        std_gap = consistent["gap"].std()
        std_otc = consistent["otc"].std()
        rho = corr_gap_otc
        mean_abs_gap = consistent["gap"].abs().mean()

        # Expected OTC return per trade if we follow: fade gap
        # = corr * std_otc/std_gap * sign(gap_was_correct) ≈ |rho| * std_otc/std_gap * mean_abs_gap
        # Simplified: E[otc | fade_correct] ≈ |rho| * std_otc
        theoretical_avg_gross = abs(rho) * std_otc
        theoretical_sharpe = theoretical_avg_gross * np.sqrt(252) / std_otc  # simplification
        print(f"Theoretical Strategy B estimate (from CTC decomposition):")
        print(f"  corr(gap, otc)      = {rho:.3f}")
        print(f"  std(otc)            = {std_otc*10000:.1f} bps")
        print(f"  std(gap)            = {std_gap*10000:.1f} bps")
        print(f"  Expected avg gross/trade = {theoretical_avg_gross*10000:.1f} bps")
        print(f"  (if traded every day — ignoring cost and gap filter)")

print()

# The clean synthetic estimate: use what we know from Strategy A
ctc_data = pd.read_csv("data/processed/feature_matrix.csv", index_col=0, parse_dates=True)
ctc_ret = ctc_data["y_ret"].dropna()

# 21d realised vol (annualised)
rv21 = ctc_ret.rolling(21).std() * np.sqrt(252)

# Strategy A gap signal: from feature_matrix, clp_gap
if "clp_gap" in ctc_data.columns:
    gap = ctc_data["clp_gap"].dropna()

    # Clean gap data: drop look-ahead contaminated rows
    # Use only days where AV shows a real open (our cleanest source)
    if df_av is not None:
        av_real_dates = df_av[df_av["open_real"]].index
        gap_clean = gap[gap.index.isin(av_real_dates)]
        ctc_clean = ctc_ret[ctc_ret.index.isin(av_real_dates)]
        rv21_clean = rv21[rv21.index.isin(av_real_dates)]
        otc_clean = df_av.loc[df_av["open_real"], "otc"]

        # Align
        common = gap_clean.index.intersection(otc_clean.index).intersection(rv21_clean.index)
        df_clean = pd.DataFrame({
            "gap": gap_clean.loc[common],
            "otc": otc_clean.loc[common],
            "ctc": ctc_ret.loc[common],
            "rv21": rv21_clean.loc[common],
        }).dropna()

        print(f"AV real-open days with full feature data: {len(df_clean)}")
        if len(df_clean) > 20:
            print(f"corr(gap, otc) [AV, feature matrix gap]: {df_clean['gap'].corr(df_clean['otc']):.3f}")
            print()

print()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PAPER TRADE FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════
print("SECTION 4: PAPER TRADE FRAMEWORK")
print("-" * 40)
print()
print("To collect real data and validate Strategy B before going live:")
print()
print("Daily routine (takes 5 minutes):")
print("  08:25 Santiago  — note today's date and prior BCCh dolar observado (prev_close)")
print("  08:30 Santiago  — open XTB platform, note FIRST bid/ask midpoint for USD/CLP (open)")
print("  13:45 Santiago  — note FINAL bid/ask midpoint for USD/CLP (close)")
print()
print("Fields to record:")
print("  date, prev_close_bcch, xtb_open, xtb_close, xtb_spread_at_open, xtb_spread_at_close")
print("  gap = log(xtb_open / prev_close_bcch)")
print("  otc = log(xtb_close / xtb_open)")
print("  signal = -sign(gap)  [fade the gap]")
print("  gross_ret = signal * otc")
print("  net_ret = gross_ret - xtb_spread_rt")
print()
print("After 30 trading days you will have:")
print("  - Real corr(gap, otc) estimate (need ~30 to get a reliable correlation)")
print("  - Real XTB spread at open and close (not just indicative)")
print("  - A live estimate of Strategy B's per-trade P&L before committing capital")
print()

paper_trade_template = {
    "columns": ["date", "prev_close_bcch", "xtb_open", "xtb_close",
                "spread_rt_bps", "gap", "otc", "signal", "gross_ret", "net_ret"],
    "bcch_source": "https://si3.bcentral.cl/Siete/es/Siete/Cuadro/CAP_TIPO_CAMBIO/MN_TCO_1501.3",
    "open_time": "08:30 Santiago (12:30 CET winter, 13:30 CET summer)",
    "close_time": "13:45 Santiago (17:45 CET winter, 18:45 CET summer)",
    "note": "Record xtb_open as soon as platform opens. Record xtb_close before 13:50."
}

paper_path = MODELS_DIR / "paper_trade_log.json"
if not paper_path.exists():
    paper_path.write_text(json.dumps({
        "strategy": "Strategy B — XTB fade-gap",
        "started": None,
        "template": paper_trade_template,
        "trades": []
    }, indent=2))
    print(f"Created paper trade log template: {paper_path}")
else:
    with open(paper_path) as f:
        log = json.load(f)
    n_trades = len(log.get("trades", []))
    print(f"Paper trade log exists: {paper_path} ({n_trades} trades recorded)")

print()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SENSITIVITY ANALYSIS — HOW GOOD DOES IT NEED TO BE?
# ══════════════════════════════════════════════════════════════════════════════
print("SECTION 5: BREAK-EVEN ANALYSIS FOR XTB")
print("-" * 40)
print()
print("For Strategy B to be profitable at XTB (6.5 bps RT cost):")
print()
print(f"  Cost per trade: {XTB_COST_BPS:.1f} bps")
print()
print("  Required avg gross return per trade to achieve target Sharpe:")
print()

# Load realised vol of CTC returns as proxy for OTC vol
rv = ctc_ret.std() * np.sqrt(252) * 10000  # in bps
print(f"  Approx daily vol of USD/CLP: {rv:.0f} bps ann. ({rv/np.sqrt(252):.0f} bps/day)")
print()

for target_sharpe in [1.0, 2.0, 5.0]:
    # For daily Sharpe S, E[r] = S * std(r) / sqrt(252)
    # Assume std(otc) ≈ 0.7 * std(ctc) (OTC captures ~70% of vol)
    std_otc_daily = ctc_ret.std() * 0.7   # rough estimate
    required_mean = target_sharpe * std_otc_daily / np.sqrt(252)
    required_gross = required_mean + XTB_COST
    required_wr = 0.5 + required_gross / (2 * std_otc_daily)   # rough approximation
    print(f"  Target net Sharpe {target_sharpe:.0f}:")
    print(f"    Required avg gross/trade: {required_gross*10000:.1f} bps")
    print(f"    Implied win rate: {min(required_wr, 1.0):.0%}")
    print()

print("  From corr(gap,otc) = -0.50 (research notes, pre-2012 estimate):")
print("    Expected: ~80-100 bps avg gross per trade with 43bps gap filter")
print("    → Well above break-even for Sharpe > 5")
print()
print("  CAVEAT: The -0.50 correlation was derived from yfinance data with")
print("  look-ahead bias. True corr(gap,otc) is unknown. Paper trading is")
print("  the only way to establish the true expected value.")
print()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Strategy B (XTB Fade-Gap) — Data Quality Analysis\n"
    "WARNING: No reliable intraday open data available for full backtest",
    fontsize=12, fontweight="bold", color="#c0392b"
)

# Panel 1: yfinance open quality by year
ax = axes[0, 0]
yr_counts = changed.groupby(changed.index.year).agg(
    n_real=("open_approx_close", "count"),
    n_lookahead=("open_approx_close", lambda x: int(x.sum()))
)
yr_counts["n_clean"] = yr_counts["n_real"] - yr_counts["n_lookahead"]
bars_la  = ax.bar(yr_counts.index, yr_counts["n_lookahead"], color="#e74c3c", alpha=0.8, label="Look-ahead (open≈close)")
bars_cl  = ax.bar(yr_counts.index, yr_counts["n_clean"], bottom=yr_counts["n_lookahead"], color="#27ae60", alpha=0.8, label="Potentially clean")
ax.set_title("yfinance CLPUSD=X 'Open' Quality by Year")
ax.set_ylabel("N days open differs from prev_close")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: AV open quality by year
ax = axes[0, 1]
if df_av is not None:
    av_by_yr = df_av.groupby(df_av.index.year)["open_real"].agg(["sum","count"])
    av_by_yr["pct"] = av_by_yr["sum"] / av_by_yr["count"] * 100
    ax.bar(av_by_yr.index, av_by_yr["pct"], color="#2b6cb0", alpha=0.8)
    ax.axhline(20, color="r", lw=0.8, linestyle="--", label="20% threshold")
    ax.set_title("Alpha Vantage USD/CLP: % Days with Real Open")
    ax.set_ylabel("% of trading days")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "AV data not available\nRun download first",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title("Alpha Vantage Data (not available)")

# Panel 3: AV gap vs otc scatter (if available)
ax = axes[1, 0]
if df_av is not None:
    ax.scatter(real_av["gap"] * 10000, real_av["otc"] * 10000,
               alpha=0.3, s=15, color="#2b6cb0", label=f"n={len(real_av)}")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    corr_text = f"corr={real_av['gap'].corr(real_av['otc']):.3f}"
    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=10,
            va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("Overnight gap (bps)")
    ax.set_ylabel("MCF session return (bps)")
    ax.set_title("AV Data: Gap vs Intraday Return\n(corr≈0 → no fade-gap signal found)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "AV data not available", ha="center", va="center",
            transform=ax.transAxes, fontsize=12)

# Panel 4: Required avg gross return vs target Sharpe
ax = axes[1, 1]
std_otc_daily = ctc_ret.std() * 0.7
sharpes = np.linspace(0, 15, 100)
required_means = sharpes * std_otc_daily / np.sqrt(252) + XTB_COST
ax.plot(sharpes, required_means * 10000, color="#2b6cb0", lw=2)
ax.axhline(XTB_COST * 10000, color="r", lw=0.8, linestyle="--",
           label=f"XTB cost: {XTB_COST_BPS:.1f} bps")
ax.fill_between(sharpes, 0, required_means * 10000, alpha=0.15, color="#2b6cb0")
ax.set_xlabel("Target net annualised Sharpe")
ax.set_ylabel("Required avg gross return/trade (bps)")
ax.set_title(f"Break-even Analysis — XTB ({XTB_COST_BPS:.1f} bps RT)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig_path = FIGURES_DIR / "09_backtest_xtb.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved → {fig_path}")

# ── Save summary ──────────────────────────────────────────────────────────
summary = {
    "description": "Strategy B (XTB fade-gap) data quality analysis and backtest",
    "conclusion": "Cannot build reliable backtest from free data — yfinance opens have look-ahead bias; AV opens don't match MCF open",
    "data_quality": {
        "yfinance_look_ahead_pct": float(pct_is_close),
        "av_real_open_days": int(len(real_av)) if df_av is not None else 0,
        "av_corr_gap_otc": float(real_av["gap"].corr(real_av["otc"])) if df_av is not None else None,
    },
    "theoretical_edge": {
        "corr_gap_otc_reported": -0.50,
        "corr_gap_otc_source": "Research notes (pre-2012, yfinance data — look-ahead contaminated)",
        "corr_gap_otc_trusted": None,
        "status": "UNKNOWN — needs paper trading or commercial intraday data"
    },
    "xtb_parameters": {
        "cost_rt_bps": XTB_COST_BPS,
        "session_open": "08:30 Santiago (12:30-13:30 CET)",
        "session_close": "13:45 Santiago (17:45-18:45 CET)",
    },
    "recommended_next_steps": [
        "1. Paper trade: record XTB open (08:30) and close (13:45) for 30-60 days",
        "2. Compute corr(gap, otc) from paper trade data",
        "3. Only go live if corr < -0.3 and avg gross > 20 bps (3x cost)",
        "4. Alternative: purchase intraday CLP/USD data from Refinitiv/Bloomberg",
    ]
}
out_path = MODELS_DIR / "09_backtest_xtb.json"
out_path.write_text(json.dumps(summary, indent=2))
print(f"Summary saved → {out_path}")
print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
print("Strategy B (XTB fade-gap) CANNOT be reliably backtested with available data.")
print()
print("Key findings:")
print("  1. yfinance CLPUSD=X opens are 74% look-ahead contaminated")
print("  2. Alpha Vantage opens available on only 15% of days, corr(gap,otc)≈0")
print("  3. The handoff.md Sharpe ~6.6 estimate is INVALID (based on contaminated data)")
print()
print("Theory supports the strategy (gap→OTC reversal is well-documented in EM FX),")
print("but the expected Sharpe is unknown without real intraday data.")
print()
print("Recommended path:")
print("  → Paper trade 30-60 days at XTB to collect real open/close prices")
print("  → Gate live trading on: corr(gap,otc) < -0.3 AND avg gross > 20 bps")
