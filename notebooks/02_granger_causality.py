"""
Experiment 2: Granger causality tests
- Does copper price Granger-cause CLP/USD? (and vice versa)
- Test at multiple lag orders (1, 2, 5, 10, 21 days)
- Also test on log-levels and log-returns
- Output: results/02_granger.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/daily_panel.csv", index_col=0, parse_dates=True)
df["log_copper"] = np.log(df["copper"])
df["log_usd_clp"] = np.log(df["usd_clp"])
df["r_copper"] = df["log_copper"].diff()
df["r_usd_clp"] = df["log_usd_clp"].diff()
df = df.dropna()

results = {}

# ── Stationarity (ADF) ─────────────────────────────────────────────────────
for col in ["log_copper", "log_usd_clp", "r_copper", "r_usd_clp"]:
    adf_stat, p_val, _, _, crit, _ = adfuller(df[col], autolag="AIC")
    results[f"adf_{col}"] = {"stat": round(adf_stat, 4), "p": round(p_val, 4), "stationary": bool(p_val < 0.05)}
    print(f"ADF {col}: stat={adf_stat:.3f} p={p_val:.4f} {'STATIONARY' if p_val<0.05 else 'non-stationary'}")

# ── Granger tests on returns (stationary) ─────────────────────────────────
max_lag = 21
data_returns = df[["r_usd_clp", "r_copper"]].copy()  # order: [y, x] → x→y?

print("\n--- Granger: copper returns → CLP returns ---")
gc_cu_to_clp = grangercausalitytests(data_returns, maxlag=max_lag, verbose=False)

print("--- Granger: CLP returns → copper returns ---")
data_rev = df[["r_copper", "r_usd_clp"]].copy()
gc_clp_to_cu = grangercausalitytests(data_rev, maxlag=max_lag, verbose=False)

gc_results = {}
for lag in [1, 2, 3, 5, 10, 21]:
    p_cu_to_clp = gc_cu_to_clp[lag][0]["ssr_ftest"][1]
    p_clp_to_cu = gc_clp_to_cu[lag][0]["ssr_ftest"][1]
    f_cu_to_clp = gc_cu_to_clp[lag][0]["ssr_ftest"][0]
    f_clp_to_cu = gc_clp_to_cu[lag][0]["ssr_ftest"][0]
    gc_results[str(lag)] = {
        "copper_causes_clp_p": round(p_cu_to_clp, 4),
        "copper_causes_clp_f": round(f_cu_to_clp, 4),
        "copper_causes_clp_sig": bool(p_cu_to_clp < 0.05),
        "clp_causes_copper_p": round(p_clp_to_cu, 4),
        "clp_causes_copper_f": round(f_clp_to_cu, 4),
        "clp_causes_copper_sig": bool(p_clp_to_cu < 0.05),
    }
    print(f"  Lag {lag:2d}: copper→CLP p={p_cu_to_clp:.4f} {'***' if p_cu_to_clp<0.01 else '*' if p_cu_to_clp<0.05 else ''}"
          f"  |  CLP→copper p={p_clp_to_cu:.4f} {'***' if p_clp_to_cu<0.01 else '*' if p_clp_to_cu<0.05 else ''}")

results["granger"] = gc_results
(OUT / "02_granger.json").write_text(json.dumps(results, indent=2))
print("\nSaved results/02_granger.json")
