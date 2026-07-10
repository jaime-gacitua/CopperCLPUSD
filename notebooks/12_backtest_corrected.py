"""
Walk-forward backtest on the CORRECTED feature matrix (feature_matrix_td.csv).

Question: does ANY edge survive once the yfinance dating artifact is removed?

Models (quarterly refit, expanding window from 2020-01-01, OOS 2021+):
  A. logit3_strict : logistic, [clp_ret, cu_ret1_lag, cu_ret5_lag]  (old spec, strict copper)
  B. logit3_same   : logistic, [clp_ret, cu_ret1_same, cu_ret5_same] (old spec, same-day copper)
  C. logit_full    : logistic on all features (L2)
  D. gbm_full      : HistGradientBoosting on all features

Costs: 6.5 bps RT (XTB-class) and 40 bps RT (Capitaria, measured 2026-07-10).

Run:  uv run python notebooks/12_backtest_corrected.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/feature_matrix_td.csv", index_col=0, parse_dates=True)

FEATSETS = {
    "logit3_strict": ["clp_ret", "cu_ret1_lag", "cu_ret5_lag"],
    "logit3_same":   ["clp_ret", "cu_ret1_same", "cu_ret5_same"],
}
ALL_FC = [c for c in df.columns if c not in ("y_ret", "y_dir")]
FEATSETS["logit_full"] = ALL_FC
FEATSETS["gbm_full"]   = ALL_FC

def walk_forward(fc, model_kind):
    data = df.dropna(subset=fc).copy()
    data = data[data.index >= "2020-01-01"]
    recs = []
    qs = pd.date_range("2021-01-01", data.index[-1], freq="QS")
    for i, q0 in enumerate(qs):
        q1 = qs[i + 1] if i + 1 < len(qs) else data.index[-1] + pd.Timedelta(days=1)
        tr = data[data.index < q0]
        te = data[(data.index >= q0) & (data.index < q1)]
        if len(tr) < 100 or len(te) == 0:
            continue
        if model_kind == "gbm":
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                               learning_rate=0.05, random_state=42)
            m.fit(tr[fc], tr["y_dir"])
            p = m.predict_proba(te[fc])[:, 1]
        else:
            sc = StandardScaler().fit(tr[fc])
            m = LogisticRegression(max_iter=2000).fit(sc.transform(tr[fc]), tr["y_dir"])
            p = m.predict_proba(sc.transform(te[fc]))[:, 1]
        for d, pi, r in zip(te.index, p, te.itertuples()):
            recs.append((d, pi, r.clp_ret, r.y_ret))
    return pd.DataFrame(recs, columns=["date", "prob", "gap", "y"]).set_index("date")

def evaluate(bt, ph, gap_th, cost_bps):
    t = bt[((bt["prob"] > ph) | (bt["prob"] < 1 - ph)) & (bt["gap"].abs() > gap_th)].copy()
    if len(t) < 20:
        return None
    t["gross"] = np.where(t["prob"] > 0.5, 1, -1) * t["y"]
    net = t["gross"] - cost_bps / 1e4
    yrs = (bt.index[-1] - bt.index[0]).days / 365.25
    return dict(n=len(t), per_yr=round(len(t) / yrs, 1),
                win=round(float((t["gross"] > 0).mean()), 3),
                avg_gross_bps=round(float(t["gross"].mean() * 1e4), 1),
                gross_sharpe=round(float(np.sqrt(252) * t["gross"].mean() / t["gross"].std()), 2),
                net_sharpe=round(float(np.sqrt(252) * net.mean() / net.std()), 2))

results = {}
for name, fc in FEATSETS.items():
    kind = "gbm" if name.startswith("gbm") else "logit"
    bt = walk_forward(fc, kind)
    print(f"\n=== {name}  (OOS {bt.index[0].date()} -> {bt.index[-1].date()}, {len(bt)} days) ===")
    print(f"  P(up) spread: 5-95 pct = [{bt['prob'].quantile(.05):.3f}, {bt['prob'].quantile(.95):.3f}]")
    print(f"  directional accuracy (all days): {((bt['prob']>0.5)==(bt['y']>0)).mean():.3f}")
    results[name] = {}
    for ph in (0.55, 0.60, 0.65):
        for g in (0.0, 0.0043, 0.010):
            for cost in (6.5, 40):
                r = evaluate(bt, ph, g, cost)
                if r:
                    key = f"p{ph}_g{int(g*1e4)}bps_c{cost}"
                    results[name][key] = r
                    if cost == 6.5:
                        print(f"  p>{ph} |gap|>{g*1e4:>3.0f}bps: n={r['n']:>4} ({r['per_yr']}/yr) "
                              f"win={r['win']:.2f} gross_sh={r['gross_sharpe']:+.2f} "
                              f"net@6.5={r['net_sharpe']:+.2f} "
                              f"net@40={results[name][f'p{ph}_g{int(g*1e4)}bps_c40']['net_sharpe'] if f'p{ph}_g{int(g*1e4)}bps_c40' in results[name] else float('nan'):+.2f}")

out = ROOT / "models/12_backtest_corrected.json"
out.write_text(json.dumps({"description": "Walk-forward on corrected (TwelveData-dated) matrix",
                           "date": "2026-07-10", "results": results}, indent=1))
print(f"\nSaved -> {out}")
