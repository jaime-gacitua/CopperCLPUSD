"""
Experiment registry — version, run, and compare ML experiments.

Each experiment is defined by a named config (feature groups, model params,
walk-forward params). Running an experiment saves a JSON snapshot under
models/experiments/<name>_<YYYYMMDD_HHMMSS>.json containing the full config
and results so results are always reproducible.

Usage:
    # List registered experiments
    uv run python -m copper_clp.experiment list

    # Run a specific experiment
    uv run python -m copper_clp.experiment run v1_copper_only
    uv run python -m copper_clp.experiment run v2_macro_signals

    # Compare all completed runs
    uv run python -m copper_clp.experiment compare
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from copper_clp.config import (
    FEATURE_MATRIX, MODELS_DIR,
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
)

EXPERIMENTS_DIR = MODELS_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


# ── Experiment definitions ─────────────────────────────────────────────────
#
# Each entry defines which feature groups to include.
# Add new experiments here — never edit existing ones (append only).

REGISTRY: dict[str, dict] = {
    "v1_copper_only": {
        "description": "Baseline — copper price signals only (lag returns, momentum, vol, RSI, z-score, slope)",
        "data_granularity": "daily",
        "feature_groups": ["copper", "clp_ar", "calendar"],
        "wf_train_days": WF_TRAIN_DAYS,
        "wf_test_days":  WF_TEST_DAYS,
        "wf_step_days":  WF_STEP_DAYS,
        "models": ["lr", "ridge", "rf", "xgb", "lgb"],
        "known_results": {
            "lr":    {"acc": 0.6002, "sharpe": 3.6141},
            "rf":    {"acc": 0.6013, "sharpe": 3.3866},
            "xgb":  {"acc": 0.5870, "sharpe": 3.0551},
            "lgb":  {"acc": 0.5896, "sharpe": 3.2351},
            "note": "Recorded 2026-06-21 — 70 OOS folds",
        },
    },
    "v2_macro_signals": {
        "description": (
            "Added DXY, VIX, EM peer FX (BRL, PEN, MXN), Oil, Gold, US10Y, IPSA. "
            "Key new features: cu_dxy_spread (copper minus DXY return), "
            "vix_chg, em_composite, clp_vs_em deviation."
        ),
        "data_granularity": "daily",
        "feature_groups": ["copper", "clp_ar", "calendar",
                           "dxy", "vix", "em_fx", "oil", "gold", "us10y", "ipsa",
                           "cross_asset"],
        "wf_train_days": WF_TRAIN_DAYS,
        "wf_test_days":  WF_TEST_DAYS,
        "wf_step_days":  WF_STEP_DAYS,
        "models": ["lr", "ridge", "rf", "xgb", "lgb"],
        "known_results": {
            "lr":    {"acc": 0.6279, "sharpe": 4.1773},
            "rf":    {"acc": 0.6328, "sharpe": 4.3757},
            "xgb":  {"acc": 0.6240, "sharpe": 4.0111},
            "lgb":  {"acc": 0.6205, "sharpe": 4.0050},
            "note": "Recorded 2026-06-21 — 70 OOS folds. Top feature: cu_dxy_spread_lag1",
        },
    },
    "v3_intraday_confirmation": {
        "description": (
            "Added same-day (day T) signals observable before CLP closes at 5pm Santiago: "
            "copper T-close return (COMEX settles 2pm Santiago), DXY T-return, VIX T-change, "
            "EM peer FX T-returns (BRL/PEN/MXN), S&P 500 T-return (SPY, closes 5pm Santiago), "
            "overnight CLP gap (T-1 close → T open), CLP open-to-close intraday move, "
            "and same-day copper/DXY spread. All v2 T-1 features retained."
        ),
        "data_granularity": "daily",
        "feature_groups": ["copper", "clp_ar", "calendar",
                           "dxy", "vix", "em_fx", "oil", "gold", "us10y", "ipsa",
                           "cross_asset", "intraday"],
        "wf_train_days": WF_TRAIN_DAYS,
        "wf_test_days":  WF_TEST_DAYS,
        "wf_step_days":  WF_STEP_DAYS,
        "models": ["lr", "ridge", "rf", "xgb", "lgb"],
        "execution_note": (
            "Target is still y_ret (T-1 close → T close). "
            "Same-day features are NOT lagged — they are observable before the T close order. "
            "This models the real workflow: see T-day data, place order near T close, exit T+1 close."
        ),
    },
}


# ── Feature group selectors ────────────────────────────────────────────────

GROUP_PREFIXES: dict[str, list[str]] = {
    "copper":      ["cu_"],
    "clp_ar":      ["clp_ret_"],
    "calendar":    ["dow_"],
    "dxy":         ["dxy_"],
    "vix":         ["vix_"],
    "em_fx":       ["brl_", "pen_", "mxn_", "em_composite", "clp_vs_em"],
    "oil":         ["oil_"],
    "gold":        ["gold_"],
    "us10y":       ["us10y_"],
    "ipsa":        ["ipsa_"],
    "cross_asset": ["cu_dxy_", "cu_vix_"],
    # v3: same-day intraday features (prefix "id_")
    "intraday":    ["id_"],
}


def select_features(feat: pd.DataFrame, groups: list[str]) -> list[str]:
    """Return feature column names matching the requested groups."""
    all_feat_cols = [c for c in feat.columns if not c.startswith("y_")]
    prefixes = []
    for g in groups:
        prefixes.extend(GROUP_PREFIXES.get(g, []))
    return [c for c in all_feat_cols if any(c.startswith(p) for p in prefixes)]


# ── Walk-forward engine ────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, ann: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(ann) * returns.mean() / returns.std())


def _run_wf(feat: pd.DataFrame, fcols: list[str], cfg: dict) -> dict:
    X_all = feat[fcols].values
    y_dir = (feat["y_ret"].values > 0).astype(int)
    y_ret = feat["y_ret"].values
    dates = feat.index
    n     = len(X_all)

    train_d = cfg["wf_train_days"]
    test_d  = cfg["wf_test_days"]
    step_d  = cfg["wf_step_days"]

    folds = []
    for t0 in range(train_d, n - test_d, step_d):
        X_tr = X_all[t0 - train_d : t0]
        y_tc = y_dir[t0 - train_d : t0]
        y_tr = y_ret[t0 - train_d : t0]
        X_te_raw = X_all[t0 : t0 + test_d]
        y_te_c   = y_dir[t0 : t0 + test_d]
        y_te_r   = y_ret[t0 : t0 + test_d]

        ok_tr = ~np.isnan(X_tr).any(axis=1)
        ok_te = ~np.isnan(X_te_raw).any(axis=1)
        X_tr, y_tc, y_tr = X_tr[ok_tr], y_tc[ok_tr], y_tr[ok_tr]
        X_te = X_te_raw[ok_te]
        y_tec, y_ter = y_te_c[ok_te], y_te_r[ok_te]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        fold = {
            "date_start": str(dates[t0].date()),
            "date_end":   str(dates[min(t0 + test_d - 1, n - 1)].date()),
        }

        lr = LogisticRegression(max_iter=1000, C=0.1)
        lr.fit(X_tr_s, y_tc)
        p = lr.predict(X_te_s)
        fold["lr"] = {"acc": float(accuracy_score(y_tec, p)),
                      "f1":  float(f1_score(y_tec, p, zero_division=0)),
                      "sharpe": _sharpe(np.where(p==1,1,-1) * y_ter)}

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_s, y_tr)
        p_r = ridge.predict(X_te_s)
        fold["ridge"] = {"mae": float(mean_absolute_error(y_ter, p_r)),
                         "dir_acc": float(np.mean(np.sign(p_r) == np.sign(y_ter))),
                         "sharpe": _sharpe(np.sign(p_r) * y_ter)}

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tc)
        p = rf.predict(X_te)
        fold["rf"] = {"acc": float(accuracy_score(y_tec, p)),
                      "f1":  float(f1_score(y_tec, p, zero_division=0)),
                      "sharpe": _sharpe(np.where(p==1,1,-1) * y_ter)}

        xm = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                eval_metric="logloss", random_state=42, verbosity=0)
        xm.fit(X_tr, y_tc)
        p = xm.predict(X_te)
        fold["xgb"] = {"acc": float(accuracy_score(y_tec, p)),
                       "f1":  float(f1_score(y_tec, p, zero_division=0)),
                       "sharpe": _sharpe(np.where(p==1,1,-1) * y_ter)}

        lm = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  random_state=42, verbosity=-1)
        lm.fit(X_tr, y_tc)
        p = lm.predict(X_te)
        fold["lgb"] = {"acc": float(accuracy_score(y_tec, p)),
                       "f1":  float(f1_score(y_tec, p, zero_division=0)),
                       "sharpe": _sharpe(np.where(p==1,1,-1) * y_ter)}

        folds.append(fold)

    model_keys = ["lr", "ridge", "rf", "xgb", "lgb"]
    summary = {}
    for m in model_keys:
        keys = list(folds[0][m].keys())
        summary[m] = {k: round(float(np.mean([f[m][k] for f in folds])), 4) for k in keys}

    # Full-sample XGBoost feature importance
    ok = ~np.isnan(X_all).any(axis=1)
    xf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             eval_metric="logloss", random_state=42, verbosity=0)
    xf.fit(X_all[ok], y_dir[ok])
    imp = sorted(zip(fcols, xf.feature_importances_.tolist()), key=lambda x: -x[1])

    return {
        "n_folds": len(folds),
        "summary": summary,
        "feature_importance": [{"feature": n, "importance": round(v, 4)} for n, v in imp[:20]],
        "folds": folds,
    }


# ── Public API ─────────────────────────────────────────────────────────────

def run_experiment(name: str) -> Path:
    """Run a registered experiment and save results to models/experiments/."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown experiment '{name}'. Available: {list(REGISTRY)}")

    cfg  = REGISTRY[name]
    feat = pd.read_csv(FEATURE_MATRIX, index_col=0, parse_dates=True)
    fcols = select_features(feat, cfg["feature_groups"])

    print(f"\nExperiment: {name}")
    print(f"  {cfg['description']}")
    print(f"  Feature groups : {cfg['feature_groups']}")
    print(f"  Features       : {len(fcols)}")
    print(f"  WF             : {cfg['wf_train_days']}d train / {cfg['wf_test_days']}d test / {cfg['wf_step_days']}d step")
    print(f"  Rows           : {len(feat)}")
    print()

    wf = _run_wf(feat, fcols, cfg)
    print(f"  {wf['n_folds']} folds complete.")
    print(f"  Summary:")
    for m, s in wf["summary"].items():
        print(f"    {m:6s}: {s}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    record = {
        "experiment":   name,
        "timestamp":    timestamp,
        "config":       cfg,
        "feature_cols": fcols,
        "n_features":   len(fcols),
        "results":      wf,
    }
    # Remove known_results from saved config (that's just documentation)
    record["config"] = {k: v for k, v in cfg.items() if k != "known_results"}

    out = EXPERIMENTS_DIR / f"{name}_{timestamp}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\n  Saved → {out}")
    return out


def list_experiments() -> None:
    """Print registered experiments and any completed runs."""
    print(f"\n{'='*65}")
    print("Registered experiments:")
    for name, cfg in REGISTRY.items():
        runs = sorted(EXPERIMENTS_DIR.glob(f"{name}_*.json"))
        status = f"{len(runs)} run(s)" if runs else "not yet run"
        print(f"  {name}")
        print(f"    {cfg['description'][:72]}")
        print(f"    Status: {status}")
        if runs:
            last = json.loads(runs[-1].read_text())
            s = last["results"]["summary"]
            best_acc = max(s[m]["acc"] for m in ["lr","rf","xgb","lgb"])
            best_sh  = max(s[m]["sharpe"] for m in ["lr","rf","xgb","lgb"])
            print(f"    Last run: {last['timestamp']}  "
                  f"best acc={best_acc:.4f}  best Sharpe={best_sh:.4f}")
    print(f"{'='*65}\n")


def compare_experiments() -> None:
    """Print a comparison table across all completed experiment runs."""
    all_runs = sorted(EXPERIMENTS_DIR.glob("*.json"))
    if not all_runs:
        print("No completed runs yet. Run: uv run python -m copper_clp.experiment run <name>")
        return

    rows = []
    for path in all_runs:
        d = json.loads(path.read_text())
        s = d["results"]["summary"]
        top_feat = d["results"]["feature_importance"][0]["feature"] if d["results"]["feature_importance"] else "—"
        for m in ["lr", "rf", "xgb", "lgb"]:
            if m in s:
                rows.append({
                    "experiment": d["experiment"],
                    "timestamp":  d["timestamp"][:15],
                    "n_features": d["n_features"],
                    "model":      m,
                    "acc":        s[m]["acc"],
                    "sharpe":     s[m]["sharpe"],
                    "top_feature": top_feat,
                })

    df = pd.DataFrame(rows)
    print(f"\n{'='*90}")
    print("Experiment comparison (OOS walk-forward averages):")
    print(f"{'='*90}")
    print(df.to_string(index=False, float_format="{:.4f}".format))

    print(f"\nBest per experiment:")
    for exp, grp in df.groupby("experiment"):
        best = grp.loc[grp["acc"].idxmax()]
        print(f"  {exp:25s}  {best['model']:5s}  acc={best['acc']:.4f}  "
              f"Sharpe={best['sharpe']:.4f}  top={best['top_feature']}")
    print(f"{'='*90}\n")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        list_experiments()
    elif cmd == "run":
        if len(sys.argv) < 3:
            print("Usage: python -m copper_clp.experiment run <experiment_name>")
            sys.exit(1)
        run_experiment(sys.argv[2])
    elif cmd == "compare":
        compare_experiments()
    elif cmd == "run-all":
        for name in REGISTRY:
            run_experiment(name)
        compare_experiments()
    else:
        print(f"Unknown command '{cmd}'. Use: list | run <name> | compare | run-all")
        sys.exit(1)
