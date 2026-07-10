"""
Walk-forward training and evaluation of all signal models.

Models:
  LogisticRegression, Ridge, RandomForest, XGBoost, LightGBM

Each fold: train on WF_TRAIN_DAYS, test on WF_TEST_DAYS, step by WF_STEP_DAYS.
Only copper features with lag >= MIN_COPPER_LAG are used (enforced in features.py).
"""
import json
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
from copper_clp.features import get_feature_cols


def sharpe(returns: np.ndarray, ann: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(ann) * returns.mean() / returns.std())


def run_walk_forward(feat: pd.DataFrame | None = None) -> dict:
    """
    Run walk-forward validation across all models.

    Returns dict with fold-level results and aggregate summary.
    """
    if feat is None:
        feat = pd.read_csv(FEATURE_MATRIX, index_col=0, parse_dates=True)

    FCOLS = get_feature_cols(feat)
    X_all = feat[FCOLS].values
    y_dir = (feat["y_ret"].values > 0).astype(int)  # 1 = USD/CLP rises (CLP weakens)
    y_ret = feat["y_ret"].values
    dates = feat.index
    n = len(X_all)

    all_folds = []
    starts = range(WF_TRAIN_DAYS, n - WF_TEST_DAYS, WF_STEP_DAYS)

    for t0 in starts:
        X_tr = X_all[t0 - WF_TRAIN_DAYS : t0]
        y_tr_cls = y_dir[t0 - WF_TRAIN_DAYS : t0]
        y_tr_reg = y_ret[t0 - WF_TRAIN_DAYS : t0]
        X_te_raw = X_all[t0 : t0 + WF_TEST_DAYS]
        y_te_cls = y_dir[t0 : t0 + WF_TEST_DAYS]
        y_te_ret = y_ret[t0 : t0 + WF_TEST_DAYS]

        # Remove NaN rows (from initial rolling window warm-up)
        tr_ok = ~np.isnan(X_tr).any(axis=1)
        te_ok = ~np.isnan(X_te_raw).any(axis=1)
        X_tr, y_tr_cls, y_tr_reg = X_tr[tr_ok], y_tr_cls[tr_ok], y_tr_reg[tr_ok]
        X_te = X_te_raw[te_ok]
        y_tc, y_tr_e = y_te_cls[te_ok], y_te_ret[te_ok]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        fold = {
            "date_start": str(dates[t0].date()),
            "date_end":   str(dates[min(t0 + WF_TEST_DAYS - 1, n - 1)].date()),
        }

        # ── Logistic Regression ──────────────────────────────────────────
        lr = LogisticRegression(max_iter=1000, C=0.1)
        lr.fit(X_tr_s, y_tr_cls)
        p = lr.predict(X_te_s)
        sig = np.where(p == 1, 1, -1)
        fold["lr"] = {
            "acc": round(float(accuracy_score(y_tc, p)), 4),
            "f1":  round(float(f1_score(y_tc, p, zero_division=0)), 4),
            "sharpe": round(sharpe(sig * y_tr_e), 4),
        }

        # ── Ridge Regression ─────────────────────────────────────────────
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_s, y_tr_reg)
        p_r = ridge.predict(X_te_s)
        fold["ridge"] = {
            "mae": round(float(mean_absolute_error(y_tr_e, p_r)), 6),
            "dir_acc": round(float(np.mean(np.sign(p_r) == np.sign(y_tr_e))), 4),
            "sharpe": round(sharpe(np.sign(p_r) * y_tr_e), 4),
        }

        # ── Random Forest ─────────────────────────────────────────────────
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr_cls)
        p = rf.predict(X_te)
        sig = np.where(p == 1, 1, -1)
        fold["rf"] = {
            "acc": round(float(accuracy_score(y_tc, p)), 4),
            "f1":  round(float(f1_score(y_tc, p, zero_division=0)), 4),
            "sharpe": round(sharpe(sig * y_tr_e), 4),
        }

        # ── XGBoost ───────────────────────────────────────────────────────
        xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                   eval_metric="logloss", random_state=42, verbosity=0)
        xgb_m.fit(X_tr, y_tr_cls)
        p = xgb_m.predict(X_te)
        sig = np.where(p == 1, 1, -1)
        fold["xgb"] = {
            "acc": round(float(accuracy_score(y_tc, p)), 4),
            "f1":  round(float(f1_score(y_tc, p, zero_division=0)), 4),
            "sharpe": round(sharpe(sig * y_tr_e), 4),
        }

        # ── LightGBM ──────────────────────────────────────────────────────
        lgb_m = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                    random_state=42, verbosity=-1)
        lgb_m.fit(X_tr, y_tr_cls)
        p = lgb_m.predict(X_te)
        sig = np.where(p == 1, 1, -1)
        fold["lgb"] = {
            "acc": round(float(accuracy_score(y_tc, p)), 4),
            "f1":  round(float(f1_score(y_tc, p, zero_division=0)), 4),
            "sharpe": round(sharpe(sig * y_tr_e), 4),
        }

        all_folds.append(fold)
        print(f"  {fold['date_start']} LR={fold['lr']['acc']:.3f} "
              f"RF={fold['rf']['acc']:.3f} XGB={fold['xgb']['acc']:.3f} "
              f"LGB={fold['lgb']['acc']:.3f}")

    # Aggregate
    models = ["lr", "ridge", "rf", "xgb", "lgb"]
    summary = {}
    for m in models:
        keys = list(all_folds[0][m].keys())
        summary[m] = {k: round(float(np.mean([f[m][k] for f in all_folds])), 4) for k in keys}

    # Full-sample XGBoost feature importance
    X_clean = X_all[~np.isnan(X_all).any(axis=1)]
    y_clean = y_dir[~np.isnan(X_all).any(axis=1)]
    xgb_full = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   eval_metric="logloss", random_state=42, verbosity=0)
    xgb_full.fit(X_clean, y_clean)
    imp = sorted(zip(FCOLS, xgb_full.feature_importances_.tolist()), key=lambda x: -x[1])

    results = {
        "folds": all_folds,
        "summary": summary,
        "feature_importance": [{"feature": n, "importance": round(v, 4)} for n, v in imp[:20]],
        "lag_policy": f"MIN_COPPER_LAG={1} day — all copper features lag >= 1d before trade date",
    }

    out = MODELS_DIR / "walk_forward_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")
    print("Summary:")
    for m, s in summary.items():
        print(f"  {m:6s}: {s}")
    return results


if __name__ == "__main__":
    run_walk_forward()
