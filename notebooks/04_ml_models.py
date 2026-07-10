"""
Experiment 4: ML signal models with walk-forward validation

Models:
  - Logistic Regression (direction)
  - Ridge Regression (return magnitude)
  - Random Forest (direction + return)
  - XGBoost (direction + return)
  - LightGBM (direction + return)

Walk-forward: train on 3 years, test on next 252 days, roll 63 days.
Metrics: accuracy, precision, recall, F1, Sharpe of signal-driven strategy,
         MAE, RMSE for regression.

Output: results/04_ml_results.json, figures/04_ml_*.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

OUT = Path("results"); OUT.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

feat = pd.read_csv("data/feature_matrix.csv", index_col=0, parse_dates=True)
FEATURE_COLS = [c for c in feat.columns if not c.startswith("y_")]

X_all = feat[FEATURE_COLS].values
y_dir = (feat["y_dir"].values > 0).astype(int)   # binary: 1=CLP strengthens vs dollar
y_ret = feat["y_ret"].values

dates = feat.index

TRAIN_DAYS = 252 * 3   # 3 years
TEST_DAYS  = 252        # 1 year
STEP_DAYS  = 63         # roll every quarter

def sharpe(returns, ann=252):
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(ann) * returns.mean() / returns.std())

def walk_forward(X, y_cls, y_reg, dates, train=TRAIN_DAYS, test=TEST_DAYS, step=STEP_DAYS):
    n = len(X)
    starts = range(train, n - test, step)

    all_results = []
    for t0 in starts:
        X_tr, y_tr_cls = X[t0 - train : t0], y_cls[t0 - train : t0]
        y_tr_reg = y_reg[t0 - train : t0]
        X_te = X[t0 : t0 + test]
        y_te_cls = y_cls[t0 : t0 + test]
        y_te_reg = y_reg[t0 : t0 + test]
        te_dates = dates[t0 : t0 + test]
        te_ret = feat["y_ret"].values[t0 : t0 + test]

        # Drop NaN rows consistently across train/test
        tr_mask = ~np.isnan(X_tr).any(axis=1)
        te_mask = ~np.isnan(X_te).any(axis=1)
        X_tr, y_tr_cls, y_tr_reg = X_tr[tr_mask], y_tr_cls[tr_mask], y_tr_reg[tr_mask]
        X_te_clean = X_te[te_mask]
        y_te_cls_c = y_te_cls[te_mask]
        y_te_reg_c = y_te_reg[te_mask]
        te_ret_c   = te_ret[te_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te_clean)

        fold = {"date_start": str(te_dates[0].date()), "date_end": str(te_dates[-1].date())}

        # ── Logistic Regression ──────────────────────────────────────────
        lr = LogisticRegression(max_iter=1000, C=0.1)
        lr.fit(X_tr_s, y_tr_cls)
        pred_lr = lr.predict(X_te_s)
        signal_lr = np.where(pred_lr == 1, 1, -1)
        strat_lr = signal_lr * te_ret_c
        fold["lr"] = {
            "acc": round(accuracy_score(y_te_cls_c, pred_lr), 4),
            "f1":  round(f1_score(y_te_cls_c, pred_lr, zero_division=0), 4),
            "sharpe": round(sharpe(strat_lr), 4),
        }

        # ── Ridge Regression ─────────────────────────────────────────────
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_s, y_tr_reg)
        pred_ridge = ridge.predict(X_te_s)
        signal_ridge = np.sign(pred_ridge)
        strat_ridge = signal_ridge * te_ret_c
        mae_ridge = mean_absolute_error(y_te_reg_c, pred_ridge)
        fold["ridge"] = {
            "mae": round(float(mae_ridge), 6),
            "sharpe": round(sharpe(strat_ridge), 4),
            "dir_acc": round(float(np.mean(np.sign(pred_ridge) == np.sign(y_te_reg_c))), 4),
        }

        # ── Random Forest ─────────────────────────────────────────────────
        rf_cls = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf_cls.fit(X_tr, y_tr_cls)
        pred_rf = rf_cls.predict(X_te_clean)
        signal_rf = np.where(pred_rf == 1, 1, -1)
        strat_rf = signal_rf * te_ret_c
        fold["rf"] = {
            "acc": round(accuracy_score(y_te_cls_c, pred_rf), 4),
            "f1":  round(f1_score(y_te_cls_c, pred_rf, zero_division=0), 4),
            "sharpe": round(sharpe(strat_rf), 4),
        }

        # ── XGBoost ───────────────────────────────────────────────────────
        xgb_cls = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                     eval_metric="logloss", random_state=42, verbosity=0)
        xgb_cls.fit(X_tr, y_tr_cls)
        pred_xgb = xgb_cls.predict(X_te_clean)
        signal_xgb = np.where(pred_xgb == 1, 1, -1)
        strat_xgb = signal_xgb * te_ret_c
        fold["xgb"] = {
            "acc": round(accuracy_score(y_te_cls_c, pred_xgb), 4),
            "f1":  round(f1_score(y_te_cls_c, pred_xgb, zero_division=0), 4),
            "sharpe": round(sharpe(strat_xgb), 4),
        }

        # ── LightGBM ──────────────────────────────────────────────────────
        lgb_cls = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      random_state=42, verbosity=-1)
        lgb_cls.fit(X_tr, y_tr_cls)
        pred_lgb = lgb_cls.predict(X_te_clean)
        signal_lgb = np.where(pred_lgb == 1, 1, -1)
        strat_lgb = signal_lgb * te_ret_c
        fold["lgb"] = {
            "acc": round(accuracy_score(y_te_cls_c, pred_lgb), 4),
            "f1":  round(f1_score(y_te_cls_c, pred_lgb, zero_division=0), 4),
            "sharpe": round(sharpe(strat_lgb), 4),
        }

        all_results.append(fold)
        print(f"  {fold['date_start']} → {fold['date_end']} | "
              f"LR acc={fold['lr']['acc']:.3f} RF={fold['rf']['acc']:.3f} "
              f"XGB={fold['xgb']['acc']:.3f} LGB={fold['lgb']['acc']:.3f}")

    return all_results

print("Running walk-forward validation...")
wf_results = walk_forward(X_all, y_dir, y_ret, dates)

# ── Aggregate across folds ─────────────────────────────────────────────────
models = ["lr", "ridge", "rf", "xgb", "lgb"]
summary = {}
for m in models:
    keys = list(wf_results[0][m].keys())
    summary[m] = {k: round(float(np.mean([r[m][k] for r in wf_results])), 4) for k in keys}

print("\n=== WALK-FORWARD SUMMARY ===")
for m, s in summary.items():
    print(f"  {m:6s}: {s}")

# ── Feature importance (full-sample XGBoost) ──────────────────────────────
scaler = StandardScaler()
X_s = scaler.fit_transform(X_all)
xgb_full = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               eval_metric="logloss", random_state=42, verbosity=0)
xgb_full.fit(X_all, y_dir)
importances = xgb_full.feature_importances_
feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[:20]
print("\nTop 20 features (XGBoost):")
for name, imp in feat_imp:
    print(f"  {name:25s} {imp:.4f}")

results = {
    "walk_forward_folds": wf_results,
    "summary": summary,
    "feature_importance_top20": [{"feature": n, "importance": round(float(i), 4)} for n, i in feat_imp],
}
(OUT / "04_ml_results.json").write_text(json.dumps(results, indent=2))
print("\nSaved results/04_ml_results.json")

# ── Figure: accuracy and Sharpe per model ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#aaa")
    ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)

cls_models = ["lr", "rf", "xgb", "lgb"]
colors = ["#4e9af1", "#7be495", "#e8a020", "#ff6b6b"]

ax = axes[0]
accs = [summary[m].get("acc", 0) for m in cls_models]
bars = ax.bar(cls_models, accs, color=colors)
ax.axhline(0.5, color="#555", lw=0.8, ls="--", label="random baseline")
ax.set_title("Direction accuracy (walk-forward avg)", color="white")
ax.set_ylabel("Accuracy", color="#aaa")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#333")
for bar, v in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f"{v:.3f}", ha="center", color="white", fontsize=9)

ax = axes[1]
sharpes = [summary[m].get("sharpe", 0) for m in cls_models]
bars = ax.bar(cls_models, sharpes, color=colors)
ax.axhline(0, color="#555", lw=0.8, ls="--")
ax.set_title("Signal Sharpe ratio (walk-forward avg)", color="white")
ax.set_ylabel("Sharpe", color="#aaa")
for bar, v in zip(bars, sharpes):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", color="white", fontsize=9)

ax = axes[2]
feat_names = [x[0] for x in feat_imp[:10]]
feat_vals  = [x[1] for x in feat_imp[:10]]
ax.barh(feat_names[::-1], feat_vals[::-1], color="#e8a020")
ax.set_title("Top 10 features (XGBoost importance)", color="white")
ax.set_xlabel("Importance", color="#aaa")

fig.suptitle("Experiment 4: ML Models Walk-Forward", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "04_ml_models.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved figures/04_ml_models.png")
