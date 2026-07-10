"""
TimesFM 2.0-500M experiment — copper → CLP/USD forecasting.

The 500M model (google/timesfm-2.0-500m-pytorch) has the same architecture as
the 200M but with 50 transformer layers instead of 20.  The timesfm PyPI
package (2.0.1) only ships a 200M class, so we create a drop-in 500M variant
by subclassing the internal module and overriding the config.

Run:
    uv run python notebooks/05b_timesfm_500m.py
"""
import sys, json, dataclasses
import numpy as np
import pandas as pd
import torch

# ── Build a 500M module by overriding the layer count ─────────────────────
from timesfm.timesfm_2p5 import timesfm_2p5_base, timesfm_2p5_torch
from timesfm import configs

# Same dimensions as 200M but 50 layers → ~500M params
@dataclasses.dataclass(frozen=True)
class TimesFM_500M_Definition(timesfm_2p5_base.TimesFM_2p5_200M_Definition):
    stacked_transformers: configs.StackedTransformersConfig = dataclasses.field(
        default_factory=lambda: configs.StackedTransformersConfig(
            num_layers=50,
            transformer=configs.TransformerConfig(
                model_dims=1280,
                hidden_dims=1280,
                num_heads=16,
                attention_norm="rms",
                feedforward_norm="rms",
                qk_norm="rms",
                use_bias=False,
                use_rotary_position_embeddings=True,
                ff_activation="swish",
                fuse_qkv=True,
            ),
        )
    )


class TimesFM_500M_torch_module(timesfm_2p5_torch.TimesFM_2p5_200M_torch_module):
    """500M variant — 50 transformer layers."""
    config = TimesFM_500M_Definition()


class TimesFM_500M_torch(timesfm_2p5_torch.TimesFM_2p5_200M_torch):
    """TimesFM 2.0 with 500M parameters (50-layer PyTorch variant)."""
    DEFAULT_REPO_ID = "google/timesfm-2.0-500m-pytorch"
    WEIGHTS_FILENAME = "model.safetensors"

    def __init__(self, torch_compile: bool = True, config=None, **kwargs):
        # Call grandparent (TimesFM_2p5) __init__ to skip the 200M module init
        timesfm_2p5_base.TimesFM_2p5.__init__(self)
        self.model = TimesFM_500M_torch_module()
        self.torch_compile = torch_compile
        if config is not None:
            self._hub_mixin_config = config


# ── Paths and config ───────────────────────────────────────────────────────
sys.path.insert(0, ".")
from copper_clp.config import (
    FEATURE_MATRIX, DAILY_PANEL, MODELS_DIR, FIGURES_DIR,
    TIMESFM_CONTEXT, TIMESFM_HORIZON,
)
from timesfm import ForecastConfig

REPO_500M   = "google/timesfm-2.0-500m-pytorch"
CONTEXT     = TIMESFM_CONTEXT   # 512 days
HORIZON     = TIMESFM_HORIZON   # 21 days
N_TEST_WINDOWS = 9
STEP_DAYS   = 126               # re-evaluate every 6 months
MIN_LAG     = 1                 # copper lag policy

print(f"Loading TimesFM 500M from {REPO_500M}…")
print(f"  (this downloads ~2 GB the first time)")

tfm = TimesFM_500M_torch.from_pretrained(REPO_500M)
forecast_config = ForecastConfig(
    max_context=CONTEXT,
    max_horizon=HORIZON,
    return_backcast=True,
)
tfm.compile(forecast_config)
print("  Model loaded and compiled.")

# ── Load data ──────────────────────────────────────────────────────────────
panel = pd.read_csv(DAILY_PANEL, index_col=0, parse_dates=True)
feat  = pd.read_csv(FEATURE_MATRIX, index_col=0, parse_dates=True)

usd_clp = panel["usd_clp"].values
dates   = panel.index
n       = len(usd_clp)

# ── Walk-forward evaluation ────────────────────────────────────────────────
results = []
start_idx = n - N_TEST_WINDOWS * STEP_DAYS

for i in range(N_TEST_WINDOWS):
    t0 = start_idx + i * STEP_DAYS
    if t0 + HORIZON >= n:
        break

    # Context: last CONTEXT days of USD/CLP price
    ctx_start = max(0, t0 - CONTEXT)
    context_series = usd_clp[ctx_start:t0].tolist()

    # Ground truth next HORIZON days
    actual = usd_clp[t0 : t0 + HORIZON]
    actual_ret = np.diff(np.log(actual), prepend=np.log(actual[0]))

    # TimesFM forecast (univariate CLP series)
    point_fc, _ = tfm.forecast(horizon=HORIZON, inputs=[np.array(context_series)])
    fc_prices = point_fc[0][:HORIZON]
    # Convert to returns
    last_price = context_series[-1]
    fc_ret = np.diff(np.log(np.concatenate([[last_price], fc_prices])))

    # Direction accuracy
    actual_dir = (actual_ret > 0).astype(int)
    fc_dir     = (fc_ret > 0).astype(int)
    dir_acc    = float(np.mean(actual_dir == fc_dir))

    # Naive baseline: repeat last return
    naive_ret  = np.full(HORIZON, actual_ret[0])
    naive_dir  = (naive_ret > 0).astype(int)
    naive_acc  = float(np.mean(actual_dir == naive_dir))

    # Signal Sharpe: go long if forecast return > 0
    signal     = np.where(fc_dir == 1, 1, -1)
    sig_sharpe = float(np.sqrt(252) * signal.mean() / (signal.std() + 1e-9))
    trade_ret  = signal * actual_ret
    trade_sharpe = float(np.sqrt(252) * trade_ret.mean() / (trade_ret.std() + 1e-9))

    r = {
        "window": i,
        "date":   str(dates[t0].date()),
        "dir_acc_500m":  round(dir_acc, 4),
        "dir_acc_naive": round(naive_acc, 4),
        "trade_sharpe":  round(trade_sharpe, 4),
    }
    results.append(r)
    print(f"  Window {i} ({r['date']}): "
          f"500M acc={dir_acc:.3f}  naive={naive_acc:.3f}  "
          f"trade Sharpe={trade_sharpe:.2f}")

# ── Summary ────────────────────────────────────────────────────────────────
avg_acc   = np.mean([r["dir_acc_500m"]  for r in results])
avg_naive = np.mean([r["dir_acc_naive"] for r in results])
avg_sh    = np.mean([r["trade_sharpe"]  for r in results])

summary = {
    "model": "TimesFM 2.0-500M (univariate CLP)",
    "repo":  REPO_500M,
    "context_days": CONTEXT,
    "horizon_days": HORIZON,
    "n_windows": len(results),
    "avg_dir_acc_500m":  round(float(avg_acc), 4),
    "avg_dir_acc_naive": round(float(avg_naive), 4),
    "avg_trade_sharpe":  round(float(avg_sh), 4),
    "note": "Next step: add copper as dynamic numerical covariate via forecast_with_covariates",
    "windows": results,
}

out_path = MODELS_DIR / "05b_timesfm_500m.json"
out_path.write_text(json.dumps(summary, indent=2))

print(f"\n{'='*55}")
print(f"  TimesFM 500M  avg dir-acc : {avg_acc:.3f}  (naive: {avg_naive:.3f})")
print(f"  Avg trade Sharpe           : {avg_sh:.2f}")
print(f"  Saved → {out_path}")
print(f"{'='*55}")

# ── Figure ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("TimesFM 500M — Walk-Forward OOS (univariate CLP)", fontsize=12)

wdates = [r["date"] for r in results]
accs_500m = [r["dir_acc_500m"] for r in results]
accs_naive = [r["dir_acc_naive"] for r in results]
sharpes = [r["trade_sharpe"] for r in results]

ax = axes[0]
ax.plot(wdates, accs_500m, marker="o", label="500M", color="#5a4fcf")
ax.plot(wdates, accs_naive, marker="s", linestyle="--", label="Naive", color="#aaa")
ax.axhline(0.5, color="r", lw=0.8, linestyle=":")
ax.set_ylabel("Directional Accuracy")
ax.set_title("Direction Accuracy per Window")
ax.legend(); ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=45)

ax = axes[1]
ax.bar(wdates, sharpes, color="#5a4fcf", edgecolor="none")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Annualised Sharpe")
ax.set_title("Trade Sharpe per Window")
ax.grid(axis="y", alpha=0.3)
ax.tick_params(axis="x", rotation=45)

fig_path = FIGURES_DIR / "05b_timesfm_500m.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure → {fig_path}")
