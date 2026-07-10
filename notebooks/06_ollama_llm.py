"""
LLM-based directional forecasting via Ollama.

Approach: feed recent CLP/copper return history as a structured text prompt,
ask the model to predict the direction of USD/CLP for the next N days.

Models tried: qwen2.5:14b, llama3.1:8b

Copper is provided with MIN_COPPER_LAG=1 (yesterday's copper, matching the
same lag policy used in all other models).

Run:
    uv run python notebooks/06_ollama_llm.py
"""
import sys, json, time
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
from copper_clp.config import (
    DAILY_PANEL, MODELS_DIR, FIGURES_DIR,
    MIN_COPPER_LAG,
)

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODELS      = ["qwen2.5:14b", "llama3.1:8b"]
HORIZON     = 5        # days to predict (shorter = less ambiguous prompt)
CONTEXT_LEN = 30       # days of history to show in prompt
N_WINDOWS   = 12       # evaluation windows
STEP_DAYS   = 63       # quarterly steps


def build_prompt(usd_clp_hist: list[float], copper_hist: list[float]) -> str:
    """
    Build a structured prompt for direction prediction.

    usd_clp_hist: last CONTEXT_LEN daily log-returns of USD/CLP
    copper_hist:  last CONTEXT_LEN daily log-returns of copper (already lagged)
    """
    clp_str = ", ".join(f"{x:+.4f}" for x in usd_clp_hist[-20:])
    cu_str  = ", ".join(f"{x:+.4f}" for x in copper_hist[-20:])
    recent_cu = copper_hist[-1]
    recent_clp = usd_clp_hist[-1]

    return f"""You are a quantitative analyst specializing in emerging-market FX.

Context: Chile is one of the world's largest copper exporters. When copper prices fall,
Chile's export revenues decline, the peso weakens, and USD/CLP rises. When copper rises,
the peso tends to strengthen and USD/CLP falls.

You are given the last 20 daily log-returns (×100 = percentage moves):
- USD/CLP log-returns (positive = peso WEAKER, USD stronger): [{clp_str}]
- COMEX Copper log-returns from the PREVIOUS day (lagged signal): [{cu_str}]

Most recent values:
- Yesterday's copper log-return: {recent_cu:+.4f}
- Yesterday's USD/CLP log-return: {recent_clp:+.4f}

Question: Based on these patterns and the copper-CLP relationship, what is the most
likely direction of USD/CLP over the next {HORIZON} trading days?

Answer with EXACTLY one word: UP (peso weakens) or DOWN (peso strengthens).
Then on the next line, give a one-sentence reason.

Answer:"""


def query_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 50},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"].strip()


def parse_direction(text: str) -> int | None:
    """Extract UP/DOWN from model response. Returns 1=UP, 0=DOWN, None=unclear."""
    first_line = text.strip().split("\n")[0].upper()
    if "UP" in first_line:
        return 1
    if "DOWN" in first_line:
        return 0
    # fallback: search whole text
    if "UP" in text.upper()[:30]:
        return 1
    if "DOWN" in text.upper()[:30]:
        return 0
    return None


# ── Load data ──────────────────────────────────────────────────────────────
panel    = pd.read_csv(DAILY_PANEL, index_col=0, parse_dates=True)
usd_clp  = panel["usd_clp"].values
r_clp    = panel["r_usd_clp"].values
r_copper = panel["r_copper"].values
dates    = panel.index
n        = len(usd_clp)

print(f"Data: {dates[0].date()} → {dates[-1].date()}  ({n} rows)")
print(f"Models: {MODELS}")
print(f"Horizon: {HORIZON}d | Context: {CONTEXT_LEN}d | Windows: {N_WINDOWS}\n")

# ── Walk-forward evaluation ────────────────────────────────────────────────
all_results = {m: [] for m in MODELS}
start_idx = n - N_WINDOWS * STEP_DAYS - HORIZON

for i in range(N_WINDOWS):
    t0 = start_idx + i * STEP_DAYS
    if t0 + HORIZON >= n:
        break
    if t0 < CONTEXT_LEN:
        continue

    # Features visible before trading on day t0
    clp_hist    = r_clp[t0 - CONTEXT_LEN : t0].tolist()
    # copper lagged by MIN_COPPER_LAG (yesterday's copper)
    copper_hist = r_copper[t0 - CONTEXT_LEN - MIN_COPPER_LAG : t0 - MIN_COPPER_LAG].tolist()

    # Ground truth: did USD/CLP go UP over next HORIZON days?
    future_ret    = r_clp[t0 : t0 + HORIZON]
    actual_cumret = float(np.sum(future_ret))
    actual_dir    = 1 if actual_cumret > 0 else 0  # 1=UP

    prompt = build_prompt(clp_hist, copper_hist)

    for model in MODELS:
        t_start = time.time()
        try:
            response = query_ollama(model, prompt)
            pred_dir = parse_direction(response)
            elapsed  = round(time.time() - t_start, 1)

            correct = (pred_dir == actual_dir) if pred_dir is not None else None
            result  = {
                "window":     i,
                "date":       str(dates[t0].date()),
                "actual_dir": actual_dir,
                "pred_dir":   pred_dir,
                "correct":    correct,
                "response":   response[:120],
                "elapsed_s":  elapsed,
            }
            all_results[model].append(result)
            mark = "✓" if correct else ("?" if correct is None else "✗")
            print(f"  [{i}] {dates[t0].date()}  {model:16s}  "
                  f"pred={'UP  ' if pred_dir==1 else 'DOWN' if pred_dir==0 else '???'}  "
                  f"actual={'UP' if actual_dir==1 else 'DN'}  {mark}  ({elapsed}s)")
        except Exception as e:
            print(f"  [{i}] {dates[t0].date()}  {model}: ERROR — {e}")
            all_results[model].append({
                "window": i, "date": str(dates[t0].date()),
                "actual_dir": actual_dir, "pred_dir": None,
                "correct": None, "response": str(e), "elapsed_s": 0,
            })

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
summary = {}
for model in MODELS:
    rows    = all_results[model]
    decided = [r for r in rows if r["correct"] is not None]
    acc     = np.mean([r["correct"] for r in decided]) if decided else float("nan")
    # Sharpe: +1 if UP prediction and actual UP, -1 otherwise
    sigs, rets = [], []
    for r in rows:
        if r["pred_dir"] is None:
            continue
        sig = 1 if r["pred_dir"] == 1 else -1
        # actual_dir 1=UP → positive return signal aligns
        ret = 1.0 if (sig == 1 and r["actual_dir"] == 1) else \
              1.0 if (sig == -1 and r["actual_dir"] == 0) else -1.0
        sigs.append(sig); rets.append(ret)
    sharpe = float(np.sqrt(252/HORIZON) * np.mean(rets) / (np.std(rets)+1e-9)) if rets else float("nan")

    summary[model] = {
        "n_windows":     len(rows),
        "n_decided":     len(decided),
        "accuracy":      round(float(acc), 4) if not np.isnan(acc) else None,
        "sharpe_annualised": round(sharpe, 4) if not np.isnan(sharpe) else None,
    }
    print(f"  {model:20s}: acc={acc:.3f}  Sharpe={sharpe:.2f}  "
          f"({len(decided)}/{len(rows)} decided)")

print(f"{'='*60}\n")

# ── Save ───────────────────────────────────────────────────────────────────
output = {
    "horizon_days": HORIZON,
    "context_days": CONTEXT_LEN,
    "copper_lag":   MIN_COPPER_LAG,
    "n_windows":    N_WINDOWS,
    "summary":      summary,
    "results":      all_results,
}
out_path = MODELS_DIR / "06_ollama_llm.json"
out_path.write_text(json.dumps(output, indent=2))
print(f"Saved → {out_path}")

# ── Figure ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle(f"LLM Directional Accuracy — {HORIZON}-day Horizon", fontsize=12)

colors = {"qwen2.5:14b": "#5a4fcf", "llama3.1:8b": "#e07b29"}
for model in MODELS:
    rows    = all_results[model]
    decided = [r for r in rows if r["correct"] is not None]
    if not decided:
        continue
    window_accs = []
    wdates      = []
    # Rolling 4-window accuracy
    for j in range(len(decided)):
        chunk = decided[max(0,j-3):j+1]
        window_accs.append(np.mean([r["correct"] for r in chunk]))
        wdates.append(chunk[-1]["date"])
    ax.plot(wdates, window_accs, marker="o", label=model,
            color=colors.get(model, "gray"), ms=5)

ax.axhline(0.5, color="r", lw=0.8, linestyle="--", label="Random 50%")
ax.set_ylabel("Rolling Accuracy (4-window)")
ax.set_xlabel("Window date")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, 1)

fig_path = FIGURES_DIR / "06_ollama_llm.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure → {fig_path}")
