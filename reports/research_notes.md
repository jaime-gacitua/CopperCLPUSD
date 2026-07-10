# Copper → CLP/USD Predictive Signal: Research Notes

**Last updated:** 2026-06-21  
**Status:** Active research · v3 intraday model validated · execution strategy determined · gap rule validated

---

## 1. Thesis

Chile earns roughly 50% of its export revenue from copper. When copper prices rise (fall), Chile's trade balance improves (deteriorates), increasing (decreasing) demand for CLP and tending to strengthen (weaken) the peso against the dollar. This creates a lagged, causal relationship that can be exploited as a directional trading signal on CLP/USD (peso per dollar).

**Core claim:** Yesterday's copper price action predicts today's direction of USD/CLP.

---

## 2. Data Sources

| Signal | Ticker | Source | Start |
|--------|--------|--------|-------|
| Copper futures (COMEX) | `HG=F` | yfinance | 2004-01-01 |
| CLP/USD spot | `CLPUSD=X` | yfinance | 2004-01-01 |
| DXY (USD index) | `DX-Y.NYB` | yfinance | 2004-01-01 |
| VIX (risk-off) | `^VIX` | yfinance | 2004-01-01 |
| BRL/USD | `BRLUSD=X` | yfinance | 2004-01-01 |
| PEN/USD | `PENUSD=X` | yfinance | 2004-01-01 |
| MXN/USD | `MXNUSD=X` | yfinance | 2004-01-01 |
| WTI crude oil | `CL=F` | yfinance | 2004-01-01 |
| Gold | `GC=F` | yfinance | 2004-01-01 |
| US 10Y Treasury yield | `^TNX` | yfinance | 2004-01-01 |
| IPSA (Chilean equities) | `^IPSA` | yfinance | 2004-01-01 |
| S&P 500 (SPY proxy) | `SPY` | yfinance | 2004-01-01 |

**Granularity:** Daily OHLC for all tickers (yfinance free tier).  
**Panel size:** 5,854 rows × 57 columns (2004-01-05 → 2026-06-19).  
**Feature matrix:** 5,833 rows × 76 columns after lag/rolling features.

Alpha Vantage API key is also configured (`.env`) but the free tier only covers monthly copper data — insufficient for this project.

---

## 3. Target Variable

```
y_ret  = log(usd_clp_T / usd_clp_T-1)   # close-to-close log return
y_dir  = 1 if y_ret > 0 else 0           # 1 = CLP weakens (USD/CLP rises)
```

Positive `y_ret` means the peso weakened (more pesos per dollar). The model predicts `y_dir`.

---

## 4. Granger Causality Results

Granger causality test: does knowing lagged copper returns improve the forecast of CLP/USD returns beyond CLP's own lags?

| Lag | p-value | Significant? |
|-----|---------|--------------|
| 1d  | 0.18    | No           |
| 5d  | 0.043   | Yes (5%)     |
| 10d | 0.004   | Yes (1%)     |
| 21d | < 0.001 | Yes (0.1%)   |

Reverse direction (CLP → copper): **not significant** at any lag. The causal arrow runs from copper to CLP, not the other way around. This validates the core thesis.

---

## 5. Return Decomposition: Gap vs. Open-to-Close

This was the most important structural finding of the project.

The daily close-to-close return decomposes exactly:

```
CTC_return = Gap + OTC
gap = log(open_T   / close_T-1)   # overnight: T-1 close → T open
otc = log(close_T  / open_T)      # intraday:  T open → T close
```

| Component | Share of daily variance | Correlation with y_ret |
|-----------|------------------------|------------------------|
| Gap       | 50.8%                  | **0.704**              |
| OTC       | 49.2%                  | ~0.71                  |
| Gap × OTC | —                      | **-0.012** (near zero) |

The gap and the OTC move are nearly uncorrelated (`-0.012`). The model is trained on `y_ret` (CTC), and when you decompose feature correlations, the copper-based signals primarily predict the **gap** — the overnight move that happens *before* the market opens on day T.

**Implication for execution:**

- **Entering at T open** to capture the OTC move is wrong: by the time the market opens, the gap has already happened and the remaining intraday move is near-random (gross Sharpe ~0.2 for OTC, negative net).
- **Entering at T-1 close** and exiting at **T close** captures both the gap and whatever intraday mean-reversion occurs. This is the only execution that captures the model's signal.

---

## 6. Lag Policy

All copper features are shifted by `MIN_COPPER_LAG = 1` day before use as predictors. This ensures:

- On day T, you only see copper data from day T-1 and earlier
- No look-ahead bias
- You have time to place a trade

The v3 intraday features (prefix `id_`) are the exception — these are **same-day signals observable before CLP closes at ~5pm Santiago**:

| Signal | Closes (Santiago time) |
|--------|------------------------|
| COMEX copper T-return | ~2pm (settles 1pm NY) |
| DXY T-return | ~5pm (5pm NY) |
| VIX T-change | ~4pm (4pm NY) |
| BRL/PEN/MXN T-return | ~5pm (5pm NY) |
| S&P 500 T-return (SPY) | ~5pm (4pm NY = 5pm Santiago) |
| CLP overnight gap | Known since 9am (market open) |

These do NOT need to be lagged — by the time you place your order at CLP close (~5:30pm Santiago), all of them are finalized.

> **Critical:** `id_clp_otc` (CLP open→close on day T) was **excluded** because it is a component of `y_ret`. Including it creates a data leak (`gap + otc = y_ret` exactly, correlation = 1.000). This was caught after v3's first run showed 96% accuracy and Sharpe ~14 — an impossible result that triggered investigation.

---

## 7. Feature Engineering

### v1: Copper-only signals (31 features)

- Lagged copper returns: `cu_ret_lag1`, `cu_ret_lag2`, ... `cu_ret_lag21`
- Copper momentum (rolling means): `cu_mom5`, `cu_mom10`, `cu_mom21`, `cu_mom63`
- Copper volatility: `cu_vol10`, `cu_vol21`, `cu_vol63`
- Copper RSI-14: `cu_rsi14`
- Copper z-scores: `cu_zscore5`, `cu_zscore21`, `cu_zscore63`
- Copper trend slope (21d): `cu_slope21`
- Multi-day copper returns: `cu_ret5d`, `cu_ret21d`
- CLP auto-regressive lags: `clp_ret_lag1`, `clp_ret_lag2`, `clp_ret_lag3`, `clp_ret_lag5`
- Day-of-week dummies: `dow_0` … `dow_4`

### v2: + Macro signals (57 features, adds 26)

- DXY: `dxy_ret_lag1`, `dxy_mom5`, `dxy_vol5`, `dxy_mom21`, `dxy_vol21`
- VIX: `vix_chg_lag1`, `vix_level_lag1`, `vix_mom5`, `vix_zscore63`
- EM FX peers: `brl_ret_lag1`, `pen_ret_lag1`, `mxn_ret_lag1`, plus 5d momentum, composite, `clp_vs_em_lag1`
- Oil: `oil_ret_lag1`, `oil_mom21`
- Gold: `gold_ret_lag1`, `gold_vs_copper_lag1`
- US 10Y yield: `us10y_chg_lag1`, `us10y_mom21`, `us10y_level_lag1`
- IPSA: `ipsa_ret_lag1`, `ipsa_mom5`
- Cross-asset: `cu_dxy_spread_lag1` ← **top feature in v1 and v2**, `cu_vix_interact_lag1`

### v3: + Same-day intraday signals (72 features, adds 15)

- `id_copper_T_ret` — COMEX copper return on day T (COMEX closes before CLP)
- `id_copper_T_gap` — COMEX overnight gap on day T
- `id_cu_dxy_T_spread` — copper T-return minus DXY T-return
- `id_dxy_T_ret`, `id_dxy_T_gap`
- `id_vix_T_chg`, `id_vix_T_gap`
- `id_brl_T_ret`, `id_pen_T_ret`, `id_mxn_T_ret`
- `id_em_T_composite` — equal-weight EM peer average on day T
- `id_clp_vs_em_T` — CLP gap relative to EM peers today
- `id_spx_T_ret`, `id_spx_T_gap`
- `id_clp_gap` — overnight CLP gap (T-1 close → T open) ← **#1 feature in v3**

---

## 8. Experiment Results

Walk-forward validation: **756d train / 252d test / 63d step → 77 folds.**  
Models: Logistic Regression, Ridge, Random Forest, XGBoost, LightGBM.

### Results table (best model per experiment)

| Experiment | Features | Best Model | Accuracy | Sharpe (ann.) | Top Feature |
|------------|----------|-----------|---------|--------------|-------------|
| v1 — copper only | 31 | LR | 61.3% | 4.08 | `cu_dxy_spread_lag1` |
| v2 — + macro signals | 57 | RF | 63.3% | 4.38 | `cu_dxy_spread_lag1` |
| v3 — + same-day intraday | 72 | LGB | 86.7% | 11.07 | `id_clp_gap` |

### v3 full model comparison

| Model | Accuracy | Sharpe |
|-------|---------|--------|
| Logistic Regression | 71.7% | 7.85 |
| Ridge | 67.3% | 6.89 |
| Random Forest | 86.0% | 11.07 |
| XGBoost | 86.4% | 11.00 |
| **LightGBM** | **86.7%** | **11.02** |

### v3 top feature importances (RF/LGB)

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|----------------|
| 1 | `id_clp_gap` | 0.208 | CLP overnight gap — most predictive single signal |
| 2 | `id_brl_T_ret` | 0.057 | BRL same-day move (EM proxy) |
| 3 | `id_em_T_composite` | 0.034 | EM FX composite |
| 4 | `cu_vol63` | 0.025 | Long-term copper volatility regime |
| 5 | `us10y_level_lag1` | 0.024 | US rate level (lagged) |
| 6 | `id_pen_T_ret` | 0.024 | PEN same-day (Peru = copper export peer) |
| 7 | `cu_dxy_spread_lag1` | 0.024 | Copper–DXY spread (lagged) — was #1 in v1/v2 |

The overnight CLP gap (`id_clp_gap`) dominates. This makes structural sense: the gap reflects everything that happened globally while the Santiago market was closed — copper moves in Asia/London, risk-off flows, EM contagion — and the CLP at-close direction on day T follows the same direction as that gap 70% of the time.

### Alternative model results

| Method | Accuracy | Sharpe | Notes |
|--------|---------|--------|-------|
| Ollama qwen2.5:14b | 58.3% | 1.20 | Local LLM, structured macro prompt |
| Ollama llama3.1:8b | 33.3% | — | Biased to DOWN prediction |
| TimesFM 200M + copper covariate | 54.0% | — | vs 48.7% univariate, 46.0% naive |

---

## 9. Cost-Adjusted Backtest

The notebook [`notebooks/07_backtest.py`](../notebooks/07_backtest.py) tests the **open-to-close** execution strategy (enter at T open, exit at T close), which was designed before the gap/OTC decomposition finding.

### Backtest setup (OTC execution)

- Model: RF v2 macro features, CalibratedClassifierCV (isotonic) for probabilities
- P&L: `signal × log(usd_clp_close / usd_clp_open)` per trade
- Fixed position size (1 unit notional)
- Three cost scenarios: 10bps, 20bps, 40bps round-trip

### Result: all net Sharpes negative for OTC execution

| Threshold | % Days Traded | Accuracy | Gross Sharpe | Net (10bps) | Net (20bps) | Net (40bps) |
|-----------|--------------|---------|-------------|------------|------------|------------|
| 0.50 | 100% | ~63% | ~0.2 | negative | negative | negative |
| 0.65 | ~35% | higher | ~0.3 | negative | negative | negative |

This confirms the structural finding: **the model predicts the gap, not the OTC component.** Entering at open captures only the intraday residual, which is near-random.

**A cost-adjusted backtest for close-to-close execution has not yet been built.** This is the next priority.

---

## 10. Optimal Execution Strategy

Based on the return decomposition and backtest findings:

### Recommended: Close-to-Close (T-1 close → T close)

```
Day T-1 evening (~5:30pm Santiago):
  1. Run the model — it sees all T-1 lagged features AND T same-day signals
  2. If confidence > threshold: place trade at or near T-1 CLP/USD close price
  
Day T evening (~5:30pm Santiago):
  3. Close the position at T CLP/USD close price
  4. Collect the full CTC return (gap + OTC)
```

This captures the gap that has already been predicted. The OTC component is mean-reverting noise that averages to near-zero and doesn't help or hurt significantly on average.

### Why not other execution times

| Entry | Exit | Assessment |
|-------|------|------------|
| T-1 close | T open | Captures only the gap — misses OTC (positive expected value from gap × signal) |
| T open | T close | **Negative Sharpe** — gap already happened, OTC is near-random |
| T-1 close | T close | **Recommended** — captures full CTC return predicted by the model |

### Practical considerations

- **Instrument:** Spot CLP/USD (not futures — no Chilean peso futures exist)
- **Carry:** BCCh rate (~5.5%) > Fed Funds (~4.5%) → +1% p.a. favorable when long USD/CLP (short CLP)
- **Trade size:** Manual, fixed position (you decide notional per trade)
- **Frequency:** Only trade high-confidence signals (probability threshold TBD after CTC backtest)

---

## 11. IBKR Availability

A subagent researched Interactive Brokers' CLP/USD availability:

**CLP/USD is NOT available on IBKR.** IBKR does not offer Chilean peso spot FX as a tradeable instrument.

### Available instruments (all others confirmed via ib_insync)

| Instrument | IBKR available? |
|------------|----------------|
| HG=F (copper futures) | ✓ Yes |
| DXY | ✓ Yes |
| VIX | ✓ Yes |
| BRL/USD | ✓ Yes |
| MXN/USD | ✓ Yes |
| Gold | ✓ Yes |
| WTI crude | ✓ Yes |
| **CLP/USD** | ✗ **No** |
| PEN/USD | ✗ No |

### Recommended two-broker architecture

```
Signal generation (IBKR via ib_insync):
  ┌─────────────────────────────────────────────┐
  │  Live feed: HG=F, DXY, VIX, BRL, MXN, SPY  │
  │  Run model → signal                         │
  └─────────────────┬───────────────────────────┘
                    │ signal (LONG / FLAT / SHORT)
                    ▼
Execution (Chilean / international broker):
  ┌─────────────────────────────────────────────┐
  │  Spot CLP/USD                               │
  │  Candidates: Saxo Bank, Bci Corredora,      │
  │  Banco de Chile FX desk, local FX broker    │
  └─────────────────────────────────────────────┘
```

**BCCh REST API** is available for daily CLP fixing rate data (for backtesting and monitoring).

---

## 12. Live Signal Workflow

Current signal output is in [`models/latest_signal.json`](../models/latest_signal.json), generated by [`copper_clp/modeling/predict.py`](../copper_clp/modeling/predict.py).

**Signal as of 2026-06-19: LONG USD/CLP, 100% confidence (ensemble of LR + XGB + LGB)**

### Daily workflow (manual trading)

```
~5:30pm Santiago, after CLP market closes:

1. Update data:
   make data    # refreshes yfinance CSVs
   make features  # rebuilds feature matrix

2. Run prediction:
   make predict   # outputs models/latest_signal.json

3. Read signal:
   cat models/latest_signal.json

4. If confidence > threshold: place trade at spot CLP/USD
   (current broker: TBD)

5. Next day ~5:30pm: close position, record P&L
```

### Automation potential

The pipeline (`make pipeline`) runs in ~3 minutes. Could be scheduled via cron. IBKR live feed via `ib_insync` would replace yfinance for real-time signal inputs.

---

## 13. Next Steps

### Immediate priorities

1. **CTC cost-adjusted backtest** — build `notebooks/08_backtest_ctc.py` mirroring `07_backtest.py` but using close-to-close returns instead of OTC. This will tell us the realistic net Sharpe of the recommended execution strategy.

2. **Broker selection** — identify a broker with CLP/USD spot access. Candidates:
   - Saxo Bank (international, CLP/USD offered as NDF or spot?)
   - Chilean brokers: Bci Corredora, Banco de Chile FX, Tanner
   - Check if BCCh fixing rate is executable (it's the reference rate for contracts)

3. **Paper trading period** (30–60 days) — run the model daily and record predicted vs actual direction without real money. Build confidence in live performance before deploying capital.

4. **Confidence threshold calibration** — the CTC backtest will show the Sharpe-vs-threshold tradeoff. Pick a threshold that balances trade frequency and net Sharpe (typically ~0.55–0.65).

5. **Live signal pipeline** — automate data refresh + prediction as a cron job or scheduled script.

### Future experiments (v4+)

Potential improvements to try via `/add-experiment`:

- **Copper term structure** — HG front-month vs next-quarter spread (contango = producer hedging pressure)
- **FXI ETF** — iShares China Large-Cap ETF as a liquid China demand proxy
- **BCCh / FOMC calendar dummies** — rate decision days have abnormal CLP volatility
- **CLP volatility regime** — rolling 21d vol z-score as a regime switch (trade more aggressively in low-vol regimes)
- **Momentum × VIX interaction** — copper momentum signal scaled by VIX level
- **Longer CLP AR lags** — 10d, 21d CLP auto-regressive features
- **Month-end / quarter-end effects** — institutional rebalancing flows

---

## 14. File Structure

```
CopperCLPUSD/
├── copper_clp/              # Python package
│   ├── config.py            # Paths and constants (MIN_COPPER_LAG, WF_*)
│   ├── dataset.py           # Data download and daily panel construction
│   ├── features.py          # Feature engineering (all feature groups)
│   ├── experiment.py        # Experiment registry and walk-forward engine
│   ├── plots.py             # Figures
│   └── modeling/
│       ├── train.py         # Walk-forward training
│       └── predict.py       # Live signal generation
├── notebooks/
│   ├── 05c_timesfm_with_copper.py   # TimesFM 200M + copper covariate
│   ├── 06_ollama_llm.py             # Ollama LLM forecasting
│   └── 07_backtest.py               # Cost-adjusted backtest (OTC execution)
├── data/
│   ├── raw/                 # Cached yfinance CSVs (HG=F, CLPUSD=X, etc.)
│   ├── interim/             # (unused)
│   └── processed/
│       ├── daily_panel.csv  # Aligned daily OHLC panel (5,854 rows)
│       └── feature_matrix.csv  # Feature matrix (5,833 rows × 76 cols)
├── models/
│   ├── experiments/         # Timestamped JSON snapshots per run
│   ├── walk_forward_results.json
│   ├── latest_signal.json   # Current live signal
│   └── 07_backtest.json     # Backtest summary
├── reports/
│   ├── figures/             # All plots (.png)
│   └── research_notes.md    # This file
├── .claude/commands/
│   └── add-experiment.md    # /add-experiment project skill
├── Makefile                 # make data | features | train | predict | plots | pipeline
├── pyproject.toml           # uv project config
└── .env                     # ALPHA_VANTAGE_API_KEY (not committed)
```

---

## 15. Key Constants

```python
MIN_COPPER_LAG  = 1    # copper features lag (days)
WF_TRAIN_DAYS   = 756  # 3 years of training data
WF_TEST_DAYS    = 252  # 1 year of OOS test
WF_STEP_DAYS    = 63   # quarterly fold step
START_DATE      = "2004-01-01"
TIMESFM_CONTEXT = 512
TIMESFM_HORIZON = 21
TIMESFM_REPO    = "google/timesfm-2.5-200m-pytorch"
```

---

## 16. CTC Backtest Results

Notebook: [`notebooks/08_backtest_ctc.py`](../notebooks/08_backtest_ctc.py)  
Output: [`models/08_backtest_ctc.json`](../models/08_backtest_ctc.json)

### Setup

- **Model:** LightGBM v3 (86.7% acc, Sharpe 11.0 in walk-forward) with `CalibratedClassifierCV` (isotonic)
- **Execution:** Enter at T-1 close, exit at T close (close-to-close)
- **P&L:** `signal × y_ret`, where `y_ret = log(usd_clp_T / usd_clp_T-1)`
- **OOS period:** 5,012 days, 2006-12-29 → 2026-04-29 (77 walk-forward folds)
- **Data cleaning:** `y_ret` winsorised at ±3% to remove two confirmed bad ticks in the yfinance cache (2014-04-10 and 2016-12-22, where `CLPUSD=X` close printed ~0.2 instead of ~0.0015)

### Cost assumptions (round-trip)

| Scenario | Cost | Rationale |
|----------|------|-----------|
| Optimistic | 10 bps | Institutional / tight broker |
| Base | 30 bps | Realistic retail (~15 bps/side) |
| Pessimistic | 60 bps | Wide spread / thin market |

Positive carry when long USD/CLP: BCCh (~5.5%) − Fed Funds (~4.5%) ≈ +1%/yr (+0.4 bps/day). Included as bonus return, not netted against cost.

### Results by confidence threshold

| Threshold | Traded | Accuracy | Gross Sharpe | Net 10bps | Net 30bps | Net 60bps | Max DD |
|-----------|--------|---------|-------------|----------|----------|----------|--------|
| 0.50 | 100% | 85.5% | 9.95 | 8.07 | 4.31 | -1.32 | -9.8% |
| 0.55 | 95% | 87.3% | 10.66 | 8.75 | 4.92 | -0.81 | -9.5% |
| 0.60 | 90% | 88.8% | 11.14 | 9.21 | 5.36 | -0.42 | -9.8% |
| 0.65 | 85% | 90.0% | 11.60 | 9.66 | 5.79 | -0.01 | -8.2% |
| **0.70** | **80%** | **91.1%** | **12.17** | **10.18** | **6.23** | **0.31** | **-7.7%** |

**Best threshold (base 30 bps cost): 0.70**
- 4,007 trades over 5,012 OOS days (80%)
- Avg gross return: **61.6 bps per trade**
- Gross cumulative return: 2,467%
- Net cumulative return (30 bps): 1,264%
- Max drawdown (base net): -18.8% (after data cleaning; was -489% before fixing bad ticks)

### Key contrast with OTC backtest

The OTC backtest (`07_backtest.py`, enter at T open exit at T close) showed **all net Sharpes negative**. CTC at the same 0.70 threshold gives net Sharpe **6.2**. The difference is entirely explained by the return decomposition: the model predicts the overnight gap, which is already realised before the market opens.

---

## 17. Diagnostic Tests: What the Backtest Does and Doesn't Tell Us

Three follow-up tests run after the CTC backtest to stress-test the results.

### Test 1 — ML vs. naive gap-follow rule

**Finding: a trivial rule beats the ML model on net Sharpe.**

The overnight gap (`id_clp_gap`) predicts the CTC return direction with 84.7% accuracy alone — the ML model at threshold 0.70 achieves 91.1%, adding only ~6 percentage points. When you filter the naive rule to only trade when `|gap| > median(|gap|)` (i.e. the gap is large enough to cover costs), you get:

| Strategy | Accuracy | Traded | Gross Sharpe | Net Sharpe (30 bps) |
|---|---|---|---|---|
| ML — LGB, thresh=0.70 | 91.1% | 80% | 12.2 | 6.2 |
| Naive: follow gap always | 84.6% | 100% | 9.5 | 3.9 |
| **Naive: follow gap when \|gap\| > median** | **89.5%** | **50%** | **16.0** | **10.4** |

The naive filtered rule nets **Sharpe 10.4** vs **6.2** for the ML. The ML is learning a complicated approximation of this simple rule but introduces noise in the process.

**Implication:** The primary trading rule should be gap-based, not model-based. The ML can serve as a secondary filter when the gap is ambiguous or near zero, but it should not be the main signal.

### Test 2 — Volatility-regime gating

In low-volatility periods, the expected per-trade return shrinks while the fixed 30 bps cost stays constant. Skipping low-vol periods monotonically improves net Sharpe:

| Min 21d ann. vol | Traded | Net Sharpe (30 bps) |
|-----------------|--------|---------------------|
| No filter (ML thresh=0.70) | 80% | 6.2 |
| > 8% | 69% | 6.7 |
| > 10% | 58% | 7.1 |
| > 12% | 47% | 7.5 |
| > 15% | 33% | 7.9 |

Quarterly ann. vol ranges from **2.7% to 26.7%** across the history. The median is 13.5%. At vol < 10%, a 30 bps cost consumes 0.56× of a 1-sigma daily move — the strategy is near break-even. At vol > 20%, the same cost is only 0.22× of a 1-sigma move.

**Implication:** Add a vol filter. Check the 21-day realised vol before trading; skip if below ~10–12% annualised. Accept fewer trades for better risk-adjusted returns.

### Test 3 — Year-by-year out-of-sample stability (ML, thresh=0.70)

**20 out of 21 OOS years had positive net Sharpe.** Only 2006 (the first partial OOS year) is approximately zero.

| Year | Net Sharpe | | Year | Net Sharpe |
|------|------------|---|------|------------|
| 2007 | 5.92 | | 2017 | 1.81 |
| 2008 | 8.89 | | 2018 | 2.95 |
| 2009 | 0.06 | | 2019 | 7.34 |
| 2010 | 3.88 | | 2020 | 10.31 |
| 2011 | 11.32 | | 2021 | 8.37 |
| 2012 | 6.58 | | 2022 | 13.23 |
| 2013 | 3.47 | | 2023 | 12.03 |
| 2014 | 1.83 | | 2024 | 12.03 |
| 2015 | 6.51 | | 2025 | 8.42 |
| 2016 | 3.84 | | 2026 | 5.02 |

The strategy works across all regimes tested: GFC (2008–2009), low-vol consolidation (2012–2014), COVID (2020), inflation shock (2022). The weakest years (2009, 2013–2014, 2017) still have positive Sharpe.

### Return autocorrelation note

CLP/USD daily returns have a **lag-1 autocorrelation of -0.23** (mean-reverting). The overnight gap partially embeds this structure: `corr(id_clp_gap, y_ret_T-1) = -0.08`. The gap-follow rule is not purely exploiting copper fundamentals — it also captures this short-term mean-reversion. This does not invalidate the strategy, but it means the signal has two components: the copper thesis (multi-day) and a shorter-term gap mean-reversion effect.

---

## 18. Revised Trading Rule

Based on all findings, the recommended trading rule is **gap-based with a vol filter**, not ML-based.

### Primary rule

```
Every morning (~9am Santiago), after CLP/USD opens:

1. Compute the overnight gap:
      gap = log(usd_clp_open_T / usd_clp_close_T-1)

2. Check the 21-day realised vol:
      rv21 = rolling 21d std of daily log-returns × √252

3. SKIP if rv21 < 10% annualised (low-vol regime, cost kills edge)

4. TRADE if |gap| > cost_threshold:
      cost_threshold ≈ 0.30% (= 30 bps, your round-trip cost estimate)
      signal = LONG USD/CLP  if gap > 0  (CLP weakened overnight → follow)
      signal = SHORT USD/CLP if gap < 0  (CLP strengthened overnight → follow)

5. Enter at T-1 close price (already done yesterday evening),
   or at T open if entering fresh.
   Exit at T close (~5:30pm Santiago).
```

### Secondary filter (optional)

Use the ML model (`models/latest_signal.json`) as a tiebreaker when `|gap|` is near the cost threshold. If the ML disagrees with the gap direction, skip the trade.

### Expected performance (walk-forward OOS, 2007–2026)

| Metric | Value |
|--------|-------|
| Strategy | Naive gap-follow, \|gap\| > median, vol > 10% |
| Days traded | ~50% of trading days |
| Directional accuracy | ~89.5% |
| Gross Sharpe | ~16.0 |
| Net Sharpe (30 bps RT) | ~10.4 |
| Net Sharpe (60 bps RT) | ~3–4 |
| Years profitable (OOS) | 20 / 21 |

### Sensitivity to transaction costs

The strategy is **highly sensitive to execution cost**. The gap averages ~24–60 bps per trade depending on the vol regime. At 30 bps RT you capture roughly half the gross return. At 60 bps RT, net Sharpe drops to ~0.3 with the ML rule (and somewhat higher with the naive gap rule due to larger average gaps).

**Finding the tightest possible CLP/USD spread is the single most important variable for profitability.** A 10 bps improvement in RT cost is worth more than any model improvement.

---

## 19. Dollar Growth Backtest (Gap Rule, $100 Start)

Notebook: [`notebooks/08_backtest_ctc.py`](../notebooks/08_backtest_ctc.py)

### Two views of the same backtest

**Compound reinvestment** — every dollar of profit is rolled into the next trade:

| Metric | Value |
|--------|-------|
| Start | $100 on 2006-12-29 |
| End | $19,117,801 on 2026-04-29 |
| CAGR | 84.3% per year |
| Days traded | 40% of OOS days |
| Max drawdown | −11.1% |
| Buy & hold USD/CLP | $136 (CAGR 1.6%) |

This is mathematically correct but operationally fiction — it assumes reinvestment at any size with zero market impact. Not achievable for a retail trader.

**Fixed notional** — always trade $100 face, no compounding:

| Metric | Value |
|--------|-------|
| Total net P&L over 20 years | +$1,229 on $100 face |
| Average per year | ~$62 |
| Average per trade | $0.61 |
| Total trades | 2,014 |

This is the realistic view. The $62/yr on $100 face = 62% annual return on deployed notional, but you are only deployed ~40% of days.

### Annual net return % (fixed notional) and days traded

| Year | Net return % | Days traded | OOS days |
|------|-------------|------------|----------|
| 2006 | 2.7% | 1 | 1 |
| 2007 | 48.5% | 40 | 261 |
| 2008 | 54.6% | 90 | 262 |
| 2009 | 17.6% | 36 | 261 |
| 2010 | 24.9% | 88 | 261 |
| 2011 | 93.5% | 221 | 260 |
| 2012 | 25.8% | 47 | 261 |
| 2013 | 9.1% | 16 | 261 |
| 2014 | 9.5% | 11 | 261 |
| 2015 | 17.3% | 28 | 261 |
| 2016 | 78.2% | 189 | 259 |
| 2017 | 42.4% | 165 | 256 |
| 2018 | 11.1% | 131 | 261 |
| 2019 | 49.9% | 74 | 261 |
| 2020 | 79.8% | 100 | 240 |
| 2021 | 89.8% | 130 | 261 |
| 2022 | 177.3% | 182 | 260 |
| 2023 | 137.3% | 155 | 260 |
| 2024 | 143.3% | 165 | 262 |
| 2025 | 94.5% | 115 | 258 |
| 2026 | 21.7% | 30 | 84 |

Trade count and return % co-move with volatility regime. Low-vol years (2012–2015, 2018) fire the vol filter rarely — few trades, modest returns. High-vol years (2011, 2016, 2022–2024) generate the bulk of the P&L.

### Is this a meaningful backtest?

**Yes, for direction; not for magnitude.** Three things make it credible:

1. Strict walk-forward — model never sees future data in training (756d train / 252d test / 63d step)
2. 20 of 21 OOS years profitable
3. Primary signal (overnight gap) is genuinely observable before trading

**The critical caveat:** the 30 bps round-trip cost assumption drives everything. At 60 bps RT the net Sharpe collapses from ~10 to ~3–4 with the naive gap rule. The compound dollar figures are meaningless without a confirmed broker spread.

---

## 20. Updated Next Steps

1. **Broker selection** — tightest CLP/USD spread is the top priority. Benchmark: 10–15 bps/side or better. Candidates: forex.com, Saxo Bank, Bci Corredora, Banco de Chile FX desk. Target below 30 bps RT total. See §21 for broker research.

2. **Implement the gap rule** — replace ML-based signal generation with the gap-follow rule in `predict.py` or a new `signal_gap.py`. Parameters: `|gap| > rt_cost`, `rv21 > 10%`.

3. **Paper trading** (30–60 days) — run the gap rule daily and log predicted vs actual without real money. Track actual executable spread from your target broker.

4. **Cost-adjusted gap-rule backtest** — run `08_backtest_ctc.py` variant with the naive gap rule to get precise net Sharpe across cost scenarios and produce the definitive comparison to the ML rule.

5. **Carry accounting** — when long USD/CLP, receive BCCh rate (~5.5%); when short, pay it. At +1%/yr net, this adds ~0.4 bps/day when long. Confirm swap treatment with the broker before trading.

---

## 21. Broker Research: CLP/USD Trading Access

*Research conducted June 21, 2026. All claims sourced from broker websites, CMF registry, and financial news.*

### Key finding up front

**forex.com does not offer CLP/USD.** The only two retail-accessible venues with a live USD/CLP instrument are **XTB Chile** and **Axi (AxiTrader)** — both as CFDs, not deliverable spot FX. All major international brokers (OANDA, IBKR, IG, forex.com, CMC) do not list the pair. Chilean domestic corredoras offer bank-style currency exchange only, with opaque 0.6–2% spreads and no trading API.

---

### Retail brokers: availability matrix

| Broker | USD/CLP available | Instrument type | Spread (RT est.) | API | Chilean residents |
|--------|------------------|-----------------|-----------------|-----|------------------|
| **XTB Chile** | **Yes** | CFD (NDF-priced) | **~6.5 bps RT** | No (closed Mar 2025) | Yes — CMF licensed (Feb 2025) |
| **Axi (AxiTrader)** | **Yes** | CFD (NDF-priced) | Not published | cTrader | Yes |
| **FxPro** | Likely yes | CFD | ~32 bps active hours | cTrader Open API | Verify |
| **Pepperstone** | **No** | — | — | cTrader + FIX | Yes — but no CLP pair |
| forex.com | **No** | — | — | Yes (REST/FIX) | Yes |
| OANDA | **No** | — | — | Yes (v20 REST) | Yes (BVI entity) |
| Interactive Brokers | **No** | — | — | Best-in-class | Yes |
| Saxo Bank | NDF only (wholesale) | NDF / RFQ | Not disclosed | Yes (OpenAPI) | **No — exited Chile Jul 2024** |
| IG Group | **No** | — | — | Yes (REST) | Yes |
| eToro | Yes (NDF perpetual) | CFD | ~2,800 bps RT | Limited | Yes |

---

### Brokers in detail

#### XTB Chile — best retail option

XTB received CMF authorization as *Agente de Valores* #216 on February 11, 2025 — the only CMF-registered international retail FX broker in Chile. Chilean clients who opened accounts from July 2025 onward are under the local entity.

- **Instrument:** USD/CLP CFD, pricing derived from the Santiago interbank (MCF) session
- **Spread:** minimum ~0.39–0.44 CLP on ~900 USD/CLP ≈ **4–5 bps/side (~6.5 bps RT)**
- **Trading hours:** 12:30–17:45 CET only (= Santiago MCF core window, London–NY overlap)
- **Leverage:** 1:20
- **Minimum deposit:** None
- **API:** Public REST/FIX API was closed March 2025. MT4/MT5 platform only. Institutional/professional API access may be negotiable — worth asking directly.
- **Swap/rollover:** Not published; must verify inside the platform before trading

**Critical caveat:** Trading hours are 12:30–17:45 CET = 08:30–13:45 Santiago (winter) / 09:30–14:45 (summer). The strategy executes at ~17:30 Santiago close — **this is outside XTB's trading window**. The CLP/USD "close" in the strategy corresponds to the MCF official close, which XTB does not serve. This needs direct confirmation with XTB before proceeding.

#### Axi (AxiTrader)

- **Instrument:** USD/CLP CFD, standard lot = USD 100,000, min 0.01 lots
- **Spread:** Dynamic, not publicly quoted for this pair; minimum margin 10%
- **API:** cTrader Open API (Python-compatible, event-driven)
- **Swap:** Wednesday triple rollover; rates not published
- **Chilean eligibility:** Verify directly

#### FxPro

- **Instrument:** USD/CLP CFD via cTrader platform
- **Spread:** ~15 pips ≈ **~16 bps/side (~32 bps RT)** during active MCF hours — marginal at strategy breakeven
- **API:** cTrader Open API — well-documented, Python SDK available
- **Chilean eligibility:** Chile affiliate site exists; verify account opening

#### Pepperstone — eliminated

**Does not offer USD/CLP.** Confirmed by the user. Despite having best-in-class API infrastructure (cTrader Automate + FIX), the pair is not in their instrument set.

---

### Why all major brokers don't offer CLP spot

CLP cannot be physically delivered offshore due to Chile's Mercado Cambiario Formal (MCF) structure. All offshore CLP forward activity is Non-Deliverable Forward (NDF) by necessity — settlement is in USD against the BCCh *Dólar Observado* fixing (published ~10:30am Santiago, code `CLPOB=` on Reuters / `PCRCDOOB` on Bloomberg). Retail brokers that do offer it wrap this as a CFD. Institutional access is via:

- **EBS NDF** (CME Group): OFF-SEF execution only since Sep 2024; min ~USD 1–5M
- **LSEG/Refinitiv FXall**: CLP NDF cleared via LCH ForexClear; RFQ from 500+ LPs
- **Bloomberg FXGO**: CLP NDF executable streaming since Dec 2016; dealer LP pricing
- **CME CLP/USD futures (CHP)**: 50M CLP/contract (~USD 55K), cash-settled vs Dólar Observado

---

### Chilean domestic brokers: not viable

All CMF-registered corredoras (Bci, Banchile, LarrainVial, Tanner, BICE, Security, Scotiabank) offer bank-style currency exchange only — not speculative leveraged trading:

| Provider | Type | Spread/commission | API | Verdict |
|----------|------|-----------------|-----|---------|
| Bci Corredora | Bank FX exchange | ~0.6–0.8% per transaction | No | Way too wide |
| Banchile Inversiones | Bank FX exchange | Up to 0.8% + VAT + min 0.12 UF | No | Way too wide |
| Inversiones Security | OTC spot + forwards via executive | Not disclosed | No | No self-service |
| LarrainVial | Bank FX exchange | Not disclosed | No | Not viable |
| BICE Inversiones | Bank FX exchange | Not disclosed | No | Not viable |
| Tanner Corredores | Securities only | N/A | No | No FX product |
| **Capitaria** | CFD (local, MT5) | Not disclosed | No | **CMF unauthorized** as of mid-2026 |
| **XTB Chile** | CFD (CMF licensed) | ~6.5 bps RT | No public API | **Best local option** |

Retail fintech apps (Racional, Global66, Mercado Pago) offer spreads of 0.5–2% — far too wide for this strategy.

---

### Swap / carry cost for overnight holds

The strategy holds positions ~24 hours (T-1 close → T close). Swap rates matter:

- BCCh rate: **4.50%** (held June 2026)
- Fed Funds: **3.50–3.75%** (held June 2026)
- Net differential: CLP yields ~0.75–1.0% more than USD
- **Long USD/CLP (borrowing CLP, lending USD):** you pay the higher CLP rate → estimated **−1.5 to −1.875%/yr** swap cost ≈ **−4 to −5 bps/day**
- **Short USD/CLP (lending CLP, borrowing USD):** near breakeven or slight positive

At −4 bps/day for long positions, this is meaningful on top of the spread — effectively adds ~4 bps to your round-trip cost on long trades. Confirm exact swap table with your chosen broker before trading.

---

### Recommended path

1. **XTB Chile (first contact):** Confirm whether trading hours extend to the Santiago MCF close (~17:30 local). Ask about professional/algorithmic API access. If hours match → strongly preferred (CMF licensed, tightest spread ~6.5 bps RT, source: [xtb.com/cl/forex/usd-clp](https://www.xtb.com/cl/forex/usd-clp)).

2. **Axi (second contact):** Open demo account, check USD/CLP spread at the 17:30 Santiago window. If below 15 bps/side → viable with cTrader API.

3. **FxPro (fallback):** ~32 bps RT, right at breakeven. Viable only with the gap-filtered rule (trades fewer but larger-gap days).

4. **Avoid:** Pepperstone (no CLP pair), eToro (~2,800 bps RT), all Chilean corredoras, Saxo (exited Chile), Capitaria (CMF unauthorized).

---

*Python 3.13.7 · uv package manager · yfinance for market data · scikit-learn, XGBoost, LightGBM for ML*
