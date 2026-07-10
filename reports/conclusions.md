# CLP/USD Trading Strategy — Research Conclusions

**Date:** 2026-06-21  
**Status:** ⛔ **INVALIDATED 2026-07-10 — see below. Do not trade this.**

> **⛔ 2026-07-10 — Backtest invalidated by data-dating artifact.**
> yfinance CLPUSD=X daily **closes are dated one day off**: the return stamped on date T is the market's move of day T-1. Verified against three independent sources that agree with each other and disagree with yfinance: TwelveData hourly (return corr 0.70 at lag-1, -0.01 same-day), AlphaVantage FX daily (0.90 same-day with TwelveData, 0.64 only at lag+1 with yfinance), and Capitaria MT5 broker history.
> Consequence: the model's "copper predicts tomorrow's CLP" effect was actually **same-day copper–CLP co-movement leaking through the misdated target**. Refitting on correctly-dated closes (any source, any sampling hour) gives copper weight ≈ 0 and no tradeable edge. An MT5 Strategy Tester run on real Capitaria data over 2021–2026 produced 15 trades and +0.7% — the true out-of-sample result.
> The earlier data-quality audit caught yfinance's fake *opens* but trusted its *closes*; the closes are consistent in level (corr 0.992) yet shifted in date, which no level-based check detects.
> Details: `mt5/README.md` and the 2026-07-10 session. Strategy B's TwelveData backtest (weak edge, Sharpe ~1.5 at 6.5 bps RT) is unaffected by this artifact but remains unviable at Capitaria's 40 bps spread.
>
> **Corrected rebuild (same day):** `notebooks/11_rebuild_features_td.py` builds `data/processed/feature_matrix_td.csv` — 41 strictly-lagged features on TwelveData MCF closes. `notebooks/12_backtest_corrected.py` re-runs the walk-forward with logistic (3-feature and full) and gradient boosting: directional accuracy 48–51%, best cherry-picked cell net Sharpe ≈ +1.0 at 6.5 bps (within data-mining noise across the grid), and **every configuration deeply negative at Capitaria's 40 bps** (best −4.4). No daily copper→CLP edge exists at this horizon on correct data.

---

## The strategy in plain terms

Every day at 13:40 Santiago, you check two things: what did the peso do yesterday, and what did copper do yesterday and over the last week. You feed those three numbers into a simple model. The model tells you whether to bet that the peso will weaken or strengthen tomorrow. You hold the position for exactly 24 hours — from today's close to tomorrow's close. That's it.

You do this about 3 times a week, only when the model is confident enough to act. The rest of the time you're flat.

**What you need every day:**
- Yesterday's USD/CLP closing rate (takes 30 seconds from BCCh website)
- Yesterday's copper price (takes 30 seconds from yfinance)
- ~2 minutes to run the model and check the signal

**Backtest results (2021–2026, fully out-of-sample):**
- Win rate: 71% — the model is right more often than wrong
- ~72 trades per year (~3 per week)
- Profitable 5 out of 6 years even at expensive broker spreads

---

## The daily routine (13:40 Santiago, every trading day)

### Step 1 — Check if you have an open position
If yes, close it now at market price. You always hold exactly one day.

### Step 2 — Compute today's signal (2 minutes)

Get these three numbers (all from yesterday's closing prices):

```
gap     = (today's USD/CLP close − yesterday's) / yesterday's   [in %]
cu_lag1 = (copper close yesterday − copper close day before) / day before   [in %]
cu_5d   = (copper close yesterday − copper close 5 days ago) / 5 days ago   [in %]
```

Run the model. It outputs a probability between 0 and 1.

### Step 3 — Decide

```
probability > 0.60  AND  |gap| > 0.43%  →  BUY USD/CLP now
probability < 0.40  AND  |gap| > 0.43%  →  SELL USD/CLP now
anything in between                      →  do nothing, stay flat
```

### Step 4 — Hold until tomorrow 13:40
Do not touch the position. Close it tomorrow at 13:40 regardless of what happens intraday.

---

## Broker — the most important next step

**The strategy only works if your broker offers USD/CLP and is open at 13:40–13:45 Santiago.** Most brokers don't offer this pair. Below are the two best options found.

### Option 1 — XTB Chile (recommended to try first)

XTB is the broker you are registering with. They offer USD/CLP as a CFD.

| | |
|---|---|
| USD/CLP available | Yes |
| Hours | 12:30–17:45 CET (winter) = **08:30–13:45 Santiago** — covers the full MCF window |
| Spread (RT) | ~6.5 bps — the tightest found for this pair |
| API | No public API (discontinued March 2025) — manual execution only |
| Regulation | Regulated in Poland (KNF), operates in Chile |

**At XTB's spread of ~6.5 bps RT, the backtest gives net Sharpe +6.71.** This is the best-case scenario.

The one risk: XTB's session closes at exactly 13:45 Santiago, the same moment the MCF closes. You need to place your closing order at 13:40 to ensure a good fill before liquidity thins out in the final minutes. Test this in paper trading first.

**Spread source:** xtb.com/cl/forex/usd-clp — minimum spread 0.30 CLP on a ~900 rate ≈ 3 bps/side ≈ 6 bps RT.

---

### Option 2 — Capitaria (Chilean broker, worth checking)

Capitaria is a Chilean broker with 14+ years of operation and ~USD 580M daily volume. They explicitly offer USD/CLP trading and in August 2023 became the first Latin American broker to extend USD/CLP trading through the lunch hour.

| | |
|---|---|
| USD/CLP available | Yes, as a CFD |
| Hours | **08:40–17:00 Santiago** (Mon–Thu), 08:40–16:00 (Fri) — covers MCF close at 13:45 ✓ |
| Spread (RT) | Variable, not publicly disclosed — EUR/USD is ~3 pips (wide); USD/CLP unknown |
| API | MetaTrader 5 (MT5) — supports automated trading via Expert Advisors |
| Regulation | UAF supervised; CMF authorization pending as of April 2026 |
| Website | capitaria.com |

**Capitaria opens at 08:40, missing the first 10 minutes of the MCF session (08:30–08:40).** For this strategy that doesn't matter — you only trade at 13:40 close, not at open.

**The MT5 platform is a significant advantage** — it means you could eventually automate the signal and execution without needing a manual step each day. XTB has no API.

**Action:** open a demo account at Capitaria, check the live USD/CLP spread at 13:40 on an active day. If the spread is below ~15 bps RT, Capitaria becomes the better long-term choice due to MT5 automation potential.

---

### Broker comparison

| | XTB | Capitaria |
|---|---|---|
| USD/CLP | ✓ | ✓ |
| Covers 13:45 MCF close | ✓ | ✓ |
| Spread | ~6.5 bps RT (tight) | Unknown (likely wider) |
| Automation | No API | MT5 — automatable |
| Regulation | Strong (KNF) | Partial (CMF pending) |
| **Best for** | Starting now, lowest cost | Long-term, automated trading |

---

## Paper trade before going live (mandatory)

Before putting real money in, run the strategy on paper for 20–30 trading days:

1. Each day at 13:40, compute the signal and write down what you *would* have done
2. Record the entry price (today's close) and the exit price (tomorrow's close)
3. After 20+ trades, check: is the win rate above 55%? Is the average trade positive?

If yes — go live with a small position. If the win rate is below 50% for 30+ trades, something has changed and you should stop and investigate before committing capital.

---

## What we ruled out

- **yfinance open prices for CLP/USD** — look-ahead bias, 74% of "opens" are actually same-day closes
- **Alpha Vantage FX_DAILY** — open prices not aligned with MCF session
- **Dukascopy** — doesn't cover USD/CLP
- **XTB API** — discontinued March 2025
- **Exness** — no evidence they offer USD/CLP
- **Pepperstone, forex.com, OANDA, IBKR** — confirmed no USD/CLP
- **Pre-2018 data for model training** — wrong market regime, poisons any model
- **Complex ML models (LightGBM)** — not enough data in the new regime; simple logistic regression works better

---

## Key numbers

| | |
|---|---|
| Training window | Post-2020 data only (~1,600 trading days) |
| Model | Logistic regression, 3 features |
| Features | Yesterday's USD/CLP return, copper yesterday, copper 5-day |
| Model refit | Every 3 months (quarterly) |
| Trade filter | Model confidence > 60% AND overnight gap > 0.43% |
| Trade frequency | ~72 trades/year (~3/week) |
| Hold period | ~24 hours (close to close) |
| Net Sharpe @ 6.5 bps | +6.71 |
| Net Sharpe @ 15 bps | +5.38 |
| Net Sharpe @ 30 bps | +3.02 |
| Win rate | 71.4% |
| OOS period | 2021–2026 (never seen by model during training) |
| Profitable years @ 30 bps | 5 out of 6 |

---

## Research findings (technical summary)

### The signal

The overnight gap (yesterday's USD/CLP close-to-close return) predicts today's return. Copper's lagged return adds independent information. Together they give the model enough signal to be right 71% of the time when it's confident.

### The regime shift

The market changed structurally around 2018. Before 2018, the gap signal alone was very strong (correlation −0.58) and the market was highly volatile (47% annualised). After 2018, volatility halved (15% annualised) and the gap signal weakened (correlation −0.11), but copper's lagged return emerged as a new, independent signal (correlation −0.25) that didn't exist before. Any model trained on pre-2018 data learns the wrong market and should be discarded.

### Data quality

| Data source | Verdict |
|------------|---------|
| yfinance CLPUSD=X daily **close** | Reliable |
| yfinance CLPUSD=X daily **open** | Contaminated — do not use |
| yfinance HG=F daily OHLC | Reliable (91% real opens) |
| TwelveData USD/CLP hourly | Reliable (saved locally) |

---

## Files

| File | Purpose |
|------|---------|
| [`reports/conclusions.md`](conclusions.md) | This file |
| [`reports/handoff.md`](handoff.md) | Earlier strategy handoff |
| [`notebooks/09_backtest_xtb.py`](../notebooks/09_backtest_xtb.py) | Data quality investigation |
| [`notebooks/10_backtest_strategyB_real.py`](../notebooks/10_backtest_strategyB_real.py) | Intraday Strategy B backtest |
| [`copper_clp/twelvedata.py`](../copper_clp/twelvedata.py) | USD/CLP hourly data downloader |
| `data/raw/td_usdclp_hourly.csv` | 32,922 hourly USD/CLP bars (2019–2026) |
| `data/raw/copper_hgf_daily.csv` | HG=F daily OHLC (2000–2026) |
| `data/raw/clpusd_daily.csv` | CLPUSD=X daily closes (2003–2026) |
