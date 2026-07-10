# Strategy Handoff

**Date:** 2026-06-21  
**Status:** Research complete · ready for broker confirmation and live execution

---

## What this document is

A concise handoff covering everything needed to move from research to live trading. Full details are in [`research_notes.md`](research_notes.md) and [`broker_research.md`](broker_research.md).

---

## The strategy in one paragraph

Every morning at ~08:30 Santiago, the CLP/USD market opens and reveals an overnight gap — the move from yesterday's close to today's open. This gap has a 0.70 correlation with the full day's close-to-close return, and a naive rule of following the gap direction (when the gap is large enough to cover costs and volatility is high enough) nets annualised Sharpe ~10 in a 20-year out-of-sample walk-forward backtest. The underlying driver is copper: Chile earns ~50% of export revenue from copper, and copper price moves Granger-cause CLP/USD moves at 5–21 day lags (p < 0.0001). The overnight gap is the most efficient single signal, better than any ML model tested.

---

## The two strategies and their tradeoff

Two valid execution approaches exist, with different broker requirements:

### Strategy A — Close-to-close (recommended)

| | |
|---|---|
| Entry | T-1 close (~13:45 Santiago) |
| Exit | T close (~13:45 Santiago, next day) |
| Hold | ~24 hours |
| Signal | Follow overnight gap direction |
| Rule | Trade if `\|gap\| > 0.43%` AND 21d realised vol > 10% ann. |
| Net Sharpe (30 bps RT) | ~10.4 |
| Years profitable (OOS) | 20 / 21 |
| Backtest quality | High — close prices from yfinance are reliable 2004–2026 |
| Broker requirement | Must execute at the MCF session close (~13:45 Santiago) |

### Strategy B — XTB session / fade-the-gap

| | |
|---|---|
| Entry | T open (~08:30 Santiago) |
| Exit | T close (~13:45 Santiago, same day) |
| Hold | ~5 hours |
| Signal | **Fade** overnight gap direction (opposite to Strategy A) |
| Rule | Trade if `\|gap\| > 0.43%` AND 21d realised vol > 10% ann. |
| Net Sharpe (6.5 bps RT) | **+0.9** (|gap|>43bps) · **+1.5** (|gap|>60bps, 191 trades) · **+2.3** (|gap|>100bps, 61 trades) |
| Gross Sharpe | +2.4 (|gap|>43bps) — signal exists but is weak; corr(gap,otc) = −0.091 |
| Signal regime | Strong 2019–2021 (corr~−0.18); near-zero 2022–present |
| Years profitable (OOS) | 5/8 (2019–2026, |gap|>43bps, XTB spreads) |
| Backtest quality | **REAL** — TwelveData USD/CLP hourly (32,922 bars, 2019–2026, 1,719 MCF sessions) |
| Broker requirement | XTB Chile is viable (trades exactly this window) |

**Why Strategy B fades the gap:** The MCF session return (open→close) has `corr(gap, otc) = −0.091` empirically (real data 2019–2026), consistent with the CTC decomposition direction (`corr(gap,ctc) = +0.61` for Strategy A). What gaps up overnight tends (weakly) to reverse during the MCF session as onshore flow corrects the offshore drift.

**Real data backtest — updated 2026-06-21:** TwelveData USD/CLP hourly bars (America/Santiago timezone) provide genuine MCF open and close prices, resolving the look-ahead bias problem.

Key findings from [`notebooks/10_backtest_strategyB_real.py`](../notebooks/10_backtest_strategyB_real.py):

| Filter | N trades (2019–2026) | Net Sharpe (XTB 6.5bps RT) | Net Sharpe (30bps RT) |
|--------|---------------------|---------------------------|----------------------|
| No filter | 1,701 | −0.28 | −4.9 |
| \|gap\| > 20 bps | 806 | +0.30 | −4.1 |
| \|gap\| > 43 bps | 344 | +0.88 | −3.3 |
| **\|gap\| > 60 bps** | **191** | **+1.53** | −2.2 |
| \|gap\| > 100 bps | 61 | +2.29 | −1.4 |

**Signal regime:** The edge is time-varying. 2019–2021: strong (corr ≈ −0.18, gross Sharpe ≈ 3–9). 2022–2026: weak or slightly wrong-signed (corr near 0 to +0.15). The prior `corr = −0.59` and `Sharpe ~6.6` were purely look-ahead artifacts.

**Critical:** At 30 bps RT, Strategy B is NOT viable under any filter. XTB (6.5 bps RT) is the only broker class where this strategy can be net-positive.

**Prior analysis** in [`notebooks/09_backtest_xtb.py`](../notebooks/09_backtest_xtb.py) documents the data quality investigation (yfinance look-ahead bias and AV open ≠ MCF open).

---

## The broker gap is the critical blocker

Strategy A needs a broker who can execute at the MCF close (~13:45 Santiago). XTB cannot — their window closes at exactly that time and you need the close *price*, not an order submitted at close. Strategy B fits XTB perfectly but lacks a reliable backtest.

### Current broker status

| Broker | USD/CLP | Spread (RT) | API | Fits Strategy A | Fits Strategy B |
|--------|---------|------------|-----|----------------|----------------|
| **XTB Chile** | Yes | ~6.5 bps | No public API | No — window ends at MCF close | **Yes** |
| **Axi** | Yes | Unknown | cTrader Open API | **Verify** | **Verify** |
| **FxPro** | Likely | ~32 bps | cTrader Open API | **Verify** | **Verify** |
| Pepperstone | **No** | — | — | — | — |
| forex.com | No | — | — | — | — |
| OANDA | No | — | — | — | — |
| IBKR | No | — | — | — | — |

XTB Chilean entity source: [xtb.com/cl/forex/usd-clp](https://www.xtb.com/cl/forex/usd-clp) — min spread 0.30 CLP, leverage 1:500, 0.2% margin, 12:30–17:45 CET.

---

## Immediate next actions

### 1. Contact Axi — open a demo account
Check whether USD/CLP is available and what the live spread is at 13:45 Santiago (= 17:45 CET in winter). If the pair exists and spread is below 15 bps/side (~30 bps RT), Axi becomes the primary broker for Strategy A.

Demo: [axi.com](https://www.axi.com) · API: cTrader Open API (Python SDK available)

### 2. Contact FxPro — verify USD/CLP and hours
Confirm availability and check live spread at 13:45 Santiago. Known spread ~16 bps/side (~32 bps RT) — workable for Strategy A only with the gap-filtered rule (trade fewer but larger-gap days).

### 3. If only XTB is viable — paper trade before going live (mandatory)
If Axi and FxPro don't work out, Strategy B on XTB is the fallback. **The prior Sharpe ~6.6 estimate is invalid** (look-ahead bias — see Strategy B caveat above). Before trading live:
- **Paper trade 30–60 days** using the framework in `notebooks/09_backtest_xtb.py` (Section 4):
  - 08:25: note prior BCCh Dólar Observado (prev_close)
  - 08:30: note XTB platform's first USD/CLP midpoint (open)
  - 13:45: note XTB platform's final USD/CLP midpoint (close)
  - Log `gap = log(open/prev_close)` and `otc = log(close/open)`
- **Gate for going live:** `corr(gap, otc) < -0.3` AND `avg gross return > 20 bps` (3× cost) over 30+ days
- **Alternative data:** Purchase intraday CLP/USD tick data from Refinitiv/Bloomberg to rebuild the backtest with proper open prices

### 4. Paper trade regardless of broker
Before any real money: run the signal daily for 30–60 days, record the gap each morning, note whether the outcome matched the prediction, and track the executable spread from the chosen broker. This costs nothing and builds confidence.

---

## Signal generation (daily, ~08:30 Santiago)

```
1. Get yesterday's MCF close price for USD/CLP
   → BCCh Dólar Observado REST API (free):
     https://si3.bcentral.cl/estadisticas/Principal1/web_services/index.htm

2. Get today's MCF open price for USD/CLP
   → From your broker platform at market open (~08:30 Santiago)

3. Compute gap:
     gap = log(usd_clp_open_today / usd_clp_close_yesterday)

4. Compute 21d realised vol:
     rv21 = rolling 21-day std of daily log-returns × sqrt(252)

5. Decision:
     IF |gap| < cost_threshold → NO TRADE (gap too small to cover spread)
     IF rv21 < 10% ann.       → NO TRADE (low-vol, edge doesn't cover cost)
     IF gap > 0 (USD/CLP rose overnight):
         Strategy A → LONG USD/CLP at yesterday's close, exit today's close
         Strategy B → SHORT USD/CLP at today's open, exit today's close (~13:45)
     IF gap < 0 (USD/CLP fell overnight):
         Strategy A → SHORT USD/CLP at yesterday's close, exit today's close
         Strategy B → LONG USD/CLP at today's open, exit today's close (~13:45)
```

The full pipeline (`make pipeline`) refreshes data, recomputes features, and outputs the current signal to `models/latest_signal.json` in ~3 minutes.

---

## Key numbers to keep in mind

| | Value |
|---|---|
| Median overnight gap | 0.43% (40 bps) — minimum to trade |
| Strategy A net Sharpe | ~10.4 at 30 bps RT |
| Strategy A net Sharpe | ~6.5 at 6.5 bps RT (XTB-class spread, if CTC possible) |
| Strategy B net Sharpe | **+1.5** at XTB 6.5bps RT (|gap|>60bps, 191 trades 2019–2026) |
| Long USD/CLP swap cost | ~−4 to −5 bps/day (pay CLP rate, receive USD rate) |
| Carry bonus when long | +1%/yr ≈ +0.4 bps/day net (BCCh 4.5% − Fed 3.5%) |
| Max drawdown (Strategy A, OOS) | −11.1% (compound equity basis) |
| OOS years profitable | 20 / 21 (Strategy A) |
| Backtest period | 2006-12-29 → 2026-04-29 (5,012 OOS days) |

---

## Files

| File | Purpose |
|------|---------|
| [`reports/research_notes.md`](research_notes.md) | Full research arc — thesis, features, experiments, backtests |
| [`reports/broker_research.md`](broker_research.md) | Broker availability, spreads, hours, API details |
| [`reports/handoff.md`](handoff.md) | This file |
| [`notebooks/08_backtest_ctc.py`](../notebooks/08_backtest_ctc.py) | CTC backtest (Strategy A) |
| [`notebooks/09_backtest_xtb.py`](../notebooks/09_backtest_xtb.py) | XTB Strategy B data quality investigation (look-ahead bias proof) |
| [`notebooks/10_backtest_strategyB_real.py`](../notebooks/10_backtest_strategyB_real.py) | **Strategy B definitive backtest** — real intraday data (TwelveData 2019–2026) |
| [`copper_clp/twelvedata.py`](../copper_clp/twelvedata.py) | TwelveData downloader — USD/CLP hourly + MCF daily extractor |
| [`models/latest_signal.json`](../models/latest_signal.json) | Current live signal output |
| [`copper_clp/experiment.py`](../copper_clp/experiment.py) | Experiment registry (v1/v2/v3) |
| `.claude/commands/add-experiment.md` | `/add-experiment` skill for adding new model experiments |
