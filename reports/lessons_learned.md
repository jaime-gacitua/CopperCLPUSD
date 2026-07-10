# Lessons Learned — CLP/USD Copper Strategy

**Date:** 2026-07-10
**Outcome:** Strategy A invalidated before any capital was risked. The MT5 backtest on real broker data caught a data artifact that two rounds of prior auditing missed.

---

## What happened, in one paragraph

The research (June 2026) found that copper returns "predicted" next-day USD/CLP moves with remarkable strength: 71% win rate, net Sharpe 6–10, profitable 20 of 21 years. We ported the strategy to an MT5 Expert Advisor and backtested it on Capitaria's real price history. It produced 15 trades and +0.7% over 5.5 years. The discrepancy traced back to a single data defect: yfinance's CLPUSD=X daily closes are dated one day off — the return stamped on date T is actually the market's move of day T-1. The model was therefore "predicting" a move that, in real time, had already happened, using copper's same-day co-movement with CLP as the leak. On correctly-dated data the edge does not exist.

## The evidence chain

The EA's model, trained on Capitaria's own bars, produced a copper coefficient of −0.04 where the research said −0.42. Sampling TwelveData hourly closes at every hour of the day reproduced the broker's answer, not the research's. The decisive test was lagged return correlations: yfinance returns correlate with TwelveData returns at 0.70 **one day earlier** and −0.01 same-day. AlphaVantage, a third source, agrees with TwelveData same-day (0.90) and with yfinance only at a one-day lag (0.64). Three independent sources against one: yfinance's dating is wrong.

A rebuild of the full feature matrix on TwelveData MCF closes (`notebooks/11_rebuild_features_td.py`), with every feature strictly observable at the 13:45 Santiago decision time, followed by a clean walk-forward (`notebooks/12_backtest_corrected.py`), confirmed it: directional accuracy 48–51% across logistic and gradient-boosting models, and every configuration deeply negative at Capitaria's measured 40 bps round-trip cost.

## Why the June audit missed it

The audit was rigorous about the failure mode it imagined: it caught that yfinance's *open* prices were fake (74% same-day closes) and correctly banned them. But it validated *closes* by level — and the misdated series has 0.992 level correlation with the true one, because USD/CLP moves ~60 bps a day while sitting near 900. A series can pass every level-based, plot-based, and summary-statistic check while being shifted a full day. Only cross-source **return** correlation at lags exposes it.

## The lessons

**Date alignment is a first-class data-quality check.** Before trusting any daily series for event-timing work, cross-correlate its returns against an independent source at lags −2…+2. The correct source pairs at lag 0. Level correlation proves nothing about dating.

**Backtest performance that looks too good usually is.** Sharpe 6–10 in daily FX should have been the loudest alarm. Same-day copper–CLP correlation is strong and well known; a "predictive" model whose power vanishes when features are lagged one day is describing the present, not forecasting.

**Test on the execution venue's own data before believing anything.** The single most informative experiment of the entire project was the cheapest: running the EA against Capitaria's actual price history. Broker data embeds the true day boundaries, the true spread, and the true tradeable prices. It answered in minutes what months of research data could not.

**Costs are a filter most edges don't pass.** At 6.5 bps (XTB) several marginal configurations flicker at Sharpe ~1; at 40 bps (Capitaria) nothing survives, in any model, in any period. Measure the venue's spread before researching a strategy, not after — it determines which hypotheses are even worth testing.

**A sweep over thresholds on OOS data is in-sample fitting in disguise.** The "best cell" across a 36-configuration grid hitting Sharpe +1.0 is exactly what noise produces. Fixed decision rules, chosen before looking, are the only ones whose OOS numbers mean anything.

**The safety rails did their job.** Paper-trade-first discipline, the EA's live-trading lock, and treating the tester as a falsification tool (not a confirmation tool) meant the total cost of this error was a few hours — not a funded account discovering the truth at 40 bps per round trip.

## What remains standing

The MT5 EA harness (`mt5/`) is verified and reusable: correct walk-forward logistic training (identical to sklearn), broker-data mode, safety locks. The corrected feature matrix (`data/processed/feature_matrix_td.csv`) is clean and ready for new hypotheses. Strategy B's intraday fade edge (TwelveData-backtested, Sharpe ~1.5 at 6.5 bps) was never touched by the artifact — but it needs a tight-spread broker. Open research questions that survive the correction: weekly-horizon copper→CLP effects (the original Granger result was at 5–21 day lags, and longer holds dilute costs 5–20x), and intraday lead-lag, which requires hourly copper data.

## Data source verdicts (updated)

| Source | Verdict |
|---|---|
| yfinance CLPUSD=X daily open | Fake — banned (June audit) |
| yfinance CLPUSD=X daily close | **Misdated by one day — banned for any timing-sensitive use (this audit)** |
| yfinance HG=F daily OHLC | Dates consistent with exchange settlements — usable |
| TwelveData USD/CLP hourly | Correct — reference series for CLP dating |
| AlphaVantage FX daily | Dates consistent with TwelveData — usable as cross-check |
| Capitaria MT5 history | Correct by construction (it is the venue) — 97% quality 2021–2026 |
