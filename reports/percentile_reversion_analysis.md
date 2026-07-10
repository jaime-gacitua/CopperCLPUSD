# Percentile mean-reversion on USDCLP — analysis log

**Date:** 2026-07-10
**Data:** yfinance CLPUSD=X daily closes inverted to USDCLP (`data/raw/clpusd_daily_full.csv`, 2004–2026; two bad ticks removed: 2014-04-10, 2016-12-22). Known yfinance one-day date-shift artifact does not affect level percentiles. Definitive backtests to be run on Capitaria's own history in the MT5 Strategy Tester.
**Deliverables:** `mt5/PercentileReversion_USDCLP.mq5` + `mt5/README_PercentileReversion.md`

## The idea

Jaime's framing (CLPUSD): sell CLPUSD at its 90th percentile, buy at its 10th. Capitaria quotes USDCLP = 1/CLPUSD, so translated: **buy USDCLP at its 10th percentile, sell (short) at its 90th**. Rules confirmed: exit when price crosses back through the median (50th percentile); one position at a time, no scale-in; 8% catastrophe stop; bands static, computed from a fixed training window. All Python baselines assume 40 bps round-trip cost (Capitaria's measured USDCLP spread) and EA timing (signal on close of day T, fill on day T+1).

## Experiment 1 — train 2021–2024, test 2025–2026 YTD

Bands: buy ≤ 732.5, exit 854.9, sell ≥ 955.7.

Result: **1 trade.** Short Jan 2025 @ ~1004 → exit at median Feb 2026 @ ~855, **+14.4% net**, max adverse ~2%. Buy side never triggered — 732.5 was last touched July 2021 (it traded ≤732.5 on 105 days, all Jan–Jul 2021; window low 694.9 on 2021-05-10). Training distribution is bimodal (~800 and ~920).

## Experiment 2 — train Jan 2015 – Jan 2025 (10 years), test 2025–2026

Motivation: more data for more robust statistics. Deciles (2,605 days):

| p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 |
|---|---|---|---|---|---|---|---|---|
| 630.0 | 653.4 | 669.8 | 691.7 | 721.8 | 779.6 | 806.2 | 857.6 | 923.7 |

Result: the short enters Jan 2025 @ ~1004 but the exit (median 721.8) belongs to the pre-2020 peso and is never reached — **position still open 18 months later** (+9.7% MTM, nothing realized). Buy level 630 last touched in 2018. Lesson: more history made the statistics more precise but *less relevant*. The 10-year distribution mixes two regimes (600–730 in 2015–19, 780–1050 in 2020–24) and its deciles describe neither.

## Experiment 3 — train Jun 2023 – May 2025, test Jun 2025 – Jun 2026

Deciles (519 days):

| p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 |
|---|---|---|---|---|---|---|---|---|
| 851.6 | 882.0 | 906.6 | 920.1 | 932.1 | 941.2 | 947.4 | 961.6 | 978.3 |

Result: **1 trade.** Short 2025-08-01 @ 972 → exit at median 2025-10-28 @ 940, **+2.9% net**, max adverse 0.2%. Two near-misses: the Feb 2026 low (852.4) missed the buy band by **0.8 pesos**, and the Sep 2025 highs (~973) stayed just under the 978.3 re-entry.

## Conclusion — fixed-window level bands are not robust

Three windows, one idea, three verdicts:

| Training window | Sell band | Outcome (test period) |
|---|---|---|
| 2021–2024 | 955.7 | +14.4% realized (1 trade) |
| 2015–2025 | 923.7 | never exits; +9.7% unrealized |
| Jun 2023–May 2025 | 978.3 | +2.9% realized (1 trade) |

When the outcome swings this much on an arbitrary date choice — and signals hinge on sub-peso margins — the bands are capturing where the box was drawn, not a property of the market. USDCLP drifts structurally (inflation differential), so any fixed level goes stale; the only question is how fast. And at ~1 trade/year, no test period of practical length can separate skill from luck.

## Agreed next direction

Replace fixed bands with ones that move with the market:

1. **Rolling percentile bands (recommended):** recompute 10/50/90 daily from the trailing N days (~250–500). Kills window sensitivity, walk-forward by construction, single parameter N, more trades. EA modification pending.
2. Alternative: **z-score bands** — fade extremes of (price − MA) / rolling std; the parametric cousin of rolling percentiles.

Optimization plan: sweep lookback + buy/sell percentiles in the MT5 optimizer over 2021–2026 (several regimes). Caveat that stands regardless: optimizing on the test window is tuning on the test set — treat optimizer output as sensitivity analysis unless validated walk-forward.

## Status of the fixed-window EA

`PercentileReversion_USDCLP.mq5` is ready for the Experiment 3 run on Capitaria data: `InpTrainStart=2023.06.01`, `InpTrainEnd=2025.05.31`, tester Jun 2025 → today, symbol USDCLP, period D1. Percentile math verified identical to numpy. Bands computed from broker D1 history (CSV fills pre-history dates); Journal logs the levels — compare against 851.6 / 932.1 / 978.3. Live-trading safety lock (same pattern as StrategyA) is on by default.
