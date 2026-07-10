# MA-Deviation Fade on USDCLP — Findings

**Date:** 2026-07-10 (Friday)

## Question

Is there a sensible moving-average strategy for buying/selling/closing USDCLP? Two candidates were considered: a slow MA crossover (trend) and an MA-deviation fade (reversion). Given USDCLP's range-bound, bimodal behavior 2021–26 and Capitaria's ~40 bps round-trip cost, the fade was chosen for testing.

## Strategy tested

- z = (close − SMA_n) / rolling std_n on daily closes
- Short USDCLP when z > +thr, long when z < −thr
- Exit when z crosses 0; 8% catastrophe stop; one position at a time
- Signals on close of day t, filled at close of day t+1 (no lookahead)
- Cost: 40 bps round trip; swap not modeled

Data: `mt5/usdclp_daily.csv` (Capitaria-based, 2019-01 → 2026-06-19 — the clean series; yfinance dailies are known-bad, see `reports/conclusions.md`).

Protocol: parameter grid selected on 2021–2024 only; 2025-01-01 → 2026-06-19 held out.

## Results

**Unfiltered** (`ma_fade_backtest.py`, grid SMA {50,100,150,200} × thr {1.5,2.0,2.5}):

- In-sample 2021–24: all 12 configs lose, −6% to −23%. The 2021–22 CLP collapse (USDCLP ~700→1050) stops out every counter-trend long.
- Holdout 2025–26: most configs gain +5–12% (best SMA50/thr1.5: +11.4%, Sharpe 0.66, 13 trades, worst trade −8.5% stop-out).

**Trend-filtered** (`ma_fade_filtered.py`, entries gated by SMA200 20-day slope: longs only if slope ≥ −eps, shorts only if ≤ +eps, eps {0, 0.5%}):

- In-sample 2021–24: all 24 configs still negative (best −1.9%, Sharpe ≈ 0). Even dip-buying restricted to uptrends lost through 2021–22.
- Holdout deliberately not run (no positive in-sample config to select), so 2025–26 remains a clean holdout for future work.

## Conclusion

**Not validated — dead end at daily horizon.** The strategy is regime-dependent, not an edge: it profits when USDCLP ranges (2025–26) and bleeds when it trends (2021–24). Since no configuration was positive in-sample, the holdout gains are regime luck; selecting a config from them would be tuning on the test set.

## Implications

1. Daily z-reversion against its own MA does not clear 40 bps on USDCLP, with or without a trend gate.
2. Cost dominates: at ~6.5 bps (XTB) rather than 40 bps (Capitaria), several fade configs would flip positive. Broker choice may matter more than strategy tweaks.
3. The static percentile-band strategy (`mt5/PercentileReversion_USDCLP.mq5`) remains the only candidate with a positive holdout result (+14.4%, 1 trade). Next step: optimize its percentiles within 2021–24 in the MT5 optimizer, keeping 2025–26 untouched.
4. Discipline reminder: do not iterate further variants against the 2025–26 holdout; each peek degrades it.

## Files

- `ma_fade_backtest.py` — unfiltered grid + holdout
- `ma_fade_filtered.py` — SMA200-slope-gated variant (holdout skipped)
- `mt5/usdclp_daily.csv` — clean daily series used
