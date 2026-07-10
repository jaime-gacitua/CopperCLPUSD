# PercentileReversion_USDCLP — MT5 EA (Capitaria)

Percentile mean-reversion, designed 2026-07-10. Jaime's framing is CLPUSD; Capitaria quotes USDCLP, so the EA translates: sell CLPUSD @ its 90th pct = **buy USDCLP @ its 10th pct (732.5)**; buy CLPUSD @ its 10th pct = **sell USDCLP @ its 90th pct (955.7)**.

## Rules
- Bands = 10th / 50th / 90th percentiles of daily closes over a **fixed** training window 2021-01-01 → 2024-12-31 (static, not rolling).
- Flat + close ≥ p90 → short USDCLP. Flat + close ≤ p10 → long USDCLP.
- Exit when close crosses back through the 50th percentile (median).
- One position at a time, no scale-in. Server-side catastrophe stop 8% from entry.
- Acts once per new D1 bar on the previous day's close; fills at market open.

## Python baseline (usdclp_daily.csv, 40 bps RT cost)
Bands: buy ≤ 732.54, exit 854.91, sell ≥ 955.71. OOS 2025 → 2026-06-19:
**1 trade** — short 2025-01-03 @ 1004, exit at median 2026-02-11 @ 855, **+14.4% net**, max adverse ~2%. Buy side never triggers (price never revisited 732). EA percentile math verified ≡ numpy.

## Install
1. MT5: **File → Open Data Folder**.
2. `PercentileReversion_USDCLP.mq5` → `MQL5\Experts\`
3. `usdclp_daily.csv` → `MQL5\Files\` (already there if StrategyA installed) — used only to fill dates before Capitaria's broker history begins.
4. F4 (MetaEditor) → open EA → F7 compile, expect 0 errors.

## Backtest (Strategy Tester, Ctrl+R)
- Expert: `PercentileReversion_USDCLP` · Symbol: **USDCLP** · Period: **D1**.
- Dates: **2025-01-01 → today**. Modelling: "Open prices only" is fine — the EA only acts on daily closes.
- Check the Journal's first lines: it prints how many training closes came from broker history vs CSV and the resulting band levels. If bands differ much from 732.5 / 854.9 / 955.7, broker vs yfinance close-timing differences are the cause — trust the broker numbers.
- Expect ~1 short held Jan 2025 → Feb 2026. If the tester shows something very different, read the Journal day by day.

## Optimization (next step)
1. Strategy Tester → **Optimization: Slow complete algorithm** (few combos, no need for genetic).
2. In Inputs, tick and set ranges:
   - `InpBuyPercentile`: start 2, step 4, stop 30
   - `InpSellPercentile`: start 70, step 4, stop 98
   - optionally `InpExitPercentile`: start 30, step 10, stop 70
3. Criterion: Balance + max Profit Factor is fine given the tiny trade count.

**Caveats before trusting any optimum:** the OOS window has essentially one regime (high → reversion to ~855). ~2 years and a handful of trades cannot distinguish skill from luck; optimizing percentiles on 2025–26 and reporting the best is in-sample tuning on the test set. Treat the optimizer output as sensitivity analysis, not expected performance. A more honest design: optimize on 2021–2024 (walk-forward inside), keep 2025–26 untouched as the single holdout.

## Safety — real account
Same lock as StrategyA: live it **logs signals but will not trade** unless both `AllowLiveTrading=true` and `IUnderstandRealMoneyRisk=true`. Tester always trades (simulated).

## Known limitations
1. Static bands go stale — CLP has trended; a level regime shift (like 2021→2022) leaves the buy band unreachable for years.
2. Median exit means months-long holds; swap/rollover costs on USDCLP are NOT modeled in the Python baseline but ARE charged by the tester if Capitaria reports them — compare.
3. One-sided results: everything rests on a single short. Do not annualize.
4. yfinance CSV closes are day-shifted (known artifact) — irrelevant for level percentiles, but the tester's broker-history bands are the authoritative ones.
