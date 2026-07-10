# Strategy A — MT5 Expert Advisor (Capitaria)

> **⛔ 2026-07-10: Strategy A's research backtest was invalidated.** yfinance daily closes are dated one day off, so the model was unknowingly using same-day copper moves to "predict" a target that had already happened. On correctly-dated data (Capitaria, TwelveData, AlphaVantage — all mutually consistent) the edge disappears. The tester run below (15 trades, +0.7% over 5.5 years) is the honest result. See warning in `reports/conclusions.md`. The EA remains useful as a harness for testing corrected models.

Close-to-close CLP/USD strategy from `reports/conclusions.md`, ported to MQL5 for the MT5 Strategy Tester. Verified against a Python refit on 2026-07-10: 68% win rate, ~76 trades/yr, net Sharpe ≈ 6.0 @ 6.5 bps RT, ≈ 2.2 @ 30 bps RT (2021–2026 OOS, profitable every year).

## Files

| File | Purpose |
|---|---|
| `StrategyA_CLPUSD.mq5` | The Expert Advisor |
| `usdclp_daily.csv` | USD/CLP daily closes 2019–2026 (yfinance, reliable) — training + tester features |
| `copper_daily.csv` | HG=F copper daily closes 2019–2026 — training + tester features |

## Install (5 minutes)

1. In MT5: **File → Open Data Folder**. This opens the terminal's data directory.
2. Copy `StrategyA_CLPUSD.mq5` → `MQL5\Experts\`
3. Copy both CSVs → `MQL5\Files\`
4. In MT5 press **F4** (MetaEditor), open the EA, press **F7** (Compile). Expect 0 errors.
   - No MQL5.community login is needed — that account is separate from your Capitaria-All login and only required for the Market/cloud services.

## Run the backtest

1. **View → Strategy Tester** (Ctrl+R).
2. Expert: `StrategyA_CLPUSD` · Symbol: **USDCLP** · Period: **M5**.
3. Dates: 2021-01-01 → **2026-06-19** (the CSVs end there; beyond that the tester's gap feature goes stale). Also limited by Capitaria's USDCLP history — if it starts later, start there.
4. Modelling: **"1 minute OHLC"** (good balance) or "Open prices only" (fastest — fine, the EA only acts once per day).
5. Inputs to check:
   - `TradeHour` / `TradeMinute` — **server time** matching ~13:40 Santiago. Compare the Market Watch clock with Santiago local time to get the offset (Capitaria-All is typically UTC-4 → 13:40; verify).
   - `Lots` — 0.10 to start.
   - Leave thresholds at defaults (0.60 / 0.40 / 0.0043) — they are the researched values.
6. Press **Start**. The Journal logs every daily decision: gap, copper features, P(up), action.

## Safety — real account

Your terminal is logged into a **real** Capitaria account. The EA is locked accordingly:

- In the Strategy Tester it always works (simulated money).
- Attached to a live chart it will **compute and log signals but refuse to trade** unless you set BOTH `AllowLiveTrading=true` AND `IUnderstandRealMoneyRisk=true`.
- Default `EmergencyStopPct=3` puts a 3% catastrophe stop on every position.

Recommended path: backtest → run on live chart with the lock ON (it becomes a free paper-trading signal logger, matching the 30–60 day paper-trade gate in `reports/handoff.md`) → only then consider enabling.

## How the EA works

- Once per day in a 5-minute window at `TradeHour:TradeMinute` it closes yesterday's position, computes features, and opens a new one if the signal fires.
- **Model:** logistic regression on 3 features — yesterday's USDCLP log-return (the "gap", also the trade filter at |ret| > 0.43%), copper 1-day return, copper 5-day return. Standardized, trained by gradient descent (verified ≡ sklearn), refit on the first decision of each quarter on an expanding window from 2020-01-01. No look-ahead: training rows require the target day to be fully realized.
- **Data:** with `UseBrokerFxData=true` (default), USDCLP training closes and gaps come from **Capitaria's own bar history** (13:40 closes rebuilt from M15/H1 bars at each quarterly refit), with the CSV filling dates before broker history begins. The EA also records its own decision-time price each day for the next day's gap — in tester and live. Copper features come from the CSV in the tester (no continuous copper series exists on MT5 — `Cobre_Sep26` history starts too late); live, it samples the real `Cobre_Sep26` quote at decision time.

## Known caveats

1. **Copper timing.** The backtest uses same-day copper closes. At 13:40 Santiago the COMEX settlement is available only part of the year (Chile/US DST offsets). Strictly lagging copper one day drops Sharpe from ~6.0 to ~3.9 and trades to ~22/yr. Live, the EA uses the real-time `Cobre_Sep26` quote — true performance should sit between the two. Judge the paper-trade log against the ~3.9 number, not the 6.0.
2. **`Cobre_Sep26` expires September 2026.** Update the `CopperSymbol` input on rollover (e.g., `Cobre_Dec26`). Around roll dates the 1d/5d copper returns can jump; if Capitaria's chart shows a gap that week, skip trusting the copper features for 5 days.
3. **Stale CSVs in live mode.** Quarterly refits use the CSVs, which end 2026-06-19. For live signal generation refresh them monthly: run `make pipeline` in the repo, then re-export (ask me — the export script is one command).
4. **Capitaria spread unknown.** The strategy needs < ~15 bps RT to look like the backtest and dies near 30–60 bps. The tester uses the spread recorded in Capitaria's history — check the average spread the tester reports before believing any result.
5. **Backtest realism.** Tester fills at broker quotes including their spread — this is the number that matters, and it may be well below the Python 6.0 if Capitaria's USDCLP spread is wide.

## Capitaria spread = 40 bps RT (measured 2026-07-10)

The default thresholds are NOT viable at 40 bps (net Sharpe 0.6, 3/6 years positive). Use:

- `ProbHi = 0.65`, `ProbLo = 0.35`, `GapMin = 0.010` (1%)

Backtest at these settings: ~24 trades/yr, 78% win, ~66 bps gross / ~26 bps net per trade, net Sharpe ≈ 4 (2021–2026). Thresholds were tuned on the same OOS window — discount accordingly. Alternative worth keeping open: execute manually at XTB (6.5 bps RT) where the original config gives Sharpe ≈ 6.
