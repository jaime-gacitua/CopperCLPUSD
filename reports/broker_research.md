# CLP/USD Broker Research

**Date:** 2026-06-21  
**Purpose:** Identify viable execution venues for a daily close-to-close CLP/USD spot strategy  
**Strategy breakeven:** ~30 bps round-trip spread. Edge largely disappears above 60 bps RT.

---

## Executive Summary

**forex.com does not offer CLP/USD.** Neither do OANDA, Interactive Brokers, IG Group, or CMC Markets.

The only confirmed retail venues with a live USD/CLP instrument are **XTB Chile** and **Axi**, both as CFDs priced against the Santiago interbank (MCF) session. XTB has the tightest spread (~8–10 bps RT) and is CMF-licensed in Chile, but its trading hours may not cover the 17:30 Santiago close that the strategy requires — this needs direct confirmation. Pepperstone has the best API infrastructure but USD/CLP availability is unconfirmed. All Chilean domestic corredoras offer bank-style currency exchange only, with spreads of 0.6–2% (60–200× too wide).

**Recommended immediate actions:**
1. Call XTB Chile → confirm trading hours extend to 17:30 Santiago
2. Email Pepperstone → confirm USD/CLP CFD availability
3. Open a demo account at Axi or FxPro → verify live spread on USD/CLP

---

## Why CLP Is Structurally Difficult to Trade Offshore

Chile's Mercado Cambiario Formal (MCF) restricts physical CLP delivery outside Chile. All offshore CLP forward activity settles as a **Non-Deliverable Forward (NDF)** in USD, against the Banco Central de Chile's *Dólar Observado* fixing published daily at ~10:30am Santiago (Reuters `CLPOB=`, Bloomberg `PCRCDOOB`). Retail brokers that offer USD/CLP wrap this as a **CFD** — a contract for difference settled in USD. This means:

- You never hold actual pesos
- Settlement is against the interbank rate, not a physical exchange
- Trading hours are limited to the MCF liquidity window (roughly London–NY overlap)
- Swap/rollover is charged in USD based on the CLP/USD interest rate differential

There are no capital controls on CLP flows — Chilean residents can legally trade through any foreign-regulated broker.

---

## Retail Brokers: Availability Matrix

| Broker | USD/CLP | Instrument | Spread (RT est.) | API | Chilean residents | Notes |
|--------|---------|-----------|-----------------|-----|-----------------|-------|
| **XTB Chile** | **Yes** | CFD | **~6–7 bps RT** | No (closed Mar 2025) | Yes — CMF licensed | Best spread; hours TBC |
| **Axi** | **Yes** | CFD | Not published | cTrader Open API | Yes — verify | Confirm spread live |
| **FxPro** | Likely yes | CFD | ~32 bps | cTrader Open API | Yes — verify | Marginal at breakeven |
| **Pepperstone** | **No** | — | — | cTrader + FIX | Yes | Confirmed unavailable |
| forex.com | No | — | — | REST + FIX | Yes | Publishes CLP news; no instrument |
| OANDA | No | — | — | v20 REST | Yes (BVI entity) | 68 pairs, no CLP |
| Interactive Brokers | No | — | — | Best-in-class | Yes | Confirmed absent |
| Saxo Bank | NDF only (wholesale) | NDF / RFQ | Not disclosed | OpenAPI | **No — exited Chile Jul 2024** | Min USD 500K notional |
| IG Group | No | — | — | REST + Streaming | Yes | 97 pairs, no CLP |
| eToro | Yes | NDF perpetual | ~2,800 bps RT | Limited | Yes | Unusable — prohibitively wide |
| CME (CHP futures) | Yes | Futures | Exchange-priced | Via broker API | Yes | 50M CLP/contract (~USD 55K); institutional scale |

---

## Brokers in Detail

### XTB Chile — best available retail option

XTB received CMF authorization as *Agente de Valores* #216 on **February 11, 2025**, making it the only CMF-registered international retail FX broker in Chile. Chilean accounts opened from July 2025 onward fall under the local entity.

Source: [xtb.com/cl/forex/usd-clp](https://www.xtb.com/cl/forex/usd-clp)

| | |
|---|---|
| Instrument | USD/CLP CFD, tracks interbank rate ("cotizaciones del dólar americano al peso chileno en el mercado interbancario") |
| Minimum spread | **0.30 CLP on ~930 spot ≈ 3.2 bps/side, ~6–7 bps RT** |
| Trading hours | **12:30–17:45 CET** (winter) / **14:30–19:45 CEST** (summer) |
| Leverage | **1:500** |
| Margin requirement | **0.2%** |
| Commission | None |
| Minimum deposit | None |
| Platform | xStation (proprietary, desktop + mobile) |
| API | Public REST/FIX closed March 2025. xStation has no public API. EA scripting via MT4/MT5 may be available — confirm with XTB. |
| Swap rates | Not published on website — verify inside the platform |
| CMF license | Yes — Agente de Valores #216 |

**Spread note:** The Chilean entity page (xtb.com/cl) shows a minimum spread of **0.30**, tighter than the international entity (0.39–0.44). At 930 CLP/USD, 0.30 CLP = ~3.2 bps/side = **~6.5 bps RT** — well inside the 30 bps breakeven. This is the best retail spread found for this pair.

**Critical issue — trading hours:** CET 12:30–17:45 translates to:
- Santiago winter (May–Aug, UTC-4): **08:30–13:45 local**
- Santiago summer (Nov–Feb, UTC-3): **09:30–14:45 local**

The strategy exits at the MCF close (~17:30 Santiago). XTB's window closes 3–4 hours before that. If this is accurate, XTB cannot support close-to-close execution despite the best spread available. **Call XTB Chile to confirm before opening an account.** It's possible the "17:45 CET" window refers to the London close and the Chilean entity operates extended hours — clarify this specifically.

---

### Axi (AxiTrader)

| | |
|---|---|
| Instrument | USD/CLP CFD |
| Contract size | USD 100,000 per standard lot; minimum 0.01 lots |
| Minimum spread | Dynamic — not publicly quoted for this pair |
| Margin | 10% for exotic pairs |
| Platform | cTrader |
| API | **cTrader Open API** — Python-compatible, event-driven, well-documented |
| Swap | Wednesday triple rollover; rates not published |
| Regulation | ASIC, FCA, CySEC |
| Chilean eligibility | Verify directly |

**Practical next step:** Open a demo account and check the live USD/CLP spread during the 17:00–18:00 Santiago window. If spread is below 15 bps/side (~30 bps RT) during that window, Axi becomes the preferred option given its API.

---

### FxPro

| | |
|---|---|
| Instrument | USD/CLP CFD |
| Spread | ~15 pips ≈ **~16 bps/side (~32 bps RT)** during MCF active hours |
| Platform | cTrader, MT4, MT5 |
| API | **cTrader Open API** — Python SDK available |
| Regulation | FCA, CySEC, FSCA |
| Chilean eligibility | Chile affiliate site exists; verify account opening |

At ~32 bps RT FxPro is right at the strategy's breakeven. Viable only with the gap-filtered rule (which trades fewer but larger-gap days, raising the average gross return per trade above the cost). Not suitable for trading every qualifying day at threshold = cost.

---

### Pepperstone — confirmed unavailable

Pepperstone does not offer USD/CLP. Confirmed by the user. Despite having the best API infrastructure (cTrader Automate for all retail accounts, FIX protocol for institutional), the pair is simply not in their instrument set.

---

### forex.com (GAIN Capital / StoneX)

**Does not offer USD/CLP.** forex.com lists ~84 currency pairs including USD/MXN but no CLP pair. The only CLP reference on their site is a 2019 news article ("USDCLP Approaching Highs after Protests") — editorial content, not a tradeable instrument.

Structural reason: forex.com's US entity is CFTC/NFA regulated and cannot offer CFDs (prohibited for US retail clients). Their non-US entity has not added CLP to its pair list. Chilean residents can open accounts but the instrument they need does not exist on the platform.

---

### Saxo Bank

**Not available to Chilean residents.** Saxo exited Chile on **July 1, 2024** as part of a broader withdrawal from 38 countries. Chilean clients were offboarded by end of 2024.

Even before the exit, Saxo's USD/CLP offering was institutional-only: NDF via phone RFQ, minimum USD 500,000 notional, no electronic execution, not accessible via their OpenAPI.

---

### OANDA

**Does not offer USD/CLP** in any tradeable form. Their ~68–77 pair roster covers majors, minors, and limited exotics (USD/HUF, TRY/JPY, GBP/ZAR) but not CLP. Chilean residents can open accounts via the BVI entity (OANDA Global Markets), but the instrument does not exist. The OANDA currency data API does publish USD/CLP reference rates — but that is a data product, not a trading account.

---

### eToro

Lists USD/CLP as a perpetual NDF CFD but with a **~2,800 bps round-trip spread** — approximately 100× the strategy's breakeven. Unusable. Both long and short overnight swaps are negative (double cost on held positions). Included here only for completeness.

---

## Chilean Domestic Brokers: Not Viable

All CMF-registered corredoras offer **bank-style currency exchange** — a service to buy or sell USD at the prevailing spot rate with a commission. This is not speculative FX trading: no leverage, no bid-ask spread quoting, no real-time execution, no API.

| Provider | Type | Spread / commission | API | Verdict |
|----------|------|-------------------|-----|---------|
| Bci Corredora | Bank FX exchange | ~0.6–0.8% per transaction | No | 60–80× too wide |
| Banchile Inversiones | Bank FX exchange | Up to 0.8% + VAT + min 0.12 UF | No | 60–80× too wide |
| LarrainVial | Bank FX exchange | Not disclosed | No | Not viable |
| Inversiones Security | OTC spot + forwards (via executive) | Not disclosed | No | No self-service |
| BICE Inversiones | Bank FX exchange | Not disclosed | No | Not viable |
| Tanner Corredores | Securities only | N/A — no FX product | No | Wrong product |
| Scotiabank Chile | Bank FX exchange (enterprise: spot + forwards) | Not disclosed | No | Enterprise only |
| **Capitaria** | CFD (local, MT5) | Not disclosed | No | **CMF unauthorized** as of mid-2026 |
| **XTB Chile** | CFD (CMF Agente de Valores #216) | ~8–10 bps RT | No public API | **Best local option** |

Retail fintech apps (Racional ~0.5%, Global66 ~2.0%, Mercado Pago ~1.9%) are even wider and not relevant.

---

## Institutional Venues (For Reference)

Not accessible to a retail trader but included for completeness and in case the strategy scales.

| Venue | Instrument | Min size | Notes |
|-------|-----------|---------|-------|
| EBS (CME Group) | USD/CLP NDF | ~USD 1–5M | OFF-SEF only since Sep 2024 |
| LSEG / Refinitiv FXall | USD/CLP NDF | ~USD 1–5M | Cleared via LCH ForexClear; RFQ from 500+ LPs |
| Bloomberg FXGO | USD/CLP NDF | Set by dealer (~USD 1–5M) | Executable streaming since Dec 2016 |
| CME CHP futures | USD/CLP futures | 50M CLP/contract (~USD 55K) | Cash-settled vs BCCh Dólar Observado; API via broker |

---

## Swap / Carry Cost

The strategy holds positions ~24 hours. Overnight swap applies to any position held past the daily rollover (typically 17:00 NY).

| | Rate (est.) |
|---|---|
| BCCh policy rate | 4.50% (held June 2026) |
| Fed Funds | 3.50–3.75% (held June 2026) |
| Net differential (CLP − USD) | ~+0.75–1.0% |
| Long USD/CLP swap (borrowing CLP) | ~−1.5 to −1.875%/yr ≈ **−4 to −5 bps/day** |
| Short USD/CLP swap (lending CLP) | Near zero or slight positive |

For a 24-hour hold, the long USD/CLP swap adds ~4–5 bps of cost on top of the spread — effectively making the long side ~12–15 bps RT all-in at XTB. The short side is close to free. This is still well within the strategy's breakeven.

Wednesday rollover is triple-charged industry-wide (covers the weekend). Avoid holding through Wednesday close if the position can be closed before then or re-entered Thursday.

**Verify exact swap tables inside the live platform** before trading — broker markup above the pure rate differential is common and not publicly disclosed.

---

## Decision Framework

```
                    USD/CLP available?
                          │
              ┌───────────┴───────────┐
             Yes                      No
              │                       │
      Spread < 30 bps RT?       Stop (wrong broker)
              │
      ┌───────┴───────┐
     Yes               No
      │                │
  API available?    Still viable if
      │             gap rule used
  ┌───┴───┐         (higher avg gap)
 Yes      No
  │        │
Best:   Manual only:
Pepperstone  XTB Chile
Axi
FxPro
```

**Recommended contact order:**

1. **XTB Chile** — call Santiago office to confirm trading hours cover 17:30 local; ask about professional/algorithmic API access → best option if hours match (~6.5 bps RT, CMF licensed)
2. **Axi** — open demo account, check USD/CLP spread at 17:30 Santiago window; if below 15 bps/side (~30 bps RT) → viable with cTrader API
3. **FxPro** — fallback; ~32 bps RT is right at breakeven, viable only with gap-filtered rule

**Pepperstone: eliminated** — confirmed does not offer USD/CLP.

---

## Reference: BCCh Dólar Observado

The official CLP/USD reference rate used by all NDF/CFD settlement and the strategy's price source:

- Published daily by Banco Central de Chile at ~10:30am Santiago
- Defined as the weighted average of prior-day MCF spot transactions
- Reuters: `CLPOB=` · Bloomberg: `PCRCDOOB`
- REST API: [BCCh estadísticas web services](https://si3.bcentral.cl/estadisticas/Principal1/web_services/index.htm) — free, no auth required, JSON output
- This is what CME CHP futures and all institutional NDF contracts settle against
