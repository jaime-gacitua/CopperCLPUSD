# Costos de transar USD/CLP: XTB vs Capitaria

*Análisis: 2026-07-10. Spot USDCLP ≈ 924. TPM Chile 4.5%, fed funds 3.63%.*

## Resumen

Spread casi idéntico entre ambos (~0.4 CLP ≈ 4 bps ida y vuelta). El diferenciador real es la plataforma (Capitaria = MT5, compatible con los EAs) vs regulación (XTB más sólida). El swap overnight —no publicado por ninguno— domina el costo en holdings de varios días.

## Comparación

| | XTB | Capitaria |
|---|---|---|
| Spread USD/CLP (estándar) | 0.39 CLP ≈ 39 pips (pip = 0.01) | 0.4 CLP (cifra de su blog, oct 2024) |
| Comisión | 0 (solo spread) | 0 (solo spread) |
| Margen / apalancamiento | 0.20% / 1:500 | 1% / 1:100 |
| Plataforma | xStation (no MT5) | **MetaTrader 5** |
| Horario | 12:30–17:45 CET (~08:30–13:45 CL) | no confirmado |
| Conversión P&L | 0.5% del \|P&L\| al cierre (P&L queda en CLP) | no publicado |
| Regulación | KNF/FCA (grupo listado); entidad int. en Belice (FSC) | Capitaria Latam SpA (Ley Fintec 21.521); CFDs sin regulación local según sus propios términos |
| Swap | No publicado; miércoles triple | No publicado; miércoles triple |

Notas del PDF de tarifas XTB (29-jun-2026): no lista spreads por par; forex CFD no paga comisión fija, solo spread + 0.5% de conversión sobre el resultado.

## Costo estimado por trade (hold 5 días = 7 cargos de swap)

Supuestos: carry = TPM 4.5% − fed funds 3.63% = 0.87%/año contra el largo; markup broker 1–3%/año (rango típico exóticos, **no verificado**).

**1 lote (USD 100,000):**

| Componente | Largo | Corto |
|---|---|---|
| Spread ida y vuelta | ~USD 42 | ~USD 42 |
| Swap 5d | −36 a −75 | −3 a −41 |
| Conversión XTB (0.5% del P&L) | ~USD 5 por 1% de movimiento | igual |
| **Total** | **~USD 90–120** | **~USD 50–90** |

**0.1 lote (USD 10,000):** todo ÷10 → total ~USD 9–12 largo, ~USD 5–9 corto. 1 pip ≈ USD 0.11.

En bps del nocional: ~5–12 bps por trade de 5 días.

## Implicancias para los EAs USDCLP

- El swap supera al spread en holds de varios días; el costo es asimétrico y favorece levemente los **cortos** (el carry juega contra el largo USDCLP).
- Para backtests, modelar fricción de swap ≈ −2.5%/año en largos y −1%/año en cortos, además del spread de 0.4 CLP.
- Con ganancia media de 1% por trade, el costo total consume ~9–12% de la ganancia esperada (trade largo 5 días).
- Contra el baseline del EA de percentiles (+14.4% OOS, ~1 trade), el costo es marginal; importa si se optimiza hacia más frecuencia.

## Pendientes

- [ ] Leer swap long/short real del símbolo USDCLP en demo MT5 de Capitaria (especificación del contrato) y reemplazar el rango estimado.
- [ ] Confirmar horario de trading USDCLP en Capitaria.
- [ ] Verificar si Capitaria cobra markup de conversión de P&L (cuenta en CLP evitaría esto).
- [ ] Comparar swaps vs IBKR (spread interbancario ~0.2–0.5 CLP + comisión 0.2 bp, mín USD 2).

## Fuentes

- [XTB USD/CLP](https://www.xtb.com/int/forex/usd-clp)
- PDF tarifas XTB Chile 29-jun-2026 (subido)
- [Capitaria — aspectos operativos](https://www.capitaria.com/aspectos-operativos/)
- [Capitaria — blog spread USDCLP](https://blog.capitaria.com/como-invertir-en-dolares-en-chile)
- [Fed funds jun-2026](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm) · [TPM jun-2026](https://portalinnova.cl/reportaje-banco-central-tpm-ipom-junio-2026/)
