//+------------------------------------------------------------------+
//| PercentileReversion_USDCLP.mq5                                   |
//| Percentile mean-reversion on USDCLP (Capitaria).                 |
//|                                                                  |
//| Jaime's framing is in CLPUSD; Capitaria quotes USDCLP = 1/CLPUSD:|
//|   "sell CLPUSD at its 90th pct" == BUY  USDCLP at its 10th pct   |
//|   "buy  CLPUSD at its 10th pct" == SELL USDCLP at its 90th pct   |
//| This EA works in USDCLP percentiles throughout.                  |
//|                                                                  |
//| Rules (confirmed 2026-07-10):                                    |
//|  - Bands = percentiles of daily closes over a FIXED training     |
//|    window (default 2021-01-01..2024-12-31). Static, not rolling. |
//|  - Flat + close >= SellPercentile level  -> open SHORT           |
//|  - Flat + close <= BuyPercentile  level  -> open LONG            |
//|  - Exit when close crosses back through ExitPercentile (median). |
//|  - One position at a time, no scale-in.                          |
//|  - Catastrophe stop: server-side SL at entry +/- StopPct%.       |
//|                                                                  |
//| Percentile method = numpy default (linear interpolation),        |
//| verified identical to the Python baseline.                       |
//|                                                                  |
//| SAFETY: on a live chart this EA only logs signals unless BOTH    |
//| AllowLiveTrading and IUnderstandRealMoneyRisk are set true.      |
//| In the Strategy Tester it always trades (simulated money).       |
//+------------------------------------------------------------------+
#property copyright "Jaime Gacitua"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Strategy inputs (percentiles are OPTIMIZABLE: e.g. Buy 2..30 step 2, Sell 70..98 step 2)
input double  InpBuyPercentile   = 10.0;         // Buy USDCLP at/below this percentile
input double  InpSellPercentile  = 90.0;         // Sell USDCLP at/above this percentile
input double  InpExitPercentile  = 50.0;         // Exit when close crosses this percentile
input datetime InpTrainStart     = D'2021.01.01';// Training window start
input datetime InpTrainEnd       = D'2024.12.31';// Training window end (inclusive)
input double  InpLots            = 0.10;         // Position size (lots)
input double  InpStopPct         = 8.0;          // Catastrophe stop, % from entry (0 = none)
//--- Data
input bool    InpUseBrokerData   = true;         // Use broker D1 history for percentiles
input string  InpCsvFile         = "usdclp_daily.csv"; // CSV fallback (MQL5\Files\)
//--- Safety
input bool    AllowLiveTrading         = false;  // Must be true to trade live
input bool    IUnderstandRealMoneyRisk = false;  // Must ALSO be true to trade live
input long    InpMagic           = 20260710;     // Magic number

CTrade   trade;
double   g_buyLevel  = 0;   // USDCLP level of BuyPercentile
double   g_sellLevel = 0;   // USDCLP level of SellPercentile
double   g_exitLevel = 0;   // USDCLP level of ExitPercentile
bool     g_bandsReady = false;
datetime g_lastBarTime = 0;

//+------------------------------------------------------------------+
bool IsTester() { return (bool)MQLInfoInteger(MQL_TESTER) || (bool)MQLInfoInteger(MQL_OPTIMIZATION); }

bool TradingAllowed()
  {
   if(IsTester()) return true;
   return AllowLiveTrading && IUnderstandRealMoneyRisk;
  }

//+------------------------------------------------------------------+
//| numpy-style linear-interpolation percentile of a sorted array    |
//+------------------------------------------------------------------+
double Percentile(const double &sorted[], const double q)
  {
   int n = ArraySize(sorted);
   if(n == 0) return 0.0;
   if(n == 1) return sorted[0];
   double rank = q / 100.0 * (n - 1);
   int lo = (int)MathFloor(rank);
   int hi = (int)MathCeil(rank);
   if(hi >= n) hi = n - 1;
   double frac = rank - lo;
   return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
  }

//+------------------------------------------------------------------+
//| Load daily closes in [InpTrainStart, InpTrainEnd] into closes[]. |
//| Broker D1 history first; CSV fills any dates before broker       |
//| history begins. Returns count.                                   |
//+------------------------------------------------------------------+
int LoadTrainingCloses(double &closes[])
  {
   datetime trainEndDay = InpTrainEnd + 86399; // make end inclusive
   double  vals[];  ArrayResize(vals, 0);
   long    days[];  ArrayResize(days, 0);      // day-key = time/86400, for dedupe
   datetime brokerFirst = 0;

   //--- 1) broker D1 bars
   if(InpUseBrokerData)
     {
      MqlRates rates[];
      int n = CopyRates(_Symbol, PERIOD_D1, InpTrainStart, trainEndDay, rates);
      if(n > 0)
        {
         brokerFirst = rates[0].time;
         for(int i = 0; i < n; i++)
           {
            int sz = ArraySize(vals);
            ArrayResize(vals, sz + 1); ArrayResize(days, sz + 1);
            vals[sz] = rates[i].close;
            days[sz] = (long)(rates[i].time / 86400);
           }
         PrintFormat("[Percentile] broker D1: %d closes, first %s", n, TimeToString(brokerFirst, TIME_DATE));
        }
      else
         Print("[Percentile] WARNING: no broker D1 history in training window; relying on CSV");
     }

   //--- 2) CSV for dates before broker history starts (or everything if no broker data)
   if(InpCsvFile != "")
     {
      int h = FileOpen(InpCsvFile, FILE_READ | FILE_CSV | FILE_ANSI, ',');
      if(h == INVALID_HANDLE)
         PrintFormat("[Percentile] WARNING: cannot open %s (err %d)", InpCsvFile, GetLastError());
      else
        {
         int added = 0;
         while(!FileIsEnding(h))
           {
            string sd = FileReadString(h);
            string sp = FileReadString(h);
            if(sd == "" ) continue;
            datetime t = StringToTime(sd);          // "2021.01.04" -> datetime
            if(t < InpTrainStart || t > trainEndDay) continue;
            if(brokerFirst > 0 && t >= brokerFirst) continue; // broker data wins
            double p = StringToDouble(sp);
            if(p <= 0) continue;
            int sz = ArraySize(vals);
            ArrayResize(vals, sz + 1); ArrayResize(days, sz + 1);
            vals[sz] = p; days[sz] = (long)(t / 86400);
            added++;
           }
         FileClose(h);
         PrintFormat("[Percentile] CSV: %d closes added before broker history", added);
        }
     }

   //--- 3) dedupe by day-key (keep first occurrence), then sort values
   int n = ArraySize(vals);
   ArrayResize(closes, 0);
   for(int i = 0; i < n; i++)
     {
      bool dup = false;
      for(int j = 0; j < i; j++) if(days[j] == days[i]) { dup = true; break; }
      if(dup) continue;
      int sz = ArraySize(closes);
      ArrayResize(closes, sz + 1);
      closes[sz] = vals[i];
     }
   ArraySort(closes);
   return ArraySize(closes);
  }

//+------------------------------------------------------------------+
bool ComputeBands()
  {
   double closes[];
   int n = LoadTrainingCloses(closes);
   if(n < 100)
     {
      PrintFormat("[Percentile] ERROR: only %d training closes (need >=100). Check history/CSV.", n);
      return false;
     }
   g_buyLevel  = Percentile(closes, InpBuyPercentile);
   g_exitLevel = Percentile(closes, InpExitPercentile);
   g_sellLevel = Percentile(closes, InpSellPercentile);
   PrintFormat("[Percentile] n=%d  BUY(p%.1f)=%.2f  EXIT(p%.1f)=%.2f  SELL(p%.1f)=%.2f",
               n, InpBuyPercentile, g_buyLevel, InpExitPercentile, g_exitLevel,
               InpSellPercentile, g_sellLevel);
   if(!(g_buyLevel < g_exitLevel && g_exitLevel < g_sellLevel))
     {
      Print("[Percentile] ERROR: band levels not ordered buy<exit<sell — check percentile inputs.");
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
bool HaveMyPosition(long &type)
  {
   if(!PositionSelect(_Symbol)) return false;
   if(PositionGetInteger(POSITION_MAGIC) != InpMagic) return false;
   type = PositionGetInteger(POSITION_TYPE);
   return true;
  }

double NormPrice(double p)
  {
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   return NormalizeDouble(p, digits);
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(100);
   if(!IsTester() && !TradingAllowed())
      Print("[Percentile] LIVE SAFETY LOCK ON — signals will be logged, no orders sent.");
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   // lazy band computation on first tick (history is reliably loaded by then)
   if(!g_bandsReady)
     {
      if(!ComputeBands()) { ExpertRemove(); return; }
      g_bandsReady = true;
     }

   // act once per new D1 bar
   datetime barTime = iTime(_Symbol, PERIOD_D1, 0);
   if(barTime == g_lastBarTime) return;
   g_lastBarTime = barTime;

   double prevClose = iClose(_Symbol, PERIOD_D1, 1);
   if(prevClose <= 0) return;

   long ptype;
   bool inPos = HaveMyPosition(ptype);

   string action = "hold";

   //--- exits
   if(inPos)
     {
      if(ptype == POSITION_TYPE_SELL && prevClose <= g_exitLevel)
        {
         action = "EXIT SHORT (close <= exit level)";
         if(TradingAllowed()) trade.PositionClose(_Symbol);
         inPos = false;
        }
      else if(ptype == POSITION_TYPE_BUY && prevClose >= g_exitLevel)
        {
         action = "EXIT LONG (close >= exit level)";
         if(TradingAllowed()) trade.PositionClose(_Symbol);
         inPos = false;
        }
     }
   //--- entries (only if flat; exit and entry can't both fire since exit level is inside the bands)
   else
     {
      if(prevClose >= g_sellLevel)
        {
         action = "OPEN SHORT (sell USDCLP = buy CLPUSD)";
         if(TradingAllowed())
           {
            double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double sl  = (InpStopPct > 0) ? NormPrice(bid * (1.0 + InpStopPct / 100.0)) : 0.0;
            trade.Sell(InpLots, _Symbol, 0.0, sl, 0.0, "pctl short");
           }
        }
      else if(prevClose <= g_buyLevel)
        {
         action = "OPEN LONG (buy USDCLP = sell CLPUSD)";
         if(TradingAllowed())
           {
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double sl  = (InpStopPct > 0) ? NormPrice(ask * (1.0 - InpStopPct / 100.0)) : 0.0;
            trade.Buy(InpLots, _Symbol, 0.0, sl, 0.0, "pctl long");
           }
        }
     }

   PrintFormat("[Percentile] %s close=%.2f  bands[%.2f / %.2f / %.2f]  pos=%s  -> %s",
               TimeToString(barTime, TIME_DATE), prevClose,
               g_buyLevel, g_exitLevel, g_sellLevel,
               inPos ? (ptype == POSITION_TYPE_SELL ? "SHORT" : "LONG") : "flat",
               action);
  }
//+------------------------------------------------------------------+
