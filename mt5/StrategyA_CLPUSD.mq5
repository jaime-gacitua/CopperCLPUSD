//+------------------------------------------------------------------+
//| StrategyA_CLPUSD.mq5                                             |
//| CLP/USD Strategy A — close-to-close, copper-informed gap model   |
//|                                                                  |
//| Logic (from reports/conclusions.md, verified 2026-07-10):        |
//|   Daily at TradeHour:TradeMinute (server time = ~13:40 Santiago) |
//|   1. Close any open position (held exactly ~24h)                 |
//|   2. Features: clp_ret  = log(mid_now / usdclp_close[D-1])       |
//|                cu_ret1  = log(cu_now / cu_close[D-1])            |
//|                cu_ret5  = log(cu_now / cu_close[D-5])            |
//|   3. Logistic regression P(USD/CLP up tomorrow), refit quarterly |
//|      on expanding window from TrainStart (post-2020 regime only) |
//|   4. Trade only if |clp_ret| > GapMin (0.43%):                   |
//|        prob > ProbHi (0.60) -> BUY  USDCLP                       |
//|        prob < ProbLo (0.40) -> SELL USDCLP                       |
//|      else stay flat.                                             |
//|                                                                  |
//| SAFETY: trades ONLY in the Strategy Tester unless BOTH           |
//| AllowLiveTrading=true AND IUnderstandRealMoneyRisk=true.         |
//| On a REAL account it additionally refuses unless both are set.   |
//+------------------------------------------------------------------+
#property copyright "Jaime Gacitua"
#property version   "1.00"
#property tester_file "usdclp_daily.csv"
#property tester_file "copper_daily.csv"

#include <Trade\Trade.mqh>

//--- inputs -----------------------------------------------------------------
input int      TradeHour                 = 13;      // Decision hour (SERVER time; match 13:40 Santiago)
input int      TradeMinute               = 40;      // Decision minute (SERVER time)
input double   Lots                      = 0.10;    // Fixed lot size
input double   ProbHi                    = 0.60;    // Long threshold  P(up)
input double   ProbLo                    = 0.40;    // Short threshold P(up)
input double   GapMin                    = 0.0043;  // Min |yesterday return| (0.43%)
input datetime TrainStart                = D'2020.01.01'; // Post-2020 regime only
input int      MinTrainRows              = 100;     // Min training rows before trading
input string   CopperSymbol              = "Cobre_Sep26"; // Capitaria copper future — UPDATE ON ROLLOVER (Dec26...); blank = CSV only
input bool     UseBrokerFxData           = true;    // Build USDCLP closes from broker history (CSV fills gaps/earlier dates)
input double   EmergencyStopPct          = 3.0;     // Emergency SL, % of entry price (0 = none)
input long     MagicNumber               = 20260710;
input bool     AllowLiveTrading          = false;   // Must be true to trade outside tester
input bool     IUnderstandRealMoneyRisk  = false;   // Must ALSO be true on a REAL account

//--- data -------------------------------------------------------------------
#define MAXROWS 6000
datetime fxDate[MAXROWS];  double fxClose[MAXROWS];  int fxN = 0;   // working series (CSV + broker merge)
datetime csvDate[MAXROWS]; double csvClose[MAXROWS]; int csvN = 0;  // CSV backup (pre-broker-history dates)
datetime cuDate[MAXROWS];  double cuClose[MAXROWS];  int cuN = 0;

//--- model ------------------------------------------------------------------
double W[3];               // weights (standardized space)
double B;                  // intercept
double MU[3], SD[3];       // feature standardization
bool   modelReady   = false;
int    lastFitYear = 0, lastFitQuarter = 0;
datetime lastDecisionDay = 0;

CTrade trade;

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);

   if(!LoadCsv("usdclp_daily.csv", fxDate, fxClose, fxN))
      { Print("FATAL: cannot read usdclp_daily.csv from MQL5/Files"); return INIT_FAILED; }
   // keep a CSV backup: broker rebuilds overwrite the working arrays
   ArrayCopy(csvDate, fxDate); ArrayCopy(csvClose, fxClose); csvN = fxN;
   if(!LoadCsv("copper_daily.csv", cuDate, cuClose, cuN))
      { Print("FATAL: cannot read copper_daily.csv from MQL5/Files"); return INIT_FAILED; }
   PrintFormat("Loaded USDCLP rows=%d (%s -> %s)  Copper rows=%d",
               fxN, TimeToString(fxDate[0],TIME_DATE), TimeToString(fxDate[fxN-1],TIME_DATE), cuN);

   if(CopperSymbol != "")
      SymbolSelect(CopperSymbol, true);   // ensure copper future is in Market Watch

   if(!TradingAllowed())
      Print("SAFETY LOCK ACTIVE: EA will compute signals but will NOT trade. ",
            "Running outside tester requires AllowLiveTrading=true",
            (AccountInfoInteger(ACCOUNT_TRADE_MODE)==ACCOUNT_TRADE_MODE_REAL ?
             " AND IUnderstandRealMoneyRisk=true (REAL account detected)." : "."));
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
bool TradingAllowed()
{
   if(MQLInfoInteger(MQL_TESTER)) return true;                 // Strategy Tester: always OK
   if(!AllowLiveTrading)          return false;
   if(AccountInfoInteger(ACCOUNT_TRADE_MODE)==ACCOUNT_TRADE_MODE_REAL
      && !IUnderstandRealMoneyRisk) return false;              // real account: double opt-in
   return true;
}

//+------------------------------------------------------------------+
bool LoadCsv(const string fname, datetime &d[], double &v[], int &n)
{
   int h = FileOpen(fname, FILE_READ|FILE_CSV|FILE_ANSI, ',');
   if(h == INVALID_HANDLE) return false;
   n = 0;
   while(!FileIsEnding(h) && n < MAXROWS)
   {
      string ds = FileReadString(h);
      if(StringLen(ds) < 8) break;
      double val = FileReadNumber(h);
      d[n] = StringToTime(ds);            // "yyyy.mm.dd" -> midnight
      v[n] = val;
      n++;
   }
   FileClose(h);
   return (n > 10);
}

// index of last row with date strictly BEFORE day (binary search)
int LastIdxBefore(const datetime &d[], int n, datetime day)
{
   int lo = 0, hi = n - 1, ans = -1;
   while(lo <= hi)
   {
      int m = (lo + hi) / 2;
      if(d[m] < day) { ans = m; lo = m + 1; } else hi = m - 1;
   }
   return ans;
}

//+------------------------------------------------------------------+
//| Logistic regression: gradient descent on standardized features   |
//+------------------------------------------------------------------+
double Sigmoid(double z) { return 1.0 / (1.0 + MathExp(-z)); }

//+------------------------------------------------------------------+
//| Rebuild daily USDCLP series from broker bars (13:40 closes),     |
//| prepending CSV rows for dates before broker history begins.      |
//| Called at each quarterly refit so it never sees past `today`.    |
//+------------------------------------------------------------------+
bool RebuildFxFromBroker(datetime today)
{
   MqlRates r[];
   datetime from = TrainStart - 40*86400;
   ENUM_TIMEFRAMES tf = PERIOD_M15;
   int n = CopyRates(_Symbol, tf, from, today, r);
   if(n < 1000) { tf = PERIOD_H1; n = CopyRates(_Symbol, tf, from, today, r); }
   if(n < 300) return false;

   // collapse to one close per day: last bar with open time <= TradeHour:TradeMinute
   int deadline = TradeHour*3600 + TradeMinute*60;
   static datetime bd[MAXROWS]; static double bc[MAXROWS]; int bm = 0;
   datetime curDay = 0; double lastC = 0; bool have = false;
   for(int i = 0; i < n; i++)
   {
      datetime day = r[i].time - (r[i].time % 86400);
      if(day != curDay)
      {
         if(have && bm < MAXROWS) { bd[bm] = curDay; bc[bm] = lastC; bm++; }
         curDay = day; have = false;
      }
      if((int)(r[i].time % 86400) <= deadline) { lastC = r[i].close; have = true; }
   }
   if(have && bm < MAXROWS) { bd[bm] = curDay; bc[bm] = lastC; bm++; }
   if(bm < 100) return false;

   // merge: CSV rows strictly before broker coverage, then broker rows
   int m = 0;
   for(int i = 0; i < csvN && m < MAXROWS; i++)
      if(csvDate[i] < bd[0]) { fxDate[m] = csvDate[i]; fxClose[m] = csvClose[i]; m++; }
   int csvRows = m;
   for(int i = 0; i < bm && m < MAXROWS; i++)
      { fxDate[m] = bd[i]; fxClose[m] = bc[i]; m++; }
   fxN = m;
   PrintFormat("FX series rebuilt from broker %s bars: %d broker days (from %s) + %d CSV days",
               EnumToString(tf), bm, TimeToString(bd[0], TIME_DATE), csvRows);
   return true;
}

bool FitModel(datetime today)
{
   if(UseBrokerFxData && !RebuildFxFromBroker(today))
      Print("Broker FX history unavailable/short — using CSV series");
   // Build training rows: decision at close of fx day i, target = day i+1 direction.
   // Row usable if fxDate[i+1] < today (target fully realized before today).
   double X[][3]; double Y[];
   int cap = fxN; ArrayResize(X, cap); ArrayResize(Y, cap);
   int m = 0;
   for(int i = 1; i < fxN - 1; i++)
   {
      if(fxDate[i] < TrainStart) continue;
      if(fxDate[i+1] >= today)   break;
      int c1 = LastIdxBefore(cuDate, cuN, fxDate[i] + 86400);  // copper close ON or before day i
      if(c1 < 5) continue;
      double clpret = MathLog(fxClose[i] / fxClose[i-1]);
      if(clpret >  0.03) clpret =  0.03;                       // winsorize bad ticks
      if(clpret < -0.03) clpret = -0.03;
      X[m][0] = clpret;
      X[m][1] = MathLog(cuClose[c1] / cuClose[c1-1]);
      X[m][2] = MathLog(cuClose[c1] / cuClose[c1-5]);
      Y[m]    = (MathLog(fxClose[i+1] / fxClose[i]) > 0) ? 1.0 : 0.0;
      m++;
   }
   if(m < MinTrainRows) { PrintFormat("FitModel: only %d rows (<%d) — not fitting", m, MinTrainRows); return false; }

   // standardize
   for(int j = 0; j < 3; j++)
   {
      double s = 0, s2 = 0;
      for(int i = 0; i < m; i++) { s += X[i][j]; s2 += X[i][j]*X[i][j]; }
      MU[j] = s / m;
      SD[j] = MathSqrt(MathMax(s2/m - MU[j]*MU[j], 1e-12));
      for(int i = 0; i < m; i++) X[i][j] = (X[i][j] - MU[j]) / SD[j];
   }
   // gradient descent, L2 (sklearn C=1.0 => lambda = 1/(C) per-sum, i.e. w/m in mean-grad)
   double w0=0, w1=0, w2=0, b=0, lr=0.5;
   for(int ep = 0; ep < 500; ep++)
   {
      double g0=0, g1=0, g2=0, gb=0;
      for(int i = 0; i < m; i++)
      {
         double e = Sigmoid(w0*X[i][0] + w1*X[i][1] + w2*X[i][2] + b) - Y[i];
         g0 += e*X[i][0]; g1 += e*X[i][1]; g2 += e*X[i][2]; gb += e;
      }
      w0 -= lr*(g0/m + w0/m);  w1 -= lr*(g1/m + w1/m);
      w2 -= lr*(g2/m + w2/m);  b  -= lr*(gb/m);
   }
   W[0]=w0; W[1]=w1; W[2]=w2; B=b;
   modelReady = true;
   PrintFormat("Model refit %s: n=%d  w=[%.4f, %.4f, %.4f]  b=%.4f",
               TimeToString(today, TIME_DATE), m, W[0], W[1], W[2], B);
   return true;
}

//+------------------------------------------------------------------+
double Predict(double clpret, double cu1, double cu5)
{
   if(clpret >  0.03) clpret =  0.03;
   if(clpret < -0.03) clpret = -0.03;
   double z = B + W[0]*(clpret-MU[0])/SD[0] + W[1]*(cu1-MU[1])/SD[1] + W[2]*(cu5-MU[2])/SD[2];
   return Sigmoid(z);
}

//+------------------------------------------------------------------+
void ClosePositionIfAny()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)     continue;
      trade.PositionClose(tk);
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   MqlDateTime dt;
   datetime now = TimeCurrent();
   TimeToStruct(now, dt);

   // decision window: [TradeMinute, TradeMinute+5) once per day
   if(dt.hour != TradeHour || dt.min < TradeMinute || dt.min >= TradeMinute + 5) return;
   datetime today = now - (now % 86400);
   if(today == lastDecisionDay) return;
   lastDecisionDay = today;

   // quarterly refit (expanding window, no look-ahead)
   int q = (dt.mon - 1) / 3 + 1;
   if(!modelReady || dt.year != lastFitYear || q != lastFitQuarter)
   {
      if(!FitModel(today)) return;
      lastFitYear = dt.year; lastFitQuarter = q;
   }

   bool allowed = TradingAllowed();

   // 1. exit yesterday's position (24h hold complete)
   if(allowed) ClosePositionIfAny();

   // 2. features
   int fi = LastIdxBefore(fxDate, fxN, today);
   if(fi < 1) return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double mid = (bid + ask) / 2.0;
   if(mid <= 0) return;
   bool isTester = (bool)MQLInfoInteger(MQL_TESTER);

   // yesterday's USDCLP close at decision time: our own recorded 13:40 price
   // (works in tester and live, survives restarts); series fallback for day 1
   double prevClose = fxClose[fi];
   string gvKey = "StratA_close_" + _Symbol, gvDayKey = gvKey + "_day";
   if(GlobalVariableCheck(gvKey) && GlobalVariableCheck(gvDayKey))
   {
      datetime storedDay = (datetime)GlobalVariableGet(gvDayKey);
      if(storedDay > today - 5*86400 && storedDay < today)   // recent prior trading day
         prevClose = GlobalVariableGet(gvKey);
   }
   double clpret = MathLog(mid / prevClose);

   // copper features
   double cu1 = 0, cu5 = 0; bool cuOK = false;
   if(!isTester && CopperSymbol != "")
   {
      double cb = SymbolInfoDouble(CopperSymbol, SYMBOL_BID);
      double ca = SymbolInfoDouble(CopperSymbol, SYMBOL_ASK);
      MqlRates rt[];
      if(cb > 0 && ca > 0 && CopyRates(CopperSymbol, PERIOD_D1, 0, 7, rt) >= 6)
      {  // rt[] as series? CopyRates returns oldest-first; last element = current bar
         int n = ArraySize(rt);
         double cuNow = (cb + ca)/2.0;
         cu1 = MathLog(cuNow / rt[n-2].close);            // yesterday's D1 close
         cu5 = MathLog(cuNow / rt[n-6].close);            // 5 trading days back
         cuOK = true;
      }
   }
   if(!cuOK)   // tester, or live fallback: CSV settlements
   {
      int ci = LastIdxBefore(cuDate, cuN, today + 86400); // copper close on or before today
      if(ci < 5) return;
      int cip = (cuDate[ci] < today) ? ci : ci - 1;       // prior copper day
      if(cip < 5) return;
      cu1 = MathLog(cuClose[ci] / cuClose[cip]);
      cu5 = MathLog(cuClose[ci] / cuClose[cip - 4]);
   }
   // record today's 13:40 price for tomorrow's gap (tester and live)
   GlobalVariableSet(gvKey, mid);
   GlobalVariableSet(gvDayKey, (double)today);

   double prob = Predict(clpret, cu1, cu5);

   // 3. decide
   string dir = "FLAT";
   if(MathAbs(clpret) > GapMin && prob > ProbHi) dir = "LONG";
   if(MathAbs(clpret) > GapMin && prob < ProbLo) dir = "SHORT";
   PrintFormat("%s  gap=%.2f bps  cu1=%.2f bps  cu5=%.2f bps  P(up)=%.3f  -> %s%s",
               TimeToString(today, TIME_DATE), clpret*1e4, cu1*1e4, cu5*1e4, prob, dir,
               allowed ? "" : "  [SAFETY LOCK — not executed]");
   if(!allowed || dir == "FLAT") return;

   // 4. execute
   double sl = 0;
   if(dir == "LONG")
   {
      if(EmergencyStopPct > 0) sl = NormalizeDouble(ask * (1.0 - EmergencyStopPct/100.0), _Digits);
      trade.Buy(Lots, _Symbol, 0.0, sl, 0.0, "StratA long");
   }
   else
   {
      if(EmergencyStopPct > 0) sl = NormalizeDouble(bid * (1.0 + EmergencyStopPct/100.0), _Digits);
      trade.Sell(Lots, _Symbol, 0.0, sl, 0.0, "StratA short");
   }
}
//+------------------------------------------------------------------+
