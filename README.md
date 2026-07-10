# CopperCLPUSD

Research project exploring whether copper prices carry a predictive signal for the CLP/USD exchange rate, and turning findings into tradeable strategies (backtests + MT5 expert advisors).

Layout follows the [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/) convention.

## Structure

```
├── copper_clp/          # Python package: data fetching, features, modeling
│   └── modeling/        # train / predict
├── data/                # NOT in git (see .gitignore)
│   ├── raw/             # immutable source data (yfinance, TwelveData, BCCh)
│   ├── interim/         # intermediate transformations
│   ├── processed/       # final feature matrices / panels
│   └── external/        # third-party data (incl. legacy_2015 Cochilco files)
├── models/              # experiment results & backtest outputs (JSON)
│   └── experiments/     # versioned experiment runs
├── mt5/                 # MetaTrader 5 expert advisors (.mq5) + docs
├── notebooks/           # numbered analysis scripts (01_… 12_…)
│   └── archive/         # original 2015 copper-vs-dollar exploration
├── references/          # papers, manuals, background material
├── reports/             # findings, research notes, conclusions
│   └── figures/         # generated charts
├── scripts/             # standalone backtest scripts
├── Makefile
└── pyproject.toml       # deps managed with uv
```

## Setup

```bash
uv sync
cp .env.example .env   # add TWELVE_DATA_API_KEY, ALPHA_VANTAGE_API_KEY
```

Data is not versioned; rebuild it with the fetch utilities in `copper_clp/dataset.py` / `copper_clp/twelvedata.py`.

## Status

See `reports/conclusions.md` and `reports/lessons_learned.md`. Key caveat: an earlier Strategy A result was invalidated by a yfinance close-dating artifact — validate any daily-bar signal against broker (MT5/TwelveData) timestamps before trusting it.
