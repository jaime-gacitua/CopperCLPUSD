# Add Experiment

Add a new versioned experiment to the copper→CLP/USD signal registry, run it, and compare it against all previous results.

## What this skill does

1. Determines the next version name (e.g. `v3_...`) by inspecting the existing registry in `copper_clp/experiment.py`
2. Designs the new experiment config based on what the user described — which new feature groups to add, any model or walk-forward parameter changes
3. Adds feature engineering code to `copper_clp/features.py` if new raw signals are needed
4. Updates `copper_clp/dataset.py` if new tickers need to be downloaded
5. Appends the new experiment entry to the `REGISTRY` dict in `copper_clp/experiment.py`
6. Rebuilds the data pipeline if needed: `uv run python -m copper_clp.dataset` and `uv run python -m copper_clp.features`
7. Runs the experiment: `uv run python -m copper_clp.experiment run <name>`
8. Prints the comparison table: `uv run python -m copper_clp.experiment compare`

## Rules to follow

**Never break existing experiments:**
- Only append to `REGISTRY` — never edit existing entries
- Never change the feature engineering logic for existing feature groups (only add new groups)
- Existing feature column names must stay stable

**Lag policy (critical — no look-ahead bias):**
- All signals must be shifted by at least 1 day before use as features
- Copper features use `_lag(series)` which applies `MIN_COPPER_LAG = 1`
- All other signals use `.shift(1)` directly
- Targets (`y_ret`, `y_dir`, `y_5d`, `y_21d`) are never shifted — they represent what happens on day T

**Feature group naming convention:**
- Add new prefix entries to `GROUP_PREFIXES` in `experiment.py` for any new group
- Feature column names: `<signal>_<type>_lag<N>` or `<signal>_mom<W>` etc.
- New group names go in `feature_groups` list of the registry entry

**Experiment naming convention:**
- `v<N>_<short_description>` where N increments from the highest existing version
- Description should name the key new signals added

**Walk-forward defaults (only change if the user explicitly asks):**
- `wf_train_days`: 756 (3 years)
- `wf_test_days`: 252 (1 year)  
- `wf_step_days`: 63 (quarterly)
- `data_granularity`: "daily"

## Key files

- `copper_clp/experiment.py` — registry and walk-forward engine (append REGISTRY entries here)
- `copper_clp/features.py` — feature engineering (add new feature blocks inside `build_features()`)
- `copper_clp/dataset.py` — data download (add tickers to `EXTRA_TICKERS` if needed)
- `copper_clp/config.py` — paths and constants
- `models/experiments/` — where per-run JSON snapshots are saved

## Existing feature groups (already in v2, do not re-add)

- `copper` — cu_ret_lagN, cu_momN, cu_volN, cu_rsi14, cu_zscoreN, cu_slope21, cu_ret5d, cu_ret21d
- `clp_ar` — clp_ret_lagN (own lags of CLP return)
- `calendar` — dow_N (day-of-week dummies)
- `dxy` — dxy_ret_lag1, dxy_momN, dxy_volN
- `vix` — vix_chg_lag1, vix_level_lag1, vix_mom5, vix_zscore63
- `em_fx` — brl_ret_lag1, pen_ret_lag1, mxn_ret_lag1, em_composite_lag1, clp_vs_em_lag1, brl_mom5, pen_mom5, mxn_mom5
- `oil` — oil_ret_lag1, oil_mom21
- `gold` — gold_ret_lag1, gold_vs_copper_lag1
- `us10y` — us10y_chg_lag1, us10y_mom21, us10y_level_lag1
- `ipsa` — ipsa_ret_lag1, ipsa_mom5
- `cross_asset` — cu_dxy_spread_lag1, cu_vix_interact_lag1

## Candidate signals not yet tried (suggestions to offer the user)

- Copper term structure: HG front-month vs next-quarter spread (contango/backwardation)
- China PMI / industrial production proxies (FXI ETF as a liquid proxy for China demand)
- BCCh / FOMC meeting calendar dummies
- CLP realized volatility regime (rolling 21d vol z-score as a regime switch)
- Momentum interaction: copper momentum × VIX regime
- Longer AR lags on CLP (10d, 21d)
- Month-end / quarter-end calendar effects

## Arguments

The user's message after `/add-experiment` describes the experiment. Extract:
- What new signals to add (or ask if unclear)
- Any model/hyperparameter changes (default: keep same)
- Any walk-forward parameter changes (default: keep same)

If the user's description is vague (e.g. "try something new"), propose 2-3 concrete options from the candidate list above and ask which to pursue before implementing.
