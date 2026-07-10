PYTHON := uv run python

.PHONY: all data features train predict plots clean

all: data features train plots

## Download / refresh raw data and build the daily panel
data:
	$(PYTHON) -m copper_clp.dataset

## Build feature matrix from the daily panel
features:
	$(PYTHON) -m copper_clp.features

## Run walk-forward training and save results to models/
train:
	$(PYTHON) -m copper_clp.modeling.train

## Generate today's actionable signal
predict:
	$(PYTHON) -m copper_clp.modeling.predict

## Regenerate all figures
plots:
	$(PYTHON) -m copper_clp.plots

## Run lag-correlation and Granger notebooks
notebooks:
	$(PYTHON) notebooks/01_lag_correlation.py
	$(PYTHON) notebooks/02_granger_causality.py

## Full pipeline in order
pipeline: data features train predict plots

## Remove generated data and models (not raw data)
clean:
	rm -f data/processed/*.csv data/processed/*.json
	rm -f models/*.json
	rm -f reports/figures/*.png
