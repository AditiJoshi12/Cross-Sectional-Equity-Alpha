# Utils

The reusable pieces of the pipeline. The notebooks import from here
(`from Utils.features import ...`), and `run_pipeline.py` strings the whole
thing together headlessly.

| module | what it does |
|--------|--------------|
| `alpha_data_loaders.py` | point-in-time index membership, FINRA short volume & short interest, SEC fundamentals (joined across CIK↔ticker) |
| `fred_features.py` | PIT FRED loading (`get_series_all_releases`, knowledge lag) and per-name rolling macro betas |
| `features.py` | price panel, cross-sectionally demeaned target, alt-data merge, per-date NA policy, coverage report |
| `walkforward.py` | embargoed walk-forward splits, leakage assertion on every fold, degenerate-model guard |
| `models.py` | Ridge / ElasticNet / GBRT factories, targets rescaled to bps before fitting |
| `tune_gbrt.py` | walk-forward grid search for the GBRT stage |
| `forecast_combination.py` | per-model OOS IC table and trailing-IC-weighted combination |
| `neutralize.py` | per-date residualisation of predictions against risk factors, exposure report |
| `portfolio.py` | decile/rank book construction, half-spread + √-impact costs, backtest stats, one-step paper trade |
| `run_pipeline.py` | end-to-end run: `python -m Utils.run_pipeline` from the repo root |
