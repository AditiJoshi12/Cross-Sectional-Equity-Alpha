# Notebooks

Run top to bottom. The old single-file `alpha_backtest.ipynb` has been split into
four stage notebooks that hand off to each other through `Data/interim/`, so you
can re-run any stage without repeating the slow ones above it.

1. **`build_dataset.ipynb`** — downloads and caches the slow, rate-limited alt data
   (FINRA short volume/interest, SEC fundamentals) to `Data/` as parquet. Run-once.

2. **`build_model_df.ipynb`** — builds the point-in-time modelling panel: PIT S&P 500
   universe + CIK keys, live price download and membership filter, equity features
   (momentum, vol, Yang-Zhang, MA-spread, Hurst, ADV), FRED macro + ETF regime,
   short/fundamental merges with strict lag discipline, market-factor residualization,
   winsorise + z-score. Needs `FRED_API_KEY` and the parquet from step 1.
   → writes `Data/interim/model_df.parquet` + `pipeline_meta.json`.

3. **`walkforward_models.ipynb`** — train/val/test split with a target-overlap embargo,
   per-feature IC audit, the five-model ladder (OLS/Ridge/Lasso/ElasticNet/RF), and the
   embargoed multi-model walk-forward. Traded signal = pre-registered walk-forward
   ElasticNet; also the rolling-IC and coefficient-stability charts.
   → writes `Data/interim/pred_df.parquet` + `best_params.json`.

4. **`portfolio_sizing.ipynb`** — turns the signal into books: Level 2 (constant gross
   notional), Level 4 (constant-vol targeting), Level 5 (mean-variance CVXPY with a
   σ/√ADV liquidity term and turnover cap), plus the liquidity-aware cost sweep.
   Needs `cvxpy`. → writes `Data/interim/portfolio_returns.parquet`.

5. **`summary_and_plots.ipynb`** — the reporting layer: final performance table,
   Sharpe-by-year, the three-panel equity-curve figure, paper-trading signal generation,
   and the headline net Sharpe. Saves figures to `Results/`.

All five `chdir` to the repo root on startup, so `Utils/` imports and `Data/` paths
resolve no matter where Jupyter was launched from.

> `alpha_backtest.ipynb` is the original monolith that notebooks 2–5 were carved out of.
> It's kept for reference (its committed outputs back the numbers in the top-level README)
> but is superseded by the split — you don't need to run it.
