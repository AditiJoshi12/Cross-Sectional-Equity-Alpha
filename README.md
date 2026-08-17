# Cross-Sectional Equity Alpha

*Cross-sectional equity alpha research on the S&P 500 — point-in-time data, embargoed walk-forward, cost-aware long/short backtest.*

---

## Overview

We predict each S&P 500 stock's **5-day forward return relative to its peers** (cross-sectionally demeaned) from price/volume, short-selling, fundamental, and macro-sensitivity features, then trade it as a dollar-neutral long/short book with realistic costs.

The mean-variance book delivers a **net-of-costs Sharpe of ~0.83** on the ~2.4-year walk-forward out-of-sample period (gross ~1.13; weekly rebalance; liquidity-scaled costs at 10 bps for the median-liquidity name). Read that number with its caveat, though: the traded signal's out-of-sample information coefficient is only **+0.010 (t ≈ 1.3, hit rate 53%)** — *not* statistically distinguishable from zero on this sample. Much of the Sharpe comes from portfolio construction and one strong year (2025), not from demonstrated forecast skill. The pipeline is built end-to-end with point-in-time data, embargoed walk-forward validation, and a degeneracy guard, and it reports IC alongside Sharpe precisely so that this distinction stays visible rather than getting laundered into a headline.

## The hypothesis

Relative returns inside a liquid index are predictable from a combination of well-known cross-sectional effects. No single feature dominates; the edge comes from combining weakly-correlated signals into a composite that survives realistic transaction costs.

| feature | idea | expected sign |
|---|---|---|
| 5d return rank | short-term reversal | − |
| 20/60/120/252d return ranks | momentum persists | + |
| 20d vol rank, Yang-Zhang vol | low-vol effect | − |
| MA(20)/MA(50) spread | overextension reverts | − |
| Hurst exponent (126d) | trending names keep trending | + |
| FINRA short-volume ratio / z / Δ, short interest | short pressure is informed | − |
| earnings yield, leverage (SEC) | quality premium, leverage penalised | +, − |
| rolling betas to oil, 10Y, HY spread, USD, VIX | factor exposure ranks the cross-section | context |

Each feature's standalone IC gets checked against its hypothesised sign before modelling. The target is demeaned per date on purpose: with a raw-return target, tree models just learn to time the market (constant prediction across all names on a day), which a dollar-neutral book can't monetise.

## Bias controls

The stuff that actually matters. Each of these bit us at some point during development.

- **Survivorship** — universe is rebuilt from a historical constituents-and-changes file, so it includes every name *ever* in the index; each date only sees that day's actual members (`merge_asof` backward). Caveat: Yahoo has no delisting returns and gaps for acquired tickers, so this mitigates rather than eliminates — the run prints a coverage diagnostic so the residual gap is measured, not assumed.
- **Fundamentals look-ahead** — SEC XBRL facts are aligned on the **filing date**, not fiscal period end. A number becomes visible only once it was public.
- **Macro look-ahead** — FRED series loaded via `get_series_all_releases()` (latest vintage with `realtime_start ≤` the trading day) plus a 1-business-day knowledge lag. And raw macro levels never enter the model — they're constant across names, so they carry zero cross-sectional information. Only per-name rolling **betas** are used.
- **Target leakage** — with a 5-day label, the last 5 training days overlap the test window. The walk-forward enforces a `label_horizon + embargo` gap and re-asserts the positional trading-day gap on every fold. A violation raises, it can't pass silently.
- **Missing-value look-ahead** — alt features are filled with the *per-date* cross-sectional median (never a global one, which peeks at future dates), plus an `_isna` flag. A coverage report catches dead data pipes.
- **Silent model collapse** — an over-penalised ElasticNet once shrank every coefficient to zero and reported "no signal, IC = NaN" for weeks. The harness now measures the fraction of constant-prediction days and raises above a tolerance. Targets are rescaled to bps before fitting so penalties operate at a sane scale.
- **Ticker renames** — FB→META, ANTM→ELV etc. break per-name histories and joins. Entities are keyed on `CIK:ticker` (CIK for rename stability, ticker suffix because GOOG/GOOGL share a CIK).
- **Costs** — half-spread on turnover plus square-root impact `∝ √(traded $ / ADV)`, charged only on the fraction traded. The MVO variant also prices `σ/√ADV` per name *inside* the objective and caps turnover per rebalance. Solver failures hold the previous book instead of dropping the day (failures cluster on stressed days; dropping them is selection bias).

## Repository layout

```
Utils/                        # reusable pipeline modules
├── alpha_data_loaders.py     # PIT membership, FINRA short vol/int, SEC fundamentals
├── fred_features.py          # PIT FRED + per-name macro betas
├── features.py               # xs signals, demeaned target, NA policy, coverage
├── walkforward.py            # embargoed WF, leakage assert, degeneracy guard
├── models.py                 # Ridge / ElasticNet / GBRT factories (bps rescale)
├── tune_gbrt.py              # walk-forward grid search
├── forecast_combination.py   # trailing-IC-weighted combination
├── neutralize.py             # per-date risk-factor residualisation
├── portfolio.py              # decile/rank books, sqrt-impact costs, paper trade
└── run_pipeline.py           # headless end-to-end run

Notebooks/
├── build_dataset.ipynb        # fetch + cache alt data (FINRA, SEC, FRED), QC
├── build_model_df.ipynb       # PIT panel + features -> model_df          [stage 1]
├── walkforward_models.ipynb   # embargoed walk-forward, 5-model ladder    [stage 2]
├── portfolio_sizing.ipynb     # L2/L4/L5 construction + costs             [stage 3]
├── summary_and_plots.ipynb    # performance table + figures               [stage 4]
└── alpha_backtest.ipynb       # original monolith — kept for reference, superseded

Data/                         # inputs & cached parquet (git-ignored, see its README)
Results/                      # figures from the committed runs, with commentary
```

## Installation

```bash
git clone https://github.com/AditiJoshi12/Cross-Sectional-Equity-Alpha.git
cd Cross-Sectional-Equity-Alpha
pip install -r requirements.txt

export FRED_API_KEY="..."   # free key from fred.stlouisfed.org — needed for macro features
```

## Usage

Run the notebooks in order. First the one-time data build:

1. **[build_dataset.ipynb](Notebooks/build_dataset.ipynb)** — downloads prices, FINRA short data, SEC fundamentals and FRED macro, runs the data-quality checks, and caches everything to `Data/` as parquet. The FINRA/SEC pulls are slow and rate-limited; you only do this once.

Then the four research stages, each reading the previous stage's parquet handoff from `Data/interim/` (see [Data/README.md](Data/README.md)):

2. **[build_model_df.ipynb](Notebooks/build_model_df.ipynb)** — assembles the point-in-time panel, builds the features and the demeaned 5-day target → `model_df.parquet`.
3. **[walkforward_models.ipynb](Notebooks/walkforward_models.ipynb)** — embargoed walk-forward over the 5-model ladder (OLS / Ridge / Lasso / ElasticNet / RandomForest); fixes the pre-registered walk-forward ElasticNet as the traded signal → `pred_df.parquet`.
4. **[portfolio_sizing.ipynb](Notebooks/portfolio_sizing.ipynb)** — constant-notional (L2), vol-targeted (L4) and mean-variance (L5) books plus liquidity-aware costs → `portfolio_returns.parquet`.
5. **[summary_and_plots.ipynb](Notebooks/summary_and_plots.ipynb)** — final performance table, Sharpe-by-year, equity curves, and the paper-trading signal for the latest date.

The original single-file **[alpha_backtest.ipynb](Notebooks/alpha_backtest.ipynb)** runs the same research end-to-end and is kept for reference, but the four-stage split above is the current path. There's also a headless run that drives the `Utils/` modules directly:

```bash
python -m Utils.run_pipeline
```

Or use the modules directly:

```python
from Utils.walkforward import run_walkforward
from Utils.models import make_model_factories
from Utils.portfolio import build_weights, backtest, print_pnl

oos, per_fold, pooled = run_walkforward(model_df, feature_cols,
                                        make_model_factories()["elasticnet"],
                                        train_min=378, test_size=21, step=21,
                                        label_horizon=5, embargo=2)
print(pooled["mean_IC"], pooled["spread_t_stat"])

w = build_weights(oos, pred_col="pred", scheme="decile", q=10)
daily, summ = backtest(w, half_spread_bps=1.0, impact_coef_bps=10.0, adv_col="adv20")
print_pnl("decile L/S, net of costs", summ)
```

## Results

Full gallery with commentary in **[Results/](Results/README.md)**. The headline numbers from the walk-forward out-of-sample run:

| metric | value |
|--------|-------|
| Gross Sharpe (mean-variance book) | ~1.13 |
| **Net Sharpe (after costs)** | **~0.83** |
| Net max drawdown | ~−7% |
| Net total return (~2.4 yr OOS) | ~20% |
| Traded-signal OOS IC (t-stat) | +0.010 (t ≈ 1.3 — *not significant*) |
| Rebalance frequency | weekly (5 trading days) |
| Cost model | σ/√ADV liquidity-scaled, 10 bps median name |

The construction ladder: gross Sharpe rises 0.75 (constant notional) → 0.97 (vol-targeted) → 1.13 (mean-variance), then costs take the MV book to 0.83 net. That climb is largely *construction* — vol targeting, position caps, turnover anchoring — not extra forecast skill; the near-zero IC is the tell. Net Sharpe is also cost-sensitive (0.98 at 5 bps, 0.83 at 10 bps, 0.53 at 20 bps) and concentrated in one year (net Sharpe-by-year: −0.07 in 2024, +1.54 in 2025, +0.78 in 2026-to-date).

### Capacity — non-linear market impact

The 10 bps headline is a *linear* cost. Real market impact is concave in trade size (a √-law), which the linear model under-charges — and it makes the strategy **size-dependent**. Pricing Almgren-style √-impact (`impact ∝ σ·√(traded$/ADV)`, Y=0.5) *on top of* the 10 bps spread shows the edge decaying as the book grows:

| Book (GMV) | √-impact / rebalance | Net Sharpe |
|---|---|---|
| $10M | 1.2 bp | 0.71 |
| $50M | 2.7 bp | 0.62 |
| **$100M** | **3.9 bp** | **0.56** |
| $250M | 6.1 bp | 0.44 |
| $500M | 8.7 bp | 0.30 |
| $1B | 12.3 bp | 0.10 |

At a realistic $100M book, non-linear impact costs ~0.2 of Sharpe (≈0.56–0.60 depending on the run), and the edge is essentially gone by ~$1B — a **small-capacity signal**, exactly what a near-zero IC predicts. These figures charge √-impact *ex-post* on the linear-cost book; the Level-6 optimiser prices it *inside* the objective, which starts to matter only at larger book sizes (at $100M it is within optimizer noise of the ex-post number).

![Equity curves](Results/equity_curves.png)

The pipeline reports IC and its t-stat alongside Sharpe so the signal strength is always visible — a positive Sharpe from construction alone (e.g. a low-vol tilt) would show up as near-zero IC, and the diagnostics are designed to surface that.

### Is it alpha, or factor beta?

Regressing the traded signal cross-sectionally on style factors settles it. The raw signal loads heavily on **long-horizon momentum (−0.62)** and **volatility (+0.43)** — it's largely a long-term-reversal-plus-vol bet, not idiosyncratic skill. Neutralising those factors out **collapses mean IC from +0.0095 to +0.0029** and takes the mean-variance book to **~0.33 net Sharpe (and negative once √-impact is priced)**. So the headline is mostly harvested factor premia; the residual idiosyncratic edge is negligible on this universe/period. `portfolio_sizing.ipynb` prints the exposure report every run and exposes a `NEUTRALIZE_SIGNAL` toggle to reproduce this. The **Level-6** optimiser (√-impact in the objective) lands within optimizer noise of the ex-post charge at a $100M book — the impact penalty there is only ~4 bp, too small to change the book — so its benefit is expected to show only at larger AUM.

## Roadmap

- [ ] CRSP-style PIT prices with delisting returns (the real survivorship fix)
- [ ] Vintage-dated macro (ALFRED) instead of a fixed knowledge lag
- [ ] Analyst revisions / options-implied features
- [x] Non-linear (√-impact) cost priced inside the optimiser — Level 6 (`portfolio_sizing.ipynb`)
- [x] Factor exposure report + optional style-neutralisation of the traded signal
- [ ] Factor-risk-*constrained* optimizer (neutrality as a hard constraint in the MVO, not just ex-post residualisation)

## License

MIT © 2026 Aditi Joshi
