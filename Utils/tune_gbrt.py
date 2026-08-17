"""
GBRT hyperparameter tuning on an early time-slice.

Tunes on the first ~40% of dates only, using the same embargoed walk-forward
folds as the main harness, scored by cross-sectional IC (not MSE — we trade
the rank, not the level). The later 60% stays untouched so the reported OOS
is out-of-sample with respect to both the fit and the hyperparameter choice.

Returns a leaderboard sorted by mean IC; freeze the best config into
gbrt_factory() and run the full walk-forward.
"""
from __future__ import annotations
from itertools import product
import numpy as np
import pandas as pd

from .walkforward import make_walkforward_folds, daily_ic


_XGB_BASE = dict(subsample=0.6, colsample_bytree=0.5, reg_alpha=0.1,
                 reg_lambda=1.0, objective="reg:squarederror", n_jobs=-1)

DEFAULT_GRID = {
    "max_depth":        [2, 3],
    "min_child_weight": [50, 100, 200],
    "n_estimators":     [300, 600],
    "learning_rate":    [0.02],          # fixed low; trade off vs n_estimators
}


def _make_xgb(params):
    """Fresh estimator; falls back to sklearn GradientBoosting if xgboost is missing."""
    try:
        from xgboost import XGBRegressor
        cfg = dict(_XGB_BASE)
        cfg.update(params)
        return XGBRegressor(**cfg)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.02),
            min_samples_leaf=params.get("min_child_weight", 50),
            subsample=_XGB_BASE["subsample"])


def _grid_dicts(grid):
    keys = list(grid)
    for combo in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def tune_gbrt(model_df, feature_cols, target_col="target",
              tune_frac=0.40, label_horizon=5, embargo=2,
              train_min=252, test_size=21, step=21,
              grid=None, verbose=True):
    """IC-scored grid search on the first tune_frac of dates.

    Returns (best_params, leaderboard_df). Inner folds use the same embargo
    as the main harness. On near-ties, prefer the simpler / more regularised
    config — a dozen combos is plenty when the signal is noisy.
    """
    grid = grid or DEFAULT_GRID
    df = model_df.sort_values(["date", "symbol"]).copy()
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))

    # early slice only — the later period stays untouched for the reported OOS
    cutoff = all_dates[int(len(all_dates) * tune_frac)]
    early = df[df["date"] <= cutoff]
    edates = pd.DatetimeIndex(sorted(early["date"].unique()))
    folds = make_walkforward_folds(edates, train_min, test_size, step,
                                   label_horizon, embargo, expanding=True)
    if not folds:
        raise ValueError(
            f"No inner folds on the early slice ({len(edates)} dates). "
            f"Lower train_min (now {train_min}) or raise tune_frac (now {tune_frac}).")

    if verbose:
        print(f"Tuning on EARLY slice: {edates.min().date()} -> {edates.max().date()} "
              f"({len(edates)} dates, {len(folds)} inner folds, "
              f"embargo gap = {label_horizon + embargo} days)")
        print(f"Configs: {sum(1 for _ in _grid_dicts(grid))}  "
              f"(scored by mean cross-sectional IC)\n")

    rows = []
    for params in _grid_dicts(grid):
        parts = []
        for fold in folds:
            tr = early[early["date"].isin(fold.train_dates)]
            te = early[early["date"].isin(fold.test_dates)]
            if tr.empty or te.empty:
                continue
            m = _make_xgb(params)
            m.fit(tr[feature_cols].values, tr[target_col].values)
            p = np.asarray(m.predict(te[feature_cols].values), dtype=float)
            part = te[["date", target_col]].copy()
            part["pred"] = p
            parts.append(part)

        if not parts:
            continue
        oos = pd.concat(parts, ignore_index=True)
        ic = daily_ic(oos, "pred", target_col)          # per-date IC series
        mean_ic = ic.mean() if len(ic) else np.nan
        ic_std = ic.std() if len(ic) else np.nan
        icir = (mean_ic / ic_std) if (ic_std and ic_std > 0) else np.nan
        rows.append({**params,
                     "mean_IC": mean_ic,
                     "IC_std": ic_std,
                     "ICIR": icir,
                     "hit_rate": (ic > 0).mean() if len(ic) else np.nan,
                     "n_ic_days": len(ic)})
        if verbose:
            print(f"  {params}  ->  mean_IC {mean_ic:+.4f} | ICIR {icir:+.2f} "
                  f"| hit {(ic > 0).mean() if len(ic) else float('nan'):.0%}")

    leaderboard = (pd.DataFrame(rows)
                   .sort_values("mean_IC", ascending=False)
                   .reset_index(drop=True))

    if leaderboard.empty or not np.isfinite(leaderboard["mean_IC"]).any():
        raise RuntimeError("All configs produced NaN IC (degenerate predictions). "
                           "GBRT is not learning on this target — fix the signal "
                           "before tuning.")

    best = leaderboard.iloc[0]
    best_params = {k: (int(best[k]) if float(best[k]).is_integer() else float(best[k]))
                   for k in grid}

    if verbose:
        print("\n=== LEADERBOARD (early-slice, IC-scored) ===")
        print(leaderboard.round(4).to_string(index=False))
        print(f"\nBest by mean IC: {best_params}  (mean_IC {best['mean_IC']:+.4f}, "
              f"ICIR {best['ICIR']:+.2f})")

    return best_params, leaderboard
