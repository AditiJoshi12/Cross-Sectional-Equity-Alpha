"""
Embargoed expanding-window walk-forward CV for a daily cross-sectional panel.

Two guardrails baked in:
  - Positional embargo assertion: every fold checks that the trading-day gap
    between last train date and first test date >= label_horizon + embargo.
    A violation raises immediately.
  - Degeneracy guard: if the model collapses to constant predictions on too
    many test days (e.g. over-regularised ElasticNet), the run raises instead
    of silently reporting NaN IC.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class Fold:
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def make_walkforward_folds(dates, train_min, test_size, step=None,
                           label_horizon=1, embargo=2, expanding=True):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(pd.unique(dates))))
    step = step or test_size
    gap = label_horizon + embargo
    folds = []
    test_start = train_min + gap
    while test_start < len(dates):
        test_end = min(test_start + test_size, len(dates))
        train_end = test_start - gap
        train_start = 0 if expanding else max(0, train_end - train_min)
        if train_end <= train_start:
            test_start += step
            continue
        folds.append(Fold(train_dates=dates[train_start:train_end],
                          test_dates=dates[test_start:test_end]))
        test_start += step
    return folds


def _assert_no_leakage(all_dates, fold, label_horizon, embargo):
    """Check that the trading-day gap between train and test >= H + embargo."""
    last_train = fold.train_dates.max()
    first_test = fold.test_dates.min()
    gap = all_dates.get_loc(first_test) - all_dates.get_loc(last_train)
    assert gap >= label_horizon + embargo, (
        f"embargo violated: gap={gap} < H+embargo={label_horizon + embargo}")
    return True


# --- diagnostics -----------------------------------------------------------
def daily_ic(df, pred_col="pred", target_col="target"):
    def _ic(g):
        if g[pred_col].nunique() < 3 or g[target_col].nunique() < 3:
            return np.nan
        return spearmanr(g[pred_col], g[target_col]).correlation
    return (df.groupby("date")[[pred_col, target_col]]
              .apply(_ic).dropna())


def decile_spread(df, pred_col="pred", ret_col="target", q=10):
    def _spread(g):
        if len(g) < q:
            return np.nan
        ranks = g[pred_col].rank(pct=True)
        top = g.loc[ranks >= 1 - 1 / q, ret_col].mean()
        bot = g.loc[ranks <= 1 / q, ret_col].mean()
        return top - bot
    return (df.groupby("date")[[pred_col, ret_col]]
              .apply(_spread).dropna())


def summarize(oos, pred_col="pred", target_col="target"):
    ic = daily_ic(oos, pred_col, target_col)
    spread = decile_spread(oos, pred_col, target_col)
    icir = ic.mean() / ic.std() if (len(ic) and ic.std() > 0) else np.nan
    return {
        "n_obs": len(oos),
        "n_days": oos["date"].nunique(),
        "mean_IC": ic.mean() if len(ic) else np.nan,
        "IC_std": ic.std() if len(ic) else np.nan,
        "ICIR_ann": icir * np.sqrt(252) if pd.notna(icir) else np.nan,
        "IC_hit_rate": (ic > 0).mean() if len(ic) else np.nan,
        "mean_decile_spread_bps": spread.mean() * 1e4 if len(spread) else np.nan,
        "spread_t_stat": (spread.mean() / (spread.std() / np.sqrt(len(spread)))
                          if (len(spread) and spread.std() > 0) else np.nan),
        "frac_degenerate_days": float(
            1 - len(ic) / max(1, oos["date"].nunique())),
    }


# --- the harness -----------------------------------------------------------
def run_walkforward(model_df, feature_cols, model_factory,
                    target_col="target", train_min=378, test_size=21, step=21,
                    label_horizon=1, embargo=2, expanding=True,
                    allow_degenerate=False, degenerate_tol=0.5, verbose=True):
    """Run the walk-forward and return (oos_df, per_fold_df, pooled_stats).

    Raises RuntimeError if more than degenerate_tol of test days have constant
    predictions (model not learning). Pass allow_degenerate=True to inspect anyway.
    """
    df = model_df.sort_values(["date", "symbol"]).copy()
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    folds = make_walkforward_folds(dates, train_min, test_size, step,
                                   label_horizon, embargo, expanding)
    if not folds:
        raise ValueError("No folds produced -- lower train_min or check span.")

    oos_parts, fold_rows = [], []
    for i, fold in enumerate(folds):
        _assert_no_leakage(dates, fold, label_horizon, embargo)
        tr = df[df["date"].isin(fold.train_dates)]
        te = df[df["date"].isin(fold.test_dates)]
        if tr.empty or te.empty:
            continue

        model = model_factory()
        model.fit(tr[feature_cols].values, tr[target_col].values)
        pred = np.asarray(model.predict(te[feature_cols].values), dtype=float)

        part = te[["date", "symbol", target_col, "ret_1d"]].copy()
        part["pred"] = pred
        part["fold"] = i
        oos_parts.append(part)

        fic = daily_ic(part)
        # degeneracy: are predictions constant within each test day?
        day_std = part.groupby("date")["pred"].std()
        degen = float((day_std.fillna(0) < 1e-12).mean())
        fold_rows.append({
            "fold": i,
            "train_end": fold.train_dates.max().date(),
            "test_start": fold.test_dates.min().date(),
            "test_end": fold.test_dates.max().date(),
            "n_train": len(tr), "n_test": len(te),
            "fold_mean_IC": fic.mean() if len(fic) else np.nan,
            "frac_degenerate_days": degen,
        })
        if verbose:
            r = fold_rows[-1]
            flag = "  <-- DEGENERATE" if r["frac_degenerate_days"] > 0.5 else ""
            ic_str = f"{r['fold_mean_IC']:+.4f}" if pd.notna(r['fold_mean_IC']) else "  nan"
            print(f"fold {i:>2} | train->{r['train_end']} "
                  f"| test {r['test_start']}->{r['test_end']} | IC {ic_str}{flag}")

    oos = pd.concat(oos_parts, ignore_index=True)
    per_fold = pd.DataFrame(fold_rows)
    pooled = summarize(oos)

    frac_degen = pooled["frac_degenerate_days"]
    if frac_degen > degenerate_tol and not allow_degenerate:
        raise RuntimeError(
            f"MODEL DEGENERATE: {frac_degen:.0%} of test days have constant "
            f"predictions (IC undefined). The model is not learning a cross-section. "
            f"Fix the model (e.g. penalty/target scaling) before trusting results. "
            f"Pass allow_degenerate=True only to inspect.")
    return oos, per_fold, pooled


# --- self-test (synthetic) -------------------------------------------------
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n_dates, n_syms = 300, 60
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rows = []
    for d in dates:
        x = rng.normal(size=n_syms)
        target = 0.6 * x + 0.4 * rng.normal(size=n_syms)
        for s in range(n_syms):
            rows.append((d, f"S{s:02d}", x[s], rng.normal(), target[s], target[s]))
    panel = pd.DataFrame(rows, columns=["date", "symbol", "f0", "f1", "target", "ret_1d"])

    oos, per_fold, pooled = run_walkforward(
        panel, ["f0", "f1"], lambda: LinearRegression(),
        train_min=126, test_size=21, step=21, verbose=False)
    assert pooled["mean_IC"] > 0.3, "planted signal not recovered"
    print(f"[selftest] healthy model recovered: mean_IC={pooled['mean_IC']:+.3f} OK")

    # a constant-output model should trigger the degeneracy guard
    class ConstModel:
        def fit(self, X, y): self.c = float(np.mean(y))
        def predict(self, X): return np.full(len(X), self.c)
    try:
        run_walkforward(panel, ["f0", "f1"], lambda: ConstModel(),
                        train_min=126, test_size=21, step=21, verbose=False)
        raise AssertionError("degeneracy guard FAILED to fire")
    except RuntimeError as e:
        print(f"[selftest] degeneracy guard fired on constant model OK")
