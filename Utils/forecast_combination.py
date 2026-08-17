"""
Forecast combination — merge multiple model OOS predictions into one signal.

Two approaches implemented:
  - rank_average: per-date average of cross-sectional percentile ranks (scale-free,
    robust, the sensible default).
  - ic_weighted: weight each model by its trailing OOS IC, computed only from days
    strictly before each date so there's no look-ahead.

model_ic_table() reports per-model IC and the cross-model correlation matrix;
low correlation is what justifies combining at all.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _xs_rank(df, cols):
    out = df.copy()
    for c in cols:
        out[c + "_r"] = out.groupby("date")[c].rank(pct=True)
    return out, [c + "_r" for c in cols]


def model_ic_table(preds, model_cols, target_col="target"):
    """Daily Spearman IC per model and the cross-model IC correlation matrix."""
    ics = {}
    for c in model_cols:
        def _ic(g, c=c):
            if g[c].nunique() < 3 or g[target_col].nunique() < 3:
                return np.nan
            return spearmanr(g[c], g[target_col]).correlation
        ics[c] = preds.groupby("date")[[c, target_col]].apply(_ic)
    ic_df = pd.DataFrame(ics).dropna(how="all")
    summary = pd.DataFrame({
        "mean_IC": ic_df.mean(),
        "ICIR_ann": ic_df.mean() / ic_df.std() * np.sqrt(252),
        "hit_rate": (ic_df > 0).mean(),
    })
    return summary, ic_df.corr()


def rank_average(preds, model_cols, out_col="pred_combo"):
    out, rcols = _xs_rank(preds, model_cols)
    out[out_col] = out[rcols].mean(axis=1)
    return out


def ic_weighted(preds, model_cols, target_col="target",
                out_col="pred_combo", min_history=21, floor=0.0):
    """Trailing-IC-weighted combination — weights use only past information."""
    out, rcols = _xs_rank(preds, model_cols)
    daily = {}
    for c, rc in zip(model_cols, rcols):
        def _ic(g, c=c):
            if g[c].nunique() < 3 or g[target_col].nunique() < 3:
                return np.nan
            return spearmanr(g[c], g[target_col]).correlation
        daily[c] = out.groupby("date")[[c, target_col]].apply(_ic)
    daily = pd.DataFrame(daily)
    # trailing mean IC, shifted by 1 day so we never peek ahead
    w = daily.expanding(min_periods=min_history).mean().shift(1).clip(lower=floor)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / len(model_cols))
    wmap = w.to_dict("index")

    def _combo(g):
        d = g.name
        weights = wmap.get(d, {c: 1.0 / len(model_cols) for c in model_cols})
        acc = np.zeros(len(g))
        for c, rc in zip(model_cols, rcols):
            acc = acc + float(weights.get(c, 0.0)) * g[rc].values
        return pd.Series(acc, index=g.index)

    out[out_col] = out.groupby("date", group_keys=False).apply(_combo)
    return out


# --- self-test -------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", periods=120)
    rows = []
    for d in dates:
        n = 50
        sig = rng.normal(size=n)
        tgt = 0.3 * sig + rng.normal(size=n)
        m1 = sig + 0.5 * rng.normal(size=n)          # decent model
        m2 = 0.2 * sig + rng.normal(size=n)          # weak/noisy model
        for s in range(n):
            rows.append((d, f"S{s:02d}", m1[s], m2[s], tgt[s]))
    preds = pd.DataFrame(rows, columns=["date", "symbol", "m1", "m2", "target"])

    summ, corr = model_ic_table(preds, ["m1", "m2"])
    combo = rank_average(preds, ["m1", "m2"])
    s2, _ = model_ic_table(combo.assign(target=combo["target"]),
                           ["m1", "m2", "pred_combo"])
    assert s2.loc["pred_combo", "mean_IC"] >= summ.loc["m2", "mean_IC"], \
        "combo should beat the worst component"
    print("[selftest] rank-average combo >= worst component  OK")
    print(s2.round(4).to_string())
