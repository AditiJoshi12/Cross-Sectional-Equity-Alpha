"""
Prediction residualisation against risk factors.

Regresses the signal cross-sectionally (per date) on market beta, momentum,
value, size, leverage, and sector dummies, then keeps the residual. The
portfolio formed on the residual has near-zero loading on these common factors,
so we're not accidentally harvesting a known risk premium and calling it alpha.
"""
import numpy as np
import pandas as pd

DEFAULT_STYLE = ["beta_60d", "ret_252d_rank", "ey_rank", "adv20_rank", "leverage_rank"]


def residualize_prediction(df, pred_col="pred", style_cols=None,
                           out_col="pred_resid", include_sectors=True):
    """Per-date OLS residual of the prediction on style characteristics
    (+ sector dummies). Handles missing columns and small cross-sections."""
    style_cols = [c for c in (style_cols or DEFAULT_STYLE) if c in df.columns]
    sector_cols = [c for c in df.columns if c.startswith("sector_")] if include_sectors else []
    reg_cols = style_cols + sector_cols
    out = df.copy()
    if not reg_cols:
        out[out_col] = out[pred_col]
        return out, []

    def _resid(g):
        y = g[pred_col].astype(float).values
        cols = [c for c in reg_cols if g[c].notna().any()]
        if len(g) <= len(cols) + 2 or not cols:
            return pd.Series(y - y.mean(), index=g.index)   # just demean
        X = g[cols].astype(float)
        X = X.fillna(X.median())
        X = np.column_stack([np.ones(len(g)), X.values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return pd.Series(y - X @ beta, index=g.index)

    out[out_col] = out.groupby("date", group_keys=False).apply(_resid)
    return out, reg_cols


def exposure_report(df, pred_col, style_cols=None):
    """Mean per-date rank correlation of the signal with each style factor.
    Run before and after residualisation to confirm exposures collapse."""
    style_cols = [c for c in (style_cols or DEFAULT_STYLE) if c in df.columns]
    rows = {}
    for c in style_cols:
        corr = (df.dropna(subset=[pred_col, c])
                  .groupby("date")
                  .apply(lambda g: g[pred_col].corr(g[c], method="spearman")))
        rows[c] = corr.mean()
    return pd.Series(rows, name="mean_xs_corr")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=40)
    rows = []
    for d in dates:
        n = 60
        beta = rng.normal(size=n)
        mom = rng.uniform(size=n)
        true_alpha = rng.normal(size=n)
        # pred is contaminated by beta and momentum exposure
        pred = true_alpha + 1.5 * beta + 2.0 * mom
        for s in range(n):
            rows.append((d, f"S{s:02d}", pred[s], beta[s], mom[s]))
    df = pd.DataFrame(rows, columns=["date", "symbol", "pred", "beta_60d", "ret_252d_rank"])

    before = exposure_report(df, "pred", ["beta_60d", "ret_252d_rank"])
    res, used = residualize_prediction(df, style_cols=["beta_60d", "ret_252d_rank"],
                                       include_sectors=False)
    after = exposure_report(res, "pred_resid", ["beta_60d", "ret_252d_rank"])
    print("exposure BEFORE residualisation:\n", before.round(3).to_string())
    print("\nexposure AFTER residualisation:\n", after.round(3).to_string())
    assert after.abs().max() < 0.1, "residualisation did not remove exposure"
    print("\n[selftest] risk-factor exposures collapse to ~0 after residualisation  OK")
