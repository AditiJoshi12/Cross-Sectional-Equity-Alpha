"""
Feature engineering and the missing-data policy.

Builds the modelling panel from price/volume + alt data sources, applies
per-date cross-sectional winsorisation on the raw continuous features, and
handles missing values with an explicit two-tier policy:

  - Core features (price/volume) must be present — rows without them are dropped.
  - Alt features (short data, fundamentals) are median-filled per date and get
    a companion _isna indicator so the model can learn from missingness instead
    of us dropping 90% of rows.

coverage_report() prints non-null rates by feature and year so a dead data
pipe can't hide. All cross-sectional ranks are within-date only.
"""
import numpy as np
import pandas as pd


def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.rolling(window).mean() / loss.rolling(window).mean()
    return 100 - (100 / (1 + rs))


def rolling_beta(stock_ret, market_ret, window=60):
    cov = stock_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    return cov / var


def winsorize_xs(df, cols, lower=0.01, upper=0.99):
    """Clip each feature to its within-date 1st/99th percentile."""
    out = df.copy()
    for c in cols:
        if c not in out:
            continue
        lo = out.groupby("date")[c].transform(lambda s: s.quantile(lower))
        hi = out.groupby("date")[c].transform(lambda s: s.quantile(upper))
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def build_price_panel(market_data, market_return_data, tickers=None):
    if tickers is None:
        tickers = list({c[0] for c in market_data.columns})
    frames = []
    for t in tickers:
        try:
            df = market_data[t].copy()
        except KeyError:
            continue
        df["ret_1d"] = df["Close"].pct_change()
        for n in (5, 20, 60, 120, 252):
            df[f"ret_{n}d"] = df["Close"].pct_change(n)
        df["dollar_volume"] = df["Close"] * df["Volume"]
        df["vol_20d"] = df["ret_1d"].rolling(20).std()
        rm, rs = df["Volume"].rolling(20).mean(), df["Volume"].rolling(20).std()
        df["volume_zscore"] = (df["Volume"] - rm) / rs
        df["relative_volume"] = df["Volume"] / rm
        df["adv20"] = df["dollar_volume"].rolling(20).mean()
        df["rsi_14"] = compute_rsi(df["Close"])
        df["ma20"] = df["Close"].rolling(20).mean()
        df["ma50"] = df["Close"].rolling(50).mean()
        df["ma_spread"] = df["ma20"] / df["ma50"] - 1
        df["symbol"] = t
        df["date"] = df.index
        df["future_ret_1d"] = df["ret_1d"].shift(-1)
        df["future_ret_5d"] = df["Close"].pct_change(5).shift(-5)
        for col, lo, hi in [("future_ret_1d", -0.9, 1.0), ("future_ret_5d", -0.95, 3.0)]:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        frames.append(df)
    panel = pd.concat(frames).reset_index(drop=True)
    panel = panel.merge(
        market_return_data[["date", "market_ret_1d"]],
        on="date", how="left").sort_values(["symbol", "date"])
    panel["beta_60d"] = (panel.groupby("symbol", group_keys=False)
                         .apply(lambda g: rolling_beta(g["ret_1d"], g["market_ret_1d"])))
    return panel


def add_cross_sectional_target(panel, sp500_meta=None, target_mode="xs_demean"):
    """Construct the prediction target and cross-sectional feature ranks.

    target_mode='xs_demean' (default): forward return minus the equal-weighted
    cross-sectional mean that day — exactly what a dollar-neutral book earns.
    'index_excess' subtracts the S&P 500 return instead (a per-date constant,
    so the cross-sectional ranking is identical; only the regression level changes).
    """
    df = panel.copy()
    if target_mode == "index_excess":
        df["target"] = df["future_ret_5d"] - df["future_market_ret_5d"]
    elif target_mode == "xs_demean":
        fwd = df.groupby("date")["future_ret_5d"].transform(
            lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
        xs_mean = fwd.groupby(df["date"]).transform("mean")
        df["target"] = fwd - xs_mean
    else:
        raise ValueError(target_mode)

    # winsorise the un-ranked continuous features before ranking
    df = winsorize_xs(df, ["rsi_14", "ma_spread", "beta_60d"])

    rank_cols = ["ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_252d",
                 "vol_20d", "volume_zscore", "adv20", "relative_volume"]
    for col in rank_cols:
        df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True)
    df["rsi_vol"] = (df["rsi_14"] * df["vol_20d"]).groupby(df["date"]).rank(pct=True)
    df["maspread_beta"] = (df["ma_spread"] * df["beta_60d"]).groupby(df["date"]).rank(pct=True)

    if sp500_meta is not None:
        df = df.merge(sp500_meta, on="symbol", how="left")
        dummies = pd.get_dummies(df["GICS Sector"], prefix="sector",
                                 drop_first=True, dtype=int)
        df = pd.concat([df, dummies], axis=1)
    return df


def merge_alt_data(df, membership=None, short_vol=None, short_int=None,
                   fundamentals=None, lazy_prices=None, restrict_to_index=True):
    out = df.copy()
    if membership is not None:
        out = out.merge(membership, on=["date", "symbol"], how="left")
        out["in_index"] = out["in_index"].fillna(0)
        if restrict_to_index:
            out = out[out["in_index"] == 1].copy()
    if short_vol is not None and not short_vol.empty:
        cols = [c for c in ["date", "symbol", "short_ratio", "short_ratio_z20",
                            "short_ratio_chg5"] if c in short_vol.columns]
        out = out.merge(short_vol[cols], on=["date", "symbol"], how="left")
        out["short_ratio_rank"] = out.groupby("date")["short_ratio"].rank(pct=True)
    if short_int is not None and not short_int.empty:
        cols = [c for c in ["date", "symbol", "short_interest_rank",
                            "days_to_cover"] if c in short_int.columns]
        out = out.merge(short_int[cols], on=["date", "symbol"], how="left")
    if fundamentals is not None and not fundamentals.empty:
        out = out.merge(fundamentals, on=["date", "symbol"], how="left")
        if "eps_diluted" in out and "Close" in out:
            out["earnings_yield"] = out["eps_diluted"] / out["Close"]
            out["ey_rank"] = out.groupby("date")["earnings_yield"].rank(pct=True)
        if "equity" in out and "assets" in out:
            out["leverage"] = out["assets"] / out["equity"].replace(0, np.nan)
            out["leverage_rank"] = out.groupby("date")["leverage"].rank(pct=True)
    if lazy_prices is not None and not lazy_prices.empty:
        out = out.merge(lazy_prices, on=["date", "symbol"], how="left")
        if "filing_sim_cosine" in out:
            out["filing_sim_rank"] = out.groupby("date")["filing_sim_cosine"].rank(pct=True)
    return out


CORE_FEATURES = [
    "ret_5d_rank", "ret_20d_rank", "ret_60d_rank", "ret_120d_rank", "ret_252d_rank",
    "vol_20d_rank", "adv20_rank", "relative_volume_rank", "volume_zscore_rank",
    "rsi_14", "ma_spread", "rsi_vol", "maspread_beta", "beta_60d",
]
ALT_FEATURES = ["short_ratio_rank", "short_ratio_z20", "short_ratio_chg5",
                "short_interest_rank", "days_to_cover",
                "ey_rank", "leverage_rank", "filing_sim_rank",
                "beta_wti_rank", "beta_ust10y_rank", "beta_hy_spread_rank",
                "beta_usd_broad_rank", "beta_vix_rank"]


def coverage_report(df, feature_cols):
    """Non-null rate per feature, overall and by year."""
    cov = pd.DataFrame({"pct_nonnull": df[feature_cols].notna().mean().round(4)})
    if "date" in df:
        yr = df.assign(_y=pd.to_datetime(df["date"]).dt.year)
        by_year = (yr.groupby("_y")[feature_cols].apply(lambda g: g.notna().mean())
                   .round(3))
        return cov, by_year
    return cov, None


def finalize_dataset(df, extra_features=None):
    """Apply the missing-value policy and return (model_df, feature_cols).

    Core features (price/volume): row dropped if any are missing.
    Alt features (short, fundamentals): cross-sectional median fill per date
    plus a companion _isna flag. All-NaN alt features are excluded entirely.
    """

    df['target'] = df['target'].replace([np.inf, -np.inf], np.nan)
    df = winsorize_xs(df, ['target'], lower=0.01, upper=0.99)
    
    sector_cols = [c for c in df.columns if c.startswith("sector_")]
    core = [c for c in CORE_FEATURES if c in df.columns]
    alt = [c for c in ALT_FEATURES if c in df.columns and df[c].notna().any()]

    out = df.copy()
    feature_cols = list(core)
    for c in alt:
        out[c + "_isna"] = out[c].isna().astype(int)
        out[c] = out.groupby("date")[c].transform(lambda s: s.fillna(s.median()))
        # if a whole cross-section is empty, fill with a neutral constant
        # (not the global median — that would peek at future dates)
        neutral = 0.5 if c.endswith("_rank") else 0.0
        out[c] = out[c].fillna(neutral)
        feature_cols += [c, c + "_isna"]
    feature_cols += sector_cols
    if extra_features:
        feature_cols += [c for c in extra_features if c in out.columns]

    carry = [c for c in ["ret_1d", "vol_20d", "adv20"] if c in out.columns]
    required = feature_cols + ["target", "date", "symbol"]
    model_df = out.dropna(subset=core + ["target"])[required + carry].copy()
    model_df = model_df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return model_df, feature_cols


# --- self-tests (synthetic data) -------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=10)
    rows = []
    for d in dates:
        for s in range(20):
            rows.append((d, f"S{s:02d}", rng.normal(), rng.normal(),
                         np.nan if s % 2 else rng.normal()))   # alt half-missing
    df = pd.DataFrame(rows, columns=["date", "symbol", "ret_20d_rank",
                                     "target", "short_ratio_rank"])
    # winsorisation
    df.loc[0, "ret_20d_rank"] = 1e6
    w = winsorize_xs(df, ["ret_20d_rank"])
    assert w["ret_20d_rank"].max() < 1e6, "winsorisation failed"
    # finalize: alt feature should be median-filled, _isna flag present
    df["beta_60d"] = 1.0
    md, fc = finalize_dataset(df)
    assert "short_ratio_rank_isna" in fc, "missing indicator not created"
    assert md["short_ratio_rank"].notna().all(), "alt not filled"
    assert len(md) == len(df), "rows dropped despite fillable alt data"
    cov, by_year = coverage_report(df, ["ret_20d_rank", "short_ratio_rank"])
    print("[selftest] winsorise + fill + _isna flags  OK")
    print("coverage:\n", cov.to_string())
