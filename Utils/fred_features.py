"""
FRED macro data -> per-stock rolling macro sensitivities.

Raw macro series (VIX, oil, 10Y, etc.) are the same value for every stock on
a given date, so in a cross-sectionally demeaned model they carry zero
information. The useful signal is each stock's rolling *beta* to those factors
— energy loads on oil, banks on the curve, etc. — which differs across names
and survives demeaning.

Pipeline:
  1. load_fred_pit()             — daily PIT macro levels (vintage-aware via
                                   get_series_all_releases + a knowledge lag)
  2. make_macro_factor_changes() — daily innovations (returns or diffs)
  3. add_macro_sensitivities()   — per-symbol rolling betas, cross-sectionally
                                   ranked — these are what you feed the model

Also includes check_fred_levels / check_factor_changes / check_macro_features
for real-data diagnostics (coverage, redundancy vs the market beta, standalone IC).
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd


# Market-priced (non-revised) series — latest release is already PIT
FRED_MARKET_SERIES = {
    "WTI":          "DCOILWTICO",     # WTI crude oil, daily
    "UST10Y":       "DGS10",          # 10Y Treasury yield, daily
    "HY_SPREAD":    "BAMLH0A0HYM2",   # ICE BofA US High Yield OAS, daily
    "USD_BROAD":    "DTWEXBGS",       # Broad trade-weighted USD index, daily
    "VIX":          "VIXCLS",         # CBOE VIX, daily
}

# Revised macro (regime use only, not cross-sectional features)
FRED_REVISED_SERIES = {
    "CPI":          "CPIAUCSL",
    "UNRATE":       "UNRATE",
    "FEDFUNDS":     "FEDFUNDS",
    "T10Y2Y":       "T10Y2Y",
    "UMCSENT":      "UMCSENT",
}

# "ret" = pct_change (price-like), "diff" = first difference (rate/spread/level)
FACTOR_INNOVATION = {
    "WTI":       "ret",
    "USD_BROAD": "ret",
    "UST10Y":    "diff",
    "HY_SPREAD": "diff",
    "VIX":       "diff",
}

# the per-stock sensitivity rank columns this module emits
MACRO_BETA_FEATURES = [f"beta_{k.lower()}_rank" for k in FRED_MARKET_SERIES]


# --- PIT FRED loader -------------------------------------------------------
def _series_pit(fred, series_id, grid, knowledge_lag_days):
    """Value known as of each grid date, using vintage data when available."""
    grid = pd.DatetimeIndex(sorted(pd.to_datetime(grid)))
    try:
        rel = fred.get_series_all_releases(series_id)  # cols: realtime_start, date, value
        rel = rel.dropna(subset=["value"]).copy()
        rel["realtime_start"] = pd.to_datetime(rel["realtime_start"])
        rel["date"] = pd.to_datetime(rel["date"])
        rel = rel.sort_values(["date", "realtime_start"])
        first_known = rel.groupby("date", as_index=False).first()
        known = (first_known[["realtime_start", "value"]]
                 .rename(columns={"realtime_start": "known_date"})
                 .sort_values("known_date"))
        used_fallback = False
    except Exception as e:
        s = fred.get_series(series_id).dropna()
        known = (pd.DataFrame({"known_date": pd.to_datetime(s.index), "value": s.values})
                 .sort_values("known_date"))
        used_fallback = True
        print(f"  ! {series_id}: all-releases unavailable ({str(e)[:40]}); "
              f"using get_series + lag (OK for non-revised market series).")

    # lag: a value is only usable knowledge_lag_days business days after publication
    known["known_date"] = (known["known_date"]
                           + pd.tseries.offsets.BDay(knowledge_lag_days))
    out = pd.merge_asof(pd.DataFrame({"date": grid}),
                        known.rename(columns={"known_date": "date"}),
                        on="date", direction="backward")
    return out.set_index("date")["value"], used_fallback


def load_fred_pit(series_map, start, end, api_key, grid=None,
                  knowledge_lag_days=1, pause=0.0):
    """Daily PIT macro level panel.

    series_map: {label: fred_series_id}. Pass your actual trading grid for
    clean joins; defaults to bdate_range(start, end).
    """
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError("fredapi is required — see requirements.txt") from e

    fred = Fred(api_key=api_key)
    if grid is None:
        grid = pd.bdate_range(start, end)
    grid = pd.DatetimeIndex(sorted(pd.to_datetime(grid)))

    cols = {}
    for label, sid in series_map.items():
        try:
            s, _ = _series_pit(fred, sid, grid, knowledge_lag_days)
            cols[label] = s
            print(f"  ok {label:10s} ({sid})")
        except Exception as e:
            print(f"  FAIL {label} ({sid}): {e}")
        if pause:
            time.sleep(pause)

    if not cols:
        return pd.DataFrame(index=grid)
    out = pd.DataFrame(cols).sort_index()
    out = out.ffill()
    return out


# --- factor innovations ----------------------------------------------------
def make_macro_factor_changes(fred_daily, innovation_map=None):
    """Turn macro levels into daily innovations (returns or diffs)."""
    innovation_map = innovation_map or FACTOR_INNOVATION
    df = fred_daily.copy()
    out = {}
    for col in df.columns:
        how = innovation_map.get(col, "diff")
        x = df[col].pct_change() if how == "ret" else df[col].diff()
        out[f"{col}_chg"] = x
    chg = pd.DataFrame(out, index=df.index)
    chg = chg.replace([np.inf, -np.inf], np.nan)
    chg.index.name = "date"
    return chg.reset_index()


# --- per-stock rolling sensitivities -> cross-sectional features -----------
def _rolling_beta(y, x, window, min_periods):
    cov = y.rolling(window, min_periods=min_periods).cov(x)
    var = x.rolling(window, min_periods=min_periods).var()
    return cov / var.replace(0, np.nan)


def add_macro_sensitivities(panel, macro_changes, factors=None,
                            ret_col="ret_1d", window=60, min_periods=40,
                            rank=True):
    """Rolling beta of each stock's returns to each macro factor, ranked
    cross-sectionally per date. The _rank columns are the model features."""
    factors = factors or [c for c in macro_changes.columns if c.endswith("_chg")]
    df = panel.merge(macro_changes, on="date", how="left").sort_values(["symbol", "date"])

    rank_cols = []
    for f in factors:
        base = f[:-4] if f.endswith("_chg") else f       # WTI_chg -> WTI
        bcol = f"beta_{base.lower()}"
        df[bcol] = (df.groupby("symbol", group_keys=False)
                      .apply(lambda g: _rolling_beta(g[ret_col], g[f],
                                                     window, min_periods)))
        if rank:
            rcol = f"{bcol}_rank"
            df[rcol] = df.groupby("date")[bcol].rank(pct=True)
            rank_cols.append(rcol)

    # drop the merged factor-change columns, keep only betas/ranks
    df = df.drop(columns=[c for c in factors if c in df.columns])
    return df, rank_cols


# --- real-data diagnostics -------------------------------------------------
def check_fred_levels(fred_daily):
    """Print span, coverage, and staleness of the PIT level panel."""
    print("=== FRED LEVELS ===")
    if fred_daily.empty:
        print("  EMPTY — no series loaded (check API key / series ids).")
        return
    print(f"  span : {fred_daily.index.min().date()} -> {fred_daily.index.max().date()}"
          f"   rows: {len(fred_daily)}")
    for col in fred_daily.columns:
        s = fred_daily[col]
        null = s.isna().mean()
        # how often the carried-forward value actually updates
        changed = s.ne(s.shift()).mean()
        last_val = s.dropna().iloc[-1] if s.notna().any() else float("nan")
        last_dt = s.dropna().index.max()
        print(f"  {col:11s} null {null:5.1%} | updates on {changed:5.1%} of days "
              f"| last {last_val:>10.4g} @ {pd.Timestamp(last_dt).date() if pd.notna(last_dt) else 'NA'}")
    print()


def check_factor_changes(macro_chg):
    """Print stats on the daily innovations (should have zero mean, finite std)."""
    print("=== FACTOR CHANGES (innovations) ===")
    cols = [c for c in macro_chg.columns if c.endswith("_chg")]
    for c in cols:
        x = macro_chg[c]
        print(f"  {c:15s} n {x.notna().sum():>5} | mean {x.mean():+.4g} "
              f"| std {x.std():.4g} | min {x.min():+.3g} | max {x.max():+.3g} "
              f"| infs {np.isinf(x).sum()}")
    print()


def check_macro_features(model_df, rank_cols, target_col="target",
                         redundancy_col="beta_60d"):
    """Coverage, redundancy vs the market beta, and standalone IC for each
    macro-beta feature. Read before deciding which to keep in the model."""
    from scipy.stats import spearmanr
    print("=== MACRO FEATURE DIAGNOSTICS (real data) ===")
    print(f"{'feature':22s} {'cover':>6} {'xs_disp':>8} "
          f"{'corr_'+redundancy_col:>14} {'mean_IC':>9} {'ICIR':>7}")
    for rcol in rank_cols:
        if rcol not in model_df.columns:
            print(f"  {rcol:20s}  MISSING from model_df")
            continue
        cover = model_df[rcol].notna().mean()
        raw = rcol[:-5] if rcol.endswith("_rank") else rcol     # beta_vix_rank -> beta_vix
        # cross-sectional dispersion of the rank itself (should be ~0.29 for U[0,1])
        xs_disp = model_df.groupby("date")[rcol].std().mean()
        # redundancy vs existing risk beta
        if redundancy_col in model_df.columns:
            corr = (model_df.dropna(subset=[rcol, redundancy_col])
                    .groupby("date")
                    .apply(lambda g: g[rcol].corr(g[redundancy_col], method="spearman"))
                    .mean())
        else:
            corr = float("nan")
        # standalone IC vs target
        ic = (model_df.dropna(subset=[rcol, target_col])
              .groupby("date")
              .apply(lambda g: spearmanr(g[rcol], g[target_col]).correlation)
              .dropna())
        mean_ic = ic.mean() if len(ic) else float("nan")
        icir = (mean_ic / ic.std() * np.sqrt(252)) if (len(ic) and ic.std() > 0) else float("nan")
        print(f"  {rcol:20s} {cover:5.1%} {xs_disp:8.3f} {corr:14.2f} "
              f"{mean_ic:+9.4f} {icir:+7.2f}")
    print()
