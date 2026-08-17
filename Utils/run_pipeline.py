"""
Headless end-to-end run: dataset -> walk-forward -> combine -> backtest.

    python -m Utils.run_pipeline

Needs network access (FINRA, SEC, Wikipedia, yfinance).
"""
import os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from Utils.alpha_data_loaders import (
    download_sp500_tables, survivorship_complete_universe, build_membership_panel,
    download_finra_short_volume, download_finra_short_interest, align_short_interest_pit,
    build_pit_fundamental_panel,
)
from Utils.features import (build_price_panel, add_cross_sectional_target, merge_alt_data,
                            finalize_dataset, coverage_report)
from Utils.walkforward import run_walkforward
from Utils.forecast_combination import model_ic_table, ic_weighted
from Utils.portfolio import build_weights, backtest, print_pnl, paper_trade_step
from Utils.models import make_model_factories

START_DATE, END_DATE = "2020-06-01", "2026-06-01"
RESTRICT_TO_INDEX = True
WF = dict(train_min=378, test_size=21, step=21, label_horizon=1, embargo=2)


def banner(m): print("\n" + "=" * 70 + f"\n{m}\n" + "=" * 70)


def build_model_df():
    import yfinance as yf
    current, changes = download_sp500_tables()
    universe = survivorship_complete_universe(current, changes)
    sp500_meta = current.rename(columns={"gics_sector": "GICS Sector"})[["symbol", "GICS Sector"]]

    # auto_adjust=True gives split- and dividend-adjusted 'Close'
    md = yf.download(universe, start=START_DATE, end=END_DATE, auto_adjust=True,
                     group_by="ticker", progress=False, threads=True)
    gspc = yf.download("^GSPC", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # survivorship coverage: how many removed names actually have data?
    if hasattr(md.columns, "get_level_values"):
        got = set(md.columns.get_level_values(0))
    else:
        got = set(universe)
    removed = set(changes["removed"].dropna())
    removed_got = removed & got
    print(f"universe requested: {len(universe)} | tickers with data: {len(got)} | "
          f"delisted/removed requested: {len(removed)} | of those with data: "
          f"{len(removed_got)}  (the gap is residual survivorship bias)")

    close = gspc["Close"]; close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    mr = pd.DataFrame({"date": pd.to_datetime(gspc.index)})
    mr["market_ret_1d"] = close.pct_change().values
    mr["future_market_ret_1d"] = mr["market_ret_1d"].shift(-1)

    panel = build_price_panel(md, mr, tickers=universe)
    panel = add_cross_sectional_target(panel, sp500_meta, target_mode="xs_demean")
    grid = pd.DatetimeIndex(sorted(panel["date"].unique()))

    membership = build_membership_panel(current, changes, grid)
    short_vol = download_finra_short_volume(START_DATE, END_DATE, symbols=universe)
    si_raw = download_finra_short_interest(START_DATE, END_DATE, symbols=universe)
    short_int = align_short_interest_pit(si_raw, grid)
    fundamentals = build_pit_fundamental_panel(universe, grid)

    panel = merge_alt_data(panel, membership, short_vol, short_int, fundamentals,
                           restrict_to_index=RESTRICT_TO_INDEX)
    model_df, feature_cols = finalize_dataset(panel)
    return model_df, feature_cols


def main():
    t0 = time.time()
    banner("STAGE 1-3  build dataset")
    model_df, feature_cols = build_model_df()
    print(f"{len(model_df):,} rows, {len(feature_cols)} features")
    cov, by_year = coverage_report(model_df, feature_cols)
    print("\nFEATURE COVERAGE:")
    print(cov.to_string())

    banner("STAGE 4  walk-forward models")
    preds = None; pooled_by_model = {}
    for name, factory in make_model_factories().items():
        print(f"\n--- {name} ---")
        try:
            oos, per_fold, pooled = run_walkforward(model_df, feature_cols, factory,
                                                    verbose=True, **WF)
        except RuntimeError as e:
            print(f"  SKIPPED: {e}")
            continue
        pooled_by_model[name] = pooled
        print(f"  OOS: mean_IC={pooled['mean_IC']:+.4f} ICIR={pooled['ICIR_ann']:+.2f} "
              f"spread={pooled['mean_decile_spread_bps']:.1f}bps t={pooled['spread_t_stat']:+.2f}")
        p = oos[["date", "symbol", "target", "ret_1d", "vol_20d", "adv20"]].copy()
        p[name] = oos["pred"]
        preds = p if preds is None else preds.merge(p[["date", "symbol", name]],
                                                    on=["date", "symbol"])

    model_cols = [m for m in pooled_by_model if not pd.isna(pooled_by_model[m]["mean_IC"])]
    if not model_cols:
        print("\nNo non-degenerate model produced a defined IC. Stop and debug.")
        return

    banner("STAGE 5  forecast combination")
    summ, corr = model_ic_table(preds, model_cols)
    print("per-model OOS IC:\n", summ.round(4).to_string())
    print("\ncross-model IC correlation (low = worth combining):\n", corr.round(2).to_string())
    combo = ic_weighted(preds, model_cols, out_col="pred")

    banner("STAGE 6  portfolio construction + costs")
    w = build_weights(combo, pred_col="pred", scheme="decile", q=10)
    daily, psumm = backtest(w, half_spread_bps=1.0, impact_coef_bps=10.0,
                            adv_col="adv20", vol_target_annual=0.10)
    print_pnl("combined signal, decile L/S, vol-targeted, net of costs", psumm)

    banner("STAGE 7  paper trading (one step)")
    last = combo[combo["date"] == combo["date"].max()][["symbol", "pred"]]
    step = paper_trade_step(last, current_holdings=None, q=10)
    print(f"target book formed: gross={step['gross']:.2f}, "
          f"{int((step['target_weights'].abs()>0).sum())} positions, "
          f"{len(step['trades'])} trades to put on from flat.")

    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
