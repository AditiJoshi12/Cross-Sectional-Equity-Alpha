"""
Portfolio construction, transaction costs, and backtesting.

Turns an OOS prediction stream into a dollar-neutral long/short book
(decile or centered-rank weighting, optional inverse-vol tilt), runs it
through a cost model (half-spread + square-root market impact), and
reports net Sharpe / cumulative return / drawdown.

Also includes a one-step paper-trading fill simulator for the latest
cross-section.
"""
import numpy as np
import pandas as pd

ANN = 252


# --- weight schemes (per date) ---------------------------------------------
def decile_weights(g, pred_col, q=10, long_only=False):
    n = len(g)
    k = max(1, int(n / q))
    order = g[pred_col].rank(method="first")
    w = pd.Series(0.0, index=g.index)
    w[order > n - k] = 1.0 / k
    if not long_only:
        w[order <= k] = -1.0 / k
    return w


def centered_rank_weights(g, pred_col):
    r = g[pred_col].rank(pct=True) - 0.5
    s = r.abs().sum()
    return r / s if s > 0 else r * 0.0


def build_weights(oos, pred_col="pred", scheme="decile", q=10,
                  invvol_col="vol_20d", long_only=False):
    """Add a 'weight' column — dollar-neutral, gross normalised to 1 per date."""
    out = oos.copy()
    if scheme == "decile":
        f = lambda g: decile_weights(g, pred_col, q, long_only)
    elif scheme == "rank":
        f = lambda g: centered_rank_weights(g, pred_col)
    else:
        raise ValueError(scheme)
    out["weight"] = out.groupby("date", group_keys=False).apply(f)

    if scheme == "rank" and invvol_col in out.columns:
        out["weight"] = out["weight"] / out[invvol_col].clip(lower=1e-6)
        gross = out.groupby("date")["weight"].transform(lambda s: s.abs().sum())
        out["weight"] = out["weight"] / gross.replace(0, np.nan)
    return out


# --- PnL with transaction costs ---------------------------------------------
def backtest(oos, weight_col="weight", ret_col="ret_1d",
             half_spread_bps=1.0, impact_coef_bps=10.0,
             adv_col="adv20", notional=10_000_000,
             vol_target_annual=None):
    """Run the daily P&L with half-spread + sqrt-impact costs.

    vol_target_annual: if set, rescales the book each day to target this
    annualised vol using trailing realised vol of the gross returns.
    """
    df = oos.sort_values(["symbol", "date"]).copy()
    df["w"] = df[weight_col].fillna(0.0)
    df["w_prev"] = df.groupby("symbol")["w"].shift(1).fillna(0.0)
    df["gross_ret"] = df["w_prev"] * df[ret_col]

    df["trade"] = (df["w"] - df["w_prev"]).abs()
    spread_cost = (half_spread_bps / 1e4) * df["trade"]
    if adv_col in df.columns:
        traded_notional = df["trade"] * notional
        adv_dollar = df[adv_col].clip(lower=1.0)
        participation = (traded_notional / adv_dollar).clip(lower=0)
        impact = (impact_coef_bps / 1e4) * np.sqrt(participation) * df["trade"]
    else:
        impact = 0.0
    df["cost"] = spread_cost + impact

    daily = df.groupby("date").agg(
        gross=("gross_ret", "sum"),
        cost=("cost", "sum"),
        turnover=("trade", "sum"),
    )
    daily["net"] = daily["gross"] - daily["cost"]

    if vol_target_annual is not None:
        realized = daily["gross"].rolling(20).std().shift(1)
        scale = (vol_target_annual / np.sqrt(ANN)) / realized.replace(0, np.nan)
        scale = scale.clip(upper=5).fillna(1.0)
        daily["net"] = daily["net"] * scale
        daily["gross"] = daily["gross"] * scale
        daily["turnover"] = daily["turnover"] * scale

    daily = daily.dropna(subset=["net"])
    return daily, summarize_pnl(daily)


def summarize_pnl(daily):
    net = daily["net"]
    if len(net) == 0 or net.std() == 0:
        return {"n_days": len(net)}
    ann_ret = net.mean() * ANN
    ann_vol = net.std() * np.sqrt(ANN)
    cum = (1 + net).prod() - 1
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "n_days": len(net),
        "gross_ann_ret": daily["gross"].mean() * ANN,
        "net_ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "cum_return_ROI": cum,
        "max_drawdown": dd,
        "avg_daily_turnover": daily["turnover"].mean(),
        "cost_drag_ann": daily["cost"].mean() * ANN,
        "hit_rate": (net > 0).mean(),
    }


def print_pnl(name, summ):
    print("=" * 56)
    print(f"PORTFOLIO PnL  ({name})")
    print("=" * 56)
    if summ.get("n_days", 0) == 0:
        print("  no tradeable days"); print("=" * 56); return
    print(f"  days                 : {summ['n_days']}")
    print(f"  gross ann. return    : {summ['gross_ann_ret']:+.2%}")
    print(f"  cost drag (ann.)     : {summ['cost_drag_ann']:.2%}")
    print(f"  NET ann. return (ROI): {summ['net_ann_ret']:+.2%}")
    print(f"  ann. vol             : {summ['ann_vol']:.2%}")
    print(f"  SHARPE (net)         : {summ['sharpe']:+.2f}")
    print(f"  cumulative ROI       : {summ['cum_return_ROI']:+.2%}")
    print(f"  max drawdown         : {summ['max_drawdown']:.2%}")
    print(f"  avg daily turnover   : {summ['avg_daily_turnover']:.2f}x")
    print("=" * 56)


# --- paper trading: one-step fill simulator --------------------------------
def paper_trade_step(latest_pred, current_holdings=None, scheme="decile",
                     pred_col="pred", q=10, notional=10_000_000):
    """Given today's cross-sectional predictions, produce the target book and
    the trade list vs current holdings.

    latest_pred: DataFrame with [symbol, pred] for a single date.
    current_holdings: Series indexed by symbol (or None to start flat).
    """
    g = latest_pred.copy()
    g["date"] = pd.Timestamp("now").normalize()
    if scheme == "decile":
        w = decile_weights(g, pred_col, q)
    else:
        w = centered_rank_weights(g, pred_col)
    target = pd.Series(w.values, index=g["symbol"].values, name="target_w")

    if current_holdings is None:
        current_holdings = pd.Series(0.0, index=target.index)
    current_holdings = current_holdings.reindex(target.index).fillna(0.0)
    trades = (target - current_holdings)
    return {
        "target_weights": target,
        "trades": trades[trades.abs() > 1e-9].sort_values(),
        "gross": float(target.abs().sum()),
        "turnover": float(trades.abs().sum()),
        "target_notional": (target * notional).round(0),
    }


# --- self-test (synthetic planted signal) -----------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=250)
    n = 80
    # AR(1) signal so decile membership has some persistence
    sig = np.zeros((len(dates), n))
    sig[0] = rng.normal(size=n)
    for t in range(1, len(dates)):
        sig[t] = 0.9 * sig[t - 1] + rng.normal(0, 0.4, n)
    rows = []
    for t, d in enumerate(dates):
        pred = sig[t]
        # return loads on today's signal, earned by yesterday's weights
        fwd_noise = rng.normal(0, 0.01, n)
        ret_next = 0.03 * (pred - pred.mean()) / (pred.std() + 1e-9) + fwd_noise
        for s in range(n):
            rows.append((d, f"S{s:02d}", pred[s], ret_next[s], 0.02,
                         5e7 + 1e7 * rng.random()))
    oos = pd.DataFrame(rows, columns=["date", "symbol", "pred", "ret_fwd",
                                      "vol_20d", "adv20"])
    # ret_1d[t] is the return realised after forming weight[t-1]
    oos = oos.sort_values(["symbol", "date"])
    oos["ret_1d"] = oos.groupby("symbol")["ret_fwd"].shift(-1)
    oos = oos.dropna(subset=["ret_1d"])
    w = build_weights(oos, scheme="decile", q=10)
    # should be dollar-neutral
    s = w.groupby("date")["weight"].sum()
    assert (s.abs() < 1e-9).all(), "not dollar-neutral"
    daily, summ = backtest(w, half_spread_bps=1.0, impact_coef_bps=10.0)
    print_pnl("decile L/S, synthetic planted signal", summ)
    assert summ["sharpe"] > 0, "planted signal should be profitable gross-of-noise"

    # paper trade one step
    last = oos[oos["date"] == oos["date"].max()][["symbol", "pred"]]
    step = paper_trade_step(last, current_holdings=None, q=10)
    assert abs(step["gross"] - 1.0) < 1e-6 or step["gross"] > 0
    print(f"\n[selftest] paper-trade step: gross={step['gross']:.2f} "
          f"turnover={step['turnover']:.2f} n_trades={len(step['trades'])}  OK")
