import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

def load_sp500_csv(filepath="S&P 500 Historical Components & Changes (Updated).csv"):
    """Load and parse the historical S&P 500 constituents CSV."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')


def get_universe_from_csv(sp500_history):
    """Every ticker that ever appeared in the index, sorted."""
    all_tickers = set()
    for tickers_str in sp500_history['tickers'].dropna():
        # Handle the comma-separated strings
        all_tickers.update([t.strip() for t in tickers_str.split(',') if t.strip()])
    return sorted(list(all_tickers))


def build_membership_panel_from_csv(sp500_history, dates):
    """Point-in-time membership panel: merge_asof carries the last known
    composition forward onto each trading date."""
    history = sp500_history[['date', 'tickers']].dropna().copy()
    
    history['tickers_list'] = history['tickers'].apply(
        lambda x: [t.strip() for t in x.split(',') if t.strip()]
    )
    history = history.drop(columns=['tickers']).sort_values('date')
    grid_df = pd.DataFrame({'date': pd.DatetimeIndex(dates).sort_values()})
    aligned = pd.merge_asof(grid_df, history, on='date', direction='backward')
    aligned = aligned.dropna(subset=['tickers_list'])
    exploded = aligned.explode('tickers_list').rename(columns={'tickers_list': 'symbol'})
    exploded['in_index'] = 1
    
    return exploded[['date', 'symbol', 'in_index']].reset_index(drop=True)

def download_sp500_tables():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (research; contact@example.com)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))

    current = tables[0].copy()
    current.columns = [str(c).strip() for c in current.columns]
    current = current.rename(columns={"Symbol": "symbol", "CIK": "cik",
                                      "Security": "security",
                                      "GICS Sector": "gics_sector"})
    current["symbol"] = current["symbol"].str.replace(".", "-", regex=False)
    keep = [c for c in ["symbol", "cik", "security", "gics_sector"] if c in current.columns]
    current = current[keep]

    changes_raw = tables[1].copy()
    changes_raw.columns = ["_".join(str(x) for x in col).strip()
                           if isinstance(col, tuple) else str(col)
                           for col in changes_raw.columns]
    date_col = [c for c in changes_raw.columns if "Date" in c][0]
    added_col = [c for c in changes_raw.columns if "Added" in c and "Ticker" in c]
    removed_col = [c for c in changes_raw.columns if "Removed" in c and "Ticker" in c]
    added_col = added_col[0] if added_col else None
    removed_col = removed_col[0] if removed_col else None
    changes = pd.DataFrame({
        "date": pd.to_datetime(changes_raw[date_col], errors="coerce"),
        "added": changes_raw[added_col] if added_col else np.nan,
        "removed": changes_raw[removed_col] if removed_col else np.nan,
    })
    for col in ["added", "removed"]:
        changes[col] = (changes[col].astype(str).str.replace(".", "-", regex=False)
                        .replace({"nan": np.nan, "": np.nan}))
    changes = changes.dropna(subset=["date"]).sort_values("date")
    return current, changes


def survivorship_complete_universe(current, changes):
    """Current members plus every ticker ever added or removed — the full set
    we need prices for. Without the removed names we can't reconstruct a true
    PIT book since the membership filter can only drop names, not resurrect ones
    whose prices were never downloaded."""
    syms = set(current["symbol"].dropna())
    syms |= set(changes["added"].dropna())
    syms |= set(changes["removed"].dropna())
    return sorted(s for s in syms if isinstance(s, str) and s)


def membership_as_of(as_of, current_symbols, changes):
    as_of = pd.Timestamp(as_of)
    members = set(current_symbols)
    future = changes[changes["date"] > as_of]
    for _, row in future[::-1].iterrows():
        if pd.notna(row["added"]):
            members.discard(row["added"])
        if pd.notna(row["removed"]):
            members.add(row["removed"])
    return members


def build_membership_panel(current, changes, dates):
    rows = []
    cur = current["symbol"].tolist()
    for d in dates:
        for sym in membership_as_of(d, cur, changes):
            rows.append((d, sym))
    panel = pd.DataFrame(rows, columns=["date", "symbol"])
    panel["in_index"] = 1
    return panel


# --- FINRA daily short-sale volume (not short interest) ---------------------
FINRA_DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/{market}shvol{date}.txt"


def fetch_finra_day(date, session, market="CNMS"):
    url = FINRA_DAILY_URL.format(market=market,
                                 date=pd.Timestamp(date).strftime("%Y%m%d"))
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200 or "Access Denied" in r.text[:200]:
            return None
        df = pd.read_csv(StringIO(r.text), sep="|")
    except Exception:
        return None

    # FINRA files have a "Total" footer row — drop anything that isn't an 8-digit date
    df = df[df["Date"].astype(str).str.match(r"^\d{8}$")].copy()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.rename(columns={"Symbol": "symbol", "ShortVolume": "short_volume",
                            "ShortExemptVolume": "short_exempt_volume",
                            "TotalVolume": "total_volume_finra"})
    df["short_ratio"] = (df["short_volume"]
                         / df["total_volume_finra"].replace(0, np.nan))
    return df[["date", "symbol", "short_volume", "total_volume_finra", "short_ratio"]]


def download_finra_short_volume(start, end, symbols=None, market="CNMS",
                                pause=0.10, verbose=False):
    
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "research short-volume loader contact@example.com"})
    frames = []
    bdays = pd.bdate_range(start, end)
    for i, d in enumerate(bdays):
        day = fetch_finra_day(d, session, market=market)
        if day is None:
            continue
        if symbols is not None:
            day = day[day["symbol"].isin(set(symbols))]
        frames.append(day)
        if verbose and i % 100 == 0:
            print(f"  finra {i}/{len(bdays)} {d.date()} kept={len(day)}")
        time.sleep(pause)

    cols = ["date", "symbol", "short_volume", "total_volume_finra", "short_ratio"]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    g = out.groupby("symbol")["short_ratio"]
    out["short_ratio_z20"] = g.transform(
        lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std())
    out["short_ratio_chg5"] = g.transform(lambda s: s.diff(5))
    return out


# --- FINRA short interest (open position, bi-monthly) ----------------------
FINRA_SI_API = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"


def download_finra_short_interest(start, end, symbols=None, session=None,
                                  page_size=5000):
    
    session = session or requests.Session()
    payload = {
        "limit": page_size,
        "compareFilters": [
            {"fieldName": "settlementDate", "compareType": "gte", "fieldValue": str(start)},
            {"fieldName": "settlementDate", "compareType": "lte", "fieldValue": str(end)},
        ],
    }
    headers = {"Content-Type": "application/json",
               "Accept": "application/json",
               "User-Agent": "research short-interest loader contact@example.com"}
    try:
        r = session.post(FINRA_SI_API, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"  [short-interest] FINRA API returned {r.status_code}; "
                  f"see docstring for FTP/flat-file alternatives.")
            return pd.DataFrame(columns=["date", "symbol", "short_interest",
                                         "avg_daily_volume", "days_to_cover"])
        data = r.json()
    except Exception as e:
        print(f"  [short-interest] request failed ({e}); verify endpoint/auth.")
        return pd.DataFrame(columns=["date", "symbol", "short_interest",
                                     "avg_daily_volume", "days_to_cover"])

    df = pd.DataFrame(data)
    # column names vary across FINRA API versions — map defensively
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if "settlement" in cl and "date" in cl: ren[c] = "settlement_date"
        elif cl in ("symbolcode", "symbol", "issuesymbolidentifier"): ren[c] = "symbol"
        elif "currentshortpositionquantity" in cl or cl == "shortinterest": ren[c] = "short_interest"
        elif "averagedailyvolume" in cl: ren[c] = "avg_daily_volume"
        elif "daystocover" in cl: ren[c] = "days_to_cover"
        elif "disseminat" in cl and "date" in cl: ren[c] = "dissemination_date"
    df = df.rename(columns=ren)
    if "settlement_date" not in df:
        return pd.DataFrame(columns=["date", "symbol", "short_interest",
                                     "avg_daily_volume", "days_to_cover"])
    # PIT alignment: known on the dissemination date if available, otherwise
    # settlement + 2 bdays (the typical publication lag)
    if "dissemination_date" in df:
        df["date"] = pd.to_datetime(df["dissemination_date"], errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["settlement_date"], errors="coerce") \
                     + pd.offsets.BDay(2)
    if symbols is not None:
        df = df[df["symbol"].isin(set(symbols))]
    keep = [c for c in ["date", "symbol", "short_interest",
                        "avg_daily_volume", "days_to_cover"] if c in df.columns]
    return df[keep].dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])


def align_short_interest_pit(si_panel, daily_grid):
    """Carry each bi-monthly short-interest print forward from its dissemination
    date across the daily grid (merge_asof backward) so it's genuinely PIT."""
    if si_panel.empty:
        return pd.DataFrame(columns=["date", "symbol", "short_interest"])
    grid = pd.DatetimeIndex(sorted(set(daily_grid)))
    out = []
    for sym, g in si_panel.sort_values("date").groupby("symbol"):
        a = pd.merge_asof(pd.DataFrame({"date": grid}),
                          g.drop(columns="symbol"), on="date", direction="backward")
        a["symbol"] = sym
        out.append(a)
    res = pd.concat(out, ignore_index=True)
    if "short_interest" in res:
        res["short_interest_rank"] = res.groupby("date")["short_interest"].rank(pct=True)
    return res


# --- SEC EDGAR XBRL fundamentals (PIT via filing date) ---------------------
SEC_HEADERS = {"User-Agent": "research fundamentals loader contact@example.com"}
CONCEPT_TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}


def load_ticker_cik_map(session):
    url = "https://www.sec.gov/files/company_tickers.json"
    data = session.get(url, headers=SEC_HEADERS, timeout=30).json()
    return {row["ticker"].replace(".", "-"): f"{int(row['cik_str']):010d}"
            for row in data.values()}


def fetch_company_facts(cik10, session):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    r = session.get(url, headers=SEC_HEADERS, timeout=30)
    return r.json() if r.status_code == 200 else None


def _extract_first_available(facts, field):
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in CONCEPT_TAGS[field]:
        if tag in gaap:
            units = gaap[tag].get("units", {})
            unit_key = next((k for k in units if k in ("USD", "USD/shares")), None)
            if unit_key is None:
                continue
            rows = pd.DataFrame(units[unit_key])
            if rows.empty or "filed" not in rows:
                continue
            rows = rows.rename(columns={"val": field})
            rows["filed"] = pd.to_datetime(rows["filed"])
            rows["end"] = pd.to_datetime(rows["end"])
            rows = (rows.sort_values(["filed", "end"])
                        .drop_duplicates("filed", keep="last"))
            return rows[["filed", "end", field]]
    return pd.DataFrame(columns=["filed", "end", field])


def build_pit_fundamental_panel(tickers, daily_dates, pause=0.12):
    session = requests.Session()
    cik_map = load_ticker_cik_map(session)
    grid = pd.DatetimeIndex(sorted(set(daily_dates)))
    panels = []
    for tkr in tickers:
        cik = cik_map.get(tkr)
        if cik is None:
            continue
        facts = fetch_company_facts(cik, session)
        time.sleep(pause)
        if facts is None:
            continue
        merged = None
        for field in CONCEPT_TAGS:
            f = _extract_first_available(facts, field)
            if f.empty:
                continue
            f = f.drop(columns="end").sort_values("filed")
            merged = f if merged is None else pd.merge_asof(
                merged.sort_values("filed"), f, on="filed", direction="backward")
        if merged is None:
            continue
        merged = merged.sort_values("filed")
        aligned = pd.merge_asof(pd.DataFrame({"date": grid}),
                                merged.rename(columns={"filed": "date"}),
                                on="date", direction="backward")
        aligned["symbol"] = tkr
        panels.append(aligned)
    return pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()


# --- self-tests (no network) -----------------------------------------------
if __name__ == "__main__":
    # FINRA parser: a pipe-delimited file with a "Total" footer row should parse clean
    raw = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
           "20240102|AAPL|100|0|250|Q\n"
           "20240102|MSFT|200|0|500|Q\n"
           "Total|0|0|0|0|0\n")
    import requests as _rq
    class _Resp:
        status_code = 200
        text = raw
    class _Sess:
        def get(self, *a, **k): return _Resp()
    day = fetch_finra_day("2024-01-02", _Sess())
    assert day is not None and len(day) == 2, "FINRA parser dropped good rows"
    assert str(day["date"].dtype).startswith("datetime"), "date not parsed once"
    assert abs(day.loc[day.symbol == "AAPL", "short_ratio"].iloc[0] - 0.4) < 1e-9
    print(f"[selftest] FINRA parse fix: kept {len(day)} rows, "
          f"short_ratio(AAPL)=0.40  OK")

    # survivorship-complete universe must include a delisted (removed) name
    current = pd.DataFrame({"symbol": ["AAA", "BBB"]})
    changes = pd.DataFrame({"date": pd.to_datetime(["2023-06-01"]),
                            "added": ["BBB"], "removed": ["ZZZ"]})
    uni = survivorship_complete_universe(current, changes)
    assert "ZZZ" in uni, "delisted name missing from download universe"
    print(f"[selftest] survivorship-complete universe includes delisted ZZZ  OK")

    # membership reconstruction
    assert membership_as_of("2022-01-01", current["symbol"], changes) == {"AAA", "ZZZ"}
    print("[selftest] membership reconstruction  OK")

    # short-interest PIT alignment (carry forward from dissemination date)
    si = pd.DataFrame({"date": pd.to_datetime(["2024-02-15"]),
                       "symbol": ["AAA"], "short_interest": [1000.0]})
    grid = pd.to_datetime(["2024-02-10", "2024-02-20"])
    a = align_short_interest_pit(si, grid)
    assert pd.isna(a.loc[a.date == "2024-02-10", "short_interest"].iloc[0])
    assert a.loc[a.date == "2024-02-20", "short_interest"].iloc[0] == 1000.0
    print("[selftest] short-interest PIT alignment  OK")
