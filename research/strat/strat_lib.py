"""Shared loader and scoring for the 4 Nov 2026 strategy studies (research only; nothing here is served).

The owner's bar (4 Nov 2026): strategies that "hold up over large sample sizes, pass CHI tests, have positive BFSP
WAX - wins over BFSP expected wins - [and are] positive in the place market too - let's back both".

Every signal is read from runs on EARLIER DAYS than the race it is applied to. Each strategy is scored:

  WIN    W = winners, E = sum of 1/BSP, WAX = W - E, A/E = W / E,
         chi2 = Pearson goodness of fit of (winners, losers) against (E, N - E), 1 df, and its p
  PLACE  the same against the Betfair PLACE SP: P = placed (finish within the places Betfair paid),
         E_p = sum of 1/place BSP, PAX = P - E_p, A/E_pl, p_pl
  ROI    level stakes at BSP, win and place and both together (1 unit each), 5% commission on winnings;
         to-win staking: stake = TO_WIN / (price - 1) in each market, so a winner pays TO_WIN before commission;
         the morning win price (2018+, GBP50+ matched in the morning) beside the BSP
  YEARS  the test years whose win A/E and place A/E are both above 1.00

A non-finisher (fell, pulled up, unseated) has no numeric placing and is a LOSER in both markets -- the first pass
dropped them and flattered every jumps cut (chase A/E 1.2 across the board). A race counts only if it has a winner.

Periods: TRAIN = 2010-2021 (choose), TEST = 2022 to Nov 2026 (report).
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

COMMISSION = 0.05
TO_WIN = 100.0
TRAIN_END = "2021-12-31"


def load(db: str = "horse_racing.db") -> pd.DataFrame:
    c = sqlite3.connect(db)
    cols = {r[1] for r in c.execute("PRAGMA table_info(race_results)")}
    want = ["id", "race_date", "race_time", "track", "race_name", "race_type", "surface_type", "race_class",
            "number_of_runners", "horse_name", "trainer", "jockey_name", "jockeys_claim", "horse_age", "horse_sex",
            "official_rating", "placing_numerical", "place", "bfsp", "bfsp_place", "bf_plcs_paid", "odds",
            "days_since_lr", "career_runs", "stallion", "dam_stallion", "headgear", "dist_furlongs", "pounds",
            "major", "race_code"]
    sel = ", ".join(c_ for c_ in want if c_ in cols)
    df = pd.read_sql_query(f"SELECT {sel} FROM race_results", c)
    tabs = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "betfair_prices" in tabs:
        bp = pd.read_sql_query("""SELECT race_results_id AS id, market_type, bsp, morningwap, morning_vol
                                  FROM betfair_prices WHERE race_results_id IS NOT NULL""", c)
        w = bp[bp.market_type == "win"].drop_duplicates("id").drop(columns="market_type")
        w.columns = ["id", "bf_bsp", "morningwap", "morning_vol"]
        p = bp[bp.market_type == "place"].drop_duplicates("id").drop(columns="market_type")
        p.columns = ["id", "bf_pbsp", "p_morningwap", "p_morning_vol"]
        df = df.merge(w, on="id", how="left").merge(p, on="id", how="left")
    c.close()
    for k in ("official_rating", "placing_numerical", "bfsp", "bfsp_place", "bf_plcs_paid", "odds", "days_since_lr",
              "career_runs", "horse_age", "jockeys_claim", "number_of_runners", "dist_furlongs", "bf_bsp",
              "morningwap", "morning_vol", "bf_pbsp", "p_morningwap", "p_morning_vol"):
        if k in df:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    df["date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    rt = df["race_type"].fillna("").str.lower()
    df["code"] = np.select([rt.str.contains("chase"), rt.str.contains("hurdle"), rt.str.contains("nh flat|bumper"),
                            df["race_code"].fillna("").str.lower().str.contains("all weather")],
                           ["chase", "hurdle", "nhflat", "aw"], default="flat")
    df["rk"] = df["race_date"].astype(str) + "|" + df["track"].astype(str) + "|" + df["race_time"].astype(str)
    df["bsp"] = df["bfsp"].where(df["bfsp"] > 1, df.get("bf_bsp"))
    df["pbsp"] = df["bfsp_place"].where(df["bfsp_place"] > 1, df.get("bf_pbsp"))
    df["fin"] = df["placing_numerical"]
    df["won"] = (df["fin"] == 1).astype(float)                  # a non-finisher is a loser
    df["placed"] = ((df["fin"] <= df["bf_plcs_paid"]) & (df["bf_plcs_paid"] > 0)).astype(float)
    df["hcap"] = df["race_name"].fillna("").str.contains("handicap", case=False)
    df["year"] = df["date"].dt.year
    df = df.sort_values(["date", "race_time", "track"]).reset_index(drop=True)
    df["bsp_rank"] = df.groupby("rk")["bsp"].rank(method="min")
    df["field"] = df.groupby("rk")["horse_name"].transform("size")
    df["race_has_winner"] = df.groupby("rk")["won"].transform("sum") >= 1
    return df


def priced(df: pd.DataFrame) -> pd.DataFrame:
    """Runners the market priced (BSP > 1) in races with a winner. Non-finishers stay in, as losers."""
    return df[(df["bsp"] > 1) & df["race_has_winner"]]


def _gof(o: float, e: float, n: int) -> tuple[float, float]:
    """Pearson chi-square of (o, n-o) against (e, n-e), 1 df."""
    if e <= 0 or n - e <= 0:
        return np.nan, np.nan
    x = (o - e) ** 2 / e + (o - e) ** 2 / (n - e)
    return x, float(_chi2.sf(x, 1))


def _pnl(win: pd.Series, price: pd.Series, to_win: float) -> tuple[float, float, float]:
    """(level P&L per unit, to-win P&L in GBP, to-win stake in GBP)."""
    lvl = np.where(win == 1, (price - 1) * (1 - COMMISSION), -1.0).sum()
    st = to_win / (price - 1)
    tw = np.where(win == 1, to_win * (1 - COMMISSION), -st).sum()
    return float(lvl), float(tw), float(st.sum())


def score(d: pd.DataFrame, to_win: float = TO_WIN) -> dict:
    d = d[d["bsp"] > 1]
    n = len(d)
    if n == 0:
        return {"n": 0}
    p = 1.0 / d["bsp"]
    W, E = float(d["won"].sum()), float(p.sum())
    x, pv = _gof(W, E, n)
    lvl, tw, tws = _pnl(d["won"], d["bsp"], to_win)
    out = {"n": n, "wins": int(W), "sr%": 100 * W / n, "WAX": W - E, "A/E": W / E, "chi2": x, "p": pv,
           "roi_bsp%": 100 * lvl / n, "tw_roi%": 100 * tw / tws, "tw_pnl": tw}
    q = d[d["pbsp"] > 1]
    if len(q) >= 30:
        pp = 1.0 / q["pbsp"]
        P, Ep = float(q["placed"].sum()), float(pp.sum())
        xp, pvp = _gof(P, Ep, len(q))
        plvl, ptw, ptws = _pnl(q["placed"], q["pbsp"], to_win)
        wl, wtw, wtws = _pnl(q["won"], q["bsp"], to_win)
        out.update({"n_pl": len(q), "PAX": P - Ep, "A/E_pl": P / Ep, "chi2_pl": xp, "p_pl": pvp,
                    "roi_pl%": 100 * plvl / len(q),
                    "roi_both%": 100 * (wl + plvl) / (2 * len(q)),
                    "tw_both_pnl": wtw + ptw, "tw_both_roi%": 100 * (wtw + ptw) / (wtws + ptws)})
    m = d[(d.get("morningwap", pd.Series(np.nan, index=d.index)) > 1) & (d.get("morning_vol", 0) >= 50)]
    if len(m) >= 30:
        ml, mtw, mts = _pnl(m["won"], m["morningwap"], to_win)
        out.update({"n_am": len(m), "roi_am%": 100 * ml / len(m), "tw_roi_am%": 100 * mtw / mts,
                    "clv%": 100 * (m["morningwap"] / m["bsp"] - 1).mean()})
    return out


KEEP = ["n", "WAX", "A/E", "p", "A/E_pl", "p_pl", "roi_bsp%", "roi_pl%", "roi_both%", "tw_both_roi%", "roi_am%", "clv%"]
PCOLS = {"p", "p_pl", "t_p", "t_p_pl"}


def table(d: pd.DataFrame, by, min_n: int = 100) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(by, observed=True, dropna=False):
        s = score(g)
        if s.get("n", 0) >= min_n:
            key = dict(zip(by, k if isinstance(k, tuple) else (k,))) if isinstance(by, list) else {"key": k}
            rows.append({**key, **s})
    return pd.DataFrame(rows)


def split(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return d[d["race_date"] <= TRAIN_END], d[d["race_date"] > TRAIN_END]


def _fmt(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    for c in t.columns:
        if c in PCOLS:
            t[c] = t[c].map(lambda v: (f"{v:.0e}" if v < 0.001 else f"{v:.3f}") if pd.notna(v) else "")
    return t.round(2)


def show(title: str, t: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(_fmt(t).to_string(index=False) if len(t) else "(none)")


def tt(d: pd.DataFrame, by, min_n: int = 100) -> pd.DataFrame:
    """The same cut on TRAIN and TEST side by side (TEST columns prefixed t_)."""
    a, b = split(d)
    ta, tb = table(a, by, min_n), table(b, by, max(30, min_n // 3))
    if ta.empty:
        return tb
    keys = [c for c in ta.columns if c in (by if isinstance(by, list) else ["key"])]
    ta = ta[keys + [c for c in KEEP if c in ta]]
    if tb.empty:
        return ta
    tb = tb[keys + [c for c in KEEP if c in tb]].rename(columns={c: "t_" + c for c in KEEP})
    return ta.merge(tb, on=keys, how="outer", sort=False)


def rule(name: str, d: pd.DataFrame, mask) -> dict:
    """One rule, scored on TRAIN and TEST, with the TEST years."""
    s = d[mask]
    a, b = split(s)
    sa, sb = score(a), score(b)
    out = {"rule": name}
    for pre, sc in (("", sa), ("t_", sb)):
        for k in ("n", "WAX", "A/E", "p", "A/E_pl", "p_pl", "roi_bsp%", "roi_pl%", "roi_both%", "tw_both_roi%",
                  "tw_both_pnl", "roi_am%"):
            out[pre + k] = sc.get(k, np.nan)
    yrs = []
    for y, g in b.groupby("year"):
        if len(g) >= 50:
            s_ = score(g)
            yrs.append((int(y), s_.get("A/E", np.nan), s_.get("A/E_pl", np.nan)))
    out["t_years_both>1"] = f"{sum(1 for _, w, p in yrs if w > 1 and p > 1)}/{len(yrs)}"
    out["t_years"] = " ".join(f"{y % 100}:{w:.2f}/{p:.2f}" for y, w, p in yrs)
    return out


def gate(r: dict) -> str:
    """The owner's bar, fixed before the TEST figures are read.

    TRAIN: n >= 2,000; win A/E > 1 with chi2 p < 0.001; place A/E > 1 with chi2 p < 0.01.
    TEST : n >= 500; WAX > 0 with chi2 p < 0.05; place A/E > 1; win+place level ROI at BSP > 0 after 5% commission;
           win and place A/E both above 1 in at least 3 of the test years.
    """
    try:
        ok_train = (r["n"] >= 2000 and r["A/E"] > 1 and r["p"] < 0.001 and r["A/E_pl"] > 1 and r["p_pl"] < 0.01)
        if not ok_train:
            return "-"
        yb, _ = (int(x) for x in r["t_years_both>1"].split("/"))
        ok_test = (r["t_n"] >= 500 and r["t_WAX"] > 0 and r["t_p"] < 0.05 and r["t_A/E_pl"] > 1
                   and r["t_roi_both%"] > 0 and yb >= 3)
        return "PASS" if ok_test else "train only"
    except (TypeError, KeyError):
        return "-"


def rules_table(rows: list[dict], sort: str | None = None) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    t.insert(1, "gate", [gate(r) for r in rows])
    if sort:
        t = t.sort_values(sort, ascending=False)
    with pd.option_context("display.max_colwidth", 90):
        print(_fmt(t).to_string(index=False))
    return t


def prior_cumsum(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Per key, the running totals of cols over runs on EARLIER DAYS (same-day runs excluded)."""
    g = df[[key, "date"] + cols].copy()
    day = g.groupby([key, "date"], sort=False)[cols].sum()
    cum = day.groupby(level=0).cumsum() - day
    out = g[[key, "date"]].merge(cum.reset_index(), on=[key, "date"], how="left")
    out.index = df.index
    return out[cols]
