"""Shared loader and scoring for the 4 Nov 2026 strategy studies (research only; nothing here is served).

Every signal is read from runs on EARLIER DAYS than the race it is applied to, so a 06:00 card would give the same
value. Each strategy is scored against the Betfair SP (the market's verdict) and, where the day file holds one, the
morning volume-weighted price (a price that was actually available before the off):

    A/E        winners / sum of 1/BSP  (above 1.00 = the horses won more often than the market said)
    z          (winners - expected) / sqrt(sum p(1-p)), the size of that against chance
    ROI        net P&L / stake, 5% commission on winnings, at BSP and at the morning price
    to-win     the owner's staking: stake = TO_WIN / (price - 1), so every winner pays TO_WIN before commission

Periods: TRAIN = up to the end of TRAIN_END, TEST = after it. A rule is chosen on TRAIN and only reported on TEST.
"""

from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

COMMISSION = 0.05
TO_WIN = 100.0
TRAIN_END = "2021-12-31"   # 2010-2021 to choose, 2022 to Nov 2026 to test


def load(db: str = "horse_racing.db") -> pd.DataFrame:
    c = sqlite3.connect(db)
    cols = {r[1] for r in c.execute("PRAGMA table_info(race_results)")}
    want = ["id", "race_date", "race_time", "track", "race_name", "race_type", "surface_type", "race_class",
            "number_of_runners", "horse_name", "trainer", "jockey_name", "jockeys_claim", "horse_age", "horse_sex",
            "official_rating", "placing_numerical", "bfsp", "odds", "days_since_lr", "career_runs", "stallion",
            "dam_stallion", "headgear", "dist_furlongs", "pounds", "major", "race_code"]
    sel = ", ".join(c_ for c_ in want if c_ in cols)
    df = pd.read_sql_query(f"SELECT {sel} FROM race_results", c)
    tabs = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "betfair_prices" in tabs:
        bp = pd.read_sql_query("""SELECT race_results_id AS id, bsp AS bf_bsp, morningwap, morning_vol, ppwap
                                  FROM betfair_prices WHERE market_type='win' AND race_results_id IS NOT NULL""", c)
        bp = bp.drop_duplicates("id")
        df = df.merge(bp, on="id", how="left")
    c.close()
    for k in ("official_rating", "placing_numerical", "bfsp", "odds", "days_since_lr", "career_runs", "horse_age",
              "jockeys_claim", "number_of_runners", "dist_furlongs", "bf_bsp", "morningwap", "morning_vol", "ppwap"):
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
    df["won"] = (df["placing_numerical"] == 1).astype(float)
    df["fin"] = df["placing_numerical"]
    df["hcap"] = df["race_name"].fillna("").str.contains("handicap", case=False)
    df["year"] = df["date"].dt.year
    df = df.sort_values(["date", "race_time", "track"]).reset_index(drop=True)
    # the market's rank inside the race
    df["bsp_rank"] = df.groupby("rk")["bsp"].rank(method="min")
    df["field"] = df.groupby("rk")["horse_name"].transform("size")
    return df


def priced(df: pd.DataFrame) -> pd.DataFrame:
    """Runners the market priced (BSP > 1) in races with a result."""
    return df[(df["bsp"] > 1) & df["fin"].notna()]


def score(d: pd.DataFrame, to_win: float = TO_WIN) -> dict:
    d = d[(d["bsp"] > 1)]
    n = len(d)
    if n == 0:
        return {"n": 0}
    p = 1.0 / d["bsp"]
    w = d["won"]
    exp = p.sum()
    var = (p * (1 - p)).sum()
    out = {"n": n, "wins": int(w.sum()), "sr%": 100 * w.mean(), "A/E": w.sum() / exp if exp else np.nan,
           "z": (w.sum() - exp) / np.sqrt(var) if var else np.nan}
    # level stakes at BSP
    pnl = np.where(w == 1, (d["bsp"] - 1) * (1 - COMMISSION), -1.0)
    out["roi_bsp%"] = 100 * pnl.sum() / n
    # to win TO_WIN at BSP
    st = to_win / (d["bsp"] - 1)
    tw = np.where(w == 1, to_win * (1 - COMMISSION), -st)
    out["tw_pnl"] = tw.sum()
    out["tw_roi%"] = 100 * tw.sum() / st.sum()
    # at the morning price, where held with some volume
    m = d[(d.get("morningwap", pd.Series(np.nan, index=d.index)) > 1) & (d.get("morning_vol", 0) >= 50)]
    if len(m) >= 30:
        pm = np.where(m["won"] == 1, (m["morningwap"] - 1) * (1 - COMMISSION), -1.0)
        stm = to_win / (m["morningwap"] - 1)
        twm = np.where(m["won"] == 1, to_win * (1 - COMMISSION), -stm)
        out["n_am"] = len(m)
        out["roi_am%"] = 100 * pm.sum() / len(m)
        out["tw_roi_am%"] = 100 * twm.sum() / stm.sum()
        out["clv%"] = 100 * (m["morningwap"] / m["bsp"] - 1).mean()
    return out


def table(d: pd.DataFrame, by, min_n: int = 100) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(by, observed=True, dropna=False):
        s = score(g)
        if s.get("n", 0) >= min_n:
            rows.append({**({"key": k} if not isinstance(by, list) else dict(zip(by, k if isinstance(k, tuple) else (k,)))), **s})
    t = pd.DataFrame(rows)
    return t.round(2)


def split(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return d[d["race_date"] <= TRAIN_END], d[d["race_date"] > TRAIN_END]


def show(title: str, t: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(t.to_string(index=False) if len(t) else "(none)")


def prior_cumsum(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Per key, the running totals of cols over runs on EARLIER DAYS (same-day runs excluded)."""
    g = df[[key, "date"] + cols].copy()
    day = g.groupby([key, "date"], sort=False)[cols].sum()
    cum = day.groupby(level=0).cumsum() - day  # totals before this day
    out = g[[key, "date"]].merge(cum.reset_index(), on=[key, "date"], how="left")
    out.index = df.index
    return out[cols]


KEEP = ["n", "sr%", "A/E", "z", "roi_bsp%", "tw_roi%", "tw_pnl", "roi_am%", "tw_roi_am%", "clv%"]


def tt(d: pd.DataFrame, by, min_n: int = 100) -> pd.DataFrame:
    """The same cut on TRAIN and TEST side by side (TEST columns prefixed t_)."""
    a, b = split(d)
    ta, tb = table(a, by, min_n), table(b, by, min(min_n, 50))
    if ta.empty:
        return tb
    keys = [c for c in ta.columns if c not in KEEP + ["wins", "n_am"]]
    ta = ta[keys + [c for c in KEEP if c in ta]]
    if tb.empty:
        return ta
    tb = tb[keys + [c for c in KEEP if c in tb]].rename(columns={c: "t_" + c for c in KEEP})
    return ta.merge(tb, on=keys, how="outer", sort=False)


def rule(name: str, d: pd.DataFrame, mask) -> dict:
    """One rule, scored on TRAIN and TEST, with TEST by year (stability)."""
    s = d[mask]
    a, b = split(s)
    sa, sb = score(a), score(b)
    yrs = {int(y): round(score(g).get("A/E", np.nan), 2) for y, g in b.groupby("year") if len(g) >= 30}
    out = {"rule": name}
    for pre, sc in (("", sa), ("t_", sb)):
        for k in ("n", "A/E", "z", "roi_bsp%", "tw_roi%", "tw_pnl", "roi_am%", "clv%"):
            out[pre + k] = round(sc.get(k, np.nan), 2) if isinstance(sc.get(k), float) else sc.get(k)
    out["test A/E by year"] = yrs
    return out


def rules_table(rows: list[dict]) -> None:
    t = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 120):
        print(t.to_string(index=False))
