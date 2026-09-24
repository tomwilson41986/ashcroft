"""What the race's OTHER markets say about each runner's chance of winning.

Benter combined his model with the public's win odds. The public prices a race
more than once: the Betfair win market (BSP), the Betfair place market (place
BSP) and the bookmakers (industry SP). Ziemba and Hausch made a system of the
place pools disagreeing with the win pool; the same disagreement is a signal
about the win probabilities themselves. All three prices are struck at the off,
so in a backtest they stand in for the prices visible a minute before it, which
is when a live version of this model would run.

Features (all Stage C: functions of this race's prices, never of its result)
    mkt_ln_pi_bsp     ln of the race-normalised BSP probability; lets a tree
                      learn a favourite-longshot bias that varies by context
    mkt_bsp_rank      price rank in the race (1 = favourite)
    mkt_ln_pi_sp      ln of the race-normalised industry-SP probability
    mkt_sp_vs_bsp     mkt_ln_pi_sp - mkt_ln_pi_bsp: bookmakers shorter than the exchange
    mkt_ln_pi_place   ln of the place market's probability (normalised to the places paid)
    mkt_place_vs_win  mkt_ln_pi_place - ln of the place chance the WIN market
                      implies (Benter's discounted Harville ordering)
    mkt_sp_book       the industry-SP book (race level: an interaction input)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.ordering import place_probabilities

#: Benter/Lo-Bacon-Shone discount exponents fitted on UK/IRE Flat (RESEARCH_FRAMEWORK §10.5).
ORDER_GAMMA, ORDER_DELTA = 0.746, 0.665

MARKET_BLOCK_FEATURES = ["mkt_ln_pi_bsp", "mkt_bsp_rank", "mkt_ln_pi_sp", "mkt_sp_vs_bsp",
                         "mkt_ln_pi_place", "mkt_place_vs_win", "mkt_sp_book"]


def _race_key(df: pd.DataFrame) -> pd.Series:
    if "raceid" in df.columns:
        return df["raceid"].astype(str)
    return (pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d") + "|" + df["track"].astype(str)
            + "|" + df["race_time"].astype(str))


def _normalised(inv: pd.Series, race: pd.Series, complete: pd.Series) -> pd.Series:
    """inv / race total, NaN for races where any runner lacks a price."""
    tot = inv.groupby(race).transform("sum")
    return (inv / tot).where(complete)


def add_same_race_market_features(df: pd.DataFrame, bsp_col: str = "bfsp", sp_col: str = "odds",
                                  place_col: str = "bfsp_place", places_cols=("bf_plcs_paid", "plcs_paid"),
                                  gamma: float = ORDER_GAMMA, delta: float = ORDER_DELTA) -> tuple[pd.DataFrame, list[str]]:
    race = _race_key(df)
    out = pd.DataFrame(index=df.index)

    bsp = pd.to_numeric(df.get(bsp_col), errors="coerce")
    ok_b = (bsp > 1).groupby(race).transform("all")
    pi_b = _normalised(1.0 / bsp.where(bsp > 1), race, ok_b)
    out["mkt_ln_pi_bsp"] = np.log(pi_b)
    out["mkt_bsp_rank"] = bsp.where(ok_b).groupby(race).rank(method="min")

    if sp_col in df.columns:
        sp = pd.to_numeric(df[sp_col], errors="coerce")
        ok_s = (sp > 1).groupby(race).transform("all")
        inv_s = 1.0 / sp.where(sp > 1)
        out["mkt_ln_pi_sp"] = np.log(_normalised(inv_s, race, ok_s))
        out["mkt_sp_vs_bsp"] = out["mkt_ln_pi_sp"] - out["mkt_ln_pi_bsp"]
        out["mkt_sp_book"] = inv_s.groupby(race).transform("sum").where(ok_s)
    else:
        out[["mkt_ln_pi_sp", "mkt_sp_vs_bsp", "mkt_sp_book"]] = np.nan

    out["mkt_ln_pi_place"] = np.nan
    out["mkt_place_vs_win"] = np.nan
    if place_col in df.columns:
        pl = pd.to_numeric(df[place_col], errors="coerce")
        places = pd.Series(np.nan, index=df.index)
        for c in places_cols:
            if c in df.columns:
                places = places.fillna(pd.to_numeric(df[c], errors="coerce"))
        places = places.groupby(race).transform("max")
        ok_p = (pl > 1).groupby(race).transform("all") & ok_b & places.between(1, 4)
        inv_p = 1.0 / pl.where(pl > 1)
        pi_place = (inv_p / inv_p.groupby(race).transform("sum") * places).where(ok_p).clip(upper=0.999)
        out["mkt_ln_pi_place"] = np.log(pi_place)
        # the place chance the win market implies, race by race
        implied = pd.Series(np.nan, index=df.index)
        rows = np.flatnonzero(ok_p.to_numpy())
        if len(rows):
            sub = pd.DataFrame({"race": race.to_numpy()[rows], "p": pi_b.to_numpy()[rows],
                                "k": places.to_numpy()[rows]}, index=df.index[rows])
            rng = np.random.default_rng(0)
            vals = []
            for _, g in sub.groupby("race", sort=False):
                p = g["p"].to_numpy(float)
                k = int(g["k"].iloc[0])
                if k >= len(p):
                    vals.append(pd.Series(np.ones(len(p)), index=g.index))
                    continue
                pp = place_probabilities(p / p.sum(), k, gamma, delta, n_sims=4000, rng=rng)
                vals.append(pd.Series(np.clip(pp, 1e-6, 0.999), index=g.index))
            implied.loc[pd.concat(vals).index] = pd.concat(vals).to_numpy()
        out["mkt_place_vs_win"] = out["mkt_ln_pi_place"] - np.log(implied)

    res = df.copy()
    for c in MARKET_BLOCK_FEATURES:
        res[c] = out[c]
    return res, list(MARKET_BLOCK_FEATURES)
