"""Place market against win market (the owner, 4 Nov 2026: "keep working in loops until you define a true edge and
profitable strategy"). Read-only research.

The question: is a GB greyhound place (to-be-placed) market's Betfair SP mispriced against the chance of placing that
the win market implies? A bet at the SP is matched at the SP whatever the queue, so a rule here is scored at the price
it would really have got: the fill question that sank the early lay's paper results does not arise.

Data: Betfair's greyhound price files (sources/greyhound_prices.py, ``greyhound_prices/prices_<year>.parquet``): each
runner's win and place SP, pre-play weighted average price (PPWAP), morning WAP, volumes, and the result (WIN_LOSE: won
the win market, placed in the place market). Win and place markets are joined by day, time, track and trap.

The chance of placing, estimated at the off from what is known before it:
- ``harville``: Harville's formula on the win market's PPWAP, normalised (the textbook baseline; known to rate the
  favourites' places too high);
- ``gbm``: LightGBM on the win and place markets' pre-play prices and volumes, Harville's chance, the field, the trap,
  the track, refitted each year on the years before only (walk-forward).
The win and place PPWAPs include trades up to the off, which a bet struck a few seconds before it would not quite see:
read the results as slightly optimistic on that count. ``--oracle`` adds the win market's SP in place of its PPWAP (a
look-ahead upper bound, never a rule).

The rules, on the test years only: back at the place SP where p x (SP - 1) x (1 - c) - (1 - p) clears a threshold, lay
where (1 - p) x (1 - c) - p x (SP - 1) does (per unit of stake; lays reported per unit of liability too); commission c
on the winnings of each bet (2% and 5%). Each with its return and a 90% interval by a bootstrap over days.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KEY = ["race_date", "race_time", "track", "trap"]
COLS = ["market", "race_date", "race_time", "track", "trap", "country", "bsp", "ppwap", "morningwap", "pp_vol",
        "morning_vol", "win_lose"]
MAXN = 8
THRESHOLDS = (0.0, 0.02, 0.05, 0.10, 0.20)
COMMISSIONS = (0.02, 0.05)
GBM = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 200,
       "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
       "seed": 0, "num_threads": 4}
ROUNDS = 400


def load(store, years: range) -> pd.DataFrame:
    """GB win and place runners, a row a runner with both markets' prices (``_w`` and ``_p``)."""
    parts = []
    for y in years:
        t = store.get_parquet(f"greyhound_prices/prices_{y}.parquet")
        if t is None or not len(t):
            log.info("%d: no table", y)
            continue
        t = t[t.country == "GB"][[c for c in COLS if c in t.columns]]
        parts.append(t)
        log.info("%d: %d GB runner rows (win %d, place %d)", y, len(t), int((t.market == "win").sum()),
                 int((t.market == "place").sum()))
    d = pd.concat(parts, ignore_index=True)
    d = d.dropna(subset=["trap"]).drop_duplicates(["market"] + KEY, keep="last")
    vals = ["bsp", "ppwap", "morningwap", "pp_vol", "morning_vol", "win_lose"]
    w = d[d.market == "win"][KEY + vals].rename(columns={c: f"{c}_w" for c in vals})
    p = d[d.market == "place"][KEY + vals].rename(columns={c: f"{c}_p" for c in vals})
    j = w.merge(p, on=KEY, how="inner")
    log.info("joined: %d runners (win rows %d, place rows %d)", len(j), len(w), len(p))
    return j


def clean(j: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Races whose win and place markets are whole: one winner, two or three placed, every runner priced."""
    j = j.copy()
    j["race"] = j.race_date + " " + j.race_time.astype(str) + " " + j.track
    g = j.groupby("race")
    j["n"] = g.trap.transform("size")
    j["winners"] = g.win_lose_w.transform("sum")
    j["places"] = g.win_lose_p.transform("sum")
    j["priced"] = g.ppwap_w.transform(lambda s: s.gt(1).all()) & g.bsp_p.transform(lambda s: s.gt(1).all())
    why = {"races": int(j.race.nunique()),
           "not one winner": int(j[j.winners != 1].race.nunique()),
           "not 2-3 placed": int(j[~j.places.isin([2, 3])].race.nunique()),
           "field outside 4-8": int(j[~j.n.between(4, MAXN)].race.nunique()),
           "a runner without a win PPWAP or place SP": int(j[~j.priced].race.nunique())}
    ok = (j.winners == 1) & j.places.isin([2, 3]) & j.n.between(4, MAXN) & j.priced & (j.places < j.n)
    j = j[ok].copy()
    why["kept races"] = int(j.race.nunique())
    why["kept runners"] = len(j)
    return j, why


def harville(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """P(in the first k) by Harville for each race's win chances ``q`` (races x MAXN, zero padded), k 2 or 3."""
    R, N = q.shape
    eye = np.eye(N, dtype=bool)
    p1 = q
    # second: sum_j q_j * q_i / (1 - q_j)
    den1 = np.clip(1 - q, 1e-9, None)
    a = q[:, :, None] * q[:, None, :] / den1[:, :, None]          # [r, j, i]: j first, i second
    a[:, eye] = 0
    p2 = a.sum(1)
    p3 = np.zeros_like(q)
    if (k == 3).any():
        # third: sum_{j,l} q_j q_l / (1 - q_j) * q_i / (1 - q_j - q_l)
        qj, ql, qi = q[:, :, None, None], q[:, None, :, None], q[:, None, None, :]
        den2 = np.clip(1 - qj - ql, 1e-9, None)
        b = qj * ql / den1[:, :, None, None] * qi / den2
        jj, ll, ii = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
        b[:, (jj == ll) | (jj == ii) | (ll == ii)] = 0
        p3 = b.sum((1, 2))
    return np.clip(p1 + p2 + np.where(k[:, None] == 3, p3, 0.0), 0, 1)


def features(j: pd.DataFrame, oracle: bool = False) -> tuple[pd.DataFrame, list[str]]:
    j = j.sort_values(["race", "trap"]).copy()
    races = j.race.to_numpy()
    codes, idx = np.unique(races, return_inverse=True)
    pos = j.groupby("race").cumcount().to_numpy()
    price_w = j.bsp_w if oracle else j.ppwap_w
    inv = 1.0 / price_w.where(price_w > 1)
    j["q_w"] = inv / inv.groupby(j.race).transform("sum")
    Q = np.zeros((len(codes), MAXN))
    Q[idx, pos] = j.q_w.fillna(0).to_numpy()
    K = j.groupby("race").places.first().reindex(codes).to_numpy()
    H = harville(Q, K)
    j["harville"] = H[idx, pos]
    invm = 1.0 / j.morningwap_w.where(j.morningwap_w > 1)
    j["q_morning"] = invm / invm.groupby(j.race).transform("sum")
    invp = 1.0 / j.ppwap_p.where(j.ppwap_p > 1)
    j["q_place_pp"] = invp / invp.groupby(j.race).transform("sum") * j.places
    j["rank_w"] = j.groupby("race").q_w.rank(ascending=False, method="first")
    j["log_w"] = np.log(price_w)
    j["log_pp_p"] = np.log(j.ppwap_p.where(j.ppwap_p > 1))
    j["log_vol_w"] = np.log1p(j.pp_vol_w)
    j["log_vol_p"] = np.log1p(j.pp_vol_p)
    j["vol_ratio"] = j.log_vol_p - j.log_vol_w
    j["drift_w"] = np.log(j.morningwap_w.where(j.morningwap_w > 1)) - j.log_w
    j["track_code"] = j.track.astype("category").cat.codes
    j["month"] = pd.to_datetime(j.race_date).dt.month
    j["year"] = pd.to_datetime(j.race_date).dt.year
    feats = ["q_w", "harville", "q_morning", "q_place_pp", "rank_w", "log_w", "log_pp_p", "log_vol_w", "log_vol_p",
             "vol_ratio", "drift_w", "n", "places", "trap", "track_code", "month"]
    return j, feats


def walk_forward(j: pd.DataFrame, feats: list[str], first_test: int) -> pd.DataFrame:
    import lightgbm as lgb
    j = j.copy()
    j["p_gbm"] = np.nan
    for y in sorted(j.year.unique()):
        if y < first_test:
            continue
        tr, te = j.year < y, j.year == y
        if tr.sum() < 100_000:
            continue
        m = lgb.train(GBM, lgb.Dataset(j.loc[tr, feats], j.loc[tr, "win_lose_p"]), ROUNDS)
        j.loc[te, "p_gbm"] = m.predict(j.loc[te, feats])
        log.info("%d: fitted on %d runners, scored %d", y, int(tr.sum()), int(te.sum()))
    return j


def _boot(day_pnl: pd.Series, day_stake: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    p, s = day_pnl.to_numpy(), day_stake.to_numpy()
    k = len(p)
    if k < 5:
        return np.nan, np.nan
    i = rng.integers(0, k, size=(n, k))
    r = p[i].sum(1) / np.clip(s[i].sum(1), 1e-9, None)
    return float(np.percentile(r, 5)), float(np.percentile(r, 95))


def logloss(y, p) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(t: pd.DataFrame, pcol: str) -> list[dict]:
    """Every rule's bets on the test rows ``t`` with chance column ``pcol``."""
    out = []
    y = t.win_lose_p.to_numpy(float)
    sp = t.bsp_p.to_numpy(float)
    p = t[pcol].to_numpy(float)
    for c in COMMISSIONS:
        ev_back = p * (sp - 1) * (1 - c) - (1 - p)
        ev_lay = (1 - p) * (1 - c) - p * (sp - 1)
        pnl_back = np.where(y == 1, (sp - 1) * (1 - c), -1.0)            # a unit stake
        pnl_lay = np.where(y == 1, -(sp - 1), 1 - c)                        # a unit stake (backer's)
        for side, ev, pnl, risk in (("back", ev_back, pnl_back, np.ones_like(sp)),
                                    ("lay", ev_lay, pnl_lay, sp - 1)):
            for th in THRESHOLDS:
                sel = np.isfinite(ev) & (ev > th)
                if sel.sum() < 30:
                    out.append({"p": pcol, "comm": c, "side": side, "edge>": th, "bets": int(sel.sum())})
                    continue
                d = pd.DataFrame({"day": t.race_date.to_numpy()[sel], "pnl": pnl[sel], "stake": np.ones(sel.sum()),
                                  "risk": risk[sel]})
                byd = d.groupby("day")[["pnl", "stake", "risk"]].sum()
                lo, hi = _boot(byd.pnl, byd.stake)
                out.append({"p": pcol, "comm": c, "side": side, "edge>": th, "bets": int(sel.sum()),
                            "days": len(byd), "mean_sp": round(float(sp[sel].mean()), 2),
                            "roi_stake%": round(100 * d.pnl.sum() / d.stake.sum(), 2),
                            "ci90_stake%": [round(100 * lo, 2), round(100 * hi, 2)],
                            "roi_risk%": round(100 * d.pnl.sum() / d.risk.sum(), 2),
                            "pnl_units": round(float(d.pnl.sum()), 1)})
    return out


def calibration(t: pd.DataFrame) -> pd.DataFrame:
    """The place SP against how often its runners placed, by the SP's implied chance."""
    imp = 1.0 / t.bsp_p
    b = pd.cut(imp, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    g = t.groupby(b, observed=True)
    out = pd.DataFrame({"runners": g.size(), "implied": g.apply(lambda x: (1 / x.bsp_p).mean()),
                        "placed": g.win_lose_p.mean(),
                        "back_roi_2%": g.apply(lambda x: np.where(x.win_lose_p == 1, (x.bsp_p - 1) * 0.98, -1).mean()),
                        "lay_roi_2%": g.apply(lambda x: np.where(x.win_lose_p == 1, -(x.bsp_p - 1), 0.98).mean())})
    return out.round(4)


def run(store, years: range, first_test: int, oracle: bool = False) -> dict:
    j = load(store, years)
    j, why = clean(j)
    log.info("clean: %s", why)
    j, feats = features(j, oracle=oracle)
    j = walk_forward(j, feats, first_test)
    t = j[j.p_gbm.notna()].copy()
    rep = {"data": why, "test_years": sorted(int(y) for y in t.year.unique()), "test_runners": len(t),
           "test_races": int(t.race.nunique()), "oracle": oracle,
           "log_loss": {"place_sp": logloss(t.win_lose_p, np.clip(1 / t.bsp_p, 0, 1)),
                        "place_sp_normalised": logloss(t.win_lose_p, (1 / t.bsp_p) / (1 / t.bsp_p).groupby(t.race)
                                                       .transform("sum") * t.places),
                        "harville": logloss(t.win_lose_p, t.harville), "gbm": logloss(t.win_lose_p, t.p_gbm)},
           "place_sp_book": round(float((1 / t.bsp_p).groupby(t.race).sum().div(t.groupby("race").places.first())
                                        .mean()), 4),
           "rules": score(t, "harville") + score(t, "p_gbm"),
           "by_year_gbm_back_5pct_2comm": [], "calibration": calibration(t).reset_index().astype(str)
           .to_dict("records")}
    for y, g in t.groupby("year"):
        for r in score(g, "p_gbm"):
            if r["comm"] == 0.02 and r["edge>"] in (0.02, 0.05, 0.10):
                rep["by_year_gbm_back_5pct_2comm"].append({"year": int(y), **r})
    return rep


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--years", default="2016-2026")
    ap.add_argument("--first-test", type=int, default=2021)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.years.split("-"))
    rep = run(Store(root=a.store), range(lo, hi + 1), a.first_test, a.oracle)
    text = json.dumps(rep, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
