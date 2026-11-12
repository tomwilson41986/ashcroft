"""How good is Betfair's closing price? Calibration of the close against the result, cell by cell.

racing² football research, 11 Nov 2026 (the warehouse's first use: docs/football_data_platform.md). The closing
chance of every match-odds runner (home, draw, away) and of over and under 2.5 goals, normalised to its book, set
against what happened. A market that is efficient at the close is calibrated in every cell; where it is not is
where an outcome model has something to say, and what a favourite-longshot bias looks like in this data.

Read from the warehouse: ``gold/event`` (the closing 1X2, the result, the match's liquidity and category) and
``gold/implied`` at the close (the over 2.5 chance and its book). Fresh closes only by default (the book's stalest
price no older than ``MAX_AGE_S`` at the off): a stale last traded price is not a closing price.

Per cell (by selection; by chance band; and, on the match-odds favourite, by liquidity tier, category and the
country codes with most matches: a whole match's selections are calibrated by construction): selections, the mean
closing chance, the realised rate, the gap with a standard error from resampled match days, and the return to
backing every selection in the cell at its closing price before commission. Overall: log loss and Brier score of
the close, and the slope of a logistic fit of the result on the closing log odds (above one: favourites win more
often than priced).

    python -m research.open_close.football_bf_efficiency --out-json out/bf_efficiency.json [--root <local folder>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAX_AGE_S = 600.0
BANDS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
TIER_CUTS = (300, 120, 40)


def selections(store, max_age_s: float = MAX_AGE_S) -> pd.DataFrame:
    """One row a selection a match at a fresh close: its normalised chance, its raw chance (1 / price share of the
    book kept), whether it won, and the match's cells."""
    from football.warehouse.store_io import read_table
    ev = read_table(store, "gold/event", ["event_id", "kickoff", "result", "over25", "p_home", "p_draw", "p_away",
                                           "close_legs", "close_book_sum", "close_age_s", "mo_ticks_pre", "category",
                                           "country_code", "void"])
    if ev is None:
        return pd.DataFrame()
    e = ev.to_pandas()
    e = e[~e["void"]]
    e["day"] = e["kickoff"].dt.strftime("%Y-%m-%d")
    a, b, c = TIER_CUTS
    e["tier"] = np.where(e["mo_ticks_pre"] >= a, "A", np.where(e["mo_ticks_pre"] >= b, "B",
                                                               np.where(e["mo_ticks_pre"] >= c, "C", "D")))
    base = ["event_id", "day", "tier", "category", "country_code"]
    rows = []
    mo = e[(e["close_legs"] == 3) & e["result"].notna() & (e["close_age_s"] <= max_age_s)]
    for sel, col, res in (("home", "p_home", "H"), ("draw", "p_draw", "D"), ("away", "p_away", "A")):
        r = mo[base].copy()
        r["selection"], r["p"], r["won"] = sel, mo[col].to_numpy(float), (mo["result"] == res).to_numpy(float)
        r["p_raw"] = r["p"] * mo["close_book_sum"].to_numpy(float)
        rows.append(r)
    im = read_table(store, "gold/implied", ["event_id", "cp_min", "p_over", "ou_book_sum", "ou_age_s"])
    if im is not None:
        i = im.to_pandas()
        i = i[(i["cp_min"] == 0) & i["p_over"].notna() & (i["ou_age_s"] <= max_age_s)]
        ou = e[e["over25"].notna()][base + ["over25"]].merge(i, on="event_id")
        for sel, p, won in (("over", ou["p_over"], ou["over25"]), ("under", 1 - ou["p_over"], 1 - ou["over25"])):
            r = ou[base].copy()
            r["selection"], r["p"], r["won"] = sel, p.to_numpy(float), won.to_numpy(float)
            r["p_raw"] = r["p"] * ou["ou_book_sum"].to_numpy(float)
            rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cell(d: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """The cell's calibration: selections, mean chance, realised rate, the gap and its standard error (days
    resampled), and the return to backing every selection at its closing price before commission."""
    if len(d) < 200:
        return {"n": int(len(d))}
    f = pd.DataFrame({"day": d["day"].to_numpy(), "p": d["p"].to_numpy(), "w": d["won"].to_numpy(), "n": 1.0,
                      "ret": (d["won"] / d["p_raw"] - 1.0).to_numpy()}).groupby("day").sum()
    idx = np.random.default_rng(seed).integers(0, len(f), size=(n_boot, len(f)))
    n = f["n"].to_numpy()[idx].sum(1)
    gap = (f["w"].to_numpy()[idx].sum(1) - f["p"].to_numpy()[idx].sum(1)) / n
    ret = f["ret"].to_numpy()[idx].sum(1) / n
    return {"n": int(len(d)), "mean_chance": round(float(d["p"].mean()), 5), "rate": round(float(d["won"].mean()), 5),
            "gap": round(float(d["won"].mean() - d["p"].mean()), 5), "gap_se": round(float(gap.std(ddof=1)), 5),
            "back_return": round(float((d["won"] / d["p_raw"] - 1.0).mean()), 5),
            "back_return_se": round(float(ret.std(ddof=1)), 5)}


def logistic_slope(p: np.ndarray, won: np.ndarray, match: np.ndarray | None = None) -> dict:
    """won ~ a + b * logit(p) by maximum likelihood (Newton): b above one means the close is too timid. The
    standard error is clustered by ``match`` where given: a match's selections are one draw, not three (their
    results sum to one), and a plain standard error counts them as independent."""
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = np.c_[np.ones_like(x), x]
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        w = mu * (1 - mu)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-9 * np.eye(2), X.T @ (won - mu))
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    bread = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    cov = bread
    if match is not None:
        score = X * (won - mu)[:, None]
        g = pd.DataFrame(score).groupby(np.asarray(match)).sum().to_numpy()
        cov = bread @ (g.T @ g) @ bread * (len(g) / max(len(g) - 1, 1))
    return {"intercept": round(float(beta[0]), 5), "slope": round(float(beta[1]), 5),
            "slope_se": round(float(np.sqrt(cov[1, 1])), 5)}


def run(store, max_age_s: float = MAX_AGE_S) -> dict:
    d = selections(store, max_age_s)
    if d.empty:
        return {"status": "no data found: gold/event is not built"}
    p, w = d["p"].to_numpy(float), d["won"].to_numpy(float)
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    out = {"selections": int(len(d)), "matches": int(d["event_id"].nunique()), "max_close_age_s": max_age_s,
           "days": int(d["day"].nunique()),
           "note": "closing last traded prices normalised to the book; development evidence on one year. The "
                   "'all' cell's gap is zero by construction (a book's chances and results both sum to one): read "
                   "its back return, and the slope with its match-clustered standard error",
           "all": cell(d),
           "log_loss": round(float(-(w * np.log(pc) + (1 - w) * np.log(1 - pc)).mean()), 6),
           "brier": round(float(((p - w) ** 2).mean()), 6),
           "logistic": {"all": logistic_slope(p, w, d["event_id"].to_numpy())},
           "by_selection": {k: cell(g) for k, g in d.groupby("selection")}}
    # a match's selections sum to one in chance and in wins, so a cell of whole matches is calibrated by
    # construction: cells of matches are read on the match-odds favourite (one selection a match) and by the slope
    mo_all = d[d["selection"].isin(["home", "draw", "away"])]
    fav = mo_all.loc[mo_all.groupby("event_id")["p"].idxmax()]
    for name, col in (("by_tier", "tier"), ("by_category", "category")):
        out[name + "_favourite"] = {k: {**cell(g), "slope_all_match_odds": logistic_slope(
            mo_all[mo_all[col] == k]["p"].to_numpy(float), mo_all[mo_all[col] == k]["won"].to_numpy(float),
            mo_all[mo_all[col] == k]["event_id"].to_numpy())}
            for k, g in fav.groupby(col)}
    for k, g in d.groupby("selection"):
        out["logistic"][k] = logistic_slope(g["p"].to_numpy(float), g["won"].to_numpy(float))
    band = pd.cut(d["p"], BANDS, include_lowest=True).astype(str)
    out["by_chance_band"] = {k: cell(g) for k, g in d.groupby(band)}
    mo = d[d["selection"].isin(["home", "draw", "away"])]
    out["by_chance_band_match_odds_tiers_AB"] = {
        k: cell(g) for k, g in mo[mo["tier"].isin(["A", "B"])].groupby(
            pd.cut(mo[mo["tier"].isin(["A", "B"])]["p"], BANDS, include_lowest=True).astype(str))}
    cc = fav["country_code"].fillna("(none)")
    out["by_country_top12_favourite"] = {k: cell(fav[cc == k]) for k in cc.value_counts().head(12).index}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a local warehouse folder in place of the capture bucket")
    ap.add_argument("--max-age-s", type=float, default=MAX_AGE_S)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    from sources.common import Store
    res = run(Store(root=a.root) if a.root else Store(), a.max_age_s)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
