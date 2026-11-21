"""Every way to trade the residual on the BSP (the owner, 21 Nov 2026: "Zoom out to high level and evaluate all possible
trading strategies to see if they can be profitable on our latest model"; ledger greyhound-residual-strategies-1121).

The residual (research/greyhound/sp_offset.py, frozen for the forward test as the mean of 20 seeds) is a card signal:
known the day before, before any price. Build 6 read it only as bets at the Betfair SP. This screen reads the same
out-of-sample signal (the three test folds, the mean of 20 seeds, the same races and features as the frozen model)
through every other route the historic data can price:

- WHEN the price is taken: the morning (MORNINGWAP, traded before 11:00), the pre-play average (PPWAP), the SP;
- HOW it is closed: held to the result, or greened at the SP (the move from the entry to the SP, whatever wins);
- WHAT order: a limit on close (an SP bet matched only if the SP is on the right side of the morning price, or under
  a fixed cap);
- HOW MUCH: level stakes, or stakes in proportion to the signal;
- WHICH market: win, or to be placed (the place SP).

16 rules (8 routes x back the top 10% / lay the bottom 10% of the centred residual, cuts from each fit's early-stopping
days), each with the same bets at the win SP beside it, so the route's own effect is read apart from the selection.
Beside them, descriptive (0 trials): how much money waits in the SP pool before the off, from the recorder's books.

    python -m research.greyhound.residual_strategies --out out/study
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-residual-strategies-1121"
SEEDS = tuple(range(1, 21))
DROP_FEATURES = ("gp_bsp_ae_car", "gp_bsp_mkt_car")
BACK_CUT, LAY_CUT = "back_top10", "lay_bottom10"
# (rule, side, route, argument): every rule registered before the run
RULES = (
    ("back_morning_hold", "back", "hold", "morningwap"), ("lay_morning_hold", "lay", "hold", "morningwap"),
    ("back_ppwap_hold", "back", "hold", "ppwap"), ("lay_ppwap_hold", "lay", "hold", "ppwap"),
    ("back_morning_out", "back", "out", "morningwap"), ("lay_morning_out", "lay", "out", "morningwap"),
    ("back_ppwap_out", "back", "out", "ppwap"), ("lay_ppwap_out", "lay", "out", "ppwap"),
    ("back_loc_morning", "back", "loc", "morningwap"), ("lay_loc_morning", "lay", "loc", "morningwap"),
    ("lay_loc_10", "lay", "cap", 10.0), ("lay_loc_20", "lay", "cap", 20.0),
    ("back_sp_scaled", "back", "scaled", None), ("lay_sp_scaled", "lay", "scaled", None),
    ("back_place_sp", "back", "place", None), ("lay_place_sp", "lay", "place", None),
)


# --------------------------------------------------------------------------------------------------------------------
# Settlement: a bet's money at risk and its gross, then netted by race with commission off a gain
# --------------------------------------------------------------------------------------------------------------------

def hold(side: str, price, won, w) -> tuple[np.ndarray, np.ndarray]:
    """A back of ``w`` at ``price``, or a lay risking ``w`` (stake w / (price - 1)), held to the result."""
    p, won, w = np.asarray(price, float), np.asarray(won, float) == 1, np.asarray(w, float)
    if side == "back":
        return w, np.where(won, w * (p - 1.0), -w)
    return w, np.where(won, -w, w / (p - 1.0))


def green(side: str, entry, exit_, w) -> tuple[np.ndarray, np.ndarray]:
    """Taken at ``entry``, closed at ``exit_`` for the same profit whatever wins: a back of ``w`` makes
    w (entry / exit - 1); a lay risking ``w`` (stake w / (entry - 1)) makes stake (1 - entry / exit)."""
    e, x, w = np.asarray(entry, float), np.asarray(exit_, float), np.asarray(w, float)
    if side == "back":
        return w, w * (e / x - 1.0)
    return w, w / (e - 1.0) * (1.0 - e / x)


def per_race(day, race, risk, gross, comm: float) -> pd.DataFrame:
    """One row a race: the money at risk and the net, commission off the race's gain."""
    r = pd.DataFrame({"day": np.asarray(day), "race_id": np.asarray(race), "risk": np.asarray(risk, float),
                      "gross": np.asarray(gross, float)}).groupby(["day", "race_id"], as_index=False).sum()
    r["net"] = np.where(r.gross > 0, r.gross * (1.0 - comm), r.gross)
    return r


# --------------------------------------------------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------------------------------------------------

def selected(r: pd.DataFrame, cuts: dict, side: str) -> pd.DataFrame:
    return r[r.s >= cuts[BACK_CUT]] if side == "back" else r[r.s <= cuts[LAY_CUT]]


def rule_bets(r: pd.DataFrame, cuts: dict, side: str, route: str, arg, comm: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(the rule's races, the same bets at the win SP, the bets themselves) on one fold's runners ``r``."""
    b = selected(r, cuts, side)
    b = b[b.bsp > 1]
    w = np.ones(len(b))
    if route in ("hold", "out"):
        b = b[b[arg] > 1]
        w = np.ones(len(b))
        risk, gross = hold(side, b[arg], b.won, w) if route == "hold" else green(side, b[arg], b.bsp, w)
    elif route == "loc":                               # an SP bet with the morning price as its limit
        b = b[b[arg] > 1]
        b = b[(b.bsp >= b[arg]) if side == "back" else (b.bsp <= b[arg])]
        w = np.ones(len(b))
        risk, gross = hold(side, b.bsp, b.won, w)
    elif route == "cap":                               # a lay at the SP with a fixed most it will lay at
        b = b[b.bsp <= float(arg)]
        w = np.ones(len(b))
        risk, gross = hold(side, b.bsp, b.won, w)
    elif route == "scaled":                            # stakes in proportion to the signal past its cut (1 at the cut)
        cut = cuts[BACK_CUT] if side == "back" else cuts[LAY_CUT]
        w = (b.s / cut).to_numpy(float)
        risk, gross = hold(side, b.bsp, b.won, w)
    elif route == "place":                             # the same dogs to be placed, at the place SP
        b = b[(b.place_bsp > 1) & b.placed.notna()]
        w = np.ones(len(b))
        risk, gross = hold(side, b.place_bsp, b.placed, w)
    else:
        raise ValueError(route)
    rule = per_race(b.day, b.race_id, risk, gross, comm)
    r0, g0 = hold(side, b.bsp, b.won, w)
    at_sp = per_race(b.day, b.race_id, r0, g0, comm)
    return rule, at_sp, b


def screen(folds: list[dict], comm: float, n_trials: int) -> dict:
    """Every rule on every fold, pooled; beside each, the same bets at the win SP; the liquidity of its runners."""
    from research.greyhound.sp_harness import read
    out = {"tag": TAG, "n_trials": int(n_trials), "rules": [], "reference": []}
    for name, side, route, arg in RULES:
        parts, sp_parts, bets, by_fold = [], [], [], []
        for f in folds:
            rule, at_sp, b = rule_bets(f["_runners"], f["cuts"], side, route, arg, comm)
            parts.append(rule)
            sp_parts.append(at_sp)
            bets.append(b)
            one = read(rule, n_trials) if len(rule) else {"bets": 0}
            by_fold.append({"test": f["test"], "roi_%": one.get("roi_%"), "bets": one.get("bets")})
        allr, alls, allb = (pd.concat(x, ignore_index=True) for x in (parts, sp_parts, bets))
        pooled = read(allr, n_trials) if len(allr) else {"bets": 0, "note": "no data found"}
        same = read(alls, n_trials) if len(alls) else {"bets": 0}
        lo = (pooled.get("roi_90%") or [None])[0]
        pos = sum(1 for x in by_fold if (x.get("roi_%") or 0) > 0)
        liq = {c: round(float(allb[c].median()), 1) for c in ("morning_vol", "pp_vol") if c in allb and allb[c].notna().any()}
        out["rules"].append({"rule": name, "side": side, "route": route, "arg": arg, "pooled": pooled,
                             "same_bets_at_sp": {k: same.get(k) for k in ("bets", "roi_%", "roi_90%", "t")},
                             "folds": by_fold, "folds_positive": pos,
                             "bets_a_day": round(len(allb) / max(1, allr.day.nunique()), 1) if len(allr) else 0,
                             "mean_price": round(float(allb.bsp.mean()), 2) if len(allb) else None,
                             "liquidity_median": liq,
                             "pass": bool(lo is not None and lo > 0 and pos >= 2)})
    for side in ("back", "lay"):                       # the registered SP rules on the same runners (counted before)
        parts = []
        for f in folds:
            b = selected(f["_runners"], f["cuts"], side)
            b = b[b.bsp > 1]
            risk, gross = hold(side, b.bsp, b.won, np.ones(len(b)))
            parts.append(per_race(b.day, b.race_id, risk, gross, comm))
        x = pd.concat(parts, ignore_index=True)
        out["reference"].append({"rule": f"{side}_sp (build 6, counted before)", "pooled": read(x, n_trials)})
    out["passed"] = [r["rule"] for r in out["rules"] if r["pass"]]
    return out


# --------------------------------------------------------------------------------------------------------------------
# Joins
# --------------------------------------------------------------------------------------------------------------------

def join_prices(runners: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Each test runner with its win file's morning and pre-play prices and volumes, and the place file's SP and
    result, by date, off time, track and trap (greyhound/parity.py ``attach_bsp``'s join)."""
    from greyhound.data import canonical_track
    k = ["race_date", "race_time", "track", "trap"]
    p = prices.assign(track=canonical_track(prices.track.astype(str))).dropna(subset=["trap"])
    p = p.assign(trap=p.trap.astype("int64"))
    out = runners.assign(trap=pd.to_numeric(runners.trap, errors="coerce").astype("Int64").astype("int64"))
    w = p[p.market == "win"].drop_duplicates(k, keep="last")
    wc = [c for c in ("morningwap", "ppwap", "morning_vol", "pp_vol") if c in w]
    out = out.merge(w[k + wc], on=k, how="left")
    pl = p[p.market == "place"].drop_duplicates(k, keep="last")
    pl = pl[k + ["bsp", "win_lose"]].rename(columns={"bsp": "place_bsp", "win_lose": "placed"}) if len(pl) else \
        pd.DataFrame(columns=k + ["place_bsp", "placed"])
    out = out.merge(pl, on=k, how="left")
    for c in ("morningwap", "ppwap", "morning_vol", "pp_vol", "place_bsp", "placed"):
        if c not in out:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # a place market counts only if it prices every runner of the race the win SP priced
    whole = (out.place_bsp > 1).groupby(out.race_id).transform("all")
    out.loc[~whole, ["place_bsp", "placed"]] = np.nan
    return out


def load_all_prices(store, years: str) -> pd.DataFrame:
    """The greyhound price files, GB win and place markets, the columns the routes read."""
    import re
    lo, hi = (int(x) for x in years.split("-"))
    keys = sorted(k for k in store.listing("greyhound_prices/") if re.search(r"prices_(\d{4})\.parquet$", k)
                  and lo <= int(re.search(r"prices_(\d{4})", k).group(1)) <= hi)
    cols = ["market", "country", "race_date", "race_time", "track", "trap", "bsp", "ppwap", "morningwap",
            "morning_vol", "pp_vol", "win_lose"]
    parts = []
    for k in keys:
        t = store.get_parquet(k)
        if t is None or not len(t):
            continue
        t = t[(t.country == "GB") & t.market.isin(["win", "place"])]
        parts.append(t[[c for c in cols if c in t.columns]].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------------------------------------------------
# The SP pool (descriptive): what waits to be matched at the SP before the off, from the recorder's books
# --------------------------------------------------------------------------------------------------------------------

BANDS = (1.0, 3.0, 5.0, 10.0, 20.0, 1000.0)


def sp_pool(books: pd.DataFrame) -> pd.DataFrame:
    """One row a runner: the most SP back stake and SP lay liability taken in any pre-off poll, and its SP."""
    b = books.assign(market_id=books.market_id.astype(str), selection_id=pd.to_numeric(books.selection_id, errors="coerce"))
    pre = b[(b.source.astype(str) != "final") & (b.status.astype(str) == "OPEN")
            & b.inplay.astype(str).isin(["0", "False", "false"])]
    for c in ("sp_back_taken", "sp_lay_taken"):
        pre = pre.assign(**{c: pd.to_numeric(pre[c], errors="coerce")})
    agg = pre.groupby(["market_id", "selection_id"])[["sp_back_taken", "sp_lay_taken"]].max()
    sp = b.assign(sp_actual=pd.to_numeric(b.sp_actual, errors="coerce")).dropna(subset=["sp_actual"])
    sp = sp.groupby(["market_id", "selection_id"]).sp_actual.last()
    return agg.join(sp, how="inner").reset_index()


def sp_pool_table(pool: pd.DataFrame) -> list[dict]:
    """By SP band: runners, the median SP back stake waiting (what an SP lay meets first) and the liability it lets a
    layer take, the median SP lay liability waiting (what an SP back meets first)."""
    rows = []
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        g = pool[(pool.sp_actual > a) & (pool.sp_actual <= b)]
        if not len(g):
            continue
        back, lay = g.sp_back_taken.fillna(0), g.sp_lay_taken.fillna(0)
        rows.append({"sp_band": f"{a:g}-{b:g}" if b < 1000 else f">{a:g}", "runners": int(len(g)),
                     "sp_back_stake_median": round(float(back.median()), 2),
                     "sp_back_stake_as_liability_median": round(float((back * (g.sp_actual - 1)).median()), 2),
                     "sp_lay_liability_median": round(float(lay.median()), 2),
                     "share_with_sp_backs": round(float((back > 0).mean()), 3)})
    return rows


def recorded_pool(last: date, days: int, s3=None) -> tuple[pd.DataFrame, list[str]]:
    from betfair_recorder import _read_day_csv
    parts, seen = [], []
    for i in range(days):
        d = last - timedelta(days=i)
        books = _read_day_csv(d, "books_greyhound.csv", s3=s3)
        if books is None or not len(books) or "sp_back_taken" not in books:
            continue
        parts.append(sp_pool(books))
        seen.append(f"{d:%Y-%m-%d}")
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), seen


# --------------------------------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------------------------------

def markdown(res: dict) -> str:
    md = [f"## Every route for the residual on the BSP ({TAG})", "",
          f"The out-of-sample residual (the mean of {res.get('seeds', 20)} seeds; tests 2024, 2025, Jan-May 2026), the top "
          "10% backed or the bottom 10% laid, a back of 1 or a lay risking 1, netted by race, "
          f"{res.get('commission', 0.02):.0%} off a gain. The deflated Sharpe at {res['screen']['n_trials']:,} greyhound "
          "trials. 'Same bets at the SP': the rule's own bets settled at the win SP, so the route's effect is the gap.", "",
          "| Rule | Bets | Bets a day | Return % | 90% | t | Deflated Sharpe | Folds + | Same bets at the SP % | Mean SP | Morning vol | Pre-play vol | Pass |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["screen"]["rules"]:
        p, s, lq = r["pooled"], r["same_bets_at_sp"], r["liquidity_median"]
        md.append(f"| {r['rule']} | {p.get('bets')} | {r['bets_a_day']} | {p.get('roi_%')} | {p.get('roi_90%')} | "
                  f"{p.get('t')} | {p.get('dsr')} | {r['folds_positive']} | {s.get('roi_%')} | {r['mean_price']} | "
                  f"{lq.get('morning_vol', '')} | {lq.get('pp_vol', '')} | {r['pass']} |")
    for r in res["screen"]["reference"]:
        p = r["pooled"]
        md.append(f"| {r['rule']} | {p.get('bets')} | | {p.get('roi_%')} | {p.get('roi_90%')} | {p.get('t')} | "
                  f"{p.get('dsr')} | | | | | | |")
    md += ["", f"Passed (development: pooled 90% above zero, two of three folds positive): {res['screen']['passed'] or 'none'}."]
    cov = res.get("coverage") or {}
    md += ["", "Coverage of the test runners: " + ", ".join(f"{k} {v}" for k, v in cov.items())]
    pool = res.get("sp_pool") or {}
    if pool.get("table"):
        md += ["", f"### The SP pool before the off (the recorder's books, {len(pool['days'])} days: "
               f"{pool['days'][-1]} to {pool['days'][0]}; {pool['runners']:,} runners)", "",
               "| SP band | Runners | SP back stake waiting (median) | As a layer's liability | SP lay liability waiting (median) | Share with SP backs |",
               "|---|---|---|---|---|---|"]
        md += [f"| {r['sp_band']} | {r['runners']:,} | {r['sp_back_stake_median']} | "
               f"{r['sp_back_stake_as_liability_median']} | {r['sp_lay_liability_median']} | {r['share_with_sp_backs']} |"
               for r in pool["table"]]
    else:
        md += ["", "The SP pool: no data found in the recorder's books."]
    return "\n".join(md)


def main(argv=None) -> int:
    from greyhound import fastloop as fl
    from greyhound.costs import commission
    from research.greyhound import sp_offset as so
    from research.greyhound.trial_count import greyhound_trials
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/residual_strategies")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--pool-days", type=int, default=21)
    a = ap.parse_args(argv)
    store = Store()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, meta = fl.load_cache(store)
    feats = [f for f in fl.feature_set(meta, "card_safe") if f in df.columns and f not in fl.LABELS and f not in DROP_FEATURES]
    df = df.assign(day=pd.to_datetime(df.t).dt.strftime("%Y-%m-%d"))
    keep, dropped = so.study_rows(df, True)
    d = df[keep]
    del df
    seeds = tuple(range(1, a.seeds + 1))
    log.info("%d rows, %d touched races out, %d features, seeds %s", len(d), dropped, len(feats), seeds)
    folds = []
    for lo, hi in so.FOLDS:
        f = so.run_fold(d, feats, lo, hi, seeds=seeds)
        f.pop("_per_race", None)
        folds.append(f)
        log.info("fold %s..%s gain %s", lo, hi, f.get("gain"))
    del d
    prices = load_all_prices(store, "2024-2026")
    log.info("prices: %d rows (%s)", len(prices), prices.market.value_counts().to_dict() if len(prices) else {})
    for f in folds:
        f["_runners"] = join_prices(f["_runners"], prices)
    allr = pd.concat([f["_runners"] for f in folds], ignore_index=True)
    cov = {"runners": int(len(allr)),
           "morning price": round(float((allr.morningwap > 1).mean()), 3),
           "pre-play price": round(float((allr.ppwap > 1).mean()), 3),
           "place SP (whole race)": round(float((allr.place_bsp > 1).mean()), 3)}
    comm = commission()
    res = {"tag": TAG, "seeds": len(seeds), "commission": comm, "coverage": cov,
           "fold_gain": [{"test": f["test"], "gain": f.get("gain"), "t": f.get("t")} for f in folds],
           "screen": screen(folds, comm, greyhound_trials())}
    try:
        from datetime import datetime, timezone
        pool, seen = recorded_pool(datetime.now(timezone.utc).date() - timedelta(days=1), a.pool_days)
        res["sp_pool"] = {"days": seen, "runners": int(len(pool)), "table": sp_pool_table(pool) if len(pool) else []}
    except Exception as e:                             # the pool is descriptive: the screen stands without it
        log.warning("the SP pool was not read: %s", e)
        res["sp_pool"] = {"note": f"not read: {e}"}
    allr.to_parquet(out / "oof_runners.parquet", index=False)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
