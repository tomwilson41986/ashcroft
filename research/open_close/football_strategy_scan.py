"""A model-free scan for football trading leads on the warehouse (the owner, 12 Nov 2026), as registered in the ledger
before its run (football-strategy-scan-1112): 25 cells, nothing fitted, no cell added after the read.

1. In-play leader: back the most likely result at its raw last traded price, at the first minute in a window where
   all three legs traded within 60 s and its raw chance is in a band; held to the result.
2. Price jump in play (a goal, by price alone): back the side that gained, and apart the other side, 2 to 5 minutes on.
3. Over/under 2.5 at a fresh close: the outsider, and apart the favourite.
4. Asian handicap main half line at a fresh close: the shorter side, the longer side.
5. Pre-match momentum: the steamer, and apart the drifter, of T-360 to T-60, backed at T-60.
6. Rating gap: the side the market-implied ratings favour over the T-60 price.

Every bet is a back held to the result, a unit staked, 2 per cent commission on a win. A last traded price is not
a quote, so each cell is read at the price and at a worse one (one tick in play and at the close; 1.15 per cent off a
pre-match checkpoint's price). Odd kick-off months find, even months confirm: a lead is a cell with the same sign
in both halves whose even-month 90 per cent interval at the worse price is above zero. A lead is not a pass.

    python -m research.open_close.football_strategy_scan --out-json out/football_strategy_scan.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from research.open_close import football_bf_close_model as cm
from research.open_close import football_bf_efficiency as eff
from research.open_close.football_trade_model import ENTRY, summary

log = logging.getLogger(__name__)

FRESH_S = 60.0                      # in play: every leg traded within this long
BANDS = ((0.70, 0.85), (0.85, 0.95), (0.95, 0.99))
JUMP, JUMP_GAP_MIN, WAIT_MIN = 0.12, 3, (2, 5)
OU_OUTSIDER, OU_FAVOURITE = 0.40, 0.60
DRIFT, GAP, BUSY = 0.10, 0.15, 60
LADDER = ((2.0, 0.01), (3.0, 0.02), (4.0, 0.05), (6.0, 0.1), (10.0, 0.2), (20.0, 0.5), (30.0, 1.0), (50.0, 2.0),
          (100.0, 5.0), (1000.0, 10.0))


def tick_worse(price) -> np.ndarray:
    """A back one tick down Betfair's ladder from ``price`` (never under 1.01)."""
    p = np.asarray(price, float)
    step = np.full(p.shape, LADDER[-1][1])
    for top, s in reversed(LADDER):
        step = np.where(p <= top + 1e-9, s, step)
    return np.maximum(p - step, 1.01)


def back(price, won) -> np.ndarray:
    """A back of one unit held to the result, after commission on a win."""
    p = np.asarray(price, float)
    return np.where(np.asarray(won, bool), (p - 1.0) * (1 - cm.COMMISSION), -1.0)


def cell(d: pd.DataFrame, worse) -> dict:
    """A cell's read in each half: bets and the return at the price and at the worse one, with day intervals; and
    whether it is a lead by the registered reading."""
    out = {}
    odd = pd.to_datetime(d["day"]).dt.month % 2 == 1
    for name, part in (("odd_months", d[odd]), ("even_months", d[~odd]), ("all", d)):
        w = {"odd_months": worse[odd.to_numpy()], "even_months": worse[~odd.to_numpy()], "all": worse}[name]
        out[name] = {"at_the_price": summary(part["day"], back(part["price"], part["won"])),
                     "at_the_worse_price": summary(part["day"], back(w, part["won"])),
                     "mean_price": float(part["price"].mean()) if len(part) else None}
    a, b = out["odd_months"]["at_the_worse_price"], out["even_months"]["at_the_worse_price"]
    out["lead"] = bool("return" in a and "return" in b and a["return"] > 0 and b["return"] > 0 and b["lo90"] > 0)
    return out


def jumps(m: pd.DataFrame) -> pd.DataFrame:
    """Price jumps in all-fresh minute rows (event_id, minute, raw_home, raw_away sorted): one row a jump with the
    side that gained and the first all-fresh row ``WAIT_MIN`` minutes later (its minute and both sides' raw
    chances); a jump with no such row is dropped."""
    m = m.sort_values(["event_id", "minute"], kind="stable").reset_index(drop=True)
    same = m["event_id"].eq(m["event_id"].shift()) & (m["minute"] - m["minute"].shift() <= JUMP_GAP_MIN)
    parts = []
    for side in ("home", "away"):
        up = same & (m[f"raw_{side}"] - m[f"raw_{side}"].shift() >= JUMP)
        j = m.loc[up, ["event_id", "minute"]].assign(side=side)
        parts.append(j)
    j = pd.concat(parts, ignore_index=True)
    if j.empty:
        return j.assign(bet_minute=[], raw_gainer=[], raw_other=[])
    j["from"] = j["minute"] + WAIT_MIN[0]
    later = m.rename(columns={"minute": "bet_minute"})
    j = pd.merge_asof(j.sort_values("from", kind="stable"), later.sort_values("bet_minute", kind="stable"),
                      left_on="from", right_on="bet_minute", by="event_id", direction="forward",
                      tolerance=WAIT_MIN[1] - WAIT_MIN[0])
    j = j[j["bet_minute"].notna()]
    home = j["side"].eq("home").to_numpy()
    j["raw_gainer"] = np.where(home, j["raw_home"], j["raw_away"])
    j["raw_other"] = np.where(home, j["raw_away"], j["raw_home"])
    return j[["event_id", "minute", "side", "bet_minute", "raw_gainer", "raw_other"]].reset_index(drop=True)


def run(store) -> dict:
    from football.warehouse import catalog
    con = catalog.connect(store)
    q = lambda s: con.sql(s).df()                                                                  # noqa: E731
    res: dict = {"registered": "football-strategy-scan-1112",
                 "cost": {"commission": cm.COMMISSION, "pre_match_entry": ENTRY, "in_play_and_close": "one tick"},
                 "cells": {}}

    def add(name: str, d: pd.DataFrame, worse) -> None:
        res["cells"][name] = cell(d.reset_index(drop=True), np.asarray(worse))

    # ---- 1. the in-play leader -------------------------------------------------------------------------------------
    m = q(f"""SELECT m.event_id, m.minute, m.phase, m.match_min, m.p_home * m.mo_book_sum AS raw_home,
                     m.p_draw * m.mo_book_sum AS raw_draw, m.p_away * m.mo_book_sum AS raw_away,
                     strftime(e.kickoff, '%Y-%m-%d') AS day, e.result, e.p_home AS c_home, e.p_away AS c_away
              FROM inplay_minute m JOIN event e USING (event_id)
              WHERE m.ok AND m.mo_max_age_s <= {FRESH_S} AND e.result IS NOT NULL AND NOT e.void""")
    raw = m[["raw_home", "raw_draw", "raw_away"]].to_numpy(float)
    m["chance"] = raw.max(1)
    m["leader"] = np.array(["H", "D", "A"])[raw.argmax(1)]
    m["window"] = np.where(m["phase"] == "half_time", "half time", np.where(
        (m["phase"] == "second_half") & (m["match_min"] > 60) & (m["match_min"] <= 75), "minutes 60-75", np.where(
            (m["phase"] == "second_half") & (m["match_min"] > 75), "minutes 75-90", "")))
    for lo, hi in BANDS:
        b = m[(m["window"] != "") & (m["chance"] >= lo) & (m["chance"] < hi)].sort_values("minute", kind="stable")
        b = b.drop_duplicates(["event_id", "window"])
        b = b.assign(price=1.0 / b["chance"], won=b["result"] == b["leader"])
        for win, part in b.groupby("window"):
            add(f"in-play leader | {win} | chance {lo:.2f}-{hi:.2f}", part, tick_worse(part["price"]))

    # ---- 2. a price jump in play ------------------------------------------------------------------------------------
    j = jumps(m[["event_id", "minute", "raw_home", "raw_away"]])
    j = j.merge(m.drop_duplicates("event_id")[["event_id", "day", "result", "c_home", "c_away"]], on="event_id")
    home = j["side"].eq("home")
    fav = np.where(home, j["c_home"] >= j["c_away"], j["c_away"] > j["c_home"])
    won_gainer = np.where(home, j["result"] == "H", j["result"] == "A")
    won_other = np.where(home, j["result"] == "A", j["result"] == "H")
    for label, mask in (("gainer the pre-match favourite", fav), ("gainer the pre-match outsider", ~fav)):
        g = j[mask & (j["raw_gainer"] > 0)]
        g = g.assign(price=1.0 / g["raw_gainer"], won=won_gainer[mask & (j["raw_gainer"] > 0).to_numpy()])
        add(f"price jump | back the gainer | {label}", g, tick_worse(g["price"]))
        o = j[mask & (j["raw_other"] > 0)]
        o = o.assign(price=1.0 / o["raw_other"], won=won_other[mask & (j["raw_other"] > 0).to_numpy()])
        add(f"price jump | back the other side | {label}", o, tick_worse(o["price"]))

    # ---- 3. over/under 2.5 at a fresh close ------------------------------------------------------------------------
    s = eff.selections(store)
    ou = s[s["selection"].isin(["over", "under"]) & (s["p_raw"] > 0)].assign(price=lambda d: 1.0 / d["p_raw"])
    for label, mask in (("outsider (chance under 0.40)", ou["p"] < OU_OUTSIDER),
                        ("favourite (chance over 0.60)", ou["p"] > OU_FAVOURITE)):
        for tiers in (("A", "B"), ("C", "D")):
            part = ou[mask & ou["tier"].isin(tiers)]
            name = f"over/under 2.5 close | {label} | tiers {'-'.join(tiers)}"
            add(name, part, tick_worse(part["price"]))

    # ---- 4. the Asian handicap's main half line at a fresh close -----------------------------------------------------
    ah = q(f"""WITH c AS (
                 SELECT c.event_id, c.market_id, c.line, c.role, c.ltp, c.p_norm, r.won,
                        strftime(e.kickoff, '%Y-%m-%d') AS day
                 FROM checkpoint c JOIN event e ON e.event_id = c.event_id AND e.ah_market_id = c.market_id
                 JOIN runner r USING (market_id, selection_id, handicap)
                 WHERE c.cp_min = 0 AND c.market_type = 'ASIAN_HANDICAP' AND c.book_complete
                   AND c.book_max_age_s <= {eff.MAX_AGE_S} AND abs(c.line * 2) % 2 = 1 AND NOT e.void
                   AND c.role IN ('home', 'away') AND r.won IS NOT NULL),
               main AS (SELECT event_id, arg_min(line, abs(p_norm - 0.5)) AS line FROM c WHERE role = 'home'
                        GROUP BY 1)
               SELECT c.* FROM c JOIN main USING (event_id, line)""")
    ah = ah.assign(price=ah["ltp"], won=ah["won"] == 1)
    add("Asian handicap main half line close | the shorter side", ah[ah["p_norm"] > 0.5],
        tick_worse(ah.loc[ah["p_norm"] > 0.5, "price"]))
    add("Asian handicap main half line close | the longer side", ah[ah["p_norm"] < 0.5],
        tick_worse(ah.loc[ah["p_norm"] < 0.5, "price"]))

    # ---- 5 and 6. pre-match: momentum, and the ratings against the price ---------------------------------------------
    mv = q("""SELECT a.event_id, a.role, a.ltp_cp AS price, a.p_cp, b.p_cp AS p_early, a.n_ticks_before, a.won,
                     strftime(a.kickoff, '%Y-%m-%d') AS day, g.gap_logit_home AS gap,
                     coalesce(g.linked AND g.n_home >= 10 AND g.n_away >= 10, false) AS rated
              FROM move a JOIN event e ON e.event_id = a.event_id AND e.mo_market_id = a.market_id
              LEFT JOIN move b ON b.market_id = a.market_id AND b.selection_id = a.selection_id
                              AND b.handicap = a.handicap AND b.cp_min = 360 AND b.cp_fresh AND b.schedule_known
              LEFT JOIN rating_gap g ON g.event_id = a.event_id AND g.cp_min = 60
              WHERE a.cp_min = 60 AND a.market_type = 'MATCH_ODDS' AND a.cp_fresh AND a.schedule_known
                AND a.won IS NOT NULL AND NOT e.void""")
    mv["won"] = mv["won"] == 1
    logit = lambda p: np.log(p / (1 - p))                                                          # noqa: E731
    mv["drift"] = logit(mv["p_cp"]) - logit(mv["p_early"])
    for label, mask in (("the steamer", mv["drift"] >= DRIFT), ("the drifter", mv["drift"] <= -DRIFT)):
        for busy, extra in (("all rows", True), (f"{BUSY}+ price updates", mv["n_ticks_before"] >= BUSY)):
            part = mv[mask & extra]
            add(f"pre-match momentum T-360 to T-60 | back {label} | {busy}", part,
                part["price"] * np.exp(-ENTRY))
    pick = mv["rated"] & (((mv["role"] == "home") & (mv["gap"] >= GAP))
                          | ((mv["role"] == "away") & (mv["gap"] <= -GAP)))
    for busy, extra in (("all rows", True), (f"{BUSY}+ price updates", mv["n_ticks_before"] >= BUSY)):
        part = mv[pick & extra]
        add(f"rating gap at T-60 | back the side the ratings favour | {busy}", part, part["price"] * np.exp(-ENTRY))

    res["n_cells"] = len(res["cells"])
    res["leads"] = [k for k, v in res["cells"].items() if v["lead"]]
    con.close()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="a local warehouse folder in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from sources.common import Store
    res = run(Store(root=a.root) if a.root else Store())
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
