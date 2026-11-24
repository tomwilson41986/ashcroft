"""One harness for bets at the Betfair SP, and its first use: first-time runners (the owner, 20 Nov 2026: the
discovery round, ideas sp-12 and sp-10; reports/greyhound_discovery_1120.md; ledger greyhound-sp-debut-1120).

A bet at the SP is always matched, so it carries no fill assumption: a back of 1 at the BSP, or a lay risking 1 (a
stake of 1 / (BSP - 1)). Settlement is by race: the rule's bets in a race are netted and commission (greyhound/costs.py)
comes off a positive net. A rule is read as the return on the money at risk with a 90% interval by day (the day as
the cluster), and its deflated Sharpe on daily returns at every greyhound rule counted (research/greyhound/
trial_count.py). The bet must be chosen from the card: nothing here may read the BSP, its rank or its band to choose.

**The frame.** GBGB runs (greyhound.data.clean) joined to the Betfair price files (greyhound.parity.attach_bsp), races
of four to six runners with a BSP for every runner and one winner. ``runs_before``: the dog's GBGB races before this
one, counted from 2014.

**The periods** (the join of GBGB runs to the BSP before 2021 was only made on 18 Nov 2026, so a card-joined rule has
never been read on 2018-20): development 2023-01-01 to 2026-11-18; the unseen block 2018-2020, read once; 2021-2022
shown beside it (partly joined before); forward days from 19 Nov 2026 are a later registration.

**First use: first-time runners.** Five card-only cells - a dog with 0, 1-2, 3-5 or 6 and more races before, and an
experienced dog (1 or more) in a race holding a first-time runner - each backed and laid: 10 rules.

    python -m research.greyhound.sp_harness --out out/study
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-sp-debut-1120"
PERIODS = {"development": ("2023-01-01", "2026-11-18"), "unseen": ("2018-01-01", "2020-12-31"),
           "partly_seen": ("2021-01-01", "2022-12-31")}
CELLS = ("first_run", "runs_1_2", "runs_3_5", "runs_6_plus", "rival_of_first_run")
SIDES = ("back", "lay")


def settle(bets: pd.DataFrame, comm: float) -> pd.DataFrame:
    """One row a race: the money at risk and the net after commission. ``bets``: day, race_id, side, bsp, won."""
    b = bets
    back = b.side.to_numpy() == "back"
    bsp, won = b.bsp.to_numpy(float), b.won.to_numpy(float) == 1
    stake = np.where(back, 1.0, 1.0 / (bsp - 1.0))                       # a lay risks 1: its liability
    pnl = np.where(back, np.where(won, stake * (bsp - 1.0), -stake), np.where(won, -1.0, stake))
    r = pd.DataFrame({"day": b.day.to_numpy(), "race_id": b.race_id.to_numpy(), "risk": 1.0, "gross": pnl}) \
        .groupby(["day", "race_id"], as_index=False).sum()
    r["net"] = np.where(r.gross > 0, r.gross * (1.0 - comm), r.gross)
    return r


def ratio_t(day, num, den) -> dict:
    """sum(num) / sum(den) with the day as the cluster: the ratio, its standard error, t and 90% interval."""
    d = pd.DataFrame({"d": np.asarray(day), "n": np.asarray(num, float), "e": np.asarray(den, float)}).groupby("d").sum()
    g = len(d)
    if g < 1 or d.e.sum() <= 0:
        return {"days": int(g), "ratio": None, "se": None, "t": None, "lo": None, "hi": None}
    r = float(d.n.sum() / d.e.sum())
    if g < 2:                                                  # one day: a ratio, and no interval
        return {"days": 1, "ratio": r, "se": None, "t": None, "lo": None, "hi": None}
    se = float(np.sqrt(g / (g - 1) * ((d.n - r * d.e) ** 2).sum()) / d.e.sum())
    return {"days": int(g), "ratio": r, "se": se, "t": r / se if se > 0 else None,
            "lo": r - 1.6448536 * se, "hi": r + 1.6448536 * se}


def read(per_race: pd.DataFrame, n_trials: int) -> dict:
    """A rule's reading: the return on the money at risk, its interval by day, the deflated Sharpe of daily returns."""
    from research.overfit import dsr_check
    if not len(per_race):
        return {"races": 0, "bets": 0}
    rt = ratio_t(per_race.day, per_race.net, per_race.risk)
    d = per_race.groupby("day")[["net", "risk"]].sum()
    dsr = dsr_check((d.net / d.risk).to_numpy(), n_trials)
    pc = lambda v: None if v is None else round(100 * v, 3)
    return {"races": int(len(per_race)), "bets": int(per_race.risk.sum()), "days": rt["days"],
            "net": round(float(per_race.net.sum()), 2), "roi_%": pc(rt["ratio"]), "roi_90%": [pc(rt["lo"]), pc(rt["hi"])],
            "t": None if rt["t"] is None else round(rt["t"], 2),
            "dsr": dsr.get("dsr"), "dsr_passed": bool(dsr.get("passed")), "n_trials": int(n_trials)}


def frame(store, first_year: int = 2014, last_year: int = 2026) -> tuple[pd.DataFrame, dict]:
    """The runners of every wholly priced race of four to six, with ``runs_before`` and the cells; and the counts of
    what was kept and dropped."""
    from greyhound.data import clean
    from greyhound.model import load_prices
    from greyhound.parity import attach_bsp
    parts, notes = [], {"rows_in": 0, "years": {}}
    held = sorted(int(re.search(r"(\d{4})", k).group(1)) for k in store.listing("gbgb/")
                  if re.search(r"gbgb/runs_\d{4}\.parquet$", k))
    for y in [v for v in held if first_year <= v <= last_year]:
        raw = pd.read_parquet(io.BytesIO(store.get(f"gbgb/runs_{y}.parquet")))
        c = clean(raw)
        c = c[~c.is_trial]
        c = attach_bsp(c, load_prices(store, f"{y}-{y}"))
        notes["rows_in"] += int(len(raw))
        notes["years"][str(y)] = {"raw_rows": int(len(raw)), "race_rows": int(len(c)),
                                  "with_bsp": int((c.bsp > 1).sum())}
        parts.append(c[["race_id", "dog_id", "race_date", "race_time", "trap", "position", "bsp", "bsp_p_norm"]]
                     .assign(race_date=c.race_date.astype(str)))
        del raw, c
    r = pd.concat(parts, ignore_index=True).sort_values(["race_date", "race_time", "race_id"], kind="mergesort")
    r["runs_before"] = r.groupby("dog_id").cumcount()                 # the dog's races before this one, from 2014
    n = r.groupby("race_id").race_id.transform("size")
    priced = r.bsp_p_norm.notna().groupby(r.race_id).transform("all")
    won = (r.position == 1).astype(int)
    one = won.groupby(r.race_id).transform("sum") == 1
    keep = priced & n.between(4, 6) & one
    notes.update({"race_rows": int(len(r)), "races": int(r.race_id.nunique()),
                  "dropped_not_wholly_priced_rows": int((~priced).sum()),
                  "dropped_field_outside_4_6_rows": int((priced & ~n.between(4, 6)).sum()),
                  "dropped_not_one_winner_rows": int((priced & n.between(4, 6) & ~one).sum()),
                  "rows_out": int(keep.sum()), "races_out": int(r.race_id[keep].nunique())})
    f = r[keep].assign(won=won[keep], day=r.race_date[keep])
    first = (f.runs_before == 0)
    holds_first = first.groupby(f.race_id).transform("any")
    f["cell"] = np.select([first, f.runs_before <= 2, f.runs_before <= 5], ["first_run", "runs_1_2", "runs_3_5"],
                          "runs_6_plus")
    f["rival_of_first_run"] = holds_first & ~first
    notes["date_min"], notes["date_max"] = str(f.day.min()), str(f.day.max())
    return f, notes


def cell_rows(f: pd.DataFrame, cell: str) -> pd.DataFrame:
    return f[f.rival_of_first_run] if cell == "rival_of_first_run" else f[f.cell == cell]


def run(f: pd.DataFrame, comm: float, n_trials: int) -> dict:
    """Every cell and side on every period; a rule is CHOSEN on development (its 90% interval above zero) and passes
    the unseen block only if chosen and its interval there is above zero as well."""
    out = {"rules": [], "cells": {}}
    for cell in CELLS:
        x = cell_rows(f, cell)
        out["cells"][cell] = {}
        for name, (lo, hi) in PERIODS.items():
            p = x[(x.day >= lo) & (x.day <= hi)]
            ae = ratio_t(p.day, p.won, p.bsp_p_norm) if len(p) else {}
            out["cells"][cell][name] = {"runners": int(len(p)), "winners": int(p.won.sum()) if len(p) else 0,
                                        "ae_at_the_bsp": None if not ae or ae["ratio"] is None else round(ae["ratio"], 4),
                                        "ae_90%": None if not ae or ae["lo"] is None else
                                        [round(ae["lo"], 4), round(ae["hi"], 4)]}
        for side in SIDES:
            row = {"rule": f"{side}_{cell}"}
            for name, (lo, hi) in PERIODS.items():
                p = x[(x.day >= lo) & (x.day <= hi)]
                row[name] = read(settle(p.assign(side=side), comm), n_trials) if len(p) else {"races": 0, "bets": 0}
            above = lambda r: bool(r.get("roi_90%") and r["roi_90%"][0] is not None and r["roi_90%"][0] > 0)
            row["chosen_on_development"] = above(row["development"])
            row["passes_the_unseen_block"] = bool(row["chosen_on_development"] and above(row["unseen"]))
            out["rules"].append(row)
    out["chosen"] = [r["rule"] for r in out["rules"] if r["chosen_on_development"]]
    out["passed"] = [r["rule"] for r in out["rules"] if r["passes_the_unseen_block"]]
    return out


def markdown(res: dict) -> str:
    md = [f"## Bets at the SP by a dog's races before ({TAG})", "",
          "A back of 1 or a lay risking 1 on every runner of a cell at the Betfair SP, netted by race, 2% off a gain. "
          "Return on the money at risk, 90% interval by day.", "",
          "| Rule | Development 2023 - 18 Nov 2026: bets | Return % | 90% | t | Unseen 2018-20: bets | Return % | 90% | t | "
          "2021-22: Return % | 90% | Chosen | Passes |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    g = lambda r, k: "" if r.get(k) is None else r.get(k)
    for r in res["rules"]:
        d, u, p = r["development"], r["unseen"], r["partly_seen"]
        md.append(f"| {r['rule']} | {g(d, 'bets')} | {g(d, 'roi_%')} | {g(d, 'roi_90%')} | {g(d, 't')} | {g(u, 'bets')} | "
                  f"{g(u, 'roi_%')} | {g(u, 'roi_90%')} | {g(u, 't')} | {g(p, 'roi_%')} | {g(p, 'roi_90%')} | "
                  f"{r['chosen_on_development']} | {r['passes_the_unseen_block']} |")
    md += ["", "Winners against the BSP's chances (actual / expected), by cell:", "",
           "| Cell | Development: runners | A/E | 90% | Unseen: runners | A/E | 90% |", "|---|---|---|---|---|---|---|"]
    for c, v in res["cells"].items():
        d, u = v["development"], v["unseen"]
        md.append(f"| {c} | {d['runners']} | {d['ae_at_the_bsp']} | {d['ae_90%']} | {u['runners']} | {u['ae_at_the_bsp']} | "
                  f"{u['ae_90%']} |")
    md += ["", f"Chosen on development: {res['chosen'] or 'none'}. Passed the unseen block: {res['passed'] or 'none'}.",
           "", "### The frame", "", "```", json.dumps(res.get("frame", {}), indent=1), "```"]
    return "\n".join(md)


def main(argv=None) -> int:
    from greyhound.costs import commission
    from research.greyhound.trial_count import greyhound_trials
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/sp_harness")
    a = ap.parse_args(argv)
    f, notes = frame(Store())
    res = run(f, commission(), greyhound_trials())
    res["frame"], res["tag"] = {k: v for k, v in notes.items() if k != "years"}, TAG
    res["frame_years"] = notes["years"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
