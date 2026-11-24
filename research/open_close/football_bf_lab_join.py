"""The Betfair-native closing-price model with the lab's team features: does the main model's knowledge of the teams
add to Betfair's own price path, checkpoint by checkpoint?

racing² football research, 24 Nov 2026 (the owner: "focus on building the best model we can"). Registered as
``football-bf-lab-join-1124`` in reports/research_ledger.jsonl before its run.

Two models exist. The lab model (``football/lab.py``) forecasts football-data's early-to-close move from the teams:
form, xG, team news, ratings, and the blocks that kept their place. The Betfair-native model
(``football_bf_close_model``) forecasts the move from a checkpoint (360, 120, 60, 15 minutes before the off) to
Betfair's close from the price path, the implied goals, the market's ratings and move memory. Neither sees the other's
features. This joins them: each Betfair event linked to one of our canonical matches (``gold/link``'s ``match_id``)
gets the lab's columns, and both arms are fitted the same way on the same rows:

- ``bf``: the Betfair-native model's features (level, path, implied, rating, memory);
- ``bf_lab``: those plus the lab's (the expected-value arm, home advantage, short form, the kept blocks and the two
  Sportmonks blocks that kept their place: ``LAB_BLOCKS``), numeric columns only.

**Lag.** The lab's features are pinned to football-data's collection day (Friday for Friday-Monday matches, Tuesday
for Tuesday-Thursday): only what was known before that day, plus that day's early prices. A checkpoint on the
collection day itself could come before those early prices were taken, so matches played on their collection day are
left out of both arms. Every other checkpoint is on a later day than everything the lab's features read.

**Validation** as the Betfair-native model's: walk-forward by calendar month, a model a role fitted on kick-offs at
least a day before the month began, trained on reliable rows (fresh checkpoint and close), scored on the reliable
rows whose schedule was known at the checkpoint. The change in skill against "no move", paired by match day (1,000
resamples), overall and by checkpoint; the trades at last traded prices beside it.

    python -m research.open_close.football_bf_lab_join --cache <lab cache> --out-json out/bf_lab_join.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from research.open_close import football_bf_close_model as bf

log = logging.getLogger(__name__)

TAG = "football-bf-lab-join-1124"
STACK_TAG = "football-bf-lab-stack-1124"
LAB_BLOCKS = ["home_adv", "form_short", "league_fx", "move_memory", "kalman_form", "sim", "style",
              "sm_xg", "sm_pred"]


def lab_table(cache: str) -> tuple[pd.DataFrame, list[str]]:
    """One row a canonical match: match_id and the lab's numeric feature columns (prefixed ``lab_``)."""
    from football import lab
    from research.open_close import football_review_arms as ra
    blocks = LAB_BLOCKS
    m, bfeat = lab.add_blocks(lab.latest(cache), blocks)
    d = lab.prepared(m)
    spec = "+".join([ra.BASE] + [b for b in blocks if b in bfeat])
    cols = [c for c in lab.resolve(spec, lab.families(m), bfeat, d.columns)
            if c != "league" and pd.api.types.is_numeric_dtype(d[c])]
    out = d[["match_id"] + cols].drop_duplicates("match_id").copy()
    out[cols] = out[cols].astype("float32")
    return out.rename(columns={c: f"lab_{c}" for c in cols}), [f"lab_{c}" for c in cols]


FC_FROM, FC_TRAIN_FROM = "2025-03", 2016
STACK_HORIZONS = (360, 120)


def lab_forecast(cache: str) -> pd.DataFrame:
    """The lab model's out-of-sample forecast of each match's early-to-close move (home and away), walk-forward by
    month: every month from ``FC_FROM`` forecast by the stack (ridge and LightGBM, the review arms' settings) fitted
    on the earlier months' matches from season ``FC_TRAIN_FROM`` (football-data's early-to-close moves, the lab's
    target). One row a match: match_id, lab_fc_h, lab_fc_a."""
    from football import lab
    from research.open_close import football_review_arms as ra
    from research.open_close.football_feature_batch import fit_predict
    m, bfeat = lab.add_blocks(lab.latest(cache), LAB_BLOCKS)
    d = lab.prepared(m)
    spec = "+".join([ra.BASE] + [b for b in LAB_BLOCKS if b in bfeat])
    cols = [c for c in lab.resolve(spec, lab.families(m), bfeat, d.columns) if c != "league"]
    d["month"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m")
    d = d[d["season"] >= FC_TRAIN_FROM]
    out = []
    for mo in sorted(x for x in d["month"].unique() if x >= FC_FROM):
        te = d[d["month"] == mo]
        row = {"match_id": te["match_id"].to_numpy()}
        for side in ("h", "a"):
            tr = d[(d["month"] < mo) & np.isfinite(d[f"y_{side}"])]
            row[f"lab_fc_{side}"] = fit_predict(tr, te, cols, tr[f"y_{side}"].to_numpy(float))
        out.append(pd.DataFrame(row))
        log.info("lab forecast %s: %d matches", mo, len(te))
    return pd.concat(out, ignore_index=True).drop_duplicates("match_id")


def run_stack(store, cache: str) -> dict:
    """The follow-up (football-bf-lab-stack-1124): the lab's forecast as two features in place of its 161 columns."""
    d, fam = bf.load(store)
    if d.empty:
        return {"tag": STACK_TAG, "status": "no data found: gold/move is not built"}
    link = bf.read_rows(store, "gold/link", ["event_id", "match_id"], horizons=False)
    if link is None:
        return {"tag": STACK_TAG, "status": "no data found: gold/link is not built"}
    fc = lab_forecast(cache)
    d = d[d["cp_fresh"]].reset_index(drop=True)
    j = join(d, link.to_pandas(), fc)
    j["lab_fc_own"] = np.where(j["role"] == "home", j["lab_fc_h"], np.where(j["role"] == "away", j["lab_fc_a"],
                                                                            np.nan)).astype("float32")
    base_cols = [c for cols in fam.values() for c in cols]
    extra = ["lab_fc_h", "lab_fc_a", "lab_fc_own"]
    res = {"tag": STACK_TAG, "rows_joined": int(len(j)), "events_joined": int(j["event_id"].nunique()),
           "forecast_coverage": round(float(j["lab_fc_h"].notna().mean()), 4),
           "note": "development evidence; last traded prices; nothing is traded"}
    if j["month"].nunique() < 4:
        res["status"] = "no data found: fewer than four months of joined rows"
        return res
    res.update(compare(j, base_cols, extra))
    sel = [str(h) for h in STACK_HORIZONS]
    p0, _ = bf.walk_forward(j, base_cols)
    p1, _ = bf.walk_forward(j, base_cols + extra)
    ok = np.isfinite(p0) & np.isfinite(p1) & j["schedule_fixed"].to_numpy() & j["usable"].to_numpy() & \
        j["cp_min"].isin(STACK_HORIZONS).to_numpy()
    sc = j[ok]
    res["primary_360_and_120"] = bf.skill_ci(sc["day"].to_numpy(), sc["y_logit"].to_numpy(float), p1[ok], base=p0[ok])
    res["horizons_in_primary"] = sel
    res["keeps_its_place"] = bool(res["primary_360_and_120"]["vs_base_lo95"] > 0)
    return res


def join(d: pd.DataFrame, link: pd.DataFrame, labt: pd.DataFrame) -> pd.DataFrame:
    """The Betfair rows of linked events, each with its match's lab columns; matches played on their collection day
    left out (see Lag)."""
    from football.team_features import collection_day
    lk = link[["event_id", "match_id"]].dropna().drop_duplicates("event_id")
    out = d.merge(lk, on="event_id", how="inner").merge(labt, on="match_id", how="inner")
    ko_day = out["kickoff"].dt.tz_convert(None).dt.normalize()
    out = out[ko_day.to_numpy() > collection_day(ko_day).to_numpy()]
    return out.reset_index(drop=True).copy()


def compare(d: pd.DataFrame, base_cols: list[str], lab_cols: list[str]) -> dict:
    """Both arms walk-forward on the same rows; the change in skill paired by day, overall and by checkpoint."""
    p0, _ = bf.walk_forward(d, base_cols)
    p1, gain = bf.walk_forward(d, base_cols + lab_cols, with_gain=True)
    d = d.assign(p0=p0, p1=p1)
    sc = d[np.isfinite(d["p0"]) & np.isfinite(d["p1"]) & d["schedule_fixed"] & d["usable"]]
    if len(sc) < 2000:
        return {"status": f"no data found: {len(sc)} scored rows (fewer than 2,000)"}
    y = sc["y_logit"].to_numpy(float)
    res = {"scored_rows": int(len(sc)), "events": int(sc["event_id"].nunique()),
           "test_months": sorted(sc["month"].unique()),
           "bf": bf.skill_ci(sc["day"].to_numpy(), y, sc["p0"].to_numpy(float)),
           "bf_lab": {**bf.skill_ci(sc["day"].to_numpy(), y, sc["p1"].to_numpy(float), base=sc["p0"].to_numpy(float)),
                      **bf.describe(y, sc["p1"].to_numpy(float))},
           "by_horizon": {}, "by_role": {}}
    for cp, s in sc.groupby("cp_min"):
        if len(s) >= 500:
            res["by_horizon"][str(cp)] = bf.skill_ci(s["day"].to_numpy(), s["y_logit"].to_numpy(float),
                                                     s["p1"].to_numpy(float), base=s["p0"].to_numpy(float))
    for r, s in sc.groupby("role"):
        if len(s) >= 500:
            res["by_role"][r] = bf.skill_ci(s["day"].to_numpy(), s["y_logit"].to_numpy(float),
                                            s["p1"].to_numpy(float), base=s["p0"].to_numpy(float))
    tot = sum(gain.values()) or 1.0
    res["gain_share_lab_last_fold"] = round(sum(v for c, v in gain.items() if c.startswith("lab_")) / tot, 4)
    lab_gain = sorted(((c, v) for c, v in gain.items() if c.startswith("lab_")), key=lambda kv: -kv[1])
    res["top_lab_features"] = {c: round(v / tot, 4) for c, v in lab_gain[:15]}
    every = d[np.isfinite(d["p0"]) & np.isfinite(d["p1"]) & d["schedule_fixed"]]
    res["trades_at_last_traded_prices_0.05"] = {
        "bf": bf.trades(every.assign(pred=every["p0"]), 0.05),
        "bf_lab": bf.trades(every.assign(pred=every["p1"]), 0.05)}
    return res


def run(store, cache: str) -> dict:
    d, fam = bf.load(store)
    if d.empty:
        return {"tag": TAG, "status": "no data found: gold/move is not built"}
    link = bf.read_rows(store, "gold/link", ["event_id", "match_id"], horizons=False)
    if link is None:
        return {"tag": TAG, "status": "no data found: gold/link is not built"}
    labt, lab_cols = lab_table(cache)
    d = d[d["cp_fresh"]].reset_index(drop=True)
    j = join(d, link.to_pandas(), labt)
    base_cols = [c for cols in fam.values() for c in cols]
    res = {"tag": TAG, "rows_betfair": int(len(d)), "events_betfair": int(d["event_id"].nunique()),
           "rows_joined": int(len(j)), "events_joined": int(j["event_id"].nunique()),
           "features": {"bf": len(base_cols), "lab": len(lab_cols)}, "lab_blocks": LAB_BLOCKS,
           "lab_coverage_joined_rows": round(float(j[lab_cols].notna().mean().mean()), 4) if len(j) else None,
           "note": "development evidence: one Betfair year and a half; last traded prices; nothing is traded"}
    if j["month"].nunique() < 4:
        res["status"] = "no data found: fewer than four months of joined rows"
        return res
    res.update(compare(j, base_cols, lab_cols))
    res["keeps_its_place"] = bool(res.get("bf_lab", {}).get("lo95", -1) > 0)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True, help="the lab cache (python -m football.lab cache)")
    ap.add_argument("--root", help="a local copy of the warehouse in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--stack", action="store_true", help="the follow-up: the lab's forecast as one feature")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from sources.common import Store
    store = Store(root=a.root) if a.root else Store()
    res = run_stack(store, a.cache) if a.stack else run(store, a.cache)
    from pathlib import Path
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k in ("tag", "status", "rows_joined", "bf", "bf_lab",
                                                             "keeps_its_place", "primary_360_and_120")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
