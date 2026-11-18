"""Why the GBGB-to-Betfair-BSP join fails before 2021, and what the duplicate (race, dog) rows are (read-only; the
owner, 18 Nov 2026: "fix the duplicates and backfill the BSP match"; reports/greyhound_data_audit.md findings 1, 3).

Per year, on the price table (``greyhound_prices/prices_<y>.parquet``, GB win) against the cleaned GBGB runs:
the share of price rows with no off time or no trap, sample raw ``event_dt`` values, the track names on each side and
their overlap, and how far down the join's keys the match survives: date; date and track; date, track and trap; then
the off time - for each GBGB race, the nearest price-file off time at the same date and track, its offset in minutes.
For the duplicates: per year, duplicates among non-null dog ids, whether the duplicated rows are identical, and a
sample of groups.

    python -m research.greyhound.join_diag --years 2015,2018,2019,2021,2023,2025 --out out/study
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


def join_year(store, y: int) -> dict:
    from greyhound.data import canonical_track, clean
    raw = store.get(f"gbgb/runs_{y}.parquet")
    p = store.get_parquet(f"greyhound_prices/prices_{y}.parquet")
    if raw is None or p is None:
        return {"note": "no data found"}
    runs = clean(pd.read_parquet(io.BytesIO(raw)))
    runs = runs[~runs.is_trial]
    w = p[(p.market == "win") & (p.country == "GB")].copy()
    out = {"price_rows": int(len(w)), "no_time_share": round(float(w.race_time.isna().mean()), 4),
           "no_trap_share": round(float(w.trap.isna().mean()), 4),
           "event_dt_samples": w["event_dt"].dropna().astype(str).drop_duplicates().head(6).tolist() if "event_dt" in w else None,
           "menu_hint_samples": w["menu_hint"].dropna().astype(str).drop_duplicates().head(6).tolist() if "menu_hint" in w else None,
           "selection_name_samples": w["selection_name"].dropna().astype(str).drop_duplicates().head(6).tolist()
           if "selection_name" in w else None}
    w["track"] = canonical_track(w.track)
    pt, rt = set(w.track.dropna()), set(runs.track.dropna())
    out["tracks_price_only"] = sorted(pt - rt)[:25]
    out["tracks_gbgb_only"] = sorted(rt - pt)[:25]
    out["tracks_common"] = len(pt & rt)
    races = runs.drop_duplicates("race_id")[["race_id", "race_date", "race_time", "track"]]
    pdays = set(w.race_date)
    out["gbgb_races"] = int(len(races))
    out["share_date_in_prices"] = round(float(races.race_date.isin(pdays).mean()), 4)
    dt = set(zip(w.race_date, w.track))
    out["share_date_track_in_prices"] = round(float(pd.Series(list(zip(races.race_date, races.track))).isin(dt).mean()), 4)
    exact = set(zip(w.race_date, w.track, w.race_time))
    out["share_date_track_time_exact"] = round(float(pd.Series(list(zip(races.race_date, races.track,
                                                                          races.race_time))).isin(exact).mean()), 4)
    # the nearest price off time at the same date and track
    wm = w.dropna(subset=["race_time"]).drop_duplicates(["race_date", "track", "race_time"])
    wm = wm.assign(m=pd.to_datetime(wm.race_date + " " + wm.race_time))
    rr = races.assign(m=pd.to_datetime(races.race_date + " " + races.race_time.astype(str).str[:5], errors="coerce"))
    rr = rr.dropna(subset=["m"]).sort_values("m")
    j = pd.merge_asof(rr, wm[["m", "race_date", "track"]].rename(columns={"m": "pm"}).sort_values("pm"),
                      left_on="m", right_on="pm", by=["race_date", "track"], direction="nearest")
    off = ((j.pm - j.m).dt.total_seconds() / 60).dropna().round()
    out["nearest_offset_minutes_top"] = {str(int(k)): int(v) for k, v in off.value_counts().head(12).items()}
    out["races_with_a_price_race_same_day_track"] = int(off.notna().sum())
    return out


def dups_year(store, y: int) -> dict:
    raw = store.get(f"gbgb/runs_{y}.parquet")
    if raw is None:
        return {"note": "no data found"}
    r = pd.read_parquet(io.BytesIO(raw))
    nn = r.dropna(subset=["dog_id"])
    d = nn[nn.duplicated(["race_id", "dog_id"], keep=False)]
    cols = [c for c in ("race_id", "race_date", "race_time", "track", "race_class", "dog_id", "dog_name", "trap",
                        "position", "sp_decimal", "run_time", "beaten", "meeting_id") if c in r]
    groups = d.groupby(["race_id", "dog_id"])
    same = int(sum(1 for _, g in groups if len(g[cols].drop_duplicates()) == 1))
    differ = {}
    for _, g in groups:
        for c in cols:
            if g[c].astype(str).nunique() > 1:
                differ[c] = differ.get(c, 0) + 1
    return {"rows": int(len(r)), "null_dog_id": int(r.dog_id.isna().sum()),
            "dup_rows_nonnull": int(nn.duplicated(["race_id", "dog_id"]).sum()), "dup_groups": int(groups.ngroups),
            "groups_identical": same, "columns_that_differ_in_groups": differ,
            "races_with_dups": int(d.race_id.nunique()),
            "sample": d.sort_values(["race_id", "dog_id"])[cols].head(16).astype(str).to_dict("records"),
            "null_id_sample": r[r.dog_id.isna()][cols].head(8).astype(str).to_dict("records")}


def aliases(store, years: list[int]) -> dict:
    """Each abbreviated price-file track label -> the GBGB track it names, learnt by date and exact off time: a plain
    GB win market at (date, off) is paired with the GBGB races at the same date and off, and each label takes the
    track most of its markets pair with. Labels with at least 20 paired markets and 80% on one track are kept."""
    from greyhound.data import clean
    from sources.greyhound_prices import market_label
    pairs = []
    for y in years:
        raw, p = store.get(f"gbgb/runs_{y}.parquet"), store.get_parquet(f"greyhound_prices/prices_{y}.parquet")
        if raw is None or p is None or "menu_hint" not in p:
            continue
        hints = pd.Series(p.menu_hint.astype(str).unique())
        lab = pd.DataFrame([market_label(h) for h in hints], columns=["label", "variant", "country"]).assign(menu_hint=hints)
        w = p[p.market.astype(str).str.startswith("win")].merge(lab, on="menu_hint", how="left")
        w = w[(w.variant == "") & (w.country == "GB")].dropna(subset=["race_time"])
        mk = w.drop_duplicates(["race_date", "race_time", "label"])[["race_date", "race_time", "label"]]
        runs = clean(pd.read_parquet(io.BytesIO(raw)))
        rc = runs[~runs.is_trial].drop_duplicates("race_id")[["race_date", "race_time", "track"]]
        rc = rc.assign(race_time=rc.race_time.astype(str).str[:5])
        pairs.append(mk.merge(rc, on=["race_date", "race_time"], how="inner"))
    if not pairs:
        return {"note": "no data found"}
    x = pd.concat(pairs, ignore_index=True)
    ct = x.groupby(["label", "track"]).size().rename("n").reset_index()
    tot = ct.groupby("label").n.transform("sum")
    best = ct.assign(tot=tot, share=ct.n / tot).sort_values(["label", "n"], ascending=[True, False]).drop_duplicates("label")
    out = {r.label: {"track": r.track, "markets": int(r.tot), "share": round(float(r.share), 3)}
           for r in best.itertuples()}
    return {"labels": out, "kept": {k: v["track"] for k, v in out.items() if v["markets"] >= 20 and v["share"] >= 0.8
                                    and k != v["track"]}}


def dups_after_clean(store, y: int) -> dict:
    """The (race, dog) duplicates clean leaves: how many, how many with a result on both rows, and whether the other
    row's trap is held by another dog in the race."""
    from greyhound.data import clean
    raw = store.get(f"gbgb/runs_{y}.parquet")
    if raw is None:
        return {"note": "no data found"}
    c = clean(pd.read_parquet(io.BytesIO(raw)))
    d = c[c.duplicated(["race_id", "dog_id"], keep=False)]
    occ = c.groupby("race_id").trap.apply(lambda t: list(t)).to_dict()
    both = d.groupby(["race_id", "dog_id"]).position.apply(lambda p: int(p.notna().sum())).value_counts().to_dict()
    clash = 0
    for (rid, did), g in d.groupby(["race_id", "dog_id"]):
        traps = list(g.trap)
        clash += int(any(occ[rid].count(t) > 1 for t in traps))
    return {"rows_out": int(len(c)), "dup_rows": int(c.duplicated(["race_id", "dog_id"]).sum()),
            "groups_by_rows_with_position": {str(k): int(v) for k, v in both.items()},
            "groups_whose_trap_is_shared": clash,
            "sample": d.sort_values(["race_id", "dog_id"])[["race_id", "dog_id", "trap", "position", "sp_decimal", "field"]]
            .head(10).astype(str).to_dict("records")}


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", default="2015,2018,2019,2021,2023,2025")
    ap.add_argument("--out", default="out/join_diag")
    ap.add_argument("--aliases", default=None, help="learn the track aliases on these years (e.g. 2014-2022)")
    a = ap.parse_args(argv)
    store = Store()
    if a.aliases:
        lo, hi = (int(v) for v in a.aliases.split("-"))
        al = aliases(store, list(range(lo, hi + 1)))
        print(json.dumps({"aliases": al}, default=str))
        post = {str(y): dups_after_clean(store, y) for y in (2015, 2019, 2023, 2025)}
        print(json.dumps({"dups_after_clean": post}, default=str))
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "aliases.json").write_text(json.dumps({"aliases": al, "dups_after_clean": post}, indent=1, default=str))
        return 0
    ys = [int(x) for x in a.years.split(",")]
    res = {"join": {}, "dups": {}}
    for y in ys:
        try:
            res["join"][str(y)] = join_year(store, y)
        except Exception as exc:
            res["join"][str(y)] = {"note": f"failed ({exc})"[:300]}
        try:
            res["dups"][str(y)] = dups_year(store, y)
        except Exception as exc:
            res["dups"][str(y)] = {"note": f"failed ({exc})"[:300]}
        print(json.dumps({"year": y, "join": res["join"][str(y)], "dups": {k: v for k, v in res["dups"][str(y)].items()
                                                                           if k not in ("sample", "null_id_sample")}},
                         default=str))
    for y in ys:
        print(json.dumps({"year": y, "dup_sample": res["dups"][str(y)].get("sample"),
                          "null_id_sample": res["dups"][str(y)].get("null_id_sample")}, default=str))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
