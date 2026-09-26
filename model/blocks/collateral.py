"""Collateral form through common opponents: how each runner compares with today's rivals by way of horses both have met.

head_to_head reads today's runners' direct meetings, which are rare: most pairs
in a race have never met. A handicapper reads them through a third horse: if A
beat C by 5 lb and B beat C by 1 lb, A is about 4 lb better than B on that
line. So, from each runner's last three runs and each rival's last three (days
before today only), every opponent the two have both met gives a line,

    implied margin of the runner over the rival
        = its margin over the opponent - the rival's margin over the opponent

in pounds (lengths beaten at the trip, capped at 20 lb a runner, as
model/form_windows.run_measures), positive when the runner comes out better.
The lines through several opponents are averaged for each rival, and then, for
each runner, over the rivals it is linked to:

    cl_rivals       today's rivals it is linked to through a common opponent
    cl_links        the lines in all (a rival met through two opponents counts two)
    cl_lbs          its mean implied margin over the linked rivals, shrunk to 0 by
                    two rivals: the collateral form in pounds against this field
    cl_lbs_min      the margin against the rival it comes out worst against
    cl_lbs_max      the margin against the rival it comes out best against
    cl_ahead_share  the share of linked rivals it comes out ahead of, shrunk to
                    one half by one each way
    cl_net          linked rivals it comes out ahead of less those ahead of it

A runner, or an opponent, without a finishing distance in a race gives no line
there. NaN (for the margins) with no line; 0 for the counts. Today's results
never enter: every line comes from runs on days before the row's day, summed
in a fixed order, so a 06:00 card and the same rows with results in agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.form_windows import run_measures
from model.freshness_features import _Days
from model.race_shape import day_index, race_key

FEATURES = ["cl_rivals", "cl_links", "cl_lbs", "cl_lbs_min", "cl_lbs_max", "cl_ahead_share", "cl_net"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "number_of_runners", "placing_numerical",
         "LB", "total_dst_bt", "dist_furlongs"]
LAST = 3            # each horse's runs that give lines
K_RIVALS = 2.0      # shrinkage of the mean margin, in rivals


def build(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy() | (day < 0)
    horse = np.where(bad, -1, pd.factorize(name, sort=True)[0]).astype(np.int64)   # codes independent of row order
    race_rows = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    lbs_rows = run_measures(df)["lbs"]                  # pounds behind the winner, capped; NaN = no finish

    H = _Days(horse, day)
    nb = len(H.u)
    b = np.arange(nb)
    race = np.nan_to_num(H.per_block(race_rows, how="max"), nan=-1).astype(np.int64)
    lbs = H.per_block(lbs_rows, how="min")

    # every finisher of every race: (race, horse, pounds behind)
    fin = H.valid & np.isfinite(lbs)
    members = pd.DataFrame({"P": race[fin], "C": H.key[fin], "lc": lbs[fin]})

    # every block's lines: its last LAST races, each opponent that finished in them and its margin over it
    parts = []
    for k in range(1, LAST + 1):
        ok = H.valid & (H.pos >= k)
        q = b[ok]
        then = q - k
        keep = np.isfinite(lbs[then])
        parts.append(pd.DataFrame({"q": q[keep], "R": race[q[keep]], "P": race[then[keep]], "me": H.key[q[keep]],
                                   "lh": lbs[then[keep]]}))
    past = pd.concat(parts, ignore_index=True)
    links = past.merge(members, on="P", how="inner")
    links = links[links["C"].to_numpy() != links["me"].to_numpy()]
    links["m"] = links["lc"].to_numpy() - links["lh"].to_numpy()          # + = it finished ahead of the opponent
    links = links[["q", "R", "C", "m"]]
    # a runner that met the same opponent twice: its mean margin over it
    links = links.groupby(["R", "C", "q"], sort=True, as_index=False)["m"].mean()

    # opponents met by at least two of a race's runners give lines between them; races in chunks, so a
    # big handicap's many shared opponents never hold the whole history's pairs in memory at once
    size = links.groupby(["R", "C"], sort=False)["q"].transform("size").to_numpy()
    shared = links[size >= 2]
    races = np.unique(shared["R"].to_numpy())
    rivals_parts = []
    for chunk in np.array_split(races, max(1, len(races) // 5000)):
        part = shared[shared["R"].isin(chunk)]
        pair = part.merge(part, on=["R", "C"], suffixes=("1", "2"))
        pair = pair[pair["q1"].to_numpy() != pair["q2"].to_numpy()]
        pair = pair.assign(d=pair["m1"].to_numpy() - pair["m2"].to_numpy())[["q1", "q2", "C", "d"]]
        pair = pair.sort_values(["q1", "q2", "C"], kind="stable")
        rivals_parts.append(pair.groupby(["q1", "q2"], sort=True).agg(d=("d", "mean"), n=("d", "size"))
                            .reset_index())
    per_rival = (pd.concat(rivals_parts, ignore_index=True) if rivals_parts
                 else pd.DataFrame({"q1": [], "q2": [], "d": [], "n": []}))
    per_rival = per_rival.sort_values(["q1", "q2"], kind="stable")
    per_rival["up"] = (per_rival["d"].to_numpy() > 0).astype(float)
    per_rival["down"] = (per_rival["d"].to_numpy() < 0).astype(float)
    g = per_rival.groupby("q1", sort=True)
    stats = pd.DataFrame({
        "rivals": g.size().astype(float), "links": g["n"].sum().astype(float), "dsum": g["d"].sum(),
        "dmin": g["d"].min(), "dmax": g["d"].max(), "ahead": g["up"].sum(), "behind": g["down"].sum(),
    })
    idx = stats.index.to_numpy().astype(np.int64)

    def put(v, fill):
        out = np.full(nb, fill, dtype=float)
        out[idx] = v
        return np.where(H.valid, out, np.nan)

    rivals, links_n = put(stats["rivals"].to_numpy(), 0.0), put(stats["links"].to_numpy(), 0.0)
    ahead, behind = put(stats["ahead"].to_numpy(), 0.0), put(stats["behind"].to_numpy(), 0.0)
    dsum = put(stats["dsum"].to_numpy(), 0.0)
    cols = {
        "cl_rivals": rivals,
        "cl_links": links_n,
        "cl_lbs": np.where(rivals > 0, dsum / (rivals + K_RIVALS), np.nan),
        "cl_lbs_min": put(stats["dmin"].to_numpy(), np.nan),
        "cl_lbs_max": put(stats["dmax"].to_numpy(), np.nan),
        "cl_ahead_share": (ahead + 1.0) / (rivals + 2.0),
        "cl_net": ahead - behind,
    }
    new = pd.DataFrame({c: H.to_rows(v) for c, v in cols.items()}, index=df.index)[FEATURES]
    assert len(new) == n_rows
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
