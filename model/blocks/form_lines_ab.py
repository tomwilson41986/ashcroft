"""Form lines split by who finished ahead and who behind, and read against today's field.

form_lines pools every rival from a horse's recent races: how they have done
since. A race reader splits them. Rivals that finished behind the horse and
have won since say it beat winners; rivals that beat it and won since say its
defeat was by good horses. And a horse's form lines mean most against the
lines of the horses it meets today.

From the horse's last race, over its rivals' next runs since (each rival's next
three at most, on days before today), split by where the rival finished
against the horse (a rival that did not finish counts as behind):

    fa_ahead_n      runs since by rivals that finished ahead of it
    fa_ahead_wins   how many of them were won
    fa_ahead_ae     their wins less the market's chances at BSP, per run, shrunk to 0 by five runs
    fa_behind_n     runs since by rivals that finished behind it
    fa_behind_wins  how many of them were won
    fa_behind_ae    the same for them
    fa_beat_winner  1 if a rival it beat has won since, else 0

NaN before a horse's first run and when it did not finish its last race.

And form_lines' readings against today's field (z-score over the runners with
a value, and the gap to the best):

    fa_l1_ae_z, fa_l1_ae_gap   fl_l1_ae (the last race's rivals, wins less chances)
    fa_l3_ae_z, fa_l3_ae_gap   fl_l3_ae (the last three races')
    fa_l1_wr_z, fa_l3_wr_z     fl_l1_wr and fl_l3_wr (win rates)

Today's results never enter: the rivals' runs counted are on days before the
row's day, summed in a fixed order (by rival run date, then horse), so a card
and the same rows with results in agree to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import form_lines
from model.freshness_features import _Days, _col
from model.race_shape import day_index, race_key

SPLIT = ["fa_ahead_n", "fa_ahead_wins", "fa_ahead_ae", "fa_behind_n", "fa_behind_wins", "fa_behind_ae",
         "fa_beat_winner"]
FIELD = {"fa_l1_ae": "fl_l1_ae", "fa_l3_ae": "fl_l3_ae", "fa_l1_wr": "fl_l1_wr", "fa_l3_wr": "fl_l3_wr"}
FIELD_STATS = {"fa_l1_ae": ("z", "gap"), "fa_l3_ae": ("z", "gap"), "fa_l1_wr": ("z",), "fa_l3_wr": ("z",)}
FEATURES = SPLIT + [f"{k}_{s}" for k, stats in FIELD_STATS.items() for s in stats]
POST_RACE: set[str] = set()
AFTER = ("form_lines",)
READS = list(dict.fromkeys(form_lines.READS + list(FIELD.values())))
K = form_lines.K
NEXT = form_lines.NEXT


def _split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy() | (day < 0)
    horse = np.where(bad, -1, pd.factorize(name, sort=True)[0]).astype(np.int64)
    race_rows = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    pos_rows = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    pos_rows = np.where(pos_rows > 0, pos_rows, np.nan)
    vals = form_lines._values(df)

    H = _Days(horse, day)
    nb = len(H.u)
    b = np.arange(nb)
    race = np.nan_to_num(H.per_block(race_rows, how="max"), nan=-1).astype(np.int64)
    pos = H.per_block(pos_rows, how="max")
    won = H.per_block(vals["wins"], how="max")
    ae = H.per_block(vals["ae"], how="max")

    # every rival run: the race it came out of, where the rival finished there, the run's day and result
    parts = []
    for k in range(1, NEXT + 1):
        j = b + k
        ok = H.valid & (j < nb)
        ok[ok] = H.key[j[ok]] == H.key[b[ok]]
        parts.append(pd.DataFrame({"R": race[b[ok]], "rpos": pos[b[ok]], "ds": H.day[j[ok]], "h": H.key[j[ok]],
                                   "won": np.nan_to_num(won[j[ok]]), "ae": np.nan_to_num(ae[j[ok]])}))
    ev = pd.concat(parts, ignore_index=True)

    # every block after a horse's first: its last race, where it finished there, and today
    okq = H.valid & (H.pos >= 1)
    qb = b[okq]
    q = pd.DataFrame({"qid": qb, "R": race[qb - 1], "p": pos[qb - 1], "t": H.day[qb], "qh": H.key[qb]})
    q = q[np.isfinite(q["p"].to_numpy())]
    m = q.merge(ev, on="R", how="inner")
    m = m[(m["ds"].to_numpy() < m["t"].to_numpy()) & (m["h"].to_numpy() != m["qh"].to_numpy())]
    m = m.sort_values(["qid", "ds", "h"], kind="stable")
    ahead = (m["rpos"] < m["p"]).to_numpy()
    behind = (~ahead) & ((m["rpos"] > m["p"]) | m["rpos"].isna()).to_numpy()

    out = {c: np.full(nb, np.nan) for c in SPLIT}
    for side, mask in (("ahead", ahead), ("behind", behind)):
        g = m[mask].groupby("qid", sort=True)
        n = pd.Series(0.0, index=q["qid"]).add(g.size().astype(float), fill_value=0.0)
        wins = pd.Series(0.0, index=q["qid"]).add(g["won"].sum(), fill_value=0.0)
        aes = pd.Series(0.0, index=q["qid"]).add(g["ae"].sum(), fill_value=0.0)
        idx = n.index.to_numpy()
        out[f"fa_{side}_n"][idx] = n.to_numpy()
        out[f"fa_{side}_wins"][idx] = wins.reindex(n.index).to_numpy()
        out[f"fa_{side}_ae"][idx] = aes.reindex(n.index).to_numpy() / (n.to_numpy() + K)
    has = np.isfinite(out["fa_behind_wins"])
    out["fa_beat_winner"] = np.where(has, (np.nan_to_num(out["fa_behind_wins"]) > 0).astype(float), np.nan)
    return {c: H.to_rows(v) for c, v in out.items()}


def _field(df: pd.DataFrame) -> dict[str, np.ndarray]:
    race = pd.factorize(race_key(df), sort=True)[0]
    cols = {}
    for key, src in FIELD.items():
        x = pd.Series(pd.to_numeric(_col(df, src), errors="coerce").to_numpy(dtype=float))
        g = x.groupby(race)
        mean, sd, top = g.transform("mean").to_numpy(), g.transform("std").to_numpy(), g.transform("max").to_numpy()
        v = x.to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            if "z" in FIELD_STATS[key]:
                cols[f"{key}_z"] = np.where(np.isfinite(v) & (sd > 0), (v - mean) / sd,
                                            np.where(np.isfinite(v) & np.isfinite(mean), 0.0, np.nan))
            if "gap" in FIELD_STATS[key]:
                cols[f"{key}_gap"] = v - top
    return cols


def build(df: pd.DataFrame) -> pd.DataFrame:
    cols = {**_split(df), **_field(df)}
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
