"""The public's R-squared in today's segments: how much the Betfair SP has said about winners in races like this.

R2 = 1 - sum ln(p_winner) / sum ln(1 / runners) over the winners of a set of races (the logarithmic public
R-squared, as Benter used it): 0 when the price says no more than the field size. It differs by segment and the
differences persist (ledger public-r2-1112: handicaps 0.134 against 0.313 for non-handicaps; the tracks' gaps
correlate +0.61 between 2022-23 and 2024-26). A low figure means a race of this kind is less settled by the price:
partly the race itself (a handicap is made to be close), partly the market. Either way it says how far a forecast
of the price can lean on what is usually known.

Over earlier days only, decayed with a half-life of HALF days, each as a gap to all races' figure on the same days:

    pr_all        all races' R2 (the baseline)
    pr_track  pr_code  pr_hcap  pr_class  pr_month  pr_field
                  today's course, code, handicap or not, class, month and field-size band, less pr_all
    pr_sum        the sum of the six gaps
    pr_tr  pr_jk  pr_sire
                  the same figure over the winners of today's trainer, jockey and sire (at least MIN_WINS), less
                  pr_all: connections whose winners the price finds easily, or does not
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _col
from model.race_shape import asof_decayed_mean, day_index, race_code, race_key

SEGS = ("track", "code", "hcap", "class", "month", "field")
WHO = (("tr", "trainer"), ("jk", "jockey_name"), ("sire", "stallion"))
FEATURES = ["pr_all"] + [f"pr_{s}" for s in SEGS] + ["pr_sum"] + [f"pr_{w}" for w, _ in WHO]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "race_type", "race_name", "race_class",
         "surface_type", "number_of_runners", "bfsp", "placing_numerical", "trainer", "jockey_name", "stallion"]
HALF = 730.0
MIN_RACES = 30.0
MIN_WINS = 10.0
TEST_REACH = {"MIN_RACES": 3.0, "MIN_WINS": 1.0}
_HCAP_RX = r"handicap|nursery|\bh'cap\b|\bhcap\b"


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    race = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    inv = pd.Series(np.where(priced, 1.0 / np.where(priced, bsp, 1.0), 0.0), index=df.index)
    tot = inv.groupby(race).transform("sum").to_numpy()
    rows = pd.Series(1.0, index=df.index).groupby(race).transform("sum").to_numpy()
    npr = pd.Series(priced.astype(float), index=df.index).groupby(race).transform("sum").to_numpy()
    win = (pos == 1) & priced & (npr >= 0.9 * rows) & (rows >= 2) & (tot > 0)
    wins = pd.Series(win.astype(float), index=df.index).groupby(race).transform("sum").to_numpy()
    win &= wins == 1                                                     # one winner a race; a dead heat is left out
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.where(win, np.log(inv.to_numpy() / np.where(tot > 0, tot, 1.0)), np.nan)
        lf = np.where(win, np.log(1.0 / rows), np.nan)
    text = (_col(df, "race_type").fillna("").astype(str) + " " + _col(df, "race_name").fillna("").astype(str)).str.lower()
    seg = {"track": _codes(_col(df, "track")),
           "code": pd.factorize(race_code(df), sort=True)[0].astype(np.int64),
           "hcap": text.str.contains(_HCAP_RX).to_numpy().astype(np.int64),
           "class": pd.to_numeric(_col(df, "race_class").astype(str).str.extract(r"(\d)")[0], errors="coerce").fillna(0)
           .to_numpy().astype(np.int64),
           "month": pd.to_datetime(df["race_date"], errors="coerce").dt.month.fillna(0).to_numpy().astype(np.int64),
           "field": np.digitize(np.nan_to_num(rows, nan=0.0), [6, 9, 12, 16]).astype(np.int64)}

    def r2(key, least=None):
        least = MIN_RACES if least is None else least
        a, n = asof_decayed_mean(key, day, lp, key, day, halflife_days=HALF)
        b, _ = asof_decayed_mean(key, day, lf, key, day, halflife_days=HALF)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((n >= least) & (b < 0), 1.0 - a / b, np.nan)

    base = r2(np.zeros(len(df), dtype=np.int64))
    cols = {"pr_all": base}
    total = np.zeros(len(df))
    for s in SEGS:
        g = r2(seg[s]) - base
        cols[f"pr_{s}"] = g
        total += np.nan_to_num(g)
    cols["pr_sum"] = np.where(np.isfinite(base), total, np.nan)
    for w, col in WHO:
        cols[f"pr_{w}"] = r2(_codes(_col(df, col)), MIN_WINS) - base
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
