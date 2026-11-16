"""Where the dog was bred: Ireland or Britain (the owner, 16 Nov 2026: "Test the GBGB origin field as a feature";
ledger greyhound-origin-batch43-1116).

GBGB serves each runner's country of origin (``dogOrigin``: ``Ir`` Irish-bred, ``Br`` British-bred, a few others) on
every result; sources/gbgb.py keeps it as ``origin``. It is a fact of the dog's registration, fixed before it ever
races and the same on every row, so a run reads the dog's origin from its own rows up to and including the race's day
(no result of the day is read: the field does not depend on one). The live card does not carry it yet
(greyhound/card.py): a release would carry it from the dog's last run or GBGB's dog record (sources/gbgb_dogs.py).

Columns: ``og_ir`` (1 Irish-bred, 0 British-bred, NaN otherwise or unknown), ``og_other`` (1 bred elsewhere),
``og_uk_days`` (the dog's earlier race days in GBGB's results), ``og_ir_new`` (``og_ir`` / (1 + ``og_uk_days``): an
Irish import with little British form), ``og_ir_vs`` (``og_ir`` less the mean of the other runners' in the race).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "og_"
SOURCE = "GBGB results (the dog's country of origin, dogOrigin: sources/gbgb.py)"
COLUMNS = {
    "og_ir": "1 when the dog is Irish-bred, 0 when British-bred (GBGB's dogOrigin), NaN otherwise",
    "og_other": "1 when the dog was bred outside Ireland and Britain",
    "og_uk_days": "the dog's earlier race days in GBGB's results",
    "og_ir_new": "og_ir / (1 + og_uk_days): an Irish import with little British form",
    "og_ir_vs": "og_ir less the mean of the other runners' og_ir in the race",
}


def code(origin) -> pd.Series:
    """'ir' / 'br' / 'other' / None from GBGB's dogOrigin."""
    def one(x):
        if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
            return None
        t = str(x).strip().lower()
        if not t:
            return None
        if t.startswith("ir"):
            return "ir"
        if t.startswith("br") or t in ("gb", "uk"):
            return "br"
        return "other"
    return pd.Series([one(x) for x in origin.tolist()], index=origin.index, dtype="object")


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    r = runs
    n = len(r)
    day = core.day(r)
    dog = pd.to_numeric(r.dog_id, errors="coerce")
    org = code(r["origin"]) if "origin" in r else pd.Series([None] * n, index=r.index, dtype="object")
    # the dog's origin as at the race's day: its last row on or before the day that has one (a registration fact)
    d = pd.DataFrame({"dog": dog.to_numpy(), "day": day.to_numpy(), "o": org.to_numpy(), "i": np.arange(n)})
    known = d[d.o.notna()].drop_duplicates(["dog", "day"], keep="last").sort_values("day")
    q = d.sort_values("day")
    m = pd.merge_asof(q, known[["dog", "day", "o"]].rename(columns={"o": "o_known"}), on="day", by="dog",
                      direction="backward", allow_exact_matches=True)
    o = np.empty(n, dtype=object)
    o[m.i.to_numpy()] = m.o_known.to_numpy()
    ir = np.where(o == "ir", 1.0, np.where(o == "br", 0.0, np.nan))
    other = np.where(o == "other", 1.0, np.where(pd.isna(o), np.nan, 0.0))
    # earlier race days (a card row is no run)
    card = r.is_card.to_numpy(bool) if "is_card" in r else np.zeros(n, bool)
    rd = pd.DataFrame({"dog": dog.to_numpy(), "day": day.to_numpy()})[~card].drop_duplicates()
    rd = rd.sort_values(["dog", "day"])
    rd["k"] = rd.groupby("dog").cumcount().astype(float)          # days before this one
    q2 = pd.DataFrame({"dog": dog.to_numpy(), "day": day.to_numpy(), "i": np.arange(n)}).sort_values("day")
    m2 = pd.merge_asof(q2, rd.sort_values("day")[["dog", "day", "k"]], on="day", by="dog", direction="backward",
                       allow_exact_matches=False)
    days = np.zeros(n)
    kk = m2.k.to_numpy()
    days[m2.i.to_numpy()] = np.where(np.isfinite(kk), kk + 1.0, 0.0)
    race = r.race_id.to_numpy()
    s = pd.Series(ir)
    tot = s.groupby(race).transform("sum").to_numpy()
    cnt = s.notna().groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vs = np.where(np.isfinite(ir) & (cnt > 1), ir - (tot - np.nan_to_num(ir)) / np.maximum(cnt - 1, 1), np.nan)
    out = {"og_ir": ir, "og_other": other, "og_uk_days": days, "og_ir_new": ir / (1.0 + days), "og_ir_vs": vs}
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy(),
                        **{c: np.asarray(out[c], float).astype(np.float32) for c in COLUMNS}}, index=r.index)
    vc = pd.Series(o).value_counts(dropna=False)
    notes = {"irish": int((o == "ir").sum()), "british": int((o == "br").sum()), "other": int((o == "other").sum()),
             "unknown": int(sum(x is None for x in o)), "codes_seen": {str(k): int(v) for k, v in
                                                              r["origin"].astype("string").value_counts().head(8).items()}
             if "origin" in r else {}, "rows": int(vc.sum())}
    return tab, notes
