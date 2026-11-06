"""One dog, one identity: each GBGB run matched to its Betfair runner, so GBGB's history joins every Betfair record
(the price files, the recorded books and the Exchange Stream, which know a dog only by its selection id).

A GBGB run and a Betfair runner are the same when they share the day, the off time, the track and the trap, and the
names agree (letters and digits only; a near match, ``difflib`` ratio ``NEAR`` or more, is kept and flagged). Each
GBGB dog's selection ids, and each selection id's dogs, are counted in the meta: a dog should have one, and an id
one dog. This table is a key, not a feature: no block reads it into a model.
"""

from __future__ import annotations

import difflib
import re

import pandas as pd

from greyhound.pit import core

PREFIX = "id_"
NEAR = 0.85
FEATURE = False
SOURCE = "GBGB results (sources/gbgb.py); Betfair's greyhound price files (sources/greyhound_prices.py, GB win)"


def load_runners(store, years: str = "2014-2026") -> pd.DataFrame | None:
    """The price files' GB win runners: day, off, track, trap, selection id and name."""
    keys = sorted(k for k in store.listing("greyhound_prices/") if re.search(r"prices_(\d{4})\.parquet$", k))
    lo, hi = (int(x) for x in years.split("-"))
    parts = []
    for k in keys:
        if not lo <= int(re.search(r"prices_(\d{4})", k).group(1)) <= hi:
            continue
        t = store.get_parquet(k)
        if t is None or not len(t):
            continue
        t = t[(t.market == "win") & (t.country == "GB")]
        parts.append(t[["race_date", "race_time", "track", "trap", "selection_id", "dog", "event_id"]].copy())
    return pd.concat(parts, ignore_index=True) if parts else None


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    bf = (ctx or {}).get("runners")
    base = runs[["race_id", "dog_id", "dog_name", "race_date", "race_time", "track", "trap"]].copy()
    if bf is None or not len(bf):
        out = base[["race_id", "dog_id"]].assign(id_selection_id=pd.NA, id_event_id=pd.NA, id_match="none")
        return out.reset_index(drop=True), {"note": "no data found: no Betfair runners to match"}
    bf = bf.dropna(subset=["trap", "selection_id"]).assign(trap=lambda x: x.trap.astype("int64"))
    bf = bf.drop_duplicates(["race_date", "race_time", "track", "trap"], keep="last")
    m = base.merge(bf, on=["race_date", "race_time", "track", "trap"], how="left")
    a, b = m.dog_name.map(core.norm_name), m.dog.map(core.norm_name)
    exact = (a == b) & (a != "")
    ratio = pd.Series([difflib.SequenceMatcher(None, x, y).ratio() if (x and y and x != y) else 0.0
                       for x, y in zip(a, b)], index=m.index)
    kind = pd.Series("none", index=m.index)
    kind[m.selection_id.notna()] = "slot_only"
    kind[m.selection_id.notna() & (ratio >= NEAR)] = "near"
    kind[m.selection_id.notna() & exact] = "exact"
    ok = kind.isin(["exact", "near"])
    out = pd.DataFrame({"race_id": m.race_id, "dog_id": m.dog_id,
                        "id_selection_id": m.selection_id.where(ok), "id_event_id": m.event_id.where(ok),
                        "id_match": kind})
    pairs = out[ok][["dog_id", "id_selection_id"]].drop_duplicates()
    ids_per_dog = pairs.groupby("dog_id").size()
    dogs_per_id = pairs.groupby("id_selection_id").size()
    covered = runs.race_date >= str(bf.race_date.min())
    meta = {"runs": int(len(out)), "runs_in_price_years": int(covered.sum()),
            "matched": {k: int(v) for k, v in kind[covered.to_numpy()].value_counts().items()},
            "match_rate_in_price_years": round(float(ok[covered.to_numpy()].mean()), 4) if covered.any() else None,
            "dogs": int(pairs.dog_id.nunique()), "dogs_with_more_than_one_id": int((ids_per_dog > 1).sum()),
            "ids_with_more_than_one_dog": int((dogs_per_id > 1).sum())}
    return out.reset_index(drop=True), meta


COLUMNS = {
    "id_selection_id": "the Betfair selection id of the dog in this race (exact or near name match only)",
    "id_event_id": "the Betfair market's event id",
    "id_match": "exact, near (name ratio >= 0.85), slot_only (the slot matched, the name did not), none",
}
