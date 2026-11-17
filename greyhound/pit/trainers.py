"""Each run's trainer AT THE TIME (the feature review of 17 Nov 2026, B3, B4, D1, D5). A helper module, not a table.

GBGB serves a dog's CURRENT trainer on every past result (core.trainer_at_the_time), so a trainer feature read from
it on old runs may look ahead (good dogs move to strong yards). The sources, in order:

1. **GreyhoundAPI's GB results** (``sources/greyhoundapi/runs_GB_<year>.parquet``, ``trainer_id`` and ``trainer`` on
   each runner, 2013 on): linked to GBGB's runs run by run on the local day, the canonical track, the off to the
   minute and the trap (the slot of greyhound/pit/irish.py ``dog_map``), the dog's name agreeing where both give one.
   Its trainer names are mapped to GBGB's (``map_names``: the normalised name, then the GBGB trainer it shares most
   runs with, then first initial and surname). Whether the API's trainer is the trainer on the day is **unverified**:
   research/greyhound/trainer_check.py measures it.
2. **GBGB's own trainer where the day's file was read within core.FRESH_DAYS of the race** (core.trainer_at_the_time).
3. **The card row** (``is_card``: today, no result, no fetch): the card's trainer (``bf_trainer``, Betfair's
   catalogue (D5), mapped to GBGB's names, where the runs carry it; else the card's ``trainer``, which
   greyhound/card.py takes from the dog's last GBGB run), else the dog's last known trainer at the time.

Everything else is NaN. The name maps are identity, not outcome (as irish.dog_map).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import core

#: the GreyhoundAPI columns the link reads
API_COLS = ["race_id", "dog_id", "dog_name", "track", "start_local", "trap", "trainer_id", "trainer"]
#: an API trainer maps to the GBGB trainer it shares most linked runs with when at least this share and count
CO_SHARE, CO_MIN = 0.6, 10
SRC = {"api": 1, "gbgb_fresh": 2, "card": 3, "carried": 4}


def load_context(store) -> dict:
    """``api_gb_trainer``: GreyhoundAPI's GB runner tables, the columns the link reads (sources store)."""
    keys = sorted(k for k in store.listing("greyhoundapi/") if re.search(r"greyhoundapi/runs_GB_\d{4}\.parquet$", k))
    parts = []
    for k in keys:
        t = store.get_parquet(k)
        if t is not None and len(t):
            parts.append(t[[c for c in API_COLS if c in t.columns]])
    return {"api_gb_trainer": pd.concat(parts, ignore_index=True) if parts else None}


def initial_key(s) -> str:
    """First initial and surname, letters only ("Mark A Wallis" -> mwallis)."""
    tok = re.findall(r"[a-z]+", str(s or "").lower())
    return tok[0][0] + tok[-1] if tok else ""


def link_runs(api: pd.DataFrame | None, runs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Each GBGB run's GreyhoundAPI row by the slot (local day, canonical track, off to the minute, trap), the names
    agreeing where both are given; one API row a run (a slot matched twice is dropped). Returns a frame on
    ``runs.index`` with ``api_trainer`` and ``api_trainer_id`` (NaN where unlinked), and notes."""
    from greyhound.data import canonical_track
    out = pd.DataFrame({"api_trainer": pd.Series(np.nan, index=runs.index, dtype=object),
                        "api_trainer_id": pd.Series(np.nan, index=runs.index, dtype=object)})
    if api is None or not len(api) or "trainer" not in api:
        return out, {"api_rows": 0, "runs_linked": 0}
    a = api.copy()
    s = a.start_local.astype(str)
    a["day"], a["hhmm"] = s.str[:10], s.str[11:16]
    a["trk"] = canonical_track(a.track.astype(str)).str.lower()
    a["trap"] = pd.to_numeric(a.trap, errors="coerce")
    a["nm"] = a.dog_name.map(core.norm_name) if "dog_name" in a else ""
    g = pd.DataFrame({"day": runs.race_date.astype(str).str[:10], "hhmm": runs.race_time.astype(str).str[:5],
                      "trk": canonical_track(runs.track.astype(str)).str.lower(),
                      "trap": pd.to_numeric(runs.trap, errors="coerce"),
                      "nm": runs.dog_name.map(core.norm_name) if "dog_name" in runs else "",
                      "_i": np.arange(len(runs))})
    m = g.merge(a[["day", "hhmm", "trk", "trap", "nm", "trainer", "trainer_id"]], on=["day", "hhmm", "trk", "trap"],
                how="inner", suffixes=("", "_api"))
    slot = len(m)
    agree = (m.nm == "") | (m.nm_api == "") | (m.nm == m.nm_api)
    m = m[agree]
    m = m[~m._i.duplicated(keep=False)]
    m = m[m.trainer.notna() & (m.trainer.astype(str).str.strip() != "")]
    out.iloc[m._i.to_numpy(), 0] = m.trainer.astype(str).str.strip().to_numpy()
    out.iloc[m._i.to_numpy(), 1] = m.trainer_id.astype(object).to_numpy()
    return out, {"api_rows": int(len(a)), "slot_matches": int(slot), "name_disagree": int((~agree).sum()),
                 "runs_linked": int(len(m)), "share_of_runs": round(len(m) / max(len(runs), 1), 4)}


def map_names(names: pd.Series, gbgb: pd.Series, cooccur: bool = True) -> tuple[pd.Series, dict]:
    """``names`` (another source's trainer, aligned with ``gbgb``, GBGB's trainer as served on the same runs; NaN where
    a run has none) to GBGB trainer names: the same normalised name; else the same first initial and surname when one
    GBGB trainer has it; else (``cooccur``) the GBGB trainer the name shares the most runs with (at least ``CO_SHARE``
    of its runs and ``CO_MIN`` runs: a dog rarely moves, so the served trainer is mostly right; a yard seen only
    through dogs that all went to one successor maps to the successor, so research/greyhound/trainer_check.py turns
    it off). Unmapped names stay as ``"?" + name`` (still one yard, a key of its own). Returns the mapped series and
    notes."""
    gb = gbgb.dropna().astype(str).str.strip()
    by_norm = gb.drop_duplicates().pipe(lambda s: pd.Series(s.to_numpy(), index=s.map(core.norm_name).to_numpy()))
    by_norm = by_norm[~by_norm.index.duplicated(keep=False)]
    ik = gb.drop_duplicates().pipe(lambda s: pd.Series(s.to_numpy(), index=s.map(initial_key).to_numpy()))
    ik = ik[~ik.index.duplicated(keep=False)]
    src = names.dropna().astype(str).str.strip()
    uniq = pd.Series(src.unique())
    mp, how = {}, {"name": 0, "cooccur": 0, "initial": 0, "unmapped": 0}
    pair = pd.DataFrame({"n": names, "g": gbgb}).dropna()
    co = {}
    if cooccur and len(pair):
        c = pair.groupby(["n", "g"]).size().rename("k").reset_index()
        tot = c.groupby("n").k.transform("sum")
        c = c.assign(share=c.k / tot).sort_values("k", ascending=False).drop_duplicates("n")
        c = c[(c.share >= CO_SHARE) & (c.k >= CO_MIN)]
        co = dict(zip(c.n.astype(str).str.strip(), c.g.astype(str).str.strip()))
    for n in uniq:
        k = core.norm_name(n)
        if k in by_norm.index:
            mp[n], w = by_norm[k], "name"
        elif initial_key(n) in ik.index:
            mp[n], w = ik[initial_key(n)], "initial"
        elif n in co:
            mp[n], w = co[n], "cooccur"
        else:
            mp[n], w = "?" + n, "unmapped"
        how[w] += 1
    mapped = names.astype(object).where(names.isna(), names.astype(str).str.strip().map(mp))
    runs_n = mapped.notna().sum()
    notes = {"names": int(len(uniq)), **{f"names_by_{k}": v for k, v in how.items()},
             "runs_mapped_share": round(float((mapped.notna() & ~mapped.astype(str).str.startswith("?")).sum()
                                              / max(runs_n, 1)), 4)}
    return mapped, notes


def at_the_time(runs: pd.DataFrame, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """``trainer`` (at the time, GBGB's names) and ``source`` (``SRC`` code; 0 unknown) for each run, on
    ``runs.index``. ``ctx``: ``fetch_lag`` (core.raw_fetch_lag), ``api_gb_trainer`` (``load_context``)."""
    ctx = ctx or {}
    lag = ctx.get("fetch_lag")
    card = runs["is_card"].fillna(False).astype(bool) if "is_card" in runs else pd.Series(False, index=runs.index)
    fresh = core.trainer_at_the_time(runs, lag).where(~card)
    link, ln = link_runs(ctx.get("api_gb_trainer"), runs)
    api_raw = link.api_trainer.where(~card)
    api, mn = map_names(api_raw, runs.trainer.where(~card)) if api_raw.notna().any() else (api_raw, {})
    tr = api.where(api.notna(), fresh)
    src = pd.Series(np.where(api.notna(), SRC["api"], np.where(fresh.notna(), SRC["gbgb_fresh"], 0)), index=runs.index)
    both = api.notna() & fresh.notna()
    notes = {"link": ln, "names": mn, "api_vs_fresh_runs": int(both.sum()),
             "api_vs_fresh_agree": round(float((api[both] == fresh[both]).mean()), 4) if both.any() else None}
    if card.any():
        ct = runs.trainer.where(card)
        if "bf_trainer" in runs:                                  # D5: Betfair's catalogue, the trainer on the day
            bf, bn = map_names(runs.bf_trainer.where(card), runs.trainer.where(~card))
            ct = bf.where(bf.notna() & ~bf.astype(str).str.startswith("?"), ct)
            notes["bf_names"] = bn
        tr = tr.where(~card, ct)
        src = src.where(~(card & ct.notna()), SRC["card"])
        last = carried(runs, tr.where(~card))
        miss = card & tr.isna() & last.notna()
        tr = tr.where(~miss, last)
        src = src.where(~miss, SRC["carried"])
    notes.update({"known_share": round(float(tr.notna().mean()), 4),
                  "by_source": {k: int((src == v).sum()) for k, v in SRC.items()}})
    return pd.DataFrame({"trainer": tr, "source": src}, index=runs.index), notes


def carried(runs: pd.DataFrame, trainer: pd.Series) -> pd.Series:
    """The dog's last known trainer at the time on an EARLIER day (NaN with none)."""
    d = pd.DataFrame({"dog_id": runs.dog_id, "t": runs.t if "t" in runs else pd.to_datetime(runs.race_date),
                      "_tr": trainer, "_d": core.day(runs)})
    k = d.dropna(subset=["_tr"]).sort_values(["dog_id", "t"]).groupby(["dog_id", "_d"]).tail(1)
    k = k.sort_values("_d")[["dog_id", "_d", "_tr"]]
    q = d[["dog_id", "_d"]].assign(_i=np.arange(len(d))).dropna(subset=["_d"]).sort_values("_d")
    m = pd.merge_asof(q, k, on="_d", by="dog_id", allow_exact_matches=False, direction="backward")
    out = pd.Series(np.nan, index=runs.index, dtype=object)
    out.iloc[m._i.to_numpy()] = m._tr.to_numpy()
    return out
