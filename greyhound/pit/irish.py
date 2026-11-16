"""The dog's Irish form, from GreyhoundAPI's Irish results (the owner, 16 Nov 2026: "Use the Irish results from
GreyhoundAPI in the model"; ledger greyhound-irish-batch45-1116). 83% of GB runs are by Irish-bred dogs
(greyhound/pit/origin.py, 3,291,234 of 3,956,525), many of them raced in Ireland before they came over; GBGB's results
hold none of those runs.

**The link.** By the dog's name (``name_map``: GreyhoundAPI keeps a dog's GB and Irish records under different ids,
so the id map below reaches almost none of its Irish runs; names held by one dog on each side only, and only the
Irish runs from three years before the dog's first GB day on). The API's GB results
(``sources/greyhoundapi/runs_GB_<year>.parquet``) are matched to GBGB's runs on the local day, the canonical track
(greyhound/data.py ``canonical_track``), the off time to the minute and the trap; an API dog maps to the GBGB dog it
matches most often, when that is at least ``MAP_SHARE`` of its matched runs. The map is identity, not outcome.

**An Irish run's figures** (``sources/greyhoundapi/runs_IE_<year>.parquet``): the time figure, 100 x (the track and
trip's standard - ln time), the standard the mean ln winning time there over the ``STD_DAYS`` before the day (at least
``MIN_STD`` winners); the runner's own time (the API times every runner: no beaten distance needed), the adjusted time
where given; the first-sectional figure against the mean ln sectional there; won, placed (first two), the class number.

**The columns** (a GB race reads the dog's Irish runs on days strictly BEFORE the race's day): ``ie_n`` (Irish runs),
``ie_days_since`` (days since the last), ``ie_win_rate`` and ``ie_place_rate`` (shrunk toward 1/6 and 2/6 by ``K``
runs), ``ie_fig_last``, ``ie_fig_mean3``, ``ie_fig_best3``, ``ie_sec_mean3``, ``ie_class_last``, ``ie_recent``
(1 when the last Irish run is within ``RECENT`` days), ``ie_import`` (1 when it has Irish runs and at most
``IMPORT_GB`` earlier GB race days), ``ie_fig_vs`` (``ie_fig_mean3`` less the mean of the other runners' that have
one). A dog with no mapped Irish run has NaN (``ie_n`` 0).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "ie_"
SOURCE = ("GreyhoundAPI results (api.greyhoundapi.com, the owner's Live Plus plan: sources/greyhoundapi.py): Ireland "
          "for the form, GB to link its dog ids to GBGB's")
STD_DAYS = 365
MIN_STD = 20
MAP_SHARE = 0.8
K = 3.0
RECENT = 120
IMPORT_GB = 3
COLUMNS = {
    "ie_n": "the dog's Irish runs before the day (GreyhoundAPI)",
    "ie_days_since": "days since its last Irish run",
    "ie_win_rate": "its Irish win rate, shrunk toward 1/6 by 3 runs",
    "ie_place_rate": "its Irish first-two rate, shrunk toward 2/6 by 3 runs",
    "ie_fig_last": "the time figure of its last Irish run (percent against the Irish track and trip's standard)",
    "ie_fig_mean3": "the mean of its last three Irish time figures",
    "ie_fig_best3": "the best of its last three Irish time figures",
    "ie_sec_mean3": "the mean of its last three Irish first-sectional figures",
    "ie_class_last": "the class number of its last Irish run",
    "ie_recent": "1 when its last Irish run was within 120 days",
    "ie_import": "1 when it has Irish runs and at most 3 earlier GB race days",
    "ie_fig_vs": "ie_fig_mean3 less the mean of the other runners' in today's race",
}
API_COLS = ["race_id", "dog_id", "track", "start_local", "trap", "position", "run_time_s", "adjusted_time_s",
            "sectional_s", "distance_m", "class_number"]


def load_context(store) -> dict:
    """The API's Irish and GB runner tables (the columns used), from the sources store."""
    out = {}
    for reg in ("IE", "GB"):
        keys = sorted(k for k in store.listing("greyhoundapi/") if re.search(rf"greyhoundapi/runs_{reg}_\d{{4}}\.parquet$", k))
        parts = []
        for k in keys:
            t = store.get_parquet(k)
            if t is not None and len(t):
                parts.append(t[[c for c in API_COLS if c in t.columns]])
        out[f"api_{reg.lower()}"] = pd.concat(parts, ignore_index=True) if parts else None
    return out


def _local(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    s = df.start_local.astype(str)
    return s.str[:10], s.str[11:16]


def dog_map(api_gb: pd.DataFrame, runs: pd.DataFrame) -> tuple[dict, dict]:
    """API dog id -> GBGB dog id, from the GB races both hold (day, canonical track, off to the minute, trap)."""
    from greyhound.data import canonical_track
    if api_gb is None or not len(api_gb):
        return {}, {"api_gb_rows": 0}
    a = api_gb.copy()
    a["day"], a["hhmm"] = _local(a)
    a["trk"] = canonical_track(a.track.astype(str)).str.lower()
    a["trap"] = pd.to_numeric(a.trap, errors="coerce")
    g = runs[["race_date", "race_time", "track", "trap", "dog_id"]].copy()
    g["day"] = g.race_date.astype(str).str[:10]
    g["hhmm"] = g.race_time.astype(str).str[:5]
    g["trk"] = canonical_track(g.track.astype(str)).str.lower()
    g["trap"] = pd.to_numeric(g.trap, errors="coerce")
    m = a[["dog_id", "day", "hhmm", "trk", "trap"]].rename(columns={"dog_id": "api_dog"}).merge(
        g[["day", "hhmm", "trk", "trap", "dog_id"]], on=["day", "hhmm", "trk", "trap"], how="inner")
    if not len(m):
        return {}, {"api_gb_rows": int(len(a)), "matched_rows": 0}
    c = m.groupby(["api_dog", "dog_id"]).size().rename("n").reset_index()
    tot = c.groupby("api_dog").n.transform("sum")
    best = c[c.n == c.groupby("api_dog").n.transform("max")].drop_duplicates("api_dog")
    best = best.assign(share=best.n / tot.loc[best.index].to_numpy())
    keep = best[best.share >= MAP_SHARE]
    mp = dict(zip(keep.api_dog.to_numpy(), keep.dog_id.astype("int64").to_numpy()))
    return mp, {"api_gb_rows": int(len(a)), "matched_rows": int(len(m)),
                "match_share_of_api_gb": round(len(m) / len(a), 4), "dogs_mapped": len(mp),
                "dogs_dropped_ambiguous": int(len(best) - len(keep))}


PRE_GB_DAYS = 1095


def name_map(ie: pd.DataFrame, runs: pd.DataFrame) -> tuple[dict, pd.Series, dict]:
    """API Irish dog id -> GBGB dog id by the dog's name (lower case, letters and digits), for names held by ONE dog on
    each side; with each GBGB dog's first GB day. GreyhoundAPI keeps a dog's GB and Irish records under different ids
    (16 Nov 2026: "Me Bundee" 4253978596 with 52 runs and 1512063880 with 15), so the id map does not reach its Irish
    runs; an Irish-bred dog keeps its registered name when it comes over."""
    if "dog_name" not in runs or ie is None or "dog_name" not in ie:
        return {}, pd.Series(dtype="datetime64[ns]"), {"name_links": 0}
    g = runs[["dog_id", "dog_name", "race_date"]].dropna(subset=["dog_name"]).copy()
    g["k"] = g.dog_name.map(core.norm_name)
    g = g[g.k != ""]
    gk = g.groupby("k").dog_id.nunique()
    g = g[g.k.isin(gk[gk == 1].index)]
    gb_dog = g.drop_duplicates("k").set_index("k").dog_id
    first = pd.to_datetime(g.groupby("dog_id").race_date.min())
    a = ie[["dog_id", "dog_name"]].dropna().drop_duplicates()
    a["k"] = a.dog_name.map(core.norm_name)
    ak = a.groupby("k").dog_id.nunique()
    a = a[a.k.isin(ak[ak == 1].index) & a.k.isin(gb_dog.index)].drop_duplicates("k")
    mp = dict(zip(a.dog_id.to_numpy(), gb_dog.loc[a.k].astype("int64").to_numpy()))
    return mp, first, {"name_links": len(mp), "gb_names_unique": int((gk == 1).sum()),
                       "gb_names_shared": int((gk > 1).sum()), "ie_names_shared": int((ak > 1).sum())}


def run_figures(api: pd.DataFrame) -> pd.DataFrame:
    """Every API run with its day, won, placed, class, time figure and first-sectional figure (standards from every
    run of the same track and trip, earlier days only)."""
    d = api.copy()
    d["race_date"], _ = _local(d)
    d["pos"] = pd.to_numeric(d.position, errors="coerce")
    t = pd.to_numeric(d.adjusted_time_s, errors="coerce").fillna(pd.to_numeric(d.run_time_s, errors="coerce"))
    d["_ltime"] = np.log(t.where(t > 5))
    sec = pd.to_numeric(d.sectional_s, errors="coerce")
    d["_lsec"] = np.log(sec.where((sec > 1) & (sec < 15)))
    d["_key"] = d.track.astype(str) + "|" + pd.to_numeric(d.distance_m, errors="coerce").round().astype("Int64").astype(str)
    win = (d.pos == 1) & d._ltime.notna()
    d["_lw"], d["_w"] = np.where(win, d._ltime, 0.0), win.astype(float)
    d["_ls"], d["_sn"] = np.nan_to_num(d._lsec), d._lsec.notna().astype(float)
    w = core.prior_window(d[["race_date", "_key", "_lw", "_w", "_ls", "_sn"]], ["_key"], ["_lw", "_w", "_ls", "_sn"],
                          STD_DAYS)
    std = np.where(w._w >= MIN_STD, w._lw / w._w.replace(0, np.nan), np.nan)
    spar = np.where(w._sn >= 30, w._ls / w._sn.replace(0, np.nan), np.nan)
    d["fig"] = 100.0 * (std - d._ltime)
    d["sec"] = 100.0 * (spar - d._lsec)
    d = d[d.pos.notna()].copy()
    d["won"], d["placed"] = (d.pos == 1).astype(float), (d.pos <= 2).astype(float)
    d["cls"] = pd.to_numeric(d.class_number, errors="coerce") if "class_number" in d else np.nan
    return d[["dog_id", "race_date", "won", "placed", "fig", "sec", "cls"]]


def irish_runs(ie: pd.DataFrame, mp: dict, first: pd.Series | None = None) -> pd.DataFrame:
    """Each Irish run of a mapped dog with its figures; with ``first`` (a GBGB dog's first GB day), only the Irish runs
    from ``PRE_GB_DAYS`` before it on (a name reused by another dog years earlier is not read)."""
    d = run_figures(ie)
    d["gb_dog"] = d.dog_id.map(mp)
    d = d[d.gb_dog.notna()].copy()
    d["gb_dog"] = d.gb_dog.astype("int64")
    if first is not None and len(first):
        start = d.gb_dog.map(first) - pd.Timedelta(days=PRE_GB_DAYS)
        d = d[pd.to_datetime(d.race_date) >= start.fillna(pd.Timestamp("1900-01-01"))]
    return d[["gb_dog", "race_date", "won", "placed", "fig", "sec", "cls"]]


def history(irr: pd.DataFrame) -> pd.DataFrame:
    """Per dog and Irish day: the running counts and the last-three figures through that day."""
    per = irr.groupby(["gb_dog", "race_date"], as_index=False).agg(
        n=("won", "size"), won=("won", "sum"), placed=("placed", "sum"), fig=("fig", "mean"), sec=("sec", "mean"),
        cls=("cls", "last")).sort_values(["gb_dog", "race_date"])
    g = per.groupby("gb_dog")
    out = pd.DataFrame({"gb_dog": per.gb_dog, "day": pd.to_datetime(per.race_date)})
    out["n"] = g.n.cumsum().to_numpy()
    out["won"] = g.won.cumsum().to_numpy()
    out["placed"] = g.placed.cumsum().to_numpy()
    out["fig_last"] = per.fig.to_numpy()
    out["fig_mean3"] = g.fig.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).reindex(per.index).to_numpy()
    out["fig_best3"] = g.fig.rolling(3, min_periods=1).max().reset_index(level=0, drop=True).reindex(per.index).to_numpy()
    out["sec_mean3"] = g.sec.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).reindex(per.index).to_numpy()
    out["cls_last"] = per.cls.to_numpy()
    out["last_day"] = out.day
    return out


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    ctx = ctx or {}
    r = runs
    n = len(r)
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy()}, index=r.index)
    for c in COLUMNS:
        tab[c] = np.float32(np.nan)
    tab["ie_n"] = np.float32(0.0)
    ie, gb = ctx.get("api_ie"), ctx.get("api_gb")
    if ie is None or not len(ie):
        return tab, {"note": "no data found: no GreyhoundAPI Irish results in the store"}
    mp, mnotes = dog_map(gb, r)
    nmp, first, nnotes = name_map(ie, r)
    mnotes.update(nnotes)
    both = {**nmp, **{k: v for k, v in mp.items()}}          # the id map where it reaches (rarely: see name_map)
    irr = irish_runs(ie, both, first)
    if not len(irr):
        return tab, {**mnotes, "note": "no Irish run of a mapped dog"}
    h = history(irr)
    day = core.day(r)
    q = pd.DataFrame({"gb_dog": pd.to_numeric(r.dog_id, errors="coerce").astype("int64").to_numpy(),
                      "day": day.to_numpy(), "i": np.arange(n)}).sort_values("day")
    m = pd.merge_asof(q, h.sort_values("day"), on="day", by="gb_dog", allow_exact_matches=False, direction="backward")
    m = m.set_index("i").sort_index()
    has = m.n.notna().to_numpy()
    nn = m.n.fillna(0).to_numpy(float)
    tab["ie_n"] = nn.astype(np.float32)
    tab["ie_days_since"] = np.where(has, (day.to_numpy() - m.last_day.to_numpy()) / np.timedelta64(1, "D"), np.nan)
    tab["ie_win_rate"] = np.where(has, (m.won.to_numpy() + K / 6.0) / (nn + K), np.nan)
    tab["ie_place_rate"] = np.where(has, (m.placed.to_numpy() + 2.0 * K / 6.0) / (nn + K), np.nan)
    for c in ("fig_last", "fig_mean3", "fig_best3", "sec_mean3"):
        tab[f"ie_{c}"] = m[c].to_numpy(float)
    tab["ie_class_last"] = m.cls_last.to_numpy(float)
    tab["ie_recent"] = np.where(has, (tab.ie_days_since.to_numpy() <= RECENT).astype(float), np.nan)
    # earlier GB race days (a card row is no run)
    card = r.is_card.to_numpy(bool) if "is_card" in r else np.zeros(n, bool)
    gd = pd.DataFrame({"dog": q.sort_index().gb_dog.to_numpy(), "day": day.to_numpy()})[~card].drop_duplicates()
    gd = gd.sort_values(["dog", "day"])
    gd["k"] = gd.groupby("dog").cumcount().astype(float)
    q2 = pd.DataFrame({"dog": pd.to_numeric(r.dog_id).astype("int64").to_numpy(), "day": day.to_numpy(),
                       "i": np.arange(n)}).sort_values("day")
    m2 = pd.merge_asof(q2, gd.sort_values("day"), on="day", by="dog", direction="backward", allow_exact_matches=False)
    gbdays = np.zeros(n)
    kk = m2.k.to_numpy()
    gbdays[m2.i.to_numpy()] = np.where(np.isfinite(kk), kk + 1.0, 0.0)
    tab["ie_import"] = np.where(has, (gbdays <= IMPORT_GB).astype(float), np.nan)
    s = pd.Series(tab.ie_fig_mean3.to_numpy(float))
    race = r.race_id.to_numpy()
    tot = s.groupby(race).transform("sum").to_numpy()
    cnt = s.notna().groupby(race).transform("sum").to_numpy()
    x = s.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        tab["ie_fig_vs"] = np.where(np.isfinite(x) & (cnt > 1), x - (tot - np.nan_to_num(x)) / np.maximum(cnt - 1, 1), np.nan)
    for c in COLUMNS:
        tab[c] = tab[c].astype(np.float32)
    notes = {**mnotes, "irish_runs_mapped": int(len(irr)), "gb_rows_with_irish_form": int(has.sum()),
             "gb_rows_imports": int(np.nansum(tab.ie_import.to_numpy(float))),
             "irish_figs": int(irr.fig.notna().sum())}
    return tab, notes
