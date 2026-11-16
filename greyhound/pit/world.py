"""The sire's and dam's progeny abroad, from GreyhoundAPI's Irish and Australian results (the owner, 16 Nov 2026: "Use
the Australian results from GreyhoundAPI too. Use all data available"; ledger greyhound-world-batch46-1116).

Our breeding tables (greyhound/pit/breeding.py, pedigree.py) read the progeny's GB runs only. Many GB runners' sires
stand in Ireland, some are Australian sires used by artificial insemination, and the progeny's runs there are on the
API: Ireland from 1999 (190,238 races) and Australia from 1990 (961,311 races).

**The link.** GreyhoundAPI's pedigree (``sources/greyhoundapi/pedigree.parquet``: every sire's offspring, each with its
dam) gives each API dog its sire and dam; a GB runner is joined to them by its GBGB sire and dam names (lower case,
letters and digits only: core.norm_name). The progeny's runs are figured as greyhound/pit/irish.py figures Irish runs
(time and first-sectional figures against the track and trip's standard there, won, placed).

**The columns** (a GB race reads the progeny's runs on days strictly BEFORE the race's day, each region apart):
``wd_sire_ie_n``, ``wd_sire_ie_win`` (win rate shrunk toward 1/6 by ``K`` runs), ``wd_sire_ie_fig``,
``wd_sire_ie_sec``, the same for Australia (``wd_sire_au_*``), and for the dam in Ireland (``wd_dam_ie_n``,
``wd_dam_ie_win``, ``wd_dam_ie_fig``); ``wd_sire_au_share`` (the sire's progeny runs in Australia over all its runs
abroad). A runner whose sire or dam has no progeny run abroad before the day has NaN (counts 0).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import core
from greyhound.pit.irish import API_COLS, run_figures

PREFIX = "wd_"
SOURCE = ("GreyhoundAPI results (Ireland, Australia) and pedigree (every sire's offspring with its dam): "
          "sources/greyhoundapi.py; joined by the GBGB sire and dam names")
K = 30.0
COLUMNS = {
    "wd_sire_ie_n": "the sire's progeny runs in Ireland before the day (GreyhoundAPI)",
    "wd_sire_ie_win": "the sire's Irish progeny win rate, shrunk toward 1/6 by 30 runs",
    "wd_sire_ie_fig": "the mean time figure of the sire's Irish progeny runs",
    "wd_sire_ie_sec": "the mean first-sectional figure of the sire's Irish progeny runs",
    "wd_sire_au_n": "the sire's progeny runs in Australia before the day",
    "wd_sire_au_win": "the sire's Australian progeny win rate, shrunk toward 1/8 by 30 runs",
    "wd_sire_au_fig": "the mean time figure of the sire's Australian progeny runs",
    "wd_sire_au_sec": "the mean first-sectional figure of the sire's Australian progeny runs",
    "wd_sire_au_share": "the share of the sire's progeny runs abroad that were in Australia",
    "wd_dam_ie_n": "the dam's progeny runs in Ireland before the day",
    "wd_dam_ie_win": "the dam's Irish progeny win rate, shrunk toward 1/6 by 30 runs",
    "wd_dam_ie_fig": "the mean time figure of the dam's Irish progeny runs",
}


def load_context(store) -> dict:
    out = {"api_pedigree": store.get_parquet("greyhoundapi/pedigree.parquet")}
    for reg in ("IE", "AU"):
        keys = sorted(k for k in store.listing("greyhoundapi/")
                      if re.search(rf"greyhoundapi/runs_{reg}_\d{{4}}\.parquet$", k))
        parts = []
        for k in keys:
            t = store.get_parquet(k)
            if t is not None and len(t):
                parts.append(t[[c for c in API_COLS if c in t.columns]])
        out[f"api_{reg.lower()}_all"] = pd.concat(parts, ignore_index=True) if parts else None
    return out


def _cum(runs: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per parent and day: the running counts and sums through that day."""
    per = runs.groupby([key, "race_date"], as_index=False).agg(
        n=("won", "size"), w=("won", "sum"), f=("fig", "sum"), fn=("fig", "count"), s=("sec", "sum"),
        sn=("sec", "count")).sort_values([key, "race_date"])
    g = per.groupby(key)
    out = per[[key]].copy()
    out["day"] = pd.to_datetime(per.race_date)
    for c in ("n", "w", "f", "fn", "s", "sn"):
        out[c] = g[c].cumsum().to_numpy()
    return out


def _asof(q: pd.DataFrame, h: pd.DataFrame, key: str) -> pd.DataFrame:
    m = pd.merge_asof(q.sort_values("day"), h.sort_values("day"), on="day", by=key, allow_exact_matches=False,
                      direction="backward")
    return m.set_index("i").sort_index()


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    ctx = ctx or {}
    r = runs
    n = len(r)
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy()}, index=r.index)
    for c in COLUMNS:
        tab[c] = np.float32(np.nan)
    for c in ("wd_sire_ie_n", "wd_sire_au_n", "wd_dam_ie_n"):
        tab[c] = np.float32(0.0)
    ped = ctx.get("api_pedigree")
    if ped is None or not len(ped):
        return tab, {"note": "no data found: no GreyhoundAPI pedigree in the store"}
    ped = ped.assign(sire_k=ped.sire.map(core.norm_name), dam_k=ped.dam.map(core.norm_name))
    day = core.day(r)
    sire_k = r.sire.map(core.norm_name).to_numpy() if "sire" in r else np.full(n, "")
    dam_k = r.dam.map(core.norm_name).to_numpy() if "dam" in r else np.full(n, "")
    notes = {"pedigree_dogs": int(len(ped))}
    totals = {}
    for reg, prior in (("ie", 6.0), ("au", 8.0)):
        api = ctx.get(f"api_{reg}_all")
        if api is None or not len(api):
            notes[f"{reg}_runs"] = 0
            continue
        fr = run_figures(api).merge(ped[["dog_id", "sire_k", "dam_k"]], on="dog_id", how="inner")
        notes[f"{reg}_runs_with_pedigree"] = int(len(fr))
        parents = (("sire", "sire_k", sire_k),) + ((("dam", "dam_k", dam_k),) if reg == "ie" else ())
        for who, key, vals in parents:
            h = _cum(fr[fr[key] != ""], key)
            q = pd.DataFrame({key: vals, "day": day.to_numpy(), "i": np.arange(n)})
            m = _asof(q, h, key)
            nn = m.n.fillna(0).to_numpy(float)
            has = nn > 0
            pre = f"wd_{who}_{reg}"
            tab[f"{pre}_n"] = nn.astype(np.float32)
            tab[f"{pre}_win"] = np.where(has, (m.w.to_numpy() + K / prior) / (nn + K), np.nan).astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                tab[f"{pre}_fig"] = np.where(m.fn.to_numpy(float) > 0, m.f.to_numpy() / m.fn.to_numpy(), np.nan
                                             ).astype(np.float32)
                if f"{pre}_sec" in COLUMNS:
                    tab[f"{pre}_sec"] = np.where(m.sn.to_numpy(float) > 0, m.s.to_numpy() / m.sn.to_numpy(), np.nan
                                                 ).astype(np.float32)
            if who == "sire":
                totals[reg] = nn
            notes[f"gb_rows_{pre}"] = int(has.sum())
    if "ie" in totals and "au" in totals:
        tot = totals["ie"] + totals["au"]
        with np.errstate(invalid="ignore", divide="ignore"):
            tab["wd_sire_au_share"] = np.where(tot > 0, totals["au"] / tot, np.nan).astype(np.float32)
    return tab, notes
