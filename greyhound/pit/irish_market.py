"""The Irish market's view of the dog, and its Irish weights, from GreyhoundAPI's Irish results (the owner, 16 Nov 2026:
"What other data from the api can we use? Research it and include in modelling"; ledger
greyhound-irish-market-batch47-1116).

The field survey (research/greyhound/api_survey.py, greyhoundapi.yml run 43964591170) found two Irish fields the Irish
form table (greyhound/pit/irish.py) does not read: the starting price (``sp_decimal``, 59-99% of Irish runners a year
from 1999) and the weight (``weight_kg``, every Irish runner from 1999). The dog is linked to its Irish runs exactly
as the Irish form table links it (``irish.name_map``, ``irish.dog_map``, from ``PRE_GB_DAYS`` before its first GB day).

**An Irish run's market reading** (races where every runner has an SP): the SP's chance with the overround taken out
(``p``), won less ``p`` (the run's actual over expected), log SP, and the SP rank less the finishing position (above
0: finished better than the market ranked it); 1 when it was the SP favourite.

**The columns** (a GB race reads the dog's Irish runs on days strictly BEFORE the race's day): ``im_n`` (Irish runs
with a market reading), ``im_ae`` (the sum of won less ``p`` over ``n + K``: the market's misses on the dog, shrunk),
``im_lsp_last``, ``im_lsp_mean3`` (log SP), ``im_beat_mean3`` (SP rank less position, last three), ``im_fav_rate``
(SP favourite share, shrunk toward 1/6 by ``K``), ``im_wt_last`` (its last Irish weight, kg), ``im_wt_trend`` (the
last Irish weight less the mean of the three before it). A dog with no Irish run has NaN (``im_n`` 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import irish

PREFIX = "im_"
SOURCE = irish.SOURCE + "; the Irish starting prices and weights"
K = 5.0
COLUMNS = {
    "im_n": "the dog's Irish runs with an SP for every runner, before the day (GreyhoundAPI)",
    "im_ae": "Irish winners less the SP's chance (overround out), summed over runs + 5",
    "im_lsp_last": "log SP of its last Irish run",
    "im_lsp_mean3": "mean log SP of its last three Irish runs",
    "im_beat_mean3": "SP rank less finishing position, mean of its last three Irish runs",
    "im_fav_rate": "its share of Irish runs as SP favourite, shrunk toward 1/6 by 5 runs",
    "im_wt_last": "its last Irish weight (kg)",
    "im_wt_trend": "its last Irish weight less the mean of the three Irish weights before it (kg)",
}


def load_context(store) -> dict:
    return irish.load_context(store)


def market_runs(ie: pd.DataFrame) -> pd.DataFrame:
    """Each Irish run with its day, the SP's chance, won, log SP, SP rank less position, favourite flag and weight."""
    d = ie.copy()
    d["race_date"], _ = irish._local(d)
    d["pos"] = pd.to_numeric(d.position, errors="coerce")
    d = d[d.pos.notna()].copy()
    sp = pd.to_numeric(d.get("sp_decimal"), errors="coerce") if "sp_decimal" in d else pd.Series(np.nan, index=d.index)
    d["sp"] = sp.where(sp > 1)
    g = d.groupby("race_id")
    full = g.sp.transform(lambda s: s.notna().all()) & (g.pos.transform("size") >= 2)
    inv = (1.0 / d.sp).where(full)
    d["p"] = inv / inv.groupby(d.race_id).transform("sum")
    d["won"] = (d.pos == 1).astype(float)
    d["lsp"] = np.log(d.sp).where(full)
    d["sp_rank"] = d.sp.where(full).groupby(d.race_id).rank(method="min")
    d["beat"] = d.sp_rank - d.pos
    d["fav"] = (d.sp_rank == 1).astype(float).where(full)
    d["wt"] = pd.to_numeric(d.get("weight_kg"), errors="coerce") if "weight_kg" in d else np.nan
    d["wt"] = d.wt.where((d.wt > 15) & (d.wt < 50))
    return d[["dog_id", "race_date", "p", "won", "lsp", "beat", "fav", "wt"]]


def history(mr: pd.DataFrame) -> pd.DataFrame:
    per = mr.assign(ae=mr.won - mr.p, has=mr.p.notna().astype(float)).groupby(["gb_dog", "race_date"], as_index=False).agg(
        n=("has", "sum"), ae=("ae", "sum"), fav=("fav", "sum"), lsp=("lsp", "mean"), beat=("beat", "mean"),
        wt=("wt", "mean")).sort_values(["gb_dog", "race_date"])
    g = per.groupby("gb_dog")
    roll = lambda c, k, f: getattr(g[c].rolling(k, min_periods=1), f)().reset_index(level=0, drop=True).reindex(per.index)  # noqa: E731
    out = pd.DataFrame({"gb_dog": per.gb_dog, "day": pd.to_datetime(per.race_date)})
    out["n"] = g.n.cumsum().to_numpy()
    out["ae"] = g.ae.cumsum().to_numpy()
    out["fav"] = g.fav.cumsum().to_numpy()
    out["lsp_last"] = g.lsp.ffill().to_numpy()
    out["lsp_mean3"] = roll("lsp", 3, "mean").to_numpy()
    out["beat_mean3"] = roll("beat", 3, "mean").to_numpy()
    out["wt_last"] = g.wt.ffill().to_numpy()
    prev3 = g.wt.shift(1).groupby(per.gb_dog).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    out["wt_trend"] = (per.wt - prev3.reindex(per.index)).to_numpy()
    return out


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    ctx = ctx or {}
    r = runs
    n = len(r)
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy()}, index=r.index)
    for c in COLUMNS:
        tab[c] = np.float32(np.nan)
    tab["im_n"] = np.float32(0.0)
    ie, gb = ctx.get("api_ie"), ctx.get("api_gb")
    if ie is None or not len(ie):
        return tab, {"note": "no data found: no GreyhoundAPI Irish results in the store"}
    if "sp_decimal" not in ie and "weight_kg" not in ie:
        return tab, {"note": "no data found: the Irish results carry no SP or weight columns"}
    mp, mnotes = irish.dog_map(gb, r)
    nmp, first, nnotes = irish.name_map(ie, r)
    both = {**nmp, **mp}
    mr = market_runs(ie)
    mr["gb_dog"] = mr.dog_id.map(both)
    mr = mr[mr.gb_dog.notna()].copy()
    mr["gb_dog"] = mr.gb_dog.astype("int64")
    if first is not None and len(first):
        start = mr.gb_dog.map(first) - pd.Timedelta(days=irish.PRE_GB_DAYS)
        mr = mr[pd.to_datetime(mr.race_date) >= start.fillna(pd.Timestamp("1900-01-01"))]
    if not len(mr):
        return tab, {**mnotes, **nnotes, "note": "no Irish run of a mapped dog"}
    h = history(mr)
    from greyhound.pit import core
    day = core.day(r)
    q = pd.DataFrame({"gb_dog": pd.to_numeric(r.dog_id, errors="coerce").astype("int64").to_numpy(),
                      "day": day.to_numpy(), "i": np.arange(n)}).sort_values("day")
    m = pd.merge_asof(q, h.sort_values("day"), on="day", by="gb_dog", allow_exact_matches=False, direction="backward")
    m = m.set_index("i").sort_index()
    nn = m.n.fillna(0).to_numpy(float)
    has = m.day.notna().to_numpy() & m.gb_dog.notna().to_numpy() & np.isfinite(m.n.to_numpy(float))
    tab["im_n"] = nn
    with np.errstate(invalid="ignore", divide="ignore"):
        tab["im_ae"] = np.where(nn > 0, m.ae.to_numpy(float) / (nn + K), np.nan)
        tab["im_fav_rate"] = np.where(nn > 0, (m.fav.to_numpy(float) + K / 6.0) / (nn + K), np.nan)
    for c in ("lsp_last", "lsp_mean3", "beat_mean3", "wt_last", "wt_trend"):
        tab[f"im_{c}"] = np.where(has, m[c].to_numpy(float), np.nan)
    for c in COLUMNS:
        tab[c] = tab[c].astype(np.float32)
    notes = {**mnotes, **nnotes, "irish_runs_mapped": int(len(mr)), "irish_runs_with_market": int(mr.p.notna().sum()),
             "irish_runs_with_weight": int(mr.wt.notna().sum()), "gb_rows_with_market": int((nn > 0).sum()),
             "gb_rows_with_weight": int(np.isfinite(tab.im_wt_last.to_numpy(float)).sum())}
    return tab, notes
