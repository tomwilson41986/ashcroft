"""The market's past readings of the dog from Betfair's price files (review D3 and D4, reports/greyhound_feature_review.md):
how it has steamed or drifted, how much attention the market gave it, and how the place market priced it against the
win market. Until now these files were labels only (greyhound/fastloop.py LABELS: ppwap, morningwap, the volumes).

Each past GB race run is matched to the GB win price file (and, where given, the GB place file) by day, off time,
track and trap, as greyhound/parity.py ``attach_bsp`` matches the BSP. Per run:

- steam: log(BSP) less log(morning WAP), centred on the race's priced runners (the books' margins differ): below 0
  the dog shortened from the morning to the off;
- late: log(BSP) less log(pre-play WAP), centred the same way: the move into the off;
- attention: log of the dog's share of the race's pre-play matched volume times the priced runners (0 = an even
  share), and the same against its BSP chance (attention beyond its price);
- place ratio: log(place BSP - 1) less log(win BSP - 1), less what the GB place markets of the same field size and
  win-price band showed on EARLIER days (``core.prior_window``): above 0 the place market thought the dog less
  likely to place than its win price implies (all or nothing), below 0 a consistent placer.

The dog's columns are its last value and a decayed mean (half-life ``HL_RUNS`` priced runs) over its days strictly
before the race's day (merge_asof, allow_exact_matches False), so a race's own prices never enter its row and a card
row (no result, no prices yet) gets what the historic build gives: every column is card-safe. A price-file column
the stored data does not carry leaves its columns NaN, with "no data found" in the notes; nothing is filled in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "px_"
SOURCE = ("Betfair's greyhound price files (sources/greyhound_prices.py: BSP, morning and pre-play WAP, pre-play "
          "matched volume; GB win markets, and GB place markets where loaded)")
HL_RUNS = 3.0               # the decayed means' half-life, in the dog's priced race days
MIN_EXPECT = 30             # earlier place/win pairs a (field, band) cell needs before its expectation is used
KEY = ["race_date", "race_time", "track", "trap"]


def load_context(store, years: str | None = None) -> dict:
    """The GB place markets' BSPs (the win markets come in ``ctx["prices"]``, greyhound.model.load_prices, which keeps
    the win markets only)."""
    import re
    keys = sorted(k for k in store.listing("greyhound_prices/") if re.search(r"prices_(\d{4})\.parquet$", k))
    if years:
        lo, hi = (int(x) for x in years.split("-"))
        keys = [k for k in keys if lo <= int(re.search(r"prices_(\d{4})", k).group(1)) <= hi]
    parts = []
    for k in keys:
        t = store.get_parquet(k)
        if t is None or not len(t) or "market" not in t:
            continue
        t = t[(t.market == "place") & (t.country == "GB")]
        parts.append(t[[c for c in KEY + ["bsp"] if c in t.columns]].copy())
    return {"place_prices": pd.concat(parts, ignore_index=True) if parts else None}


def _match(runs: pd.DataFrame, prices: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame:
    """Each run's ``cols`` from a price table by day, off time, track and trap (NaN where unmatched or not held)."""
    out = pd.DataFrame(np.nan, index=runs.index, columns=cols)
    if prices is None or not len(prices) or not set(KEY) <= set(prices.columns):
        return out
    from greyhound.data import canonical_track
    have = [c for c in cols if c in prices.columns]
    p = prices[KEY + have].copy()
    p["track"] = canonical_track(p.track)
    p = p.dropna(subset=["trap"]).drop_duplicates(KEY, keep="last")
    p["trap"] = p.trap.astype("int64")
    q = runs[KEY].copy()
    q["trap"] = pd.to_numeric(q.trap, errors="coerce")
    ok = q.trap.notna()
    q = q[ok].astype({"trap": "int64"}).reset_index()
    m = q.merge(p, on=KEY, how="left").set_index("index")
    for c in have:
        out.loc[m.index, c] = pd.to_numeric(m[c], errors="coerce").to_numpy(float)
    return out


def history(df: pd.DataFrame, cols: list[str], hl: float = HL_RUNS, key: str = "dog_id") -> pd.DataFrame:
    """For each row: per ``col``, the ``key``'s last value, its decayed mean (half-life ``hl`` of its days with a
    value) and the count of those days, over days strictly before the row's day. Each day's values are averaged
    first. Columns ``<col>_last``, ``<col>_ewm``, ``<col>_n``; aligned to ``df.index``; NaN (n 0) with none."""
    d0 = df[[key]].assign(_d=core.day(df))
    out = pd.DataFrame(index=df.index)
    q = d0.assign(_i=np.arange(len(df))).dropna(subset=["_d", key]).sort_values("_d")
    for col in cols:
        out[f"{col}_last"], out[f"{col}_ewm"], out[f"{col}_n"] = np.nan, np.nan, 0.0
        d = d0.assign(_v=pd.to_numeric(df[col], errors="coerce")).dropna(subset=["_v", "_d", key])
        if not len(d):
            continue
        per = d.groupby([key, "_d"], sort=True)["_v"].mean().reset_index()
        g = per.groupby(key, sort=False)["_v"]
        per["_ewm"] = g.ewm(halflife=hl).mean().reset_index(level=0, drop=True).reindex(per.index).to_numpy()
        per["_n"] = g.cumcount().to_numpy() + 1.0
        m = pd.merge_asof(q, per.sort_values("_d"), on="_d", by=key, allow_exact_matches=False, direction="backward")
        res = np.full((len(df), 3), np.nan)
        res[m._i.to_numpy()] = m[["_v", "_ewm", "_n"]].to_numpy(float)
        out[f"{col}_last"], out[f"{col}_ewm"] = res[:, 0], res[:, 1]
        out[f"{col}_n"] = np.nan_to_num(res[:, 2])
    return out


def _centred(v: pd.Series, race: pd.Series) -> pd.Series:
    """``v`` less its race's mean over the runners that have it; NaN where fewer than two do."""
    g = v.groupby(race)
    return (v - g.transform("mean")).where(g.transform("count") >= 2)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    ctx = ctx or {}
    notes: dict = {}
    hist_row = ~runs.is_trial.fillna(False).astype(bool)
    if "is_card" in runs:
        hist_row &= ~runs.is_card.fillna(False).astype(bool)
    win = ctx.get("prices")
    if win is not None and "market" in win:
        win = win[(win.market == "win") & (win.get("country", pd.Series("GB", index=win.index)) == "GB")]
    w = _match(runs, win, ["bsp", "morningwap", "ppwap", "pp_vol"])
    w.loc[~hist_row.to_numpy()] = np.nan
    for c in ("bsp", "morningwap", "ppwap", "pp_vol"):
        held = win is not None and c in win.columns and win[c].notna().any()
        notes[f"win_{c}"] = int(w[c].notna().sum()) if held else "no data found"
    pl = ctx.get("place_prices")
    pb = _match(runs, pl, ["bsp"])["bsp"].where(hist_row)
    notes["place_bsp"] = (int(pb.notna().sum()) if pl is not None and "bsp" in pl.columns and pl.bsp.notna().any()
                          else "no data found")
    race = runs.race_id
    lb = np.log(w.bsp.where(w.bsp > 1))
    lm = np.log(w.morningwap.where(w.morningwap > 1))
    lp = np.log(w.ppwap.where(w.ppwap > 1))
    per = pd.DataFrame({"dog_id": runs.dog_id, "race_date": runs.race_date}, index=runs.index)
    per["steam"] = _centred(lb - lm, race)
    per["steam_abs"] = per.steam.abs()
    per["late"] = _centred(lb - lp, race)
    vol = w.pp_vol.where((w.pp_vol >= 0) & lb.notna())
    tot = vol.groupby(race).transform("sum")
    npr = vol.groupby(race).transform("count")
    share = (vol / tot).where((tot > 0) & (npr >= 2))
    per["attn"] = np.log((share * npr).where(share > 0))
    q = np.exp(-lb)
    qn = q / q.where(vol.notna()).groupby(race).transform("sum")
    per["attn_xs"] = np.log((share / qn).where((share > 0) & (qn > 0)))
    # the place ratio, against the earlier days' GB place markets of the same field size and win-price band
    r = (np.log((pb - 1.0).where(pb > 1.0)) - np.log((w.bsp - 1.0).where(w.bsp > 1.0)))
    band = np.floor(np.clip(lb, 0.0, np.log(100.0)) * 4.0)
    cell = pd.DataFrame({"field": pd.to_numeric(runs.field, errors="coerce").fillna(-1).astype(int),
                         "band": band.fillna(-1).astype(int), "race_date": runs.race_date,
                         "_r": r.fillna(0.0), "_k": r.notna().astype(float)}, index=runs.index)
    pw = core.prior_window(cell, ["field", "band"], ["_r", "_k"], None)
    expect = (pw._r / pw._k).where(pw._k >= MIN_EXPECT)
    per["plr"] = r - expect
    notes["place_ratio_rows"] = int(per.plr.notna().sum())
    h = history(per, ["steam", "steam_abs", "late", "attn", "attn_xs", "plr"])
    out = pd.DataFrame({"race_id": runs.race_id, "dog_id": runs.dog_id}, index=runs.index)
    n_priced = history(per.assign(_one=lb.where(lb.isna(), 1.0)), ["_one"])["_one_n"]
    out[f"{PREFIX}n"] = n_priced
    out[f"{PREFIX}steam_last"] = h.steam_last
    out[f"{PREFIX}steam_ewm"] = h.steam_ewm
    out[f"{PREFIX}steam_abs_ewm"] = h.steam_abs_ewm
    out[f"{PREFIX}late_last"] = h.late_last
    out[f"{PREFIX}late_ewm"] = h.late_ewm
    out[f"{PREFIX}attn_last"] = h.attn_last
    out[f"{PREFIX}attn_ewm"] = h.attn_ewm
    out[f"{PREFIX}attn_xs_ewm"] = h.attn_xs_ewm
    out[f"{PREFIX}plr_last"] = h.plr_last
    out[f"{PREFIX}plr_ewm"] = h.plr_ewm
    out[f"{PREFIX}plr_n"] = h.plr_n
    notes.update({"rows": int(len(out)), "hl_runs": HL_RUNS, "min_expect": MIN_EXPECT,
                  "card_safe": "every column (days strictly before the race's day only)"})
    return out, notes


COLUMNS = {
    f"{PREFIX}n": "the dog's earlier race days with a Betfair SP in the GB win price files",
    f"{PREFIX}steam_last": "last priced run: log(BSP / morning WAP), centred on its race (below 0: shortened to the off)",
    f"{PREFIX}steam_ewm": "the same, decayed mean over the dog's priced runs (half-life 3 runs)",
    f"{PREFIX}steam_abs_ewm": "decayed mean of |the centred morning-to-BSP move|: how much the market revises the dog",
    f"{PREFIX}late_last": "last priced run: log(BSP / pre-play WAP), centred on its race: the move into the off",
    f"{PREFIX}late_ewm": "the same, decayed mean (half-life 3 runs)",
    f"{PREFIX}attn_last": "last priced run: log(the dog's share of the race's pre-play matched volume x runners priced)",
    f"{PREFIX}attn_ewm": "the same, decayed mean (half-life 3 runs)",
    f"{PREFIX}attn_xs_ewm": "decayed mean of log(volume share / BSP chance): attention beyond the dog's price",
    f"{PREFIX}plr_last": "last run with both: log((place BSP - 1) / (win BSP - 1)) less the earlier days' mean for its "
                         "field size and win-price band (above 0: the place market saw it as all or nothing)",
    f"{PREFIX}plr_ewm": "the same, decayed mean (half-life 3 runs)",
    f"{PREFIX}plr_n": "the dog's earlier race days with a place ratio",
}
