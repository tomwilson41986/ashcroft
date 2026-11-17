"""The dog's run style in Ireland, from GreyhoundAPI's in-running comments (the owner, 17 Nov 2026: "build Irish
run-style features from the comments"; ledger greyhound-irish-style-batch51-1117).

Each Irish race's comments (``sources/greyhoundapi.py fetch_comments`` -> ``greyhoundapi/comments_IE.parquet``: a row
a runner, its events in order as codes, e.g. 'fast_start|led', and where in the race each happened, e.g. '|bend_1')
are read into flags a run: away quickly, away slowly, led, led early (at the first bend or before), on the rails,
wide, met trouble (baulked, bumped, crowded, checked, impeded, blocked, hampered, fell), ran on, faded. The codes are
grouped by the words in them (``GROUPS``), so a code the API adds later falls in its group; the share of events no
group reads is reported in the build notes.

The dog is linked to its Irish runs as the Irish form table links it (greyhound/pit/irish.py ``name_map``, ``dog_map``,
from ``irish.PRE_GB_DAYS`` before its first GB day).

**The columns** (a GB race reads the dog's Irish runs on days strictly BEFORE the race's day): ``irs_n`` (Irish runs
with comments), each group's rate shrunk toward the share in every Irish run of the first fifth of the days by ``K``
runs (``irs_<group>_rate``), the last Irish run's flags for early pace, the lead and trouble (``irs_last_*``), the style
(rails less wide), and the early-pace rate against today's other runners (``irs_early_vs``). A dog with no Irish run
with comments has NaN (``irs_n`` 0).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import core, irish

PREFIX = "irs_"
SOURCE = "GreyhoundAPI in-running comments of Irish races (sources/greyhoundapi.py fetch_comments), linked by name"
K = 3.0
#: each group: a regular expression on the event code (lower case, underscores)
GROUPS = {
    "early": r"fast_start|quick|early_pace|broke_well|good_break|pace_early|showed_early",
    "slow": r"slow|dwelt|missed|poor_break|bad_break|crawled",
    "led": r"(^|[_|])led($|[_|])|lead",
    "rails": r"rail|inside|in_?side",
    "wide": r"wide|outside|middle",
    "trouble": r"baulk|bump|crowd|check|impede|block|hamper|fell|fall|knock|interfer|squeez|clip",
    "ranon": r"ran_on|finished_well|stayed_on|strong_finish|came_late|closed|late_pace",
    "faded": r"faded|weaken|tired|no_extra|one_pace|eased",
}
EARLY_PLACES = r"bend_1|first_bend|start|break|run_up|early"
COLUMNS = {
    "irs_n": "the dog's Irish runs with in-running comments before the day (GreyhoundAPI)",
    **{f"irs_{g}_rate": f"its Irish runs with a '{g}' comment, shrunk toward the share in every Irish run by 3 runs"
       for g in GROUPS},
    "irs_led_early_rate": "its Irish runs where it led at the first bend or before, shrunk",
    "irs_last_early": "its last Irish run: 1 when it was quick away",
    "irs_last_led": "its last Irish run: 1 when it led",
    "irs_last_trouble": "its last Irish run: 1 when it met trouble",
    "irs_style": "irs_rails_rate less irs_wide_rate (above 0: a railer)",
    "irs_early_vs": "irs_early_rate less the mean of today's other runners' that have one",
}


def load_context(store) -> dict:
    out = irish.load_context(store)
    out["api_ie_comments"] = store.get_parquet("greyhoundapi/comments_IE.parquet")
    return out


def run_flags(cm: pd.DataFrame) -> pd.DataFrame:
    """Each commented Irish run's flags (0/1), with its day."""
    ev = cm.events.fillna("").astype(str).str.lower()
    pl = cm.places.fillna("").astype(str).str.lower()
    f = pd.DataFrame({"dog_id": cm.dog_id.to_numpy(),
                      "race_date": pd.to_datetime(cm.start_utc, utc=True, errors="coerce").dt.strftime("%Y-%m-%d")})
    for g, rx in GROUPS.items():
        f[g] = ev.str.contains(rx, regex=True).astype(float).to_numpy()
    # led early: a 'led' event whose place is early, read pairwise
    early_led = []
    for e, p in zip(ev, pl):
        es, ps = e.split("|"), p.split("|")
        early_led.append(float(any(re.search(GROUPS["led"], a) and re.search(EARLY_PLACES, b or "")
                                   for a, b in zip(es, ps + [""] * (len(es) - len(ps))))))
    f["led_early"] = early_led
    return f[f.race_date.notna()]


def coverage(cm: pd.DataFrame) -> float | None:
    codes = [c for e in cm.events.fillna("").astype(str) for c in e.lower().split("|") if c]
    if not codes:
        return None
    rx = re.compile("|".join(f"(?:{v})" for v in GROUPS.values()))
    return round(sum(1 for c in codes if rx.search(c)) / len(codes), 4)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    ctx = ctx or {}
    r = runs
    n = len(r)
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy()}, index=r.index)
    for c in COLUMNS:
        tab[c] = np.float32(np.nan)
    tab["irs_n"] = np.float32(0.0)
    cm = ctx.get("api_ie_comments")
    if cm is None or not len(cm):
        return tab, {"note": "no data found: no GreyhoundAPI Irish comments in the store (greyhoundapi.yml task comments)"}
    names = cm[["dog_id", "dog_name"]].dropna()
    ie = ctx.get("api_ie")
    ie_names = pd.concat([ie[["dog_id", "dog_name"]], names], ignore_index=True) if ie is not None and "dog_name" in ie \
        else names
    mp, mnotes = irish.dog_map(ctx.get("api_gb"), r)
    nmp, first, nnotes = irish.name_map(ie_names, r)
    both = {**nmp, **mp}
    fl = run_flags(cm)
    fl["gb_dog"] = fl.dog_id.map(both)
    fl = fl[fl.gb_dog.notna()].copy()
    fl["gb_dog"] = fl.gb_dog.astype("int64")
    if first is not None and len(first):
        start = fl.gb_dog.map(first) - pd.Timedelta(days=irish.PRE_GB_DAYS)
        fl = fl[pd.to_datetime(fl.race_date) >= start.fillna(pd.Timestamp("1900-01-01"))]
    notes = {**mnotes, **nnotes, "comment_rows": int(len(cm)), "event_codes_read_share": coverage(cm),
             "irish_commented_runs_mapped": int(len(fl))}
    if not len(fl):
        return tab, {**notes, "note": "no Irish commented run of a mapped dog"}
    keys = list(GROUPS) + ["led_early"]
    days = np.sort(fl.race_date.unique())
    cut = days[int(len(days) * 0.2)] if len(days) > 4 else days[-1]
    p0 = fl[fl.race_date <= cut][keys].mean().fillna(0.0)
    per = fl.groupby(["gb_dog", "race_date"], as_index=False)[keys].mean().sort_values(["gb_dog", "race_date"])
    g = per.groupby("gb_dog")
    h = pd.DataFrame({"gb_dog": per.gb_dog, "day": pd.to_datetime(per.race_date), "n": g.cumcount().to_numpy() + 1.0})
    for k in keys:
        h[f"c_{k}"] = g[k].cumsum().to_numpy()
        h[f"last_{k}"] = per[k].to_numpy()
    day = core.day(r)
    q = pd.DataFrame({"gb_dog": pd.to_numeric(r.dog_id, errors="coerce").astype("int64").to_numpy(),
                      "day": day.to_numpy(), "i": np.arange(n)}).sort_values("day")
    m = pd.merge_asof(q, h.sort_values("day"), on="day", by="gb_dog", allow_exact_matches=False,
                      direction="backward").set_index("i").sort_index()
    nn = m.n.fillna(0).to_numpy(float)
    has = nn > 0
    tab["irs_n"] = nn
    for k in keys:
        name = "irs_led_early_rate" if k == "led_early" else f"irs_{k}_rate"
        tab[name] = np.where(has, (m[f"c_{k}"].to_numpy(float) + K * p0[k]) / (nn + K), np.nan)
    for k, c in (("early", "irs_last_early"), ("led", "irs_last_led"), ("trouble", "irs_last_trouble")):
        tab[c] = np.where(has, m[f"last_{k}"].to_numpy(float), np.nan)
    tab["irs_style"] = tab.irs_rails_rate.to_numpy(float) - tab.irs_wide_rate.to_numpy(float)
    x = pd.Series(tab.irs_early_rate.to_numpy(float))
    race = r.race_id.to_numpy()
    tot, cnt = x.groupby(race).transform("sum").to_numpy(), x.notna().groupby(race).transform("sum").to_numpy()
    v = x.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        tab["irs_early_vs"] = np.where(np.isfinite(v) & (cnt > 1), v - (tot - np.nan_to_num(v)) / np.maximum(cnt - 1, 1),
                                      np.nan)
    for c in COLUMNS:
        tab[c] = tab[c].astype(np.float32)
    notes.update({"gb_rows_with_style": int(has.sum()), "prior_shares": {k: round(float(p0[k]), 4) for k in keys}})
    return tab, notes
