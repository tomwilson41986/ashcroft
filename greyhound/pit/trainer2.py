"""The trainer rebuilt on the trainer AT THE TIME (the feature review of 17 Nov 2026: B3, B4, D1, D5 and the
connections row). Research: built on demand.

**The trainer** (greyhound/pit/trainers.py ``at_the_time``): GreyhoundAPI's per-run trainer where linked, else GBGB's
where the day was read within core.FRESH_DAYS, and on the card row the card's trainer (Betfair's catalogue where the
runs carry ``bf_trainer``), else the dog's last known. Every aggregate is keyed on the trainer of the runs it counts
and reads days strictly BEFORE the row's day.

**Against the market**: each race's priced runners' chances from the SP de-biased by Shin's method
(core.debiased_probs, review B5), not the proportional normalisation. A/E is a gamma posterior: prior Gamma(``K``,
``K``) (mean 1), wins and expected winners exponentially decayed with half-life ``HALF_LIFE`` days; the column is
E[log A/E] = digamma(a) - log(b) with its sd sqrt(trigamma(a)). The 30-day form is the same on the 30-day window.

**Shrunk to the trainer**: the trainer at this track and in each situation (after a break of 28 days or more, up in
grade, down in grade, the first run for the yard) is a gamma posterior whose prior is the trainer's own A/E posterior
mean with ``K_CELL`` expected winners of weight (decayed with ``HALF_LIFE_CELL``). A situation's columns are given
where today's run is in that situation (else NaN), like the connections table.

**The move**: ``ta_new_trainer`` (today's trainer differs from the trainer of the dog's last race with one known) and
``ta_runs_since_move`` (the dog's earlier race days for today's trainer since it last changed yard).

Card-safe (all columns): nothing reads today's result, SP or time; today's trainer is the declared trainer, known
before the off. The card row's trainer comes from the card where the historic row's comes from GreyhoundAPI or GBGB:
equal when the sources agree (notes ``api_vs_fresh_agree``; the check of research/greyhound/trainer_check.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma

from greyhound.pit import core, trainers
from greyhound.pit.connections import situations

PREFIX = "ta_"
#: the code hash covers the helper (trainers.py) and the reused situations (connections.py)
DEPENDS = ("connections", "trainers")
SOURCE = ("GBGB results (grade, SP) and the trainer at the time: GreyhoundAPI GB results (sources/greyhoundapi.py), "
          "GBGB read fresh, the card (greyhound/pit/trainers.py)")
K = 5.0
K_CELL = 3.0
HALF_LIFE = 180.0
HALF_LIFE_CELL = 365.0
K_RATE = 20.0
SITS = ("layoff", "rise", "drop", "first_for_yard")


def decayed_prior(df: pd.DataFrame, keys: list[str], cols: list[str], half_life: float) -> pd.DataFrame:
    """Sums of ``cols`` over the group's runs on EARLIER days, each weighted 2^(-(age in days) / ``half_life``).
    Aligned to ``df.index``; 0 where there is none."""
    d = df[keys + cols].copy()
    for k in keys:
        if not pd.api.types.is_numeric_dtype(d[k]):
            d[k] = d[k].astype(object).where(d[k].notna(), "?").astype(str)
    d["_d"] = core.day(df)
    d0 = d["_d"].min()
    lam = np.log(2.0) / half_life
    agg = d.groupby(keys + ["_d"], dropna=False)[cols].sum().reset_index()
    age = (agg["_d"] - d0).dt.days.to_numpy(float)
    w = np.exp(lam * age)
    for c in cols:
        agg[c] = agg[c].to_numpy(float) * w
    agg = agg.sort_values(keys + ["_d"])
    cum = agg.groupby(keys, dropna=False)[cols].cumsum()
    agg[[f"_c_{c}" for c in cols]] = cum.to_numpy()
    q = d[keys].copy()
    q["_q"] = d["_d"].values
    q["_i"] = np.arange(len(q))
    q = q.dropna(subset=["_q"]).sort_values("_q")
    right = agg[keys + ["_d"] + [f"_c_{c}" for c in cols]].sort_values("_d")
    m = pd.merge_asof(q, right, left_on="_q", right_on="_d", by=keys, allow_exact_matches=False, direction="backward")
    res = np.zeros((len(d), len(cols)))
    back = np.exp(-lam * (m["_q"] - d0).dt.days.to_numpy(float))[:, None]
    res[m._i.to_numpy()] = m[[f"_c_{c}" for c in cols]].fillna(0.0).to_numpy() * back
    return pd.DataFrame(res, index=df.index, columns=cols)


def gamma_log(a, b) -> tuple[np.ndarray, np.ndarray]:
    """E[log x] and sd(log x) of x ~ Gamma(shape a, rate b)."""
    a = np.asarray(a, float)
    return digamma(a) - np.log(np.asarray(b, float)), np.sqrt(polygamma(1, a))


def spells(runs: pd.DataFrame, trainer: pd.Series, races: pd.Series) -> pd.DataFrame:
    """Over the dog's race days (``races``: trials out): ``new`` (today's trainer differs from the last known trainer
    of an earlier race day; NaN when either is unknown) and ``since`` (the dog's earlier race days for today's trainer
    since the spell began: its first race day, or a known change; NaN when the spell began at an unknown), on
    ``runs.index``."""
    d = pd.DataFrame({"dog_id": runs.dog_id, "t": runs.t, "_d": core.day(runs), "tr": trainer})[races]
    dd = d.sort_values(["dog_id", "t"]).groupby(["dog_id", "_d"], sort=False).tail(1).sort_values(["dog_id", "_d"])
    g = dd.groupby("dog_id")
    dd["ff"] = g.tr.ffill()
    prev = dd.groupby("dog_id").ff.shift(1)
    first = g.cumcount() == 0
    change = dd.ff.notna() & prev.notna() & (dd.ff != prev)
    start = first | change | (dd.ff.notna() & prev.isna())
    anchored = (first & dd.ff.notna()) | change
    spell = start.astype(int).groupby(dd.dog_id).cumsum()
    key = [dd.dog_id, spell]
    since = dd.groupby(key).cumcount().astype(float)
    anch = anchored.groupby(key).transform("first")
    dd["new"] = change.astype(float).where(dd.tr.notna() & prev.notna())
    dd["since"] = since.where(anch & dd.tr.notna())
    m = pd.DataFrame({"dog_id": runs.dog_id, "_d": core.day(runs)}).reset_index().merge(
        dd[["dog_id", "_d", "new", "since"]], on=["dog_id", "_d"], how="left").set_index("index")
    return m[["new", "since"]].reindex(runs.index)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    tt, tnotes = trainers.at_the_time(runs, ctx)
    df = runs.copy()
    df["trainer"] = tt.trainer
    known = df.trainer.notna()
    sit = situations(df)                                # layoff, rise, drop from the dog's last race (trials out)
    card = df["is_card"].fillna(False).astype(bool) if "is_card" in df else pd.Series(False, index=df.index)
    trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
    sp = df["sp_decimal"] if "sp_decimal" in df else pd.Series(np.nan, index=df.index)
    p = pd.Series(core.debiased_probs(sp.where(~trial & ~card).to_numpy(), df.race_id.to_numpy(), "shin"),
                  index=df.index)
    racing = ~trial & ~card & p.notna() & df.won.notna() & known
    df["_w"] = df.won.where(racing, 0.0).fillna(0.0)
    df["_p"] = df.placed2.where(racing, 0.0).fillna(0.0) if "placed2" in df else 0.0
    df["_e"] = p.where(racing, 0.0).fillna(0.0)
    df["_r"] = racing.astype(float)
    sp_ = spells(df, df.trainer, ~trial)
    flags = {"layoff": sit["layoff"], "rise": sit["rise"], "drop": sit["drop"], "first_for_yard": sp_.new.fillna(0.0)}

    out = pd.DataFrame({"race_id": df.race_id, "dog_id": df.dog_id})
    out[f"{PREFIX}known"] = known.astype(float)
    dec = decayed_prior(df, ["trainer"], ["_w", "_e", "_r"], HALF_LIFE)
    a, b = K + dec._w, K + dec._e
    mean_tr = a / b
    out[f"{PREFIX}ae_log"], out[f"{PREFIX}ae_sd"] = gamma_log(a, b)
    out[f"{PREFIX}runs_decayed"] = dec._r
    w30 = core.prior_window(df, ["trainer"], ["_w", "_e", "_r"], 30)
    out[f"{PREFIX}ae30_log"], out[f"{PREFIX}ae30_sd"] = gamma_log(K + w30._w, K + w30._e)
    out[f"{PREFIX}runs30"] = w30._r
    w365 = core.prior_window(df, ["trainer"], ["_w", "_p", "_r"], 365)
    out[f"{PREFIX}runs365"] = w365._r
    out[f"{PREFIX}win_rate"] = core.shrunk_rate(w365._w, w365._r, 1 / 6, K_RATE)
    out[f"{PREFIX}place_rate"] = core.shrunk_rate(w365._p, w365._r, 2 / 6, K_RATE)
    trk = decayed_prior(df.assign(_trk=df.track.astype(str)), ["trainer", "_trk"], ["_w", "_e", "_r"], HALF_LIFE_CELL)
    out[f"{PREFIX}track_ae_log"], out[f"{PREFIX}track_ae_sd"] = gamma_log(K_CELL * mean_tr + trk._w, K_CELL + trk._e)
    out[f"{PREFIX}track_runs_decayed"] = trk._r
    out[f"{PREFIX}new_trainer"] = sp_.new
    out[f"{PREFIX}runs_since_move"] = sp_.since
    for s in SITUATIONS_OUT:
        f = flags[s].fillna(0.0)
        cell = df.assign(_s=f)
        for c in ("_w", "_e", "_r"):
            cell[c] = cell[c] * cell._s
        w = decayed_prior(cell, ["trainer"], ["_w", "_e", "_r"], HALF_LIFE_CELL)
        lg, sd = gamma_log(K_CELL * mean_tr + w._w, K_CELL + w._e)
        on = f == 1
        out[f"{PREFIX}{s}"] = f
        out[f"{PREFIX}{s}_ae_log"] = pd.Series(lg, index=df.index).where(on)
        out[f"{PREFIX}{s}_ae_sd"] = pd.Series(sd, index=df.index).where(on)
        out[f"{PREFIX}{s}_runs_decayed"] = w._r.where(on)
    for c in [c for c in out.columns if c.startswith(PREFIX) and c != f"{PREFIX}known"]:
        out[c] = out[c].where(known)
    tnotes.update({"k": K, "k_cell": K_CELL, "half_life": HALF_LIFE, "half_life_cell": HALF_LIFE_CELL,
                   "sp_debias": "shin", "racing_runs_counted": int(racing.sum()),
                   "card_safe": "all columns (the trainer on the card: the card's, else the dog's last known)"})
    return out.reset_index(drop=True), tnotes


SITUATIONS_OUT = SITS
COLUMNS = {
    f"{PREFIX}known": "1 when the run's trainer at the time is known (GreyhoundAPI, GBGB read fresh, or the card)",
    f"{PREFIX}ae_log": "the trainer's E[log A/E] against the Shin de-biased SP, wins and expected decayed (half-life "
                       "180 days), gamma posterior from Gamma(5, 5)",
    f"{PREFIX}ae_sd": "the sd of that log A/E",
    f"{PREFIX}runs_decayed": "the trainer's runners, decayed (half-life 180 days)",
    f"{PREFIX}ae30_log": "the trainer's E[log A/E] over the 30 days before the day (the yard's form), Gamma(5, 5) prior",
    f"{PREFIX}ae30_sd": "the sd of that 30-day log A/E",
    f"{PREFIX}runs30": "the trainer's runners, last 30 days",
    f"{PREFIX}runs365": "the trainer's runners, last 365 days",
    f"{PREFIX}win_rate": "the trainer's win rate, last 365 days, shrunk toward 1/6 by 20 runs",
    f"{PREFIX}place_rate": "the trainer's first-two rate, last 365 days, shrunk toward 2/6 by 20 runs",
    f"{PREFIX}track_ae_log": "the trainer's E[log A/E] at today's track (half-life 365 days), shrunk toward the "
                             "trainer's own A/E by 3 expected winners",
    f"{PREFIX}track_ae_sd": "the sd of that log A/E",
    f"{PREFIX}track_runs_decayed": "the trainer's runners at today's track, decayed (half-life 365 days)",
    f"{PREFIX}new_trainer": "1 when today's trainer differs from the trainer of the dog's last race with one known",
    f"{PREFIX}runs_since_move": "the dog's earlier race days for today's trainer since it last changed yard",
    **{f"{PREFIX}{s}": f"today's run: {s.replace('_', ' ')} (1/0)" for s in SITS},
    **{f"{PREFIX}{s}_ae_log": f"the trainer's E[log A/E] with runners {s.replace('_', ' ')} (half-life 365 days), "
                              f"shrunk toward the trainer's own A/E; where today's run is one" for s in SITS},
    **{f"{PREFIX}{s}_ae_sd": f"the sd of that log A/E ({s.replace('_', ' ')})" for s in SITS},
    **{f"{PREFIX}{s}_runs_decayed": f"the trainer's runners {s.replace('_', ' ')}, decayed; where today's run is one"
       for s in SITS},
}
