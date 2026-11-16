"""A fitted first bend (the BFSP plan, reports/greyhound_bfsp_plan.md, step 2.4), in place of the race simulator's
guessed settings (greyhound/blocks/sim.py: pace weights 0.75 / 0.25, a 2-length check, noise 1.0, none fitted).

1. **Who leads at the bend**: a conditional logit over each race's runners, the label the fastest first sectional,
   on what is known before the race: the dog's early pace (its first-sectional figure over its last 5 earlier days,
   greyhound/metrics.py ``esr``) against the field, its spread, its lead and break history, its trap (and the two end
   traps), the early pace of the dogs drawn either side of it.
2. **The finish given the bend**: a second conditional logit on the winner: the bend utility from (1), the run-in
   (``fsr``) and whole-race (``gsr``) figures against the field, the trap.

Both are refitted each year on the races of the years before it only (greyhound/blocks/wpred.py's pattern), so a row
is priced by a model that never saw its year; the first year with fewer than ``MIN_FIT`` earlier races is left empty.
Columns: ``bd_p_lead``, ``bd_exp_pos`` (the expected bend position, Plackett-Luce on the bend utilities, by
simulation), ``bd_u`` (the bend utility), ``bd_p_win``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "bd_"
MIN_FIT = 20_000
MAX_FIT = 300_000
SIMS = 100
SOURCE = "GBGB results (sources/gbgb.py): sectional, trap, position, comment, through the engine's figures"
LEAD_X = ("esr_vs_field", "esr_sd", "lead_rate", "brk_rate", "trap", "trap1", "trap6", "in_esr", "out_esr", "esr_rank")
WIN_X = ("u", "fsr_vs_field", "gsr_vs_field", "trap")


def _vs_field(v: pd.Series, race: pd.Series) -> pd.Series:
    g = v.groupby(race)
    return v - (g.transform("sum") - v.fillna(0)) / (g.transform("count") - v.notna()).replace(0, np.nan)


def design(df: pd.DataFrame) -> pd.DataFrame:
    """The pre-race inputs of each runner (earlier days only) and the labels (the race's own result)."""
    race = df.race_id
    x = pd.DataFrame(index=df.index)
    for c in ("esr", "fsr", "gsr"):
        x[c] = core.dog_lagged(df, c, 5) if c in df else np.nan
    sec = pd.to_numeric(df.sectional, errors="coerce")
    first = sec.groupby(race).transform("min")
    lead = (sec == first).astype(float).where(sec.notna().groupby(race).transform("all"))
    x["lead_rate"] = core.dog_lagged(df.assign(_l=lead), "_l", 10)
    brk = pd.to_numeric(df.get("brk"), errors="coerce") if "brk" in df else pd.Series(np.nan, index=df.index)
    x["brk_rate"] = core.dog_lagged(df.assign(_b=brk), "_b", 10)
    x["esr_sd"] = core.dog_lagged(df.assign(_e2=df.esr ** 2), "_e2", 5).sub(x.esr ** 2).clip(lower=0) ** 0.5 \
        if "esr" in df else np.nan
    for c in ("esr", "fsr", "gsr"):
        x[f"{c}_vs_field"] = _vs_field(x[c], race)
    x["esr_rank"] = x.esr.groupby(race).rank(ascending=False)
    trap = pd.to_numeric(df.trap, errors="coerce")
    x["trap"] = trap
    x["trap1"] = (trap == 1).astype(float)
    # the outside runner: the highest trap in the race. Until 11 Nov 2026 it was `trap == field`, which in a race of
    # five missed the dog in trap 6 and, with trap 6 empty, flagged no one (the audit of 11 Nov 2026, finding 38)
    x["trap6"] = (trap == trap.groupby(race).transform("max")).astype(float)
    # the dogs in the boxes beside (trap - 1, trap + 1): an empty box is room, read as the field's slowest dog there,
    # not the next dog along
    esr = x.esr.fillna(x.esr.groupby(race).transform("mean"))
    box = esr.groupby([race, trap]).first()
    room = esr.groupby(race).transform("min")
    beside = lambda step: pd.Series(box.reindex(pd.MultiIndex.from_arrays([race, trap + step])).to_numpy(float), index=df.index)
    inside, outside = beside(-1).fillna(room), beside(1).fillna(room)
    x["in_esr"] = (inside - esr).where(trap > trap.groupby(race).transform("min"))
    x["out_esr"] = (outside - esr).where(trap < trap.groupby(race).transform("max"))
    x["lead"] = lead
    x["won"] = pd.to_numeric(df.won, errors="coerce")
    return x


def _matrix(x: pd.DataFrame, cols, mu=None, sd=None):
    m = x[list(cols)].astype(float)
    if mu is None:
        mu, sd = m.mean(), m.std().replace(0, 1.0)
    return ((m - mu) / sd).fillna(0.0).to_numpy(), mu, sd


def _groups(race: pd.Series) -> np.ndarray:
    return pd.factorize(race)[0]


def expected_position(u: np.ndarray, gi: np.ndarray, sims: int = SIMS, seed: int = 0) -> np.ndarray:
    """Each runner's expected bend position under Plackett-Luce with utilities ``u`` (Gumbel draws, ranked in race)."""
    acc = np.zeros(len(u))
    d = pd.DataFrame({"g": gi})
    # the same draws for every race (a runner's are those of its place in the race's rows): one stream down the frame
    # gave a card race another expected position than its result row (the audit of 11 Nov 2026, findings 17 and 42)
    slot = d.groupby("g").cumcount().to_numpy()
    G = np.random.default_rng(seed).gumbel(size=(int(slot.max()) + 1 if len(slot) else 1, sims))
    for k in range(sims):
        z = u + G[slot, k]
        acc += d.assign(z=z).groupby("g").z.rank(ascending=False).to_numpy()
    return acc / sims


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    x = design(df)
    racing = ~df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(True, index=df.index)
    year = core.day(df).dt.year
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    for c in ("p_lead", "exp_pos", "u", "p_win"):
        out[f"{PREFIX}{c}"] = np.nan
    notes = {"years": {}}
    lab = x.lead.notna() & x.lead.groupby(df.race_id).transform("sum").eq(1) & \
        x.lead.groupby(df.race_id).transform("count").eq(df.field)
    for y in sorted(year.unique()):
        fit_l = (year < y) & racing & lab
        races = df.race_id[fit_l].unique()
        if len(races) < MIN_FIT / 6:
            continue
        rng = np.random.default_rng(int(y))
        if len(races) > MAX_FIT / 6:
            races = rng.choice(races, int(MAX_FIT / 6), replace=False)
        tr = fit_l & df.race_id.isin(races)
        Xl, mu, sd = _matrix(x[tr], LEAD_X)
        bl = core.clogit_fit(Xl, _groups(df.race_id[tr]), x.lead[tr].to_numpy(float), lam=1.0)
        # the second stage: the winner, on the bend utility and the finishing figures
        ul_tr = Xl @ bl
        xw = x[tr].assign(u=ul_tr)
        won1 = xw.won.groupby(df.race_id[tr]).transform("sum").eq(1)
        Xw, mw, sw = _matrix(xw[won1], WIN_X)
        bw = core.clogit_fit(Xw, _groups(df.race_id[tr][won1]), xw.won[won1].to_numpy(float), lam=1.0)
        te = year == y
        Xt, _, _ = _matrix(x[te], LEAD_X, mu, sd)
        gt = _groups(df.race_id[te])
        u = Xt @ bl
        out.loc[te, f"{PREFIX}u"] = u
        out.loc[te, f"{PREFIX}p_lead"] = core.group_softmax(u, gt)
        out.loc[te, f"{PREFIX}exp_pos"] = expected_position(u, gt, sims=SIMS, seed=int(y))
        Xwt, _, _ = _matrix(x[te].assign(u=u), WIN_X, mw, sw)
        out.loc[te, f"{PREFIX}p_win"] = core.group_softmax(Xwt @ bw, gt)
        notes["years"][int(y)] = {"fit_races": int(len(races)), "lead_beta": dict(zip(LEAD_X, np.round(bl, 4))),
                                  "win_beta": dict(zip(WIN_X, np.round(bw, 4)))}
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}p_lead": "the chance of leading at the first bend (a conditional logit fitted on earlier years)",
    f"{PREFIX}exp_pos": "the expected position at the first bend (Plackett-Luce on the bend utilities)",
    f"{PREFIX}u": "the bend utility (the logit's score)",
    f"{PREFIX}p_win": "the chance of winning given the bend and the finishing figures (a second logit, earlier years)",
}
