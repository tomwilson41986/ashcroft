"""The first bend refitted (the feature review of 17 Nov 2026, reports/greyhound_feature_review.md, row "bend": "Trap
enters linearly and the same everywhere; raw unshrunk inputs; Monte Carlo expected position"). A research variant of
greyhound/pit/bend.py (``bd_``), whose output it leaves alone.

As bend.py, a conditional logit on who leads at the first bend (the race's single fastest first sectional) and a
second on the winner given the bend, both refitted each year on the races of the years before it only. What changes:

- **the trap**: a dummy a trap (1 to 5, and the race's outside runner) in place of one linear term, and a
  track-specific offset a trap: log((leads + ``K_TRAP``) / (the model's expected leads + ``K_TRAP``)) over the
  earlier years' races at the track from that trap (two passes), so trap 1 can help at one track and hurt at another;
- **shrunk inputs**: the early, run-home and whole-race figures (the engine's ``esr``, ``fsr``, ``gsr``) over the
  last 5 earlier days are pulled toward the race's field mean by ``K_PACE`` days of prior, their spread toward the
  field's median spread, the lead rate toward 1 / field and the break rate toward 0 (``K_RATE`` days), and the
  evidence (days with a first sectional) enters;
- **the exact expected bend position** under Plackett-Luce, 1 + sum over the other runners of sigmoid(u_j - u_i)
  (each pair's order is a logistic of the utility gap), in place of bend.py's Monte Carlo;
- **crowding**: with each dog's arrival at the bend -u + Gumbel noise, the gap between two dogs' arrivals is
  logistic about u_j - u_i, so the chance the two arrive within ``DELTA`` utility units of each other is
  sigmoid(DELTA - gap) - sigmoid(-DELTA - gap); ``bd2_crowd`` is the chance at least one dog drawn 1 or 2 traps away
  arrives with it (the pairs taken as independent), ``bd2_crowd1`` the same for the traps beside it. ``DELTA`` is a
  fixed setting, not fitted.

Card-safe throughout: the inputs read earlier days, the trap, track and field come from the card.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import bend, core

DEPENDS = ("bend",)
PREFIX = "bd2_"
SOURCE = "GBGB results (sources/gbgb.py): sectional, trap, track, position, comment, through the engine's figures"
MIN_FIT = 20_000
MAX_FIT = 300_000
K_PACE = 2.0                   # earlier days of prior toward the field's mean figure
K_RATE = 3.0                   # earlier days of prior on the lead and break rates
K_TRAP = 50.0                  # leads of prior on a track-and-trap offset
DELTA = 0.5                    # utility units: two dogs inside this of each other at the bend "arrive together"
SLOTS = 6
LEAD_X = ("esr_vs_field", "esr_sd", "esr_n", "lead_rate", "brk_rate", "in_esr", "out_esr", "esr_rank",
          "t1", "t2", "t3", "t4", "t5", "outside")
WIN_X = ("u", "fsr_vs_field", "gsr_vs_field", "outside")
COLUMNS = {
    f"{PREFIX}u": "the bend utility (the lead logit's score with its track-and-trap offset; earlier years' fit)",
    f"{PREFIX}p_lead": "the chance of leading at the first bend (softmax of bd2_u in the race)",
    f"{PREFIX}exp_pos": "the expected first-bend position, exactly: 1 + sum over the other runners of sigmoid(u_j - u_i)",
    f"{PREFIX}trap_off": "the track-specific offset of this trap on the bend utility (earlier years' leads against "
                         f"the model's, {K_TRAP:g} leads of prior)",
    f"{PREFIX}crowd": "the chance at least one dog drawn 1 or 2 traps away reaches the bend with it (within "
                      f"{DELTA:g} utility units)",
    f"{PREFIX}crowd1": "the same for the dogs drawn immediately beside it",
    f"{PREFIX}p_win": "the chance of winning given the bend utility and the shrunk finishing figures (earlier years)",
}


def _shrunk(df: pd.DataFrame, col: str, race: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The dog's mean and spread of ``col`` over its last 5 earlier days, shrunk to the field; and the days behind."""
    if col not in df:
        nan = pd.Series(np.nan, index=df.index)
        return nan, nan, pd.Series(0.0, index=df.index)
    m = core.dog_lagged(df, col, 5)
    v = pd.to_numeric(df[col], errors="coerce")
    n = core.dog_lagged(df.assign(_c=v.notna().astype(float).where(v.notna())), "_c", 5, "sum").fillna(0.0)
    sd = (core.dog_lagged(df.assign(_q=v ** 2), "_q", 5) - m ** 2).clip(lower=0) ** 0.5
    fm = m.groupby(race).transform("mean")
    fsd = sd.groupby(race).transform("median")
    ms = (m.fillna(0.0) * n + K_PACE * fm) / (n + K_PACE)
    sds = (sd.fillna(0.0) * n + K_PACE * fsd) / (n + K_PACE)
    return ms.where(m.notna() | fm.notna()), sds, n


def design(df: pd.DataFrame) -> pd.DataFrame:
    """The pre-race inputs of each runner (earlier days only) and the labels (the race's own result)."""
    race = df.race_id
    x = pd.DataFrame(index=df.index)
    field = df.groupby("race_id").dog_id.transform("size").astype(float)
    for c in ("esr", "fsr", "gsr"):
        x[c], sd, n = _shrunk(df, c, race)
        if c == "esr":
            x["esr_sd"], x["esr_n"] = sd, n
    sec = pd.to_numeric(df.sectional, errors="coerce")
    first = sec.groupby(race).transform("min")
    lead = (sec == first).astype(float).where(sec.notna().groupby(race).transform("all"))
    lm = core.dog_lagged(df.assign(_l=lead), "_l", 10)
    ln = core.dog_lagged(df.assign(_c=lead.notna().astype(float).where(lead.notna())), "_c", 10, "sum").fillna(0.0)
    x["lead_rate"] = (lm.fillna(0.0) * ln + K_RATE / field) / (ln + K_RATE)
    brk = pd.to_numeric(df["brk"], errors="coerce") if "brk" in df else pd.Series(np.nan, index=df.index)
    bm = core.dog_lagged(df.assign(_b=brk), "_b", 10)
    bn = core.dog_lagged(df.assign(_c=brk.notna().astype(float).where(brk.notna())), "_c", 10, "sum").fillna(0.0)
    x["brk_rate"] = bm.fillna(0.0) * bn / (bn + K_RATE)
    for c in ("esr", "fsr", "gsr"):
        x[f"{c}_vs_field"] = bend._vs_field(x[c], race)
    x["esr_rank"] = x.esr.groupby(race).rank(ascending=False)
    trap = pd.to_numeric(df.trap, errors="coerce")
    x["trap"] = trap
    for k in range(1, 6):
        x[f"t{k}"] = (trap == k).astype(float)
    x["outside"] = (trap == trap.groupby(race).transform("max")).astype(float)
    esr = x.esr.fillna(x.esr.groupby(race).transform("mean"))
    box = esr.groupby([race, trap]).first()
    room = esr.groupby(race).transform("min")
    beside = lambda step: pd.Series(box.reindex(pd.MultiIndex.from_arrays([race, trap + step])).to_numpy(float),  # noqa: E731
                                    index=df.index)
    x["in_esr"] = (beside(-1).fillna(room) - esr).where(trap > trap.groupby(race).transform("min"))
    x["out_esr"] = (beside(1).fillna(room) - esr).where(trap < trap.groupby(race).transform("max"))
    x["lead"] = lead
    x["won"] = pd.to_numeric(df.won, errors="coerce")
    return x


def expected_position(u: np.ndarray, gi: np.ndarray) -> np.ndarray:
    """The exact Plackett-Luce expected position: 1 + sum over the race's other runners of sigmoid(u_j - u_i)."""
    n = len(u)
    if not n:
        return np.zeros(0)
    slot = pd.Series(gi).groupby(gi).cumcount().to_numpy()
    U = np.full((int(gi.max()) + 1, int(slot.max()) + 1), np.nan)
    U[gi, slot] = u
    d = U[gi] - u[:, None]                                          # rows x slots: u_j - u_i
    with np.errstate(over="ignore"):
        s = np.where(np.isfinite(d), 1.0 / (1.0 + np.exp(-d)), 0.0)
    return 0.5 + s.sum(axis=1)                                      # the row itself adds sigmoid(0) = 0.5


def together(gap: np.ndarray) -> np.ndarray:
    """The chance two arrivals whose utilities differ by ``gap`` fall within DELTA of each other (NaN in, 0 out)."""
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))                       # noqa: E731
    p = sig(DELTA - gap) - sig(-DELTA - gap)
    return np.where(np.isfinite(p), p, 0.0)


def crowding(u: np.ndarray, gi: np.ndarray, trap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row: the chance some dog drawn 1 to 2 traps away arrives with it, and 1 trap away (traps 1 to 6)."""
    n = len(u)
    ok = np.isfinite(trap) & (trap >= 1) & (trap <= SLOTS) & (trap == np.round(trap)) & np.isfinite(u)
    nr = int(gi.max()) + 1 if n else 0
    s = np.where(ok, trap + 1, 0).astype(np.int64)                  # two spare slots either side
    cell = gi * (SLOTS + 4) + s
    first = ok & ~pd.Series(np.where(ok, cell, -1 - np.arange(n))).duplicated().to_numpy()
    G = np.full((nr, SLOTS + 4), np.nan)
    G[gi[first], s[first]] = u[first]
    miss1, miss2 = np.ones(n), np.ones(n)
    for step in (-2, -1, 1, 2):
        other = G[gi, np.clip(s + step, 0, SLOTS + 3)]
        p = np.where(ok, together(other - u), 0.0)
        miss2 *= 1 - p
        if abs(step) == 1:
            miss1 *= 1 - p
    c2, c1 = 1 - miss2, 1 - miss1
    return np.where(ok, c2, np.nan), np.where(ok, c1, np.nan)


def _offsets(u: np.ndarray, gi: np.ndarray, lead: np.ndarray, cell: pd.Series) -> pd.Series:
    """The track-and-trap offsets: two passes of log((leads + K) / (expected leads + K)) over the fit rows."""
    off = pd.Series(0.0, index=pd.Index(cell.unique()))
    for _ in range(2):
        o = cell.map(off).to_numpy(float)
        p = core.group_softmax(u + o, gi)
        g = pd.DataFrame({"c": cell.to_numpy(), "l": lead, "p": p}).groupby("c")[["l", "p"]].sum()
        off = off.add(np.log((g.l + K_TRAP) / (g.p + K_TRAP)), fill_value=0.0)
    return off


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    x = design(df)
    racing = ~df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(True, index=df.index)
    year = core.day(df).dt.year
    cell = df.track.astype(str) + "|" + df.trap.astype(str)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    for c in COLUMNS:
        out[c] = np.nan
    notes = {"years": {}, "k_trap": K_TRAP, "k_pace": K_PACE, "delta": DELTA}
    field = df.groupby("race_id").dog_id.transform("size")
    lab = x.lead.notna() & x.lead.groupby(df.race_id).transform("sum").eq(1) & \
        x.lead.groupby(df.race_id).transform("count").eq(field)
    for y in sorted(year.dropna().unique()):
        fit_l = (year < y) & racing & lab
        races = df.race_id[fit_l].unique()
        if len(races) < MIN_FIT / 6:
            continue
        rng = np.random.default_rng(int(y))
        sample = rng.choice(races, int(MAX_FIT / 6), replace=False) if len(races) > MAX_FIT / 6 else races
        tr = fit_l & df.race_id.isin(sample)
        Xl, mu, sd = bend._matrix(x[tr], LEAD_X)
        bl = core.clogit_fit(Xl, bend._groups(df.race_id[tr]), x.lead[tr].to_numpy(float), lam=1.0)
        # the track-and-trap offsets over every earlier labelled race (not only the sample)
        Xa, _, _ = bend._matrix(x[fit_l], LEAD_X, mu, sd)
        off = _offsets(Xa @ bl, bend._groups(df.race_id[fit_l]), x.lead[fit_l].to_numpy(float), cell[fit_l])
        # the second stage on the sample: the winner, on the bend utility and the shrunk finishing figures
        ul_tr = Xl @ bl + cell[tr].map(off).fillna(0.0).to_numpy()
        xw = x[tr].assign(u=ul_tr)
        won1 = xw.won.groupby(df.race_id[tr]).transform("sum").eq(1)
        Xw, mw, sw = bend._matrix(xw[won1], WIN_X)
        bw = core.clogit_fit(Xw, bend._groups(df.race_id[tr][won1]), xw.won[won1].to_numpy(float), lam=1.0)
        te = (year == y).to_numpy()
        Xt, _, _ = bend._matrix(x[te], LEAD_X, mu, sd)
        gt = bend._groups(df.race_id[te])
        to = cell[te].map(off).fillna(0.0).to_numpy()
        u = Xt @ bl + to
        out.loc[te, f"{PREFIX}u"] = u
        out.loc[te, f"{PREFIX}trap_off"] = to
        out.loc[te, f"{PREFIX}p_lead"] = core.group_softmax(u, gt)
        out.loc[te, f"{PREFIX}exp_pos"] = expected_position(u, gt)
        c2, c1 = crowding(u, gt, x.trap[te].to_numpy(float))
        out.loc[te, f"{PREFIX}crowd"] = c2
        out.loc[te, f"{PREFIX}crowd1"] = c1
        Xwt, _, _ = bend._matrix(x[te].assign(u=u), WIN_X, mw, sw)
        out.loc[te, f"{PREFIX}p_win"] = core.group_softmax(Xwt @ bw, gt)
        notes["years"][int(y)] = {"fit_races": int(len(sample)), "lead_beta": dict(zip(LEAD_X, np.round(bl, 4))),
                                  "win_beta": dict(zip(WIN_X, np.round(bw, 4))),
                                  "trap_offsets": {k: round(float(v), 4) for k, v in off.items()}}
    cols = list(COLUMNS)
    out[cols] = out[cols].astype(np.float32)
    return out.sort_index().reset_index(drop=True)[["race_id", "dog_id"] + cols], notes
