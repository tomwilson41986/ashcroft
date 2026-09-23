"""What the day has shown by the time a race is run.

Every other block steps back whole days, so a model built from them never sees
what the market learns during the afternoon. By the 4.10 the first four races at
the meeting have been run: whether low draws or front-runners have been beating
their prices, whether a yard's earlier runners today have run above or below
their market, whether a jockey is riding well. Punters watch all of this; the
question is whether the price has fully caught up by the off.

Only races whose off time is at least GAP_MINUTES earlier on the same day
contribute (the same meeting for draw and pace, any meeting for trainer and
jockey), so neither the race itself nor anything run after it -- or so shortly
before it that its result may not be known -- moves a feature. An off time that
does not parse makes the race neither a source nor a target, since the reader
would otherwise take it for midnight, before every other race. The residuals
used throughout measure a run against its own price:

    bmr = (BSP rank - finishing rank) / (ranked runners - 1)   beat the market's order
    ae  = won - pi                                             beat the market's probability

Features (a model using them has to see the day's results as they come in, so
it bets close to the off, at BSP):

  id_race_index        races already run at this meeting today
  id_draw_n            earlier ranked runners in flat races with stalls at this
                       meeting and distance class (sprint / round)
  id_draw_slope        slope of bmr on relative draw among them, shrunk to zero
  id_draw_edge_mine    this runner's relative draw (centred) times that slope
  id_draw_ae_mine      A/E of those earlier runners drawn in this runner's third
  id_pace_n            earlier runners at this meeting with an early position
  id_pace_slope        slope of bmr on early position (from the in-running comment)
  id_pace_edge_mine    this runner's usual early position (its last three runs,
                       centred) times that slope
  id_front_ae          A/E of earlier front-runners (made the running or disputed)
  id_{trainer,jockey}_runs_today, _ae_today, _bmr_today
                       the connection's earlier runners today at any meeting
  id_{trainer,jockey}_runs_3d, _ae_3d, _bmr_3d
                       the same over the last 72 hours (earlier days and earlier today)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.custom_metrics import _EPF_COMPILED
from model.lagsafe import race_minutes

GAP_MINUTES = 10.0
SPRINT_MAX_FURLONGS = 7.5
K_SLOPE = 2.5          # pseudo sum of squares at a zero slope (about thirty runners of draw spread)
K_AE = 30.0            # pseudo-runners at a zero A/E residual
K_BMR = 10.0           # pseudo-runners at a zero rank residual
FRONT_EPF = 5.5        # made the running or disputed the lead
RECENT_MINUTES = 3 * 1440.0   # the connections' last three days, earlier today included

INDAY_FEATURES = ["id_race_index",
                  "id_draw_n", "id_draw_slope", "id_draw_edge_mine", "id_draw_ae_mine",
                  "id_pace_n", "id_pace_slope", "id_pace_edge_mine", "id_front_ae",
                  "id_trainer_runs_today", "id_trainer_ae_today", "id_trainer_bmr_today",
                  "id_jockey_runs_today", "id_jockey_ae_today", "id_jockey_bmr_today",
                  "id_trainer_runs_3d", "id_trainer_ae_3d", "id_trainer_bmr_3d",
                  "id_jockey_runs_3d", "id_jockey_ae_3d", "id_jockey_bmr_3d"]


def _race_key(df: pd.DataFrame) -> pd.Series:
    if "raceid" in df.columns:
        return df["raceid"].astype(str)
    return (pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d") + "|" + df["track"].astype(str)
            + "|" + df["race_time"].astype(str))


def off_minutes(race_time: pd.Series) -> pd.Series:
    """race_minutes, but NaN where the time does not parse (race_minutes reads that as midnight)."""
    s = pd.Series(race_time).astype(str).str.strip().str.rstrip(".").str.replace(".", ":", regex=False)
    ok = s.str.match(r"^\d{1,2}(?::\d{1,2})?$").fillna(False)
    return race_minutes(race_time).where(ok.to_numpy())


def early_position(comment) -> float:
    """First positional phrase of an in-running comment on the 1-6 EPF scale; NaN if none."""
    if not isinstance(comment, str) or not comment:
        return np.nan
    c = comment.lower()
    best = None
    for value, rx in _EPF_COMPILED:
        m = rx.search(c)
        if m is None:
            continue
        key = (m.start(), -(m.end() - m.start()))
        if best is None or key < best[0]:
            best = (key, value)
    return np.nan if best is None else float(best[1])


def earlier_sums(cell_group, cell_t, cell_vals: dict, q_group, q_t, gap: float = GAP_MINUTES) -> tuple[dict, np.ndarray]:
    """For each query (group, time): the sum of each cell value over cells of the same group
    at least `gap` minutes earlier, and how many such cells there were.

    Prefix sums restart at every group and run in time order, so a query's sums are built
    only from the cells it may see -- nothing from another group, or from a later cell,
    enters the arithmetic, and the result is bit-for-bit independent of them. Times are
    minutes on any axis (within a day, or across days)."""
    cell_t = np.asarray(cell_t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)
    codes, _ = pd.factorize(pd.concat([pd.Series(np.asarray(cell_group, dtype=object)),
                                       pd.Series(np.asarray(q_group, dtype=object))], ignore_index=True))
    nc = len(cell_t)
    cg, qg = codes[:nc].astype(np.float64), codes[nc:].astype(np.float64)
    # rebase so every cell time is >= 0 and space the groups wider than the time range, so a
    # group's keys never meet its neighbours' whatever the query offset
    known = np.r_[cell_t[~np.isnan(cell_t)], q_t[~np.isnan(q_t)]]
    t0 = float(known.min()) if len(known) else 0.0
    t_rng = float(known.max()) - t0 if len(known) else 0.0
    span = t_rng + abs(gap) + 1e4
    cell_t, q_t = cell_t - t0, q_t - t0
    order = np.lexsort((cell_t, cg))
    cg_s, ct_s = cg[order], cell_t[order]
    key_c = cg_s * span + ct_s
    q_ok = ~np.isnan(q_t) & (qg >= 0)
    key_q = np.where(q_ok, qg * span + (q_t - gap), -np.inf)
    hi = np.searchsorted(key_c, key_q, side="right")
    lo = np.searchsorted(key_c, np.where(q_ok, qg * span - 0.5, -np.inf), side="left")
    n_before = np.where(q_ok, np.maximum(hi - lo, 0), 0)
    take = np.clip(hi - 1, 0, max(nc - 1, 0))
    out = {}
    for name, v in cell_vals.items():
        vs = pd.Series(np.asarray(v, dtype=float)[order]).fillna(0.0)
        cum = vs.groupby(cg_s, sort=False).cumsum().to_numpy() if nc else np.zeros(0)
        out[name] = np.where(n_before > 0, cum[take] if nc else 0.0, 0.0)
    return out, n_before


def window_sums(cell_group, cell_t, cell_vals: dict, q_group, q_t, gap: float, window: float) -> dict:
    """Sums over the same group's cells between `window` and `gap` minutes before each query.
    Both prefix sums stop before the query, so the difference is as blind to later cells."""
    hi, _ = earlier_sums(cell_group, cell_t, cell_vals, q_group, q_t, gap)
    lo, _ = earlier_sums(cell_group, cell_t, cell_vals, q_group, q_t, window)
    return {k: hi[k] - lo[k] for k in hi}


def add_inday_features(df: pd.DataFrame, gap: float = GAP_MINUTES) -> tuple[pd.DataFrame, list[str]]:
    race = _race_key(df)
    d = pd.DataFrame(index=df.index)
    d["_race"] = race.to_numpy()
    d["_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy()
    d["_t"] = off_minutes(df["race_time"]).to_numpy()
    d["_track"] = df["track"].astype(str).to_numpy()
    bsp = pd.to_numeric(df["bfsp"], errors="coerce").where(lambda s: s > 1.0)
    pos = pd.to_numeric(df.get("placing_numerical"), errors="coerce").where(lambda s: s >= 1)
    inv = 1.0 / bsp
    pi = inv / inv.groupby(race).transform("sum")
    won = (pos == 1).astype(float).where(pos.notna())
    d["_ae"] = (won - pi).where(pi.notna() & won.notna())
    ranked = bsp.notna() & pos.notna()
    mrank = bsp.where(ranked).groupby(race).rank(method="average")
    frank = pos.where(ranked).groupby(race).rank(method="average")
    cnt = ranked.groupby(race).transform("sum")
    d["_bmr"] = ((mrank - frank) / (cnt - 1).where(cnt > 1)).where(ranked)

    # relative draw among the runners that took part, in races run from stalls
    stall = pd.to_numeric(df.get("stall"), errors="coerce").where(lambda s: s > 0)
    has_stall = stall.notna()
    stall_share = has_stall.groupby(race).transform("mean")
    srank = stall.groupby(race).rank(method="average")
    scnt = has_stall.groupby(race).transform("sum")
    rel = ((srank - 1) / (scnt - 1).where(scnt > 1)).where((stall_share >= 0.8) & has_stall)
    d["_x_draw"] = rel - 0.5
    d["_third"] = np.floor(rel.clip(0, 1) * 3).clip(0, 2)
    dist = pd.to_numeric(df.get("dist_furlongs"), errors="coerce")
    d["_dclass"] = np.where(dist <= SPRINT_MAX_FURLONGS, "sprint", np.where(dist.notna(), "round", "unknown"))

    # early position in THIS run (a source only for later races) and the runner's usual one
    comment = df["comment"] if "comment" in df.columns else pd.Series(np.nan, index=df.index)
    epf = comment.map(early_position).astype(float)
    d["_x_pace"] = (epf - 3.5) / 2.5
    d["_front"] = (epf >= FRONT_EPF).astype(float).where(epf.notna())
    horse = df["horse_name"].astype(str)
    order = pd.DataFrame({"h": horse, "date": d["_date"], "t": d["_t"].fillna(-1.0)}).sort_values(
        ["h", "date", "t"], kind="stable").index
    h_sorted = horse.loc[order]
    prev = epf.loc[order].groupby(h_sorted, sort=False).shift(1)
    usual = (prev.groupby(h_sorted, sort=False).rolling(3, min_periods=1).mean()
             .reset_index(level=0, drop=True))
    d["_x_usual"] = ((usual - 3.5) / 2.5).reindex(df.index)

    valid = d["_t"].notna()
    res = pd.DataFrame(index=df.index)

    # races already run at this meeting
    meet = d["_date"] + "|" + d["_track"]
    races = d.loc[valid].drop_duplicates("_race")
    s, n_before = earlier_sums(meet.loc[races.index].to_numpy(), races["_t"].to_numpy(), {},
                               meet.to_numpy(), d["_t"].to_numpy(), gap)
    res["id_race_index"] = pd.Series(n_before, index=df.index).where(valid)

    # draw: earlier flat races with stalls, same meeting and distance class
    src = valid & d["_x_draw"].notna()
    both = src & d["_bmr"].notna()
    g_draw = meet + "|" + d["_dclass"]
    cell = pd.DataFrame({
        "race": d["_race"], "g": g_draw, "t": d["_t"],
        "xy": (d["_x_draw"] * d["_bmr"]).where(both), "xx": (d["_x_draw"] ** 2).where(both),
        "n": both.astype(float).where(both),
        **{f"ae{k}": d["_ae"].where(src & (d["_third"] == k)) for k in range(3)},
        **{f"na{k}": (src & (d["_third"] == k) & d["_ae"].notna()).astype(float) for k in range(3)},
    }).loc[src]
    agg = cell.groupby("race", sort=False).agg(g=("g", "first"), t=("t", "first"),
                                               **{c: (c, "sum") for c in cell.columns if c not in ("race", "g", "t")})
    s, _ = earlier_sums(agg["g"].to_numpy(), agg["t"].to_numpy(),
                        {c: agg[c].to_numpy() for c in agg.columns if c not in ("g", "t")},
                        g_draw.to_numpy(), d["_t"].to_numpy(), gap)
    slope = s["xy"] / (s["xx"] + K_SLOPE)
    has_draw = d["_x_draw"].notna().to_numpy()
    res["id_draw_n"] = pd.Series(s["n"], index=df.index).where(valid)
    res["id_draw_slope"] = pd.Series(slope, index=df.index).where(valid)
    res["id_draw_edge_mine"] = pd.Series(slope * d["_x_draw"].to_numpy(), index=df.index).where(valid & has_draw)
    third = d["_third"].to_numpy()
    ae_sum = np.select([third == 0, third == 1, third == 2], [s["ae0"], s["ae1"], s["ae2"]], default=np.nan)
    ae_n = np.select([third == 0, third == 1, third == 2], [s["na0"], s["na1"], s["na2"]], default=np.nan)
    res["id_draw_ae_mine"] = pd.Series(ae_sum / (ae_n + K_AE), index=df.index).where(valid & has_draw)

    # pace: earlier races at this meeting
    src = valid & d["_x_pace"].notna()
    both = src & d["_bmr"].notna()
    cell = pd.DataFrame({
        "race": d["_race"], "g": meet, "t": d["_t"],
        "xy": (d["_x_pace"] * d["_bmr"]).where(both), "xx": (d["_x_pace"] ** 2).where(both),
        "n": both.astype(float),
        "fae": d["_ae"].where(src & (d["_front"] == 1)),
        "fn": (src & (d["_front"] == 1) & d["_ae"].notna()).astype(float),
    }).loc[src]
    agg = cell.groupby("race", sort=False).agg(g=("g", "first"), t=("t", "first"),
                                               **{c: (c, "sum") for c in ("xy", "xx", "n", "fae", "fn")})
    s, _ = earlier_sums(agg["g"].to_numpy(), agg["t"].to_numpy(),
                        {c: agg[c].to_numpy() for c in ("xy", "xx", "n", "fae", "fn")},
                        meet.to_numpy(), d["_t"].to_numpy(), gap)
    slope = s["xy"] / (s["xx"] + K_SLOPE)
    res["id_pace_n"] = pd.Series(s["n"], index=df.index).where(valid)
    res["id_pace_slope"] = pd.Series(slope, index=df.index).where(valid)
    res["id_pace_edge_mine"] = pd.Series(slope * d["_x_usual"].to_numpy(), index=df.index).where(
        valid & d["_x_usual"].notna())
    res["id_front_ae"] = pd.Series(s["fae"] / (s["fn"] + K_AE), index=df.index).where(valid)

    # connections: earlier runners today at any meeting (and, below, over the last three days)
    day = (pd.to_datetime(pd.Series(d["_date"], index=df.index), errors="coerce") - pd.Timestamp("2000-01-01")).dt.days
    abs_t = day * 1440.0 + d["_t"]
    for ent, col in (("trainer", "trainer"), ("jockey", "jockey_name")):
        if col not in df.columns:
            continue
        who = df[col].astype(str).str.strip()
        known = who.ne("") & who.str.lower().ne("nan") & df[col].notna()
        key = pd.Series(d["_date"], index=df.index) + "|" + who
        src = valid & known & (d["_ae"].notna() | d["_bmr"].notna())
        cell = pd.DataFrame({"race": d["_race"], "g": key, "t": d["_t"],
                             "ae": d["_ae"].where(src), "na": (src & d["_ae"].notna()).astype(float),
                             "bmr": d["_bmr"].where(src), "nb": (src & d["_bmr"].notna()).astype(float)}).loc[src]
        agg = cell.groupby(["race", "g"], sort=False).agg(t=("t", "first"), ae=("ae", "sum"), na=("na", "sum"),
                                                          bmr=("bmr", "sum"), nb=("nb", "sum")).reset_index()
        s, _ = earlier_sums(agg["g"].to_numpy(), agg["t"].to_numpy(),
                            {c: agg[c].to_numpy() for c in ("ae", "na", "bmr", "nb")},
                            key.to_numpy(), d["_t"].to_numpy(), gap)
        ok = valid & known
        res[f"id_{ent}_runs_today"] = pd.Series(s["na"], index=df.index).where(ok)
        res[f"id_{ent}_ae_today"] = pd.Series(s["ae"] / (s["na"] + K_AE), index=df.index).where(ok)
        res[f"id_{ent}_bmr_today"] = pd.Series(s["bmr"] / (s["nb"] + K_BMR), index=df.index).where(ok)

        # the same over the last three days: a streak the market may be slow to credit
        g3 = who.to_numpy()
        cell3 = cell.assign(g=who.loc[cell.index].to_numpy(), t=abs_t.loc[cell.index].to_numpy())
        agg3 = cell3.groupby(["race", "g"], sort=False).agg(t=("t", "first"), ae=("ae", "sum"), na=("na", "sum"),
                                                            bmr=("bmr", "sum"), nb=("nb", "sum")).reset_index()
        s3 = window_sums(agg3["g"].to_numpy(), agg3["t"].to_numpy(),
                         {c: agg3[c].to_numpy() for c in ("ae", "na", "bmr", "nb")},
                         g3, abs_t.to_numpy(), gap, RECENT_MINUTES)
        res[f"id_{ent}_runs_3d"] = pd.Series(s3["na"], index=df.index).where(ok)
        res[f"id_{ent}_ae_3d"] = pd.Series(s3["ae"] / (s3["na"] + K_AE), index=df.index).where(ok)
        res[f"id_{ent}_bmr_3d"] = pd.Series(s3["bmr"] / (s3["nb"] + K_BMR), index=df.index).where(ok)

    out = df.copy()
    names = [c for c in INDAY_FEATURES if c in res.columns]
    for c in names:
        out[c] = res[c].astype(float)
    return out, names
