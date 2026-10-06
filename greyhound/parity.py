"""The horse model's metric families on the GBGB results (the owner's ask, 6 Oct 2026: "calculate all metrics like the
horse racing model on our results data, e.g. pace, race shape, normalised finish position").

Greyhound results carry what the horse database lacks: a first sectional for every runner, so each past race's early
order is measured, not read from a comment. Built on the base engine's per-run figures (``greyhound.metrics``), in
three parts, every one lag-safe (one row a dog a day, lagged a day; group statistics from earlier days through
``model.lagsafe``):

1. **Per-run measures**, each describing one past run from that run's own race only (never features themselves):

       nfp     normalised finishing position, 1 the winner .. 0 last (the horse engine's NFP)
       wax     won less 1/field, a random runner's chance (WAX; its career sum over chance is WIV)
       lbw     lengths behind the winner;  lbsc  lengths better than the race's average runner
       gsr     the speed rating (the horse model's RSR / performance figure)
       ep      the early position: the order at the first sectional, 0 fastest away .. 1 slowest (EPF, measured)
       lead    led at the first sectional (1/0)
       gain    places made from the first sectional to the line, over the field (a closer's figure)
       shape   the finishing position against what the run's early position was worth in that race's pace at that
               track and distance, from earlier days (the shape/draw remodel: a run read against the race it met)
       mkt     the market's view: log(field x the SP's normalised chance);  ae  won less that chance
       nres    the finishing position less the one the market's order implied (GBGB's market position)
       bsp_ae, bsp_mkt   the same against the Betfair SP, where the price files price the race
       rs      the race's strength: the mean of the other runners' pre-race ratings (the race-strength family)

2. **Form windows** of every measure (``model/form_windows.py``'s ladder): career, the last run, the means of the last
   3 and 5, and the last 3, 5 and 10 weighted linearly by recency (``gp_<m>_car`` .. ``gp_<m>_w10``). A run whose
   measure is unknown keeps its place in a window and adds nothing to it.

3. **Today's race**, from the windows of the field:
   - the race shape: each dog's projected early position and its chance of leading at the first bend, the field's
     competition for the lead, how clear the likeliest leader is, how much of the field wants to be on the pace;
   - what leading is worth here: the win rate of first-bend leaders at the track and distance from earlier days,
     times this dog's share of the lead;
   - class: today's race strength, the dog's rating against it, today's strength against the races it has run in;
   - WIV: career wins over the wins its fields gave it by chance;
   - the trainer's last 30 days: strike rate and A/E against the SP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.lagsafe import race_lagged_expanding_mean

#: the measures windowed, in feature order
MEASURES = ["nfp", "wax", "lbw", "lbsc", "gsr", "ep", "lead", "gain", "shape", "mkt", "ae", "nres", "bsp_ae",
            "bsp_mkt", "rs"]
WINDOWS = ("car", "l1", "m3", "m5", "w3", "w5", "w10")
PREFIX = "gp_"


def attach_bsp(runs: pd.DataFrame, prices: pd.DataFrame | None) -> pd.DataFrame:
    """Each run's Betfair SP from the price files (GB win markets), by date, off time, track and trap; normalised
    within the race where the whole field is priced."""
    df = runs.copy()
    df["bsp"] = np.nan
    if prices is None or not len(prices):
        df["bsp_p_norm"] = np.nan
        return df
    w = prices[(prices.market == "win") & (prices.country == "GB")][["race_date", "race_time", "track", "trap", "bsp"]]
    w = w.dropna(subset=["trap"]).drop_duplicates(["race_date", "race_time", "track", "trap"], keep="last")
    w = w.assign(trap=w.trap.astype("int64"))
    df = df.drop(columns="bsp").merge(w, on=["race_date", "race_time", "track", "trap"], how="left")
    q = df.bsp.where(df.bsp > 1)
    whole = q.notna().groupby(df.race_id).transform("all")
    p = (1.0 / q).where(whole)
    df["bsp_p_norm"] = p / p.groupby(df.race_id).transform("sum")
    return df


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """The per-run measures (part 1). ``df`` carries the base engine's per-run figures (gsr, esr, lbw) and its
    pre-race gsr_ewm."""
    out = {}
    n = df.field.astype(float)
    denom = (n - 1).where(n > 1)
    out["nfp"] = (n - df.position) / denom
    out["wax"] = df.won - 1.0 / n
    out["lbsc"] = df.lbw.groupby(df.race_id).transform("mean") - df.lbw
    # the early order at the first sectional (lower sectional = faster away); a race with no sectionals is unknown
    sec_rank = df.sectional.groupby(df.race_id).rank(method="average")
    out["ep"] = (sec_rank - 1) / denom
    out["lead"] = (sec_rank == 1).astype(float).where(df.sectional.notna())
    out["gain"] = (sec_rank - df.position) / denom
    # the race's pace: the field's early speed against the track and distance's earlier races (lengths)
    race_pace = df.esr.groupby(df.race_id).transform("mean")
    tmp = df.assign(_pace=race_pace)
    pace_par = race_lagged_expanding_mean(tmp, ["track", "distance_m"], "_pace")
    pace = race_pace - pace_par
    pace_b = np.select([pace < -0.75, pace > 0.75], [0, 2], 1).astype(float)
    ep_b = np.select([sec_rank == 1, sec_rank == 2, sec_rank <= 4], [0, 1, 2], 3).astype(float)
    known = pace.notna() & df.sectional.notna()
    tmp = tmp.assign(_nfp=out["nfp"].where(known), _pb=np.where(known, pace_b, -1.0), _eb=np.where(known, ep_b, -1.0))
    expect = race_lagged_expanding_mean(tmp, ["track", "distance_m", "_eb", "_pb"], "_nfp", min_races=20)
    out["shape"] = (out["nfp"] - expect).where(known)
    # the market: GBGB's SP and its order; the Betfair SP where the price files have the race
    out["mkt"] = np.log(n * df.sp_p_norm)
    out["ae"] = df.won - df.sp_p_norm
    mpos = pd.to_numeric(df.get("market_pos"), errors="coerce") if "market_pos" in df else pd.Series(np.nan, df.index)
    out["nres"] = out["nfp"] - (n - mpos) / denom
    out["bsp_ae"] = df.won - df.bsp_p_norm
    out["bsp_mkt"] = np.log(n * df.bsp_p_norm)
    # the race's strength: the other runners' pre-race ratings (known before the off)
    r = df.gsr_ewm
    tot, cnt = r.groupby(df.race_id).transform("sum"), r.notna().groupby(df.race_id).transform("sum")
    out["rs"] = (tot - r.fillna(0)) / (cnt - r.notna()).replace(0, np.nan)
    m = pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)
    m["gsr"], m["lbw"] = df.gsr, df.lbw
    return m


def _windows(dd, col: str) -> dict:
    """The seven windows of one measure on the dog-days (``greyhound.metrics.DogDays``)."""
    prev = {i: dd.prev(col, i) for i in range(1, 11)}
    known = dd.dd[col].notna().astype(float)
    s = dd.dd[col].fillna(0.0)
    cs, ck = s.groupby(dd.dd.dog_id).cumsum() - s, known.groupby(dd.dd.dog_id).cumsum() - known
    w = {"car": cs / ck.replace(0, np.nan), "l1": prev[1], "m3": dd.window(col, 3), "m5": dd.window(col, 5)}
    for k in (3, 5, 10):
        num = sum((k - i + 1) * prev[i].fillna(0.0) for i in range(1, k + 1))
        den = sum((k - i + 1) * prev[i].notna() for i in range(1, k + 1))
        w[f"w{k}"] = num / den.replace(0, np.nan)
    return w


def _trainer_30d(df: pd.DataFrame) -> pd.DataFrame:
    """The trainer's last 30 days before today: runs, strike rate, wins over the SP's expected wins."""
    d = df[["trainer", "race_date", "won", "sp_p_norm"]].copy()
    d["_d"] = pd.to_datetime(d.race_date)
    day = d.groupby(["trainer", "_d"]).agg(w=("won", "sum"), x=("sp_p_norm", "sum"), n=("won", "size")).reset_index()
    day = day.sort_values(["trainer", "_d"])
    roll = (day.set_index("_d").groupby("trainer")[["w", "x", "n"]]
            .rolling("30D", closed="left").sum().reset_index())
    key = pd.MultiIndex.from_frame(roll[["trainer", "_d"]])
    idx = pd.MultiIndex.from_arrays([d.trainer, d._d])
    get = lambda c: pd.Series(roll[c].values, index=key).reindex(idx).values  # noqa: E731
    w, x, n = get("w"), get("x"), get("n")
    return pd.DataFrame({"gp_trainer30_runs": n, "gp_trainer30_win": (w + 1.0) / (n + 6.0),
                         "gp_trainer30_ae": (w + 3.0) / (x + 3.0)}, index=df.index)


def calculate(df: pd.DataFrame, dog_days) -> tuple[pd.DataFrame, list[str]]:
    """Parts 1-3 for the runs ``df`` (the base engine's frame); returns the new columns and their names.
    ``dog_days`` is ``greyhound.metrics.DogDays``."""
    m = per_run(df)
    tmp = df[["dog_id", "race_date"]].join(m)
    dd = dog_days(tmp, MEASURES, [])
    day = {}
    for col in MEASURES:
        for name, v in _windows(dd, col).items():
            day[f"{PREFIX}{col}_{name}"] = v
    # career counts for the shrunk lead share and WIV
    lead_known = dd.dd["lead"].notna().astype(float)
    day["_lead_n"] = lead_known.groupby(dd.dd.dog_id).cumsum() - lead_known
    cols = {k: dd.back(v) for k, v in day.items()}
    out = pd.DataFrame(cols, index=df.index)
    # WIV: career wins over the wins chance gave (from the dog's earlier runs: wins and 1/field summed)
    fair = (1.0 / df.field).where(df.position.notna())
    t2 = df[["dog_id", "raceid", "race_date"]].assign(_won=df.won, _fair=fair)
    from model.lagsafe import race_lagged_expanding_sum
    ws, fs = race_lagged_expanding_sum(t2, "dog_id", "_won"), race_lagged_expanding_sum(t2, "dog_id", "_fair")
    out["gp_wiv"] = (ws.fillna(0) + 0.5) / (fs.fillna(0) + 0.5)
    # ---- today's race shape, from the field's projections
    proj_ep = out[f"{PREFIX}ep_w5"].fillna(0.5)
    lead_n = out.pop("_lead_n").fillna(0)
    lead_rate = ((out[f"{PREFIX}lead_car"].fillna(0) * lead_n + 3.0 / df.field) / (lead_n + 3.0))
    g = df.race_id
    out["gp_proj_ep"] = proj_ep
    out["gp_proj_ep_rank"] = proj_ep.groupby(g).rank(method="average")
    best = proj_ep.groupby(g).transform("min")
    o = pd.DataFrame({"g": g, "p": proj_ep}).sort_values(["g", "p"], kind="stable")
    sec = o[o.groupby("g").cumcount() == 1].set_index("g").p
    second = g.map(sec)
    out["gp_proj_ep_gap"] = proj_ep - best
    out["gp_leader_clarity"] = second - best
    out["gp_n_pace"] = (proj_ep < 0.3).groupby(g).transform("sum").astype(float)
    out["gp_lead_rate"] = lead_rate
    out["gp_lead_share"] = lead_rate / lead_rate.groupby(g).transform("sum")
    out["gp_lead_comp"] = lead_rate.groupby(g).transform("sum") - lead_rate
    # what leading at the first bend is worth here, from earlier days
    t3 = df[["raceid", "race_date", "track", "distance_m"]].assign(_lw=df.won.where(m["lead"] == 1),
                                                                    _ow=df.won.where(m["lead"] == 0))
    v_lead = race_lagged_expanding_mean(t3, ["track", "distance_m"], "_lw", min_races=20)
    v_other = race_lagged_expanding_mean(t3, ["track", "distance_m"], "_ow", min_races=20)
    out["gp_lead_value"] = v_lead
    out["gp_pos_value"] = out.gp_lead_share * v_lead + (1 - out.gp_lead_share) * v_other
    # class: today's race strength (the other runners' pre-race ratings) against the dog and its past races
    out["gp_rs_today"] = m["rs"]
    out["gp_rating_vs_rs"] = df.gsr_ewm - m["rs"]
    out["gp_rs_vs_past"] = m["rs"] - out[f"{PREFIX}rs_w5"]
    # the trainer's last 30 days
    out = out.join(_trainer_30d(df))
    # today's market-free race-relative reads of the new windows
    for col in (f"{PREFIX}nfp_w5", f"{PREFIX}shape_w5", f"{PREFIX}gain_w5", f"{PREFIX}nres_w5", f"{PREFIX}bsp_ae_w10"):
        x = out[col]
        out[f"{col}_rank"] = x.groupby(g).rank(ascending=False, method="average")
        out[f"{col}_z"] = (x - x.groupby(g).transform("mean")) / x.groupby(g).transform("std").replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, list(out.columns)
