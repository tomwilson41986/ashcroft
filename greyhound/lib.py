"""The horse model's feature library, carried to greyhounds where nothing in the greyhound engine covers it yet (the
owner's ask of 7 Oct 2026: "ideate and use our feature library").

Three ideas from ``model/blocks`` (each scored on the horses there):

- **Elo** (``model/blocks/elo.py``): every dog's strength inferred from whom it beat across the whole race network,
  not only today's rivals (head to head, ``greyhound.hrb``). A multi-runner Elo: each finisher's expected share of the
  field beaten, E_i = mean_j 1 / (1 + 10 ** ((R_j - R_i) / 400)), against its actual share
  S_i = (finishers - place) / (finishers - 1); R_i += K (S_i - E_i). Start 1500, finishers only.
- **A Kalman rating on the speed figure** (``model/blocks/kalman.py``): ability as a state that drifts between runs,
  more over a long gap, read with its uncertainty. Each run's GSR (lengths against the track's standard, the engine's
  speed figure) is a noisy reading of it. The windows average figures; they cannot say how sure they are.
- **Form lines** (``model/blocks/form_lines.py``): how a dog's last races have worked out, i.e. what the rivals it met
  there have won since, before today.

Everything is read as it stood before the row's day: a day's own results never reach its features (the lag-safety
test covers the block), so a morning card and the resulted rows give the same values.

    lb_elo, lb_elo_n, lb_elo_z, lb_elo_gap, lb_elo_field
    lb_kr, lb_kr_sd, lb_kr_n, lb_kr_z, lb_kr_gap, lb_kr_vs_par
    lb_fl1_n, lb_fl1_wins, lb_fl1_wr, lb_fl3_n, lb_fl3_wr
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREFIX = "lb_"
K_ELO = 24.0
START = 1500.0
FL_NEXT = 3                     # each rival's next runs counted for a race's form line
KR_R = 6.0                      # a run's GSR reading variance (lengths^2)
KR_Q = 0.5                      # ability drift variance a 30 days (lengths^2)
KR_INIT = 9.0                   # a dog's first rating variance


def _day_codes(df: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(df.race_date).values.astype("datetime64[D]").astype(np.int64)


def _race_stats(v: pd.Series, race: pd.Series, prefix: str) -> dict:
    """z, gap to the best and the race mean of ``v`` within each race (over the runners it rates)."""
    g = v.groupby(race)
    mean, sd, best = g.transform("mean"), g.transform("std"), g.transform("max")
    return {f"{prefix}_z": ((v - mean) / sd.replace(0, np.nan)).astype(np.float32),
            f"{prefix}_gap": (v - best).astype(np.float32), f"{prefix}_field": mean.astype(np.float32)}


def elo(df: pd.DataFrame) -> pd.DataFrame:
    """Each row's Elo and finishes before its day."""
    dog = pd.factorize(df.dog_id)[0]
    day = _day_codes(df)
    pos = df.position.to_numpy(float)
    race = pd.factorize(df.race_id)[0]
    R = np.full(dog.max() + 1, START)
    N = np.zeros(dog.max() + 1)
    out_r, out_n = np.full(len(df), np.nan), np.zeros(len(df))
    order = np.lexsort((race, day))
    bounds = np.flatnonzero(np.diff(day[order])) + 1
    for rows in np.split(order, bounds):                      # one day at a time: read before, update after
        d = dog[rows]
        out_r[rows] = np.where(N[d] > 0, R[d], np.nan)
        out_n[rows] = N[d]
        fin = rows[~np.isnan(pos[rows])]
        if not len(fin):
            continue
        fin = fin[np.argsort(race[fin], kind="stable")]
        rb = np.flatnonzero(np.diff(race[fin])) + 1
        delta = np.zeros(len(fin))
        start = 0
        for end in list(rb) + [len(fin)]:
            idx = np.arange(start, end)
            start = end
            n = len(idx)
            if n < 2:
                continue
            r = R[dog[fin[idx]]]
            e = (1.0 / (1.0 + 10 ** ((r[None, :] - r[:, None]) / 400.0))).sum(1) - 0.5   # less the self term
            e /= (n - 1)
            place = pd.Series(pos[fin[idx]]).rank(method="average").to_numpy()
            s = (n - place) / (n - 1)
            delta[idx] = K_ELO * (s - e)
        np.add.at(R, dog[fin], delta)
        np.add.at(N, dog[fin], 1.0)
    return pd.DataFrame({"r": out_r, "n": out_n}, index=df.index)


def kalman(df: pd.DataFrame, par: pd.Series | None = None) -> pd.DataFrame:
    """Each row's speed-rating state before its day: mean, sd and readings behind it. A local-level filter per dog:
    the variance grows by ``q`` a 30 days between runs (more over a dog's first runs), each run's GSR read with
    variance ``r``; a dog's first state is its first race's grade par (or the population mean) with the population
    variance."""
    gsr = df.gsr.to_numpy(float)
    # fixed on the GSR's scale (lengths): estimated from the data they would read the priced day's own figures
    r, q, init = KR_R, KR_Q, KR_INIT
    mu0 = par.to_numpy(float) if par is not None else np.full(len(df), np.nan)
    mu0 = np.where(np.isfinite(mu0), mu0, 0.0)
    dog = df.dog_id.to_numpy()
    day = _day_codes(df)
    order = np.lexsort((day, dog))
    m_out, sd_out, n_out = np.full(len(df), np.nan), np.full(len(df), np.nan), np.zeros(len(df))
    cur_dog, m, v, n, last = None, 0.0, init, 0, 0
    i, L = 0, len(order)
    while i < L:
        j = i
        dg, dy = dog[order[i]], day[order[i]]
        while j < L and dog[order[j]] == dg and day[order[j]] == dy:
            j += 1
        rows = order[i:j]
        if dg != cur_dog:
            cur_dog, m, v, n, last = dg, mu0[rows[0]], init, 0, dy
        else:
            gap = max(dy - last, 0)
            v = v + q * (gap / 30.0) * (2.0 if n < 5 else 1.0)
        m_out[rows], sd_out[rows], n_out[rows] = (m if n else np.nan), np.sqrt(v), n
        for z in gsr[rows]:                                   # the day's readings enter after the day is priced
            if np.isfinite(z):
                k = v / (v + r)
                m, v, n = m + k * (z - m), (1 - k) * v, n + 1
        last = dy
        i = j
    return pd.DataFrame({"m": m_out, "sd": sd_out, "n": n_out}, index=df.index)


def form_lines(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, its dog's last race (and its last three) before the row's day: the rivals' next runs after that
    race, up to ``FL_NEXT`` each, that came on days before the row's day, and how many were won."""
    day = _day_codes(df)
    base = pd.DataFrame({"race": df.race_id.to_numpy(), "dog": df.dog_id.to_numpy(), "day": day,
                         "won": df.won.to_numpy(float), "res": df.position.notna().to_numpy()})
    hist = base[base.res].sort_values(["dog", "day"], kind="stable").reset_index(drop=True)
    # each rival's next runs after each of its races
    nxt = []
    g = hist.groupby("dog", sort=False)
    for k in range(1, FL_NEXT + 1):
        nxt.append(pd.DataFrame({"race": hist.race, "day": g.day.shift(-k), "won": g.won.shift(-k)}).dropna())
    e = pd.concat(nxt, ignore_index=True)                    # (race, a rival's later day, won)
    e = e[e.day > hist.set_index("race").day.groupby(level=0).first().reindex(e.race).to_numpy()]
    rc = {r: i for i, r in enumerate(pd.unique(hist.race))}
    e["rc"] = e.race.map(rc)
    e = e.sort_values(["rc", "day"], kind="stable").reset_index(drop=True)
    e["cw"] = e.groupby("rc").won.cumsum()
    e["cn"] = e.groupby("rc").cumcount() + 1
    key = e.rc.to_numpy(np.int64) * 100_000 + e.day.to_numpy(np.int64)
    cw, cn, rc_arr = e.cw.to_numpy(), e.cn.to_numpy(), e.rc.to_numpy(np.int64)

    def since(race_ids: pd.Series, today: np.ndarray):
        """(runs, wins) of the race's rivals on days before ``today``."""
        r = race_ids.map(rc)
        has = r.notna().to_numpy()
        n, w = np.zeros(len(r)), np.zeros(len(r))
        rr = r.to_numpy(float)[has].astype(np.int64)
        q = rr * 100_000 + today[has] - 1                    # the last entry on a day before today
        pos = np.searchsorted(key, q, side="right") - 1
        valid = (pos >= 0) & (rc_arr[np.clip(pos, 0, None)] == rr)
        n_h, w_h = np.zeros(len(rr)), np.zeros(len(rr))
        n_h[valid], w_h[valid] = cn[pos[valid]], cw[pos[valid]]
        n[has], w[has] = n_h, w_h
        return n, w

    # each row's previous races (on earlier days), from the dog's day-ordered history
    rows = base.reset_index().rename(columns={"index": "row"})
    days = hist.groupby(["dog", "day"], sort=False).race.last().reset_index()
    out = {}
    for lag in (1, 2, 3):
        # the race 'lag' dog-days before each row's day: the last day before it, then lag-1 days further back (a dog's
        # own later runs count in its older races' lines, as a rival's would)
        dd = days[["dog", "day", "race"]].copy()
        dd["prior"] = dd.groupby("dog").race.shift(lag - 1)
        m = pd.merge_asof(rows.sort_values("day"), dd.sort_values("day").rename(columns={"day": "dday"})
                          [["dog", "dday", "prior"]], left_on="day", right_on="dday", by="dog",
                          allow_exact_matches=False, direction="backward").set_index("row").reindex(rows.row)
        n, w = since(m.prior, day)
        out[lag] = (n, w)
    n1, w1 = out[1]
    n3 = out[1][0] + out[2][0] + out[3][0]
    w3 = out[1][1] + out[2][1] + out[3][1]
    return pd.DataFrame({"fl1_n": n1, "fl1_wins": w1, "fl1_wr": (w1 + 0.5) / (n1 + 3.0),
                         "fl3_n": n3, "fl3_wr": (w3 + 0.5) / (n3 + 3.0)}, index=df.index)


def calculate(df: pd.DataFrame, dog_days=None) -> tuple[pd.DataFrame, list[str]]:
    """The block for the engine's frame (after the base features): its columns and their names."""
    g = df.race_id
    cols = {}
    e = elo(df)
    cols[f"{PREFIX}elo"] = e.r.astype(np.float32)
    cols[f"{PREFIX}elo_n"] = e.n.astype(np.float32)
    cols.update(_race_stats(e.r, g, f"{PREFIX}elo"))
    k = kalman(df, df.grade_par if "grade_par" in df else None)
    cols[f"{PREFIX}kr"] = k.m.astype(np.float32)
    cols[f"{PREFIX}kr_sd"] = k.sd.astype(np.float32)
    cols[f"{PREFIX}kr_n"] = k.n.astype(np.float32)
    st = _race_stats(k.m, g, f"{PREFIX}kr")
    cols[f"{PREFIX}kr_z"], cols[f"{PREFIX}kr_gap"] = st[f"{PREFIX}kr_z"], st[f"{PREFIX}kr_gap"]
    if "grade_par" in df:
        cols[f"{PREFIX}kr_vs_par"] = (k.m - df.grade_par).astype(np.float32)
    f = form_lines(df)
    for c in f.columns:
        cols[f"{PREFIX}{c}"] = f[c].astype(np.float32)
    out = pd.DataFrame(cols, index=df.index)
    return out, list(out.columns)
