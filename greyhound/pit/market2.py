"""The market as data, rebuilt (the feature review of 17 Nov 2026, B5 and the ``market`` row): the dog's and the yard's
record against the SP and the Betfair SP, as greyhound/pit/market.py (``mk_``) reads it, with four changes.

- **De-biased SP.** The SP's chances are Shin's (core.debiased_probs, vectorised: ``shin_probs``), not the proportionally normalised book, so the
  favourite-longshot bias does not read as "A/E": habitual favourites no longer read high, outsiders low.
- **Gamma-posterior A/E.** A dog's (or yard's) winning rate against the market is theta, wins ~ Poisson(theta x
  expected), theta ~ Gamma(a, a) a priori (mean 1). The column is log((a + W) / (a + E)), the log of the posterior
  mean; its sd is the posterior sd of log theta, sqrt(trigamma(a + W)); the z-score is the one over the other.
- **Decay by days.** W and E are sums over earlier days, each run weighted 0.5 ** (age in days / half-life).
  The half-life and the prior shape a are fitted together on an early slice (the first 20% of the days, never on
  or after ``TEST_FROM``) by the log-likelihood of each race's winner under the market's chances times the dog's
  posterior multiplier, renormalised within the race.
- **Drift on race-normalised shares.** log(SP share / BSP share), each book normalised within the race (where the
  whole race is priced on both), so the bookmakers' overround does not enter (mk_drift5 read log(BSP / SP), an
  overround book against a near-100% one, so every dog drifted).

Every column reads the dog's (or the trainer's) EARLIER days only; a card row reads what the historic row would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import polygamma

from greyhound.pit import core

PREFIX = "mq_"
SOURCE = ("GBGB results (SP, market position); Betfair's greyhound price files (sources/greyhound_prices.py: the BSP); "
          "Shin de-biased SP (core.debiased_probs)")
TEST_FROM = "2026-06-01"
EARLY_FRAC = 0.2
HALF_LIVES = (30.0, 60.0, 120.0, 240.0, 480.0, 960.0, np.inf)
SHAPES = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
MIN_FIT_RACES = 200
DEFAULT = {"half_life": 240.0, "a": 10.0}


def shin_probs(odds, race_id) -> np.ndarray:
    """core.debiased_probs(odds, race_id, "shin"), vectorised over races: every race's z bisected at once (the core
    function loops over races in Python, minutes on five million runs). Same rules: a race with fewer than two
    priced runners or a book under 1 is normalised proportionally; a runner with no price is NaN."""
    o = np.asarray(odds, dtype=float)
    q = np.where(np.isfinite(o) & (o > 1.0), 1.0 / np.where(o > 1.0, o, np.nan), np.nan)
    gi = pd.factorize(pd.Series(race_id).to_numpy())[0]
    out = np.full(len(o), np.nan)
    ok = np.isfinite(q) & (gi >= 0)
    if not ok.any():
        return out
    g, qq = gi[ok], q[ok]
    ng = int(g.max()) + 1
    Q = np.bincount(g, weights=qq, minlength=ng)
    cnt = np.bincount(g, minlength=ng)
    prop = ((cnt < 2) | (Q <= 1.0))[g]
    lo, hi = np.zeros(ng), np.full(ng, 0.5)

    def shin(z):
        zz = z[g]
        return (np.sqrt(zz * zz + 4.0 * (1.0 - zz) * qq * qq / Q[g]) - zz) / (2.0 * (1.0 - zz))
    for _ in range(60):
        z = 0.5 * (lo + hi)
        big = np.bincount(g, weights=shin(z), minlength=ng) > 1.0
        lo, hi = np.where(big, z, lo), np.where(big, hi, z)
    p = shin(0.5 * (lo + hi))
    p = p / np.bincount(g, weights=p, minlength=ng)[g]
    out[ok] = np.where(prop, qq / Q[g], p)
    return out


def _dayn(df: pd.DataFrame) -> np.ndarray:
    return (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()


def decayed_sums(key: np.ndarray, dn: np.ndarray, vals: np.ndarray, half_life: float) -> np.ndarray:
    """Per ``key`` (integer codes, -1: no key), over the key's EARLIER days: the sums of each column of ``vals``, each
    run weighted 0.5 ** (age in days / ``half_life``) (inf: no decay). Aligned to the rows; 0 with no history."""
    lam = 0.0 if not np.isfinite(half_life) else np.log(2.0) / half_life
    vals = np.nan_to_num(np.asarray(vals, float).reshape(len(key), -1))
    k = vals.shape[1]
    cols = [f"v{i}" for i in range(k)]
    d = pd.DataFrame(vals, columns=cols).assign(key=key, dn=dn)
    per = d[d.key >= 0].groupby(["key", "dn"], sort=True)[cols].sum().reset_index()
    if not len(per):
        return np.zeros((len(key), k))
    d0 = per.groupby("key").dn.transform("min").to_numpy()
    age = (per.dn.to_numpy() - d0).astype(float)
    cum = (per[cols].to_numpy() * np.exp(lam * age)[:, None])
    cum = pd.DataFrame(cum).groupby(per.key.to_numpy()).cumsum()
    prev = cum.groupby(per.key.to_numpy()).shift(1).fillna(0.0).to_numpy() * np.exp(-lam * age)[:, None]
    tab = pd.DataFrame(prev, columns=cols).assign(key=per.key.to_numpy(), dn=per.dn.to_numpy())
    m = d[["key", "dn"]].merge(tab, on=["key", "dn"], how="left")
    return m[cols].fillna(0.0).to_numpy()


def posterior(W: np.ndarray, E: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    """The gamma posterior's log mean multiplier and the posterior sd of log theta."""
    return np.log((a + W) / (a + E)), np.sqrt(polygamma(1, a + W))


def _early(dn: np.ndarray, have: np.ndarray) -> np.ndarray:
    """The first ``EARLY_FRAC`` of the days from the first with data (``have``), never reaching ``TEST_FROM``. The
    days are counted on every row, so a card day (no result yet) counts as its run's day would: the slice, and so
    the fitted constants, are the same on the card as in the historic build."""
    cut = early_cut(dn, have)
    return np.zeros(len(dn), bool) if cut is None else dn < cut


def early_cut(dn: np.ndarray, have: np.ndarray) -> int | None:
    test = (pd.Timestamp(TEST_FROM) - pd.Timestamp("2000-01-01")).days
    if not have.any():
        return None
    days = np.unique(dn[(dn >= dn[have].min()) & (dn < test)])
    if not len(days):
        return None
    return int(max(days[min(int(len(days) * EARLY_FRAC), len(days) - 1)], days[0] + 1))


def fit(key: np.ndarray, dn: np.ndarray, race: np.ndarray, p: np.ndarray, won: np.ndarray, use: np.ndarray,
        early: np.ndarray) -> dict:
    """The half-life and prior shape that best predict the early races' winners: each runner's chance ``p`` times
    its posterior multiplier from its earlier days, renormalised within the race (the mean log chance of the winner)."""
    e = early                                   # a runner with no key reads a multiplier of 1
    if not (e & (key >= 0)).any():
        return {**DEFAULT, "fitted": False}
    ii = np.flatnonzero(e)
    r = pd.factorize(race[ii])[0]
    ok = use[ii]
    nr = int(r.max()) + 1
    full = np.bincount(r, weights=(~ok).astype(float), minlength=nr) == 0
    wins = np.bincount(r, weights=np.where(ok, won[ii], 0.0), minlength=nr)
    good = full & (wins == 1) & (np.bincount(r, minlength=nr) >= 2)
    if good.sum() < MIN_FIT_RACES:
        return {**DEFAULT, "fitted": False, "races": int(good.sum())}
    keep = good[r]
    ri = pd.factorize(r[keep])[0]
    pp, ww = p[ii][keep], won[ii][keep]
    vals = np.c_[np.where(use, won, 0.0), np.where(use, p, 0.0)][ii]
    grid, best = [], None
    for hl in HALF_LIVES:
        WE = decayed_sums(key[ii], dn[ii], vals, hl)[keep]
        for a in SHAPES:
            u = pp * (a + WE[:, 0]) / (a + WE[:, 1])
            pa = u / np.bincount(ri, weights=u)[ri]
            ll = float(np.log(np.clip(pa[ww == 1], 1e-12, None)).mean())
            grid.append((hl if np.isfinite(hl) else "inf", a, round(ll, 6)))
            if best is None or ll > best[0]:
                best = (ll, hl, a)
    base = float(np.log(np.clip(pp[ww == 1], 1e-12, None)).mean())
    return {"half_life": best[1], "a": best[2], "fitted": True, "races": int(good.sum()), "ll": round(best[0], 6),
            "ll_market": round(base, 6), "grid": grid}


def _record(out: pd.DataFrame, name: str, key, dn, race, p, won, use, early_have, notes, mask=None, prm=None):
    early = _early(dn, early_have)
    prm = prm or fit(key, dn, race, p, won, use, early)
    vals = np.c_[np.where(use, won, 0.0), np.where(use, p, 0.0)]
    WE = decayed_sums(key, dn, vals, prm["half_life"])
    lae, sd = posterior(WE[:, 0], WE[:, 1], prm["a"])
    have = (WE[:, 1] > 0) & (key >= 0)
    if mask is not None:
        have &= mask
    out[f"{PREFIX}{name}_lae"] = np.where(have, lae, np.nan)
    out[f"{PREFIX}{name}_lae_sd"] = np.where(have, sd, np.nan)
    out[f"{PREFIX}{name}_z"] = np.where(have, lae / sd, np.nan)
    out[f"{PREFIX}{name}_e"] = np.where(have, WE[:, 1], np.nan)
    notes[name] = {k: (v if k != "half_life" or np.isfinite(v) else "inf") for k, v in prm.items()}
    notes[name]["early_to"] = str(pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(dn[early].max()))) \
        if early.any() else None
    return prm


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = runs.copy()
    prices = (ctx or {}).get("prices")
    if "bsp" not in df:
        from greyhound.parity import attach_bsp
        df = attach_bsp(df, prices)
    df = df.reset_index(drop=True)
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(len(df), bool)
    won = pd.to_numeric(df.won, errors="coerce").to_numpy(float)
    racing = ~trial & np.isfinite(won)
    race = df.race_id.to_numpy()
    dn = _dayn(df)
    p_sp = shin_probs(df.sp_decimal, df.race_id)
    p_bsp = pd.to_numeric(df.bsp_p_norm, errors="coerce").to_numpy(float)
    use_sp = racing & np.isfinite(p_sp)
    use_bsp = racing & np.isfinite(p_bsp)
    dog = pd.factorize(df.dog_id)[0]
    out = pd.DataFrame({"race_id": df.race_id, "dog_id": df.dog_id})
    notes = {"rows": len(df), "bsp_rows": int(np.isfinite(p_bsp).sum())}
    prm_sp = _record(out, "sp", dog, dn, race, p_sp, won, use_sp, use_sp, notes)
    pb = fit(dog, dn, race, p_bsp, won, use_bsp, _early(dn, use_bsp))
    _record(out, "bsp", dog, dn, race, p_bsp, won, use_bsp, use_bsp, notes, prm=pb if pb["fitted"] else
            {**prm_sp, "fitted": False, "from": "sp"})
    # the trainer at the time only (core.trainer_at_the_time: GBGB serves the current trainer on past results)
    tr = core.trainer_at_the_time(df, (ctx or {}).get("fetch_lag"))
    tk = pd.factorize(tr)[0]
    known = tk >= 0
    pt = fit(tk, dn, race, p_sp, won, use_sp & known, _early(dn, use_sp & known))
    pt = pt if pt["fitted"] else {**prm_sp, "fitted": False, "from": "dog sp"}
    _record(out, "tr_sp", tk, dn, race, p_sp, won, use_sp & known, use_sp & known, notes, mask=known, prm=pt)
    _record(out, "tr_bsp", tk, dn, race, p_bsp, won, use_bsp & known, use_bsp & known, notes, mask=known, prm=pt)
    # the exchange against the bookmakers on race-normalised shares (both books whole)
    sp_share = pd.to_numeric(df.sp_p_norm, errors="coerce")
    whole = sp_share.notna().groupby(df.race_id).transform("all") & pd.Series(p_bsp).notna()
    df["_drift"] = np.log(sp_share / pd.Series(p_bsp)).where(whole & pd.Series(racing))
    out[f"{PREFIX}drift5"] = core.dog_lagged(df, "_drift", 5)
    fld = pd.to_numeric(df.field, errors="coerce")
    df["_mvf"] = ((df.market_pos - df.position) / (fld - 1).where(fld > 1)).where(pd.Series(racing))
    out[f"{PREFIX}vs_market5"] = core.dog_lagged(df, "_mvf", 5)
    df["_p"] = np.where(use_sp, p_sp, np.nan)
    out[f"{PREFIX}last_sp_p"] = core.dog_lagged(df, "_p", 1)
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(float)
    return out.reset_index(drop=True), notes


_REC = {"sp": "the dog's record against the de-biased (Shin) SP",
        "bsp": "the dog's record against the Betfair SP (normalised book)",
        "tr_sp": "the trainer's record against the de-biased SP (the trainer at the time)",
        "tr_bsp": "the trainer's record against the Betfair SP (the trainer at the time)"}
COLUMNS = {
    **{c: v for k, lab in _REC.items() for c, v in (
        (f"{PREFIX}{k}_lae", (f"{lab}: gamma-posterior log A/E, log((a + W) / (a + E)), W and E decayed by days "
                             "(half-life and prior shape fitted on the first 20% of days)")),
        (f"{PREFIX}{k}_lae_sd", "its posterior sd (of log theta)"),
        (f"{PREFIX}{k}_z", "the log A/E over its posterior sd"),
        (f"{PREFIX}{k}_e", "the decayed expected winners behind it (the evidence)"))},
    f"{PREFIX}drift5": "mean log(SP share / BSP share), each book normalised within the race, last 5 (+: the "
                       "exchange longer than the bookmakers; no overround)",
    f"{PREFIX}vs_market5": "mean (market position less finishing position) / (field - 1), last 5 (+: beat the "
                           "market's order)",
    f"{PREFIX}last_sp_p": "the de-biased SP's chance for the dog last time",
}
