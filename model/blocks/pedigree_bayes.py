"""Pedigree as a hierarchy of priors: sire of sire, sire, dam sire, dam, the cross, each with its uncertainty.

The pedigree blocks so far read the sire's and the dam's levels and aptitudes one at a time, with shrinkage
constants typed in, on results or on the raw market level (sire_apt, dam_line, pedigree_market, sire_windows: each
worth 0.0006 or less). The owner's ask of 12 Nov 2026: "a strong set of Bayesian statistics and priors ... how to
handle different and small sample sizes in pedigrees". Here every estimate is a posterior mean: the entity's own
record (decayed, half-life HALF days) pulled toward the level above it by a prior strength K fitted from the data
before FIT_UNTIL (the variance within entities over the variance between them, by moments), with its posterior
standard deviation beside it:

    sire of sire  ->  sire  ->  the cross (sire x dam sire), whose prior is the sire's plus the dam sire's
    dam sire      ->  dam (her other foals only: the horse's own runs are taken out)

on two residuals of each relation's run, neither of which leans with the price:

    m   the run's price against the price its field's market ratings implied (model/blocks/market_rating.py):
        how the market rates the stock beyond what their own price history says - what prices an unraced horse
    s   the run's finishing position against earlier runners at its price (model/blocks/_resid.py)

    pb_sire_<r>  pb_sire_<r>_sd      the sire: posterior mean and standard deviation
    pb_ss_<r>  pb_ds_<r>             the sire of the sire (where the sire himself raced in this data), the dam sire
    pb_dam_<r>  pb_dam_<r>_sd        the dam, from her other foals
    pb_cross_<r>                     the cross
    pb_ped_<r>                       the horse's pedigree prior: half the sire's plus half the dam's, plus what the
                                     cross adds beyond its two parents
    pb_ped_unexp_m                   pb_ped_m divided by (1 + the horse's own runs): the prior fading with exposure
    pb_sire_debut_m  pb_dam_debut_m  the same m record on first runs only, each shrunk to its all-runs posterior
    pb_sire_trip_s  pb_sire_going_s  pb_ds_trip_s  pb_ds_going_s
                                     the sire's (dam sire's) stock at today's trip band and going group, against
                                     their own level: the cell's record shrunk to zero
    pb_sire_trip_m                   the market's view of the sire's stock at this trip band, against its level
    pb_sire_impr  pb_dam_impr        improvement with experience: the time figure of the sire's stock (the dam's
                                     other foals) less each runner's own earlier level, at this many runs
    pb_sire_ss  pb_dam_ss            the spread of the stock's s (root mean square): a sire or a dam whose stock
                                     is all much of a muchness, or all over the place
    pb_sire_n  pb_dam_n              the decayed number of runs behind the sire's and the dam's estimates

Earlier days only for every record. Who a horse's sire and dam are is on the card.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import _resid
from model.blocks import time_figure as _tf
from model.blocks.cond_form import GOING, classify_going
from model.freshness_features import _Days, _col
from model.race_shape import DIST_BANDS, asof_decayed_mean, bands, day_index, race_key

R = ("m", "s")
FEATURES = ([f"pb_{e}_{r}" for r in R for e in ("sire", "ss", "ds", "dam", "cross", "ped")]
            + [f"pb_{e}_{r}_sd" for r in R for e in ("sire", "dam")]
            + ["pb_ped_unexp_m", "pb_sire_debut_m", "pb_dam_debut_m", "pb_sire_trip_s", "pb_sire_going_s", "pb_ds_trip_s",
               "pb_ds_going_s", "pb_sire_trip_m", "pb_sire_impr", "pb_dam_impr", "pb_sire_ss", "pb_dam_ss", "pb_sire_n",
               "pb_dam_n"])
POST_RACE: set[str] = set()
READS = sorted(set(_tf.READS) | {"stallion", "dam", "dam_stallion", "number_of_runners", "bfsp", "mr_lshare"})
AFTER = ("market_rating",)
FIT_UNTIL = "2021-01-01"
HALF = 4 * 365.0
K_DEFAULT = {"sire": 30.0, "ss": 60.0, "ds": 40.0, "dam": 8.0, "cross": 15.0, "cell": 30.0, "debut": 10.0}
K_BOUNDS = (3.0, 300.0)
MIN_FIT = 3000


def _norm(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str).str.lower().str.strip()
    t = t.str.replace(r"\s*\([a-z]{2,3}\)\s*$", "", regex=True).str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    return t.where(~t.isin(["", "nan", "none"]), "")


def _codes(t: pd.Series) -> np.ndarray:
    return np.where((t == "").to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _key(*parts) -> np.ndarray:
    bad = np.zeros(len(parts[0]), dtype=bool)
    out = np.zeros(len(parts[0]), dtype=np.int64)
    for p in parts:
        p = np.asarray(p, dtype=np.int64)
        bad |= p < 0
        out = out * (int(p.max()) + 2) + np.where(p < 0, 0, p)
    return np.where(bad, -1, pd.factorize(out, sort=True)[0]).astype(np.int64)


def _fit_k(key, value, fit, default):
    """Prior strength in runs: the variance within entities over the variance between them, on the fit rows."""
    ok = fit & (key >= 0) & np.isfinite(value)
    if ok.sum() < MIN_FIT:
        return default, float(np.nanstd(value[ok])) if ok.sum() > 10 else 1.0
    g = pd.Series(value[ok]).groupby(key[ok])
    n, mean, var = g.size(), g.mean(), g.var()
    big = n >= 2
    within = float((var[big] * (n[big] - 1)).sum() / max((n[big] - 1).sum(), 1))
    between = float(np.average((mean[big] - mean[big].mean()) ** 2, weights=n[big])) - within * float((1.0 / n[big]).mean())
    k = within / between if between > 1e-9 else K_BOUNDS[1]
    return float(np.clip(k, *K_BOUNDS)), float(np.sqrt(max(within, 1e-12)))


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    fit = day < int(np.datetime64(FIT_UNTIL, "D").astype(np.int64))
    race = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    hname, sname = _norm(df["horse_name"]), _norm(_col(df, "stallion"))
    horse, sire, dam, ds = _codes(hname), _codes(sname), _codes(_norm(_col(df, "dam"))), _codes(_norm(_col(df, "dam_stallion")))
    own_sire = pd.Series(sname.to_numpy(), index=hname.to_numpy())
    own_sire = own_sire[(own_sire.index != "") & (own_sire != "")]
    own_sire = own_sire[~own_sire.index.duplicated()]
    ss = _codes(sname.map(own_sire).fillna(""))                    # the sire's own sire, where the sire raced here
    cross = _key(sire, ds)

    s, _ = _resid.price_residual(df, day, race)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    inv = pd.Series(np.where(priced, 1.0 / np.where(priced, bsp, 1.0), 0.0), index=df.index)
    tot = inv.groupby(race).transform("sum").to_numpy()
    rows = pd.Series(1.0, index=df.index).groupby(race).transform("sum").to_numpy()
    npr = pd.Series(priced.astype(float), index=df.index).groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(priced & (tot > 0) & (npr >= 0.9 * rows), np.log(rows * inv.to_numpy() / np.where(tot > 0, tot, 1.0)), np.nan)
    m = np.clip(x - pd.to_numeric(_col(df, "mr_lshare"), errors="coerce").to_numpy(float), -3.0, 3.0)
    val = {"m": m, "s": s}
    H = _Days(horse, day)
    runs = H.to_rows(H.pos.astype(float))

    def rec(key, v, half=HALF):
        mean, n = asof_decayed_mean(key, day, v, key, day, halflife_days=half)
        return np.nan_to_num(mean), np.nan_to_num(n)

    def post(mean, n, prior, k):
        return (n * mean + k * prior) / (n + k)

    cols: dict[str, np.ndarray] = {}
    theta = {}
    for r in R:
        v = val[r]
        k, sig = {}, {}
        for e, key in (("sire", sire), ("ss", ss), ("ds", ds), ("dam", dam), ("cross", cross)):
            k[e], sig[e] = _fit_k(key, v, fit, K_DEFAULT[e])
        t_ss = post(*rec(ss, v), 0.0, k["ss"])
        t_ss = np.where(ss >= 0, t_ss, 0.0)
        sm, sn = rec(sire, v)
        t_s = post(sm, sn, t_ss, k["sire"])
        t_ds = post(*rec(ds, v), 0.0, k["ds"])
        dm, dn = rec(dam, v)
        hm, hn = rec(horse, v)
        on = np.maximum(dn - hn, 0.0)                               # her other foals
        om = np.where(on > 1e-9, (dm * dn - hm * hn) / np.where(on > 1e-9, on, 1.0), 0.0)
        t_d = post(om, on, np.where(ds >= 0, t_ds, 0.0), k["dam"])
        xm, xn = rec(cross, v)
        t_x = post(xm, xn, np.where(sire >= 0, t_s, 0.0) + np.where(ds >= 0, t_ds, 0.0), k["cross"])
        cols[f"pb_sire_{r}"] = np.where(sire >= 0, t_s, np.nan)
        cols[f"pb_sire_{r}_sd"] = np.where(sire >= 0, sig["sire"] / np.sqrt(sn + k["sire"]), np.nan)
        cols[f"pb_ss_{r}"] = np.where(sire >= 0, t_ss, np.nan)             # 0, the prior, where the sire's sire is not held
        cols[f"pb_ds_{r}"] = np.where(ds >= 0, t_ds, np.nan)
        cols[f"pb_dam_{r}"] = np.where(dam >= 0, t_d, np.nan)
        cols[f"pb_dam_{r}_sd"] = np.where(dam >= 0, sig["dam"] / np.sqrt(on + k["dam"]), np.nan)
        cols[f"pb_cross_{r}"] = np.where(cross >= 0, t_x, np.nan)
        extra = np.where(cross >= 0, t_x - np.where(sire >= 0, t_s, 0.0) - np.where(ds >= 0, t_ds, 0.0), 0.0)
        ped = 0.5 * np.where(sire >= 0, t_s, 0.0) + 0.5 * np.where(dam >= 0, t_d, 0.0) + extra
        cols[f"pb_ped_{r}"] = np.where((sire >= 0) | (dam >= 0), ped, np.nan)
        theta[r] = (t_s, t_ds, t_d, sn, on)
    cols["pb_ped_unexp_m"] = cols["pb_ped_m"] / (1.0 + np.nan_to_num(runs))
    cols["pb_sire_n"], cols["pb_dam_n"] = np.where(sire >= 0, theta["s"][3], np.nan), np.where(dam >= 0, theta["s"][4], np.nan)

    first = runs == 0
    fm = np.where(first, m, np.nan)
    a, n1 = rec(sire, fm)
    cols["pb_sire_debut_m"] = np.where(sire >= 0, post(a, n1, theta["m"][0], K_DEFAULT["debut"]), np.nan)
    a, n1 = rec(dam, fm)                                             # a horse has one first run: its own is not earlier
    cols["pb_dam_debut_m"] = np.where(dam >= 0, post(a, n1, theta["m"][2], K_DEFAULT["debut"]), np.nan)

    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    tb = pd.Series(bands(dist, DIST_BANDS)).fillna(-1).to_numpy().astype(np.int64)
    gg = pd.to_numeric(_col(df, "going_description").map(classify_going).map(GOING), errors="coerce").fillna(-1) \
        .to_numpy().astype(np.int64)
    kc = K_DEFAULT["cell"]
    for e, key, own in (("sire", sire, theta["s"][0]), ("ds", ds, theta["s"][1])):
        for tag, cell in (("trip", tb), ("going", gg)):
            cm, cn = rec(_key(key, cell), s)
            raw, _ = rec(key, s)
            cols[f"pb_{e}_{tag}_s"] = np.where((key >= 0) & (cell >= 0), cn * (cm - raw) / (cn + kc), np.nan)
    cm, cn = rec(_key(sire, tb), m)
    raw, _ = rec(sire, m)
    cols["pb_sire_trip_m"] = np.where((sire >= 0) & (tb >= 0), cn * (cm - raw) / (cn + kc), np.nan)

    fig = np.asarray(_tf.figures(df)["fig"], dtype=float)
    own_fig, _ = asof_decayed_mean(horse, day, fig, horse, day, halflife_days=365.0)
    gain = fig - own_fig
    exp_b = np.where(np.isfinite(runs), np.minimum(runs, 7), -1).astype(np.int64)
    exp_b = np.where(exp_b >= 4, np.where(exp_b >= 7, 5, 4), exp_b)   # 0, 1, 2, 3, 4-6, 7+
    for e, key in (("sire", sire), ("dam", dam)):
        cm, cn = rec(_key(key, exp_b), gain)
        cols[f"pb_{e}_impr"] = np.where((key >= 0) & (exp_b >= 0), cn * cm / (cn + kc), np.nan)
        sq, sqn = rec(key, s ** 2)
        cols[f"pb_{e}_ss"] = np.where((key >= 0) & (sqn > 0), np.sqrt(sq), np.nan)

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
