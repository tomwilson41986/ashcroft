"""Pedigree posteriors: what a dog's family says about it, as a fitted hierarchy with its uncertainty.

Screened on 12 Nov 2026 (wave 4) against what the stack with the trials and optics tables still misses. The breeding
table (br_) is a single level with fixed constants and the dog's own runs left in; the litter table (lt_) has no
sire or dam level. This table is the hierarchy sire -> dam -> litter -> dog, each level an empirical-Bayes
normal-normal posterior (a mean AND an sd), the prior strengths fitted and not fixed, the progeny's runs clustered
by dog (a dog with 60 runs counts little more than one with 10), and the dog's own runs always left out.

**The measures** (per run; a run is read on later days only, a card row never):

- ``spd``: (the track and distance's median calculated time - the run's) / (1.4826 x the median absolute deviation),
  clipped to +-4; the standard is fitted on the rows before the month (a track and distance with ``MIN_STD`` runs);
- ``rc``: the race's mean beaten lengths (each clipped to 0..20, every row of the race, three at least) less the
  dog's; ``ae``: won less the SP's normalised chance;
- ``imp``: for a dog first seen at ``YOUNG_MONTHS`` months or younger (and ``DEBUT_GAP`` days at least after the
  frame starts, so that its first row is a debut), ``spd`` on its runs 4 to 15 less the mean of its first three.

**The model.** y = mu + sire + dam + litter + dog + noise, variances ts, td, tl, th, s2. The variance components
are fitted by the method of moments on the dogs' means: s2 the pooled within-dog variance, ts the covariance of
paternal half-sibs, td of maternal half-sibs, tl that of littermates less ts and td, th the rest of the between-dog
variance; each at least ``FLOOR`` of the between-dog variance.

**The posterior at a level** (the dog's own runs out): each other dog of the group with n earlier runs and mean m
weighs w = 1 / (v + s2 / n), v the variance below the level (sire: td + tl + th; dam: tl + th; litter: th);
mean = sum w (m - offset) / (1 / tau2 + sum w), sd = (1 / tau2 + sum w)^-0.5. The offset is mu for the sire, mu plus
the dog's sire's posterior for the dam, and that plus the dam's for the litter (both as at the month's start). A
first-crop sire, an unseen dam or litter has mean 0 (the level above) and sd tau.

**The dog.** The family prior f = sire + dam + litter, sd_f the root of the summed variances; ``pg_ped_weight`` =
(1 / (sd_f^2 + th)) / (1 / (sd_f^2 + th) + n / s2), the family's weight in the dog's own posterior given its n
earlier runs; ``pg_post_shift`` = weight x (f - the dog's own mean less mu); ``pg_fam_pull`` = weight x f.

**Context** (the parent's progeny in today's trip band, at today's track, on today's trap side): a dog's
observation is its mean in the context less its mean overall, noise s2 (1 / n_ctx - 1 / n); the parent-by-context
variance is the sibling covariance of those, the dog-by-context variance the rest; the same posterior, shrunk to
the parent's overall effect (0). Trip bands: under 400 m, 400 to 549 m, 550 m and over. Trap sides: 1-2, 3-4, 5-6.
Today's going is not a card fact and is not read.

**Point in time.** Every fitted quantity (the time standards, mu, the five variances of each measure, the context
variances and means, the offsets) is refitted at each month's start from the rows before that month, as the screen's
script did. A row's posterior then reads the family's runs on days strictly before its day (the month-start state
plus the month's earlier days). A month with fewer than ``MIN_DOGS`` dogs of two runs before it (the first month
of a frame) has no fit and its rows are NaN. The pedigree (sire, dam, whelping month) is the dog's first row's: it
is fixed at birth and printed on the card. A row with no sire is NaN throughout.

**How it is built.** One pass a month: the dogs' counts and sums before the month are a ``bincount`` (carried
forward for the measures that no fit changes), each group's state at the month start another, and only the month's
own rows are walked day by day. The screen's script rebuilt its cumulative tables over every row for every month;
this gives the same numbers at a cost of a few passes over the earlier rows a month. That later days and the day's
own results move nothing is tested in tests/test_greyhound_pedigree.py (cut frames, rewritten days, a card).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "pg_"
SOURCE = ("GBGB results (sire, dam, whelping month, calculated time, distance beaten, SP, track, distance, trap: "
          "sources/gbgb.py) of the family's other dogs on earlier days")
FLOOR = 0.005                # a variance component is at least this share of the between-dog variance
MIN_STD = 200                # runs a track and distance needs before the month for a time standard
MIN_DOGS = 50                # dogs with two runs before the month a fit needs
YOUNG_MONTHS = 22.0          # a dog first seen at this age or younger is a young dog
DEBUT_GAP = 45               # days after the frame's first day before a first row is taken for a debut
IMP_RUNS = (4, 15)           # the runs the improvement is read on, against the first three
BIG = 1 << 21                # day numbers are below this


def _norm(s: pd.Series) -> pd.Series:
    """Names for matching: lower case, letters and digits only; a missing or empty name is NaN."""
    u = s.astype(object).where(s.notna())
    out = u.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return out.where(u.notna() & (out != ""))


def _codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s)[0].astype(np.int64)          # -1 where missing


def _cum_excl(code: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Each row's group's sum of ``vals`` (rows x k) over the group's EARLIER rows (rows are in time order)."""
    if not len(code):
        return vals.copy()
    o = np.argsort(code, kind="stable")
    v = vals[o]
    c = np.cumsum(v, axis=0) - v
    k = code[o]
    new = np.r_[True, k[1:] != k[:-1]]
    c -= c[np.flatnonzero(new)][np.cumsum(new) - 1]
    out = np.empty_like(c)
    out[o] = c
    return out


class _Tab:
    """The month's rows' ``cols`` summed by key, read over the days strictly before a day."""

    def __init__(self, key: np.ndarray, day: np.ndarray, cols: list):
        ok = key >= 0
        k, code = key[ok], key[ok] * BIG + day[ok]
        o = np.argsort(code, kind="stable")
        v = np.column_stack(cols)[ok][o].astype(float)
        self.k, self.code, self.w = k[o], code[o], v.shape[1]
        c = np.cumsum(v, axis=0)
        if len(c):
            new = np.r_[True, self.k[1:] != self.k[:-1]]
            c -= (c - v)[np.flatnonzero(new)][np.cumsum(new) - 1]
        self.cum = c

    def before(self, qkey: np.ndarray, qday: np.ndarray) -> np.ndarray:
        out = np.zeros((len(qkey), self.w))
        if not len(self.code):
            return out
        pos = np.searchsorted(self.code, qkey * BIG + qday, "left") - 1
        ok = (pos >= 0) & (qkey >= 0)
        ok[ok] &= self.k[pos[ok]] == qkey[ok]
        out[ok] = self.cum[pos[ok]]
        return out


def _pair_cov(x: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """Sum over groups of sum_{i != j} x_i x_j, and the number of ordered pairs."""
    ok = g >= 0
    if not ok.any():
        return 0.0, 0.0
    s = np.bincount(g[ok], weights=x[ok])
    q = np.bincount(g[ok], weights=x[ok] ** 2)
    n = np.bincount(g[ok]).astype(float)
    return float((s * s - q).sum()), float((n * (n - 1)).sum())


def _state(dog: np.ndarray, y: np.ndarray, nd: int) -> np.ndarray:
    """Per dog: the count, sum and sum of squares of ``y`` over the rows given."""
    ok = np.isfinite(y)
    d, v = dog[ok], y[ok]
    return np.column_stack([np.bincount(d, minlength=nd).astype(float), np.bincount(d, weights=v, minlength=nd),
                            np.bincount(d, weights=v * v, minlength=nd)])


def _hyper(st: np.ndarray, ped: dict) -> dict | None:
    """The variance components from the dogs' counts, sums and sums of squares before the month."""
    n, s, q = st[:, 0], st[:, 1], st[:, 2]
    two = n >= 2
    if int(two.sum()) < MIN_DOGS:
        return None
    mu = float(s.sum() / n.sum())
    has = n > 0
    s2 = float((q[has] - s[has] ** 2 / n[has]).sum() / (n[has] - 1).sum())
    s2 = max(s2, 1e-12)
    x, nn = s[two] / n[two] - mu, n[two]
    tot = float(x.var() - s2 * (1.0 / nn).mean())
    S_, Sn = _pair_cov(x, ped["sire"][two])
    D_, Dn = _pair_cov(x, ped["dam"][two])
    P_, Pn = _pair_cov(x, ped["cross"][two])
    L_, Ln = _pair_cov(x, ped["lit"][two])
    c_pat, c_mat, c_lit = (S_ - P_) / max(Sn - Pn, 1), (D_ - P_) / max(Dn - Pn, 1), L_ / max(Ln, 1)
    fl = FLOOR * max(tot, 1e-9)
    ts, td = max(c_pat, fl), max(c_mat, fl)
    tl = max(c_lit - ts - td, fl)
    th = max(tot - ts - td - tl, fl)
    return {"mu": mu, "s2": s2, "tot": tot, "ts": ts, "td": td, "tl": tl, "th": th, "dogs": int(two.sum()),
            "pairs": {"paternal_half": int((Sn - Pn) / 2), "maternal_half": int((Dn - Pn) / 2),
                      "littermates": int(Ln / 2)}}


def _wa(n, s, off, v, s2):
    """A dog's precision weight and weighted deviation at a level."""
    pos = n > 0
    nn = np.where(pos, n, 1.0)
    w = np.where(pos, 1.0 / (v + s2 / nn), 0.0)
    return w, w * (s / nn - off)


class _Month:
    """A measure's rows of the month against the dogs' state at its start."""

    def __init__(self, st: np.ndarray, dm: np.ndarray, daym: np.ndarray, ym: np.ndarray):
        self.n0, self.s0 = st[:, 0], st[:, 1]
        self.dm, self.day = dm, daym
        self.ok = np.isfinite(ym).astype(float)
        self.y = np.where(np.isfinite(ym), ym, 0.0)
        ex = _cum_excl(dm, np.column_stack([self.ok, self.y]))
        self.nb, self.sb = self.n0[dm] + ex[:, 0], self.s0[dm] + ex[:, 1]         # before the row
        self.na, self.sa = self.nb + self.ok, self.sb + self.y                      # after it
        own = _Tab(dm, daym, [self.ok, self.y]).before(dm, daym)
        self.nq, self.sq = self.n0[dm] + own[:, 0], self.s0[dm] + own[:, 1]         # before the row's DAY


def _chain(mo: _Month, h: dict | None, ped: dict, want_own: bool = False) -> dict:
    """The sire, dam and litter posteriors of the month's rows (the dog's own runs out)."""
    m = len(mo.dm)
    nan = np.full(m, np.nan)
    if h is None:
        return {k: nan for k in ("s_mu", "s_sd", "d_mu", "d_sd", "l_mu", "l_sd", "f_mu", "f_sd", "pw", "shift")}
    s2 = h["s2"]
    off = np.full(len(mo.n0), h["mu"])
    r = {}
    for tag, key, tau, v in (("s", ped["sire"], h["ts"], h["td"] + h["tl"] + h["th"]),
                             ("d", ped["dam"], h["td"], h["tl"] + h["th"]), ("l", ped["lit"], h["tl"], h["th"])):
        nk = int(key.max()) + 2
        w, a = _wa(mo.n0, mo.s0, off, v, s2)
        has = key >= 0
        G = np.column_stack([np.bincount(key[has], weights=w[has], minlength=nk),
                             np.bincount(key[has], weights=a[has], minlength=nk)])
        km, om = key[mo.dm], off[mo.dm]
        w1, a1 = _wa(mo.na, mo.sa, om, v, s2)
        w0, a0 = _wa(mo.nb, mo.sb, om, v, s2)
        g = np.where((km >= 0)[:, None], G[km], 0.0) + _Tab(km, mo.day, [w1 - w0, a1 - a0]).before(km, mo.day)
        ow, oa = _wa(mo.nq, mo.sq, om, v, s2)
        W = np.where(km >= 0, np.maximum(g[:, 0] - ow, 0.0), 0.0)
        A = np.where(km >= 0, g[:, 1] - oa, 0.0)
        prec = 1.0 / tau + W
        r[f"{tag}_mu"], r[f"{tag}_sd"] = A / prec, np.sqrt(1.0 / prec)
        post = G[:, 1] / (1.0 / tau + G[:, 0])                       # the group's posterior at the month's start
        off = off + np.where(has, post[key], 0.0)
    known = ped["sire"][mo.dm] >= 0
    r = {k: np.where(known, x, np.nan) for k, x in r.items()}
    r["f_mu"] = r["s_mu"] + r["d_mu"] + r["l_mu"]
    r["f_sd"] = np.sqrt(r["s_sd"] ** 2 + r["d_sd"] ** 2 + r["l_sd"] ** 2)
    if want_own:
        pv = r["f_sd"] ** 2 + h["th"]
        r["pw"] = (1.0 / pv) / (1.0 / pv + mo.nq / s2)
        r["shift"] = np.where(mo.nq > 0, r["pw"] * (r["f_mu"] - (mo.sq / np.maximum(mo.nq, 1.0) - h["mu"])), np.nan)
    return r


class _Ctx:
    """The (dog, context) pairs of a card fact (trip band, track, trap side)."""

    def __init__(self, dog: np.ndarray, c: np.ndarray, nd: int):
        self.nc = int(c.max()) + 1 if len(c) and c.max() >= 0 else 1
        ok = c >= 0
        code = dog[ok] * self.nc + c[ok]
        u, inv = np.unique(code, return_inverse=True)
        self.row = np.full(len(dog), -1, dtype=np.int64)
        self.row[ok] = inv
        self.dog, self.c = u // self.nc, u % self.nc
        self.count = np.bincount(self.dog, minlength=nd)
        self.start = np.cumsum(self.count) - self.count
        self.c_row = c

    def state(self, y: np.ndarray, lo: int, hi: int) -> np.ndarray:
        """Per pair: the count and sum of ``y`` over the rows lo..hi in the pair's context."""
        p, v = self.row[lo:hi], y[lo:hi]
        ok = (p >= 0) & np.isfinite(v)
        return np.column_stack([np.bincount(p[ok], minlength=len(self.dog)).astype(float),
                                np.bincount(p[ok], weights=v[ok], minlength=len(self.dog))])


def _context(mo: _Month, h: dict | None, cx: _Ctx, pst: np.ndarray, parent: np.ndarray, rows: slice):
    """The parent's progeny in the row's context against each dog's own overall mean: the posterior's mean and sd."""
    m = len(mo.dm)
    nan = np.full(m, np.nan)
    if h is None:
        return nan, nan, None
    s2, nc = h["s2"], cx.nc
    nb0, sb0 = pst[:, 0], pst[:, 1]

    def obs(nb, sb, n, s):
        valid = (nb > 0) & (n > nb)
        nbs, ns = np.where(valid, nb, 1.0), np.where(valid, n, 2.0)
        return valid, sb / nbs - s / ns, 1.0 / nbs - 1.0 / ns

    valid, o, noise = obs(nb0, sb0, mo.n0[cx.dog], mo.s0[cx.dog])
    if int(valid.sum()) < MIN_DOGS:
        return nan, nan, None
    cnt = np.bincount(cx.c[valid], minlength=nc)
    mc = np.bincount(cx.c[valid], weights=o[valid], minlength=nc) / np.maximum(cnt, 1)
    x = o[valid] - mc[cx.c[valid]]
    Tt = float(np.mean(x * x - s2 * noise[valid]))
    par = parent[cx.dog]
    gk = np.where(par >= 0, par * nc + cx.c, -1)
    cv, cn = _pair_cov(x, gk[valid])
    fl = FLOOR * max(Tt, 1e-9)
    tsc = max(cv / max(cn, 1), fl)
    tdc = max(Tt - tsc, fl)

    def wa(nb, sb, n, s, c):
        ok, ob, nz = obs(nb, sb, n, s)
        w = np.where(ok, 1.0 / (tdc + s2 * nz), 0.0)
        return w, w * (ob - mc[c])

    w, a = wa(nb0, sb0, mo.n0[cx.dog], mo.s0[cx.dog], cx.c)
    ng = (int(parent.max()) + 2) * nc
    has = gk >= 0
    G = np.column_stack([np.bincount(gk[has], weights=w[has], minlength=ng),
                         np.bincount(gk[has], weights=a[has], minlength=ng)])
    # the month's rows, one line for each of the dog's contexts: the state of (dog, context) before and after the run
    cm = cx.c_row[rows]
    rep = cx.count[mo.dm]
    ri = np.repeat(np.arange(m), rep)
    pi = cx.start[mo.dm][ri] + (np.arange(len(ri)) - np.repeat(np.cumsum(rep) - rep, rep))
    inb = ((cm[ri] == cx.c[pi]) & (mo.ok[ri] > 0)).astype(float)
    yb = inb * mo.y[ri]
    ex = _cum_excl(pi, np.column_stack([inb, yb]))
    nbb, sbb = nb0[pi] + ex[:, 0], sb0[pi] + ex[:, 1]
    w0, a0 = wa(nbb, sbb, mo.nb[ri], mo.sb[ri], cx.c[pi])
    w1, a1 = wa(nbb + inb, sbb + yb, mo.na[ri], mo.sa[ri], cx.c[pi])
    T = _Tab(gk[pi], mo.day[ri], [w1 - w0, a1 - a0])
    pm = parent[mo.dm]
    okq = (pm >= 0) & (cm >= 0)
    qg = np.where(okq, pm * nc + cm, -1)
    g = np.where(okq[:, None], G[qg], 0.0) + T.before(qg, mo.day)
    rp = cx.row[rows]                                                # the row's own (dog, context)
    own = _Tab(rp, mo.day, [mo.ok, mo.y]).before(rp, mo.day)
    ow, oa = wa(nb0[rp] + own[:, 0], sb0[rp] + own[:, 1], mo.nq, mo.sq, np.maximum(cm, 0))
    prec = 1.0 / tsc + np.maximum(g[:, 0] - ow, 0.0)
    return (np.where(okq, (g[:, 1] - oa) / prec, np.nan), np.where(okq, np.sqrt(1.0 / prec), np.nan),
            {"tau_parent_ctx": tsc, "tau_dog_ctx": tdc, "obs": int(valid.sum()), "pairs": int(cn / 2)})


def _standards(ordt: np.ndarray, grp: np.ndarray, adj: np.ndarray, iD: int, ng: int):
    """Each track and distance's median calculated time and median absolute deviation over the rows before iD
    (NaN under ``MIN_STD`` runs). ``ordt``: the rows with a time, by group then time."""
    sel = ordt[ordt < iD]
    med, mad = np.full(ng + 1, np.nan), np.full(ng + 1, np.nan)
    if not len(sel):
        return med, mad
    g, t = grp[sel], adj[sel]
    cnt = np.bincount(g, minlength=ng)
    st = np.cumsum(cnt) - cnt
    has = np.flatnonzero(cnt > 0)

    def mid(v):
        return 0.5 * (v[st[has] + (cnt[has] - 1) // 2] + v[st[has] + cnt[has] // 2])

    m = np.full(ng, np.nan)
    m[has] = mid(t)
    dev = np.abs(t - m[g])
    dev = dev[np.argsort(g + dev / (dev.max() + 1.0), kind="stable")]
    d = np.full(ng, np.nan)
    d[has] = mid(dev)
    keep = cnt >= MIN_STD
    med[:ng], mad[:ng] = np.where(keep, m, np.nan), np.where(keep, d, np.nan)
    return med, mad


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    t_start = time.time()
    n_all = len(runs)
    df = runs.reset_index(drop=True).sort_values(["t", "race_id", "trap"], kind="stable")   # any index, repeats too
    back = df.index.to_numpy()                                      # the sorted row's place in ``runs``
    n = len(df)
    date = core.day(df)
    day = (date - date.min()).dt.days.to_numpy().astype(np.int64)
    if n and not (np.diff(day) >= 0).all():                         # rows with no time of day: order by the day first
        o = np.argsort(day, kind="stable")
        df, back, date, day = df.iloc[o], back[o], date.iloc[o], day[o]
    card = df.is_card.fillna(False).to_numpy(bool) if "is_card" in df else np.zeros(n, bool)

    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(float) if col in df else np.full(n, np.nan)

    # ---- the dog and its pedigree (its first row's: fixed at birth, printed on the card)
    dog = _codes(df.dog_id)
    nd = int(dog.max()) + 1 if n else 0
    first = np.full(nd, n, dtype=np.int64)
    np.minimum.at(first, dog, np.arange(n))
    sire = _norm(df.sire) if "sire" in df else pd.Series(np.nan, index=df.index, dtype=object)
    dam = _norm(df.dam) if "dam" in df else pd.Series(np.nan, index=df.index, dtype=object)
    born_s = df.born.astype(object).where(df.born.notna()) if "born" in df else pd.Series(np.nan, index=df.index, dtype=object)
    lit = (sire + "|" + dam + "|" + born_s.astype(str)).where(sire.notna() & dam.notna() & born_s.notna())
    cross = (sire + "|" + dam).where(sire.notna() & dam.notna())
    ped = {k: _codes(v)[first] for k, v in (("sire", sire), ("dam", dam), ("lit", lit), ("cross", cross))}
    born = pd.to_datetime(born_s, format="%b-%Y", errors="coerce")
    age0 = ((date - born).dt.days / 30.44).to_numpy(float)[first]
    with np.errstate(invalid="ignore"):
        young = (day[first] >= DEBUT_GAP) & (age0 <= YOUNG_MONTHS)

    # ---- the measures no fit changes (a card row has no result)
    bl = np.where(card, np.nan, np.clip(num("beaten_lengths"), 0, 20))
    rc = _codes(df.race_id)
    nr = int(rc.max()) + 1 if n else 0
    okb = np.isfinite(bl)
    cntb = np.bincount(rc[okb], minlength=nr)
    with np.errstate(invalid="ignore", divide="ignore"):
        meanb = np.bincount(rc[okb], weights=bl[okb], minlength=nr) / cntb
    y_rc = np.where(cntb[rc] >= 3, meanb[rc] - bl, np.nan)
    y_ae = np.where(card, np.nan, num("won") - num("sp_p_norm"))
    adj = np.where(card, np.nan, num("adjusted_time"))
    dist = num("distance_m")
    grp = _codes(df.track.astype(str) + "|" + pd.Series(dist, index=df.index).astype(str))
    grp = np.where(np.isfinite(dist) & df.track.notna().to_numpy(), grp, -1)
    ng = int(grp.max()) + 1 if n else 0
    ht = np.flatnonzero(np.isfinite(adj) & (grp >= 0))
    ordt = ht[np.lexsort((adj[ht], grp[ht]))]
    # ---- the card facts
    band = np.where(np.isfinite(dist), np.where(dist < 400, 0, np.where(dist < 550, 1, 2)), -1).astype(np.int64)
    trap = num("trap")
    side = np.where(np.isfinite(trap) & (trap >= 1) & (trap <= 6), (np.nan_to_num(trap, nan=1.0).clip(1, 6) - 1) // 2,
                    -1).astype(np.int64)
    track = np.where(df.track.notna().to_numpy(), _codes(df.track.astype(str)), -1)
    cxs = {"band": _Ctx(dog, band, nd), "track": _Ctx(dog, track, nd), "side": _Ctx(dog, side, nd)}
    perm = np.argsort(dog, kind="stable")                            # a dog's rows together, in time order
    dperm = dog[perm]
    newd = np.r_[True, dperm[1:] != dperm[:-1]] if n else np.zeros(0, bool)
    firstp, grpp = np.flatnonzero(newd), np.cumsum(newd) - 1

    out = {c: np.full(n, np.nan) for c in COLUMNS}
    period = date.dt.year.to_numpy() * 12 + date.dt.month.to_numpy() - 1
    d0 = date.min()
    fixed = {"rc": (y_rc, _state(dog[:0], y_rc[:0], nd)), "ae": (y_ae, _state(dog[:0], y_ae[:0], nd))}
    fixed_ctx = cxs["side"].state(y_rc, 0, 0)
    done, hyper_last, months, fitted = 0, {}, 0, 0
    for p in np.unique(period):
        D = int((pd.Timestamp(year=int(p // 12), month=int(p % 12) + 1, day=1) - d0).days)
        iD, iE = int(np.searchsorted(day, D, "left")), int(np.searchsorted(period, p, "right"))
        rows = slice(iD, iE)
        dm, daym = dog[rows], day[rows]
        months += 1
        # the time standard and the speed measure, from the rows before the month
        med, mad = _standards(ordt, grp, adj, iD, ng)
        with np.errstate(invalid="ignore", divide="ignore"):
            y_spd = np.clip((med[grp] - adj) / (1.4826 * mad[grp]), -4, 4)
        y_spd[grp < 0] = np.nan
        oks = np.isfinite(y_spd)
        ck = np.cumsum(oks[perm])
        k = np.empty(n)
        k[perm] = ck - (ck - oks[perm])[firstp][grpp]                # the dog's runs with a speed, this one included
        e3 = np.bincount(dog[oks & (k <= 3)], weights=y_spd[oks & (k <= 3)], minlength=nd) / 3.0
        y_imp = np.where(young[dog] & oks & (k >= IMP_RUNS[0]) & (k <= IMP_RUNS[1]), y_spd - e3[dog], np.nan)
        for name in ("rc", "ae"):                                    # carried forward: the rows since the last month
            y, st = fixed[name]
            fixed[name] = (y, st + _state(dog[done:iD], y[done:iD], nd))
        fixed_ctx = fixed_ctx + cxs["side"].state(y_rc, done, iD)
        done = iD
        st_spd, st_imp = _state(dog[:iD], y_spd[:iD], nd), _state(dog[:iD], y_imp[:iD], nd)
        hs = {"spd": _hyper(st_spd, ped), "rc": _hyper(fixed["rc"][1], ped), "ae": _hyper(fixed["ae"][1], ped),
              "imp": _hyper(st_imp, ped)}
        fitted += hs["spd"] is not None
        mo = _Month(st_spd, dm, daym, y_spd[rows])
        r = _chain(mo, hs["spd"], ped, want_own=True)
        for col, key in (("sire_mu", "s_mu"), ("sire_sd", "s_sd"), ("dam_mu", "d_mu"), ("dam_sd", "d_sd"),
                         ("lit_mu", "l_mu"), ("lit_sd", "l_sd"), ("fam_mu", "f_mu"), ("fam_sd", "f_sd"),
                         ("ped_weight", "pw"), ("post_shift", "shift")):
            out[PREFIX + col][rows] = r[key]
        out[PREFIX + "fam_pull"][rows] = r["pw"] * r["f_mu"]
        hc = {}
        mu, sd, hc["sire_band"] = _context(mo, hs["spd"], cxs["band"], cxs["band"].state(y_spd, 0, iD), ped["sire"], rows)
        out[PREFIX + "sband_mu"][rows], out[PREFIX + "sband_sd"][rows] = mu, sd
        out[PREFIX + "dband_mu"][rows], _, hc["dam_band"] = _context(
            mo, hs["spd"], cxs["band"], cxs["band"].state(y_spd, 0, iD), ped["dam"], rows)
        out[PREFIX + "strack_mu"][rows], _, hc["sire_track"] = _context(
            mo, hs["spd"], cxs["track"], cxs["track"].state(y_spd, 0, iD), ped["sire"], rows)
        mo = _Month(fixed["rc"][1], dm, daym, y_rc[rows])
        out[PREFIX + "rc_fam_mu"][rows] = _chain(mo, hs["rc"], ped)["f_mu"]
        out[PREFIX + "strap_mu"][rows], _, hc["sire_trapside"] = _context(
            mo, hs["rc"], cxs["side"], fixed_ctx, ped["sire"], rows)
        out[PREFIX + "ae_fam_mu"][rows] = _chain(_Month(fixed["ae"][1], dm, daym, y_ae[rows]), hs["ae"], ped)["f_mu"]
        r = _chain(_Month(st_imp, dm, daym, y_imp[rows]), hs["imp"], ped)
        out[PREFIX + "imp_fam_mu"][rows], out[PREFIX + "imp_fam_sd"][rows] = r["f_mu"], r["f_sd"]
        hyper_last = {"month": f"{int(p // 12)}-{int(p % 12) + 1:02d}", **hs, "context": hc}

    tab = pd.DataFrame({"race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    for c in COLUMNS:
        v = np.full(n_all, np.nan, dtype=np.float32)
        v[back] = out[c].astype(np.float32)
        tab[c] = v
    notes = {"rows": int(n_all), "months": int(months), "months_with_a_fit": int(fitted),
             "rows_with_a_family": round(float(tab[PREFIX + "fam_mu"].notna().mean()), 4) if n_all else 0.0,
             "dogs": int(nd), "sires": int(ped["sire"].max()) + 1 if nd else 0,
             "dams": int(ped["dam"].max()) + 1 if nd else 0, "litters": int(ped["lit"].max()) + 1 if nd else 0,
             "young_dogs": int(young.sum()), "last_fit": hyper_last, "seconds": round(time.time() - t_start, 1)}
    return tab, notes


COLUMNS = {
    f"{PREFIX}sire_mu": "the sire's posterior mean speed effect from his other progeny's earlier runs (robust sds of "
                        "calculated time against the track and distance's standard, higher faster; 0 with none)",
    f"{PREFIX}sire_sd": "that posterior's sd (the sire prior's sd for a first-crop sire)",
    f"{PREFIX}dam_mu": "the dam's posterior mean speed effect from her other progeny, net of their sires",
    f"{PREFIX}dam_sd": "that posterior's sd",
    f"{PREFIX}lit_mu": "the litter's posterior mean speed effect from the littermates, net of sire and dam",
    f"{PREFIX}lit_sd": "that posterior's sd",
    f"{PREFIX}fam_mu": "the family prior for the dog's speed: sire + dam + litter posterior means",
    f"{PREFIX}fam_sd": "the family prior's sd (the root of the three posterior variances summed)",
    f"{PREFIX}ped_weight": "the family prior's weight in the dog's own speed posterior given its earlier runs (1 "
                           "for a dog with none)",
    f"{PREFIX}post_shift": "that weight x (the family prior - the dog's own mean speed): how far the pedigree moves "
                           "the dog's level (none for a dog with no earlier timed run)",
    f"{PREFIX}fam_pull": "that weight x the family prior (large only for unexposed dogs)",
    f"{PREFIX}rc_fam_mu": "the family prior on lengths against the race's mean beaten distance",
    f"{PREFIX}ae_fam_mu": "the family prior on wins less the SP's chance (the market's miss on the family)",
    f"{PREFIX}imp_fam_mu": "the family prior on a young dog's improvement (speed on runs 4 to 15 less its first "
                           "three), from the family's other young dogs",
    f"{PREFIX}imp_fam_sd": "that prior's sd",
    f"{PREFIX}sband_mu": "the sire's progeny's speed in today's trip band (under 400 m, 400-549 m, 550 m and over) "
                         "against each dog's own overall speed: posterior mean",
    f"{PREFIX}sband_sd": "that posterior's sd",
    f"{PREFIX}dband_mu": "the same for the dam's progeny in today's trip band: posterior mean",
    f"{PREFIX}strack_mu": "the sire's progeny's speed at today's track against each dog's own overall: posterior mean",
    f"{PREFIX}strap_mu": "the sire's progeny's lengths against the race from today's trap side (1-2, 3-4, 5-6) "
                         "against each dog's own overall: posterior mean",
}
