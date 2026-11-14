"""Pace and race shape (the owner, 14 Nov 2026): expected leader, early position, run style, pace pressure, and which
setup suits which dog. Local audit on GBGB Jul 2025 - Nov 2026 (scratchpad src/gbgb). Pars and constants from TRAIN
(to 2026-02-28); model coefficients fitted on FIT races (2025-10-01 .. 2026-02-28); scored on EVAL (2026-03-01 on).
Every dog feature is read from its EARLIER days only."""
import sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from research.greyhound.times_audit.load import load, L
from greyhound.metrics import parse_comment

TRAIN_END, FIT_FROM, EVAL_FROM = pd.Timestamp("2026-02-28"), pd.Timestamp("2025-10-01"), pd.Timestamp("2026-03-01")


def base():
    d = load()
    g = d[~d.trial & ~d.hcap & ~d.hurdle & d.position.notna() & d.dog_id.notna() & d.adjusted_time.notna()
          & d.going_ok].copy()
    g = g[(g.sectional.isna()) | g.sectional.between(1, 15)]
    full = g.groupby("race_id").sectional.transform(lambda s: s.notna().all())
    g["sec_ok"] = full
    tr = g.day <= TRAIN_END
    # sectional par by key x trap (train, every runner), percent figure (higher faster)
    sp = g[tr & g.sectional.notna()].groupby(["key", "trap"]).sectional.median()
    g["sec_par"] = pd.MultiIndex.from_frame(g[["key", "trap"]]).map(lambda t: sp.get(t, np.nan))
    g["S"] = 100 * np.log(g.sec_par / g.sectional)
    # run-in figure: run home against the key's median run home (train)
    g["runin"] = g.adjusted_time - g.sectional
    rp = g[tr & g.runin.notna()].groupby("key").runin.median()
    g["R"] = 100 * np.log(g.key.map(rp) / g.runin)
    tp = g[tr & (g.position == 1)].groupby("key").adjusted_time.median()
    g["T"] = 100 * np.log(g.key.map(tp) / g.adjusted_time)
    # early rank (1 fastest sectional) where the whole field is timed; leader; position gain
    g["erank"] = np.where(g.sec_ok, g.groupby("race_id").sectional.rank(method="first"), np.nan)
    g["nrun"] = g.groupby("race_id").dog_id.transform("size")
    g["epct"] = (g.erank - 1) / (g.nrun - 1)
    g["lead"] = (g.erank == 1).astype(float).where(g.sec_ok)
    g["gain"] = g.erank - g.position
    pc = g.comment.map(parse_comment).apply(pd.Series)
    g = pd.concat([g, pc[["brk", "epace", "led", "trouble", "path"]].add_prefix("c_")], axis=1)
    return g.sort_values(["dog_id", "day"])


def history(g, cols, last=3, days=180, kind="mean", half=3.0):
    """Per row: the dog's ``kind`` over its last ``last`` values from EARLIER days within ``days`` (mean, last, ewm by
    runs with half-life ``half``)."""
    out = {c: np.full(len(g), np.nan) for c in cols}
    idx = {k: v for v, k in enumerate(g.index)}
    for _, s in g.groupby("dog_id", sort=False):
        dd = s.day.to_numpy()
        vals = {c: s[c].to_numpy(float) for c in cols}
        pos = [idx[i] for i in s.index]
        for i in range(len(s)):
            j = np.searchsorted(dd, dd[i]); lo = np.searchsorted(dd, dd[i] - np.timedelta64(days, "D"))
            a = max(lo, j - last)
            if a >= j:
                continue
            for c in cols:
                v = vals[c][a:j]; ok = np.isfinite(v)
                if not ok.any():
                    continue
                if kind == "mean":
                    out[c][pos[i]] = v[ok].mean()
                elif kind == "last":
                    out[c][pos[i]] = v[ok][-1]
                else:
                    w = 0.5 ** (np.arange(len(v))[::-1] / half); out[c][pos[i]] = (v[ok] * w[ok]).sum() / w[ok].sum()
    return pd.DataFrame(out, index=g.index)


def clogit(df, feats, label, fit_mask, l2=1e-3):
    """A conditional logit over races (race-centred features, NaN to the race mean); fitted on ``fit_mask`` rows, the
    probabilities for every row."""
    X = df[feats].astype(float)
    X = X.fillna(X.groupby(df.race_id).transform("mean")).fillna(0.0)
    X = X - X.groupby(df.race_id).transform("mean")
    order = np.argsort(df.race_id.to_numpy(), kind="stable")
    rid = df.race_id.to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1]
    Xo, yo, fo = X.to_numpy()[order], df[label].to_numpy(float)[order], fit_mask.to_numpy()[order]
    reps = np.diff(np.r_[starts, len(rid)])
    fit_r = np.add.reduceat(fo.astype(float), starts) > 0

    def nll(b):
        z = Xo @ b
        m = np.maximum.reduceat(z, starts); e = np.exp(z - np.repeat(m, reps))
        se = np.add.reduceat(e, starts); lse = np.log(se) + m
        ll = np.add.reduceat(z * yo, starts) - lse
        p = e / np.repeat(se, reps)
        gr = Xo * (yo - p)[:, None]
        gsum = np.add.reduceat(gr, starts)[fit_r].sum(0)
        return -ll[fit_r].sum() / fit_r.sum() + l2 * b @ b, -gsum / fit_r.sum() + 2 * l2 * b
    res = minimize(nll, np.zeros(len(feats)), jac=True, method="L-BFGS-B")
    z = Xo @ res.x; m = np.maximum.reduceat(z, starts); e = np.exp(z - np.repeat(m, reps))
    p = e / np.repeat(np.add.reduceat(e, starts), reps)
    out = np.empty(len(p)); out[order] = p
    return pd.Series(out, index=df.index), dict(zip(feats, np.round(res.x, 4)))


def evaluate(df, p, label, mask):
    x = df[mask].assign(p=p[mask])
    x = x[x.groupby("race_id")[label].transform("sum") == 1]
    ll = -np.log(x.p[x[label] == 1].clip(1e-9)).mean()
    top = x.loc[x.groupby("race_id").p.idxmax()]
    return {"races": int(x.race_id.nunique()), "logloss": round(float(ll), 5),
            "uniform": round(float(np.log(x.groupby("race_id").size()).mean()), 5),
            "top1": round(float(top[label].mean()), 4)}
